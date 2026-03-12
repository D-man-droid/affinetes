"""
SWE-INFINITE Cache Module (Read-Only)

Two-level cache for reading expansion tasks (produced by the mining pipeline):
  L1: Local filesystem (fast, per-machine)
  L2: R2 public bucket via HTTP
"""

import json
import os
from pathlib import Path
from typing import Optional, Dict, Any
from urllib.request import urlopen, Request
from urllib.error import HTTPError, URLError


class LocalCache:
    """Local filesystem cache keyed by instance_id."""

    def __init__(self, cache_dir: str = "/tmp/swe-infinite-cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_path(self, instance_id: str) -> Path:
        try:
            num = int(instance_id)
            return self.cache_dir / f"task_{num:011d}.json"
        except (ValueError, TypeError):
            return self.cache_dir / f"{instance_id}.json"

    def load(self, instance_id: str) -> Optional[Dict[str, Any]]:
        path = self._get_path(instance_id)
        if path.exists():
            with open(path, 'r') as f:
                return json.load(f)
        return None

    def save(self, instance_id: str, data: Dict[str, Any]) -> None:
        path = self._get_path(instance_id)
        with open(path, 'w') as f:
            json.dump(data, f, indent=2, default=str)

    def exists(self, instance_id: str) -> bool:
        return self._get_path(instance_id).exists()


class R2PublicCache:
    """Read-only R2 cache via public HTTP URL."""

    DEFAULT_BASE_URL = "https://pub-7882418a56434a479bf9a7febd660b36.r2.dev"
    DEFAULT_PREFIX = "bugs"

    def __init__(
        self,
        base_url: Optional[str] = None,
        prefix: Optional[str] = None,
    ):
        self.base_url = (base_url or os.getenv("R2_PUBLIC_URL") or self.DEFAULT_BASE_URL).rstrip("/")
        self.prefix = prefix if prefix is not None else (os.getenv("R2_PUBLIC_PREFIX") or self.DEFAULT_PREFIX)
        print(f"[CACHE] R2 public: {self.base_url}/{self.prefix}")

    @property
    def enabled(self) -> bool:
        return True

    @staticmethod
    def _format_key(task_id: str) -> str:
        """Format task_id to R2 filename: task_00000000001.json"""
        try:
            num = int(task_id)
            return f"task_{num:011d}.json"
        except (ValueError, TypeError):
            return f"{task_id}.json"

    def _get_url(self, instance_id: str) -> str:
        filename = self._format_key(instance_id)
        if self.prefix:
            return f"{self.base_url}/{self.prefix}/{filename}"
        return f"{self.base_url}/{filename}"

    def load(self, instance_id: str) -> Optional[Dict[str, Any]]:
        url = self._get_url(instance_id)
        try:
            req = Request(url, headers={"Accept": "application/json", "User-Agent": "swe-infinite/1.0"})
            with urlopen(req, timeout=30) as resp:
                return json.loads(resp.read())
        except HTTPError as e:
            if e.code == 404:
                return None
            print(f"[CACHE] R2 HTTP error for {instance_id}: {e.code} {e.reason}")
            return None
        except (URLError, TimeoutError) as e:
            print(f"[CACHE] R2 fetch error for {instance_id}: {e}")
            return None

    def exists(self, instance_id: str) -> bool:
        url = self._get_url(instance_id)
        try:
            req = Request(url, method="HEAD", headers={"User-Agent": "swe-infinite/1.0"})
            with urlopen(req, timeout=10):
                return True
        except Exception:
            return False


class TwoLevelCache:
    """Two-level read cache: Local (L1) + R2 public HTTP (L2).

    Read path: L1 hit -> return | L2 hit -> save to L1, return | miss -> None
    """

    def __init__(
        self,
        local_cache_dir: str = "/tmp/swe-infinite-cache",
        r2_base_url: Optional[str] = None,
        r2_prefix: Optional[str] = None,
        # Deprecated: kept for backward compatibility, ignored
        r2_endpoint: Optional[str] = None,
        r2_access_key: Optional[str] = None,
        r2_secret_key: Optional[str] = None,
        r2_bucket: Optional[str] = None,
    ):
        self.local = LocalCache(local_cache_dir)
        self.r2 = R2PublicCache(
            base_url=r2_base_url,
            prefix=r2_prefix,
        )

    def load(self, instance_id: str) -> Optional[Dict[str, Any]]:
        """Load task by instance_id (L1 -> L2)."""
        data = self.local.load(instance_id)
        if data is not None:
            return data

        if self.r2.enabled:
            data = self.r2.load(instance_id)
            if data is not None:
                self.local.save(instance_id, data)
                return data

        return None

    def exists(self, instance_id: str) -> bool:
        if self.local.exists(instance_id):
            return True
        return self.r2.exists(instance_id)

