"""
SWE-INFINITE Cache Module (Read-Only)

Two-level cache for reading expansion tasks (produced by the mining pipeline):
  L1: Local filesystem (fast, per-machine)
  L2: R2 via S3 API (private bucket, requires credentials)
"""

import json
import os
from pathlib import Path
from typing import Optional, Dict, Any


class LocalCache:
    """Local filesystem cache keyed by instance_id."""

    def __init__(self, cache_dir: str = "/tmp/swe-infinite-cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_path(self, instance_id: str) -> Path:
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


class R2Cache:
    """Read-only R2 cache via S3 API (private bucket)."""

    def __init__(
        self,
        endpoint: Optional[str] = None,
        access_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        bucket: str = "swe-infinite",
        prefix: str = "",
    ):
        self.bucket = bucket
        self.prefix = prefix
        self._client = None

        endpoint = endpoint or os.getenv("R2_ENDPOINT")
        access_key = access_key or os.getenv("R2_ACCESS_KEY")
        secret_key = secret_key or os.getenv("R2_SECRET_KEY")

        if endpoint and access_key and secret_key:
            try:
                import boto3
                self._client = boto3.client(
                    "s3",
                    endpoint_url=endpoint,
                    aws_access_key_id=access_key,
                    aws_secret_access_key=secret_key,
                    region_name="auto",
                )
                print(f"[CACHE] R2 connected: {bucket}/{prefix}")
            except Exception as e:
                print(f"[CACHE] R2 setup failed ({e}), local only")
        else:
            print("[CACHE] R2 credentials not provided, local only")

    @property
    def enabled(self) -> bool:
        return self._client is not None

    def _get_key(self, instance_id: str) -> str:
        if self.prefix:
            return f"{self.prefix}/{instance_id}.json"
        return f"{instance_id}.json"

    def load(self, instance_id: str) -> Optional[Dict[str, Any]]:
        if not self._client:
            return None
        try:
            resp = self._client.get_object(
                Bucket=self.bucket,
                Key=self._get_key(instance_id),
            )
            return json.loads(resp["Body"].read())
        except self._client.exceptions.NoSuchKey:
            return None
        except Exception as e:
            print(f"[CACHE] R2 read error for {instance_id}: {e}")
            return None

    def exists(self, instance_id: str) -> bool:
        if not self._client:
            return False
        try:
            self._client.head_object(
                Bucket=self.bucket,
                Key=self._get_key(instance_id),
            )
            return True
        except Exception:
            return False

    def list_tasks(self, max_keys: int = 1000) -> list[str]:
        """List available instance_ids in R2."""
        if not self._client:
            return []
        try:
            list_kwargs = {"Bucket": self.bucket, "MaxKeys": max_keys}
            if self.prefix:
                list_kwargs["Prefix"] = f"{self.prefix}/"
            resp = self._client.list_objects_v2(**list_kwargs)
            strip_len = len(self.prefix) + 1 if self.prefix else 0
            ids = []
            for obj in resp.get("Contents", []):
                key = obj["Key"]
                if key.endswith(".json"):
                    ids.append(key[strip_len:-5])
            return ids
        except Exception as e:
            print(f"[CACHE] R2 list error: {e}")
            return []


class TwoLevelCache:
    """Two-level read cache: Local (L1) + R2 S3 (L2).

    Read path: L1 hit -> return | L2 hit -> save to L1, return | miss -> None
    """

    def __init__(
        self,
        local_cache_dir: str = "/tmp/swe-infinite-cache",
        r2_endpoint: Optional[str] = None,
        r2_access_key: Optional[str] = None,
        r2_secret_key: Optional[str] = None,
        r2_bucket: str = "swe-infinite",
        r2_prefix: str = "",
    ):
        self.local = LocalCache(local_cache_dir)
        self.r2 = R2Cache(
            endpoint=r2_endpoint,
            access_key=r2_access_key,
            secret_key=r2_secret_key,
            bucket=r2_bucket,
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

    def list_tasks(self, max_keys: int = 1000) -> list[str]:
        """List available task instance_ids."""
        return self.r2.list_tasks(max_keys)
