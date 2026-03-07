"""Abstract Base Class for Fixer Agents"""

import os
import subprocess
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List

from utils import SANITIZE_GIT_SCRIPT, NORMALIZE_TIMESTAMPS_SCRIPT


@dataclass
class FixerConfig:
    """Configuration for fixer agents"""
    model: str
    api_base: str
    api_key: str
    temperature: float = 0.0
    max_iterations: int = 100
    cost_limit: float = 3.0
    timeout: int = 300
    seed: Optional[int] = None
    cwd: str = "/app"

    # Reserved for future agents
    swe_agent_config: Optional[Dict[str, Any]] = None
    external_agent_endpoint: Optional[str] = None
    model_kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FixerResult:
    """Result from fixer agent execution"""
    patch: str
    model_calls: int = 0
    model_cost: float = 0.0
    total_tokens: int = 0
    conversation: List[Any] = field(default_factory=list)
    success: bool = True
    error: Optional[str] = None


class BaseFixerAgent(ABC):
    """Abstract base class for fixer agents"""

    def __init__(self, config: FixerConfig):
        self.config = config
        self._container_name: Optional[str] = None

    def _prepare_container(
        self,
        gold_patch: Optional[str],
        bug_patch: Optional[str],
    ) -> None:
        """Apply patches, sanitize git history, and normalize timestamps.

        Must be called after self._container_name is set and the container is running.
        """
        patch_names = ["gold_patch", "bug_patch"]
        for idx, patch in enumerate([gold_patch, bug_patch]):
            if not patch:
                continue
            with tempfile.NamedTemporaryFile(mode='w', suffix='.diff', delete=False) as f:
                f.write(patch)
                temp_path = f.name
            try:
                subprocess.run(
                    ["docker", "cp", temp_path, f"{self._container_name}:/tmp/patch_{idx}.diff"],
                    check=True, capture_output=True, timeout=30,
                )
                result = subprocess.run(
                    ["docker", "exec", self._container_name, "bash", "-c",
                     f"cd /app && git apply -v /tmp/patch_{idx}.diff 2>&1"],
                    capture_output=True, text=True, timeout=120,
                )
                if result.returncode != 0:
                    print(f"[SWE-SYNTH] Warning: {patch_names[idx]} may have failed: {result.stdout[:500]}")
                else:
                    print(f"[SWE-SYNTH] {patch_names[idx]} applied successfully")
            finally:
                os.unlink(temp_path)

        result = subprocess.run(
            ["docker", "exec", self._container_name, "bash", "-c", SANITIZE_GIT_SCRIPT],
            capture_output=True, text=True, timeout=60,
        )
        print(f"[SWE-SYNTH] Git history sanitized: {result.stdout[:200]}")

        # Warm up the login shell so conda activation and .pyc compilation happen
        # before normalization. The agent runs commands via "bash -lc", which sources
        # /etc/profile.d scripts and activates the conda env, potentially creating
        # __pycache__/*.pyc files with current timestamps. Running a login shell here
        # ensures those files exist before we normalize.
        subprocess.run(
            ["docker", "exec", self._container_name, "bash", "-lc", "true"],
            capture_output=True, text=True, timeout=60,
        )

        subprocess.run(
            ["docker", "exec", self._container_name, "bash", "-lc", NORMALIZE_TIMESTAMPS_SCRIPT],
            capture_output=True, text=True, timeout=120,
        )
        print("[SWE-SYNTH] Timestamps normalized")

    @abstractmethod
    async def fix(
        self,
        problem_statement: str,
        docker_image: str,
        repo_path: Optional[str] = None,
        gold_patch: Optional[str] = None,
        bug_patch: Optional[str] = None,
        base_commit: Optional[str] = None,
    ) -> FixerResult:
        """
        Run the fixer agent to repair a bug.

        The agent sees code in state: base_commit + gold_patch + bug_patch
        and must produce a fix_patch to restore correct behavior.
        """
        pass

    @abstractmethod
    def cleanup(self):
        """Clean up resources (Docker containers, temp files, etc.)"""
        pass
