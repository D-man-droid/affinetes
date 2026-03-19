"""
MCP Wrapper - Reuses QQR's MCP code logic.

Core MCP functionality copied from QQR project to avoid slime dependency.
Sources:
- qqr/mcp/server.py - MCPServerStdioCacheable
- qqr/mcp/utils.py - get_mcp_tools
- qqr/rollout/agent_rollout.py - MCPState
"""

import asyncio
import hashlib
import json
import logging
import os
from typing import Any, Callable, Dict, List, Optional, Union

import diskcache
from agents.mcp import MCPServer, MCPUtil
from agents.mcp.server import MCPServerStdio, MCPServerStdioParams
from agents.models.chatcmpl_converter import Converter
from mcp.types import CallToolResult
from mcp.types import Tool as MCPTool
from openai.types.chat.chat_completion_tool_param import ChatCompletionToolParam

# Cache directory configuration (inside container)
CACHE_DIR = os.getenv("QQR_CACHE_DIR", "/var/lib/qqr/cache")

logger = logging.getLogger(__name__)

__all__ = [
    "MCPServerStdioParams",
    "MCPServerStdioCacheable",
    "MCPState",
]


# ==================== Copied from qqr/mcp/utils.py ====================
async def get_mcp_tools(mcp_server: MCPServer) -> List[ChatCompletionToolParam]:
    """Convert MCP server tools to OpenAI format."""
    server_tools = await mcp_server.list_tools()
    server_tools = [
        MCPUtil.to_function_tool(
            MCPTool(
                name=info.name,
                title=info.title,
                description=info.description,
                inputSchema=info.inputSchema,
                outputSchema=info.outputSchema,
                annotations=info.annotations,
            ),
            mcp_server,
            convert_schemas_to_strict=False,
        )
        for info in server_tools
    ]
    converted_tools = [Converter.tool_to_openai(tool) for tool in server_tools]
    return converted_tools


# ==================== MCPServerCacheableMixin with diskcache ====================
class MCPServerCacheableMixin:
    """
    A Mixin that adds tool result caching capabilities and concurrency control.

    Uses diskcache for:
    - Local persistence (survives restarts)
    - Multi-process sharing (SQLite-based)
    - TTL expiration support

    Features:
    - cache_key_salt: prefix for cache keys (e.g., epoch salt for anti-memorization)
    - key_normalizer: callback to normalize arguments before keying (e.g., round coordinates)
    - cache_ttl: fixed int OR callable returning int (for dynamic per-entry TTL)
    - Cache hit/miss metrics via get_cache_stats()
    """

    def __init__(
        self,
        blocklist: set = None,
        cache_ttl: Union[int, Callable[[], int]] = 600,
        cache_maxsize: int = 8192,
        concurrency_limit: int = 64,
        cache_key_salt: str = "",
        key_normalizer: Optional[Callable[[str, dict], dict]] = None,
        cache_size_limit_mb: Optional[int] = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        # Create separate cache directory for each server
        server_name = getattr(self, 'name', 'default')
        cache_path = os.path.join(CACHE_DIR, server_name)

        # Size limit: explicit MB override > maxsize * 4KB estimate
        if cache_size_limit_mb is not None:
            size_limit = cache_size_limit_mb * 1024 * 1024
        else:
            size_limit = cache_maxsize * 4096

        # Use diskcache for multi-process sharing
        self._tool_cache = diskcache.Cache(
            cache_path,
            size_limit=size_limit,
            eviction_policy='least-recently-used',
        )
        self._cache_ttl = cache_ttl
        self._cache_blocklist = blocklist or set()
        self._cache_key_salt = cache_key_salt
        self._key_normalizer = key_normalizer
        self.concurrency_limit = concurrency_limit
        self._semaphore = None

        # Cache metrics
        self._cache_hits = 0
        self._cache_misses = 0

    @property
    def semaphore(self) -> asyncio.Semaphore:
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self.concurrency_limit)
        return self._semaphore

    def _get_ttl(self) -> int:
        """Get current TTL value. Supports both fixed int and callable."""
        if callable(self._cache_ttl):
            return self._cache_ttl()
        return self._cache_ttl

    def _make_cache_key(self, tool_name: str, arguments: dict) -> str:
        # Apply normalizer if configured (with fallback on error)
        if self._key_normalizer and arguments:
            try:
                arguments = self._key_normalizer(tool_name, arguments)
            except Exception:
                pass  # Use original arguments on normalizer failure

        if arguments is None:
            key = tool_name
        else:
            args_str = json.dumps(arguments, sort_keys=True, ensure_ascii=False)
            key = f"{tool_name}:{args_str}"

        # Include salt prefix for cache isolation across epochs
        if self._cache_key_salt:
            key = f"{self._cache_key_salt}:{key}"

        if len(key) > 1024:
            return hashlib.md5(key.encode("utf-8")).hexdigest()
        return key

    @staticmethod
    def _serialize_result(result: CallToolResult) -> str:
        """Serialize CallToolResult to JSON string for cache storage.

        We do NOT pickle CallToolResult directly because:
        - Pydantic v2 models have unstable pickle behavior across versions
        - Production evidence: access_count=0 on all 4,336 cached entries
          proves pickle.loads() silently failed on every cache.get()
        - JSON is version-independent and human-debuggable
        """
        items = []
        for item in result.content:
            items.append(item.model_dump(mode="json"))
        return json.dumps({"content": items, "isError": result.isError}, ensure_ascii=False)

    @staticmethod
    def _deserialize_result(data: str) -> CallToolResult:
        """Deserialize JSON string back to CallToolResult."""
        from mcp.types import TextContent
        parsed = json.loads(data)
        content = []
        for item in parsed["content"]:
            content.append(TextContent(**item))
        return CallToolResult(content=content, isError=parsed["isError"])

    async def call_tool(
        self, tool_name: str, arguments: Dict[str, Any]
    ) -> CallToolResult:
        if tool_name in self._cache_blocklist:
            async with self.semaphore:
                return await super().call_tool(tool_name, arguments)

        cache_key = self._make_cache_key(tool_name, arguments)

        # Read from cache: stored as JSON string (not pickled Pydantic object)
        cached_json = self._tool_cache.get(cache_key, default=None)
        if cached_json is not None:
            try:
                self._cache_hits += 1
                return self._deserialize_result(cached_json)
            except Exception:
                # Corrupted entry — treat as miss, will be overwritten
                self._cache_hits -= 1

        self._cache_misses += 1

        async with self.semaphore:
            # Double-check after acquiring semaphore
            cached_json = self._tool_cache.get(cache_key, default=None)
            if cached_json is not None:
                try:
                    self._cache_hits += 1
                    return self._deserialize_result(cached_json)
                except Exception:
                    self._cache_hits -= 1

            result = await super().call_tool(tool_name, arguments)
            if not result.isError:
                try:
                    serialized = self._serialize_result(result)
                    self._tool_cache.set(cache_key, serialized, expire=self._get_ttl())
                except Exception as e:
                    logger.warning("Cache write failed for %s: %s", tool_name, e)

        return result

    def get_cache_stats(self) -> Dict[str, Any]:
        """Return cache hit/miss statistics for monitoring."""
        total = self._cache_hits + self._cache_misses
        try:
            disk_bytes = self._tool_cache.volume()
        except Exception:
            disk_bytes = 0
        return {
            "hits": self._cache_hits,
            "misses": self._cache_misses,
            "hit_rate": round(self._cache_hits / total, 4) if total > 0 else 0.0,
            "total_calls": total,
            "disk_size_mb": round(disk_bytes / 1024 / 1024, 1),
        }

    async def cleanup(self):
        await super().cleanup()
        self._semaphore = None
        # Close diskcache connection
        if hasattr(self, '_tool_cache') and self._tool_cache is not None:
            self._tool_cache.close()


class MCPServerStdioCacheable(MCPServerCacheableMixin, MCPServerStdio):
    """Cached and Rate-Limited version of MCPServerStdio."""
    pass


# ==================== Copied from qqr/rollout/agent_rollout.py ====================
class SingletonMeta(type):
    """Simple singleton metaclass."""
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]


class MCPState(metaclass=SingletonMeta):
    """
    The global state for the MCP server.
    Reused from qqr/rollout/agent_rollout.py
    """

    def __init__(self, mcp_server_config_fn: Callable) -> None:
        self._mcp_server_config_fn = mcp_server_config_fn
        self._mcp_servers: List[MCPServer] = None
        self._mcp_lock = asyncio.Lock()
        self.tools = []
        self.tool_to_server: Dict[str, MCPServer] = {}

    async def get_mcp_servers(self) -> List[MCPServer]:
        """Thread-safe lazy initialization of the MCP server."""
        if self._mcp_servers is None:
            async with self._mcp_lock:
                if self._mcp_servers is None:
                    try:
                        servers = self._mcp_server_config_fn()
                        for server in servers:
                            await server.connect()
                            converted_tools = await get_mcp_tools(server)
                            self.tools += converted_tools
                            for tool in converted_tools:
                                self.tool_to_server[tool["function"]["name"]] = server
                            logger.info(f"MCP Server {server.name} connected successfully.")
                        self._mcp_servers = servers
                    except Exception as e:
                        logger.error(f"Failed to initialize MCP Servers: {e}")
                        self._mcp_servers = None
                        raise e
        return self._mcp_servers

    async def cleanup(self):
        """Shut down all MCP servers and release resources."""
        if self._mcp_servers:
            for server in self._mcp_servers:
                try:
                    await asyncio.wait_for(server.cleanup(), timeout=5)
                    logger.info(f"MCP Server {server.name} cleaned up.")
                except BaseException as e:
                    logger.warning(f"Error cleaning up MCP Server {server.name}: {e}")
            self._mcp_servers = None
            self.tools = []
            self.tool_to_server = {}

    async def call_tool(self, tool_call: dict) -> dict:
        """Call a tool by name with arguments. Auto-retries on subprocess crash."""
        return await self._call_tool_inner(tool_call, allow_retry=True)

    async def _call_tool_inner(self, tool_call: dict, allow_retry: bool) -> dict:
        """Inner implementation of call_tool with optional crash recovery."""
        await self.get_mcp_servers()

        tool_name = tool_call["function"]["name"]
        tool_call_id = tool_call["id"]
        tool_content = ""

        target_server = self.tool_to_server.get(tool_name)

        if not target_server:
            return {
                "role": "tool",
                "content": f"[Error] Tool '{tool_name}' not found in any connected MCP servers.",
                "tool_call_id": tool_call_id,
            }

        try:
            tool_arguments_str = tool_call["function"]["arguments"]
            tool_arguments = (
                json.loads(tool_arguments_str) if tool_arguments_str else {}
            )

            result = await target_server.call_tool(tool_name, tool_arguments)

            if len(result.content) == 1:
                tool_content = result.content[0].model_dump_json()
            elif len(result.content) > 1:
                tool_results = [item.model_dump(mode="json") for item in result.content]
                tool_content = json.dumps(tool_results, ensure_ascii=False, indent=4)
            else:
                tool_content = "[]"
        except json.JSONDecodeError as e:
            tool_content = f"[Error] Invalid JSON arguments: {e}"
        except (ConnectionError, BrokenPipeError, OSError) as e:
            if allow_retry:
                logger.warning("MCP subprocess connection lost (%s), resetting and retrying", e)
                await self._cleanup_and_reset()
                return await self._call_tool_inner(tool_call, allow_retry=False)
            tool_content = f"[Error] MCP connection failed after retry: {e}"
        except Exception as e:
            # Check if the exception message indicates a connection-type error
            err_msg = str(e).lower()
            if allow_retry and any(kw in err_msg for kw in ("broken pipe", "connection", "eof", "transport")):
                logger.warning("MCP subprocess likely crashed (%s), resetting and retrying", e)
                await self._cleanup_and_reset()
                return await self._call_tool_inner(tool_call, allow_retry=False)
            tool_content = f"[Error] Tool execution failed: {e}"

        return {
            "role": "tool",
            "content": tool_content,
            "tool_call_id": tool_call_id,
        }

    async def _cleanup_and_reset(self):
        """Cleanup old servers before resetting state (prevents resource leaks)."""
        async with self._mcp_lock:
            if self._mcp_servers:
                for server in self._mcp_servers:
                    try:
                        await asyncio.wait_for(server.cleanup(), timeout=5)
                    except BaseException as e:
                        logger.warning("Cleanup during reset failed for %s: %s",
                                       getattr(server, 'name', '?'), e)
            self._mcp_servers = None
            self.tools = []
            self.tool_to_server = {}
