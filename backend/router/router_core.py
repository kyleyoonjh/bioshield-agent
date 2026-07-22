"""
Model Router core — AiRemedy-Agent.

Tiered dispatch sitting between the two pipeline Executors
(services/assay_orchestrator.py, services/pipeline.py) and the deterministic
Engine layer (services/*.py), which they previously called directly:

  Tier 1 (sub-second)  — static pre-indexed lookup in knowledge/gene_index.json
  Tier 2 (local engine) — registry/tool_registry.py's deterministic tool calls
  Tier 3 (async AI)     — LLM calls with a hard timeout budget + graceful
                          degradation (route_async, not yet wired into any
                          live call site — see PLAN.md scope notes)

Tier 2 exceptions are intentionally left to propagate uncaught: both
orchestrators already catch failures at the pipeline level and expect the
real exception, not a wrapped error object.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from typing import Any, Awaitable, Callable

from registry import tool_registry
from router.interfaces import EngineResult

logger = logging.getLogger(__name__)

_KNOWLEDGE_PATH = os.path.join(os.path.dirname(__file__), "..", "knowledge", "gene_index.json")
_knowledge_cache: dict | None = None


def _load_knowledge() -> dict:
    global _knowledge_cache
    if _knowledge_cache is None:
        try:
            with open(_KNOWLEDGE_PATH, "r", encoding="utf-8") as f:
                _knowledge_cache = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as exc:
            logger.warning("[router] knowledge index unavailable: %s", exc)
            _knowledge_cache = {}
    return _knowledge_cache


def _tier1_lookup(cache_key: tuple[str, str] | None) -> dict | None:
    """Static pre-indexed cache lookup. cache_key = (organism, gene)."""
    if not cache_key:
        return None
    organism, gene = cache_key
    return _load_knowledge().get(organism, {}).get("cached_designs", {}).get(gene)


def route(tool_name: str, cache_key: tuple[str, str] | None = None, **kwargs: Any) -> EngineResult:
    """
    Dispatch one Engine call through the tiered router.

    cache_key, when supplied by the caller, is checked against the Tier-1
    static knowledge cache before falling through to the Tier-2 registry
    call. No orchestrator wires cache_key today — the demo cache entry's
    shape (visualization-oriented) doesn't match the Primer3 candidate-list
    shape downstream steps expect, so forcing a hit there would silently
    corrupt pipeline output. It's available for call sites that know their
    cached shape matches.
    """
    t0 = time.perf_counter()

    cached = _tier1_lookup(cache_key)
    if cached is not None:
        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.info("[router] tier=1 cache hit | tool=%s key=%s", tool_name, cache_key)
        return EngineResult(tool_name=tool_name, tier=1, source="cache", data=cached, elapsed_ms=elapsed_ms)

    result = tool_registry.call(tool_name, **kwargs)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    logger.info("[router] tier=2 engine call | tool=%s elapsed=%.1fms", tool_name, elapsed_ms)
    return EngineResult(tool_name=tool_name, tier=2, source="engine", data=result, elapsed_ms=elapsed_ms)


async def route_async(
    tool_name: str,
    coro_factory: Callable[[], Awaitable[Any]],
    timeout: float = 30.0,
) -> EngineResult:
    """
    Tier-3 dispatch for async AI calls. Wraps an arbitrary awaitable-producing
    factory with a hard timeout budget; on timeout, degrades gracefully
    (returns an EngineResult with degraded=True) instead of hanging the
    caller. Not currently invoked by agent/__init__.py's LLM call sites —
    that wiring is out of scope for this pass (see PLAN.md).
    """
    t0 = time.perf_counter()
    try:
        result = await asyncio.wait_for(coro_factory(), timeout=timeout)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.info("[router] tier=3 llm call OK | tool=%s elapsed=%.1fms", tool_name, elapsed_ms)
        return EngineResult(tool_name=tool_name, tier=3, source="llm", data=result, elapsed_ms=elapsed_ms)
    except asyncio.TimeoutError:
        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.warning("[router] tier=3 TIMEOUT (%.0fs budget) | tool=%s", timeout, tool_name)
        return EngineResult(
            tool_name=tool_name, tier=3, source="llm", elapsed_ms=elapsed_ms,
            degraded=True, error=f"exceeded {timeout:.0f}s budget",
        )
