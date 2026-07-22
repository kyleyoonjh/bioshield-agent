"""
Model Router — unified interface layer, AiRemedy-Agent.

Defines the single JSON-serializable envelope every tool/engine/LLM call
returns through router_core.route(), regardless of which tier served it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class EngineResult:
    tool_name: str
    tier:      int              # 1 = cache, 2 = local engine, 3 = async AI
    source:    str               # "cache" | "engine" | "llm"
    data:      Any               = None
    elapsed_ms: float             = 0.0
    degraded:  bool              = False
    error:     str | None        = None

    def to_dict(self) -> dict:
        return {
            "tool_name":  self.tool_name,
            "tier":       self.tier,
            "source":     self.source,
            "data":       self.data,
            "elapsed_ms": round(self.elapsed_ms, 2),
            "degraded":   self.degraded,
            "error":      self.error,
        }
