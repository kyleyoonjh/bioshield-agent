"""
MASTER_PLAN §3.38 — Event Contract implementation.

All events are immutable dataclasses. The EventBus manages per-job channels
so the SSE endpoint can consume a typed stream instead of raw dicts.
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Literal

EventType = Literal[
    "TOOL_PROGRESS",
    "COMPLETED",
    "ERROR",
    "KEEPALIVE",
]


@dataclass(frozen=True)
class AgentEvent:
    type: EventType
    job_id: str
    payload: dict[str, Any]
    timestamp: float = field(default_factory=time.time)

    def to_sse_dict(self) -> dict:
        """Flatten to the dict shape the frontend EventSource consumer expects."""
        d = {"event": self.type, **self.payload}
        return d


class EventBus:
    """Per-job asyncio.Queue wrapper with typed events and sentinel-based close."""

    def __init__(self) -> None:
        self._queues: dict[str, asyncio.Queue[AgentEvent | None]] = {}

    def create_channel(self, job_id: str) -> None:
        self._queues[job_id] = asyncio.Queue()

    def has_channel(self, job_id: str) -> bool:
        return job_id in self._queues

    def publish(self, event: AgentEvent) -> None:
        q = self._queues.get(event.job_id)
        if q is not None:
            q.put_nowait(event)

    def publish_raw(self, job_id: str, type: EventType, payload: dict) -> None:
        self.publish(AgentEvent(type=type, job_id=job_id, payload=payload))

    async def consume(self, job_id: str, timeout: float = 120.0) -> AgentEvent | None:
        q = self._queues.get(job_id)
        if q is None:
            return None
        return await asyncio.wait_for(q.get(), timeout=timeout)

    def close_channel(self, job_id: str) -> None:
        q = self._queues.pop(job_id, None)
        if q is not None:
            q.put_nowait(None)  # sentinel — unblocks any waiting consumer


event_bus = EventBus()
