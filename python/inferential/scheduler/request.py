from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class InferenceRequest:
    client_id: str
    model_id: str
    identity: bytes
    envelope: bytes
    payload: bytes
    received_at: float
    urgency: float
    steps_remaining: int | None
    time_until_next: float
    latency_budget_ms: float
    priority: int
    score: float = 0.0
    retry_count: int = 0
    _counter: int = field(default=0, repr=False, compare=False)

    def __lt__(self, other: InferenceRequest) -> bool:
        # For heapq: higher score = higher priority, so negate
        # Tie-break by counter (FIFO)
        if self.score != other.score:
            return self.score > other.score
        return self._counter < other._counter
