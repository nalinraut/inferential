from __future__ import annotations

import enum
import time
from dataclasses import dataclass, field


class ResponseState(enum.Enum):
    PENDING = "pending"
    ISSUED = "issued"
    CONSUMED = "consumed"
    INVALIDATED = "invalidated"
    EXPIRED = "expired"


@dataclass
class ResponseRecord:
    client_id: str
    response_id: str
    state: ResponseState = ResponseState.PENDING
    created_at: float = field(default_factory=time.monotonic)
    issued_at: float | None = None
    latency_ms: float | None = None


class ResponseTracker:
    def __init__(self, disconnect_timeout_s: float = 10.0) -> None:
        self._disconnect_timeout_s = disconnect_timeout_s
        self._active: dict[str, ResponseRecord] = {}  # client_id -> latest
        self._connected: set[str] = set()

    def on_request(self, client_id: str) -> None:
        if client_id not in self._connected:
            self._connected.add(client_id)
        # Previous response is now consumed
        if client_id in self._active:
            prev = self._active[client_id]
            if prev.state == ResponseState.ISSUED:
                prev.state = ResponseState.CONSUMED

    def on_response_sent(self, client_id: str, response_id: str, latency_ms: float) -> None:
        record = ResponseRecord(
            client_id=client_id,
            response_id=response_id,
            state=ResponseState.ISSUED,
            issued_at=time.monotonic(),
            latency_ms=latency_ms,
        )
        self._active[client_id] = record

    def on_invalidated(self, client_id: str, reason: str) -> None:
        if client_id in self._active:
            self._active[client_id].state = ResponseState.INVALIDATED

    def get_active(self, client_id: str) -> ResponseRecord | None:
        return self._active.get(client_id)

    @property
    def connected_clients(self) -> set[str]:
        return self._connected.copy()
