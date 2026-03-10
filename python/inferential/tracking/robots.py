from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class ClientState:
    client_id: str
    first_seen: float = field(default_factory=time.monotonic)
    last_seen: float = field(default_factory=time.monotonic)
    total_requests: int = 0
    model_id: str | None = None


class ClientRegistry:
    def __init__(self) -> None:
        self._clients: dict[str, ClientState] = {}

    def on_observation(self, client_id: str, model_id: str) -> bool:
        """Record observation. Returns True if this is a new client."""
        is_new = client_id not in self._clients
        if is_new:
            self._clients[client_id] = ClientState(client_id=client_id, model_id=model_id)
        else:
            state = self._clients[client_id]
            state.last_seen = time.monotonic()
            state.total_requests += 1
            state.model_id = model_id
        return is_new

    def get(self, client_id: str) -> ClientState | None:
        return self._clients.get(client_id)

    @property
    def active_count(self) -> int:
        return len(self._clients)

    @property
    def all_clients(self) -> dict[str, ClientState]:
        return self._clients.copy()
