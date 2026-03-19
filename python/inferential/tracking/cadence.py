from __future__ import annotations

import time
from collections import defaultdict


class CadenceTracker:
    def __init__(self, alpha: float = 0.3, overdue_multiplier: float = 1.5) -> None:
        self.alpha = alpha
        self.overdue_multiplier = overdue_multiplier
        self._cadence: dict[str, float] = {}
        self._last_request: dict[str, float] = {}
        self._sample_count: dict[str, int] = defaultdict(int)

    def on_request(self, client_id: str) -> None:
        now = time.monotonic()
        if client_id in self._last_request:
            interval = now - self._last_request[client_id]
            if client_id in self._cadence:
                self._cadence[client_id] = (
                    self.alpha * interval + (1 - self.alpha) * self._cadence[client_id]
                )
            else:
                self._cadence[client_id] = interval
            self._sample_count[client_id] += 1
        self._last_request[client_id] = now

    def time_until_next(self, client_id: str) -> float:
        if client_id not in self._cadence:
            return 0.0  # Unknown cadence, assume urgent
        elapsed = time.monotonic() - self._last_request[client_id]
        return self._cadence[client_id] - elapsed

    def is_overdue(self, client_id: str) -> bool:
        if client_id not in self._cadence:
            return False
        elapsed = time.monotonic() - self._last_request[client_id]
        return elapsed > self._cadence[client_id] * self.overdue_multiplier

    def time_since_last(self, client_id: str) -> float:
        if client_id not in self._last_request:
            return float("inf")
        return time.monotonic() - self._last_request[client_id]

    def learned_cadence(self, client_id: str) -> float | None:
        return self._cadence.get(client_id)

    def sample_count(self, client_id: str) -> int:
        return self._sample_count.get(client_id, 0)

    def confidence(self, client_id: str) -> float:
        count = self._sample_count.get(client_id, 0)
        return min(count / 10.0, 1.0)
