from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class EndpointHealth:
    model_id: str
    avg_latency_ms: float = 0.0
    error_rate: float = 0.0
    total_requests: int = 0
    last_error: str | None = None
    _alpha: float = field(default=0.2, repr=False)

    def update(self, latency_ms: float, success: bool) -> None:
        self.total_requests += 1
        self.avg_latency_ms = self._alpha * latency_ms + (1 - self._alpha) * self.avg_latency_ms
        error_val = 0.0 if success else 1.0
        self.error_rate = self._alpha * error_val + (1 - self._alpha) * self.error_rate
        if not success:
            self.last_error = f"request #{self.total_requests}"
