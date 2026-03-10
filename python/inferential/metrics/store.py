from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass


@dataclass
class MetricPoint:
    timestamp: float
    value: float
    labels: dict[str, str]


@dataclass
class MetricStats:
    avg: float
    p50: float
    p95: float
    p99: float
    count: int
    min: float
    max: float


class MetricSeries:
    def __init__(self, max_size: int = 10000) -> None:
        self._buffer: deque[MetricPoint] = deque(maxlen=max_size)

    def record(self, value: float, labels: dict[str, str] | None = None) -> MetricPoint:
        point = MetricPoint(
            timestamp=time.monotonic(),
            value=value,
            labels=labels or {},
        )
        self._buffer.append(point)
        return point

    def get_latest(self, labels: dict[str, str] | None = None) -> float | None:
        for point in reversed(self._buffer):
            if labels is None or _labels_match(point.labels, labels):
                return point.value
        return None

    def get_stats(
        self,
        labels: dict[str, str] | None = None,
        window_seconds: int = 300,
    ) -> MetricStats | None:
        cutoff = time.monotonic() - window_seconds
        values = [
            p.value
            for p in self._buffer
            if p.timestamp >= cutoff and (labels is None or _labels_match(p.labels, labels))
        ]
        if not values:
            return None
        values.sort()
        n = len(values)
        return MetricStats(
            avg=sum(values) / n,
            p50=values[n // 2],
            p95=values[int(n * 0.95)],
            p99=values[int(n * 0.99)],
            count=n,
            min=values[0],
            max=values[-1],
        )

    def __len__(self) -> int:
        return len(self._buffer)


def _labels_match(point_labels: dict[str, str], query_labels: dict[str, str]) -> bool:
    return all(point_labels.get(k) == v for k, v in query_labels.items())
