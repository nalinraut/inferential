from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from inferential.metrics.callbacks import CallbackRegistry, MetricCallback
from inferential.metrics.store import MetricSeries, MetricStats

if TYPE_CHECKING:
    from inferential.config.schema import MetricsConfig


@dataclass
class MetricSnapshot:
    metrics: dict[str, list[dict]]


class MetricsCollector:
    def __init__(self, config: MetricsConfig | None = None) -> None:
        self._ring_buffer_size = config.ring_buffer_size if config else 10000
        self._enabled = config.enabled if config else True
        self._series: dict[str, MetricSeries] = {}
        self._callbacks = CallbackRegistry()

    def record(
        self,
        name: str,
        value: float,
        labels: dict[str, str] | None = None,
    ) -> None:
        if not self._enabled:
            return
        labels = labels or {}
        if name not in self._series:
            self._series[name] = MetricSeries(max_size=self._ring_buffer_size)
        self._series[name].record(value, labels)
        self._callbacks.fire(name, value, labels)

    def get_latest(
        self,
        name: str,
        labels: dict[str, str] | None = None,
    ) -> float | None:
        series = self._series.get(name)
        if series is None:
            return None
        return series.get_latest(labels)

    def get_stats(
        self,
        name: str,
        labels: dict[str, str] | None = None,
        window_seconds: int = 300,
    ) -> MetricStats | None:
        series = self._series.get(name)
        if series is None:
            return None
        return series.get_stats(labels, window_seconds)

    def snapshot(self) -> MetricSnapshot:
        result: dict[str, list[dict]] = {}
        for name, series in self._series.items():
            result[name] = [
                {"timestamp": p.timestamp, "value": p.value, "labels": p.labels}
                for p in series._buffer
            ]
        return MetricSnapshot(metrics=result)

    def on_metric(self, callback: MetricCallback) -> MetricCallback:
        self._callbacks.register(callback)
        return callback
