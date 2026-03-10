from __future__ import annotations

import time
from unittest.mock import patch

from inferential.config.schema import MetricsConfig
from inferential.metrics.callbacks import CallbackRegistry
from inferential.metrics.collector import MetricsCollector
from inferential.metrics.store import MetricSeries


class TestMetricSeries:
    def test_record_and_get_latest(self):
        s = MetricSeries()
        s.record(1.0)
        s.record(2.0)
        assert s.get_latest() == 2.0

    def test_get_latest_with_labels(self):
        s = MetricSeries()
        s.record(1.0, {"robot": "r1"})
        s.record(2.0, {"robot": "r2"})
        assert s.get_latest({"robot": "r1"}) == 1.0
        assert s.get_latest({"robot": "r2"}) == 2.0

    def test_get_latest_empty(self):
        s = MetricSeries()
        assert s.get_latest() is None

    def test_ring_buffer_max_size(self):
        s = MetricSeries(max_size=5)
        for i in range(10):
            s.record(float(i))
        assert len(s) == 5
        assert s.get_latest() == 9.0

    def test_get_stats(self):
        s = MetricSeries()
        for v in [1.0, 2.0, 3.0, 4.0, 5.0]:
            s.record(v)

        stats = s.get_stats()
        assert stats is not None
        assert stats.avg == 3.0
        assert stats.min == 1.0
        assert stats.max == 5.0
        assert stats.count == 5
        assert stats.p50 == 3.0

    def test_get_stats_with_window(self):
        s = MetricSeries()

        # Record in the past (beyond window)
        old_time = time.monotonic() - 600
        with patch.object(time, "monotonic", return_value=old_time):
            s.record(100.0)

        # Record now
        s.record(5.0)

        stats = s.get_stats(window_seconds=300)
        assert stats is not None
        assert stats.count == 1
        assert stats.avg == 5.0


class TestCallbackRegistry:
    def test_fire_callbacks(self):
        registry = CallbackRegistry()
        called = []
        registry.register(lambda name, val, labels: called.append((name, val)))

        registry.fire("latency", 12.5, {})
        assert len(called) == 1
        assert called[0] == ("latency", 12.5)

    def test_callback_exception_ignored(self):
        registry = CallbackRegistry()

        def bad_callback(name, val, labels):
            raise RuntimeError("boom")

        called = []
        registry.register(bad_callback)
        registry.register(lambda name, val, labels: called.append(name))

        registry.fire("metric", 1.0, {})
        # Second callback should still fire
        assert called == ["metric"]


class TestMetricsCollector:
    def test_record_and_query(self):
        collector = MetricsCollector(MetricsConfig())
        collector.record("latency", 10.0)
        collector.record("latency", 20.0)
        assert collector.get_latest("latency") == 20.0

    def test_get_stats(self):
        collector = MetricsCollector(MetricsConfig())
        for v in [10.0, 20.0, 30.0]:
            collector.record("latency", v)
        stats = collector.get_stats("latency")
        assert stats is not None
        assert stats.avg == 20.0

    def test_on_metric_callback(self):
        collector = MetricsCollector(MetricsConfig())
        calls = []

        @collector.on_metric
        def on_metric(name, value, labels):
            calls.append((name, value))

        collector.record("latency", 42.0)
        assert calls == [("latency", 42.0)]

    def test_disabled_metrics(self):
        collector = MetricsCollector(MetricsConfig(enabled=False))
        collector.record("latency", 10.0)
        assert collector.get_latest("latency") is None

    def test_snapshot(self):
        collector = MetricsCollector(MetricsConfig())
        collector.record("a", 1.0)
        collector.record("b", 2.0)
        snap = collector.snapshot()
        assert "a" in snap.metrics
        assert "b" in snap.metrics
        assert len(snap.metrics["a"]) == 1

    def test_labels_filtering(self):
        collector = MetricsCollector(MetricsConfig())
        collector.record("latency", 10.0, {"robot": "r1"})
        collector.record("latency", 20.0, {"robot": "r2"})
        assert collector.get_latest("latency", {"robot": "r1"}) == 10.0
        assert collector.get_latest("latency", {"robot": "r2"}) == 20.0
