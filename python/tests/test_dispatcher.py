from __future__ import annotations

from inferential.dispatch.health import EndpointHealth


class TestEndpointHealth:
    def test_initial_state(self):
        h = EndpointHealth(model_id="test-model")
        assert h.avg_latency_ms == 0.0
        assert h.error_rate == 0.0
        assert h.total_requests == 0

    def test_update_success(self):
        h = EndpointHealth(model_id="test-model")
        h.update(10.0, success=True)
        assert h.total_requests == 1
        assert h.avg_latency_ms > 0
        assert h.error_rate == 0.0

    def test_update_failure(self):
        h = EndpointHealth(model_id="test-model")
        h.update(10.0, success=False)
        assert h.error_rate > 0
        assert h.last_error is not None

    def test_ewma_latency(self):
        h = EndpointHealth(model_id="test-model", _alpha=0.5)
        h.update(10.0, success=True)
        h.update(20.0, success=True)
        # EWMA: first = 0.5*10 + 0.5*0 = 5.0, second = 0.5*20 + 0.5*5 = 12.5
        assert abs(h.avg_latency_ms - 12.5) < 0.01

    def test_error_rate_recovery(self):
        h = EndpointHealth(model_id="test-model", _alpha=0.5)
        h.update(10.0, success=False)  # error_rate = 0.5
        h.update(10.0, success=True)  # error_rate = 0.25
        h.update(10.0, success=True)  # error_rate = 0.125
        assert h.error_rate < 0.2
