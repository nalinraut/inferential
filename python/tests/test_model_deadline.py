from __future__ import annotations

import time

import pytest

from inferential.config.schema import SchedulingConfig
from inferential.scheduler.base import ModelAwareScheduler, QueueFullError, create_scheduler
from inferential.scheduler.model_deadline import ModelDeadlineScheduler
from inferential.scheduler.request import InferenceRequest


def _make_request(
    client_id: str = "1",
    model_id: str = "policy",
    urgency: float = 0.0,
    priority: int = 1,
    steps_remaining: int | None = None,
    time_until_next: float = 0.0,
    received_at: float | None = None,
) -> InferenceRequest:
    return InferenceRequest(
        client_id=client_id,
        model_id=model_id,
        identity=client_id.encode(),
        envelope=b"",
        payload=b"",
        received_at=received_at or time.monotonic(),
        urgency=urgency,
        steps_remaining=steps_remaining,
        time_until_next=time_until_next,
        latency_budget_ms=50.0,
        priority=priority,
    )


class TestModelRouting:
    def test_routes_by_model_id(self):
        config = SchedulingConfig(strategy="model_deadline")
        sched = ModelDeadlineScheduler(config)
        sched.submit(_make_request(client_id="1", model_id="policy"))
        sched.submit(_make_request(client_id="2", model_id="telemetry"))
        assert sched.queue_len_for_model("policy") == 1
        assert sched.queue_len_for_model("telemetry") == 1

    def test_same_model_accumulates(self):
        config = SchedulingConfig(strategy="model_deadline")
        sched = ModelDeadlineScheduler(config)
        sched.submit(_make_request(client_id="1", model_id="policy"))
        sched.submit(_make_request(client_id="2", model_id="policy"))
        assert sched.queue_len_for_model("policy") == 2

    def test_models_returns_active(self):
        config = SchedulingConfig(strategy="model_deadline")
        sched = ModelDeadlineScheduler(config)
        sched.submit(_make_request(model_id="policy"))
        sched.submit(_make_request(model_id="telemetry"))
        assert sorted(sched.models()) == ["policy", "telemetry"]

    def test_unknown_model_returns_zero(self):
        config = SchedulingConfig(strategy="model_deadline")
        sched = ModelDeadlineScheduler(config)
        assert sched.queue_len_for_model("nonexistent") == 0


class TestModelIsolation:
    def test_model_drains_independently(self):
        config = SchedulingConfig(strategy="model_deadline")
        sched = ModelDeadlineScheduler(config)
        sched.submit(_make_request(client_id="p", model_id="policy"))
        sched.submit(_make_request(client_id="t", model_id="telemetry"))

        batch = sched.next_batch_for_model("policy")
        assert len(batch) == 1
        assert batch[0].client_id == "p"
        assert sched.queue_len_for_model("policy") == 0
        assert sched.queue_len_for_model("telemetry") == 1

    def test_empty_model_returns_empty(self):
        config = SchedulingConfig(strategy="model_deadline")
        sched = ModelDeadlineScheduler(config)
        assert sched.next_batch_for_model("policy") == []

    def test_depleted_model_removed_from_models(self):
        config = SchedulingConfig(strategy="model_deadline")
        sched = ModelDeadlineScheduler(config)
        sched.submit(_make_request(model_id="policy"))
        sched.next_batch_for_model("policy")
        assert "policy" not in sched.models()


class TestScoringWithinModel:
    def test_urgency_ordering(self):
        config = SchedulingConfig(strategy="model_deadline")
        sched = ModelDeadlineScheduler(config)
        sched.submit(_make_request(client_id="low", model_id="policy", urgency=0.1))
        sched.submit(_make_request(client_id="high", model_id="policy", urgency=0.9))
        batch = sched.next_batch_for_model("policy")
        assert batch[0].client_id == "high"

    def test_priority_boosts_score(self):
        config = SchedulingConfig(strategy="model_deadline")
        sched = ModelDeadlineScheduler(config)
        sched.submit(_make_request(client_id="lo-pri", model_id="policy", priority=5))
        sched.submit(_make_request(client_id="hi-pri", model_id="policy", priority=0))
        batch = sched.next_batch_for_model("policy")
        assert batch[0].client_id == "hi-pri"

    def test_overdue_cadence_boosts_score(self):
        config = SchedulingConfig(strategy="model_deadline")
        sched = ModelDeadlineScheduler(config)
        sched.submit(_make_request(client_id="on-time", model_id="policy", time_until_next=0.05))
        sched.submit(_make_request(client_id="overdue", model_id="policy", time_until_next=-0.1))
        batch = sched.next_batch_for_model("policy")
        assert batch[0].client_id == "overdue"

    def test_steps_remaining_boosts_score(self):
        config = SchedulingConfig(strategy="model_deadline")
        sched = ModelDeadlineScheduler(config)
        sched.submit(_make_request(client_id="far", model_id="policy", steps_remaining=10))
        sched.submit(_make_request(client_id="done", model_id="policy", steps_remaining=0))
        batch = sched.next_batch_for_model("policy")
        assert batch[0].client_id == "done"


class TestNextBatchFallback:
    def test_fallback_drains_from_any_model(self):
        config = SchedulingConfig(strategy="model_deadline")
        sched = ModelDeadlineScheduler(config)
        sched.submit(_make_request(client_id="p", model_id="policy"))
        sched.submit(_make_request(client_id="t", model_id="telemetry"))

        batch = sched.next_batch()
        assert len(batch) == 1
        batch2 = sched.next_batch()
        assert len(batch2) == 1
        assert sched.next_batch() == []

    def test_fallback_empty(self):
        config = SchedulingConfig(strategy="model_deadline")
        sched = ModelDeadlineScheduler(config)
        assert sched.next_batch() == []


class TestOverflow:
    def test_drop_oldest_across_models(self):
        config = SchedulingConfig(
            strategy="model_deadline", max_queue_size=2, overflow_policy="drop_oldest"
        )
        sched = ModelDeadlineScheduler(config)
        old_time = time.monotonic() - 1.0
        sched.submit(_make_request(client_id="old", model_id="telemetry", received_at=old_time))
        sched.submit(_make_request(client_id="new", model_id="policy"))
        # Queue is full. Adding another should drop the oldest.
        sched.submit(_make_request(client_id="newest", model_id="policy"))
        assert sched.queue_len() == 2
        assert sched.queue_len_for_model("telemetry") == 0

    def test_reject_newest_on_overflow(self):
        config = SchedulingConfig(
            strategy="model_deadline", max_queue_size=2, overflow_policy="reject_newest"
        )
        sched = ModelDeadlineScheduler(config)
        sched.submit(_make_request(client_id="1"))
        sched.submit(_make_request(client_id="2"))
        with pytest.raises(QueueFullError):
            sched.submit(_make_request(client_id="3"))


class TestTTLExpiry:
    def test_ttl_expiry_across_models(self):
        config = SchedulingConfig(strategy="model_deadline", request_ttl_ms=100.0)
        sched = ModelDeadlineScheduler(config)
        old_time = time.monotonic() - 0.2
        sched.submit(_make_request(client_id="old_p", model_id="policy", received_at=old_time))
        sched.submit(_make_request(client_id="new_p", model_id="policy"))
        sched.submit(_make_request(client_id="old_t", model_id="telemetry", received_at=old_time))
        sched.submit(_make_request(client_id="new_t", model_id="telemetry"))

        expired = sched.tick()
        assert expired == 2
        assert sched.queue_len() == 2
        assert sched.queue_len_for_model("policy") == 1
        assert sched.queue_len_for_model("telemetry") == 1


class TestDisconnect:
    def test_disconnect_removes_from_all_models(self):
        config = SchedulingConfig(strategy="model_deadline")
        sched = ModelDeadlineScheduler(config)
        sched.submit(_make_request(client_id="robot1", model_id="policy"))
        sched.submit(_make_request(client_id="robot1", model_id="telemetry"))
        sched.submit(_make_request(client_id="robot2", model_id="policy"))

        sched.on_client_disconnected("robot1")
        assert sched.queue_len() == 1
        batch = sched.next_batch_for_model("policy")
        assert batch[0].client_id == "robot2"

    def test_disconnect_cleans_up_empty_models(self):
        config = SchedulingConfig(strategy="model_deadline")
        sched = ModelDeadlineScheduler(config)
        sched.submit(_make_request(client_id="robot1", model_id="telemetry"))
        sched.on_client_disconnected("robot1")
        assert "telemetry" not in sched.models()


class TestResubmit:
    def test_resubmit_bypasses_overflow(self):
        config = SchedulingConfig(strategy="model_deadline", max_queue_size=1)
        sched = ModelDeadlineScheduler(config)
        sched.submit(_make_request(client_id="1"))
        req = _make_request(client_id="2")
        req.retry_count = 1
        sched.resubmit(req)
        assert sched.queue_len() == 2

    def test_resubmit_preserves_retry_count(self):
        config = SchedulingConfig(strategy="model_deadline")
        sched = ModelDeadlineScheduler(config)
        req = _make_request(client_id="1")
        req.retry_count = 3
        sched.resubmit(req)
        batch = sched.next_batch()
        assert batch[0].retry_count == 3


class TestFactory:
    def test_create_via_factory(self):
        config = SchedulingConfig(strategy="model_deadline")
        sched = create_scheduler(config)
        assert isinstance(sched, ModelDeadlineScheduler)
        assert isinstance(sched, ModelAwareScheduler)

    def test_status(self):
        config = SchedulingConfig(strategy="model_deadline")
        sched = ModelDeadlineScheduler(config)
        sched.submit(_make_request(model_id="policy"))
        sched.submit(_make_request(model_id="telemetry"))
        status = sched.status()
        assert status["strategy"] == "model_deadline"
        assert status["queue_size"] == 2
        assert status["models"] == {"policy": 1, "telemetry": 1}
