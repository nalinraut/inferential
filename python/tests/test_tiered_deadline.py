from __future__ import annotations

import time

import pytest

from inferential.config.schema import SchedulingConfig
from inferential.scheduler.base import QueueFullError, Scheduler, create_scheduler
from inferential.scheduler.request import InferenceRequest
from inferential.scheduler.tiered_deadline import TieredDeadlineScheduler


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


class TestTierRouting:
    def test_priority_0_goes_to_tier_0(self):
        config = SchedulingConfig(strategy="tiered_deadline")
        sched = TieredDeadlineScheduler(config)
        sched.submit(_make_request(client_id="1", priority=0))
        assert sched.queue_len_for_tier(0) == 1
        assert sched.queue_len_for_tier(1) == 0

    def test_priority_1_goes_to_tier_1(self):
        config = SchedulingConfig(strategy="tiered_deadline")
        sched = TieredDeadlineScheduler(config)
        sched.submit(_make_request(client_id="1", priority=1))
        assert sched.queue_len_for_tier(0) == 0
        assert sched.queue_len_for_tier(1) == 1

    def test_high_priority_clamped_to_last_tier(self):
        config = SchedulingConfig(strategy="tiered_deadline")
        config.priority_tiered.num_tiers = 2
        sched = TieredDeadlineScheduler(config)
        sched.submit(_make_request(client_id="1", priority=5))
        assert sched.queue_len_for_tier(1) == 1

    def test_num_tiers(self):
        config = SchedulingConfig(strategy="tiered_deadline")
        config.priority_tiered.num_tiers = 3
        sched = TieredDeadlineScheduler(config)
        assert sched.num_tiers() == 3


class TestTierIsolation:
    def test_tier_0_drains_independently(self):
        config = SchedulingConfig(strategy="tiered_deadline")
        sched = TieredDeadlineScheduler(config)
        sched.submit(_make_request(client_id="p0", priority=0))
        sched.submit(_make_request(client_id="p1", priority=1))

        batch = sched.next_batch_for_tier(0)
        assert len(batch) == 1
        assert batch[0].client_id == "p0"
        assert sched.queue_len_for_tier(0) == 0
        assert sched.queue_len_for_tier(1) == 1

    def test_tier_1_drains_independently(self):
        config = SchedulingConfig(strategy="tiered_deadline")
        sched = TieredDeadlineScheduler(config)
        sched.submit(_make_request(client_id="p0", priority=0))
        sched.submit(_make_request(client_id="p1", priority=1))

        batch = sched.next_batch_for_tier(1)
        assert len(batch) == 1
        assert batch[0].client_id == "p1"
        assert sched.queue_len_for_tier(0) == 1

    def test_empty_tier_returns_empty(self):
        config = SchedulingConfig(strategy="tiered_deadline")
        sched = TieredDeadlineScheduler(config)
        assert sched.next_batch_for_tier(0) == []
        assert sched.next_batch_for_tier(1) == []

    def test_invalid_tier_returns_empty(self):
        config = SchedulingConfig(strategy="tiered_deadline")
        sched = TieredDeadlineScheduler(config)
        assert sched.next_batch_for_tier(99) == []
        assert sched.queue_len_for_tier(99) == 0


class TestScoringWithinTier:
    def test_urgency_ordering_within_tier(self):
        config = SchedulingConfig(strategy="tiered_deadline")
        sched = TieredDeadlineScheduler(config)
        sched.submit(_make_request(client_id="low", priority=0, urgency=0.1))
        sched.submit(_make_request(client_id="high", priority=0, urgency=0.9))
        batch = sched.next_batch_for_tier(0)
        assert batch[0].client_id == "high"

    def test_overdue_cadence_boosts_score(self):
        config = SchedulingConfig(strategy="tiered_deadline")
        sched = TieredDeadlineScheduler(config)
        sched.submit(_make_request(client_id="on-time", priority=0, time_until_next=0.05))
        sched.submit(_make_request(client_id="overdue", priority=0, time_until_next=-0.1))
        batch = sched.next_batch_for_tier(0)
        assert batch[0].client_id == "overdue"

    def test_steps_remaining_boosts_score(self):
        config = SchedulingConfig(strategy="tiered_deadline")
        sched = TieredDeadlineScheduler(config)
        sched.submit(_make_request(client_id="far", priority=0, steps_remaining=10))
        sched.submit(_make_request(client_id="done", priority=0, steps_remaining=0))
        batch = sched.next_batch_for_tier(0)
        assert batch[0].client_id == "done"


class TestNextBatchFallback:
    def test_fallback_drains_highest_first(self):
        config = SchedulingConfig(strategy="tiered_deadline")
        sched = TieredDeadlineScheduler(config)
        sched.submit(_make_request(client_id="p1", priority=1))
        sched.submit(_make_request(client_id="p0", priority=0))

        batch = sched.next_batch()
        assert batch[0].client_id == "p0"
        batch = sched.next_batch()
        assert batch[0].client_id == "p1"

    def test_fallback_empty(self):
        config = SchedulingConfig(strategy="tiered_deadline")
        sched = TieredDeadlineScheduler(config)
        assert sched.next_batch() == []


class TestOverflow:
    def test_drop_oldest_from_lowest_tier(self):
        config = SchedulingConfig(
            strategy="tiered_deadline", max_queue_size=2, overflow_policy="drop_oldest"
        )
        sched = TieredDeadlineScheduler(config)
        sched.submit(_make_request(client_id="lo1", priority=1))
        sched.submit(_make_request(client_id="lo2", priority=1))
        # Queue is full. Adding P0 should drop from tier 1.
        sched.submit(_make_request(client_id="hi", priority=0))
        assert sched.queue_len() == 2
        assert sched.queue_len_for_tier(0) == 1

    def test_reject_newest_on_overflow(self):
        config = SchedulingConfig(
            strategy="tiered_deadline", max_queue_size=2, overflow_policy="reject_newest"
        )
        sched = TieredDeadlineScheduler(config)
        sched.submit(_make_request(client_id="1"))
        sched.submit(_make_request(client_id="2"))
        with pytest.raises(QueueFullError):
            sched.submit(_make_request(client_id="3"))


class TestTTLExpiry:
    def test_ttl_expiry_per_tier(self):
        config = SchedulingConfig(strategy="tiered_deadline", request_ttl_ms=100.0)
        sched = TieredDeadlineScheduler(config)
        old_time = time.monotonic() - 0.2
        sched.submit(_make_request(client_id="old_p0", priority=0, received_at=old_time))
        sched.submit(_make_request(client_id="new_p0", priority=0))
        sched.submit(_make_request(client_id="old_p1", priority=1, received_at=old_time))
        sched.submit(_make_request(client_id="new_p1", priority=1))

        expired = sched.tick()
        assert expired == 2
        assert sched.queue_len() == 2
        assert sched.queue_len_for_tier(0) == 1
        assert sched.queue_len_for_tier(1) == 1


class TestDisconnect:
    def test_disconnect_removes_from_all_tiers(self):
        config = SchedulingConfig(strategy="tiered_deadline")
        sched = TieredDeadlineScheduler(config)
        sched.submit(_make_request(client_id="robot1", priority=0))
        sched.submit(_make_request(client_id="robot1", priority=1))
        sched.submit(_make_request(client_id="robot2", priority=0))

        sched.on_client_disconnected("robot1")
        assert sched.queue_len() == 1
        batch = sched.next_batch_for_tier(0)
        assert batch[0].client_id == "robot2"


class TestResubmit:
    def test_resubmit_bypasses_overflow(self):
        config = SchedulingConfig(strategy="tiered_deadline", max_queue_size=1)
        sched = TieredDeadlineScheduler(config)
        sched.submit(_make_request(client_id="1"))
        req = _make_request(client_id="2")
        req.retry_count = 1
        sched.resubmit(req)
        assert sched.queue_len() == 2

    def test_resubmit_preserves_retry_count(self):
        config = SchedulingConfig(strategy="tiered_deadline")
        sched = TieredDeadlineScheduler(config)
        req = _make_request(client_id="1")
        req.retry_count = 3
        sched.resubmit(req)
        batch = sched.next_batch()
        assert batch[0].retry_count == 3


class TestFactory:
    def test_create_via_factory(self):
        config = SchedulingConfig(strategy="tiered_deadline")
        sched = create_scheduler(config)
        assert isinstance(sched, TieredDeadlineScheduler)
        assert isinstance(sched, Scheduler)

    def test_status(self):
        config = SchedulingConfig(strategy="tiered_deadline")
        sched = TieredDeadlineScheduler(config)
        sched.submit(_make_request(priority=0))
        sched.submit(_make_request(priority=1))
        status = sched.status()
        assert status["strategy"] == "tiered_deadline"
        assert status["queue_size"] == 2
        assert status["tiers"] == [1, 1, 0]
