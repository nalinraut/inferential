from __future__ import annotations

import time
from unittest.mock import patch

import pytest

from inferential.config.schema import SchedulingConfig
from inferential.scheduler.base import QueueFullError, create_scheduler, register_policy
from inferential.scheduler.batch_optimized import BatchOptimizedScheduler
from inferential.scheduler.deadline_aware import DeadlineAwareScheduler
from inferential.scheduler.priority_tiered import PriorityTieredScheduler
from inferential.scheduler.request import InferenceRequest
from inferential.scheduler.round_robin import RoundRobinScheduler


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


# --- DeadlineAwareScheduler ---


class TestDeadlineAware:
    def test_submit_and_pop(self):
        config = SchedulingConfig(strategy="deadline_aware")
        sched = DeadlineAwareScheduler(config)
        sched.submit(_make_request())
        assert sched.queue_len() == 1
        batch = sched.next_batch()
        assert len(batch) == 1
        assert sched.queue_len() == 0

    def test_urgency_ordering(self):
        config = SchedulingConfig(strategy="deadline_aware")
        sched = DeadlineAwareScheduler(config)
        sched.submit(_make_request(client_id="10", urgency=0.1))
        sched.submit(_make_request(client_id="20", urgency=0.9))
        batch = sched.next_batch()
        assert batch[0].client_id == "20"

    def test_queue_full_reject_newest(self):
        config = SchedulingConfig(
            strategy="deadline_aware", max_queue_size=2, overflow_policy="reject_newest"
        )
        sched = DeadlineAwareScheduler(config)
        sched.submit(_make_request(client_id="1"))
        sched.submit(_make_request(client_id="2"))
        with pytest.raises(QueueFullError):
            sched.submit(_make_request(client_id="3"))

    def test_drop_oldest_on_overflow(self):
        config = SchedulingConfig(
            strategy="deadline_aware", max_queue_size=2, overflow_policy="drop_oldest"
        )
        sched = DeadlineAwareScheduler(config)
        now = time.monotonic()
        sched.submit(_make_request(client_id="1", received_at=now - 0.3))
        sched.submit(_make_request(client_id="2", received_at=now - 0.2))
        # Queue is full; drop-oldest should evict client "1"
        sched.submit(_make_request(client_id="3", received_at=now))
        assert sched.queue_len() == 2
        ids = {sched.next_batch()[0].client_id for _ in range(2)}
        assert "1" not in ids
        assert "3" in ids

    def test_ttl_expiry(self):
        config = SchedulingConfig(strategy="deadline_aware", request_ttl_ms=100.0)
        sched = DeadlineAwareScheduler(config)
        # Submit a request that's already 200ms old
        old_time = time.monotonic() - 0.2
        sched.submit(_make_request(client_id="1", received_at=old_time))
        sched.submit(_make_request(client_id="2"))  # fresh
        expired = sched.tick()
        assert expired == 1
        assert sched.queue_len() == 1
        batch = sched.next_batch()
        assert batch[0].client_id == "2"

    def test_overdue_cadence_boosts_score(self):
        config = SchedulingConfig(strategy="deadline_aware")
        sched = DeadlineAwareScheduler(config)
        # Negative time_until_next means overdue
        r1 = _make_request(client_id="10", time_until_next=0.05)
        r2 = _make_request(client_id="20", time_until_next=-0.1)
        sched.submit(r1)
        sched.submit(r2)
        batch = sched.next_batch()
        assert batch[0].client_id == "20"

    def test_custom_policy(self):
        @register_policy("test_always_100")
        def always_100(req: InferenceRequest) -> float:
            return 100.0

        config = SchedulingConfig(strategy="deadline_aware")
        sched = DeadlineAwareScheduler(config)
        sched.use_policy("test_always_100")
        r = _make_request()
        sched.submit(r)
        assert r.score == 100.0


# --- RoundRobinScheduler ---


class TestRoundRobin:
    def test_basic_round_robin(self):
        config = SchedulingConfig(strategy="round_robin")
        sched = RoundRobinScheduler(config)
        sched.submit(_make_request(client_id="1"))
        sched.submit(_make_request(client_id="2"))
        sched.submit(_make_request(client_id="1"))

        # Should alternate between clients
        b1 = sched.next_batch()
        assert b1[0].client_id == "1"
        b2 = sched.next_batch()
        assert b2[0].client_id == "2"
        b3 = sched.next_batch()
        assert b3[0].client_id == "1"

    def test_empty_queue(self):
        config = SchedulingConfig(strategy="round_robin")
        sched = RoundRobinScheduler(config)
        assert sched.next_batch() == []

    def test_disconnect_clears_queue(self):
        config = SchedulingConfig(strategy="round_robin")
        sched = RoundRobinScheduler(config)
        sched.submit(_make_request(client_id="1"))
        sched.submit(_make_request(client_id="1"))
        assert sched.queue_len() == 2
        sched.on_client_disconnected("1")
        assert sched.queue_len() == 0

    def test_drop_oldest_on_overflow(self):
        config = SchedulingConfig(
            strategy="round_robin", max_queue_size=2, overflow_policy="drop_oldest"
        )
        sched = RoundRobinScheduler(config)
        now = time.monotonic()
        sched.submit(_make_request(client_id="1", received_at=now - 0.3))
        sched.submit(_make_request(client_id="2", received_at=now - 0.2))
        sched.submit(_make_request(client_id="3", received_at=now))
        assert sched.queue_len() == 2

    def test_ttl_expiry(self):
        config = SchedulingConfig(strategy="round_robin", request_ttl_ms=100.0)
        sched = RoundRobinScheduler(config)
        old_time = time.monotonic() - 0.2
        sched.submit(_make_request(client_id="1", received_at=old_time))
        sched.submit(_make_request(client_id="2"))
        expired = sched.tick()
        assert expired == 1
        assert sched.queue_len() == 1


# --- BatchOptimizedScheduler ---


class TestBatchOptimized:
    def test_batch_when_full(self):
        config = SchedulingConfig(strategy="batch_optimized")
        config.batch_optimized.max_batch_size = 2
        config.batch_optimized.max_wait_ms = 10000  # won't trigger by time
        sched = BatchOptimizedScheduler(config)

        sched.submit(_make_request(model_id="m1"))
        assert sched.next_batch() == []  # Not full yet

        sched.submit(_make_request(model_id="m1"))
        batch = sched.next_batch()
        assert len(batch) == 2

    def test_batch_when_timeout(self):
        config = SchedulingConfig(strategy="batch_optimized")
        config.batch_optimized.max_batch_size = 100
        config.batch_optimized.max_wait_ms = 5.0
        sched = BatchOptimizedScheduler(config)

        with patch.object(time, "monotonic", return_value=0.0):
            sched.submit(_make_request(model_id="m1"))

        # Not enough time yet
        with patch.object(time, "monotonic", return_value=0.003):
            assert sched.next_batch() == []

        # Enough time
        with patch.object(time, "monotonic", return_value=0.006):
            batch = sched.next_batch()
            assert len(batch) == 1

    def test_groups_by_model(self):
        config = SchedulingConfig(strategy="batch_optimized")
        config.batch_optimized.max_batch_size = 2
        sched = BatchOptimizedScheduler(config)

        sched.submit(_make_request(model_id="m1"))
        sched.submit(_make_request(model_id="m2"))
        sched.submit(_make_request(model_id="m1"))

        batch = sched.next_batch()
        assert len(batch) == 2
        assert all(r.model_id == "m1" for r in batch)

    def test_drop_oldest_on_overflow(self):
        config = SchedulingConfig(
            strategy="batch_optimized", max_queue_size=2, overflow_policy="drop_oldest"
        )
        config.batch_optimized.max_batch_size = 10
        sched = BatchOptimizedScheduler(config)
        now = time.monotonic()
        sched.submit(_make_request(client_id="1", received_at=now - 0.3))
        sched.submit(_make_request(client_id="2", received_at=now - 0.2))
        sched.submit(_make_request(client_id="3", received_at=now))
        assert sched.queue_len() == 2

    def test_ttl_expiry(self):
        config = SchedulingConfig(strategy="batch_optimized", request_ttl_ms=100.0)
        sched = BatchOptimizedScheduler(config)
        old_time = time.monotonic() - 0.2
        sched.submit(_make_request(client_id="1", received_at=old_time))
        sched.submit(_make_request(client_id="2"))
        expired = sched.tick()
        assert expired == 1
        assert sched.queue_len() == 1


# --- PriorityTieredScheduler ---


class TestPriorityTiered:
    def test_strict_priority(self):
        config = SchedulingConfig(strategy="priority_tiered")
        sched = PriorityTieredScheduler(config)

        sched.submit(_make_request(client_id="10", priority=2))
        sched.submit(_make_request(client_id="20", priority=0))

        batch = sched.next_batch()
        assert batch[0].client_id == "20"
        batch = sched.next_batch()
        assert batch[0].client_id == "10"

    def test_fifo_within_tier(self):
        config = SchedulingConfig(strategy="priority_tiered")
        sched = PriorityTieredScheduler(config)

        sched.submit(_make_request(client_id="100", priority=1))
        sched.submit(_make_request(client_id="200", priority=1))

        batch = sched.next_batch()
        assert batch[0].client_id == "100"
        batch = sched.next_batch()
        assert batch[0].client_id == "200"

    def test_priority_clamped_to_tiers(self):
        config = SchedulingConfig(strategy="priority_tiered")
        config.priority_tiered.num_tiers = 3
        sched = PriorityTieredScheduler(config)

        # Priority 10 should be clamped to tier 2 (last tier)
        sched.submit(_make_request(client_id="99", priority=10))
        assert sched.queue_len() == 1
        batch = sched.next_batch()
        assert batch[0].client_id == "99"

    def test_drop_oldest_on_overflow(self):
        config = SchedulingConfig(
            strategy="priority_tiered", max_queue_size=2, overflow_policy="drop_oldest"
        )
        sched = PriorityTieredScheduler(config)
        # Fill with low-priority requests
        sched.submit(_make_request(client_id="1", priority=2))
        sched.submit(_make_request(client_id="2", priority=2))
        # Submit high-priority — should drop from lowest tier
        sched.submit(_make_request(client_id="3", priority=0))
        assert sched.queue_len() == 2
        # High-priority request should be served first
        batch = sched.next_batch()
        assert batch[0].client_id == "3"

    def test_ttl_expiry(self):
        config = SchedulingConfig(strategy="priority_tiered", request_ttl_ms=100.0)
        sched = PriorityTieredScheduler(config)
        old_time = time.monotonic() - 0.2
        sched.submit(_make_request(client_id="1", received_at=old_time))
        sched.submit(_make_request(client_id="2"))
        expired = sched.tick()
        assert expired == 1
        assert sched.queue_len() == 1


# --- Factory ---


class TestFactory:
    def test_create_all_strategies(self):
        for strategy in ["deadline_aware", "round_robin", "batch_optimized", "priority_tiered"]:
            config = SchedulingConfig(strategy=strategy)
            sched = create_scheduler(config)
            assert sched is not None

    def test_unknown_strategy(self):
        config = SchedulingConfig(strategy="nonexistent")
        with pytest.raises(ValueError, match="Unknown scheduler strategy"):
            create_scheduler(config)


# --- Resubmit (retry support) ---


class TestResubmit:
    def test_resubmit_deadline_aware(self):
        config = SchedulingConfig(strategy="deadline_aware", max_queue_size=1)
        sched = DeadlineAwareScheduler(config)
        # Fill queue
        sched.submit(_make_request(client_id="1"))
        assert sched.queue_len() == 1
        # Resubmit bypasses overflow
        req = _make_request(client_id="2")
        req.retry_count = 1
        sched.resubmit(req)
        assert sched.queue_len() == 2

    def test_resubmit_round_robin(self):
        config = SchedulingConfig(strategy="round_robin", max_queue_size=1)
        sched = RoundRobinScheduler(config)
        sched.submit(_make_request(client_id="1"))
        req = _make_request(client_id="1")
        req.retry_count = 1
        sched.resubmit(req)
        assert sched.queue_len() == 2

    def test_resubmit_batch_optimized(self):
        config = SchedulingConfig(strategy="batch_optimized", max_queue_size=1)
        sched = BatchOptimizedScheduler(config)
        sched.submit(_make_request(client_id="1"))
        req = _make_request(client_id="1")
        req.retry_count = 1
        sched.resubmit(req)
        assert sched.queue_len() == 2

    def test_resubmit_priority_tiered(self):
        config = SchedulingConfig(strategy="priority_tiered", max_queue_size=1)
        sched = PriorityTieredScheduler(config)
        sched.submit(_make_request(client_id="1"))
        req = _make_request(client_id="1")
        req.retry_count = 1
        sched.resubmit(req)
        assert sched.queue_len() == 2

    def test_retry_count_preserved(self):
        config = SchedulingConfig(strategy="deadline_aware")
        sched = DeadlineAwareScheduler(config)
        req = _make_request(client_id="1")
        req.retry_count = 2
        sched.resubmit(req)
        batch = sched.next_batch()
        assert batch[0].retry_count == 2
