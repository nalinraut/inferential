from __future__ import annotations

import asyncio
import time

import pytest

from inferential.config.schema import InferentialConfig, ModelConfig, SchedulingConfig
from inferential.scheduler.base import ModelAwareScheduler, create_scheduler
from inferential.scheduler.deadline_aware import DeadlineAwareScheduler
from inferential.scheduler.model_deadline import ModelDeadlineScheduler
from inferential.scheduler.request import InferenceRequest


def _make_request(
    client_id: str = "1",
    model_id: str = "policy",
    priority: int = 1,
    received_at: float | None = None,
) -> InferenceRequest:
    return InferenceRequest(
        client_id=client_id,
        model_id=model_id,
        identity=client_id.encode(),
        envelope=b"",
        payload=b"",
        received_at=received_at or time.monotonic(),
        urgency=0.0,
        steps_remaining=None,
        time_until_next=0.0,
        latency_budget_ms=50.0,
        priority=priority,
    )


class TestPipelineConfig:
    def test_pipeline_disabled_by_default(self):
        config = InferentialConfig()
        assert config.scheduling.pipeline_dispatch.enabled is False

    def test_predictive_disabled_by_default(self):
        config = InferentialConfig()
        assert config.scheduling.predictive_pre_dispatch.enabled is False

    def test_model_max_inflight_defaults(self):
        config = InferentialConfig()
        assert config.get_model_max_inflight("unknown") == 2

    def test_model_max_inflight_known(self):
        config = InferentialConfig(
            models={"known": {"policy": ModelConfig(max_inflight=4)}, "default_max_inflight": 2}
        )
        assert config.get_model_max_inflight("policy") == 4
        assert config.get_model_max_inflight("telemetry") == 2


class TestModelAwareSchedulerDetection:
    def test_model_deadline_is_model_aware(self):
        config = SchedulingConfig(strategy="model_deadline")
        sched = create_scheduler(config)
        assert isinstance(sched, ModelAwareScheduler)

    def test_deadline_aware_is_not_model_aware(self):
        config = SchedulingConfig(strategy="deadline_aware")
        sched = create_scheduler(config)
        assert not isinstance(sched, ModelAwareScheduler)

    def test_round_robin_is_not_model_aware(self):
        config = SchedulingConfig(strategy="round_robin")
        sched = create_scheduler(config)
        assert not isinstance(sched, ModelAwareScheduler)


class TestServerModelDispatch:
    def test_ensure_model_dispatch_creates_structures(self):
        from inferential.server import Server

        config = InferentialConfig(
            models={"known": {"policy": ModelConfig(max_inflight=3)}, "default_max_inflight": 2}
        )
        server = Server(config=config)

        # Need an event loop for _ensure_model_dispatch
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(self._test_ensure(server))
        finally:
            loop.close()

    async def _test_ensure(self, server):
        server._transport = True  # stub for assert
        server._scheduler = ModelDeadlineScheduler(SchedulingConfig(strategy="model_deadline"))
        server._ensure_model_dispatch("policy")
        assert "policy" in server._model_events
        assert "policy" in server._model_sems
        assert "policy" in server._model_tasks
        # Cancel the task to avoid warnings
        server._model_tasks["policy"].cancel()
        try:
            await server._model_tasks["policy"]
        except asyncio.CancelledError:
            pass


class TestUseScheduler:
    def test_use_scheduler_creates_model_deadline(self):
        from inferential.server import Server

        config = InferentialConfig()
        server = Server(config=config)
        server.use_scheduler("model_deadline")
        assert isinstance(server._scheduler, ModelDeadlineScheduler)

    def test_use_scheduler_with_kwargs(self):
        from inferential.server import Server

        config = InferentialConfig()
        server = Server(config=config)
        server.use_scheduler("model_deadline", cadence_weight=60.0)
        assert isinstance(server._scheduler, ModelDeadlineScheduler)

    def test_use_scheduler_backwards_compat(self):
        from inferential.server import Server

        config = InferentialConfig()
        server = Server(config=config)
        server.use_scheduler("deadline_aware")
        assert isinstance(server._scheduler, DeadlineAwareScheduler)


class TestSemaphoreBounding:
    @pytest.mark.asyncio
    async def test_semaphore_limits_concurrency(self):
        """Verify that semaphore limits concurrent access."""
        sem = asyncio.Semaphore(2)
        max_concurrent = 0
        current = 0

        async def task():
            nonlocal max_concurrent, current
            async with sem:
                current += 1
                max_concurrent = max(max_concurrent, current)
                await asyncio.sleep(0.01)
                current -= 1

        await asyncio.gather(*(task() for _ in range(5)))
        assert max_concurrent == 2

    @pytest.mark.asyncio
    async def test_event_driven_wakeup(self):
        """Verify that asyncio.Event wakes up waiters immediately."""
        event = asyncio.Event()
        woke_up = False

        async def waiter():
            nonlocal woke_up
            await event.wait()
            woke_up = True

        task = asyncio.create_task(waiter())
        await asyncio.sleep(0.001)
        assert not woke_up

        event.set()
        await asyncio.sleep(0.001)
        assert woke_up
        await task
