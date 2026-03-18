from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Callable

import zmq

from inferential.config.schema import InferentialConfig
from inferential.dispatch.dispatcher import RayDispatcher
from inferential.metrics.collector import MetricsCollector
from inferential.observation.assembler import ObservationAssembler
from inferential.observation.slots import ObservationError
from inferential.scheduler.base import QueueFullError, Scheduler, create_scheduler
from inferential.scheduler.request import InferenceRequest
from inferential.tracking.cadence import CadenceTracker
from inferential.tracking.response import ResponseTracker
from inferential.tracking.robots import ClientRegistry
from inferential.transport.messages import OutgoingResponse
from inferential.transport.zmq_transport import ZmqTransport

logger = logging.getLogger("inferential")


class Server:
    def __init__(
        self,
        bind: str = "tcp://*:5555",
        models: list[str] | None = None,
        config: InferentialConfig | None = None,
    ) -> None:
        if config is None:
            config = InferentialConfig()
            config.transport.bind = bind
        self._config = config
        self._models = models or []

        self._transport: ZmqTransport | None = None
        self._assembler = ObservationAssembler(config.observations)
        self._cadence = CadenceTracker(
            alpha=config.response_tracking.cadence_alpha,
            overdue_multiplier=config.response_tracking.overdue_multiplier,
        )
        self._response_tracker = ResponseTracker(
            disconnect_timeout_s=config.response_tracking.disconnect_timeout_s,
        )
        self._client_registry = ClientRegistry()
        self._dispatcher = RayDispatcher(config.ray)
        self._metrics = MetricsCollector(config.metrics)
        self._scheduler: Scheduler | None = None

        # Apply default scheduler
        self._scheduler = create_scheduler(config.scheduling)

    @property
    def metrics(self) -> MetricsCollector:
        return self._metrics

    def use_scheduler(
        self,
        strategy: str,
        policy: str | None = None,
        **kwargs: Any,
    ) -> None:
        sched_config = self._config.scheduling.model_copy()
        sched_config.strategy = strategy
        if kwargs:
            da = sched_config.deadline_aware.model_copy(update=kwargs)
            sched_config.deadline_aware = da
        self._scheduler = create_scheduler(sched_config)
        if policy is not None:
            from inferential.scheduler.deadline_aware import DeadlineAwareScheduler

            if isinstance(self._scheduler, DeadlineAwareScheduler):
                self._scheduler.use_policy(policy)

    def on_metric(self, callback: Callable) -> Callable:
        self._metrics.on_metric(callback)
        return callback

    async def run(self) -> None:
        self._transport = ZmqTransport(self._config.transport)
        logger.info("Inferential server starting on %s", self._config.transport.bind)

        assert self._scheduler is not None

        await asyncio.gather(
            self._receive_loop(),
            self._dispatch_loop(),
            self._tick_loop(),
        )

    async def _receive_loop(self) -> None:
        assert self._transport is not None
        assert self._scheduler is not None

        while True:
            try:
                incoming = await self._transport.recv()

                try:
                    decoded = self._assembler.decode(
                        incoming.envelope, incoming.payload, incoming.received_at
                    )
                except ObservationError as e:
                    logger.warning("Invalid observation: %s", e)
                    self._metrics.record("observation_errors", 1.0)
                    continue

                # Track client
                is_new = self._client_registry.on_observation(decoded.client_id, decoded.model_id)
                if is_new:
                    self._scheduler.on_client_connected(decoded.client_id)
                    logger.info("New client connected: %s", decoded.client_id)

                # Update cadence tracker
                self._cadence.on_request(decoded.client_id)
                self._response_tracker.on_request(decoded.client_id)

                # Record metrics
                self._metrics.record(
                    "observation_staleness_ms",
                    decoded.staleness_ms,
                    {"client": decoded.client_id},
                )
                self._metrics.record(
                    "payload_size_bytes",
                    len(incoming.payload),
                    {"client": decoded.client_id},
                )

                # Build inference request
                request = InferenceRequest(
                    client_id=decoded.client_id,
                    model_id=decoded.model_id,
                    identity=incoming.identity,
                    envelope=incoming.envelope,
                    payload=incoming.payload,
                    received_at=incoming.received_at,
                    urgency=decoded.urgency,
                    steps_remaining=decoded.steps_remaining,
                    time_until_next=self._cadence.time_until_next(decoded.client_id),
                    latency_budget_ms=self._config.get_client_budget(decoded.client_id),
                    priority=self._config.get_client_priority(decoded.client_id),
                )

                try:
                    self._scheduler.submit(request)
                except QueueFullError:
                    logger.warning("Queue full, dropping request from client %s", decoded.client_id)
                    self._metrics.record("queue_full_drops", 1.0, {"client": decoded.client_id})

            except Exception:
                logger.exception("Error in receive loop")
                await asyncio.sleep(0.001)

    async def _dispatch_loop(self) -> None:
        assert self._transport is not None
        assert self._scheduler is not None

        max_retries = self._config.scheduling.max_retries

        while True:
            batch = self._scheduler.next_batch()
            if not batch:
                await asyncio.sleep(0.001)
                continue

            # Collect more ready requests so the dispatcher can
            # saturate multiple Ray Serve replicas concurrently.
            max_concurrent = self._config.scheduling.max_concurrent_dispatch
            while len(batch) < max_concurrent:
                more = self._scheduler.next_batch()
                if not more:
                    break
                batch.extend(more)

            self._metrics.record("queue_depth", self._scheduler.queue_len())
            self._metrics.record("batch_size", float(len(batch)))

            # Record scheduling wait time per request
            dispatch_time = time.monotonic()
            for req in batch:
                wait_ms = (dispatch_time - req.received_at) * 1000
                self._metrics.record("scheduling_wait_ms", wait_ms, {"client": req.client_id})

            # Build lookup for retry
            batch_by_id: dict[str, InferenceRequest] = {req.client_id: req for req in batch}

            results = await self._dispatcher.dispatch(batch)

            for result in results:
                if result.success:
                    self._response_tracker.on_response_sent(
                        result.client_id, result.response_id, result.latency_ms
                    )
                    # End-to-end: queue wait + inference
                    original = batch_by_id.get(result.client_id)
                    e2e_ms = (
                        (time.monotonic() - original.received_at) * 1000
                        if original
                        else result.latency_ms
                    )
                    self._metrics.record("e2e_latency_ms", e2e_ms, {"client": result.client_id})
                    self._metrics.record(
                        "inference_latency_ms",
                        result.latency_ms,
                        {"client": result.client_id},
                    )
                    try:
                        await self._transport.send(
                            OutgoingResponse(
                                identity=result.identity,
                                envelope=result.envelope,
                                payload=result.payload,
                            )
                        )
                    except zmq.ZMQError as e:
                        logger.warning(
                            "Failed to send response to client %s: %s",
                            result.client_id,
                            e,
                        )
                        self._metrics.record(
                            "send_errors", 1.0, {"client": result.client_id}
                        )
                else:
                    original = batch_by_id.get(result.client_id)
                    if (
                        original is not None
                        and max_retries > 0
                        and original.retry_count < max_retries
                    ):
                        original.retry_count += 1
                        self._scheduler.resubmit(original)
                        self._metrics.record(
                            "dispatch_retries",
                            1.0,
                            {"client": result.client_id},
                        )
                        logger.info(
                            "Retrying dispatch for client %s (attempt %d/%d)",
                            result.client_id,
                            original.retry_count,
                            max_retries,
                        )
                    else:
                        self._metrics.record(
                            "dispatch_errors",
                            1.0,
                            {"client": result.client_id, "error": result.error or "unknown"},
                        )

    async def _tick_loop(self) -> None:
        assert self._scheduler is not None

        while True:
            await asyncio.sleep(0.1)  # 10Hz tick
            expired = self._scheduler.tick()
            if expired:
                self._metrics.record("requests_expired", float(expired))
            self._metrics.record("active_clients", self._client_registry.active_count)

            # Check for disconnected clients
            for client_id in list(self._client_registry.all_clients.keys()):
                if self._cadence.time_since_last(client_id) > (
                    self._config.response_tracking.disconnect_timeout_s
                ):
                    self._scheduler.on_client_disconnected(client_id)
                    self._metrics.record("client_disconnected", 1.0, {"client": client_id})
