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
from inferential.scheduler.base import (
    ModelAwareScheduler,
    QueueFullError,
    Scheduler,
    create_scheduler,
)
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
        self._model_events: dict[str, asyncio.Event] = {}
        self._model_sems: dict[str, asyncio.Semaphore] = {}
        self._model_tasks: dict[str, asyncio.Task] = {}
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
            if strategy == "model_deadline":
                da = sched_config.model_deadline.model_copy(update=kwargs)
                sched_config.model_deadline = da
            else:
                da = sched_config.deadline_aware.model_copy(update=kwargs)
                sched_config.deadline_aware = da
        self._scheduler = create_scheduler(sched_config)
        if policy is not None and hasattr(self._scheduler, "use_policy"):
            self._scheduler.use_policy(policy)

    def on_metric(self, callback: Callable) -> Callable:
        self._metrics.on_metric(callback)
        return callback

    async def run(self) -> None:
        self._transport = ZmqTransport(self._config.transport)
        logger.info("Inferential server starting on %s", self._config.transport.bind)

        assert self._scheduler is not None

        pipeline_config = self._config.scheduling.pipeline_dispatch

        if pipeline_config.enabled:
            logger.info("Pipeline dispatch enabled (per-model concurrency)")

            # _dispatch_loop runs alongside model loops: it handles non-model-aware
            # schedulers (e.g. round_robin) and yields when a ModelAwareScheduler is active.
            coroutines: list[Any] = [
                self._receive_loop(),
                self._dispatch_loop(),
                self._tick_loop(),
            ]

            # Optional predictive pre-dispatch
            pd_config = self._config.scheduling.predictive_pre_dispatch
            if pd_config.enabled:
                coroutines.append(self._predictive_wake_loop())

            await asyncio.gather(*coroutines)
        else:
            # Legacy single dispatch loop
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
                    priority=decoded.priority,
                )

                try:
                    self._scheduler.submit(request)
                except QueueFullError:
                    logger.warning("Queue full, dropping request from client %s", decoded.client_id)
                    self._metrics.record("queue_full_drops", 1.0, {"client": decoded.client_id})
                    continue

                # Wake the model dispatch loop (start one if first time seeing this model)
                if isinstance(self._scheduler, ModelAwareScheduler):
                    self._ensure_model_dispatch(decoded.model_id)
                    self._model_events[decoded.model_id].set()

            except Exception:
                logger.exception("Error in receive loop")
                await asyncio.sleep(0.001)

    # --- Pipeline dispatch (per-model) ---

    def _ensure_model_dispatch(self, model_id: str) -> None:
        """Lazily create event/semaphore/dispatch loop for a new model."""
        if model_id in self._model_events:
            return
        max_inflight = self._config.get_model_max_inflight(model_id)
        sem = asyncio.Semaphore(max_inflight)
        event = asyncio.Event()
        self._model_events[model_id] = event
        self._model_sems[model_id] = sem
        self._model_tasks[model_id] = asyncio.create_task(
            self._model_dispatch_loop(model_id, sem, event)
        )
        logger.info(
            "Started dispatch loop for model %s (max_inflight=%d)",
            model_id,
            max_inflight,
        )

    async def _model_dispatch_loop(
        self,
        model_id: str,
        sem: asyncio.Semaphore,
        event: asyncio.Event,
    ) -> None:
        assert self._transport is not None
        assert self._scheduler is not None

        max_retries = self._config.scheduling.max_retries

        while True:
            await sem.acquire()

            # If the scheduler was swapped to a non-model-aware type, yield to _dispatch_loop.
            if not isinstance(self._scheduler, ModelAwareScheduler):
                sem.release()
                await asyncio.sleep(0.001)
                continue

            batch = self._scheduler.next_batch_for_model(model_id)
            if not batch:
                sem.release()
                try:
                    await asyncio.wait_for(event.wait(), timeout=0.001)
                    event.clear()
                except asyncio.TimeoutError:
                    pass
                continue

            request = batch[0]

            # Record scheduling wait
            wait_ms = (time.monotonic() - request.received_at) * 1000
            self._metrics.record(
                "scheduling_wait_ms",
                wait_ms,
                {"client": request.client_id, "model": model_id},
            )
            self._metrics.record(
                "queue_depth",
                self._scheduler.queue_len_for_model(model_id),
                {"model": model_id},
            )

            # Fire dispatch as background task (pipeline)
            asyncio.create_task(self._dispatch_single(request, model_id, sem, max_retries))

    async def _dispatch_single(
        self,
        request: InferenceRequest,
        model_id: str,
        sem: asyncio.Semaphore,
        max_retries: int,
    ) -> None:
        assert self._transport is not None
        assert self._scheduler is not None

        try:
            results = await self._dispatcher.dispatch([request])

            for result in results:
                if result.success:
                    self._response_tracker.on_response_sent(
                        result.client_id, result.response_id, result.latency_ms
                    )
                    e2e_ms = (time.monotonic() - request.received_at) * 1000
                    self._metrics.record(
                        "e2e_latency_ms",
                        e2e_ms,
                        {"client": result.client_id, "model": model_id},
                    )
                    self._metrics.record(
                        "inference_latency_ms",
                        result.latency_ms,
                        {"client": result.client_id, "model": model_id},
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
                        self._metrics.record("send_errors", 1.0, {"client": result.client_id})
                else:
                    if max_retries > 0 and request.retry_count < max_retries:
                        request.retry_count += 1
                        self._scheduler.resubmit(request)
                        self._metrics.record(
                            "dispatch_retries",
                            1.0,
                            {"client": result.client_id, "model": model_id},
                        )
                        logger.info(
                            "Retrying dispatch for client %s (attempt %d/%d)",
                            result.client_id,
                            request.retry_count,
                            max_retries,
                        )
                    else:
                        self._metrics.record(
                            "dispatch_errors",
                            1.0,
                            {"client": result.client_id, "error": result.error or "unknown"},
                        )
        except Exception:
            logger.exception("Error in dispatch_single for client %s", request.client_id)
        finally:
            sem.release()

    # --- Predictive pre-dispatch (opt-in) ---

    async def _predictive_wake_loop(self) -> None:
        """Wake model dispatch loops early based on cadence prediction."""
        assert self._scheduler is not None

        pd_config = self._config.scheduling.predictive_pre_dispatch
        lookahead_s = pd_config.lookahead_ms / 1000.0

        while True:
            await asyncio.sleep(0.002)  # 500Hz check

            for client_id in list(self._client_registry.all_clients.keys()):
                if self._cadence.sample_count(client_id) < pd_config.min_cadence_samples:
                    continue

                time_until = self._cadence.time_until_next(client_id)
                if 0 < time_until <= lookahead_s:
                    # Wake all model dispatch loops for this client's models
                    models = self._client_registry.client_models(client_id)
                    for model_id in models:
                        if model_id in self._model_events:
                            self._model_events[model_id].set()

    # --- Legacy single dispatch loop ---

    async def _dispatch_loop(self) -> None:
        assert self._transport is not None
        assert self._scheduler is not None

        max_retries = self._config.scheduling.max_retries

        while True:
            # When pipeline dispatch is enabled and a ModelAwareScheduler is active,
            # per-model loops own dispatch — yield to them.
            if self._config.scheduling.pipeline_dispatch.enabled and isinstance(
                self._scheduler, ModelAwareScheduler
            ):
                await asyncio.sleep(0.001)
                continue

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

            # Build lookup keyed by object id so same-client requests in one batch are distinct
            batch_by_id: dict[int, InferenceRequest] = {id(req): req for req in batch}

            results = await self._dispatcher.dispatch(batch)

            for result in results:
                if result.success:
                    self._response_tracker.on_response_sent(
                        result.client_id, result.response_id, result.latency_ms
                    )
                    # End-to-end: queue wait + inference
                    original = batch_by_id.get(result.req_id)
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
                        self._metrics.record("send_errors", 1.0, {"client": result.client_id})
                else:
                    original = batch_by_id.get(result.req_id)
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
