from __future__ import annotations

import logging
import time
from collections import defaultdict, deque
from typing import TYPE_CHECKING

from inferential.scheduler.base import QueueFullError, Scheduler, register_scheduler
from inferential.scheduler.request import InferenceRequest

if TYPE_CHECKING:
    from inferential.config.schema import SchedulingConfig

logger = logging.getLogger("inferential.scheduler")


@register_scheduler("batch_optimized")
class BatchOptimizedScheduler(Scheduler):
    def __init__(self, config: SchedulingConfig) -> None:
        self._max_queue = config.max_queue_size
        self._max_batch = config.batch_optimized.max_batch_size
        self._max_wait_ms = config.batch_optimized.max_wait_ms
        self._ttl_ms = config.request_ttl_ms
        self._overflow_policy = config.overflow_policy
        self._by_model: dict[str, deque[InferenceRequest]] = defaultdict(deque)
        self._first_arrival: dict[str, float] = {}
        self._total = 0

    def submit(self, request: InferenceRequest) -> None:
        if self._total >= self._max_queue:
            if self._overflow_policy == "drop_oldest":
                self._drop_oldest()
            else:
                raise QueueFullError(request.client_id)

        model_id = request.model_id
        if model_id not in self._first_arrival:
            self._first_arrival[model_id] = time.monotonic()
        self._by_model[model_id].append(request)
        self._total += 1

    def _drop_oldest(self) -> None:
        oldest_model = None
        oldest_time = float("inf")
        for model_id, q in self._by_model.items():
            if q and q[0].received_at < oldest_time:
                oldest_time = q[0].received_at
                oldest_model = model_id
        if oldest_model is not None:
            dropped = self._by_model[oldest_model].popleft()
            self._total -= 1
            logger.debug("Dropped oldest request from client %s", dropped.client_id)
            if not self._by_model[oldest_model]:
                self._first_arrival.pop(oldest_model, None)

    def resubmit(self, request: InferenceRequest) -> None:
        model_id = request.model_id
        if model_id not in self._first_arrival:
            self._first_arrival[model_id] = time.monotonic()
        self._by_model[model_id].append(request)
        self._total += 1

    def next_batch(self) -> list[InferenceRequest]:
        now = time.monotonic()

        # Check each model queue for batch readiness
        for model_id in list(self._by_model.keys()):
            q = self._by_model[model_id]
            if not q:
                continue

            batch_full = len(q) >= self._max_batch
            first = self._first_arrival.get(model_id, now)
            wait_exceeded = (now - first) * 1000 >= self._max_wait_ms

            if batch_full or wait_exceeded:
                batch: list[InferenceRequest] = []
                while q and len(batch) < self._max_batch:
                    batch.append(q.popleft())
                    self._total -= 1

                # Reset arrival timer
                if q:
                    self._first_arrival[model_id] = now
                else:
                    self._first_arrival.pop(model_id, None)

                return batch

        return []

    def tick(self) -> int:
        expired = 0
        for model_id in list(self._by_model.keys()):
            q = self._by_model[model_id]
            before = len(q)
            self._by_model[model_id] = deque(r for r in q if not self._is_expired(r, self._ttl_ms))
            removed = before - len(self._by_model[model_id])
            self._total -= removed
            expired += removed
            if not self._by_model[model_id]:
                self._first_arrival.pop(model_id, None)
        return expired

    def on_client_disconnected(self, client_id: str) -> None:
        # Remove requests from this client
        for model_id in list(self._by_model.keys()):
            q = self._by_model[model_id]
            before = len(q)
            self._by_model[model_id] = deque(r for r in q if r.client_id != client_id)
            removed = before - len(self._by_model[model_id])
            self._total -= removed

    def status(self) -> dict:
        return {
            "strategy": "batch_optimized",
            "queue_size": self._total,
            "max_queue_size": self._max_queue,
            "max_batch_size": self._max_batch,
            "models_queued": len([q for q in self._by_model.values() if q]),
        }

    def queue_len(self) -> int:
        return self._total
