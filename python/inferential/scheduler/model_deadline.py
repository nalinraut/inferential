from __future__ import annotations

import heapq
import logging
import time
from collections import defaultdict
from typing import TYPE_CHECKING, Callable

from inferential.scheduler.base import (
    ModelAwareScheduler,
    QueueFullError,
    get_policy,
    register_scheduler,
)
from inferential.scheduler.request import InferenceRequest

if TYPE_CHECKING:
    from inferential.config.schema import SchedulingConfig

logger = logging.getLogger("inferential.scheduler")


@register_scheduler("model_deadline")
class ModelDeadlineScheduler(ModelAwareScheduler):
    """Per-model deadline-aware scheduler.

    Routes requests by model_id. Within each model queue, requests are
    ordered by a deadline-aware score that factors in cadence, urgency,
    priority, steps remaining, and age.
    """

    def __init__(self, config: SchedulingConfig) -> None:
        self._config = config
        self._weights = config.model_deadline
        self._max_queue = config.max_queue_size
        self._ttl_ms = config.request_ttl_ms
        self._overflow_policy = config.overflow_policy
        self._queues: dict[str, list[InferenceRequest]] = defaultdict(list)
        self._counter = 0
        self._total = 0
        self._policy: Callable[[InferenceRequest], float] | None = None

    def use_policy(self, policy_name: str) -> None:
        policy = get_policy(policy_name)
        if policy is None:
            raise ValueError(f"Unknown policy '{policy_name}'")
        self._policy = policy

    def _compute_score(self, r: InferenceRequest) -> float:
        if self._policy is not None:
            return self._policy(r)

        s = 0.0
        cadence_urgency = max(0, -r.time_until_next) / 0.1
        s += cadence_urgency * self._weights.cadence_weight
        s += r.urgency * self._weights.urgency_weight
        if r.steps_remaining is not None:
            if r.steps_remaining == 0:
                s += self._weights.steps_weight
            elif r.steps_remaining <= 2:
                s += self._weights.steps_weight * 0.7
            elif r.steps_remaining <= 4:
                s += self._weights.steps_weight * 0.4
        s += (10 - min(r.priority, 10)) * self._weights.priority_weight / 10
        age_ms = (time.monotonic() - r.received_at) * 1000
        s += min(age_ms / r.latency_budget_ms, 1.0) * self._weights.age_weight
        return s

    def submit(self, request: InferenceRequest) -> None:
        if self._total >= self._max_queue:
            if self._overflow_policy == "drop_oldest":
                self._drop_oldest()
            else:
                raise QueueFullError(request.client_id)

        request.score = self._compute_score(request)
        request._counter = self._counter
        self._counter += 1
        heapq.heappush(self._queues[request.model_id], request)
        self._total += 1

    def _drop_oldest(self) -> None:
        # Drop oldest request across all model queues
        oldest_model = None
        oldest_time = float("inf")
        for model_id, heap in self._queues.items():
            if heap:
                for r in heap:
                    if r.received_at < oldest_time:
                        oldest_time = r.received_at
                        oldest_model = model_id
        if oldest_model is not None:
            heap = self._queues[oldest_model]
            oldest_idx = min(range(len(heap)), key=lambda j: heap[j].received_at)
            heap[oldest_idx] = heap[-1]
            heap.pop()
            if heap:
                heapq.heapify(heap)
            else:
                del self._queues[oldest_model]
            self._total -= 1
            logger.debug("Dropped oldest request from model %s", oldest_model)

    def resubmit(self, request: InferenceRequest) -> None:
        request.score = self._compute_score(request)
        request._counter = self._counter
        self._counter += 1
        heapq.heappush(self._queues[request.model_id], request)
        self._total += 1

    # --- ModelAwareScheduler interface ---

    def models(self) -> list[str]:
        return [m for m, q in self._queues.items() if q]

    def next_batch_for_model(self, model_id: str) -> list[InferenceRequest]:
        heap = self._queues.get(model_id)
        if heap:
            self._total -= 1
            item = heapq.heappop(heap)
            if not heap:
                del self._queues[model_id]
            return [item]
        return []

    def queue_len_for_model(self, model_id: str) -> int:
        heap = self._queues.get(model_id)
        return len(heap) if heap else 0

    # --- Scheduler base interface ---

    def tick(self) -> int:
        expired = 0
        empty_models = []
        for model_id, heap in self._queues.items():
            before = len(heap)
            self._queues[model_id] = [r for r in heap if not self._is_expired(r, self._ttl_ms)]
            removed = before - len(self._queues[model_id])
            self._total -= removed
            expired += removed
            if removed and self._queues[model_id]:
                heapq.heapify(self._queues[model_id])
            if not self._queues[model_id]:
                empty_models.append(model_id)
        for model_id in empty_models:
            del self._queues[model_id]
        return expired

    def on_client_disconnected(self, client_id: str) -> None:
        empty_models = []
        for model_id, heap in self._queues.items():
            before = len(heap)
            self._queues[model_id] = [r for r in heap if r.client_id != client_id]
            removed = before - len(self._queues[model_id])
            self._total -= removed
            if removed and self._queues[model_id]:
                heapq.heapify(self._queues[model_id])
            if not self._queues[model_id]:
                empty_models.append(model_id)
        for model_id in empty_models:
            del self._queues[model_id]

    def status(self) -> dict:
        return {
            "strategy": "model_deadline",
            "queue_size": self._total,
            "max_queue_size": self._max_queue,
            "models": {m: len(q) for m, q in self._queues.items() if q},
        }

    def queue_len(self) -> int:
        return self._total
