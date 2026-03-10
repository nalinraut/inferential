from __future__ import annotations

import heapq
import logging
import time
from typing import TYPE_CHECKING, Callable

from inferential.scheduler.base import QueueFullError, Scheduler, get_policy, register_scheduler
from inferential.scheduler.request import InferenceRequest

if TYPE_CHECKING:
    from inferential.config.schema import SchedulingConfig

logger = logging.getLogger("inferential.scheduler")


@register_scheduler("deadline_aware")
class DeadlineAwareScheduler(Scheduler):
    def __init__(self, config: SchedulingConfig) -> None:
        self._config = config
        self._weights = config.deadline_aware
        self._max_queue = config.max_queue_size
        self._ttl_ms = config.request_ttl_ms
        self._overflow_policy = config.overflow_policy
        self._queue: list[InferenceRequest] = []
        self._counter = 0
        self._policy: Callable[[InferenceRequest], float] | None = None

    def use_policy(self, policy_name: str) -> None:
        policy = get_policy(policy_name)
        if policy is None:
            raise ValueError(f"Unknown policy '{policy_name}'")
        self._policy = policy

    def submit(self, request: InferenceRequest) -> None:
        if len(self._queue) >= self._max_queue:
            if self._overflow_policy == "drop_oldest":
                oldest = min(self._queue, key=lambda r: r.received_at)
                self._queue.remove(oldest)
                heapq.heapify(self._queue)
                logger.debug("Dropped oldest request from client %s", oldest.client_id)
            else:
                raise QueueFullError(request.client_id)
        request.score = self._compute_score(request)
        request._counter = self._counter
        self._counter += 1
        heapq.heappush(self._queue, request)

    def resubmit(self, request: InferenceRequest) -> None:
        request.score = self._compute_score(request)
        request._counter = self._counter
        self._counter += 1
        heapq.heappush(self._queue, request)

    def _compute_score(self, r: InferenceRequest) -> float:
        if self._policy is not None:
            return self._policy(r)

        s = 0.0

        # Cadence urgency: negative time_until_next = overdue
        cadence_urgency = max(0, -r.time_until_next) / 0.1  # normalized
        s += cadence_urgency * self._weights.cadence_weight

        # Robot-reported urgency
        s += r.urgency * self._weights.urgency_weight

        # Steps remaining (if provided)
        if r.steps_remaining is not None:
            if r.steps_remaining == 0:
                s += self._weights.steps_weight
            elif r.steps_remaining <= 2:
                s += self._weights.steps_weight * 0.7
            elif r.steps_remaining <= 4:
                s += self._weights.steps_weight * 0.4

        # Static priority
        s += (10 - min(r.priority, 10)) * self._weights.priority_weight / 10

        # Queue age (anti-starvation)
        age_ms = (time.monotonic() - r.received_at) * 1000
        s += min(age_ms / r.latency_budget_ms, 1.0) * self._weights.age_weight

        return s

    def next_batch(self) -> list[InferenceRequest]:
        if self._queue:
            return [heapq.heappop(self._queue)]
        return []

    def tick(self) -> int:
        before = len(self._queue)
        self._queue = [r for r in self._queue if not self._is_expired(r, self._ttl_ms)]
        expired = before - len(self._queue)
        if expired:
            heapq.heapify(self._queue)
        return expired

    def status(self) -> dict:
        return {
            "strategy": "deadline_aware",
            "queue_size": len(self._queue),
            "max_queue_size": self._max_queue,
        }

    def queue_len(self) -> int:
        return len(self._queue)
