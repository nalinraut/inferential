from __future__ import annotations

import heapq
import logging
import time
from typing import TYPE_CHECKING, Callable

from inferential.scheduler.base import (
    QueueFullError,
    Scheduler,
    get_policy,
    register_scheduler,
)
from inferential.scheduler.request import InferenceRequest

if TYPE_CHECKING:
    from inferential.config.schema import SchedulingConfig

logger = logging.getLogger("inferential.scheduler")


@register_scheduler("tiered_deadline")
class TieredDeadlineScheduler(Scheduler):
    """Per-tier deadline-aware scheduler (legacy).

    Combines priority-tier isolation (requests never block across tiers)
    with deadline-aware scoring within each tier. Superseded by
    ModelDeadlineScheduler for per-model dispatch.
    """

    def __init__(self, config: SchedulingConfig) -> None:
        self._config = config
        self._num_tiers_val = config.priority_tiered.num_tiers
        self._weights = config.deadline_aware
        self._max_queue = config.max_queue_size
        self._ttl_ms = config.request_ttl_ms
        self._overflow_policy = config.overflow_policy
        self._tiers: list[list[InferenceRequest]] = [[] for _ in range(self._num_tiers_val)]
        self._counter = 0
        self._total = 0
        self._policy: Callable[[InferenceRequest], float] | None = None

    def use_policy(self, policy_name: str) -> None:
        policy = get_policy(policy_name)
        if policy is None:
            raise ValueError(f"Unknown policy '{policy_name}'")
        self._policy = policy

    def _tier_for(self, priority: int) -> int:
        return min(priority, self._num_tiers_val - 1)

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
                self._drop_from_lowest_tier()
            else:
                raise QueueFullError(request.client_id)

        tier = self._tier_for(request.priority)
        request.score = self._compute_score(request)
        request._counter = self._counter
        self._counter += 1
        heapq.heappush(self._tiers[tier], request)
        self._total += 1

    def _drop_from_lowest_tier(self) -> None:
        for i in range(self._num_tiers_val - 1, -1, -1):
            heap = self._tiers[i]
            if heap:
                oldest_idx = min(range(len(heap)), key=lambda j: heap[j].received_at)
                heap[oldest_idx] = heap[-1]
                heap.pop()
                if heap:
                    heapq.heapify(heap)
                self._total -= 1
                logger.debug("Dropped oldest request from tier %d", i)
                return

    def resubmit(self, request: InferenceRequest) -> None:
        tier = self._tier_for(request.priority)
        request.score = self._compute_score(request)
        request._counter = self._counter
        self._counter += 1
        heapq.heappush(self._tiers[tier], request)
        self._total += 1

    # --- Tier access methods ---

    def num_tiers(self) -> int:
        return self._num_tiers_val

    def next_batch_for_tier(self, tier: int) -> list[InferenceRequest]:
        if 0 <= tier < self._num_tiers_val and self._tiers[tier]:
            self._total -= 1
            return [heapq.heappop(self._tiers[tier])]
        return []

    def queue_len_for_tier(self, tier: int) -> int:
        if 0 <= tier < self._num_tiers_val:
            return len(self._tiers[tier])
        return 0

    def next_batch(self) -> list[InferenceRequest]:
        for tier in range(self._num_tiers_val):
            batch = self.next_batch_for_tier(tier)
            if batch:
                return batch
        return []

    # --- Scheduler base interface ---

    def tick(self) -> int:
        expired = 0
        for i in range(self._num_tiers_val):
            before = len(self._tiers[i])
            self._tiers[i] = [r for r in self._tiers[i] if not self._is_expired(r, self._ttl_ms)]
            removed = before - len(self._tiers[i])
            self._total -= removed
            expired += removed
            if removed:
                heapq.heapify(self._tiers[i])
        return expired

    def on_client_disconnected(self, client_id: str) -> None:
        for i in range(self._num_tiers_val):
            before = len(self._tiers[i])
            self._tiers[i] = [r for r in self._tiers[i] if r.client_id != client_id]
            removed = before - len(self._tiers[i])
            self._total -= removed
            if removed:
                heapq.heapify(self._tiers[i])

    def status(self) -> dict:
        return {
            "strategy": "tiered_deadline",
            "queue_size": self._total,
            "max_queue_size": self._max_queue,
            "tiers": [len(t) for t in self._tiers],
        }

    def queue_len(self) -> int:
        return self._total
