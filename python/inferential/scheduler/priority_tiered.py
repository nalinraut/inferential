from __future__ import annotations

import logging
from collections import deque
from typing import TYPE_CHECKING

from inferential.scheduler.base import QueueFullError, Scheduler, register_scheduler
from inferential.scheduler.request import InferenceRequest

if TYPE_CHECKING:
    from inferential.config.schema import SchedulingConfig

logger = logging.getLogger("inferential.scheduler")


@register_scheduler("priority_tiered")
class PriorityTieredScheduler(Scheduler):
    def __init__(self, config: SchedulingConfig) -> None:
        self._max_queue = config.max_queue_size
        self._num_tiers = config.priority_tiered.num_tiers
        self._ttl_ms = config.request_ttl_ms
        self._overflow_policy = config.overflow_policy
        self._tiers: list[deque[InferenceRequest]] = [deque() for _ in range(self._num_tiers)]
        self._total = 0

    def _tier_for(self, priority: int) -> int:
        # priority 0 = highest = tier 0
        # Clamp to available tiers
        return min(priority, self._num_tiers - 1)

    def submit(self, request: InferenceRequest) -> None:
        if self._total >= self._max_queue:
            if self._overflow_policy == "drop_oldest":
                self._drop_oldest()
            else:
                raise QueueFullError(request.client_id)

        tier = self._tier_for(request.priority)
        self._tiers[tier].append(request)
        self._total += 1

    def _drop_oldest(self) -> None:
        # Drop from the lowest-priority (highest index) non-empty tier
        for i in range(self._num_tiers - 1, -1, -1):
            if self._tiers[i]:
                dropped = self._tiers[i].popleft()
                self._total -= 1
                logger.debug("Dropped oldest request from client %s", dropped.client_id)
                return

    def resubmit(self, request: InferenceRequest) -> None:
        tier = self._tier_for(request.priority)
        self._tiers[tier].append(request)
        self._total += 1

    def next_batch(self) -> list[InferenceRequest]:
        # Strict priority: drain highest tier first (FIFO within tier)
        for tier in self._tiers:
            if tier:
                self._total -= 1
                return [tier.popleft()]
        return []

    def tick(self) -> int:
        expired = 0
        for i, tier in enumerate(self._tiers):
            before = len(tier)
            self._tiers[i] = deque(r for r in tier if not self._is_expired(r, self._ttl_ms))
            removed = before - len(self._tiers[i])
            self._total -= removed
            expired += removed
        return expired

    def on_client_disconnected(self, client_id: str) -> None:
        for i, tier in enumerate(self._tiers):
            before = len(tier)
            self._tiers[i] = deque(r for r in tier if r.client_id != client_id)
            self._total -= before - len(self._tiers[i])

    def status(self) -> dict:
        return {
            "strategy": "priority_tiered",
            "queue_size": self._total,
            "max_queue_size": self._max_queue,
            "tiers": [len(t) for t in self._tiers],
        }

    def queue_len(self) -> int:
        return self._total
