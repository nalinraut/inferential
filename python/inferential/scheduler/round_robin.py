from __future__ import annotations

import logging
from collections import OrderedDict, deque
from typing import TYPE_CHECKING

from inferential.scheduler.base import QueueFullError, Scheduler, register_scheduler
from inferential.scheduler.request import InferenceRequest

if TYPE_CHECKING:
    from inferential.config.schema import SchedulingConfig

logger = logging.getLogger("inferential.scheduler")


@register_scheduler("round_robin")
class RoundRobinScheduler(Scheduler):
    def __init__(self, config: SchedulingConfig) -> None:
        self._max_queue = config.max_queue_size
        self._ttl_ms = config.request_ttl_ms
        self._overflow_policy = config.overflow_policy
        self._queues: OrderedDict[str, deque[InferenceRequest]] = OrderedDict()
        self._total = 0

    def submit(self, request: InferenceRequest) -> None:
        if self._total >= self._max_queue:
            if self._overflow_policy == "drop_oldest":
                self._drop_oldest()
            else:
                raise QueueFullError(request.client_id)

        client_id = request.client_id
        if client_id not in self._queues:
            self._queues[client_id] = deque()
        self._queues[client_id].append(request)
        self._total += 1

    def _drop_oldest(self) -> None:
        oldest_client = None
        oldest_time = float("inf")
        for client_id, q in self._queues.items():
            if q and q[0].received_at < oldest_time:
                oldest_time = q[0].received_at
                oldest_client = client_id
        if oldest_client is not None:
            dropped = self._queues[oldest_client].popleft()
            self._total -= 1
            logger.debug("Dropped oldest request from client %s", dropped.client_id)

    def resubmit(self, request: InferenceRequest) -> None:
        client_id = request.client_id
        if client_id not in self._queues:
            self._queues[client_id] = deque()
        self._queues[client_id].append(request)
        self._total += 1

    def next_batch(self) -> list[InferenceRequest]:
        if self._total == 0:
            return []

        # Find the first non-empty queue
        for _ in range(len(self._queues)):
            client_id, q = self._queues.popitem(last=False)
            if q:
                request = q.popleft()
                self._total -= 1
                # Move this client to the end (round-robin rotation)
                self._queues[client_id] = q
                return [request]
            # Empty queue, put back and skip
            self._queues[client_id] = q

        return []

    def tick(self) -> int:
        expired = 0
        for client_id in list(self._queues.keys()):
            q = self._queues[client_id]
            before = len(q)
            self._queues[client_id] = deque(r for r in q if not self._is_expired(r, self._ttl_ms))
            removed = before - len(self._queues[client_id])
            self._total -= removed
            expired += removed
        return expired

    def on_client_connected(self, client_id: str) -> None:
        if client_id not in self._queues:
            self._queues[client_id] = deque()

    def on_client_disconnected(self, client_id: str) -> None:
        if client_id in self._queues:
            self._total -= len(self._queues[client_id])
            del self._queues[client_id]

    def status(self) -> dict:
        return {
            "strategy": "round_robin",
            "queue_size": self._total,
            "max_queue_size": self._max_queue,
            "active_clients": len(self._queues),
        }

    def queue_len(self) -> int:
        return self._total
