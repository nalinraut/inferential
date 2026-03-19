from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Callable

from inferential.scheduler.request import InferenceRequest

if TYPE_CHECKING:
    from inferential.config.schema import SchedulingConfig


class QueueFullError(Exception):
    def __init__(self, client_id: str) -> None:
        self.client_id = client_id
        super().__init__(f"Queue full, rejecting request from client {client_id}")


class Scheduler(ABC):
    @abstractmethod
    def submit(self, request: InferenceRequest) -> None:
        """Enqueue a request. May drop oldest or raise QueueFullError on overflow."""

    def resubmit(self, request: InferenceRequest) -> None:
        """Re-queue a failed request for retry. Bypasses overflow checks."""
        self.submit(request)

    @abstractmethod
    def next_batch(self) -> list[InferenceRequest]:
        """Return next batch to dispatch. Empty list if none ready."""

    @abstractmethod
    def tick(self) -> int:
        """Periodic call for time-driven logic (TTL expiry, batch flush).

        Returns the number of requests expired by TTL.
        """

    def on_client_connected(self, client_id: str) -> None:
        pass

    def on_client_disconnected(self, client_id: str) -> None:
        pass

    @abstractmethod
    def status(self) -> dict: ...

    @abstractmethod
    def queue_len(self) -> int: ...

    @staticmethod
    def _is_expired(request: InferenceRequest, ttl_ms: float) -> bool:
        age_ms = (time.monotonic() - request.received_at) * 1000
        return age_ms > ttl_ms


class ModelAwareScheduler(Scheduler):
    """Extension for schedulers that support per-model dispatch."""

    @abstractmethod
    def next_batch_for_model(self, model_id: str) -> list[InferenceRequest]:
        """Return next batch from a specific model queue. Empty list if none ready."""

    @abstractmethod
    def models(self) -> list[str]:
        """Return model IDs that currently have queued requests."""

    @abstractmethod
    def queue_len_for_model(self, model_id: str) -> int:
        """Return queue depth for a specific model."""

    def next_batch(self) -> list[InferenceRequest]:
        """Default: drain from model with highest-scored request."""
        for model_id in self.models():
            batch = self.next_batch_for_model(model_id)
            if batch:
                return batch
        return []


# --- Registry ---

_SCHEDULER_REGISTRY: dict[str, type[Scheduler]] = {}
_POLICY_REGISTRY: dict[str, Callable[[InferenceRequest], float]] = {}


def register_scheduler(name: str) -> Callable:
    def decorator(cls: type[Scheduler]) -> type[Scheduler]:
        _SCHEDULER_REGISTRY[name] = cls
        return cls

    return decorator


def register_policy(name: str) -> Callable:
    def decorator(fn: Callable[[InferenceRequest], float]) -> Callable:
        _POLICY_REGISTRY[name] = fn
        return fn

    return decorator


def get_policy(name: str) -> Callable[[InferenceRequest], float] | None:
    return _POLICY_REGISTRY.get(name)


def create_scheduler(config: SchedulingConfig) -> Scheduler:
    if config.strategy not in _SCHEDULER_REGISTRY:
        available = list(_SCHEDULER_REGISTRY.keys())
        raise ValueError(f"Unknown scheduler strategy '{config.strategy}'. Available: {available}")
    cls = _SCHEDULER_REGISTRY[config.strategy]
    return cls(config)  # type: ignore[call-arg]
