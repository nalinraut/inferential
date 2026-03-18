from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class TransportConfig(BaseModel):
    type: Literal["zmq"] = "zmq"
    bind: str = "tcp://*:5555"
    recv_hwm: int = Field(default=1000, ge=1)
    send_hwm: int = Field(default=1000, ge=1)


class DeadlineAwareConfig(BaseModel):
    cadence_weight: float = 45.0
    urgency_weight: float = 25.0
    steps_weight: float = 15.0
    priority_weight: float = 10.0
    age_weight: float = 15.0


class BatchConfig(BaseModel):
    max_batch_size: int = Field(default=8, ge=1)
    max_wait_ms: float = Field(default=10.0, gt=0)


class TieredConfig(BaseModel):
    num_tiers: int = Field(default=3, ge=1)


class SchedulingConfig(BaseModel):
    strategy: str = "deadline_aware"
    max_queue_size: int = Field(default=1000, ge=1)
    request_ttl_ms: float = Field(default=5000.0, gt=0)
    overflow_policy: Literal["drop_oldest", "reject_newest"] = "drop_oldest"
    max_retries: int = Field(default=0, ge=0)
    max_concurrent_dispatch: int = Field(default=8, ge=1)
    deadline_aware: DeadlineAwareConfig = DeadlineAwareConfig()
    batch_optimized: BatchConfig = BatchConfig()
    priority_tiered: TieredConfig = TieredConfig()


class ClientDefaults(BaseModel):
    latency_budget_ms: float = Field(default=50.0, gt=0)
    priority: int = Field(default=1, ge=0)


class ClientEntry(BaseModel):
    id: str
    model: str
    latency_budget_ms: float | None = None
    priority: int | None = None


class ClientsConfig(BaseModel):
    defaults: ClientDefaults = ClientDefaults()
    known: list[ClientEntry] = []
    accept_unknown: bool = True


class ResponseTrackingConfig(BaseModel):
    overdue_multiplier: float = Field(default=1.5, gt=1.0)
    disconnect_timeout_s: float = Field(default=10.0, gt=0)
    cadence_alpha: float = Field(default=0.3, gt=0, lt=1.0)


class ObservationConfig(BaseModel):
    max_staleness_ms: float = Field(default=1000.0, gt=0)
    temporal_alignment_threshold_ms: float = Field(default=20.0, gt=0)


class MetricsConfig(BaseModel):
    ring_buffer_size: int = Field(default=10000, ge=100)
    enabled: bool = True


class InferentialConfig(BaseModel):
    ray: dict = Field(default_factory=dict)
    transport: TransportConfig = TransportConfig()
    scheduling: SchedulingConfig = SchedulingConfig()
    clients: ClientsConfig = ClientsConfig()
    response_tracking: ResponseTrackingConfig = ResponseTrackingConfig()
    observations: ObservationConfig = ObservationConfig()
    metrics: MetricsConfig = MetricsConfig()

    def get_client_budget(self, client_id: str) -> float:
        for entry in self.clients.known:
            if entry.id == client_id and entry.latency_budget_ms is not None:
                return entry.latency_budget_ms
        return self.clients.defaults.latency_budget_ms

    def get_client_priority(self, client_id: str) -> int:
        for entry in self.clients.known:
            if entry.id == client_id and entry.priority is not None:
                return entry.priority
        return self.clients.defaults.priority
