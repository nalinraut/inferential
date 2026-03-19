from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

from inferential.observation.slots import ObservationError, SlotInfo, validate_slots
from inferential.proto import ENCODING_UNSPECIFIED, Observation, dtype_to_numpy, encoding_to_string

if TYPE_CHECKING:
    from inferential.config.schema import ObservationConfig


@dataclass
class DecodedObservation:
    client_id: str
    model_id: str
    timestamp_ns: int
    urgency: float
    priority: int
    steps_remaining: int | None
    slots: list[SlotInfo]
    envelope: bytes
    payload: bytes
    received_at: float
    staleness_ms: float


class ObservationAssembler:
    def __init__(self, config: ObservationConfig) -> None:
        self._config = config

    def decode(
        self,
        envelope: bytes,
        payload: bytes,
        received_at: float,
    ) -> DecodedObservation:
        obs = Observation()
        try:
            obs.ParseFromString(envelope)
        except Exception as e:
            raise ObservationError(f"Failed to decode protobuf envelope: {e}") from e

        if not obs.HasField("client") or not obs.client.id:
            raise ObservationError("Missing client in observation")
        if not obs.model_id:
            raise ObservationError("Missing model_id in observation")

        slots = [
            SlotInfo(
                key=s.key,
                dtype=dtype_to_numpy(s.dtype).str,
                shape=tuple(s.shape) if s.shape else (),
                byte_offset=s.byte_offset,
                byte_length=s.byte_length,
                timestamp_ns=s.timestamp_ns,
                encoding=(
                    encoding_to_string(s.encoding) if s.encoding != ENCODING_UNSPECIFIED else "raw"
                ),
            )
            for s in obs.tensors
        ]

        validate_slots(slots, len(payload))

        # Temporal alignment check
        if len(slots) > 1:
            timestamps = [s.timestamp_ns for s in slots if s.timestamp_ns > 0]
            if len(timestamps) > 1:
                spread_ns = max(timestamps) - min(timestamps)
                spread_ms = spread_ns / 1_000_000
                if spread_ms > self._config.temporal_alignment_threshold_ms:
                    import logging

                    logging.getLogger("inferential.observation").warning(
                        "Temporal misalignment: %.1fms spread across slots for client %s",
                        spread_ms,
                        obs.client.id,
                    )

        # Staleness
        now_ns = int(time.time() * 1_000_000_000)
        staleness_ms = (now_ns - obs.timestamp_ns) / 1_000_000 if obs.timestamp_ns > 0 else 0.0

        steps_remaining = obs.steps_remaining if obs.HasField("steps_remaining") else None

        return DecodedObservation(
            client_id=obs.client.id,
            model_id=obs.model_id,
            timestamp_ns=obs.timestamp_ns,
            urgency=obs.urgency,
            priority=obs.priority,
            steps_remaining=steps_remaining,
            slots=slots,
            envelope=envelope,
            payload=payload,
            received_at=received_at,
            staleness_ms=staleness_ms,
        )
