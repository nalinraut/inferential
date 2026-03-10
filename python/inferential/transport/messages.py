from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class IncomingObservation:
    identity: bytes
    envelope: bytes
    payload: bytes
    received_at: float


@dataclass(frozen=True)
class OutgoingResponse:
    identity: bytes
    envelope: bytes
    payload: bytes
