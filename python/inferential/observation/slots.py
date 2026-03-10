from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SlotInfo:
    key: str
    dtype: str
    shape: tuple[int, ...]
    byte_offset: int
    byte_length: int
    timestamp_ns: int
    encoding: str


class ObservationError(Exception):
    pass


def validate_slots(slots: list[SlotInfo], payload_size: int) -> None:
    for slot in slots:
        end = slot.byte_offset + slot.byte_length
        if end > payload_size:
            raise ObservationError(
                f"Slot '{slot.key}' byte range [{slot.byte_offset}:{end}) "
                f"exceeds payload size {payload_size}"
            )
        if slot.byte_length == 0:
            raise ObservationError(f"Slot '{slot.key}' has zero byte_length")
