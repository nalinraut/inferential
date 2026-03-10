from __future__ import annotations

import numpy as np
import pytest

from inferential.config.schema import ObservationConfig
from inferential.observation.assembler import ObservationAssembler
from inferential.observation.slots import ObservationError, SlotInfo, validate_slots
from inferential.proto import RAW, Client, Observation, Tensor, dtype_from_numpy


def _make_obs(
    client_id: str = "1",
    model_id: str = "policy-v2",
    slots: list[tuple[str, np.ndarray]] | None = None,
) -> tuple[bytes, bytes]:
    obs = Observation()
    obs.client.CopyFrom(Client(id=client_id))
    obs.model_id = model_id
    obs.timestamp_ns = int(1e18)  # some past time

    payload_parts = []
    offset = 0

    for key, arr in slots or []:
        data = arr.tobytes()
        tensor = Tensor()
        tensor.key = key
        tensor.dtype = dtype_from_numpy(arr.dtype)
        tensor.shape.extend(arr.shape)
        tensor.byte_offset = offset
        tensor.byte_length = len(data)
        tensor.timestamp_ns = obs.timestamp_ns
        tensor.encoding = RAW
        obs.tensors.append(tensor)
        payload_parts.append(data)
        offset += len(data)

    envelope = obs.SerializeToString()
    payload = b"".join(payload_parts)
    return envelope, payload


def test_decode_basic():
    assembler = ObservationAssembler(ObservationConfig())
    joints = np.zeros((1, 7), dtype=np.float32)
    envelope, payload = _make_obs(slots=[("joint_positions", joints)])

    decoded = assembler.decode(envelope, payload, received_at=0.0)
    assert decoded.client_id == "1"
    assert decoded.model_id == "policy-v2"
    assert len(decoded.slots) == 1
    assert decoded.slots[0].key == "joint_positions"


def test_decode_multiple_slots():
    assembler = ObservationAssembler(ObservationConfig())
    joints = np.zeros((1, 7), dtype=np.float32)
    image = np.zeros((3, 224, 224), dtype=np.uint8)

    envelope, payload = _make_obs(slots=[("joint_positions", joints), ("camera_rgb", image)])

    decoded = assembler.decode(envelope, payload, received_at=0.0)
    assert len(decoded.slots) == 2
    assert decoded.slots[0].key == "joint_positions"
    assert decoded.slots[1].key == "camera_rgb"


def test_decode_missing_client():
    assembler = ObservationAssembler(ObservationConfig())
    obs = Observation()
    obs.model_id = "policy"

    with pytest.raises(ObservationError, match="Missing client"):
        assembler.decode(obs.SerializeToString(), b"", received_at=0.0)


def test_decode_missing_model_id():
    assembler = ObservationAssembler(ObservationConfig())
    obs = Observation()
    obs.client.CopyFrom(Client(id="1"))

    with pytest.raises(ObservationError, match="Missing model_id"):
        assembler.decode(obs.SerializeToString(), b"", received_at=0.0)


def test_decode_invalid_protobuf():
    assembler = ObservationAssembler(ObservationConfig())
    with pytest.raises(ObservationError, match="Failed to decode"):
        assembler.decode(b"\xff\xff\xff", b"", received_at=0.0)


def test_validate_slots_ok():
    slots = [
        SlotInfo(
            key="data",
            dtype="float32",
            shape=(4,),
            byte_offset=0,
            byte_length=16,
            timestamp_ns=0,
            encoding="raw",
        )
    ]
    validate_slots(slots, payload_size=16)  # Should not raise


def test_validate_slots_exceeds_payload():
    slots = [
        SlotInfo(
            key="data",
            dtype="float32",
            shape=(4,),
            byte_offset=0,
            byte_length=32,
            timestamp_ns=0,
            encoding="raw",
        )
    ]
    with pytest.raises(ObservationError, match="exceeds payload size"):
        validate_slots(slots, payload_size=16)


def test_validate_slots_zero_length():
    slots = [
        SlotInfo(
            key="data",
            dtype="float32",
            shape=(4,),
            byte_offset=0,
            byte_length=0,
            timestamp_ns=0,
            encoding="raw",
        )
    ]
    with pytest.raises(ObservationError, match="zero byte_length"):
        validate_slots(slots, payload_size=16)
