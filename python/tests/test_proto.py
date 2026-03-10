from __future__ import annotations

import numpy as np
import pytest

from inferential.proto import (
    FLOAT32,
    JPEG,
    RAW,
    UINT8,
    Client,
    ModelResponse,
    Observation,
    Tensor,
    dtype_from_numpy,
    dtype_to_numpy,
    encoding_from_string,
    encoding_to_string,
)


def test_observation_roundtrip():
    obs = Observation()
    obs.client.CopyFrom(Client(id="1", type="franka"))
    obs.model_id = "policy-v2"
    obs.timestamp_ns = 1_000_000_000
    obs.urgency = 0.8

    tensor = Tensor()
    tensor.key = "joint_positions"
    tensor.dtype = FLOAT32
    tensor.shape.extend([1, 7])
    tensor.byte_offset = 0
    tensor.byte_length = 28
    tensor.timestamp_ns = 1_000_000_000
    tensor.encoding = RAW
    obs.tensors.append(tensor)

    data = obs.SerializeToString()
    assert len(data) > 0

    parsed = Observation()
    parsed.ParseFromString(data)
    assert parsed.client.id == "1"
    assert parsed.client.type == "franka"
    assert parsed.model_id == "policy-v2"
    assert parsed.urgency == pytest.approx(0.8)
    assert len(parsed.tensors) == 1
    assert parsed.tensors[0].key == "joint_positions"
    assert parsed.tensors[0].dtype == FLOAT32
    assert list(parsed.tensors[0].shape) == [1, 7]
    assert parsed.tensors[0].encoding == RAW


def test_model_response_roundtrip():
    resp = ModelResponse()
    resp.client.CopyFrom(Client(id="1"))
    resp.response_id = "abc123"
    resp.model_id = "policy-v2"
    resp.timestamp_ns = 2_000_000_000
    resp.inference_latency_ms = 12.5

    tensor = Tensor()
    tensor.key = "actions"
    tensor.dtype = FLOAT32
    tensor.shape.extend([1, 7])
    tensor.byte_offset = 0
    tensor.byte_length = 28
    tensor.encoding = RAW
    resp.tensors.append(tensor)

    data = resp.SerializeToString()
    parsed = ModelResponse()
    parsed.ParseFromString(data)
    assert parsed.client.id == "1"
    assert parsed.response_id == "abc123"
    assert parsed.inference_latency_ms == pytest.approx(12.5)
    assert len(parsed.tensors) == 1
    assert parsed.tensors[0].key == "actions"
    assert parsed.tensors[0].dtype == FLOAT32
    assert list(parsed.tensors[0].shape) == [1, 7]


def test_model_response_multi_tensor():
    resp = ModelResponse()
    resp.client.CopyFrom(Client(id="1"))
    resp.response_id = "resp-multi"
    resp.model_id = "planner"

    actions_tensor = Tensor()
    actions_tensor.key = "actions"
    actions_tensor.dtype = FLOAT32
    actions_tensor.shape.extend([7])
    actions_tensor.byte_offset = 0
    actions_tensor.byte_length = 28
    actions_tensor.encoding = RAW
    resp.tensors.append(actions_tensor)

    confidence_tensor = Tensor()
    confidence_tensor.key = "confidence"
    confidence_tensor.dtype = FLOAT32
    confidence_tensor.shape.extend([1])
    confidence_tensor.byte_offset = 28
    confidence_tensor.byte_length = 4
    confidence_tensor.encoding = RAW
    resp.tensors.append(confidence_tensor)

    data = resp.SerializeToString()
    parsed = ModelResponse()
    parsed.ParseFromString(data)
    assert len(parsed.tensors) == 2
    assert parsed.tensors[0].key == "actions"
    assert parsed.tensors[1].key == "confidence"


def test_observation_metadata():
    obs = Observation()
    obs.client.CopyFrom(Client(id="1"))
    obs.model_id = "vlm"
    obs.metadata["language"] = "pick up the red cube"

    data = obs.SerializeToString()
    parsed = Observation()
    parsed.ParseFromString(data)
    assert parsed.metadata["language"] == "pick up the red cube"


def test_optional_steps_remaining():
    obs = Observation()
    obs.client.CopyFrom(Client(id="1"))
    obs.model_id = "policy"
    # Don't set steps_remaining

    data = obs.SerializeToString()
    parsed = Observation()
    parsed.ParseFromString(data)
    assert not parsed.HasField("steps_remaining")

    obs2 = Observation()
    obs2.client.CopyFrom(Client(id="1"))
    obs2.model_id = "policy"
    obs2.steps_remaining = 5

    data2 = obs2.SerializeToString()
    parsed2 = Observation()
    parsed2.ParseFromString(data2)
    assert parsed2.HasField("steps_remaining")
    assert parsed2.steps_remaining == 5


def test_dtype_numpy_roundtrip():
    for np_dtype_str in ["float16", "float32", "float64", "uint8", "int32", "int64", "bool"]:
        np_dt = np.dtype(np_dtype_str)
        enum_val = dtype_from_numpy(np_dt)
        back = dtype_to_numpy(enum_val)
        assert back == np_dt


def test_dtype_unknown_raises():
    with pytest.raises(ValueError, match="No DType mapping"):
        dtype_from_numpy(np.dtype("complex128"))

    with pytest.raises(ValueError, match="Unknown DType"):
        dtype_to_numpy(999)


def test_encoding_roundtrip():
    for s in ["raw", "jpeg", "png"]:
        enum_val = encoding_from_string(s)
        back = encoding_to_string(enum_val)
        assert back == s


def test_encoding_unknown_raises():
    with pytest.raises(ValueError, match="Unknown encoding string"):
        encoding_from_string("webp")

    with pytest.raises(ValueError, match="Unknown Encoding"):
        encoding_to_string(999)


def test_tensor_encoding_enum():
    tensor = Tensor()
    tensor.encoding = JPEG
    assert tensor.encoding == JPEG

    tensor2 = Tensor()
    tensor2.encoding = RAW
    assert tensor2.encoding == RAW


def test_tensor_dtype_enum():
    tensor = Tensor()
    tensor.dtype = UINT8
    assert tensor.dtype == UINT8


def test_client_message():
    client = Client(id="42", type="franka")
    assert client.id == "42"
    assert client.type == "franka"
