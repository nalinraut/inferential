from __future__ import annotations

import asyncio

import numpy as np
import pytest
import zmq
import zmq.asyncio

from inferential.client import Connection
from inferential.proto import FLOAT32, RAW, Client, ModelResponse, Observation, Tensor


@pytest.fixture
async def router_socket():
    ctx = zmq.asyncio.Context()
    sock = ctx.socket(zmq.ROUTER)
    sock.setsockopt(zmq.ROUTER_HANDOVER, 1)
    sock.setsockopt(zmq.LINGER, 0)
    sock.bind("tcp://127.0.0.1:15556")
    yield sock
    sock.close()
    ctx.term()


@pytest.fixture
def connection():
    conn = Connection(server="tcp://127.0.0.1:15556", client_id=42, client_type="test")
    yield conn
    conn.close()


@pytest.mark.asyncio
async def test_observe_sends_protobuf(router_socket, connection):
    model = connection.model("policy-v2")

    joints = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    model.observe(urgency=0.5, joint_positions=joints)

    # Give time for message to arrive
    await asyncio.sleep(0.1)

    if router_socket.poll(1000, zmq.POLLIN):
        frames = await router_socket.recv_multipart()
        # frames = [identity, b'', envelope, payload]
        identity = frames[0]
        assert identity == b"42"

        envelope = frames[2]
        obs = Observation()
        obs.ParseFromString(envelope)
        assert obs.client.id == "42"
        assert obs.client.type == "test"
        assert obs.model_id == "policy-v2"
        assert obs.urgency == pytest.approx(0.5)
        assert len(obs.tensors) == 1
        assert obs.tensors[0].key == "joint_positions"


@pytest.mark.asyncio
async def test_get_result_parses_response(router_socket, connection):
    model = connection.model("policy-v2")

    # Send observation first
    model.observe(joint_positions=np.zeros((3,), dtype=np.float32))
    await asyncio.sleep(0.1)

    # Drain the observation from the router
    if router_socket.poll(1000, zmq.POLLIN):
        frames = await router_socket.recv_multipart()
        identity = frames[0]

        # Build and send response
        resp = ModelResponse()
        resp.client.CopyFrom(Client(id="42"))
        resp.response_id = "resp-001"
        resp.model_id = "policy-v2"
        resp.inference_latency_ms = 8.5

        actions = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        tensor = Tensor()
        tensor.key = "actions"
        tensor.dtype = FLOAT32
        tensor.shape.extend([3])
        tensor.byte_offset = 0
        tensor.byte_length = actions.nbytes
        tensor.encoding = RAW
        resp.tensors.append(tensor)

        await router_socket.send_multipart(
            [identity, b"", resp.SerializeToString(), actions.tobytes()]
        )

    await asyncio.sleep(0.1)
    result = model.get_result(timeout_ms=1000)
    assert result is not None
    assert result["response_id"] == "resp-001"
    assert result["model_id"] == "policy-v2"
    assert "actions" in result
    np.testing.assert_array_almost_equal(result["actions"], [0.1, 0.2, 0.3])


def test_get_result_returns_none_when_empty(connection):
    model = connection.model("policy-v2")
    result = model.get_result(timeout_ms=10)
    assert result is None


def test_observe_with_metadata(connection):
    model = connection.model("vlm")
    # Should not raise
    model.observe(language="pick up the red cube")
