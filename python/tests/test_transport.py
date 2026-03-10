from __future__ import annotations

import asyncio

import pytest
import zmq
import zmq.asyncio

from inferential.config.schema import TransportConfig
from inferential.transport.messages import IncomingObservation, OutgoingResponse
from inferential.transport.zmq_transport import ZmqTransport


@pytest.fixture
def transport_config() -> TransportConfig:
    return TransportConfig(bind="tcp://127.0.0.1:15555")


@pytest.fixture
async def transport(transport_config):
    t = ZmqTransport(transport_config)
    yield t
    await t.close()


@pytest.fixture
async def dealer_socket(transport_config):
    ctx = zmq.asyncio.Context()
    sock = ctx.socket(zmq.DEALER)
    sock.identity = b"test-robot"
    sock.setsockopt(zmq.LINGER, 0)
    addr = transport_config.bind.replace("*", "127.0.0.1")
    sock.connect(addr)
    await asyncio.sleep(0.05)  # Allow connection to establish
    yield sock
    sock.close()
    ctx.term()


@pytest.mark.asyncio
async def test_recv(transport, dealer_socket):
    # Client sends: [b'', envelope, payload]
    await dealer_socket.send_multipart([b"", b"envelope-data", b"payload-data"])
    obs = await asyncio.wait_for(transport.recv(), timeout=2.0)

    assert obs.identity == b"test-robot"
    assert obs.envelope == b"envelope-data"
    assert obs.payload == b"payload-data"
    assert obs.received_at > 0


@pytest.mark.asyncio
async def test_send(transport, dealer_socket):
    response = OutgoingResponse(
        identity=b"test-robot",
        envelope=b"resp-envelope",
        payload=b"resp-payload",
    )
    await transport.send(response)

    frames = await asyncio.wait_for(dealer_socket.recv_multipart(), timeout=2.0)
    # ROUTER sends: [b'', envelope, payload]
    assert frames[1] == b"resp-envelope"
    assert frames[2] == b"resp-payload"


@pytest.mark.asyncio
async def test_roundtrip(transport, dealer_socket):
    # Client sends observation
    await dealer_socket.send_multipart([b"", b"obs", b"tensor-data"])
    obs = await asyncio.wait_for(transport.recv(), timeout=2.0)
    assert obs.envelope == b"obs"

    # Server sends response back
    await transport.send(
        OutgoingResponse(identity=obs.identity, envelope=b"actions", payload=b"result")
    )
    frames = await asyncio.wait_for(dealer_socket.recv_multipart(), timeout=2.0)
    assert frames[1] == b"actions"
    assert frames[2] == b"result"


def test_incoming_observation_frozen():
    obs = IncomingObservation(identity=b"id", envelope=b"env", payload=b"pay", received_at=1.0)
    with pytest.raises(AttributeError):
        obs.identity = b"new"


def test_outgoing_response_frozen():
    resp = OutgoingResponse(identity=b"id", envelope=b"env", payload=b"pay")
    with pytest.raises(AttributeError):
        resp.identity = b"new"
