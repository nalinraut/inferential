from __future__ import annotations

import time
from typing import TYPE_CHECKING

import zmq
import zmq.asyncio

from inferential.transport.messages import IncomingObservation, OutgoingResponse

if TYPE_CHECKING:
    from inferential.config.schema import TransportConfig


class ZmqTransport:
    def __init__(self, config: TransportConfig) -> None:
        self._config = config
        self._ctx = zmq.asyncio.Context()
        self._socket = self._ctx.socket(zmq.ROUTER)
        self._socket.setsockopt(zmq.ROUTER_HANDOVER, 1)
        self._socket.setsockopt(zmq.ROUTER_MANDATORY, 1)
        self._socket.setsockopt(zmq.RCVHWM, config.recv_hwm)
        self._socket.setsockopt(zmq.SNDHWM, config.send_hwm)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.bind(config.bind)

    async def recv(self) -> IncomingObservation:
        frames = await self._socket.recv_multipart()
        # frames = [identity, b'', envelope_bytes, payload_bytes]
        identity = frames[0]
        envelope = frames[2] if len(frames) > 2 else b""
        payload = frames[3] if len(frames) > 3 else b""
        return IncomingObservation(
            identity=identity,
            envelope=envelope,
            payload=payload,
            received_at=time.monotonic(),
        )

    async def send(self, response: OutgoingResponse) -> None:
        await self._socket.send_multipart(
            [
                response.identity,
                b"",
                response.envelope,
                response.payload,
            ]
        )

    async def close(self) -> None:
        self._socket.close()
        self._ctx.term()
