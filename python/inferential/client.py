from __future__ import annotations

import asyncio
import time
from typing import Any

import numpy as np
import zmq
import zmq.asyncio

from inferential.proto import (
    RAW,
    Client,
    ModelResponse,
    Observation,
    Tensor,
    dtype_from_numpy,
    dtype_to_numpy,
)


def _build_observation(
    client_id: str,
    client_type: str,
    model_id: str,
    urgency: float,
    steps_remaining: int | None,
    **kwargs: Any,
) -> tuple[bytes, bytes]:
    obs = Observation()
    obs.client.CopyFrom(Client(id=client_id, type=client_type))
    obs.model_id = model_id
    obs.timestamp_ns = int(time.time() * 1_000_000_000)
    obs.urgency = urgency
    if steps_remaining is not None:
        obs.steps_remaining = steps_remaining

    payload_parts: list[bytes] = []
    offset = 0

    for key, value in kwargs.items():
        if isinstance(value, np.ndarray):
            data = value.tobytes()
            tensor = Tensor()
            tensor.key = key
            tensor.dtype = dtype_from_numpy(value.dtype)
            tensor.shape.extend(value.shape)
            tensor.byte_offset = offset
            tensor.byte_length = len(data)
            tensor.timestamp_ns = obs.timestamp_ns
            tensor.encoding = RAW
            obs.tensors.append(tensor)
            payload_parts.append(data)
            offset += len(data)
        elif isinstance(value, str):
            obs.metadata[key] = value

    payload = b"".join(payload_parts)
    envelope = obs.SerializeToString()
    return envelope, payload


def _parse_response(envelope: bytes, payload: bytes) -> dict[str, Any]:
    resp = ModelResponse()
    resp.ParseFromString(envelope)

    output: dict[str, Any] = {
        "response_id": resp.response_id,
        "model_id": resp.model_id,
        "inference_latency_ms": resp.inference_latency_ms,
    }

    for tensor in resp.tensors:
        if tensor.byte_length > 0 and tensor.dtype:
            np_dtype = dtype_to_numpy(tensor.dtype)
            shape = tuple(tensor.shape) if tensor.shape else ()
            data = payload[tensor.byte_offset : tensor.byte_offset + tensor.byte_length]
            output[tensor.key] = np.frombuffer(data, dtype=np_dtype).reshape(shape)

    for k, v in resp.metadata.items():
        output[k] = v

    return output


def _configure_socket(
    socket: zmq.Socket,
    client_id: str | int,
    reconnect_ivl_ms: int,
    reconnect_max_ms: int,
) -> None:
    socket.identity = str(client_id).encode()
    socket.setsockopt(zmq.RECONNECT_IVL, reconnect_ivl_ms)
    socket.setsockopt(zmq.RECONNECT_IVL_MAX, reconnect_max_ms)
    socket.setsockopt(zmq.RCVHWM, 100)
    socket.setsockopt(zmq.SNDHWM, 100)
    socket.setsockopt(zmq.LINGER, 0)


def _normalize_server(server: str) -> str:
    if not server.startswith("tcp://"):
        return f"tcp://{server}"
    return server


# ---------------------------------------------------------------------------
# Synchronous client
# ---------------------------------------------------------------------------


class Connection:
    def __init__(
        self,
        server: str = "localhost:5555",
        client_id: str | int = "",
        client_type: str = "",
        reconnect_ivl_ms: int = 100,
        reconnect_max_ms: int = 5000,
    ) -> None:
        self._server = server
        self._client_id = str(client_id)
        self._client_type = client_type
        self._ctx = zmq.Context()
        self._socket = self._ctx.socket(zmq.DEALER)
        _configure_socket(self._socket, client_id, reconnect_ivl_ms, reconnect_max_ms)
        self._socket.connect(_normalize_server(server))

    def model(
        self,
        model_id: str,
        latency_budget_ms: float = 50.0,
        priority: int = 1,
    ) -> Model:
        return Model(
            connection=self,
            model_id=model_id,
            latency_budget_ms=latency_budget_ms,
            priority=priority,
        )

    def _send(self, envelope: bytes, payload: bytes) -> None:
        self._socket.send_multipart([b"", envelope, payload])

    def _recv(self, timeout_ms: int = 0) -> tuple[bytes, bytes] | None:
        if timeout_ms > 0:
            if self._socket.poll(timeout_ms, zmq.POLLIN) == 0:
                return None
        elif not self._socket.poll(0, zmq.POLLIN):
            return None

        frames = self._socket.recv_multipart(zmq.NOBLOCK)
        # frames = [b'', envelope, payload]
        envelope = frames[1] if len(frames) > 1 else b""
        payload = frames[2] if len(frames) > 2 else b""
        return envelope, payload

    def close(self) -> None:
        self._socket.close()
        self._ctx.term()


class Model:
    def __init__(
        self,
        connection: Connection,
        model_id: str,
        latency_budget_ms: float = 50.0,
        priority: int = 1,
    ) -> None:
        self._conn = connection
        self._model_id = model_id
        self._latency_budget_ms = latency_budget_ms
        self._priority = priority

    def observe(
        self,
        urgency: float = 0.0,
        steps_remaining: int | None = None,
        **kwargs: Any,
    ) -> None:
        envelope, payload = _build_observation(
            self._conn._client_id,
            self._conn._client_type,
            self._model_id,
            urgency,
            steps_remaining,
            **kwargs,
        )
        self._conn._send(envelope, payload)

    def get_result(self, timeout_ms: int = 100) -> dict | None:
        result = self._conn._recv(timeout_ms)
        if result is None:
            return None
        return _parse_response(*result)


# ---------------------------------------------------------------------------
# Async client
# ---------------------------------------------------------------------------


class AsyncConnection:
    def __init__(
        self,
        server: str = "localhost:5555",
        client_id: str | int = "",
        client_type: str = "",
        reconnect_ivl_ms: int = 100,
        reconnect_max_ms: int = 5000,
    ) -> None:
        self._server = server
        self._client_id = str(client_id)
        self._client_type = client_type
        self._ctx = zmq.asyncio.Context()
        self._socket = self._ctx.socket(zmq.DEALER)
        _configure_socket(self._socket, client_id, reconnect_ivl_ms, reconnect_max_ms)
        self._socket.connect(_normalize_server(server))

    def model(
        self,
        model_id: str,
        latency_budget_ms: float = 50.0,
        priority: int = 1,
    ) -> AsyncModel:
        return AsyncModel(
            connection=self,
            model_id=model_id,
            latency_budget_ms=latency_budget_ms,
            priority=priority,
        )

    async def _send(self, envelope: bytes, payload: bytes) -> None:
        await self._socket.send_multipart([b"", envelope, payload])

    async def _recv(self, timeout_ms: int = 0) -> tuple[bytes, bytes] | None:
        try:
            frames = await asyncio.wait_for(
                self._socket.recv_multipart(),
                timeout=timeout_ms / 1000 if timeout_ms > 0 else 0,
            )
        except (asyncio.TimeoutError, TimeoutError):
            return None
        envelope = frames[1] if len(frames) > 1 else b""
        payload = frames[2] if len(frames) > 2 else b""
        return envelope, payload

    def close(self) -> None:
        self._socket.close()
        self._ctx.term()

    async def __aenter__(self) -> AsyncConnection:
        return self

    async def __aexit__(self, *exc: Any) -> None:
        self.close()


class AsyncModel:
    def __init__(
        self,
        connection: AsyncConnection,
        model_id: str,
        latency_budget_ms: float = 50.0,
        priority: int = 1,
    ) -> None:
        self._conn = connection
        self._model_id = model_id
        self._latency_budget_ms = latency_budget_ms
        self._priority = priority

    async def observe(
        self,
        urgency: float = 0.0,
        steps_remaining: int | None = None,
        **kwargs: Any,
    ) -> None:
        envelope, payload = _build_observation(
            self._conn._client_id,
            self._conn._client_type,
            self._model_id,
            urgency,
            steps_remaining,
            **kwargs,
        )
        await self._conn._send(envelope, payload)

    async def get_result(self, timeout_ms: int = 100) -> dict | None:
        result = await self._conn._recv(timeout_ms)
        if result is None:
            return None
        return _parse_response(*result)
