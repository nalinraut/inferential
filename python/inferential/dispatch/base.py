from __future__ import annotations

import logging
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np

from inferential.dispatch.health import EndpointHealth
from inferential.proto import (
    RAW,
    Client,
    ModelResponse,
    Observation,
    Tensor,
    dtype_from_numpy,
    dtype_to_numpy,
)
from inferential.scheduler.request import InferenceRequest

logger = logging.getLogger("inferential.dispatch")


@dataclass
class DispatchResult:
    client_id: str
    response_id: str
    identity: bytes
    envelope: bytes
    payload: bytes
    latency_ms: float
    success: bool
    req_id: int = 0  # id(InferenceRequest) — used to look up the original request
    error: str | None = None


class Dispatcher(ABC):
    """Abstract base for inference dispatchers."""

    def __init__(self) -> None:
        self._health: dict[str, EndpointHealth] = {}

    @abstractmethod
    async def dispatch(self, requests: list[InferenceRequest]) -> list[DispatchResult]: ...

    def endpoint_health(self, model_id: str) -> EndpointHealth | None:
        return self._health.get(model_id)

    def _update_health(self, model_id: str, latency_ms: float, success: bool) -> None:
        if model_id not in self._health:
            self._health[model_id] = EndpointHealth(model_id=model_id)
        self._health[model_id].update(latency_ms, success)

    def _reconstruct_numpy(self, req: InferenceRequest) -> dict:
        obs = Observation()
        obs.ParseFromString(req.envelope)
        result: dict[str, Any] = {}
        for tensor in obs.tensors:
            np_dtype = dtype_to_numpy(tensor.dtype)
            shape = tuple(tensor.shape) if tensor.shape else ()
            data = req.payload[tensor.byte_offset : tensor.byte_offset + tensor.byte_length]
            array = np.frombuffer(data, dtype=np_dtype).reshape(shape)
            result[tensor.key] = array
        for k, v in obs.metadata.items():
            result[k] = v
        return result

    def _ndarray_to_tensor(self, key: str, arr: np.ndarray, offset: int) -> tuple[Tensor, bytes]:
        data = arr.tobytes()
        tensor = Tensor()
        tensor.key = key
        tensor.dtype = dtype_from_numpy(arr.dtype)
        tensor.shape.extend(arr.shape)
        tensor.byte_offset = offset
        tensor.byte_length = len(data)
        tensor.encoding = RAW
        return tensor, data

    def _build_result(
        self,
        req: InferenceRequest,
        raw_result: Any,
        latency_ms: float,
    ) -> DispatchResult:
        resp = ModelResponse()
        resp.client.CopyFrom(Client(id=req.client_id))
        resp.response_id = uuid.uuid4().hex
        resp.timestamp_ns = int(time.time() * 1_000_000_000)
        resp.inference_latency_ms = latency_ms
        resp.model_id = req.model_id

        payload_parts: list[bytes] = []
        offset = 0

        if isinstance(raw_result, dict):
            for key, value in raw_result.items():
                if isinstance(value, str):
                    resp.metadata[key] = value
                else:
                    arr = np.asarray(value)
                    tensor, data = self._ndarray_to_tensor(key, arr, offset)
                    resp.tensors.append(tensor)
                    payload_parts.append(data)
                    offset += len(data)
        else:
            arr = np.asarray(raw_result)
            tensor, data = self._ndarray_to_tensor("actions", arr, offset)
            resp.tensors.append(tensor)
            payload_parts.append(data)

        payload = b"".join(payload_parts)

        return DispatchResult(
            client_id=req.client_id,
            response_id=resp.response_id,
            identity=req.identity,
            envelope=resp.SerializeToString(),
            payload=payload,
            latency_ms=latency_ms,
            success=True,
            req_id=id(req),
        )

    def _error_result(self, req: InferenceRequest, latency_ms: float, error: str) -> DispatchResult:
        return DispatchResult(
            client_id=req.client_id,
            response_id=uuid.uuid4().hex,
            identity=req.identity,
            envelope=b"",
            payload=b"",
            latency_ms=latency_ms,
            success=False,
            req_id=id(req),
            error=error,
        )
