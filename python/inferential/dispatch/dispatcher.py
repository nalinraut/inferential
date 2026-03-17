from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections import defaultdict
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
    error: str | None = None


class RayDispatcher:
    def __init__(self, ray_config: dict | None = None) -> None:
        self._ray_config = ray_config or {}
        self._handles: dict[str, Any] = {}
        self._health: dict[str, EndpointHealth] = {}

    def _get_handle(self, model_id: str) -> Any:
        if model_id not in self._handles:
            from ray import serve

            self._handles[model_id] = serve.get_app_handle(model_id)
        return self._handles[model_id]

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
        ray_result: Any,
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

        if isinstance(ray_result, dict):
            for key, value in ray_result.items():
                if isinstance(value, str):
                    resp.metadata[key] = value
                else:
                    arr = np.asarray(value)
                    tensor, data = self._ndarray_to_tensor(key, arr, offset)
                    resp.tensors.append(tensor)
                    payload_parts.append(data)
                    offset += len(data)
        else:
            arr = np.asarray(ray_result)
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
        )

    async def dispatch(self, requests: list[InferenceRequest]) -> list[DispatchResult]:
        results: list[DispatchResult] = []
        by_model: dict[str, list[InferenceRequest]] = defaultdict(list)
        for req in requests:
            by_model[req.model_id].append(req)

        for model_id, model_requests in by_model.items():
            handle = self._get_handle(model_id)

            # Fire all remote calls concurrently so Ray Serve can
            # load-balance across replicas.
            starts: list[float] = []
            obs_dicts: list[dict] = []
            for req in model_requests:
                obs_dicts.append(self._reconstruct_numpy(req))
                starts.append(time.monotonic())

            async def _call(req: InferenceRequest, obs: dict, t0: float) -> DispatchResult:
                try:
                    result = await handle.infer.remote(obs)
                    latency = (time.monotonic() - t0) * 1000
                    self._update_health(model_id, latency, True)
                    return self._build_result(req, result, latency)
                except Exception as e:
                    latency = (time.monotonic() - t0) * 1000
                    self._update_health(model_id, latency, False)
                    logger.error("Dispatch failed for client %s: %s", req.client_id, e)
                    return DispatchResult(
                        client_id=req.client_id,
                        response_id=uuid.uuid4().hex,
                        identity=req.identity,
                        envelope=b"",
                        payload=b"",
                        latency_ms=latency,
                        success=False,
                        error=str(e),
                    )

            batch_results = await asyncio.gather(
                *(_call(req, obs, t0) for req, obs, t0 in zip(model_requests, obs_dicts, starts))
            )
            results.extend(batch_results)
        return results

    def _update_health(self, model_id: str, latency_ms: float, success: bool) -> None:
        if model_id not in self._health:
            self._health[model_id] = EndpointHealth(model_id=model_id)
        self._health[model_id].update(latency_ms, success)

    def endpoint_health(self, model_id: str) -> EndpointHealth | None:
        return self._health.get(model_id)

    def get_handle(self, model_id: str) -> Any | None:
        return self._handles.get(model_id)
