from __future__ import annotations

import asyncio
import logging
import time
from collections import defaultdict
from typing import Any

from inferential.dispatch.base import Dispatcher, DispatchResult
from inferential.scheduler.request import InferenceRequest

logger = logging.getLogger("inferential.dispatch")

# Re-export for backward compatibility
__all__ = ["DispatchResult", "RayDispatcher"]


class RayDispatcher(Dispatcher):
    def __init__(self, ray_config: dict | None = None) -> None:
        super().__init__()
        self._ray_config = ray_config or {}
        self._handles: dict[str, Any] = {}

    def _get_handle(self, model_id: str) -> Any:
        if model_id not in self._handles:
            from ray import serve

            self._handles[model_id] = serve.get_app_handle(model_id)
        return self._handles[model_id]

    async def dispatch(self, requests: list[InferenceRequest]) -> list[DispatchResult]:
        results: list[DispatchResult] = []
        by_model: dict[str, list[InferenceRequest]] = defaultdict(list)
        for req in requests:
            by_model[req.model_id].append(req)

        for model_id, model_requests in by_model.items():
            handle = self._get_handle(model_id)

            # Reconstruct observations up front, then fire all remote
            # calls concurrently so Ray Serve can load-balance across replicas.
            obs_dicts = [self._reconstruct_numpy(req) for req in model_requests]

            async def _call(req: InferenceRequest, obs: dict) -> DispatchResult:
                t0 = time.monotonic()
                try:
                    result = await handle.infer.remote(obs)
                    latency = (time.monotonic() - t0) * 1000
                    self._update_health(model_id, latency, True)
                    return self._build_result(req, result, latency)
                except Exception as e:
                    latency = (time.monotonic() - t0) * 1000
                    self._update_health(model_id, latency, False)
                    logger.error("Dispatch failed for client %s: %s", req.client_id, e)
                    return self._error_result(req, latency, str(e))

            batch_results = await asyncio.gather(
                *(_call(req, obs) for req, obs in zip(model_requests, obs_dicts))
            )
            results.extend(batch_results)
        return results

    def get_handle(self, model_id: str) -> Any | None:
        return self._handles.get(model_id)
