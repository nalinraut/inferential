from __future__ import annotations

import inspect
import logging
import time
from collections import defaultdict
from typing import Any, Callable

from inferential.dispatch.base import Dispatcher, DispatchResult
from inferential.scheduler.request import InferenceRequest

logger = logging.getLogger("inferential.dispatch")


class LocalDispatcher(Dispatcher):
    """Dispatches inference to in-process callables.

    Each model is a callable that takes an observation dict and returns
    a result (ndarray, dict of ndarrays/strings, or any array-like).
    Callables can be sync or async.

    Example::

        dispatcher = LocalDispatcher({
            "manipulation-policy": trt_model.infer,
            "telemetry": onnx_session.run,
        })
    """

    def __init__(self, models: dict[str, Callable[..., Any]]) -> None:
        super().__init__()
        self._models = dict(models)

    def register_model(self, model_id: str, fn: Callable[..., Any]) -> None:
        self._models[model_id] = fn

    async def dispatch(self, requests: list[InferenceRequest]) -> list[DispatchResult]:
        results: list[DispatchResult] = []
        by_model: dict[str, list[InferenceRequest]] = defaultdict(list)
        for req in requests:
            by_model[req.model_id].append(req)

        for model_id, model_requests in by_model.items():
            model_fn = self._models.get(model_id)
            if model_fn is None:
                for req in model_requests:
                    results.append(
                        self._error_result(req, 0.0, f"No model registered for '{model_id}'")
                    )
                continue

            for req in model_requests:
                obs = self._reconstruct_numpy(req)
                t0 = time.monotonic()
                try:
                    result = model_fn(obs)
                    if inspect.isawaitable(result):
                        result = await result
                    latency = (time.monotonic() - t0) * 1000
                    self._update_health(model_id, latency, True)
                    results.append(self._build_result(req, result, latency))
                except Exception as e:
                    latency = (time.monotonic() - t0) * 1000
                    self._update_health(model_id, latency, False)
                    logger.error("Local dispatch failed for client %s: %s", req.client_id, e)
                    results.append(self._error_result(req, latency, str(e)))

        return results
