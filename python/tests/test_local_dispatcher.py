from __future__ import annotations

import asyncio
import time

import numpy as np
import pytest

from inferential.dispatch.base import Dispatcher
from inferential.dispatch.local import LocalDispatcher
from inferential.proto import FLOAT32, RAW, ModelResponse, Observation, Tensor
from inferential.scheduler.request import InferenceRequest


def _make_request(
    model_id: str = "test-model",
    client_id: str = "robot-01",
    arrays: dict[str, np.ndarray] | None = None,
    metadata: dict[str, str] | None = None,
) -> InferenceRequest:
    """Build an InferenceRequest with a valid protobuf envelope + payload."""
    if arrays is None:
        arrays = {"image": np.zeros((3, 64, 64), dtype=np.float32)}

    obs = Observation()
    obs.client.id = client_id
    obs.model_id = model_id
    obs.timestamp_ns = int(time.time() * 1_000_000_000)

    payload_parts: list[bytes] = []
    offset = 0
    for key, arr in arrays.items():
        data = arr.tobytes()
        tensor = Tensor()
        tensor.key = key
        tensor.dtype = FLOAT32
        tensor.shape.extend(arr.shape)
        tensor.byte_offset = offset
        tensor.byte_length = len(data)
        tensor.encoding = RAW
        obs.tensors.append(tensor)
        payload_parts.append(data)
        offset += len(data)

    if metadata:
        for k, v in metadata.items():
            obs.metadata[k] = v

    envelope = obs.SerializeToString()
    payload = b"".join(payload_parts)

    return InferenceRequest(
        client_id=client_id,
        model_id=model_id,
        identity=b"id-001",
        envelope=envelope,
        payload=payload,
        received_at=time.monotonic(),
        urgency=0.5,
        steps_remaining=None,
        time_until_next=0.02,
        latency_budget_ms=50.0,
        priority=0,
    )


class TestLocalDispatcherSync:
    @pytest.mark.asyncio
    async def test_dispatch_sync_callable(self):
        def echo_model(obs: dict) -> np.ndarray:
            return np.ones(7, dtype=np.float32)

        dispatcher = LocalDispatcher({"test-model": echo_model})
        results = await dispatcher.dispatch([_make_request()])

        assert len(results) == 1
        assert results[0].success is True
        assert results[0].latency_ms > 0
        assert results[0].client_id == "robot-01"

        # Verify response payload round-trips
        resp = ModelResponse()
        resp.ParseFromString(results[0].envelope)
        assert resp.model_id == "test-model"
        assert len(resp.tensors) == 1
        assert resp.tensors[0].key == "actions"

    @pytest.mark.asyncio
    async def test_dispatch_dict_return(self):
        def multi_output(obs: dict) -> dict:
            return {
                "actions": np.ones(7, dtype=np.float32),
                "status": "ok",
            }

        dispatcher = LocalDispatcher({"test-model": multi_output})
        results = await dispatcher.dispatch([_make_request()])

        assert results[0].success is True
        resp = ModelResponse()
        resp.ParseFromString(results[0].envelope)
        assert resp.metadata["status"] == "ok"
        assert len(resp.tensors) == 1
        assert resp.tensors[0].key == "actions"


class TestLocalDispatcherAsync:
    @pytest.mark.asyncio
    async def test_dispatch_async_callable(self):
        async def async_model(obs: dict) -> np.ndarray:
            await asyncio.sleep(0.001)
            return np.zeros(3, dtype=np.float32)

        dispatcher = LocalDispatcher({"test-model": async_model})
        results = await dispatcher.dispatch([_make_request()])

        assert len(results) == 1
        assert results[0].success is True


class TestLocalDispatcherErrors:
    @pytest.mark.asyncio
    async def test_unknown_model(self):
        dispatcher = LocalDispatcher({})
        results = await dispatcher.dispatch([_make_request(model_id="nonexistent")])

        assert len(results) == 1
        assert results[0].success is False
        assert "No model registered" in results[0].error

    @pytest.mark.asyncio
    async def test_callable_raises(self):
        def bad_model(obs: dict) -> np.ndarray:
            raise RuntimeError("GPU OOM")

        dispatcher = LocalDispatcher({"test-model": bad_model})
        results = await dispatcher.dispatch([_make_request()])

        assert len(results) == 1
        assert results[0].success is False
        assert "GPU OOM" in results[0].error
        assert results[0].latency_ms >= 0


class TestLocalDispatcherMultiModel:
    @pytest.mark.asyncio
    async def test_routes_to_correct_model(self):
        def policy(obs: dict) -> np.ndarray:
            return np.ones(7, dtype=np.float32)

        def telemetry(obs: dict) -> np.ndarray:
            return np.zeros(3, dtype=np.float32)

        dispatcher = LocalDispatcher(
            {
                "manipulation-policy": policy,
                "telemetry": telemetry,
            }
        )

        req_policy = _make_request(model_id="manipulation-policy")
        req_telem = _make_request(model_id="telemetry", client_id="robot-02")
        results = await dispatcher.dispatch([req_policy, req_telem])

        assert len(results) == 2
        assert all(r.success for r in results)

        # Verify correct model produced correct output
        for r in results:
            resp = ModelResponse()
            resp.ParseFromString(r.envelope)
            t0 = resp.tensors[0]
            arr = np.frombuffer(
                r.payload[t0.byte_offset : t0.byte_offset + t0.byte_length],
                dtype=np.float32,
            )
            if resp.model_id == "manipulation-policy":
                np.testing.assert_array_equal(arr, np.ones(7, dtype=np.float32))
            else:
                np.testing.assert_array_equal(arr, np.zeros(3, dtype=np.float32))


class TestLocalDispatcherHealth:
    @pytest.mark.asyncio
    async def test_health_tracking(self):
        def model(obs: dict) -> np.ndarray:
            return np.ones(1, dtype=np.float32)

        dispatcher = LocalDispatcher({"test-model": model})

        # No health before any dispatch
        assert dispatcher.endpoint_health("test-model") is None

        await dispatcher.dispatch([_make_request()])
        health = dispatcher.endpoint_health("test-model")
        assert health is not None
        assert health.total_requests == 1
        assert health.error_rate == 0.0
        assert health.avg_latency_ms > 0

    @pytest.mark.asyncio
    async def test_health_after_error(self):
        def failing(obs: dict) -> np.ndarray:
            raise ValueError("bad input")

        dispatcher = LocalDispatcher({"test-model": failing})
        await dispatcher.dispatch([_make_request()])

        health = dispatcher.endpoint_health("test-model")
        assert health is not None
        assert health.error_rate > 0


class TestLocalDispatcherRegister:
    @pytest.mark.asyncio
    async def test_register_model_after_init(self):
        dispatcher = LocalDispatcher({})

        # Fails before registration
        results = await dispatcher.dispatch([_make_request()])
        assert results[0].success is False

        # Register and retry
        dispatcher.register_model("test-model", lambda obs: np.ones(1, dtype=np.float32))
        results = await dispatcher.dispatch([_make_request()])
        assert results[0].success is True


class TestDispatcherABC:
    def test_is_abstract(self):
        assert issubclass(LocalDispatcher, Dispatcher)

    def test_cannot_instantiate_base(self):
        with pytest.raises(TypeError):
            Dispatcher()
