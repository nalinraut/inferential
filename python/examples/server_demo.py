"""Inferential server with model_deadline + pipeline dispatch.

Two mock models are served:
  - manipulation-policy  (max_inflight=3, latency ~5ms)
  - telemetry            (max_inflight=1, latency ~1ms)

Each model gets its own dispatch loop bounded by its replica count.
Priority is client-sent — 0 is highest, scores higher within the queue.

Usage:
    ray start --head
    python examples/server_demo.py
"""

from __future__ import annotations

import asyncio
import time

import numpy as np
from ray import serve

from inferential import Server
from inferential.config.schema import InferentialConfig, ModelConfig, ModelsConfig


@serve.deployment
class ManipulationPolicy:
    """Simulates a heavier manipulation policy (~5ms inference)."""

    def infer(self, obs: dict) -> dict:
        time.sleep(0.005)
        dim = 7
        for v in obs.values():
            if isinstance(v, np.ndarray) and v.ndim >= 1:
                dim = v.shape[-1]
                break
        return {"actions": np.random.randn(dim).astype(np.float32)}


@serve.deployment
class TelemetryModel:
    """Simulates a lightweight telemetry model (~1ms inference)."""

    def infer(self, obs: dict) -> dict:
        time.sleep(0.001)
        return {"status": np.array([1.0], dtype=np.float32)}


def main() -> None:
    serve.run(
        ManipulationPolicy.bind(), name="manipulation-policy", route_prefix="/manipulation-policy"
    )
    serve.run(TelemetryModel.bind(), name="telemetry", route_prefix="/telemetry")

    config = InferentialConfig(
        models=ModelsConfig(
            known={
                "manipulation-policy": ModelConfig(max_inflight=3),
                "telemetry": ModelConfig(max_inflight=1),
            },
            default_max_inflight=2,
        ),
    )
    config.scheduling.strategy = "model_deadline"
    config.scheduling.pipeline_dispatch.enabled = True

    server = Server(config=config)

    @server.on_metric
    def log_metrics(name: str, value: float, labels: dict) -> None:
        model = labels.get("model", "")
        client = labels.get("client", "")
        tag = f"[{model or client}]" if (model or client) else ""
        if name in ("e2e_latency_ms", "scheduling_wait_ms", "inference_latency_ms"):
            print(f"  {tag} {name}: {value:.1f}ms")
        elif name in ("queue_depth",):
            if value > 0:
                print(f"  {tag} {name}: {value:.0f}")

    print("Server started — manipulation-policy (max_inflight=3), telemetry (max_inflight=1)")
    asyncio.run(server.run())


if __name__ == "__main__":
    main()
