"""Inferential edge server with LocalDispatcher.

Two in-process models are registered:
  - manipulation-policy  (simulated ~5ms inference)
  - telemetry            (simulated ~1ms inference)

No Ray dependency — models are plain callables dispatched in-process.
Use the existing client_demo.py or async_client_demo.py to connect.

Usage:
    pip install inferential
    python examples/edge_server_demo.py
"""

from __future__ import annotations

import asyncio
import time

import numpy as np

from inferential import LocalDispatcher, Server
from inferential.config.schema import InferentialConfig, ModelConfig, ModelsConfig


def manipulation_policy(obs: dict) -> dict:
    """Simulates a heavier manipulation policy (~5ms inference)."""
    time.sleep(0.005)
    dim = 7
    for v in obs.values():
        if isinstance(v, np.ndarray) and v.ndim >= 1:
            dim = v.shape[-1]
            break
    return {"actions": np.random.randn(dim).astype(np.float32)}


def telemetry_model(obs: dict) -> dict:
    """Simulates a lightweight telemetry model (~1ms inference)."""
    time.sleep(0.001)
    return {"status": np.array([1.0], dtype=np.float32)}


def main() -> None:
    dispatcher = LocalDispatcher(
        {
            "manipulation-policy": manipulation_policy,
            "telemetry": telemetry_model,
        }
    )

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

    server = Server(config=config, dispatcher=dispatcher)

    @server.on_metric
    def log_metrics(name: str, value: float, labels: dict) -> None:
        model = labels.get("model", "")
        client = labels.get("client", "")
        tag = f"[{model or client}]" if (model or client) else ""
        if name in ("e2e_latency_ms", "scheduling_wait_ms", "inference_latency_ms"):
            print(f"  {tag} {name}: {value:.1f}ms")
        elif name == "queue_depth" and value > 0:
            print(f"  {tag} {name}: {value:.0f}")

    print("Edge server started — no Ray required")
    print("  manipulation-policy (max_inflight=3)")
    print("  telemetry (max_inflight=1)")
    print("  Listening on tcp://*:5555")
    asyncio.run(server.run())


if __name__ == "__main__":
    main()
