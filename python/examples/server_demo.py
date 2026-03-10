"""Inferential server with a mock Ray Serve model.

Usage:
    ray start --head
    python examples/server_demo.py
"""

from __future__ import annotations

import asyncio

import numpy as np
from ray import serve

from inferential import Server


@serve.deployment
class MockPolicy:
    """Returns random actions for any observation.

    The infer() method receives a dict of numpy arrays (deserialized from
    the wire format) and returns a dict. Return values can be any array-like
    (numpy, torch, jax, etc.) — the dispatcher converts via np.asarray().
    """

    def infer(self, obs: dict) -> dict:
        dim = 7
        for v in obs.values():
            if isinstance(v, np.ndarray) and v.ndim == 1:
                dim = v.shape[0]
                break
        return {"actions": np.random.randn(dim).astype(np.float32)}


def main() -> None:
    serve.run(MockPolicy.bind(), name="policy-v2")

    server = Server(
        bind="tcp://*:5555",
        models=["policy-v2"],
    )
    server.use_scheduler("deadline_aware")

    @server.on_metric
    def log_metrics(name: str, value: float, labels: dict) -> None:
        client = labels.get("client", "")
        prefix = f"  [{client}]" if client else " "
        if name in ("inference_latency_ms", "observation_staleness_ms",
                     "scheduling_wait_ms", "e2e_latency_ms"):
            print(f"{prefix} {name}: {value:.1f}ms")
        elif name in ("queue_full_drops", "dispatch_errors", "dispatch_retries",
                       "requests_expired", "client_disconnected", "observation_errors"):
            print(f"{prefix} {name}: {value:.0f}")

    asyncio.run(server.run())


if __name__ == "__main__":
    main()
