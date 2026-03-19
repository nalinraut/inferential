"""Async robot fleet demo — multiple robots on a single connection.

Demonstrates per-model priority with asyncio:
  - manipulation-policy  priority=0 (high — real-time control)
  - telemetry            priority=1 (lower — background monitoring)

All robots run concurrently on a single event loop (no threads).

Usage:
    python examples/async_client_demo.py
    python examples/async_client_demo.py --robots 6 --hz 50 --steps 200
"""

from __future__ import annotations

import argparse
import asyncio

import numpy as np

from inferential import AsyncConnection


async def robot_loop(
    conn: AsyncConnection,
    robot_id: str,
    hz: float,
    steps: int,
) -> None:
    # Two models with different priorities
    policy = conn.model("manipulation-policy", latency_budget_ms=1000 / hz, priority=0)
    telemetry = conn.model("telemetry", latency_budget_ms=200.0, priority=1)

    interval = 1.0 / hz
    policy_received = 0

    print(f"[{robot_id}] starting — {hz}Hz, {steps} steps")

    for step in range(steps):
        state = np.random.randn(7).astype(np.float32)

        # High-priority control observation
        await policy.observe(
            urgency=min(1.0, (steps - step) / steps),
            steps_remaining=steps - step,
            state=state,
        )

        # Low-priority telemetry every 5 steps
        if step % 5 == 0:
            await telemetry.observe(
                urgency=0.1,
                joint_positions=state,
            )

        result = await policy.get_result(timeout_ms=int(interval * 1000))
        if result is not None:
            policy_received += 1
            if step % 20 == 0:
                actions = result["actions"]
                print(f"[{robot_id}] step {step}: actions={actions[:3]}...")

        await asyncio.sleep(interval)

    print(f"[{robot_id}] done — policy responses: {policy_received}/{steps}")


async def main() -> None:
    parser = argparse.ArgumentParser(description="Inferential async robot fleet demo")
    parser.add_argument("--server", default="tcp://localhost:5555")
    parser.add_argument("--robots", type=int, default=3, help="Number of robots")
    parser.add_argument("--hz", type=float, default=20.0, help="Control frequency per robot")
    parser.add_argument("--steps", type=int, default=100, help="Steps per robot")
    args = parser.parse_args()

    async with AsyncConnection(
        server=args.server, client_id="fleet-controller", client_type="franka"
    ) as conn:
        await asyncio.gather(
            *(
                robot_loop(conn, f"robot-{i + 1:02d}", args.hz, args.steps)
                for i in range(args.robots)
            )
        )


if __name__ == "__main__":
    asyncio.run(main())
