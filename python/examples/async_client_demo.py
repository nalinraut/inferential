"""Async client demo — multiple robots sharing a single connection.

Unlike the sync client_demo.py which uses threads, this runs all robots
concurrently on a single thread using asyncio.

Usage:
    python examples/async_client_demo.py
    python examples/async_client_demo.py --clients 6 --hz 50 --steps 200
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
    model = conn.model("policy-v2", latency_budget_ms=1000 / hz)
    interval = 1.0 / hz
    received = 0

    print(f"[{robot_id}] starting — {hz}Hz, {steps} steps")

    for step in range(steps):
        state = np.random.randn(7).astype(np.float32)
        await model.observe(
            urgency=0.5,
            steps_remaining=steps - step,
            state=state,
        )

        result = await model.get_result(timeout_ms=int(interval * 1000))
        if result is not None:
            received += 1
            if step % 20 == 0:
                actions = result["actions"]
                latency = result["inference_latency_ms"]
                print(
                    f"[{robot_id}] step {step}: "
                    f"actions={actions[:3]}... latency={latency:.1f}ms"
                )

        await asyncio.sleep(interval)

    print(f"[{robot_id}] done — received {received}/{steps} responses")


async def main() -> None:
    parser = argparse.ArgumentParser(description="Inferential async client demo")
    parser.add_argument("--server", default="tcp://localhost:5555")
    parser.add_argument("--clients", type=int, default=3, help="Number of robots")
    parser.add_argument("--hz", type=float, default=20.0, help="Request frequency per robot")
    parser.add_argument("--steps", type=int, default=100, help="Steps per robot")
    args = parser.parse_args()

    async with AsyncConnection(
        server=args.server, client_id="async-cell", client_type="sim"
    ) as conn:
        await asyncio.gather(
            *(
                robot_loop(conn, f"bot-{i + 1:02d}", args.hz, args.steps)
                for i in range(args.clients)
            )
        )


if __name__ == "__main__":
    asyncio.run(main())
