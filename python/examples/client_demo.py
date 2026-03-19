"""Simulated robot cell sending observations to an Inferential server.

Demonstrates per-model priority:
  - manipulation-policy  priority=0 (high — drives real-time control)
  - telemetry            priority=1 (lower — background monitoring)

Usage:
    # Single robot
    python examples/client_demo.py

    # Multiple robots in parallel
    python examples/client_demo.py --robots 4

    # Custom settings
    python examples/client_demo.py --robots 2 --hz 50 --steps 200
"""

from __future__ import annotations

import argparse
import threading
import time

import numpy as np

from inferential import Connection


def run_robot(
    robot_id: str,
    server: str,
    hz: float,
    steps: int,
) -> None:
    conn = Connection(server=server, client_id=robot_id, client_type="franka")

    # Two models with different priorities — priority travels in every observation
    policy = conn.model("manipulation-policy", latency_budget_ms=1000 / hz, priority=0)
    telemetry = conn.model("telemetry", latency_budget_ms=200.0, priority=1)

    interval = 1.0 / hz
    policy_received = 0
    telemetry_received = 0

    print(f"[{robot_id}] starting — {hz}Hz, {steps} steps")

    for step in range(steps):
        state = np.random.randn(7).astype(np.float32)

        # High-priority control observation (priority=0 from model handle)
        policy.observe(
            urgency=min(1.0, (steps - step) / steps),  # urgency rises as episode ends
            steps_remaining=steps - step,
            state=state,
        )

        # Low-priority telemetry every 5 steps (priority=1 from model handle)
        if step % 5 == 0:
            telemetry.observe(
                urgency=0.1,
                joint_positions=state,
            )

        result = policy.get_result(timeout_ms=int(interval * 1000))
        if result is not None:
            policy_received += 1
            if step % 20 == 0:
                actions = result["actions"]
                print(f"[{robot_id}] step {step}: actions={actions[:3]}...")

        # Drain any pending telemetry responses non-blocking
        tel_result = telemetry.get_result(timeout_ms=0)
        if tel_result is not None:
            telemetry_received += 1

        time.sleep(interval)

    conn.close()
    print(
        f"[{robot_id}] done — policy {policy_received}/{steps}, "
        f"telemetry {telemetry_received}/{steps // 5}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Inferential robot client demo")
    parser.add_argument("--server", default="tcp://localhost:5555")
    parser.add_argument("--robots", type=int, default=1, help="Number of robots")
    parser.add_argument("--hz", type=float, default=20.0, help="Control frequency per robot")
    parser.add_argument("--steps", type=int, default=100, help="Steps per robot")
    args = parser.parse_args()

    if args.robots == 1:
        run_robot("robot-01", args.server, args.hz, args.steps)
    else:
        threads = []
        for i in range(args.robots):
            rid = f"robot-{i + 1:02d}"
            t = threading.Thread(target=run_robot, args=(rid, args.server, args.hz, args.steps))
            t.start()
            threads.append(t)
        for t in threads:
            t.join()


if __name__ == "__main__":
    main()
