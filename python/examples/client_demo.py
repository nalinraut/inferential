"""Simulated client sending observations to an Inferential server.

Usage:
    # Single client
    python examples/client_demo.py

    # Multiple clients in parallel
    python examples/client_demo.py --clients 4

    # Custom settings
    python examples/client_demo.py --clients 2 --hz 50 --steps 200
"""

from __future__ import annotations

import argparse
import threading
import time

import numpy as np

from inferential import Connection


def run_client(
    client_id: str,
    server: str,
    hz: float,
    steps: int,
) -> None:
    conn = Connection(server=server, client_id=client_id, client_type="sim")
    model = conn.model("policy-v2", latency_budget_ms=1000 / hz)
    interval = 1.0 / hz
    received = 0

    print(f"[{client_id}] starting — {hz}Hz, {steps} steps")

    for step in range(steps):
        state = np.random.randn(7).astype(np.float32)
        model.observe(
            urgency=0.5,
            steps_remaining=steps - step,
            state=state,
        )

        result = model.get_result(timeout_ms=int(interval * 1000))
        if result is not None:
            received += 1
            if step % 20 == 0:
                actions = result["actions"]
                latency = result["inference_latency_ms"]
                print(
                    f"[{client_id}] step {step}: "
                    f"actions={actions[:3]}... latency={latency:.1f}ms"
                )

        time.sleep(interval)

    conn.close()
    print(f"[{client_id}] done — received {received}/{steps} responses")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inferential client demo")
    parser.add_argument("--server", default="tcp://localhost:5555")
    parser.add_argument("--clients", type=int, default=1, help="Number of clients")
    parser.add_argument("--hz", type=float, default=20.0, help="Request frequency per client")
    parser.add_argument("--steps", type=int, default=100, help="Steps per client")
    args = parser.parse_args()

    if args.clients == 1:
        run_client("sim-01", args.server, args.hz, args.steps)
    else:
        threads = []
        for i in range(args.clients):
            cid = f"sim-{i + 1:02d}"
            t = threading.Thread(target=run_client, args=(cid, args.server, args.hz, args.steps))
            t.start()
            threads.append(t)
        for t in threads:
            t.join()


if __name__ == "__main__":
    main()
