# Quick Start

Get a server and client running locally in under 5 minutes.

## Prerequisites

- Python 3.11+
- A running Ray cluster (single-node is fine)

## 1. Install

```bash
pip install inferential[server]
```

This pulls in Ray Serve, pyzmq, protobuf, numpy, and pydantic.

## 2. Start Ray

```bash
ray start --head
```

## 3. Create a Server

Save as `server.py`:

```python
import asyncio
import numpy as np
from ray import serve
from inferential import Server

@serve.deployment
class MockPolicy:
    def infer(self, obs: dict) -> dict:
        dim = 7
        for v in obs.values():
            if isinstance(v, np.ndarray) and v.ndim == 1:
                dim = v.shape[0]
                break
        return {"actions": np.random.randn(dim).astype(np.float32)}

serve.run(MockPolicy.bind(), name="policy-v2")

server = Server(bind="tcp://*:5555", models=["policy-v2"])

@server.on_metric
def log(name, value, labels):
    if name == "inference_latency_ms":
        client = labels.get("client", "?")
        print(f"  [{client}] {value:.1f}ms")

asyncio.run(server.run())
```

Run it:

```bash
python server.py
```

You should see:

```
Application 'policy-v2' is ready at http://127.0.0.1:8000/.
```

## 4. Connect a Client

In a second terminal, save as `client.py`:

```python
import time
import numpy as np
from inferential import Connection

conn = Connection(server="tcp://localhost:5555", client_id="demo-01", client_type="sim")
# priority=0 is highest; lower-priority clients yield GPU slots under contention
model = conn.model("policy-v2", latency_budget_ms=50.0, priority=0)

for step in range(10):
    state = np.random.randn(7).astype(np.float32)
    model.observe(urgency=0.5, state=state)

    result = model.get_result(timeout_ms=100)
    if result is not None:
        actions = result["actions"]
        latency = result["inference_latency_ms"]
        print(f"step {step}: actions={actions[:3]}... latency={latency:.1f}ms")

    time.sleep(0.05)

conn.close()
```

Run it:

```bash
python client.py
```

Expected output:

```
step 0: actions=[ 0.42 -1.07  0.83]... latency=11.2ms
step 1: actions=[-0.31  0.55  1.22]... latency=1.4ms
step 2: actions=[ 0.91 -0.18  0.03]... latency=1.2ms
...
```

The first request is slower (Ray Serve cold start). Subsequent requests settle around 1-2ms for this mock model.

On the server terminal you'll see metric callbacks firing:

```
  [demo-01] 11.2ms
  [demo-01] 1.4ms
  [demo-01] 1.2ms
```

## 5. Async Client (optional)

The SDK also provides `AsyncConnection` for `asyncio`-based control loops. Save as `async_client.py`:

```python
import asyncio
import numpy as np
from inferential import AsyncConnection

async def main():
    async with AsyncConnection(
        server="tcp://localhost:5555", client_id="async-01", client_type="sim"
    ) as conn:
        model = conn.model("policy-v2", latency_budget_ms=50.0, priority=0)

        for step in range(10):
            state = np.random.randn(7).astype(np.float32)
            await model.observe(urgency=0.5, state=state)

            result = await model.get_result(timeout_ms=100)
            if result is not None:
                actions = result["actions"]
                latency = result["inference_latency_ms"]
                print(f"step {step}: actions={actions[:3]}... latency={latency:.1f}ms")

            await asyncio.sleep(0.05)

asyncio.run(main())
```

Run it:

```bash
python async_client.py
```

The API mirrors the sync client — `observe()` and `get_result()` are `await`-ed instead of blocking. `AsyncConnection` supports `async with` for automatic cleanup.

## 6. Teardown

```bash
# Stop the server with Ctrl+C
ray stop
```

## Next Steps

### Multi-model with priority scheduling

For production use with multiple models (e.g. a manipulation policy and a telemetry stream), use `model_deadline` + per-model concurrency limits:

```python
from inferential.config.schema import InferentialConfig, ModelConfig, ModelsConfig

config = InferentialConfig(
    models=ModelsConfig(
        known={
            "manipulation-policy": ModelConfig(max_inflight=4),  # 4 GPU replicas
            "telemetry":           ModelConfig(max_inflight=1),
        },
    ),
)
config.transport.bind = "tcp://*:5555"
config.scheduling.strategy = "model_deadline"
config.scheduling.pipeline_dispatch.enabled = True

server = Server(config=config, models=["manipulation-policy", "telemetry"])
```

`max_inflight` should match `num_replicas` on your Ray Serve deployment. This keeps one request queued per GPU at the scheduler level — so priority ordering is always respected before a request reaches Ray Serve.

- [Architecture](../../docs/architecture.md) — system design, wire protocol, schedulers, configuration reference
- [Examples](../../docs/examples.md) — multi-language client demos, server extensions, custom schedulers
- [C++ SDK](../../cpp/) · [Rust SDK](../../rust/) — other language clients
