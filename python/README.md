# Inferential Python SDK

Python client and server SDK for [Inferential](../README.md) inference orchestration. The Python package includes both the **client SDK** (for sending observations and receiving results) and the **server** (Ray Serve-based scheduling and dispatch).

## Install

```bash
# Client SDK only (pyzmq, protobuf, numpy)
pip install inferential

# Server with Ray Serve
pip install inferential[server]

# Development
pip install inferential[dev]
```

## Quick Start

See the full [Quick Start guide](docs/quickstart.md) for step-by-step setup.

### Server

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
        print(f"Client {labels.get('client')}: {value:.1f}ms")

asyncio.run(server.run())
```

### Client (sync)

```python
import numpy as np
from inferential import Connection

conn = Connection(server="tcp://localhost:5555", client_id="agent-01", client_type="franka")
model = conn.model("policy-v2", latency_budget_ms=30.0)

state = np.random.randn(7).astype(np.float32)
model.observe(urgency=0.8, state=state)

result = model.get_result(timeout_ms=50)
if result is not None:
    actions = result["actions"]  # np.ndarray

conn.close()
```

### Client (async)

```python
import asyncio
import numpy as np
from inferential import AsyncConnection

async def main():
    async with AsyncConnection(server="tcp://localhost:5555", client_id="agent-01") as conn:
        model = conn.model("policy-v2", latency_budget_ms=30.0)

        state = np.random.randn(7).astype(np.float32)
        await model.observe(urgency=0.8, state=state)

        result = await model.get_result(timeout_ms=50)
        if result is not None:
            actions = result["actions"]  # np.ndarray

asyncio.run(main())
```

## API Reference

### `Connection(server, client_id, client_type, reconnect_ivl_ms=100, reconnect_max_ms=5000)`

Creates a ZMQ DEALER connection to the server. The `server` address can be with or without the `tcp://` prefix.

### `AsyncConnection(server, client_id, client_type, ...)`

Async variant using `zmq.asyncio.Context`. Supports `async with` for automatic cleanup.

### `conn.model(model_id, latency_budget_ms=50.0, priority=1) → Model / AsyncModel`

Creates a handle to a specific model on the server.

### `model.observe(urgency=0.0, steps_remaining=None, **kwargs)`

Sends an observation to the server. Keyword arguments are automatically dispatched:

- **`np.ndarray`** values → serialized as tensors (dtype/shape preserved)
- **`str`** values → passed as metadata key-value pairs
- **`urgency`** (float, 0.0–1.0) → scheduling priority hint
- **`steps_remaining`** (int) → remaining steps in trajectory

```python
model.observe(
    urgency=0.5,
    steps_remaining=120,
    state_vector=np.zeros(7, dtype=np.float32),
    image=np.zeros((3, 224, 224), dtype=np.uint8),
    prompt="describe the scene",  # → metadata
)
```

### `model.get_result(timeout_ms=100) → dict | None`

Waits for a response. Returns a dict mapping tensor keys to numpy arrays, or `None` on timeout. Also includes `response_id`, `model_id`, `inference_latency_ms`, and any metadata from the server.

### `conn.close()`

Closes the ZMQ socket. Called automatically by `AsyncConnection.__aexit__`.

## Server Configuration

See [Architecture](../docs/architecture.md) for full details on schedulers, queue management, metrics, and configuration schema.

## Documentation

- [Quick Start](docs/quickstart.md) — Install, run server + client, get your first result
- [Architecture](../docs/architecture.md) — System design, wire protocol, schedulers, metrics
- [Examples](../docs/examples.md) — Multi-language client demos, server extensions
- [Contributing](../docs/contributing.md) — Commit conventions, branching, code style
