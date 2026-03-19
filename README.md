# Inferential

[![CI](https://github.com/nalinraut/inferential/actions/workflows/ci.yml/badge.svg)](https://github.com/nalinraut/inferential/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/inferential)](https://pypi.org/project/inferential/)
[![crates.io](https://img.shields.io/crates/v/inferential)](https://crates.io/crates/inferential)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)](https://pypi.org/project/inferential/)
[![Rust](https://img.shields.io/badge/rust-1.70%2B-orange)](https://crates.io/crates/inferential)
[![C++17](https://img.shields.io/badge/C%2B%2B-17-blue)](cpp/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

Multi-client inference orchestration on top of Ray Serve.

Inferential sits between your clients and your ML models. It receives observations over ZMQ, schedules inference requests using cadence-aware priority scoring, dispatches to Ray Serve, and streams results back — all with sub-millisecond transport overhead. Built for any scenario where multiple clients need concurrent access to shared models: robotics fleets, game agents, IoT devices, real-time ML pipelines.

![Inferential data flow](https://raw.githubusercontent.com/nalinraut/inferential/main/assets/InferentialFlow.gif)

## Features

- **ZMQ transport** — ROUTER/DEALER sockets with automatic reconnection and zero-copy tensor payloads
- **Multi-language SDKs** — Python, C++, and Rust clients sharing the same protobuf wire protocol
- **Pluggable schedulers** — Deadline-aware (default), model-deadline (per-model queues), batch-optimized, priority-tiered, round-robin
- **Cadence learning** — EMA-based tracking of each client's request pattern to predict urgency
- **Protobuf wire protocol** — Typed tensor metadata (dtype, shape, encoding) with binary payload
- **Queue management** — Request TTL, drop-oldest overflow policy, dispatch retry
- **In-memory metrics** — Ring-buffer storage with label filtering and percentile stats (p50/p95/p99)
- **Async support** — Python (`AsyncConnection`) and Rust (`AsyncConnection`) async clients

## Metrics

Every request generates metrics across the pipeline, stored in a ring buffer (10,000 points per metric) with p50/p95/p99 percentiles and per-client label filtering.

| Metric | Labels | What it captures |
|--------|--------|-----------------|
| `inference_latency_ms` | `client`, `model` | Pure model execution time (Ray Serve) |
| `scheduling_wait_ms` | `client`, `model` | Time spent in the scheduler queue |
| `e2e_latency_ms` | `client`, `model` | Total server-side delay (queue + inference) |
| `observation_staleness_ms` | `client` | Age of sensor data on arrival |
| `queue_depth` | `model` | Pending requests at dispatch time (per-model with pipeline dispatch) |
| `queue_full_drops` | `client` | Requests dropped due to queue overflow |

All metrics support label-based filtering. With `model_deadline` + pipeline dispatch, metrics include a `model` label for per-queue observability:

```python
@server.on_metric
def handle(name, value, labels):
    model = labels.get("model", "all")
    if name == "queue_depth":
        print(f"{model} depth: {value}")

# Per-model stats
stats = server.metrics.get_stats("e2e_latency_ms", labels={"model": "manipulation-policy"})
```

## Architecture

```
                        Inferential Server
                  ┌──────────────────────────────┐
                  │                              │
  Client A ──ZMQ──┤  Assembler ──► Scheduler     │
  Client B ──ZMQ──┤           ┌────┴────┐       │
  Client C ──ZMQ──┤       "policy"  "telemetry"  │
                  │           │          │       │
                  │       Dispatch   Dispatch ───┼──► Ray Serve
                  │        (sem 3)    (sem 1)    │    (per-model replicas)
                  │           └────┬────┘        │
  Client A ◄─ZMQ─┤  Transport ◄───┘             │
  Client B ◄─ZMQ─┤                              │
  Client C ◄─ZMQ─┤  Metrics   Cadence   Health  │
                  └──────────────────────────────┘

  Clients can be Python, C++, or Rust — any language that speaks
  the ZMQ + protobuf wire protocol.
```

## Install

```bash
# Python
pip install inferential

# Rust
cargo add inferential

# C++ (Bazel — add to MODULE.bazel)
bazel_dep(name = "inferential", version = "1.2.1")
```

## Client SDKs

All SDKs implement the same API pattern: **connect → model → observe → get_result**.

| Language | Sync | Async | Package |
|----------|------|-------|---------|
| [Python](python/) | `Connection` / `Model` | `AsyncConnection` / `AsyncModel` | [PyPI](https://pypi.org/project/inferential/) |
| [C++](cpp/) | `Connection` / `Model` | — | [Bazel Central Registry](https://registry.bazel.build/modules/inferential) |
| [Rust](rust/) | `Connection` / `Model` | `AsyncConnection` / `AsyncModel` | [crates.io](https://crates.io/crates/inferential) |

### Quick Example

<details>
<summary><b>Python</b></summary>

```python
import numpy as np
from inferential import Connection

conn = Connection(server="tcp://localhost:5555", client_id="agent-01", client_type="franka")
# priority=0 is highest; yields GPU slots to lower-priority peers under contention
model = conn.model("manipulation-policy", latency_budget_ms=30.0, priority=0)

state = np.random.randn(7).astype(np.float32)
model.observe(urgency=0.8, steps_remaining=50, state=state)

result = model.get_result(timeout_ms=50)
if result is not None:
    actions = result["actions"]  # np.ndarray
```
</details>

<details>
<summary><b>C++</b></summary>

```cpp
#include "inferential/client.hpp"

auto conn = inferential::Connection("tcp://localhost:5555", "agent-01", "franka");
auto model = conn.model("policy-v2", 30.0f, 1);

std::vector<float> state = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f};
model.observe()
    .urgency(0.8f)
    .tensor_f32("state", state.data(), state.size(), {7})
    .send();

auto result = model.get_result(50);
if (result) {
    auto [ptr, count] = (*result)["actions"].as<float>();
}
```
</details>

<details>
<summary><b>Rust</b></summary>

```rust
use inferential::Connection;

let conn = Connection::new("tcp://localhost:5555", "agent-01", "franka");
let model = conn.model("policy-v2", 30.0, 1);

let state: Vec<f32> = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7];
model.observe()
    .urgency(0.8)
    .tensor_f32("state", &state, &[7])
    .send();

if let Some(result) = model.get_result(50) {
    let actions = result["actions"].as_f32();
}
```
</details>

## Server

The server runs on Python with Ray Serve. See the [Python SDK](python/) for server setup.

```python
import asyncio
from inferential import Server
from inferential.config.schema import InferentialConfig, ModelConfig, ModelsConfig

config = InferentialConfig(
    models=ModelsConfig(
        known={
            "manipulation-policy": ModelConfig(max_inflight=4),  # match GPU replica count
            "telemetry":           ModelConfig(max_inflight=1),
        },
    ),
)
config.transport.bind = "tcp://*:5555"
config.scheduling.strategy = "model_deadline"
config.scheduling.pipeline_dispatch.enabled = True

server = Server(config=config, models=["manipulation-policy", "telemetry"])

@server.on_metric
def log(name, value, labels):
    if name == "e2e_latency_ms":
        print(f"[{labels.get('model')}/{labels.get('client')}] {value:.1f}ms")

asyncio.run(server.run())
```

## Documentation

- **[Architecture](docs/architecture.md)** — System design, wire protocol, schedulers, queue management, metrics, configuration
- **[Contributing](docs/contributing.md)** — Commit conventions, branching, code style, development setup
- **[Quick Start (Python)](python/docs/quickstart.md)** — Install, run server + client, get your first result
- **SDK Guides**: [Python](python/) · [C++](cpp/) · [Rust](rust/)

## Development

```bash
# Python
make proto          # Generate protobuf code
make test           # Run Python tests
make lint           # Lint Python code

# C++
make build-cpp      # Build C++ SDK (Bazel)
make test-cpp       # Run C++ tests

# Rust
make build-rust     # Build Rust SDK
make test-rust      # Run Rust tests

# All languages
make test-all       # Run all tests
make clean          # Clean all build artifacts
```

## License

MIT
