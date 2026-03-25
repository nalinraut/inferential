# Examples

Each SDK includes ready-to-run example scripts. This guide walks through the patterns.

## Server Setup

All examples connect to an Inferential server. Start one first:

```bash
cd python
ray start --head
python examples/server_demo.py
```

Wait for both applications to be ready. The server runs two mock models: `manipulation-policy` (heavier, higher priority) and `telemetry` (lightweight).

See the [Python quickstart](../python/docs/quickstart.md) for full server setup details.

## Client Examples

### Python

**Files**: `python/examples/client_demo.py`, `python/examples/async_client_demo.py`

```bash
# Sync — 4 clients, threaded
python python/examples/client_demo.py --clients 4 --hz 20 --steps 50

# Async — 3 robots on single thread
python python/examples/async_client_demo.py --clients 3 --hz 20 --steps 50
```

The sync demo spawns one thread per client. The async demo runs all robots concurrently on a single thread using `asyncio.gather()`.

#### Sync Client

```python
from inferential import Connection
import numpy as np

conn = Connection(server="tcp://localhost:5555", client_id="arm-01", client_type="franka")

# priority=0 → highest priority; scheduler scores this client above lower-priority peers
policy  = conn.model("manipulation-policy", latency_budget_ms=30.0, priority=0)
telemetry = conn.model("telemetry", latency_budget_ms=100.0, priority=1)

steps = 50
for step in range(steps):
    state = np.random.randn(14).astype(np.float32)
    urgency = min(1.0, (steps - step) / steps)
    policy.observe(urgency=urgency, steps_remaining=steps - step, state=state)

    result = policy.get_result(timeout_ms=100)
    if result is not None:
        actions = result["actions"]
        print(f"step {step}: actions={actions[:3]}...")

    if step % 5 == 0:
        telemetry.observe(urgency=0.1, state=state)

conn.close()
```

#### Async Client

```python
from inferential import AsyncConnection
import numpy as np, asyncio

async def robot_loop(conn, client_id, model_name, priority, steps):
    model = conn.model(model_name, latency_budget_ms=30.0, priority=priority)
    for step in range(steps):
        state = np.random.randn(14).astype(np.float32)
        urgency = min(1.0, (steps - step) / steps)
        await model.observe(urgency=urgency, steps_remaining=steps - step, state=state)
        result = await model.get_result(timeout_ms=100)
        await asyncio.sleep(0.02)

async def main():
    async with AsyncConnection(server="tcp://localhost:5555", client_id="cell") as conn:
        await asyncio.gather(
            robot_loop(conn, "arm-01", "manipulation-policy", 0, 100),
            robot_loop(conn, "arm-02", "manipulation-policy", 0, 100),
            robot_loop(conn, "arm-03", "telemetry",           1, 100),
        )

asyncio.run(main())
```

### C++

**File**: `cpp/examples/client_demo.cpp`

```bash
bazel build //cpp/examples:client_demo
bazel-bin/cpp/examples/client_demo
```

```cpp
#include "inferential/client.hpp"
#include <iostream>
#include <vector>

int main() {
    auto conn = inferential::Connection("tcp://localhost:5555", "arm-01", "franka");
    // priority=0 → highest; matches P0 manipulation robots in the demo
    auto policy   = conn.model("manipulation-policy", 30.0f, 0);
    auto telemetry = conn.model("telemetry", 100.0f, 1);

    int steps = 100;
    for (int step = 0; step < steps; ++step) {
        std::vector<float> state(14, static_cast<float>(step) * 0.1f);
        float urgency = std::min(1.0f, static_cast<float>(steps - step) / steps);
        policy.observe()
            .urgency(urgency)
            .steps_remaining(steps - step)
            .tensor_f32("state", state.data(), state.size(), {14})
            .send();

        auto result = policy.get_result(100);
        if (result) {
            auto [ptr, count] = (*result)["actions"].as<float>();
            std::cout << "step " << step << ": actions=[";
            for (size_t i = 0; i < std::min(count, size_t(3)); ++i)
                std::cout << ptr[i] << " ";
            std::cout << "...]" << std::endl;
        }

        if (step % 5 == 0) {
            telemetry.observe().urgency(0.1f)
                .tensor_f32("state", state.data(), state.size(), {14}).send();
        }
    }
}
```

### Rust

**Files**: `rust/examples/client_demo.rs`, `rust/examples/async_client_demo.rs`

```bash
cd rust
cargo run --example client_demo
cargo run --example async_client_demo
```

#### Sync

```rust
use inferential::Connection;

fn main() {
    let conn = Connection::new("tcp://localhost:5555", "arm-01", "franka");
    // priority=0 → highest; matches P0 manipulation robots in the demo
    let policy    = conn.model("manipulation-policy", 30.0, 0);
    let telemetry = conn.model("telemetry", 100.0, 1);

    let steps = 100;
    for step in 0..steps {
        let state: Vec<f32> = (0..14).map(|i| i as f32 * 0.1).collect();
        let urgency = (1.0f32).min((steps - step) as f32 / steps as f32);
        policy.observe()
            .urgency(urgency)
            .steps_remaining(steps - step)
            .tensor_f32("state", &state, &[14])
            .send();

        if let Some(result) = policy.get_result(100) {
            let actions = result["actions"].as_f32();
            println!("step {}: actions={:?}", step, &actions[..3.min(actions.len())]);
        }

        if step % 5 == 0 {
            telemetry.observe().urgency(0.1)
                .tensor_f32("state", &state, &[14]).send();
        }
    }
}
```

#### Async

```rust
use inferential::AsyncConnection;

#[tokio::main]
async fn main() {
    let mut conn = AsyncConnection::new("tcp://localhost:5555", "arm-01", "franka").await;
    let mut policy = conn.model("manipulation-policy", 30.0, 0);

    let steps = 100u32;
    for step in 0..steps {
        let state: Vec<f32> = (0..14).map(|i| i as f32 * 0.1).collect();
        let urgency = (1.0f32).min((steps - step) as f32 / steps as f32);
        policy.observe()
            .urgency(urgency)
            .steps_remaining(steps - step)
            .tensor_f32("state", &state, &[14])
            .send()
            .await;

        if let Some(result) = policy.get_result(100).await {
            let actions = result["actions"].as_f32();
            println!("step {}: actions={:?}", step, &actions[..3.min(actions.len())]);
        }
    }
}
```

## Server Extensions (Python)

### Metric Callback

```python
@server.on_metric
def log_metrics(name: str, value: float, labels: dict) -> None:
    client = labels.get("client", "")
    model  = labels.get("model", "")
    if name == "e2e_latency_ms":
        print(f"  [{model}/{client}] {value:.1f}ms")
    if name == "queue_depth":
        print(f"  [{model}] depth={int(value)}")
```

### Swap Scheduler

```python
server.use_scheduler("round_robin")        # baseline, no priority
server.use_scheduler("model_deadline")     # recommended: per-model queues + priority scoring
server.use_scheduler("deadline_aware")     # single queue, deadline + urgency scoring
server.use_scheduler("priority_tiered")    # legacy: fixed priority tiers
```

### Custom Scoring

```python
from inferential import register_policy, InferenceRequest

@register_policy("latency_first")
def score(req: InferenceRequest) -> float:
    return 1.0 / max(req.latency_budget_ms, 1.0)

server.use_scheduler("model_deadline", policy="latency_first")
```

### Real Model

Replace the mock with your GPU-backed model. With Ray Serve, set `num_replicas` to match available GPUs and `max_inflight` to the same value so the scheduler owns the queue. With LocalDispatcher, pass your model's inference callable directly:

```python
@serve.deployment(num_replicas=4, ray_actor_options={"num_gpus": 1})
class ManipulationPolicy:
    def __init__(self):
        self.model = load_model("weights.pt")

    def infer(self, obs: dict) -> dict:
        return {"actions": self.model.predict(obs["state"])}
```

### Queue Tuning

```python
from inferential.config.schema import InferentialConfig, ModelConfig, ModelsConfig

config = InferentialConfig(
    models=ModelsConfig(
        known={
            "manipulation-policy": ModelConfig(max_inflight=4),  # match num_replicas/GPUs
            "telemetry":           ModelConfig(max_inflight=1),
        },
        default_max_inflight=2,
    ),
)
config.transport.bind = "tcp://*:5555"
config.scheduling.strategy = "model_deadline"
config.scheduling.pipeline_dispatch.enabled = True
config.scheduling.request_ttl_ms = 2000
config.scheduling.overflow_policy = "drop_oldest"
config.scheduling.max_retries = 2

server = Server(config=config, models=["manipulation-policy", "telemetry"])
```
