# Examples

Each SDK includes ready-to-run example scripts. This guide walks through the patterns.

## Server Setup

All examples connect to an Inferential server. Start one first:

```bash
cd python
ray start --head
python examples/server_demo.py
```

Wait for `Application 'policy-v2' is ready`. The server runs a `MockPolicy` that echoes random actions of the same dimension as the input.

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

conn = Connection(server="tcp://localhost:5555", client_id="sim-01", client_type="sim")
model = conn.model("policy-v2", latency_budget_ms=30.0)

for step in range(100):
    state = np.random.randn(7).astype(np.float32)
    model.observe(urgency=0.5, state=state)

    result = model.get_result(timeout_ms=100)
    if result is not None:
        actions = result["actions"]
        print(f"step {step}: actions={actions[:3]}...")

conn.close()
```

#### Async Client

```python
from inferential import AsyncConnection
import numpy as np, asyncio

async def robot_loop(conn, bot_id, steps):
    model = conn.model("policy-v2", latency_budget_ms=30.0)
    for step in range(steps):
        state = np.random.randn(7).astype(np.float32)
        await model.observe(urgency=0.5, state=state)
        result = await model.get_result(timeout_ms=100)
        await asyncio.sleep(0.05)

async def main():
    async with AsyncConnection(server="tcp://localhost:5555", client_id="cell") as conn:
        await asyncio.gather(
            robot_loop(conn, "bot-01", 100),
            robot_loop(conn, "bot-02", 100),
            robot_loop(conn, "bot-03", 100),
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
    auto conn = inferential::Connection("tcp://localhost:5555", "cpp-agent", "sim");
    auto model = conn.model("policy-v2", 30.0f, 1);

    for (int step = 0; step < 100; ++step) {
        std::vector<float> state(7, static_cast<float>(step) * 0.1f);
        model.observe()
            .urgency(0.5f)
            .tensor_f32("state", state.data(), state.size(), {7})
            .send();

        auto result = model.get_result(100);
        if (result) {
            auto [ptr, count] = (*result)["actions"].as<float>();
            std::cout << "step " << step << ": actions=[";
            for (size_t i = 0; i < std::min(count, size_t(3)); ++i)
                std::cout << ptr[i] << " ";
            std::cout << "...]" << std::endl;
        }
    }
    // conn closes automatically via destructor
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
    let conn = Connection::new("tcp://localhost:5555", "rust-agent", "sim");
    let model = conn.model("policy-v2", 30.0, 1);

    for step in 0..100 {
        let state: Vec<f32> = (0..7).map(|i| i as f32 * 0.1).collect();
        model.observe()
            .urgency(0.5)
            .tensor_f32("state", &state, &[7])
            .send();

        if let Some(result) = model.get_result(100) {
            let actions = result["actions"].as_f32();
            println!("step {}: actions={:?}", step, &actions[..3.min(actions.len())]);
        }
    }
}
```

#### Async

```rust
use inferential::AsyncConnection;

#[tokio::main]
async fn main() {
    let mut conn = AsyncConnection::new("tcp://localhost:5555", "rust-async", "sim").await;
    let mut model = conn.model("policy-v2", 30.0, 1);

    for step in 0..100 {
        let state: Vec<f32> = (0..7).map(|i| i as f32 * 0.1).collect();
        model.observe()
            .urgency(0.5)
            .tensor_f32("state", &state, &[7])
            .send()
            .await;

        if let Some(result) = model.get_result(100).await {
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
    if name in ("inference_latency_ms", "e2e_latency_ms"):
        print(f"  [{client}] {name}: {value:.1f}ms")
```

### Swap Scheduler

```python
server.use_scheduler("round_robin")
server.use_scheduler("batch_optimized")
server.use_scheduler("priority_tiered")
```

### Custom Scoring

```python
from inferential import register_policy, InferenceRequest

@register_policy("latency_first")
def score(req: InferenceRequest) -> float:
    return 1.0 / max(req.latency_budget_ms, 1.0)

server.use_scheduler("deadline_aware", policy="latency_first")
```

### Real Model

Replace `MockPolicy` with your model:

```python
@serve.deployment(num_replicas=2, ray_actor_options={"num_gpus": 1})
class MyModel:
    def __init__(self):
        self.model = load_model("weights.pt")

    def infer(self, obs: dict) -> dict:
        return {"actions": self.model.predict(obs["state"])}
```

### Queue Tuning

```python
from inferential.config import InferentialConfig

server = Server(
    bind="tcp://*:5555",
    models=["policy-v2"],
    config=InferentialConfig(
        scheduling={
            "strategy": "deadline_aware",
            "request_ttl_ms": 2000,
            "overflow_policy": "drop_oldest",
            "max_retries": 2,
        }
    ),
)
```
