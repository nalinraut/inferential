# Inferential Rust SDK

Rust client SDK for [Inferential](https://github.com/nalinraut/inferential) inference orchestration. Communicates with the server using ZMQ DEALER sockets and the shared [protobuf wire protocol](https://github.com/nalinraut/inferential/blob/main/proto/inferential.proto).

Includes both synchronous (`Connection`/`Model`) and async (`AsyncConnection`/`AsyncModel`) clients.

## Install

Add to your `Cargo.toml`:

```toml
[dependencies]
inferential = "0.3"
```

Or via the CLI:

```bash
cargo add inferential
```

Requires `libzmq` installed on the system (`apt install libzmq3-dev` or `brew install zmq`).

## Build from Source

Requires Rust 1.70+, `libzmq` installed on the system, and `protoc` (protobuf compiler).

```bash
# Build
cargo build

# Run tests (single-threaded — tests share ZMQ ports)
cargo test -- --test-threads=1

# Run examples
cargo run --example client_demo
cargo run --example async_client_demo

# Lint
cargo clippy
```

Or use the root Makefile:

```bash
make build-rust
make test-rust
```

## Usage

### Sync

```rust
use inferential::{Connection, proto};

let conn = Connection::new("tcp://localhost:5555", "agent-01", "franka");
let model = conn.model("policy-v2", 30.0, 1);

let joints: Vec<f32> = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7];
model.observe()
    .urgency(0.8)
    .tensor_f32("joint_positions", &joints, &[7])
    .send();

if let Some(result) = model.get_result(100) {
    let actions = result["actions"].as_f32();
}

conn.close();  // also called on Drop
```

### Async

```rust
use inferential::AsyncConnection;

#[tokio::main]
async fn main() {
    let mut conn = AsyncConnection::new("tcp://localhost:5555", "agent-01", "franka").await;
    let mut model = conn.model("policy-v2", 30.0, 1);

    let joints: Vec<f32> = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7];
    model.observe()
        .urgency(0.8)
        .tensor_f32("joint_positions", &joints, &[7])
        .send()
        .await;

    if let Some(result) = model.get_result(100).await {
        let actions = result["actions"].as_f32();
    }

    // Must drop model before closing conn (borrow rules)
    drop(model);
    conn.close().await;  // consumes self
}
```

## API Reference

### `Connection` (sync)

```rust
Connection::new(server: &str, client_id: &str, client_type: &str) -> Self
Connection::with_options(server, client_id, client_type, reconnect_ivl_ms, reconnect_max_ms) -> Self
```

Creates a ZMQ DEALER connection. Auto-prefixes `tcp://` if missing. Implements `Drop` for cleanup.

| Method | Description |
|--------|-------------|
| `model(model_id, latency_budget_ms, priority)` | Create a `Model<'_>` handle (borrows connection) |
| `close()` | Explicit close (also called on drop) |

### `AsyncConnection`

```rust
AsyncConnection::new(server: &str, client_id: &str, client_type: &str) -> Self  // async
```

Tokio-based async connection using the `zeromq` crate.

| Method | Description |
|--------|-------------|
| `model(model_id, latency_budget_ms, priority)` | Create an `AsyncModel<'_>` (requires `&mut self`) |
| `close(self)` | Close and consume the connection (async, takes ownership) |

### `Model<'a>` / `AsyncModel<'a>`

Obtained via `conn.model(...)`. Borrows the connection.

| Method | Sync | Async |
|--------|------|-------|
| `observe()` | `ObservationBuilder<'_>` | `AsyncObservationBuilder<'_>` |
| `get_result(timeout_ms)` | `Option<HashMap<String, TensorData>>` | Same, `async` |

### `ObservationBuilder` / `AsyncObservationBuilder`

Fluent builder — chain methods and finalize with `.send()`.

| Method | Description |
|--------|-------------|
| `.urgency(f32)` | Set urgency hint (0.0–1.0) |
| `.steps_remaining(u32)` | Set remaining trajectory steps |
| `.tensor(key, data: &[u8], dtype: i32, shape: &[i64])` | Add raw tensor |
| `.tensor_f32(key, data: &[f32], shape: &[i64])` | Add float32 tensor (convenience) |
| `.metadata(key, value)` | Add string metadata |
| `.send()` | Serialize and send (async version: `.send().await`) |

All builder methods consume and return `self` (move semantics).

### `TensorData`

Holds tensor data received from `get_result()`.

```rust
pub struct TensorData {
    pub key: String,
    pub data: Vec<u8>,       // raw bytes
    pub shape: Vec<i64>,
    pub dtype: i32,          // proto::DType as i32
}
```

| Method | Return | Description |
|--------|--------|-------------|
| `element_size()` | `usize` | Bytes per element |
| `numel()` | `usize` | Total element count |
| `as_f32()` | `&[f32]` | Reinterpret as float32 slice |
| `as_f64()` | `&[f64]` | Reinterpret as float64 slice |
| `as_i32()` | `&[i32]` | Reinterpret as int32 slice |
| `as_i64()` | `&[i64]` | Reinterpret as int64 slice |
| `as_u8()` | `&[u8]` | Raw byte slice |

## Complete Example

### Sync

```rust
use inferential::Connection;
use std::{thread, time::Duration};

fn main() {
    let conn = Connection::new("tcp://localhost:5555", "rust-demo", "sim");
    let model = conn.model("policy-v2", 30.0, 1);

    for step in 0..100 {
        let state: Vec<f32> = (0..7).map(|i| i as f32 * 0.1).collect();

        model.observe()
            .urgency(0.5)
            .steps_remaining(100 - step)
            .tensor_f32("state", &state, &[7])
            .metadata("task", "pick_and_place")
            .send();

        if let Some(result) = model.get_result(100) {
            let actions = result["actions"].as_f32();
            if step % 20 == 0 {
                println!("step {}: actions={:?}", step, &actions[..3.min(actions.len())]);
            }
        }

        thread::sleep(Duration::from_millis(50));
    }
}
```

### Async

```rust
use inferential::AsyncConnection;
use tokio::time::{sleep, Duration};

#[tokio::main]
async fn main() {
    let mut conn = AsyncConnection::new("tcp://localhost:5555", "rust-async", "sim").await;

    {
        let mut model = conn.model("policy-v2", 30.0, 1);

        for step in 0u32..100 {
            let state: Vec<f32> = (0..7).map(|i| i as f32 * 0.1).collect();

            model.observe()
                .urgency(0.5)
                .steps_remaining(100 - step)
                .tensor_f32("state", &state, &[7])
                .metadata("task", "pick_and_place")
                .send()
                .await;

            if let Some(result) = model.get_result(100).await {
                let actions = result["actions"].as_f32();
                if step % 20 == 0 {
                    println!("step {}: actions={:?}", step, &actions[..3.min(actions.len())]);
                }
            }

            sleep(Duration::from_millis(50)).await;
        }
    } // model dropped here, releasing borrow on conn

    conn.close().await;
}
```

## Notes

- **Sync vs Async**: The sync client uses the `zmq` crate (libzmq binding). The async client uses the `zeromq` crate (native Rust, tokio-based).
- **Borrow rules**: `Model` borrows `&Connection`, `AsyncModel` borrows `&mut AsyncConnection`. Drop the model before calling `conn.close()`.
- **Thread safety**: `Connection` is not `Send`/`Sync` (ZMQ sockets are single-threaded). Create one connection per thread.
- **Test threading**: Tests must run single-threaded (`--test-threads=1`) because they bind to fixed ZMQ ports.
- **Wire protocol**: See [Architecture](https://github.com/nalinraut/inferential/blob/main/docs/architecture.md) for the shared protobuf wire protocol.

## Documentation

- [Architecture](https://github.com/nalinraut/inferential/blob/main/docs/architecture.md) — Wire protocol, system design
- [Examples](https://github.com/nalinraut/inferential/blob/main/docs/examples.md) — Multi-language examples
- [Contributing](https://github.com/nalinraut/inferential/blob/main/docs/contributing.md) — Development workflow
