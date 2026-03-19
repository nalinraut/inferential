# Architecture

## System Overview

Inferential sits between your clients and your ML models. Clients send observations over ZMQ, the server schedules and dispatches inference to Ray Serve, and streams results back.

```
                        Inferential Server
                  ┌──────────────────────────────┐
                  │                              │
  Client A ──ZMQ──┤  Assembler ──► Scheduler     │
  Client B ──ZMQ──┤                  │           │
  Client C ──ZMQ──┤              next_batch()    │
                  │                  │           │
                  │              Dispatcher ─────┼──► Ray Serve
                  │                  │           │    (model replicas)
                  │              responses       │
                  │                  │           │
  Client A ◄─ZMQ─┤  Transport ◄────┘           │
  Client B ◄─ZMQ─┤                              │
  Client C ◄─ZMQ─┤  Metrics   Cadence   Health  │
                  └──────────────────────────────┘
```

Clients can be written in **Python**, **C++**, or **Rust** — any language that speaks the ZMQ + protobuf wire protocol.

### Data Flow

1. **Client** serializes tensor data + metadata into a protobuf envelope and binary payload, sends over ZMQ
2. **Transport** (ROUTER socket) receives the multipart message, tags it with identity and `received_at`
3. **Assembler** parses the protobuf envelope, validates tensor slots against the payload, extracts client/model info
4. **Server** updates cadence tracking, client registry, and metrics, then builds an `InferenceRequest`
5. **Scheduler** queues the request and scores/orders it according to its strategy
6. **Dispatcher** pulls the next batch, deserializes tensors, calls `model.infer()` via Ray Serve handles
7. **Response** is serialized back to protobuf + binary and sent to the client over ZMQ

## Wire Protocol

All client SDKs speak the same binary protocol. Messages use protobuf serialization over ZMQ multipart frames:

```
[identity | "" | envelope (protobuf bytes) | payload (binary tensors)]
```

The envelope contains metadata (client info, model ID, tensor descriptors with dtype/shape/offset), while the payload is a single concatenated binary buffer holding all tensor data. This two-part design keeps metadata parseable without touching the tensor bytes.

### Observation (client → server)

```protobuf
message Observation {
    Client client = 1;           // id + type
    uint64 timestamp_ns = 2;
    float urgency = 3;           // 0.0 (low) to 1.0 (critical)
    repeated Tensor tensors = 4; // descriptors with byte_offset into payload
    string model_id = 5;
    optional uint32 steps_remaining = 6;
    map<string, string> metadata = 7;
    uint32 priority = 8;         // 0 = highest priority
}
```

### ModelResponse (server → client)

```protobuf
message ModelResponse {
    Client client = 1;
    string response_id = 2;
    uint64 timestamp_ns = 3;
    repeated Tensor tensors = 4;
    float inference_latency_ms = 5;
    string model_id = 6;
    map<string, string> metadata = 7;
}
```

### Tensor Descriptor

Each tensor entry in the envelope describes a slice of the binary payload:

```protobuf
message Tensor {
    string key = 1;           // "camera/wrist", "proprioception"
    DType dtype = 2;
    repeated int64 shape = 3;
    uint64 byte_offset = 4;   // offset into the payload buffer
    uint64 byte_length = 5;
    uint64 timestamp_ns = 6;  // per-tensor temporal alignment
    Encoding encoding = 7;    // RAW, JPEG, PNG
}
```

### Supported Types

| DType | Python (NumPy) | C++ | Rust | Bytes |
|-------|---------------|-----|------|-------|
| `FLOAT16` | `float16` | — | — | 2 |
| `FLOAT32` | `float32` | `float` | `f32` | 4 |
| `FLOAT64` | `float64` | `double` | `f64` | 8 |
| `BFLOAT16` | `float16` | — | — | 2 |
| `UINT8` | `uint8` | `uint8_t` | `u8` | 1 |
| `INT32` | `int32` | `int32_t` | `i32` | 4 |
| `INT64` | `int64` | `int64_t` | `i64` | 8 |
| `BOOL` | `bool` | `bool` | `bool` | 1 |

Supported encodings: `RAW`, `JPEG`, `PNG`.

## Tensor Handling

The system is **framework-agnostic**. Tensors travel as raw bytes over the wire with dtype/shape metadata in the protobuf envelope.

**Sending (all clients)**: Tensor data is serialized to raw bytes with shape and dtype tracked in the envelope. Multiple tensors are concatenated into a single payload buffer, with each tensor's `byte_offset` and `byte_length` recorded in its descriptor.

**Server side (input)**: The dispatcher deserializes binary back to numpy dicts. The `infer()` method receives a `dict[str, np.ndarray]`.

**Server side (output)**: The dispatcher accepts any array-like return from `infer()` — numpy, PyTorch, JAX, TensorFlow — and converts via `np.asarray()`.

**Receiving (all clients)**: The response payload is sliced by each tensor's `byte_offset`/`byte_length` and reinterpreted according to dtype.

## Client SDK API Pattern

All three SDKs follow the same pattern:

```
Connection(server, client_id, client_type)
    └── model(model_id, latency_budget_ms, priority) → Model
            ├── observe() → ObservationBuilder
            │       ├── .urgency(float)
            │       ├── .tensor / .tensor_f32(key, data, shape)
            │       ├── .metadata(key, value)
            │       └── .send()
            └── get_result(timeout_ms) → Result | None
```

| Concept | Python | C++ | Rust (sync) | Rust (async) |
|---------|--------|-----|-------------|--------------|
| Connect | `Connection(...)` | `Connection(...)` | `Connection::new(...)` | `AsyncConnection::new(...).await` |
| Model handle | `conn.model(...)` | `conn.model(...)` | `conn.model(...)` | `conn.model(...)` |
| Send observation | `model.observe(urgency=0.8, state=arr)` | `model.observe().urgency(0.8f).tensor_f32(...).send()` | `model.observe().urgency(0.8).tensor_f32(...).send()` | `...send().await` |
| Get result | `model.get_result(50)` | `model.get_result(50)` | `model.get_result(50)` | `model.get_result(50).await` |
| Close | `conn.close()` | `conn.close()` / destructor | `Drop` / `conn.close()` | `conn.close().await` |

**Python** uses `**kwargs` for the observation API — pass numpy arrays and strings directly as keyword arguments. **C++** and **Rust** use a fluent builder pattern with explicit `.tensor()` / `.metadata()` calls.

## Schedulers

Six built-in strategies. Each implements the `Scheduler` ABC and can be used standalone or via the server.

| Strategy | Data Structure | Description |
|----------|---------------|-------------|
| `deadline_aware` | heapq | Weighted scoring: cadence, urgency, priority, age |
| `model_deadline` | per-model heapq | Per-model queues with deadline-aware scoring (default, recommended) |
| `tiered_deadline` | per-tier heapq | Legacy per-priority-tier isolation with deadline-aware scoring |
| `batch_optimized` | dict of deques | Groups requests per model, flushes on size or time threshold |
| `priority_tiered` | list of deques | Strict priority tiers (0=highest), FIFO within each tier |
| `round_robin` | OrderedDict of deques | Fair rotation across clients |

### Model Deadline Scheduling

The `model_deadline` scheduler routes requests by `model_id` into per-model heaps. Within each model queue, requests are ordered by a deadline-aware score that factors in cadence, urgency, priority, steps remaining, and age. Priority (sent by the client) is a scoring factor, not a routing key.

```
              _receive_loop()
                    |
            scheduler.submit()
              /           \
    "policy" heap       "telemetry" heap
         |                     |
 _model_dispatch_loop       _model_dispatch_loop
     semaphore(3)               semaphore(1)
```

When combined with **pipeline dispatch**, each model gets its own dispatch loop with a semaphore bounded to `max_inflight` (matching the model's Ray Serve replica count). Models are registered dynamically — when the first request for a new model arrives, the server spins up its dispatch loop.

```python
config = InferentialConfig(
    models={
        "known": {
            "manipulation-policy": {"max_inflight": 3},
            "telemetry": {"max_inflight": 1},
        },
        "default_max_inflight": 2,
    },
    scheduling={
        "strategy": "model_deadline",
        "pipeline_dispatch": {"enabled": True},
    },
)
```

### Priority and Urgency

Priority and urgency are separate dimensions, both set by the client:

- **Priority** (`uint32`, 0=highest): Determines scoring weight within the queue. Set at `conn.model()` time, can be overridden per-observation.
- **Urgency** (`float`, 0.0-1.0): How time-critical this specific observation is. Higher urgency = dispatched sooner within the same priority level.

```python
conn = Connection(server="tcp://localhost:5555", client_id="station-01", client_type="franka")
manip = conn.model("manipulation-policy", priority=0)   # high priority
telemetry = conn.model("telemetry", priority=1)          # lower priority

manip.observe(urgency=0.5, state=state)                  # normal
manip.observe(urgency=1.0, priority=0, state=state)      # emergency override
```

### Custom Scheduling

Schedulers are configured server-side (Python):

```python
# Swap strategy
server.use_scheduler("round_robin")

# Custom scoring policy for deadline-aware or model_deadline
from inferential import register_policy, InferenceRequest

@register_policy("safety_first")
def score(req: InferenceRequest) -> float:
    if req.urgency > 0.9:
        return 1000.0
    return req.priority * 10.0

server.use_scheduler("model_deadline", policy="safety_first")
```

See the [Python SDK docs](../python/) for full scheduler and server configuration details.

## Queue Management

The scheduler queue supports TTL, overflow policies, and dispatch retry:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `request_ttl_ms` | 5000 | Drop queued requests older than this |
| `overflow_policy` | `"drop_oldest"` | What to do when the queue is full |
| `max_retries` | 0 | Re-queue failed dispatches up to N times |
| `max_concurrent_dispatch` | 8 | Max requests dispatched concurrently (legacy loop) |

### Pipeline Dispatch

When `pipeline_dispatch.enabled = True` and the scheduler is model-aware (`model_deadline`), the server runs a separate dispatch coroutine per model. Each model has a semaphore bounded to its `max_inflight` value, matching the model's Ray Serve replica count.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `pipeline_dispatch.enabled` | `false` | Enable per-model dispatch loops |
| `models.known.<id>.max_inflight` | — | Max concurrent dispatches for a known model |
| `models.default_max_inflight` | 2 | Fallback for unknown models |

Dispatch loops are created dynamically when the first request for a model arrives, using `asyncio.Event` for instant wake-up (no polling delay).

## Metrics

In-memory ring-buffer metrics with label filtering and percentile stats.

### Tracked Metrics

| Metric | Labels | Description |
|--------|--------|-------------|
| `inference_latency_ms` | `client`, `model`* | Time spent in Ray Serve model |
| `scheduling_wait_ms` | `client`, `model`* | Time request sat in queue before dispatch |
| `e2e_latency_ms` | `client`, `model`* | Total server-side time (queue wait + inference) |
| `observation_staleness_ms` | `client` | Age of observation on arrival |
| `payload_size_bytes` | `client` | Size of binary tensor payload |
| `queue_depth` | `model`* | Scheduler queue length at dispatch time |
| `batch_size` | — | Number of requests per dispatch batch (legacy loop) |
| `active_clients` | — | Currently connected client count |
| `queue_full_drops` | `client` | Requests dropped due to full queue |
| `dispatch_errors` | `client`, `error` | Failed dispatches (after retries exhausted) |
| `dispatch_retries` | `client`, `model`* | Dispatch retry attempts |
| `send_errors` | `client` | ZMQ response send failures |
| `requests_expired` | — | Requests dropped by TTL |
| `observation_errors` | — | Malformed observations |
| `client_disconnected` | `client` | Client disconnect events |

\* `model` label is present when using pipeline dispatch with a model-aware scheduler (`model_deadline`). In the legacy dispatch loop, these metrics have no `model` label.

### Querying Metrics (Python)

```python
@server.on_metric
def log(name, value, labels):
    model = labels.get("model", "all")
    print(f"[{model}] {name}: {value}")

# Global stats
stats = server.metrics.get_stats("inference_latency_ms", window_seconds=60)
print(f"p95: {stats.p95:.1f}ms")

# Per-model queue depth
policy_depth = server.metrics.get_latest("queue_depth", labels={"model": "manipulation-policy"})
telemetry_depth = server.metrics.get_latest("queue_depth", labels={"model": "telemetry"})

# Per-model latency stats
policy_e2e = server.metrics.get_stats("e2e_latency_ms", labels={"model": "manipulation-policy"})
```

## Configuration

Server configuration (Python):

```python
from inferential.config import InferentialConfig

config = InferentialConfig(
    transport={"bind": "tcp://*:5555", "recv_hwm": 2000},
    scheduling={
        "strategy": "model_deadline",
        "max_queue_size": 500,
        "request_ttl_ms": 3000,
        "overflow_policy": "drop_oldest",
        "max_retries": 1,
        "pipeline_dispatch": {"enabled": True},
    },
    models={
        "known": {
            "manipulation-policy": {"max_inflight": 3},
            "telemetry": {"max_inflight": 1},
        },
        "default_max_inflight": 2,
    },
    clients={
        "defaults": {"latency_budget_ms": 50.0},
        "known": [
            {"id": "agent-01", "model": "policy-v2", "latency_budget_ms": 30.0},
        ],
        "accept_unknown": True,
    },
    response_tracking={"cadence_alpha": 0.3, "disconnect_timeout_s": 10.0},
    metrics={"ring_buffer_size": 10000},
)

server = Server(config=config)
```

### Model Deadline with Pipeline Dispatch

```python
from inferential import Server
from inferential.config.schema import InferentialConfig

config = InferentialConfig(
    models={
        "known": {
            "manipulation-policy": {"max_inflight": 3},
            "telemetry": {"max_inflight": 1},
        },
        "default_max_inflight": 2,
    },
    scheduling={
        "strategy": "model_deadline",
        "pipeline_dispatch": {"enabled": True},
    },
)

server = Server(config=config)
```
