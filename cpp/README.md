# Inferential C++ SDK

C++ client SDK for [Inferential](../README.md) inference orchestration. Communicates with the server using ZMQ DEALER sockets and the shared [protobuf wire protocol](../proto/inferential.proto).

## Install

Add to your `MODULE.bazel`:

```python
bazel_dep(name = "inferential", version = "1.0.1")
```

Then depend on `@inferential//cpp:inferential` in your BUILD files:

```python
cc_binary(
    name = "my_app",
    srcs = ["main.cpp"],
    deps = ["@inferential//cpp:inferential"],
)
```

## Build from Source

Requires [Bazel 7.x](https://bazel.build/) (pinned via `.bazelversion`). All dependencies (libzmq, cppzmq, protobuf, Google Test) are fetched automatically by Bazel — no system packages required.

```bash
# Build library
bazel build //cpp:inferential

# Run tests
bazel test //cpp/tests:test_client

# Build example
bazel build //cpp/examples:client_demo
```

Or use the root Makefile:

```bash
make build-cpp
make test-cpp
```

## Usage

```cpp
#include "inferential/client.hpp"

int main() {
    // Connect to server
    auto conn = inferential::Connection("tcp://localhost:5555", "agent-01", "franka");
    auto model = conn.model("policy-v2", 30.0f, 1);

    // Send an observation
    std::vector<float> joints = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f};
    model.observe()
        .urgency(0.8f)
        .tensor_f32("joint_positions", joints.data(), joints.size(), {7})
        .send();

    // Get result
    auto result = model.get_result(100);
    if (result) {
        auto [ptr, count] = (*result)["actions"].as<float>();
        // ptr points to count floats
    }
    // conn closes automatically via destructor
}
```

## API Reference

### `Connection`

```cpp
inferential::Connection(
    const std::string& server = "localhost:5555",
    const std::string& client_id = "",
    const std::string& client_type = "",
    int reconnect_ivl_ms = 100,
    int reconnect_max_ms = 5000
);
```

Creates a ZMQ DEALER connection. The server address is auto-prefixed with `tcp://` if missing. RAII — the destructor calls `close()`. Non-copyable, movable.

| Method | Description |
|--------|-------------|
| `model(model_id, latency_budget_ms, priority)` | Create a `Model` handle |
| `close()` | Close socket and terminate ZMQ context |

### `Model`

Obtained via `conn.model(...)`. Not independently constructible.

| Method | Description |
|--------|-------------|
| `observe()` | Returns an `ObservationBuilder` |
| `get_result(timeout_ms)` | Returns `std::optional<std::unordered_map<std::string, TensorData>>` |

### `ObservationBuilder`

Fluent builder returned by `model.observe()`. Chain methods and finalize with `.send()`.

| Method | Description |
|--------|-------------|
| `.urgency(float)` | Set urgency hint (0.0–1.0) |
| `.steps_remaining(uint32_t)` | Set remaining trajectory steps |
| `.tensor(key, data, size_bytes, dtype, shape)` | Add raw tensor |
| `.tensor_f32(key, data, count, shape)` | Add float32 tensor (convenience) |
| `.metadata(key, value)` | Add string metadata |
| `.send()` | Serialize and send the observation |

#### Tensor Methods

```cpp
// Raw tensor — you manage the byte layout
builder.tensor("image", img_data, img_bytes,
               inferential::UINT8, {3, 224, 224});

// Float32 convenience — handles byte conversion
std::vector<float> joints = {0.1f, 0.2f, 0.3f};
builder.tensor_f32("joints", joints.data(), joints.size(), {3});
```

### `TensorData`

Holds tensor data received from `get_result()`.

```cpp
struct TensorData {
    std::string key;
    std::vector<uint8_t> data;      // raw bytes
    std::vector<int64_t> shape;
    inferential::DType dtype;

    size_t element_size() const;    // bytes per element
    size_t numel() const;           // total element count

    // Zero-copy typed view — returns {pointer, count}
    template<typename T>
    std::pair<const T*, size_t> as() const;
};
```

#### Typed Access

```cpp
auto result = model.get_result(100);
if (result) {
    // Float tensor
    auto [fptr, fcount] = (*result)["actions"].as<float>();

    // Double tensor
    auto [dptr, dcount] = (*result)["values"].as<double>();

    // Shape info
    auto& shape = (*result)["actions"].shape;  // std::vector<int64_t>
}
```

## Complete Example

```cpp
#include "inferential/client.hpp"
#include <iostream>
#include <vector>
#include <thread>
#include <chrono>

int main() {
    auto conn = inferential::Connection("tcp://localhost:5555", "cpp-demo", "sim");
    auto model = conn.model("policy-v2", 30.0f, 1);

    for (int step = 0; step < 100; ++step) {
        std::vector<float> state(7, static_cast<float>(step) * 0.1f);

        model.observe()
            .urgency(0.5f)
            .steps_remaining(100 - step)
            .tensor_f32("state", state.data(), state.size(), {7})
            .metadata("task", "pick_and_place")
            .send();

        auto result = model.get_result(100);
        if (result) {
            auto [ptr, count] = (*result)["actions"].as<float>();
            if (step % 20 == 0) {
                std::cout << "step " << step << ": actions=[";
                for (size_t i = 0; i < std::min(count, size_t(3)); ++i)
                    std::cout << ptr[i] << " ";
                std::cout << "...]" << std::endl;
            }
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
}
```

## Code Formatting

This project uses `clang-format` (Google style, configured in `.clang-format`).

```bash
# Check formatting
find cpp -name '*.cpp' -o -name '*.hpp' | xargs clang-format --dry-run -Werror

# Auto-format
find cpp -name '*.cpp' -o -name '*.hpp' | xargs clang-format -i
```

CI enforces formatting — PRs with unformatted C++ code will fail.

## Notes

- **C++ standard**: C++17 (configured in `.bazelrc`)
- **No async client**: C++ lacks native async/await. Use threads for concurrent clients.
- **Thread safety**: Each `Connection` owns its own ZMQ context and socket. Use one `Connection` per thread.
- **Wire protocol**: See [Architecture](../docs/architecture.md) for the shared protobuf wire protocol.

## Documentation

- [Architecture](../docs/architecture.md) — Wire protocol, system design
- [Examples](../docs/examples.md) — Multi-language examples
- [Contributing](../docs/contributing.md) — Development workflow
