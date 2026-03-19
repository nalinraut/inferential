#pragma once

#include <chrono>
#include <cstdint>
#include <initializer_list>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>
#include <zmq.hpp>

#include "inferential/tensor.hpp"

namespace inferential {

class Model;

/// ZMQ DEALER connection to an Inferential server.
class Connection {
public:
    explicit Connection(const std::string& server = "localhost:5555",
                        const std::string& client_id = "", const std::string& client_type = "",
                        int reconnect_ivl_ms = 100, int reconnect_max_ms = 5000);

    ~Connection();

    Connection(const Connection&) = delete;
    Connection& operator=(const Connection&) = delete;
    Connection(Connection&&) noexcept;
    Connection& operator=(Connection&&) noexcept;

    /// Create a Model handle bound to this connection.
    Model model(const std::string& model_id, float latency_budget_ms = 50.0f, int priority = 1);

    /// Close the socket and terminate the context.
    void close();

private:
    friend class Model;
    friend class ObservationBuilder;

    void send(const std::string& envelope, const std::string& payload);
    std::optional<std::pair<std::string, std::string>> recv(int timeout_ms);

    static std::string normalize_server(const std::string& server);
    static uint64_t now_ns();

    zmq::context_t ctx_;
    zmq::socket_t socket_;
    std::string client_id_;
    std::string client_type_;
    bool closed_ = false;
};

/// Fluent builder for constructing and sending an Observation.
class ObservationBuilder {
public:
    ObservationBuilder& urgency(float u);
    ObservationBuilder& priority(int p);
    ObservationBuilder& steps_remaining(uint32_t s);

    /// Add a raw tensor (caller manages byte layout).
    ObservationBuilder& tensor(const std::string& key, const void* data, size_t size_bytes,
                               ::inferential::DType dtype, std::initializer_list<int64_t> shape);

    /// Add a float tensor (convenience).
    ObservationBuilder& tensor_f32(const std::string& key, const float* data, size_t count,
                                   std::initializer_list<int64_t> shape);

    /// Add string metadata.
    ObservationBuilder& metadata(const std::string& key, const std::string& value);

    /// Serialize and send the observation.
    void send();

private:
    friend class Model;
    ObservationBuilder(Connection* conn, const std::string& client_id,
                       const std::string& client_type, const std::string& model_id, int priority);

    Connection* conn_;
    std::string client_id_;
    std::string client_type_;
    std::string model_id_;
    float urgency_ = 0.0f;
    int priority_ = 1;
    std::optional<uint32_t> steps_remaining_;

    struct TensorEntry {
        std::string key;
        std::vector<uint8_t> data;
        ::inferential::DType dtype;
        std::vector<int64_t> shape;
    };
    std::vector<TensorEntry> tensors_;
    std::vector<std::pair<std::string, std::string>> metadata_;
};

/// Handle to a specific model on the server.
class Model {
public:
    /// Start building an observation to send.
    ObservationBuilder observe();

    /// Wait for a result. Returns std::nullopt on timeout.
    std::optional<std::unordered_map<std::string, TensorData>> get_result(int timeout_ms = 100);

private:
    friend class Connection;
    Model(Connection* conn, const std::string& model_id, float latency_budget_ms, int priority);

    Connection* conn_;
    std::string model_id_;
    float latency_budget_ms_;
    int priority_;
};

}  // namespace inferential
