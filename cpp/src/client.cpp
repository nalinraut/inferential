#include "inferential/client.hpp"

#include <chrono>
#include <cstring>
#include <utility>
#include <zmq.hpp>
#include <zmq_addon.hpp>

#include "proto/inferential.pb.h"

namespace inferential {

// ---------------------------------------------------------------------------
// Connection
// ---------------------------------------------------------------------------

std::string Connection::normalize_server(const std::string& server) {
    if (server.rfind("tcp://", 0) != 0) {
        return "tcp://" + server;
    }
    return server;
}

uint64_t Connection::now_ns() {
    auto now = std::chrono::system_clock::now();
    auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch()).count();
    return static_cast<uint64_t>(ns);
}

Connection::Connection(const std::string& server, const std::string& client_id,
                       const std::string& client_type, int reconnect_ivl_ms, int reconnect_max_ms)
    : ctx_(1),
      socket_(ctx_, zmq::socket_type::dealer),
      client_id_(client_id),
      client_type_(client_type) {
    socket_.set(zmq::sockopt::routing_id, client_id);
    socket_.set(zmq::sockopt::reconnect_ivl, reconnect_ivl_ms);
    socket_.set(zmq::sockopt::reconnect_ivl_max, reconnect_max_ms);
    socket_.set(zmq::sockopt::rcvhwm, 100);
    socket_.set(zmq::sockopt::sndhwm, 100);
    socket_.set(zmq::sockopt::linger, 0);
    socket_.connect(normalize_server(server));
}

Connection::~Connection() {
    close();
}

Connection::Connection(Connection&& other) noexcept
    : ctx_(std::move(other.ctx_)),
      socket_(std::move(other.socket_)),
      client_id_(std::move(other.client_id_)),
      client_type_(std::move(other.client_type_)),
      closed_(other.closed_) {
    other.closed_ = true;
}

Connection& Connection::operator=(Connection&& other) noexcept {
    if (this != &other) {
        close();
        ctx_ = std::move(other.ctx_);
        socket_ = std::move(other.socket_);
        client_id_ = std::move(other.client_id_);
        client_type_ = std::move(other.client_type_);
        closed_ = other.closed_;
        other.closed_ = true;
    }
    return *this;
}

Model Connection::model(const std::string& model_id, float latency_budget_ms, int priority) {
    return Model(this, model_id, latency_budget_ms, priority);
}

void Connection::close() {
    if (!closed_) {
        socket_.close();
        ctx_.close();
        closed_ = true;
    }
}

void Connection::send(const std::string& envelope, const std::string& payload) {
    std::array<zmq::const_buffer, 3> msgs = {
        zmq::str_buffer(""),
        zmq::buffer(envelope),
        zmq::buffer(payload),
    };
    zmq::send_multipart(socket_, msgs);
}

std::optional<std::pair<std::string, std::string>> Connection::recv(int timeout_ms) {
    zmq::pollitem_t items[] = {{socket_.handle(), 0, ZMQ_POLLIN, 0}};
    zmq::poll(items, 1, std::chrono::milliseconds(timeout_ms));

    if (!(items[0].revents & ZMQ_POLLIN)) {
        return std::nullopt;
    }

    std::vector<zmq::message_t> frames;
    auto result =
        zmq::recv_multipart(socket_, std::back_inserter(frames), zmq::recv_flags::dontwait);
    if (!result || frames.size() < 3) {
        return std::nullopt;
    }

    // frames[0] = empty, frames[1] = envelope, frames[2] = payload
    std::string envelope(static_cast<const char*>(frames[1].data()), frames[1].size());
    std::string payload(static_cast<const char*>(frames[2].data()), frames[2].size());
    return std::make_pair(std::move(envelope), std::move(payload));
}

// ---------------------------------------------------------------------------
// ObservationBuilder
// ---------------------------------------------------------------------------

ObservationBuilder::ObservationBuilder(Connection* conn, const std::string& client_id,
                                       const std::string& client_type, const std::string& model_id,
                                       int priority)
    : conn_(conn),
      client_id_(client_id),
      client_type_(client_type),
      model_id_(model_id),
      priority_(priority) {}

ObservationBuilder& ObservationBuilder::urgency(float u) {
    urgency_ = u;
    return *this;
}

ObservationBuilder& ObservationBuilder::priority(int p) {
    priority_ = p;
    return *this;
}

ObservationBuilder& ObservationBuilder::steps_remaining(uint32_t s) {
    steps_remaining_ = s;
    return *this;
}

ObservationBuilder& ObservationBuilder::tensor(const std::string& key, const void* data,
                                               size_t size_bytes, ::inferential::DType dtype,
                                               std::initializer_list<int64_t> shape) {
    TensorEntry entry;
    entry.key = key;
    entry.data.resize(size_bytes);
    std::memcpy(entry.data.data(), data, size_bytes);
    entry.dtype = dtype;
    entry.shape.assign(shape);
    tensors_.push_back(std::move(entry));
    return *this;
}

ObservationBuilder& ObservationBuilder::tensor_f32(const std::string& key, const float* data,
                                                   size_t count,
                                                   std::initializer_list<int64_t> shape) {
    return tensor(key, data, count * sizeof(float), FLOAT32, shape);
}

ObservationBuilder& ObservationBuilder::metadata(const std::string& key, const std::string& value) {
    metadata_.emplace_back(key, value);
    return *this;
}

void ObservationBuilder::send() {
    ::inferential::Observation obs;

    auto* client = obs.mutable_client();
    client->set_id(client_id_);
    client->set_type(client_type_);

    obs.set_model_id(model_id_);
    obs.set_timestamp_ns(Connection::now_ns());
    obs.set_urgency(urgency_);
    obs.set_priority(priority_);

    if (steps_remaining_.has_value()) {
        obs.set_steps_remaining(steps_remaining_.value());
    }

    // Build payload and tensor descriptors
    std::string payload;
    uint64_t offset = 0;

    for (const auto& entry : tensors_) {
        auto* t = obs.add_tensors();
        t->set_key(entry.key);
        t->set_dtype(entry.dtype);
        for (auto dim : entry.shape) {
            t->add_shape(dim);
        }
        t->set_byte_offset(offset);
        t->set_byte_length(entry.data.size());
        t->set_timestamp_ns(obs.timestamp_ns());
        t->set_encoding(RAW);

        payload.append(reinterpret_cast<const char*>(entry.data.data()), entry.data.size());
        offset += entry.data.size();
    }

    for (const auto& [k, v] : metadata_) {
        (*obs.mutable_metadata())[k] = v;
    }

    std::string envelope;
    obs.SerializeToString(&envelope);
    conn_->send(envelope, payload);
}

// ---------------------------------------------------------------------------
// Model
// ---------------------------------------------------------------------------

Model::Model(Connection* conn, const std::string& model_id, float latency_budget_ms, int priority)
    : conn_(conn),
      model_id_(model_id),
      latency_budget_ms_(latency_budget_ms),
      priority_(priority) {}

ObservationBuilder Model::observe() {
    return ObservationBuilder(conn_, conn_->client_id_, conn_->client_type_, model_id_, priority_);
}

std::optional<std::unordered_map<std::string, TensorData>> Model::get_result(int timeout_ms) {
    auto result = conn_->recv(timeout_ms);
    if (!result) {
        return std::nullopt;
    }

    const auto& [envelope, payload] = *result;

    ::inferential::ModelResponse resp;
    if (!resp.ParseFromString(envelope)) {
        return std::nullopt;
    }

    std::unordered_map<std::string, TensorData> output;

    // Fixed fields as special keys
    output["response_id"] = TensorData{
        .key = "response_id",
        .data = std::vector<uint8_t>(resp.response_id().begin(), resp.response_id().end()),
        .shape = {},
        .dtype = DTYPE_UNSPECIFIED,
    };
    output["model_id"] = TensorData{
        .key = "model_id",
        .data = std::vector<uint8_t>(resp.model_id().begin(), resp.model_id().end()),
        .shape = {},
        .dtype = DTYPE_UNSPECIFIED,
    };

    for (const auto& tensor : resp.tensors()) {
        if (tensor.byte_length() > 0 && tensor.dtype() != DTYPE_UNSPECIFIED) {
            TensorData td;
            td.key = tensor.key();
            td.dtype = tensor.dtype();
            td.shape.assign(tensor.shape().begin(), tensor.shape().end());

            auto start = static_cast<size_t>(tensor.byte_offset());
            auto len = static_cast<size_t>(tensor.byte_length());
            if (start + len <= payload.size()) {
                td.data.assign(payload.begin() + start, payload.begin() + start + len);
            }
            output[tensor.key()] = std::move(td);
        }
    }

    for (const auto& [k, v] : resp.metadata()) {
        output[k] = TensorData{
            .key = k,
            .data = std::vector<uint8_t>(v.begin(), v.end()),
            .shape = {},
            .dtype = DTYPE_UNSPECIFIED,
        };
    }

    return output;
}

}  // namespace inferential
