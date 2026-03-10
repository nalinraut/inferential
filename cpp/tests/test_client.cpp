#include <gtest/gtest.h>

#include <cstring>
#include <thread>
#include <vector>
#include <zmq.hpp>
#include <zmq_addon.hpp>

#include "inferential/client.hpp"
#include "proto/inferential.pb.h"

namespace {

constexpr const char* kTestEndpoint = "tcp://127.0.0.1:15558";

// Helper: run a ROUTER socket that receives one multipart message.
struct RouterFixture {
    zmq::context_t ctx{1};
    zmq::socket_t router{ctx, zmq::socket_type::router};

    RouterFixture() {
        router.set(zmq::sockopt::linger, 0);
        router.set(zmq::sockopt::rcvtimeo, 2000);
        router.bind(kTestEndpoint);
    }

    ~RouterFixture() {
        router.close();
        ctx.close();
    }

    // Receive a full multipart message (identity + empty + envelope + payload).
    // Returns {identity, envelope, payload} or empty vector on timeout.
    std::vector<zmq::message_t> recv_msg() {
        std::vector<zmq::message_t> frames;
        auto result = zmq::recv_multipart(router, std::back_inserter(frames));
        if (!result) return {};
        return frames;
    }
};

// ----- Tests -----

TEST(ClientTest, ObserveSendsProtobuf) {
    RouterFixture server;

    // Small delay for bind to complete
    std::this_thread::sleep_for(std::chrono::milliseconds(50));

    inferential::Connection conn(kTestEndpoint, "test-42", "sensor");
    auto model = conn.model("policy-v2");

    std::vector<float> joints = {1.0f, 2.0f, 3.0f};
    model.observe()
        .urgency(0.5f)
        .tensor_f32("joint_positions", joints.data(), joints.size(), {3})
        .send();

    auto frames = server.recv_msg();
    // ROUTER frames: [identity, empty, envelope, payload]
    ASSERT_GE(frames.size(), 4u);

    ::inferential::Observation obs;
    ASSERT_TRUE(obs.ParseFromArray(frames[2].data(), frames[2].size()));

    EXPECT_EQ(obs.client().id(), "test-42");
    EXPECT_EQ(obs.client().type(), "sensor");
    EXPECT_EQ(obs.model_id(), "policy-v2");
    EXPECT_FLOAT_EQ(obs.urgency(), 0.5f);
    EXPECT_GT(obs.timestamp_ns(), 0u);

    ASSERT_EQ(obs.tensors_size(), 1);
    const auto& t = obs.tensors(0);
    EXPECT_EQ(t.key(), "joint_positions");
    EXPECT_EQ(t.dtype(), ::inferential::FLOAT32);
    ASSERT_EQ(t.shape_size(), 1);
    EXPECT_EQ(t.shape(0), 3);
    EXPECT_EQ(t.byte_length(), 3 * sizeof(float));
    EXPECT_EQ(t.encoding(), ::inferential::RAW);

    // Verify payload bytes
    auto& payload = frames[3];
    ASSERT_EQ(payload.size(), 3 * sizeof(float));
    const float* data = reinterpret_cast<const float*>(payload.data());
    EXPECT_FLOAT_EQ(data[0], 1.0f);
    EXPECT_FLOAT_EQ(data[1], 2.0f);
    EXPECT_FLOAT_EQ(data[2], 3.0f);

    conn.close();
}

TEST(ClientTest, GetResultParsesResponse) {
    RouterFixture server;
    std::this_thread::sleep_for(std::chrono::milliseconds(50));

    inferential::Connection conn(kTestEndpoint, "test-43", "robot");
    auto model = conn.model("policy-v2");

    // Send an observation first (server needs identity)
    model.observe().urgency(0.1f).send();
    auto req_frames = server.recv_msg();
    ASSERT_GE(req_frames.size(), 4u);

    // Craft a ModelResponse
    ::inferential::ModelResponse resp;
    resp.set_response_id("resp-001");
    resp.set_model_id("policy-v2");
    resp.set_inference_latency_ms(5.2f);

    std::vector<float> actions = {0.1f, 0.2f, 0.3f, 0.4f};
    auto* t = resp.add_tensors();
    t->set_key("actions");
    t->set_dtype(::inferential::FLOAT32);
    t->add_shape(4);
    t->set_byte_offset(0);
    t->set_byte_length(actions.size() * sizeof(float));

    std::string envelope;
    resp.SerializeToString(&envelope);
    std::string payload(reinterpret_cast<const char*>(actions.data()),
                        actions.size() * sizeof(float));

    // Send back: [identity, empty, envelope, payload]
    zmq::message_t identity(req_frames[0].data(), req_frames[0].size());
    server.router.send(identity, zmq::send_flags::sndmore);
    server.router.send(zmq::buffer(""), zmq::send_flags::sndmore);
    server.router.send(zmq::buffer(envelope), zmq::send_flags::sndmore);
    server.router.send(zmq::buffer(payload), zmq::send_flags::none);

    auto result = model.get_result(2000);
    ASSERT_TRUE(result.has_value());

    auto& output = *result;
    ASSERT_TRUE(output.count("actions"));
    auto& td = output["actions"];
    EXPECT_EQ(td.dtype, ::inferential::FLOAT32);
    ASSERT_EQ(td.shape.size(), 1u);
    EXPECT_EQ(td.shape[0], 4);

    auto [ptr, count] = td.as<float>();
    ASSERT_EQ(count, 4u);
    EXPECT_FLOAT_EQ(ptr[0], 0.1f);
    EXPECT_FLOAT_EQ(ptr[1], 0.2f);
    EXPECT_FLOAT_EQ(ptr[2], 0.3f);
    EXPECT_FLOAT_EQ(ptr[3], 0.4f);

    conn.close();
}

TEST(ClientTest, GetResultReturnsNulloptOnTimeout) {
    // Connect to a port where nothing is listening
    inferential::Connection conn("tcp://127.0.0.1:15599", "test-44");
    auto model = conn.model("policy-v2");

    auto result = model.get_result(50);
    EXPECT_FALSE(result.has_value());

    conn.close();
}

TEST(ClientTest, ObserveWithMetadata) {
    RouterFixture server;
    std::this_thread::sleep_for(std::chrono::milliseconds(50));

    inferential::Connection conn(kTestEndpoint, "test-45", "sim");
    auto model = conn.model("policy-v2");

    model.observe()
        .urgency(0.3f)
        .metadata("language", "pick up the red cube")
        .metadata("task_id", "task-99")
        .send();

    auto frames = server.recv_msg();
    ASSERT_GE(frames.size(), 4u);

    ::inferential::Observation obs;
    ASSERT_TRUE(obs.ParseFromArray(frames[2].data(), frames[2].size()));

    EXPECT_EQ(obs.metadata().at("language"), "pick up the red cube");
    EXPECT_EQ(obs.metadata().at("task_id"), "task-99");
    EXPECT_EQ(obs.tensors_size(), 0);

    conn.close();
}

}  // namespace
