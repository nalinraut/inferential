#include <cstdio>
#include <vector>

#include "inferential/client.hpp"

int main() {
    auto conn = inferential::Connection("tcp://localhost:5555", "cpp-agent-01", "franka");
    auto model = conn.model("policy-v2", 30.0f, 1);

    std::vector<float> joints = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f};

    for (int step = 0; step < 5; ++step) {
        model.observe()
            .urgency(0.8f)
            .steps_remaining(static_cast<uint32_t>(5 - step))
            .tensor_f32("joint_positions", joints.data(), joints.size(), {7})
            .metadata("step", std::to_string(step))
            .send();

        std::printf("[step %d] observation sent\n", step);

        auto result = model.get_result(100);
        if (result) {
            auto& output = *result;
            if (output.count("actions")) {
                auto [ptr, count] = output["actions"].as<float>();
                std::printf("[step %d] got %zu actions\n", step, count);
            }
        } else {
            std::printf("[step %d] timeout (no server)\n", step);
        }
    }

    conn.close();
    std::printf("done\n");
    return 0;
}
