#include <cstdio>
#include <vector>

#include "inferential/client.hpp"

int main() {
    auto conn = inferential::Connection("tcp://localhost:5555", "arm-01", "franka");
    auto policy = conn.model("manipulation-policy", 30.0f, 0);
    auto telemetry = conn.model("telemetry", 100.0f, 1);

    std::vector<float> joints(14, 0.1f);

    for (int step = 0; step < 5; ++step) {
        float urgency = std::min(1.0f, static_cast<float>(5 - step) / 5.0f);
        policy.observe()
            .urgency(urgency)
            .steps_remaining(static_cast<uint32_t>(5 - step))
            .tensor_f32("joint_positions", joints.data(), joints.size(), {14})
            .metadata("step", std::to_string(step))
            .send();

        std::printf("[step %d] observation sent\n", step);

        auto result = policy.get_result(100);
        if (result) {
            auto& output = *result;
            if (output.count("actions")) {
                auto [ptr, count] = output["actions"].as<float>();
                std::printf("[step %d] got %zu actions\n", step, count);
            }
        } else {
            std::printf("[step %d] timeout (no server)\n", step);
        }

        if (step % 2 == 0) {
            telemetry.observe()
                .urgency(0.1f)
                .tensor_f32("joint_positions", joints.data(), joints.size(), {14})
                .send();
        }
    }

    conn.close();
    std::printf("done\n");
    return 0;
}
