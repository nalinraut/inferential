use inferential::Connection;

fn main() {
    let conn = Connection::new("tcp://localhost:5555", "arm-01", "franka");
    let policy = conn.model("manipulation-policy", 30.0, 0);
    let telemetry = conn.model("telemetry", 100.0, 1);

    let joints: Vec<f32> = vec![0.1f32; 14];

    for step in 0..5u32 {
        let urgency = (1.0f32).min((5 - step) as f32 / 5.0);
        policy
            .observe()
            .urgency(urgency)
            .steps_remaining(5 - step)
            .tensor_f32("joint_positions", &joints, &[14])
            .metadata("step", &step.to_string())
            .send();

        println!("[step {}] observation sent", step);

        match policy.get_result(100) {
            Some(output) => {
                if let Some(actions) = output.get("actions") {
                    let values = actions.as_f32();
                    println!("[step {}] got {} actions", step, values.len());
                }
            }
            None => println!("[step {}] timeout (no server)", step),
        }

        if step % 2 == 0 {
            telemetry
                .observe()
                .urgency(0.1)
                .tensor_f32("joint_positions", &joints, &[14])
                .send();
        }
    }

    println!("done");
}
