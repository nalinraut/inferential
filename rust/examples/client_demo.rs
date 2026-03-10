use inferential::Connection;

fn main() {
    let conn = Connection::new("tcp://localhost:5555", "rust-agent-01", "franka");
    let model = conn.model("policy-v2", 30.0, 1);

    let joints: Vec<f32> = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7];

    for step in 0..5 {
        model
            .observe()
            .urgency(0.8)
            .steps_remaining(5 - step)
            .tensor_f32("joint_positions", &joints, &[7])
            .metadata("step", &step.to_string())
            .send();

        println!("[step {}] observation sent", step);

        match model.get_result(100) {
            Some(output) => {
                if let Some(actions) = output.get("actions") {
                    let values = actions.as_f32();
                    println!("[step {}] got {} actions", step, values.len());
                }
            }
            None => println!("[step {}] timeout (no server)", step),
        }
    }

    println!("done");
}
