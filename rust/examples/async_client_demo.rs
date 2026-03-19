use inferential::AsyncConnection;

#[tokio::main]
async fn main() {
    let mut conn = AsyncConnection::new("tcp://localhost:5555", "arm-01", "franka").await;

    let joints: Vec<f32> = vec![0.1f32; 14];

    {
        let mut policy = conn.model("manipulation-policy", 30.0, 0);

        for step in 0..5u32 {
            let urgency = (1.0f32).min((5 - step) as f32 / 5.0);
            policy
                .observe()
                .urgency(urgency)
                .steps_remaining(5 - step)
                .tensor_f32("joint_positions", &joints, &[14])
                .metadata("step", &step.to_string())
                .send()
                .await;

            println!("[step {}] observation sent", step);

            match policy.get_result(100).await {
                Some(output) => {
                    if let Some(actions) = output.get("actions") {
                        let values = actions.as_f32();
                        println!("[step {}] got {} actions", step, values.len());
                    }
                }
                None => println!("[step {}] timeout (no server)", step),
            }
        }
    }

    conn.close().await;
    println!("done");
}
