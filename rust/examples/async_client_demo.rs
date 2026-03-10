use inferential::AsyncConnection;

#[tokio::main]
async fn main() {
    let mut conn = AsyncConnection::new("tcp://localhost:5555", "rust-async-01", "franka").await;

    let joints: Vec<f32> = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7];

    {
        let mut model = conn.model("policy-v2", 30.0, 1);

        for step in 0..5u32 {
            model
                .observe()
                .urgency(0.8)
                .steps_remaining(5 - step)
                .tensor_f32("joint_positions", &joints, &[7])
                .metadata("step", &step.to_string())
                .send()
                .await;

            println!("[step {}] observation sent", step);

            match model.get_result(100).await {
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
