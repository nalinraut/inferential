use std::thread;
use std::time::Duration;

use prost::Message;

use inferential::proto;
use inferential::AsyncConnection;

#[tokio::test]
async fn test_async_observe_sends_protobuf() {
    let ctx = zmq::Context::new();
    let router = ctx.socket(zmq::ROUTER).unwrap();
    router.set_linger(0).unwrap();
    router.set_rcvtimeo(2000).unwrap();
    let endpoint = "tcp://127.0.0.1:15563";
    router.bind(endpoint).unwrap();

    thread::sleep(Duration::from_millis(100));

    let mut conn = AsyncConnection::new(endpoint, "async-42", "sensor").await;
    let mut model = conn.model("policy-v2", 50.0, 1);

    let joints: Vec<f32> = vec![1.0, 2.0, 3.0];
    model
        .observe()
        .urgency(0.5)
        .tensor_f32("joint_positions", &joints, &[3])
        .send()
        .await;

    // Small delay for message delivery
    thread::sleep(Duration::from_millis(100));

    let frames = router.recv_multipart(0).unwrap();
    assert!(
        frames.len() >= 4,
        "expected 4+ frames, got {}",
        frames.len()
    );

    let obs = proto::Observation::decode(frames[2].as_slice()).unwrap();
    let client = obs.client.unwrap();
    assert_eq!(client.id, "async-42");
    assert_eq!(obs.model_id, "policy-v2");
    assert!((obs.urgency - 0.5).abs() < 1e-6);
    assert_eq!(obs.tensors.len(), 1);
    assert_eq!(obs.tensors[0].key, "joint_positions");
}

#[tokio::test]
async fn test_async_get_result_returns_none_on_timeout() {
    // Bind a ROUTER but never send a response — tests timeout behavior.
    let ctx = zmq::Context::new();
    let router = ctx.socket(zmq::ROUTER).unwrap();
    router.set_linger(0).unwrap();
    let endpoint = "tcp://127.0.0.1:15565";
    router.bind(endpoint).unwrap();

    thread::sleep(Duration::from_millis(100));

    let mut conn = AsyncConnection::new(endpoint, "async-44", "").await;
    let mut model = conn.model("policy-v2", 50.0, 1);

    let result = model.get_result(100).await;
    assert!(result.is_none());
}

#[tokio::test]
async fn test_async_observe_with_metadata() {
    let ctx = zmq::Context::new();
    let router = ctx.socket(zmq::ROUTER).unwrap();
    router.set_linger(0).unwrap();
    router.set_rcvtimeo(2000).unwrap();
    let endpoint = "tcp://127.0.0.1:15564";
    router.bind(endpoint).unwrap();

    thread::sleep(Duration::from_millis(100));

    let mut conn = AsyncConnection::new(endpoint, "async-45", "sim").await;
    let mut model = conn.model("policy-v2", 50.0, 1);

    model
        .observe()
        .urgency(0.3)
        .metadata("language", "pick up the red cube")
        .send()
        .await;

    thread::sleep(Duration::from_millis(100));

    let frames = router.recv_multipart(0).unwrap();
    assert!(frames.len() >= 4);

    let obs = proto::Observation::decode(frames[2].as_slice()).unwrap();
    assert_eq!(
        obs.metadata.get("language").unwrap(),
        "pick up the red cube"
    );
}
