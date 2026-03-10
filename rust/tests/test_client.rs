use std::thread;
use std::time::Duration;

use prost::Message;

use inferential::proto;
use inferential::Connection;

const TEST_ENDPOINT: &str = "tcp://127.0.0.1:15560";

#[test]
fn test_observe_sends_protobuf() {
    let ctx = zmq::Context::new();
    let router = ctx.socket(zmq::ROUTER).unwrap();
    router.set_linger(0).unwrap();
    router.set_rcvtimeo(2000).unwrap();
    router.bind(TEST_ENDPOINT).unwrap();

    thread::sleep(Duration::from_millis(50));

    let conn = Connection::new(TEST_ENDPOINT, "rust-42", "sensor");
    let model = conn.model("policy-v2", 50.0, 1);

    let joints: Vec<f32> = vec![1.0, 2.0, 3.0];
    model
        .observe()
        .urgency(0.5)
        .tensor_f32("joint_positions", &joints, &[3])
        .send();

    // ROUTER receives: [identity, empty, envelope, payload]
    let frames = router.recv_multipart(0).unwrap();
    assert!(
        frames.len() >= 4,
        "expected 4+ frames, got {}",
        frames.len()
    );

    let obs = proto::Observation::decode(frames[2].as_slice()).unwrap();
    let client = obs.client.unwrap();
    assert_eq!(client.id, "rust-42");
    assert_eq!(client.r#type, "sensor");
    assert_eq!(obs.model_id, "policy-v2");
    assert!((obs.urgency - 0.5).abs() < 1e-6);
    assert!(obs.timestamp_ns > 0);

    assert_eq!(obs.tensors.len(), 1);
    let t = &obs.tensors[0];
    assert_eq!(t.key, "joint_positions");
    assert_eq!(t.dtype, proto::DType::Float32 as i32);
    assert_eq!(t.shape, vec![3]);
    assert_eq!(t.byte_length, 12); // 3 * 4 bytes
    assert_eq!(t.encoding, proto::Encoding::Raw as i32);

    // Verify payload
    let payload = &frames[3];
    assert_eq!(payload.len(), 12);
    let data: &[f32] = unsafe { std::slice::from_raw_parts(payload.as_ptr() as *const f32, 3) };
    assert!((data[0] - 1.0).abs() < 1e-6);
    assert!((data[1] - 2.0).abs() < 1e-6);
    assert!((data[2] - 3.0).abs() < 1e-6);
}

#[test]
fn test_get_result_parses_response() {
    let endpoint = "tcp://127.0.0.1:15561";
    let ctx = zmq::Context::new();
    let router = ctx.socket(zmq::ROUTER).unwrap();
    router.set_linger(0).unwrap();
    router.set_rcvtimeo(2000).unwrap();
    router.bind(endpoint).unwrap();

    thread::sleep(Duration::from_millis(50));

    let conn = Connection::new(endpoint, "rust-43", "robot");
    let model = conn.model("policy-v2", 50.0, 1);

    // Send observation to register identity
    model.observe().urgency(0.1).send();
    let req_frames = router.recv_multipart(0).unwrap();
    assert!(req_frames.len() >= 4);

    // Craft response
    let actions: Vec<f32> = vec![0.1, 0.2, 0.3, 0.4];
    let resp = proto::ModelResponse {
        response_id: "resp-001".into(),
        model_id: "policy-v2".into(),
        inference_latency_ms: 5.2,
        tensors: vec![proto::Tensor {
            key: "actions".into(),
            dtype: proto::DType::Float32 as i32,
            shape: vec![4],
            byte_offset: 0,
            byte_length: 16,
            timestamp_ns: 0,
            encoding: proto::Encoding::Raw as i32,
        }],
        ..Default::default()
    };
    let envelope = resp.encode_to_vec();
    let payload: Vec<u8> = actions.iter().flat_map(|f| f.to_ne_bytes()).collect();

    // Send: [identity, empty, envelope, payload]
    router
        .send_multipart(
            &[req_frames[0].as_slice(), &b""[..], &envelope, &payload],
            0,
        )
        .unwrap();

    let result = model.get_result(2000);
    assert!(result.is_some());
    let output = result.unwrap();

    let td = output.get("actions").expect("missing 'actions'");
    assert_eq!(td.dtype, proto::DType::Float32 as i32);
    assert_eq!(td.shape, vec![4]);

    let values = td.as_f32();
    assert_eq!(values.len(), 4);
    assert!((values[0] - 0.1).abs() < 1e-6);
    assert!((values[1] - 0.2).abs() < 1e-6);
    assert!((values[2] - 0.3).abs() < 1e-6);
    assert!((values[3] - 0.4).abs() < 1e-6);
}

#[test]
fn test_get_result_returns_none_on_timeout() {
    let conn = Connection::new("tcp://127.0.0.1:15599", "rust-44", "");
    let model = conn.model("policy-v2", 50.0, 1);

    let result = model.get_result(50);
    assert!(result.is_none());
}

#[test]
fn test_observe_with_metadata() {
    let endpoint = "tcp://127.0.0.1:15562";
    let ctx = zmq::Context::new();
    let router = ctx.socket(zmq::ROUTER).unwrap();
    router.set_linger(0).unwrap();
    router.set_rcvtimeo(2000).unwrap();
    router.bind(endpoint).unwrap();

    thread::sleep(Duration::from_millis(50));

    let conn = Connection::new(endpoint, "rust-45", "sim");
    let model = conn.model("policy-v2", 50.0, 1);

    model
        .observe()
        .urgency(0.3)
        .metadata("language", "pick up the red cube")
        .metadata("task_id", "task-99")
        .send();

    let frames = router.recv_multipart(0).unwrap();
    assert!(frames.len() >= 4);

    let obs = proto::Observation::decode(frames[2].as_slice()).unwrap();
    assert_eq!(
        obs.metadata.get("language").unwrap(),
        "pick up the red cube"
    );
    assert_eq!(obs.metadata.get("task_id").unwrap(), "task-99");
    assert_eq!(obs.tensors.len(), 0);
}
