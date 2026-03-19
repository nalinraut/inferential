use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

use prost::Message;

use crate::proto;
use crate::tensor::TensorData;

fn normalize_server(server: &str) -> String {
    if server.starts_with("tcp://") {
        server.to_string()
    } else {
        format!("tcp://{}", server)
    }
}

fn now_ns() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos() as u64
}

/// ZMQ DEALER connection to an Inferential server (synchronous).
pub struct Connection {
    _ctx: zmq::Context,
    socket: zmq::Socket,
    client_id: String,
    client_type: String,
}

impl Connection {
    /// Create a new connection with default reconnection settings.
    pub fn new(server: &str, client_id: &str, client_type: &str) -> Self {
        Self::with_options(server, client_id, client_type, 100, 5000)
    }

    /// Create a new connection with custom reconnection intervals.
    pub fn with_options(
        server: &str,
        client_id: &str,
        client_type: &str,
        reconnect_ivl_ms: i32,
        reconnect_max_ms: i32,
    ) -> Self {
        let ctx = zmq::Context::new();
        let socket = ctx
            .socket(zmq::DEALER)
            .expect("failed to create DEALER socket");

        socket.set_identity(client_id.as_bytes()).unwrap();
        socket.set_reconnect_ivl(reconnect_ivl_ms).unwrap();
        socket.set_reconnect_ivl_max(reconnect_max_ms).unwrap();
        socket.set_rcvhwm(100).unwrap();
        socket.set_sndhwm(100).unwrap();
        socket.set_linger(0).unwrap();
        socket.connect(&normalize_server(server)).unwrap();

        Connection {
            _ctx: ctx,
            socket,
            client_id: client_id.to_string(),
            client_type: client_type.to_string(),
        }
    }

    /// Create a `Model` handle bound to this connection.
    pub fn model(&self, model_id: &str, latency_budget_ms: f32, priority: i32) -> Model<'_> {
        Model {
            conn: self,
            model_id: model_id.to_string(),
            _latency_budget_ms: latency_budget_ms,
            _priority: priority,
        }
    }

    /// Close the connection.
    pub fn close(&self) {
        // Socket is closed on drop; this is a no-op for explicit API parity.
    }

    fn send(&self, envelope: &[u8], payload: &[u8]) {
        self.socket
            .send_multipart([&b""[..], envelope, payload], 0)
            .expect("send failed");
    }

    fn recv(&self, timeout_ms: i32) -> Option<(Vec<u8>, Vec<u8>)> {
        self.socket.set_rcvtimeo(timeout_ms).unwrap();

        let frames = match self.socket.recv_multipart(0) {
            Ok(f) => f,
            Err(zmq::Error::EAGAIN) | Err(zmq::Error::ETERM) => return None,
            Err(e) => panic!("recv error: {}", e),
        };

        if frames.len() < 3 {
            return None;
        }

        // frames[0] = empty, frames[1] = envelope, frames[2] = payload
        Some((frames[1].clone(), frames[2].clone()))
    }
}

impl Drop for Connection {
    fn drop(&mut self) {
        // zmq::Socket and Context clean up on drop.
    }
}

/// Handle to a specific model on the server (synchronous).
pub struct Model<'a> {
    conn: &'a Connection,
    model_id: String,
    _latency_budget_ms: f32,
    _priority: i32,
}

impl<'a> Model<'a> {
    /// Start building an observation to send.
    pub fn observe(&self) -> ObservationBuilder<'_> {
        ObservationBuilder {
            conn: self.conn,
            model_id: &self.model_id,
            urgency: 0.0,
            priority: self._priority as u32,
            steps_remaining: None,
            tensors: Vec::new(),
            metadata: Vec::new(),
        }
    }

    /// Wait for a result. Returns `None` on timeout.
    pub fn get_result(&self, timeout_ms: i32) -> Option<HashMap<String, TensorData>> {
        let (envelope, payload) = self.conn.recv(timeout_ms)?;
        let resp = proto::ModelResponse::decode(envelope.as_slice()).ok()?;

        let mut output = HashMap::new();

        // Store fixed fields as string-typed TensorData
        output.insert(
            "response_id".into(),
            TensorData {
                key: "response_id".into(),
                data: resp.response_id.as_bytes().to_vec(),
                shape: vec![],
                dtype: proto::DType::DtypeUnspecified as i32,
            },
        );
        output.insert(
            "model_id".into(),
            TensorData {
                key: "model_id".into(),
                data: resp.model_id.as_bytes().to_vec(),
                shape: vec![],
                dtype: proto::DType::DtypeUnspecified as i32,
            },
        );

        for tensor in &resp.tensors {
            if tensor.byte_length > 0 && tensor.dtype != proto::DType::DtypeUnspecified as i32 {
                let start = tensor.byte_offset as usize;
                let len = tensor.byte_length as usize;
                let data = if start + len <= payload.len() {
                    payload[start..start + len].to_vec()
                } else {
                    vec![]
                };
                output.insert(
                    tensor.key.clone(),
                    TensorData {
                        key: tensor.key.clone(),
                        data,
                        shape: tensor.shape.clone(),
                        dtype: tensor.dtype,
                    },
                );
            }
        }

        for (k, v) in &resp.metadata {
            output.insert(
                k.clone(),
                TensorData {
                    key: k.clone(),
                    data: v.as_bytes().to_vec(),
                    shape: vec![],
                    dtype: proto::DType::DtypeUnspecified as i32,
                },
            );
        }

        Some(output)
    }
}

/// Fluent builder for constructing and sending an Observation.
pub struct ObservationBuilder<'a> {
    conn: &'a Connection,
    model_id: &'a str,
    urgency: f32,
    priority: u32,
    steps_remaining: Option<u32>,
    tensors: Vec<TensorEntry>,
    metadata: Vec<(String, String)>,
}

struct TensorEntry {
    key: String,
    data: Vec<u8>,
    dtype: i32,
    shape: Vec<i64>,
}

impl<'a> ObservationBuilder<'a> {
    pub fn urgency(mut self, u: f32) -> Self {
        self.urgency = u;
        self
    }

    pub fn priority(mut self, p: u32) -> Self {
        self.priority = p;
        self
    }

    pub fn steps_remaining(mut self, s: u32) -> Self {
        self.steps_remaining = Some(s);
        self
    }

    /// Add a raw tensor.
    pub fn tensor(mut self, key: &str, data: &[u8], dtype: i32, shape: &[i64]) -> Self {
        self.tensors.push(TensorEntry {
            key: key.to_string(),
            data: data.to_vec(),
            dtype,
            shape: shape.to_vec(),
        });
        self
    }

    /// Add a float32 tensor (convenience).
    pub fn tensor_f32(self, key: &str, data: &[f32], shape: &[i64]) -> Self {
        let bytes: &[u8] =
            unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4) };
        self.tensor(key, bytes, proto::DType::Float32 as i32, shape)
    }

    /// Add string metadata.
    pub fn metadata(mut self, key: &str, value: &str) -> Self {
        self.metadata.push((key.to_string(), value.to_string()));
        self
    }

    /// Serialize and send the observation.
    pub fn send(self) {
        let ts = now_ns();

        let mut payload = Vec::new();
        let mut tensor_descs = Vec::new();
        let mut offset: u64 = 0;

        for entry in &self.tensors {
            tensor_descs.push(proto::Tensor {
                key: entry.key.clone(),
                dtype: entry.dtype,
                shape: entry.shape.clone(),
                byte_offset: offset,
                byte_length: entry.data.len() as u64,
                timestamp_ns: ts,
                encoding: proto::Encoding::Raw as i32,
            });
            payload.extend_from_slice(&entry.data);
            offset += entry.data.len() as u64;
        }

        let obs = proto::Observation {
            client: Some(proto::Client {
                id: self.conn.client_id.clone(),
                r#type: self.conn.client_type.clone(),
            }),
            timestamp_ns: ts,
            urgency: self.urgency,
            tensors: tensor_descs,
            model_id: self.model_id.to_string(),
            steps_remaining: self.steps_remaining,
            metadata: self.metadata.into_iter().collect(),
            priority: self.priority,
        };

        let envelope = obs.encode_to_vec();
        self.conn.send(&envelope, &payload);
    }
}
