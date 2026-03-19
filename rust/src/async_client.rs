use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

use prost::Message;
use zeromq::{DealerSocket, Socket, SocketRecv, SocketSend, ZmqMessage};

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

/// ZMQ DEALER connection to an Inferential server (async, tokio-based).
pub struct AsyncConnection {
    socket: DealerSocket,
    client_id: String,
    client_type: String,
}

impl AsyncConnection {
    /// Create a new async connection.
    pub async fn new(server: &str, client_id: &str, client_type: &str) -> Self {
        let mut socket = DealerSocket::new();
        socket
            .connect(&normalize_server(server))
            .await
            .expect("failed to connect");

        AsyncConnection {
            socket,
            client_id: client_id.to_string(),
            client_type: client_type.to_string(),
        }
    }

    /// Create an `AsyncModel` handle bound to this connection.
    pub fn model(
        &mut self,
        model_id: &str,
        latency_budget_ms: f32,
        priority: i32,
    ) -> AsyncModel<'_> {
        AsyncModel {
            conn: self,
            model_id: model_id.to_string(),
            _latency_budget_ms: latency_budget_ms,
            _priority: priority,
        }
    }

    /// Close the connection, consuming it.
    pub async fn close(self) {
        self.socket.close().await;
    }

    async fn send(&mut self, envelope: &[u8], payload: &[u8]) {
        let mut msg = ZmqMessage::from(Vec::from(&b""[..]));
        msg.push_back(envelope.to_vec().into());
        msg.push_back(payload.to_vec().into());
        self.socket.send(msg).await.expect("send failed");
    }

    async fn recv(&mut self, timeout_ms: i32) -> Option<(Vec<u8>, Vec<u8>)> {
        let timeout = tokio::time::Duration::from_millis(timeout_ms as u64);

        tokio::select! {
            result = self.socket.recv() => {
                match result {
                    Ok(msg) => {
                        let frames: Vec<_> = msg.into_vec();
                        if frames.len() < 3 {
                            return None;
                        }
                        Some((frames[1].to_vec(), frames[2].to_vec()))
                    }
                    Err(_) => None,
                }
            }
            _ = tokio::time::sleep(timeout) => None,
        }
    }
}

/// Handle to a specific model on the server (async).
pub struct AsyncModel<'a> {
    conn: &'a mut AsyncConnection,
    model_id: String,
    _latency_budget_ms: f32,
    _priority: i32,
}

impl<'a> AsyncModel<'a> {
    /// Start building an observation to send.
    pub fn observe(&mut self) -> AsyncObservationBuilder<'_> {
        AsyncObservationBuilder {
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
    pub async fn get_result(&mut self, timeout_ms: i32) -> Option<HashMap<String, TensorData>> {
        let (envelope, payload) = self.conn.recv(timeout_ms).await?;
        let resp = proto::ModelResponse::decode(envelope.as_slice()).ok()?;

        let mut output = HashMap::new();

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

/// Fluent builder for constructing and sending an Observation (async).
pub struct AsyncObservationBuilder<'a> {
    conn: &'a mut AsyncConnection,
    model_id: &'a str,
    urgency: f32,
    priority: u32,
    steps_remaining: Option<u32>,
    tensors: Vec<AsyncTensorEntry>,
    metadata: Vec<(String, String)>,
}

struct AsyncTensorEntry {
    key: String,
    data: Vec<u8>,
    dtype: i32,
    shape: Vec<i64>,
}

impl<'a> AsyncObservationBuilder<'a> {
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

    pub fn tensor(mut self, key: &str, data: &[u8], dtype: i32, shape: &[i64]) -> Self {
        self.tensors.push(AsyncTensorEntry {
            key: key.to_string(),
            data: data.to_vec(),
            dtype,
            shape: shape.to_vec(),
        });
        self
    }

    pub fn tensor_f32(self, key: &str, data: &[f32], shape: &[i64]) -> Self {
        let bytes: &[u8] =
            unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4) };
        self.tensor(key, bytes, proto::DType::Float32 as i32, shape)
    }

    pub fn metadata(mut self, key: &str, value: &str) -> Self {
        self.metadata.push((key.to_string(), value.to_string()));
        self
    }

    pub async fn send(self) {
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
        self.conn.send(&envelope, &payload).await;
    }
}
