use crate::proto;

/// Holds tensor data received from the server.
#[derive(Debug, Clone)]
pub struct TensorData {
    pub key: String,
    pub data: Vec<u8>,
    pub shape: Vec<i64>,
    pub dtype: i32,
}

impl TensorData {
    /// Bytes per element for the tensor's dtype.
    pub fn element_size(&self) -> usize {
        match self.dtype {
            x if x == proto::DType::Float16 as i32 => 2,
            x if x == proto::DType::Bfloat16 as i32 => 2,
            x if x == proto::DType::Float32 as i32 => 4,
            x if x == proto::DType::Int32 as i32 => 4,
            x if x == proto::DType::Float64 as i32 => 8,
            x if x == proto::DType::Int64 as i32 => 8,
            x if x == proto::DType::Uint8 as i32 => 1,
            x if x == proto::DType::Bool as i32 => 1,
            _ => 0,
        }
    }

    /// Total number of elements.
    pub fn numel(&self) -> usize {
        let es = self.element_size();
        if es == 0 {
            return 0;
        }
        self.data.len() / es
    }

    /// Reinterpret raw bytes as `&[f32]`.
    pub fn as_f32(&self) -> &[f32] {
        assert_eq!(self.element_size(), 4, "dtype is not 4 bytes");
        let ptr = self.data.as_ptr() as *const f32;
        unsafe { std::slice::from_raw_parts(ptr, self.data.len() / 4) }
    }

    /// Reinterpret raw bytes as `&[f64]`.
    pub fn as_f64(&self) -> &[f64] {
        assert_eq!(self.element_size(), 8, "dtype is not 8 bytes");
        let ptr = self.data.as_ptr() as *const f64;
        unsafe { std::slice::from_raw_parts(ptr, self.data.len() / 8) }
    }

    /// Reinterpret raw bytes as `&[i32]`.
    pub fn as_i32(&self) -> &[i32] {
        assert_eq!(self.element_size(), 4, "dtype is not 4 bytes");
        let ptr = self.data.as_ptr() as *const i32;
        unsafe { std::slice::from_raw_parts(ptr, self.data.len() / 4) }
    }

    /// Reinterpret raw bytes as `&[i64]`.
    pub fn as_i64(&self) -> &[i64] {
        assert_eq!(self.element_size(), 8, "dtype is not 8 bytes");
        let ptr = self.data.as_ptr() as *const i64;
        unsafe { std::slice::from_raw_parts(ptr, self.data.len() / 8) }
    }

    /// Raw bytes as `&[u8]`.
    pub fn as_u8(&self) -> &[u8] {
        &self.data
    }
}
