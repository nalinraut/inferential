from __future__ import annotations

from pathlib import Path

import numpy as np


def _ensure_generated() -> None:
    pb2_path = Path(__file__).parent / "inferential_pb2.py"
    if not pb2_path.exists():
        raise ImportError(
            "inferential_pb2.py not found. Run 'make proto' or 'bash scripts/gen_proto.sh' first."
        )


_ensure_generated()

from inferential.proto.inferential_pb2 import (  # noqa: E402
    BFLOAT16,
    BOOL,
    DTYPE_UNSPECIFIED,
    ENCODING_UNSPECIFIED,
    FLOAT16,
    FLOAT32,
    FLOAT64,
    INT32,
    INT64,
    JPEG,
    PNG,
    RAW,
    UINT8,
    Client,
    DType,
    Encoding,
    ModelResponse,
    Observation,
    Tensor,
)

__all__ = [
    "BFLOAT16",
    "BOOL",
    "Client",
    "DTYPE_UNSPECIFIED",
    "ENCODING_UNSPECIFIED",
    "FLOAT16",
    "FLOAT32",
    "FLOAT64",
    "INT32",
    "INT64",
    "JPEG",
    "ModelResponse",
    "Observation",
    "PNG",
    "RAW",
    "Tensor",
    "UINT8",
    "dtype_from_numpy",
    "dtype_to_numpy",
    "encoding_from_string",
    "encoding_to_string",
]

# --- DType <-> numpy mapping ---

_DTYPE_TO_NUMPY: dict[DType, np.dtype] = {
    FLOAT16: np.dtype("float16"),
    FLOAT32: np.dtype("float32"),
    FLOAT64: np.dtype("float64"),
    BFLOAT16: np.dtype("uint16"),  # bfloat16 has no numpy equivalent; raw bytes preserved
    UINT8: np.dtype("uint8"),
    INT32: np.dtype("int32"),
    INT64: np.dtype("int64"),
    BOOL: np.dtype("bool"),
}

_NUMPY_TO_DTYPE: dict[str, DType] = {
    "float16": FLOAT16,
    "float32": FLOAT32,
    "float64": FLOAT64,
    "uint8": UINT8,
    "int32": INT32,
    "int64": INT64,
    "bool": BOOL,
}


def dtype_to_numpy(dtype_enum: DType) -> np.dtype:
    """Convert a protobuf DType enum value to a numpy dtype."""
    if dtype_enum in _DTYPE_TO_NUMPY:
        return _DTYPE_TO_NUMPY[dtype_enum]
    raise ValueError(f"Unknown DType enum value: {dtype_enum}")


def dtype_from_numpy(np_dtype: np.dtype) -> DType:
    """Convert a numpy dtype to a protobuf DType enum value."""
    key = np_dtype.name
    if key in _NUMPY_TO_DTYPE:
        return _NUMPY_TO_DTYPE[key]
    raise ValueError(f"No DType mapping for numpy dtype: {np_dtype}")


# --- Encoding <-> string mapping ---

_ENCODING_TO_STRING: dict[Encoding, str] = {
    RAW: "raw",
    JPEG: "jpeg",
    PNG: "png",
}

_STRING_TO_ENCODING: dict[str, Encoding] = {v: k for k, v in _ENCODING_TO_STRING.items()}


def encoding_to_string(encoding_enum: Encoding) -> str:
    """Convert a protobuf Encoding enum value to a string."""
    if encoding_enum in _ENCODING_TO_STRING:
        return _ENCODING_TO_STRING[encoding_enum]
    raise ValueError(f"Unknown Encoding enum value: {encoding_enum}")


def encoding_from_string(encoding_str: str) -> Encoding:
    """Convert an encoding string to a protobuf Encoding enum value."""
    key = encoding_str.lower()
    if key in _STRING_TO_ENCODING:
        return _STRING_TO_ENCODING[key]
    raise ValueError(f"Unknown encoding string: {encoding_str}")
