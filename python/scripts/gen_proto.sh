#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PROTO_SRC="$REPO_ROOT/proto"
PROTO_OUT="$REPO_ROOT/python/inferential/proto"

"$REPO_ROOT/python/.venv/bin/python" -m grpc_tools.protoc \
    --proto_path="$PROTO_SRC" \
    --python_out="$PROTO_OUT" \
    --pyi_out="$PROTO_OUT" \
    "$PROTO_SRC/inferential.proto"

echo "Generated inferential_pb2.py"
