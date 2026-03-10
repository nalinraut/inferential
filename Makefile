.PHONY: proto test lint clean build-cpp test-cpp build-rust test-rust test-all check-sync

# --- Python ---
proto:
	@bash python/scripts/gen_proto.sh

test: proto
	cd python && uv run pytest tests/ -v

lint:
	cd python && uv run ruff check inferential/ tests/

# --- C++ (Bazel) ---
build-cpp:
	bazel build //cpp:inferential

test-cpp:
	bazel test //cpp/tests:test_client

# --- Rust ---
build-rust:
	cd rust && PROTOC=$(CURDIR)/python/.venv/bin/python-grpc-tools-protoc PATH=$(HOME)/.cargo/bin:$$PATH cargo build

test-rust:
	cd rust && PROTOC=$(CURDIR)/python/.venv/bin/python-grpc-tools-protoc PATH=$(HOME)/.cargo/bin:$$PATH cargo test -- --test-threads=1

# --- Sync Check ---
check-sync:
	diff proto/inferential.proto rust/proto/inferential.proto
	diff LICENSE rust/LICENSE

# --- All ---
test-all: check-sync test test-cpp test-rust

# --- Clean ---
clean:
	cd python && find . -type d -name __pycache__ -exec rm -rf {} +
	rm -f python/inferential/proto/inferential_pb2.py python/inferential/proto/inferential_pb2.pyi
	bazel clean 2>/dev/null || true
	cd rust && PATH=$(HOME)/.cargo/bin:$$PATH cargo clean 2>/dev/null || true
