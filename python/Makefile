.PHONY: proto test lint clean

proto:
	@bash scripts/gen_proto.sh

test: proto
	uv run pytest tests/ -v

lint:
	uv run ruff check inferential/ tests/

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	rm -f inferential/proto/inferential_pb2.py inferential/proto/inferential_pb2.pyi
