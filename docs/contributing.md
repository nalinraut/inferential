# Contributing

## Setup

```bash
git clone https://github.com/nalinraut/inferential.git
cd inferential
```

### Python

```bash
cd python
pip install uv
uv sync --extra dev
make proto          # generate protobuf code
pre-commit install  # install git hooks
```

### C++ (Bazel)

```bash
# Install Bazelisk (manages Bazel versions)
# The project pins Bazel 7.x via .bazelversion
bazel build //cpp:inferential
bazel test //cpp/tests:test_client
```

### Rust

```bash
cd rust
# Requires libzmq-dev and protobuf-compiler
cargo build
cargo test -- --test-threads=1
```

## Pre-commit Hooks (Python)

The following run automatically on every commit:

| Hook | What it does |
|------|-------------|
| **Ruff lint** | Catches errors, unused imports, style violations (`--fix` applied automatically) |
| **Ruff format** | Enforces consistent code formatting |
| **Mypy** | Static type checking with `--ignore-missing-imports` |
| **Trailing whitespace** | Removes trailing whitespace from all files |
| **End-of-file fixer** | Ensures files end with a single newline |
| **check-yaml** | Validates YAML syntax |
| **check-toml** | Validates TOML syntax |

If a hook fails, it will either auto-fix the file (ruff lint/format, whitespace) or print the error (mypy). Stage the fixes and commit again.

## Commit Messages

We follow [Conventional Commits](https://www.conventionalcommits.org/).

### Format

```
<type>(<scope>): <subject>

[optional body]

[optional footer]
```

### Types

| Type | When to use |
|------|-------------|
| `feat` | New feature or capability |
| `fix` | Bug fix |
| `refactor` | Code change that neither fixes a bug nor adds a feature |
| `perf` | Performance improvement |
| `docs` | Documentation only |
| `test` | Adding or updating tests |
| `ci` | CI/CD pipeline changes (workflows, pre-commit) |
| `build` | Build system or dependency changes |
| `chore` | Maintenance tasks that don't fit above |

### Scopes

Use the module or area being changed:

| Scope | Area |
|-------|------|
| `scheduler` | Scheduler strategies, base class, policies |
| `dispatch` | Dispatcher, health tracking |
| `transport` | ZMQ transport layer |
| `proto` | Protobuf schema, codegen, dtype/encoding mappings |
| `client` | Client SDK (any language — specify in description) |
| `client-py` | Python client specifically |
| `client-cpp` | C++ client specifically |
| `client-rs` | Rust client specifically |
| `server` | Server orchestration loop |
| `metrics` | Metrics pipeline, ring buffer |
| `tracking` | Cadence tracker, response tracker |
| `examples` | Example scripts |
| `deps` | Dependency changes |

Scope is optional but encouraged for non-trivial changes.

### Subject line rules

- Imperative mood: "add feature" not "added feature"
- Lowercase first letter
- No period at the end
- Max 72 characters

### Examples

```
feat(client-rs): add async client with tokio support

fix(dispatch): handle non-numpy model responses with np.asarray

refactor(proto): use DType/Encoding types instead of raw int

docs: restructure documentation for multi-language SDKs

test(client-cpp): add metadata roundtrip test

ci: add C++ and Rust test jobs to CI workflow

build(client-rs): add zeromq and tokio dependencies
```

### Breaking changes

Add `!` after the type/scope and explain in the footer:

```
feat(proto)!: rename Observation.metadata to Observation.labels

BREAKING CHANGE: Observation.metadata field renamed to labels.
Existing serialized messages are incompatible.
```

## Branching

| Branch | Purpose |
|--------|---------|
| `main` | Stable, release-ready code |
| `feat/<name>` | New features |
| `fix/<name>` | Bug fixes |
| `refactor/<name>` | Refactoring |
| `docs/<name>` | Documentation |

Always branch from `main`. Keep branches short-lived.

## Pull Requests

1. One logical change per PR
2. PR title follows the same conventional commit format
3. Ensure CI passes (Python lint + tests, C++ tests, Rust tests)
4. Squash merge into `main`

## Code Style

### Python

- **Python 3.11+** — use `X | Y` union syntax, not `Union[X, Y]`
- **Line length**: 100 characters (configured in `pyproject.toml`)
- **Imports**: sorted by ruff (isort-compatible)
- **Type hints**: required on all function signatures
- **Generated files**: `inferential_pb2.py` / `.pyi` are excluded from linting

### C++

- **C++17** (configured in `.bazelrc`)
- **Formatter**: `clang-format` (Google style, config in `.clang-format`)
- **Namespace**: `inferential::`
- **Naming**: `snake_case` for functions/variables, `PascalCase` for classes/structs
- **Headers**: `#pragma once` guards

Format before committing:

```bash
find cpp -name '*.cpp' -o -name '*.hpp' | xargs clang-format -i
```

### Rust

- **Rust 1.70+**
- **Formatter**: `rustfmt` (run `cargo fmt`)
- Run `cargo clippy` before submitting
- `unsafe` is acceptable for zero-copy tensor reinterpretation

## Running Checks

```bash
# Python
cd python
uv run ruff check inferential/
uv run ruff format --check inferential/ tests/
uv run mypy inferential/ --ignore-missing-imports
uv run pytest tests/

# C++
find cpp -name '*.cpp' -o -name '*.hpp' | xargs clang-format --dry-run -Werror
bazel test //cpp/tests:test_client

# Rust
cd rust
cargo fmt --check
cargo clippy -- -D warnings
cargo test -- --test-threads=1

# Everything
make test-all
```

## Releases

We use [Semantic Versioning](https://semver.org/):

- **MAJOR**: Breaking changes to public API or wire protocol
- **MINOR**: New features, backward-compatible
- **PATCH**: Bug fixes, backward-compatible

### Pre-release Checklist

1. Update version numbers in all three places:
   - `python/pyproject.toml`
   - `rust/Cargo.toml`
   - `MODULE.bazel`
2. Sync vendored files:
   ```bash
   cp proto/inferential.proto rust/proto/inferential.proto
   cp LICENSE rust/LICENSE
   make check-sync  # verify they match
   ```
3. Verify Rust packaging: `cd rust && cargo publish --dry-run`
4. Commit, tag, and push:
   ```bash
   git tag v1.0.1
   git push origin v1.0.1
   ```

### Automated Publishes

Pushing a `v*` tag triggers Python and Rust publishes. Creating a **GitHub Release** additionally triggers the BCR submission.

| Package | Workflow | Trigger | Registry |
|---------|----------|---------|----------|
| Python | `publish.yml` | `v*` tag | [PyPI](https://pypi.org/project/inferential/) |
| Rust | `publish-rust.yml` | `v*` tag | [crates.io](https://crates.io/crates/inferential) |
| C++ | `publish-bcr.yml` | GitHub Release | [Bazel Central Registry](https://registry.bazel.build/) |

**Required secrets**: `PYPI_API_TOKEN`, `CRATES_API_TOKEN`, `BCR_PUBLISH_TOKEN` (Classic PAT with `public_repo` scope)
