"""Hatchling build hook to generate protobuf code at build time."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


class ProtobufBuildHook(BuildHookInterface):
    PLUGIN_NAME = "protobuf"

    def initialize(self, version: str, build_data: dict) -> None:
        repo_root = Path(self.root).parent
        proto_src = repo_root / "proto"
        proto_out = Path(self.root) / "inferential" / "proto"
        proto_file = proto_src / "inferential.proto"

        if not proto_file.exists():
            return

        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "grpc_tools.protoc",
                f"--proto_path={proto_src}",
                f"--python_out={proto_out}",
                f"--pyi_out={proto_out}",
                str(proto_file),
            ]
        )
