from __future__ import annotations

import pytest

from inferential.config.schema import InferentialConfig, SchedulingConfig


@pytest.fixture
def config() -> InferentialConfig:
    return InferentialConfig()


@pytest.fixture
def scheduling_config() -> SchedulingConfig:
    return SchedulingConfig()
