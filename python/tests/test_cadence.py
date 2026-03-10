from __future__ import annotations

import time
from unittest.mock import patch

from inferential.tracking.cadence import CadenceTracker


def test_initial_state():
    tracker = CadenceTracker()
    assert tracker.learned_cadence("robot-1") is None
    assert tracker.confidence("robot-1") == 0.0
    assert tracker.time_since_last("robot-1") == float("inf")


def test_single_request_no_cadence():
    tracker = CadenceTracker()
    tracker.on_request("robot-1")
    # After one request, no interval yet
    assert tracker.learned_cadence("robot-1") is None
    assert tracker.confidence("robot-1") == 0.0


def test_cadence_learning():
    tracker = CadenceTracker(alpha=0.5)

    with patch.object(time, "monotonic", return_value=0.0):
        tracker.on_request("robot-1")

    with patch.object(time, "monotonic", return_value=0.1):
        tracker.on_request("robot-1")

    cadence = tracker.learned_cadence("robot-1")
    assert cadence is not None
    assert abs(cadence - 0.1) < 0.01


def test_ewma_convergence():
    tracker = CadenceTracker(alpha=0.3)
    t = 0.0

    for _ in range(20):
        with patch.object(time, "monotonic", return_value=t):
            tracker.on_request("robot-1")
        t += 0.05  # 50ms cadence

    cadence = tracker.learned_cadence("robot-1")
    assert cadence is not None
    assert abs(cadence - 0.05) < 0.005  # Should converge close to 50ms


def test_confidence_increases():
    tracker = CadenceTracker()
    t = 0.0

    for i in range(12):
        with patch.object(time, "monotonic", return_value=t):
            tracker.on_request("robot-1")
        t += 0.05

    # After 11 intervals (12 requests), confidence should be 1.0
    assert tracker.confidence("robot-1") == 1.0


def test_time_until_next():
    tracker = CadenceTracker(alpha=1.0)  # alpha=1.0 so cadence = last interval

    with patch.object(time, "monotonic", return_value=0.0):
        tracker.on_request("robot-1")

    with patch.object(time, "monotonic", return_value=0.1):
        tracker.on_request("robot-1")

    # Now cadence = 0.1. If 0.05s have passed, time_until_next = 0.05
    with patch.object(time, "monotonic", return_value=0.15):
        t = tracker.time_until_next("robot-1")
        assert abs(t - 0.05) < 0.01


def test_is_overdue():
    tracker = CadenceTracker(alpha=1.0, overdue_multiplier=1.5)

    with patch.object(time, "monotonic", return_value=0.0):
        tracker.on_request("robot-1")

    with patch.object(time, "monotonic", return_value=0.1):
        tracker.on_request("robot-1")

    # cadence = 0.1. Overdue if > 0.15
    with patch.object(time, "monotonic", return_value=0.20):
        assert not tracker.is_overdue("robot-1")

    with patch.object(time, "monotonic", return_value=0.26):
        assert tracker.is_overdue("robot-1")


def test_multiple_clients():
    tracker = CadenceTracker()

    with patch.object(time, "monotonic", return_value=0.0):
        tracker.on_request("robot-1")
        tracker.on_request("robot-2")

    with patch.object(time, "monotonic", return_value=0.1):
        tracker.on_request("robot-1")

    with patch.object(time, "monotonic", return_value=0.2):
        tracker.on_request("robot-2")

    assert tracker.learned_cadence("robot-1") is not None
    assert tracker.learned_cadence("robot-2") is not None
    # robot-1 has 0.1 cadence, robot-2 has 0.2 cadence
    assert tracker.learned_cadence("robot-1") < tracker.learned_cadence("robot-2")
