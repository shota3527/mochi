"""Shared command-state definitions."""

from __future__ import annotations

from enum import Enum


class ControllerState(str, Enum):
    READ_STATE = "read_state"
    READY = "ready"
    TEST_SINGLE_JOINT = "test_single_joint"
    HOLD = "hold"
    FAULT = "fault"
