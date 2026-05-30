"""Project-level command safety filters for G1 hammer work."""

from __future__ import annotations

import numpy as np


def clamp_joint_limits(q_des, q_min, q_max, margin=0.08):
    q_des = np.asarray(q_des, dtype=float)
    q_min = np.asarray(q_min, dtype=float)
    q_max = np.asarray(q_max, dtype=float)
    return np.clip(q_des, q_min + margin, q_max - margin)


def rate_limit(q_des, q_prev, max_step=0.01):
    q_des = np.asarray(q_des, dtype=float)
    q_prev = np.asarray(q_prev, dtype=float)
    return q_prev + np.clip(q_des - q_prev, -max_step, max_step)


def joint_margin(q, q_min, q_max):
    q = np.asarray(q, dtype=float)
    q_min = np.asarray(q_min, dtype=float)
    q_max = np.asarray(q_max, dtype=float)
    return np.minimum(q - q_min, q_max - q)


def near_joint_limit(q, q_min, q_max, margin=0.06):
    q = np.asarray(q, dtype=float)
    q_min = np.asarray(q_min, dtype=float)
    q_max = np.asarray(q_max, dtype=float)
    return (q < q_min + margin) | (q > q_max - margin)


def filter_position_command(
    q_des,
    q_prev,
    q_min,
    q_max,
    limit_margin=0.08,
    max_step=0.01,
):
    """Apply limit clamp then command rate limiting."""
    clamped = clamp_joint_limits(q_des, q_min, q_max, margin=limit_margin)
    return rate_limit(clamped, q_prev, max_step=max_step)
