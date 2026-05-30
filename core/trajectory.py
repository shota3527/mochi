"""Trajectory helpers shared by simulation and real backends."""

from __future__ import annotations

import numpy as np


def interpolate_joint(q_start, q_end, alpha: float):
    alpha = float(np.clip(alpha, 0.0, 1.0))
    return np.asarray(q_start, dtype=float) + alpha * (
        np.asarray(q_end, dtype=float) - np.asarray(q_start, dtype=float)
    )
