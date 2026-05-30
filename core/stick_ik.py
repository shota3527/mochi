"""Stick-pose to two-hand IK helpers for G1 hammer/stick inspection poses."""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np
from scipy.optimize import least_squares


LEFT_ARM_JOINTS = (
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
)
RIGHT_ARM_JOINTS = (
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)


@dataclass(frozen=True)
class StickTarget:
    """World-space wood-stick centerline target.

    `root_m` is the rear end of the wood stick. For the deadlocked
    two-hand setup, the right clamp center grips this rear end and the stick
    extends forward along `axis_m`.
    """

    root_m: np.ndarray
    axis_m: np.ndarray
    length_m: float = 0.60

    def point_at(self, distance_m: float) -> np.ndarray:
        return self.root_m + self.axis_m * float(distance_m)


@dataclass(frozen=True)
class DualGripTargets:
    """World-space targets derived from a stick target."""

    right_grip_m: np.ndarray
    left_grip_m: np.ndarray
    stick_axis_m: np.ndarray
    right_cross_axis_m: np.ndarray
    left_cross_axis_m: np.ndarray
    right_grip_distance_m: float
    left_grip_distance_m: float
    clamp_center_offset_m: float


@dataclass(frozen=True)
class DualHoldIKResult:
    joint_values: dict[str, float]
    cost: float
    right_tool_error_m: float
    right_grip_error_m: float
    left_grip_error_m: float
    right_axis_error: float
    left_axis_error: float
    right_cross_axis_error: float
    left_cross_axis_error: float
    targets: DualGripTargets


@dataclass(frozen=True)
class ClosedLoopIKResult:
    joint_values: dict[str, float]
    cost: float
    loop_grip_error_m: float
    loop_axis_error: float
    head_axis_error: float
    hand_center_y_error_m: float
    right_grip_m: np.ndarray
    left_grip_m: np.ndarray
    right_axis_m: np.ndarray
    left_axis_m: np.ndarray
    head_axis_m: np.ndarray


@dataclass(frozen=True)
class IKWeights:
    """Weights for strict deadlocked two-hand stick IK."""

    right_grip: float = 100.0
    left_grip: float = 100.0
    right_axis: float = 30.0
    left_axis: float = 30.0
    right_cross_axis: float = 30.0
    left_cross_axis: float = 30.0
    elbow_clearance: float = 20.0
    regularization: float = 0.01


def normalize(v) -> np.ndarray:
    arr = np.asarray(v, dtype=float)
    norm = np.linalg.norm(arr)
    if norm <= 1e-9:
        raise ValueError("Cannot normalize a near-zero vector.")
    return arr / norm


def stick_axis_from_pitch(pitch_rad: float) -> np.ndarray:
    """Return a forward/up stick axis in the robot sagittal plane."""
    return normalize([np.cos(pitch_rad), 0.0, np.sin(pitch_rad)])


def cross_axis_for_stick(axis_m, preferred=(0.0, -1.0, 0.0)) -> np.ndarray:
    """Return a unit cross axis perpendicular to the stick axis.

    This locks roll around the wood stick. The default keeps the hammer head
    sideways in world -Y while the stick swings in the sagittal X/Z plane.
    """
    axis = normalize(axis_m)
    preferred = normalize(preferred)
    cross = preferred - axis * np.dot(preferred, axis)
    if np.linalg.norm(cross) <= 1e-9:
        cross = np.cross(axis, [0.0, 0.0, 1.0])
    return normalize(cross)


def cross_axis_from_roll_phase(axis_m, roll_phase_rad: float) -> np.ndarray:
    """Return a cross axis with fixed roll phase around a sagittal stick.

    Phase 0 points toward world +Y. Positive phase rotates toward
    `axis x +Y`. Keeping this phase constant locks hammer/clamp roll while the
    stick pitches forward/back.
    """
    axis = normalize(axis_m)
    side = np.array([0.0, 1.0, 0.0])
    tangent = normalize(np.cross(axis, side))
    return np.cos(roll_phase_rad) * side + np.sin(roll_phase_rad) * tangent


def make_centered_stick_target(
    root_m=(0.12, 0.0, 0.69),
    pitch_rad: float = 0.44,
    length_m: float = 0.60,
) -> StickTarget:
    """Create a sagittal-plane stick target from rear-end position and pitch."""
    return StickTarget(
        root_m=np.asarray(root_m, dtype=float),
        axis_m=stick_axis_from_pitch(pitch_rad),
        length_m=float(length_m),
    )


def grips_from_stick(
    stick: StickTarget,
    right_grip_distance_m: float = 0.0,
    left_grip_distance_m: float = 0.24,
    clamp_center_offset_m: float = 0.0,
    cross_axis_m=None,
    right_cross_axis_m=None,
    left_cross_axis_m=None,
) -> DualGripTargets:
    """Compute clamp-center targets from a stick centerline.

    `right_grip_distance_m` defaults to 0 because the right clamp/tool root is
    the rear end of the wood stick. Both clamp centers must lie on the same
    centerline and align their local +Z axis with the stick axis.
    """
    shared_cross = cross_axis_for_stick(stick.axis_m) if cross_axis_m is None else normalize(cross_axis_m)
    return DualGripTargets(
        right_grip_m=stick.point_at(right_grip_distance_m),
        left_grip_m=stick.point_at(left_grip_distance_m),
        stick_axis_m=stick.axis_m.copy(),
        right_cross_axis_m=shared_cross if right_cross_axis_m is None else normalize(right_cross_axis_m),
        left_cross_axis_m=shared_cross if left_cross_axis_m is None else normalize(left_cross_axis_m),
        right_grip_distance_m=float(right_grip_distance_m),
        left_grip_distance_m=float(left_grip_distance_m),
        clamp_center_offset_m=float(clamp_center_offset_m),
    )


def joint_qpos_addrs(model, joint_names) -> np.ndarray:
    return np.array([model.jnt_qposadr[model.joint(name).id] for name in joint_names], dtype=int)


def joint_ranges(model, joint_names) -> tuple[np.ndarray, np.ndarray]:
    ranges = np.array([model.jnt_range[model.joint(name).id] for name in joint_names], dtype=float)
    return ranges[:, 0], ranges[:, 1]


def last_geom_on_body(model, body_name: str) -> int:
    body_id = model.body(body_name).id
    geom_ids = [i for i in range(model.ngeom) if model.geom_bodyid[i] == body_id]
    if not geom_ids:
        raise ValueError(f"No geoms found on body {body_name!r}.")
    return geom_ids[-1]


def body_axis_and_grip(model, data, body_name: str, clamp_center_offset_m: float) -> tuple[np.ndarray, np.ndarray]:
    body_id = model.body(body_name).id
    pos = data.xpos[body_id].copy()
    axis = data.xmat[body_id].reshape(3, 3)[:, 2].copy()
    grip = pos + axis * float(clamp_center_offset_m)
    return axis, grip


def arm_nominal_q(lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    nominal = np.array(
        [
            -0.65,
            -0.25,
            -0.55,
            1.25,
            0.0,
            0.05,
            0.0,
            -0.25,
            0.45,
            0.35,
            1.1344640138,
            0.0,
            0.05,
            0.0,
        ],
        dtype=float,
    )
    return np.clip(nominal, lower, upper)


def build_starts(
    q0: np.ndarray,
    nominal: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    joint_names: tuple[str, ...],
    initial_joint_values: dict[str, float] | None,
    use_extra_starts: bool,
) -> list[np.ndarray]:
    if initial_joint_values:
        warm = np.array([initial_joint_values.get(name, q0[i]) for i, name in enumerate(joint_names)], dtype=float)
        starts = [warm]
        if use_extra_starts:
            starts.append(0.5 * warm + 0.5 * nominal)
        return starts

    starts = [nominal]
    if use_extra_starts:
        starts.extend(
            [
                q0,
                0.5 * q0 + 0.5 * nominal,
                np.clip(
                    nominal
                    + np.array([-0.2, 0.2, -0.2, 0.2, 0, 0, 0, -0.2, -0.3, 0.2, 0.1, 0, 0, 0]),
                    lower,
                    upper,
                ),
            ]
        )
    return starts


def solve_dual_hold_ik(
    model,
    data,
    targets: DualGripTargets,
    *,
    left_hand_geom_id: int | None = None,
    right_tool_body: str = "right_hammer_tool",
    left_tool_body: str = "left_hammer_clamp",
    left_elbow_body: str = "left_elbow_link",
    right_elbow_body: str = "right_elbow_link",
    left_clearance_bodies: tuple[str, ...] = (),
    right_clearance_bodies: tuple[str, ...] = (),
    elbow_clearance_y_m: float = 0.14,
    elbow_clearance_weight: float = 5.0,
    regularization_weight: float = 0.12,
    right_grip_weight: float = 10.0,
    left_grip_weight: float = 9.0,
    right_axis_weight: float = 5.0,
    left_axis_weight: float = 4.0,
    right_cross_axis_weight: float = 0.0,
    left_cross_axis_weight: float = 0.0,
    initial_joint_values: dict[str, float] | None = None,
    use_extra_starts: bool = True,
    max_nfev: int = 5000,
) -> DualHoldIKResult:
    """Solve both arms for a centered two-hand stick pose."""
    right_tool_body_id = model.body(right_tool_body).id
    left_tool_body_id = model.body(left_tool_body).id
    left_elbow_body_id = model.body(left_elbow_body).id
    right_elbow_body_id = model.body(right_elbow_body).id
    left_clearance_body_ids = [model.body(name).id for name in left_clearance_bodies]
    right_clearance_body_ids = [model.body(name).id for name in right_clearance_bodies]

    joint_names = LEFT_ARM_JOINTS + RIGHT_ARM_JOINTS
    addrs = joint_qpos_addrs(model, joint_names)
    lower, upper = joint_ranges(model, joint_names)
    q0 = data.qpos[addrs].copy()
    weights = IKWeights(
        right_grip=right_grip_weight,
        left_grip=left_grip_weight,
        right_axis=right_axis_weight,
        left_axis=left_axis_weight,
        right_cross_axis=right_cross_axis_weight,
        left_cross_axis=left_cross_axis_weight,
        elbow_clearance=elbow_clearance_weight,
        regularization=regularization_weight,
    )
    nominal = arm_nominal_q(lower, upper)
    starts = build_starts(q0, nominal, lower, upper, joint_names, initial_joint_values, use_extra_starts)

    def apply(x: np.ndarray) -> None:
        data.qpos[addrs] = x
        mujoco.mj_forward(model, data)

    def residual(x: np.ndarray) -> np.ndarray:
        apply(x)
        right_axis, right_grip = body_axis_and_grip(model, data, right_tool_body, targets.clamp_center_offset_m)
        left_axis, left_grip = body_axis_and_grip(model, data, left_tool_body, targets.clamp_center_offset_m)
        right_cross = data.xmat[right_tool_body_id].reshape(3, 3)[:, 0]
        left_cross = data.xmat[left_tool_body_id].reshape(3, 3)[:, 0]
        return np.concatenate(
            [
                (right_grip - targets.right_grip_m) * weights.right_grip,
                (right_axis - targets.stick_axis_m) * weights.right_axis,
                (right_cross - targets.right_cross_axis_m) * weights.right_cross_axis,
                (left_grip - targets.left_grip_m) * weights.left_grip,
                (left_axis - targets.stick_axis_m) * weights.left_axis,
                (left_cross - targets.left_cross_axis_m) * weights.left_cross_axis,
                np.array(
                    [
                        max(0.0, elbow_clearance_y_m - data.xpos[left_elbow_body_id][1]),
                        max(0.0, elbow_clearance_y_m + data.xpos[right_elbow_body_id][1]),
                    ]
                )
                * weights.elbow_clearance,
                np.array(
                    [max(0.0, elbow_clearance_y_m - data.xpos[body_id][1]) for body_id in left_clearance_body_ids]
                    + [max(0.0, elbow_clearance_y_m + data.xpos[body_id][1]) for body_id in right_clearance_body_ids],
                    dtype=float,
                )
                * weights.elbow_clearance,
                (x - nominal) * weights.regularization,
            ]
        )

    best = None
    for start in starts:
        result = least_squares(
            residual,
            np.clip(start, lower, upper),
            bounds=(lower, upper),
            max_nfev=max_nfev,
            xtol=1e-10,
            ftol=1e-10,
            gtol=1e-10,
        )
        if best is None or result.cost < best.cost:
            best = result

    assert best is not None
    apply(best.x)
    right_axis, right_grip = body_axis_and_grip(model, data, right_tool_body, targets.clamp_center_offset_m)
    left_axis, left_grip = body_axis_and_grip(model, data, left_tool_body, targets.clamp_center_offset_m)
    right_cross = data.xmat[right_tool_body_id].reshape(3, 3)[:, 0].copy()
    left_cross = data.xmat[left_tool_body_id].reshape(3, 3)[:, 0].copy()
    right_tool_root = data.xpos[right_tool_body_id].copy()
    target_tool_root = targets.right_grip_m - targets.stick_axis_m * targets.clamp_center_offset_m

    return DualHoldIKResult(
        joint_values={name: float(value) for name, value in zip(joint_names, best.x)},
        cost=float(best.cost),
        right_tool_error_m=float(np.linalg.norm(right_tool_root - target_tool_root)),
        right_grip_error_m=float(np.linalg.norm(right_grip - targets.right_grip_m)),
        left_grip_error_m=float(np.linalg.norm(left_grip - targets.left_grip_m)),
        right_axis_error=float(np.linalg.norm(right_axis - targets.stick_axis_m)),
        left_axis_error=float(np.linalg.norm(left_axis - targets.stick_axis_m)),
        right_cross_axis_error=float(np.linalg.norm(right_cross - targets.right_cross_axis_m)),
        left_cross_axis_error=float(np.linalg.norm(left_cross - targets.left_cross_axis_m)),
        targets=targets,
    )


def solve_dual_hold_closed_loop_ik(
    model,
    data,
    *,
    left_grip_distance_m: float = 0.20,
    clamp_center_offset_m: float = 0.0,
    right_tool_body: str = "right_hammer_tool",
    left_tool_body: str = "left_hammer_clamp",
    hammer_grip_body: str = "right_hammer_grip",
    left_elbow_body: str = "left_elbow_link",
    right_elbow_body: str = "right_elbow_link",
    left_clearance_bodies: tuple[str, ...] = (),
    right_clearance_bodies: tuple[str, ...] = (),
    head_axis_target_m=(0.0, 0.0, -1.0),
    elbow_clearance_y_m: float = 0.14,
    elbow_clearance_weight: float = 5.0,
    regularization_weight: float = 0.04,
    loop_grip_weight: float = 120.0,
    loop_axis_weight: float = 35.0,
    head_axis_weight: float = 3.0,
    hand_center_y_target_m: float | None = None,
    hand_center_y_weight: float = 5.0,
    initial_joint_values: dict[str, float] | None = None,
    use_extra_starts: bool = True,
    max_nfev: int = 5000,
) -> ClosedLoopIKResult:
    """Solve both arms without pinning the stick to a world-space root.

    The right hand carries the hammer. The left hand is constrained to the
    right-hand stick line at `left_grip_distance_m`, both clamp axes stay
    aligned, and the hammer head axis is biased toward `head_axis_target_m`.
    """
    right_tool_body_id = model.body(right_tool_body).id
    left_tool_body_id = model.body(left_tool_body).id
    hammer_grip_body_id = model.body(hammer_grip_body).id
    left_elbow_body_id = model.body(left_elbow_body).id
    right_elbow_body_id = model.body(right_elbow_body).id
    left_clearance_body_ids = [model.body(name).id for name in left_clearance_bodies]
    right_clearance_body_ids = [model.body(name).id for name in right_clearance_bodies]
    head_axis_target = normalize(head_axis_target_m)

    joint_names = LEFT_ARM_JOINTS + RIGHT_ARM_JOINTS
    addrs = joint_qpos_addrs(model, joint_names)
    lower, upper = joint_ranges(model, joint_names)
    q0 = data.qpos[addrs].copy()
    nominal = arm_nominal_q(lower, upper)
    starts = build_starts(q0, nominal, lower, upper, joint_names, initial_joint_values, use_extra_starts)
    regularization_center = starts[0] if initial_joint_values else nominal

    def apply(x: np.ndarray) -> None:
        data.qpos[addrs] = x
        mujoco.mj_forward(model, data)

    def residual(x: np.ndarray) -> np.ndarray:
        apply(x)
        right_axis, right_grip = body_axis_and_grip(model, data, right_tool_body, clamp_center_offset_m)
        left_axis, left_grip = body_axis_and_grip(model, data, left_tool_body, clamp_center_offset_m)
        right_left_grip = right_grip + right_axis * float(left_grip_distance_m)
        hand_center_y = 0.5 * (right_grip[1] + left_grip[1])
        head_axis = data.xmat[hammer_grip_body_id].reshape(3, 3)[:, 0]
        hand_center_residual = (
            np.array([(hand_center_y - float(hand_center_y_target_m)) * hand_center_y_weight])
            if hand_center_y_target_m is not None and hand_center_y_weight > 0.0
            else np.empty(0)
        )
        return np.concatenate(
            [
                (left_grip - right_left_grip) * loop_grip_weight,
                (left_axis - right_axis) * loop_axis_weight,
                (head_axis - head_axis_target) * head_axis_weight,
                hand_center_residual,
                np.array(
                    [
                        max(0.0, elbow_clearance_y_m - data.xpos[left_elbow_body_id][1]),
                        max(0.0, elbow_clearance_y_m + data.xpos[right_elbow_body_id][1]),
                    ]
                )
                * elbow_clearance_weight,
                np.array(
                    [max(0.0, elbow_clearance_y_m - data.xpos[body_id][1]) for body_id in left_clearance_body_ids]
                    + [max(0.0, elbow_clearance_y_m + data.xpos[body_id][1]) for body_id in right_clearance_body_ids],
                    dtype=float,
                )
                * elbow_clearance_weight,
                (x - regularization_center) * regularization_weight,
            ]
        )

    best = None
    for start in starts:
        result = least_squares(
            residual,
            np.clip(start, lower, upper),
            bounds=(lower, upper),
            max_nfev=max_nfev,
            xtol=1e-10,
            ftol=1e-10,
            gtol=1e-10,
        )
        if best is None or result.cost < best.cost:
            best = result

    assert best is not None
    apply(best.x)
    right_axis, right_grip = body_axis_and_grip(model, data, right_tool_body, clamp_center_offset_m)
    left_axis, left_grip = body_axis_and_grip(model, data, left_tool_body, clamp_center_offset_m)
    right_left_grip = right_grip + right_axis * float(left_grip_distance_m)
    head_axis = data.xmat[hammer_grip_body_id].reshape(3, 3)[:, 0].copy()
    hand_center_y = 0.5 * (right_grip[1] + left_grip[1])
    hand_center_y_target = hand_center_y if hand_center_y_target_m is None else float(hand_center_y_target_m)

    return ClosedLoopIKResult(
        joint_values={name: float(value) for name, value in zip(joint_names, best.x)},
        cost=float(best.cost),
        loop_grip_error_m=float(np.linalg.norm(left_grip - right_left_grip)),
        loop_axis_error=float(np.linalg.norm(left_axis - right_axis)),
        head_axis_error=float(np.linalg.norm(head_axis - head_axis_target)),
        hand_center_y_error_m=float(hand_center_y - hand_center_y_target),
        right_grip_m=right_grip.copy(),
        left_grip_m=left_grip.copy(),
        right_axis_m=right_axis.copy(),
        left_axis_m=left_axis.copy(),
        head_axis_m=head_axis,
    )
