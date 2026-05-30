#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import shutil
from dataclasses import dataclass
from pathlib import Path

import cv2
import mujoco
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCENE_TEMPLATE_PATH = PROJECT_ROOT / "demos" / "mochitsuki" / "scene_mochitsuki.xml"
SCENE_PATH = PROJECT_ROOT / "unitree_mujoco" / "unitree_robots" / "g1" / "scene_physicalai_mochitsuki.xml"
DEFAULT_RENDER_DIR = PROJECT_ROOT / "demos" / "mochitsuki" / "renders"
DIAGNOSTIC_CAMERAS = (
    "mochitsuki_camera",
    "grip_camera",
    "hand_front_camera",
    "hand_rear_camera",
    "hand_side_camera",
    "overhead_camera",
)
VIDEO_CAMERAS = (
    "mochitsuki_camera",
    "hand_front_camera",
    "hand_side_camera",
    "grip_camera",
    "hand_rear_camera",
    "overhead_camera",
)
CAMERA_LABELS = {
    "mochitsuki_camera": "main wide",
    "grip_camera": "grip detail",
    "hand_front_camera": "front wide",
    "hand_rear_camera": "rear wide",
    "hand_side_camera": "side wide",
    "overhead_camera": "overhead",
}


@dataclass(frozen=True)
class PoseKey:
    phase: float
    name: str
    base_xy: np.ndarray
    base_yaw: float
    pelvis_z: float
    joints: dict[str, float]
    hammer_pos: np.ndarray
    hammer_quat: np.ndarray
    mochi_scale: np.ndarray


@dataclass(frozen=True)
class GripSpec:
    body_name: str
    adapter_name: str
    handle_y: float
    adapter_side_local: np.ndarray
    palm_local: np.ndarray
    shaft_local: np.ndarray
    normal_local: np.ndarray


@dataclass(frozen=True)
class GripConstraint:
    body_name: str
    hand_local: np.ndarray
    target: np.ndarray
    weight: float


@dataclass(frozen=True)
class ActuatedJointSpec:
    actuator_name: str
    joint_name: str
    dof_adr: int
    torque_limit: float


def quat_from_euler_xyz(rx: float, ry: float, rz: float) -> np.ndarray:
    cx, sx = math.cos(rx * 0.5), math.sin(rx * 0.5)
    cy, sy = math.cos(ry * 0.5), math.sin(ry * 0.5)
    cz, sz = math.cos(rz * 0.5), math.sin(rz * 0.5)
    return np.array(
        [
            cx * cy * cz + sx * sy * sz,
            sx * cy * cz - cx * sy * sz,
            cx * sy * cz + sx * cy * sz,
            cx * cy * sz - sx * sy * cz,
        ],
        dtype=np.float64,
    )


def quat_to_matrix(q: np.ndarray) -> np.ndarray:
    q = q / np.linalg.norm(q)
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def quat_from_matrix(matrix: np.ndarray) -> np.ndarray:
    trace = float(np.trace(matrix))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        return np.array(
            [
                0.25 * s,
                (matrix[2, 1] - matrix[1, 2]) / s,
                (matrix[0, 2] - matrix[2, 0]) / s,
                (matrix[1, 0] - matrix[0, 1]) / s,
            ],
            dtype=np.float64,
        )
    axis = int(np.argmax(np.diag(matrix)))
    if axis == 0:
        s = math.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
        q = np.array(
            [
                (matrix[2, 1] - matrix[1, 2]) / s,
                0.25 * s,
                (matrix[0, 1] + matrix[1, 0]) / s,
                (matrix[0, 2] + matrix[2, 0]) / s,
            ],
            dtype=np.float64,
        )
    elif axis == 1:
        s = math.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
        q = np.array(
            [
                (matrix[0, 2] - matrix[2, 0]) / s,
                (matrix[0, 1] + matrix[1, 0]) / s,
                0.25 * s,
                (matrix[1, 2] + matrix[2, 1]) / s,
            ],
            dtype=np.float64,
        )
    else:
        s = math.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
        q = np.array(
            [
                (matrix[1, 0] - matrix[0, 1]) / s,
                (matrix[0, 2] + matrix[2, 0]) / s,
                (matrix[1, 2] + matrix[2, 1]) / s,
                0.25 * s,
            ],
            dtype=np.float64,
        )
    return q / np.linalg.norm(q)


def quat_from_yz(y_axis: np.ndarray, z_axis_hint: np.ndarray) -> np.ndarray:
    y_axis = y_axis / np.linalg.norm(y_axis)
    z_axis = z_axis_hint - y_axis * float(np.dot(z_axis_hint, y_axis))
    z_axis = z_axis / np.linalg.norm(z_axis)
    x_axis = np.cross(y_axis, z_axis)
    x_axis = x_axis / np.linalg.norm(x_axis)
    z_axis = np.cross(x_axis, y_axis)
    rotation = np.column_stack([x_axis, y_axis, z_axis])
    return quat_from_matrix(rotation)


def quat_from_yx(y_axis: np.ndarray, x_axis_hint: np.ndarray) -> np.ndarray:
    y_axis = y_axis / np.linalg.norm(y_axis)
    x_axis = x_axis_hint - y_axis * float(np.dot(x_axis_hint, y_axis))
    x_norm = np.linalg.norm(x_axis)
    if x_norm < 1e-6:
        x_axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        x_axis = x_axis - y_axis * float(np.dot(x_axis, y_axis))
        x_norm = np.linalg.norm(x_axis)
    x_axis = x_axis / x_norm
    z_axis = np.cross(x_axis, y_axis)
    z_axis = z_axis / np.linalg.norm(z_axis)
    x_axis = np.cross(y_axis, z_axis)
    x_axis = x_axis / np.linalg.norm(x_axis)
    rotation = np.column_stack([x_axis, y_axis, z_axis])
    return quat_from_matrix(rotation)


def slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    q0 = q0 / np.linalg.norm(q0)
    q1 = q1 / np.linalg.norm(q1)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    if dot > 0.9995:
        q = q0 + t * (q1 - q0)
        return q / np.linalg.norm(q)
    theta_0 = math.acos(dot)
    sin_theta_0 = math.sin(theta_0)
    theta = theta_0 * t
    s0 = math.cos(theta) - dot * math.sin(theta) / sin_theta_0
    s1 = math.sin(theta) / sin_theta_0
    return s0 * q0 + s1 * q1


def smoothstep(t: float) -> float:
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)


def merge_joints(*parts: dict[str, float]) -> dict[str, float]:
    out: dict[str, float] = {}
    for part in parts:
        out.update(part)
    return out


CYCLE_SECONDS = 1.90
KINE_MASS_KG = 1.0
HANDLE_AXIS_LOCAL = np.array([0.0, 1.0, 0.0], dtype=np.float64)
ADAPTER_HANDLE_CLEARANCE_M = 0.062
GRIP_AXIS_SPAN = 0.035
GRIP_NORMAL_SPAN = 0.035
ELBOW_BRANCH_WEIGHT = 0.72
WRIST_BRANCH_WEIGHT = 0.36
MAX_GRIP_CENTER_ERROR_M = 0.070
MAX_TRAJECTORY_GRIP_CENTER_ERROR_M = 0.070
MAX_GRIP_REFERENCE_ERROR_M = 0.070
MAX_TRAJECTORY_GRIP_REFERENCE_ERROR_M = 0.070
MIN_HAND_SIDE_OFFSET_M = 0.030
MIN_HAND_SIDE_SEPARATION_M = 0.120
MIN_HANDLE_LATERAL_ALIGNMENT = 0.75
MIN_LEFT_WRIST_LEFT_OFFSET_M = 0.045
MAX_RIGHT_WRIST_LEFT_OFFSET_M = 0.000
MIN_LEFT_ELBOW_LEFT_OFFSET_M = 0.070
MAX_RIGHT_ELBOW_LEFT_OFFSET_M = -0.065
MIN_ELBOW_FORWARD_OFFSET_M = -0.060
MIN_GRIP_AXIS_ALIGNMENT = 0.70
MIN_GRIP_NORMAL_ALIGNMENT = 0.815
MIN_ADAPTER_HANDLE_GAP_M = 0.030
MIN_FOREARM_HAND_ALIGNMENT = 0.875
MAX_WRIST_JOINT_ABS_RAD = 0.60
MAX_WRIST_NORM_RAD = 0.60
MIN_ELBOW_FLEXION_WARNING_RAD = -0.08
MAX_COM_FOOT_MID_OFFSET_M = 0.13
MAX_FOOT_Z_DRIFT_M = 0.006
MAX_ACTUATED_JOINT_SPEED_RAD_S = 3.0
MAX_ACTUATED_JOINT_ACCEL_RAD_S2 = 24.0
MAX_ESTIMATED_TORQUE_RATIO = 0.85
TRAJECTORY_SAMPLE_RATE_HZ = 90
COLLISION_SCAN_SAMPLE_COUNT = 180
WATCHED_COLLISION_KEYWORDS = (
    "usu",
    "kine",
    "adapter",
    "handle",
    "head",
    "mochi",
    "rim",
    "splash",
    "target",
    "wrist",
    "hand",
    "finger",
)
IK_SMOOTHNESS_WEIGHT = 0.350
ARM_NOMINAL_BLEND = 1.0
IK_POSTURE_WEIGHTS = {
    "left_shoulder_pitch_joint": 0.058,
    "left_shoulder_roll_joint": 0.070,
    "left_shoulder_yaw_joint": 0.035,
    "left_elbow_joint": 0.152,
    "left_wrist_roll_joint": 0.200,
    "left_wrist_pitch_joint": 0.220,
    "left_wrist_yaw_joint": 0.200,
    "right_shoulder_pitch_joint": 0.058,
    "right_shoulder_roll_joint": 0.070,
    "right_shoulder_yaw_joint": 0.035,
    "right_elbow_joint": 0.152,
    "right_wrist_roll_joint": 0.200,
    "right_wrist_pitch_joint": 0.220,
    "right_wrist_yaw_joint": 0.200,
}
GRIP_SPECS = (
    GripSpec(
        "left_wrist_yaw_link",
        "lower_adapter",
        0.50,
        np.array([-1.0, 0.0, 0.0], dtype=np.float64),
        np.array([0.1120, -0.0320, -0.0100], dtype=np.float64),
        np.array([0.3943, 0.8863, -0.2429], dtype=np.float64),
        np.array([0.8197, -0.2197, 0.5290], dtype=np.float64),
    ),
    GripSpec(
        "right_wrist_yaw_link",
        "upper_adapter",
        0.29,
        np.array([-1.0, 0.0, 0.0], dtype=np.float64),
        np.array([0.1120, 0.0320, -0.0100], dtype=np.float64),
        np.array([-0.1044, 0.6363, -0.7643], dtype=np.float64),
        np.array([0.9059, 0.3781, 0.1910], dtype=np.float64),
    ),
)
IK_JOINT_LIMIT_OVERRIDES = {
    "waist_yaw_joint": (-0.08, 0.05),
    "waist_roll_joint": (-0.035, 0.035),
    "waist_pitch_joint": (-0.02, 0.30),
    "left_wrist_roll_joint": (-1.20, 1.20),
    "left_wrist_pitch_joint": (-1.20, 1.20),
    "left_wrist_yaw_joint": (-1.20, 1.20),
    "right_wrist_roll_joint": (-1.20, 1.20),
    "right_wrist_pitch_joint": (-1.20, 1.20),
    "right_wrist_yaw_joint": (-1.20, 1.20),
}

STABLE_LEGS = {
    "left_hip_pitch_joint": -0.34,
    "left_hip_roll_joint": 0.12,
    "left_knee_joint": 0.64,
    "left_ankle_pitch_joint": -0.32,
    "left_ankle_roll_joint": -0.05,
    "right_hip_pitch_joint": -0.34,
    "right_hip_roll_joint": -0.12,
    "right_knee_joint": 0.64,
    "right_ankle_pitch_joint": -0.32,
    "right_ankle_roll_joint": 0.05,
}

LOW_ARMS = {
    "left_shoulder_pitch_joint": -0.56,
    "left_shoulder_roll_joint": 0.105,
    "left_shoulder_yaw_joint": 0.08,
    "left_elbow_joint": 1.48,
    "left_wrist_roll_joint": 0.05,
    "left_wrist_pitch_joint": -0.20,
    "left_wrist_yaw_joint": 0.08,
    "right_shoulder_pitch_joint": -0.68,
    "right_shoulder_roll_joint": -0.125,
    "right_shoulder_yaw_joint": -0.08,
    "right_elbow_joint": 1.38,
    "right_wrist_roll_joint": -0.05,
    "right_wrist_pitch_joint": -0.20,
    "right_wrist_yaw_joint": -0.08,
}

HIGH_ARMS = {
    "left_shoulder_pitch_joint": -0.72,
    "left_shoulder_roll_joint": 0.145,
    "left_shoulder_yaw_joint": 0.14,
    "left_elbow_joint": 1.50,
    "left_wrist_roll_joint": 0.10,
    "left_wrist_pitch_joint": -0.12,
    "left_wrist_yaw_joint": 0.10,
    "right_shoulder_pitch_joint": -0.92,
    "right_shoulder_roll_joint": -0.175,
    "right_shoulder_yaw_joint": -0.18,
    "right_elbow_joint": 1.28,
    "right_wrist_roll_joint": -0.10,
    "right_wrist_pitch_joint": -0.12,
    "right_wrist_yaw_joint": -0.10,
}

BASE_XY = np.array([0.39586232, -0.00954763], dtype=np.float64)
BASE_YAW = 0.70
WORLD_UP = np.array([0.0, 0.0, 1.0], dtype=np.float64)
ROBOT_FORWARD_WORLD = np.array([math.cos(BASE_YAW), math.sin(BASE_YAW), 0.0], dtype=np.float64)
ROBOT_LEFT_WORLD = np.array([-math.sin(BASE_YAW), math.cos(BASE_YAW), 0.0], dtype=np.float64)
ROBOT_RIGHT_WORLD = -ROBOT_LEFT_WORLD
REFERENCE_HANDLE_HORIZONTAL = ROBOT_LEFT_WORLD + 0.08 * ROBOT_FORWARD_WORLD
REFERENCE_HANDLE_HORIZONTAL = REFERENCE_HANDLE_HORIZONTAL / np.linalg.norm(REFERENCE_HANDLE_HORIZONTAL)
PELVIS_Z = 0.735

def reference_video_quat(
    head_forward_tilt: float,
    head_left_tilt: float = 0.0,
    handle_tail_drop: float = 0.0,
) -> np.ndarray:
    handle_axis = REFERENCE_HANDLE_HORIZONTAL - WORLD_UP * handle_tail_drop
    handle_axis = handle_axis / np.linalg.norm(handle_axis)
    z_axis_hint = (
        WORLD_UP
        - ROBOT_FORWARD_WORLD * head_forward_tilt
        + ROBOT_LEFT_WORLD * head_left_tilt
    )

    return quat_from_yz(handle_axis, z_axis_hint)


IMPACT_QUAT = reference_video_quat(0.00, handle_tail_drop=0.20)
COMPRESS_QUAT = reference_video_quat(0.03, handle_tail_drop=0.23)
RELEASE_QUAT = reference_video_quat(0.22, 0.02, handle_tail_drop=0.40)
LIFT_LOW_QUAT = reference_video_quat(0.45, 0.04, handle_tail_drop=0.60)
LIFT_MID_QUAT = reference_video_quat(0.70, 0.06, handle_tail_drop=0.75)
LIFT_TOP_QUAT = reference_video_quat(0.82, 0.06, handle_tail_drop=0.85)
AIM_HIGH_QUAT = reference_video_quat(0.55, 0.04, handle_tail_drop=0.70)
AIM_LOW_QUAT = reference_video_quat(0.26, 0.02, handle_tail_drop=0.45)
STRIKE_PRE_QUAT = reference_video_quat(0.10, 0.01, handle_tail_drop=0.30)
STRIKE_QUAT = IMPACT_QUAT

HAMMER_TARGET_WORLD = np.array([0.78, 0.00, 0.0], dtype=np.float64) + ROBOT_RIGHT_WORLD * 0.06


def hammer_roll_head_pos(backoff_m: float, z_m: float) -> np.ndarray:
    return HAMMER_TARGET_WORLD - ROBOT_FORWARD_WORLD * backoff_m + np.array([0.0, 0.0, z_m], dtype=np.float64)


KEYS = [
    PoseKey(
        0.00,
        "impact",
        BASE_XY,
        BASE_YAW,
        PELVIS_Z,
        merge_joints(STABLE_LEGS, LOW_ARMS, {"waist_pitch_joint": 0.195, "waist_yaw_joint": -0.11, "waist_roll_joint": 0.00}),
        hammer_roll_head_pos(0.000, 0.865),
        IMPACT_QUAT,
        np.array([1.12, 1.06, 0.66]),
    ),
    PoseKey(
        0.07,
        "compress",
        BASE_XY,
        BASE_YAW,
        PELVIS_Z,
        merge_joints(STABLE_LEGS, LOW_ARMS, {"waist_pitch_joint": 0.205, "waist_yaw_joint": -0.12, "waist_roll_joint": 0.00}),
        hammer_roll_head_pos(0.000, 0.855),
        COMPRESS_QUAT,
        np.array([1.18, 1.08, 0.58]),
    ),
    PoseKey(
        0.16,
        "release",
        BASE_XY,
        BASE_YAW,
        PELVIS_Z,
        merge_joints(STABLE_LEGS, LOW_ARMS, {"waist_pitch_joint": 0.185, "waist_yaw_joint": -0.12, "waist_roll_joint": 0.01}),
        hammer_roll_head_pos(0.015, 0.925),
        RELEASE_QUAT,
        np.array([1.03, 1.00, 0.90]),
    ),
    PoseKey(
        0.30,
        "lift_low",
        BASE_XY,
        BASE_YAW,
        PELVIS_Z,
        merge_joints(STABLE_LEGS, HIGH_ARMS, {"waist_pitch_joint": 0.135, "waist_yaw_joint": -0.14, "waist_roll_joint": 0.01}),
        hammer_roll_head_pos(0.025, 1.012),
        LIFT_LOW_QUAT,
        np.array([0.98, 0.98, 1.02]),
    ),
    PoseKey(
        0.44,
        "lift_mid",
        BASE_XY,
        BASE_YAW,
        PELVIS_Z,
        merge_joints(STABLE_LEGS, HIGH_ARMS, {"waist_pitch_joint": 0.115, "waist_yaw_joint": -0.15, "waist_roll_joint": 0.01}),
        hammer_roll_head_pos(0.030, 1.058),
        LIFT_MID_QUAT,
        np.array([0.96, 0.96, 1.08]),
    ),
    PoseKey(
        0.56,
        "lift_top",
        BASE_XY,
        BASE_YAW,
        PELVIS_Z,
        merge_joints(STABLE_LEGS, HIGH_ARMS, {"waist_pitch_joint": 0.105, "waist_yaw_joint": -0.15, "waist_roll_joint": 0.01}),
        hammer_roll_head_pos(0.0375, 1.090),
        LIFT_TOP_QUAT,
        np.array([0.96, 0.96, 1.08]),
    ),
    PoseKey(
        0.66,
        "settle_top",
        BASE_XY,
        BASE_YAW,
        PELVIS_Z,
        merge_joints(STABLE_LEGS, HIGH_ARMS, {"waist_pitch_joint": 0.115, "waist_yaw_joint": -0.14, "waist_roll_joint": 0.00}),
        hammer_roll_head_pos(0.035, 1.073),
        LIFT_TOP_QUAT,
        np.array([0.97, 0.97, 1.05]),
    ),
    PoseKey(
        0.73,
        "aim_high",
        BASE_XY,
        BASE_YAW,
        PELVIS_Z,
        merge_joints(STABLE_LEGS, HIGH_ARMS, {"waist_pitch_joint": 0.135, "waist_yaw_joint": -0.13, "waist_roll_joint": 0.00}),
        hammer_roll_head_pos(0.030, 1.031),
        AIM_HIGH_QUAT,
        np.array([0.98, 0.98, 1.00]),
    ),
    PoseKey(
        0.835,
        "aim_low",
        BASE_XY,
        BASE_YAW,
        PELVIS_Z,
        merge_joints(STABLE_LEGS, LOW_ARMS, {"waist_pitch_joint": 0.185, "waist_yaw_joint": -0.12, "waist_roll_joint": 0.00}),
        hammer_roll_head_pos(0.01875, 0.950),
        AIM_LOW_QUAT,
        np.array([1.02, 1.00, 0.90]),
    ),
    PoseKey(
        0.905,
        "strike_pre",
        BASE_XY,
        BASE_YAW,
        PELVIS_Z,
        merge_joints(STABLE_LEGS, LOW_ARMS, {"waist_pitch_joint": 0.195, "waist_yaw_joint": -0.11, "waist_roll_joint": 0.00}),
        hammer_roll_head_pos(0.00375, 0.890),
        STRIKE_PRE_QUAT,
        np.array([1.08, 1.03, 0.78]),
    ),
    PoseKey(
        0.970,
        "strike",
        BASE_XY,
        BASE_YAW,
        PELVIS_Z,
        merge_joints(STABLE_LEGS, LOW_ARMS, {"waist_pitch_joint": 0.205, "waist_yaw_joint": -0.12, "waist_roll_joint": 0.00}),
        hammer_roll_head_pos(0.000, 0.865),
        STRIKE_QUAT,
        np.array([1.16, 1.07, 0.62]),
    ),
    PoseKey(
        1.00,
        "impact_next",
        BASE_XY,
        BASE_YAW,
        PELVIS_Z,
        merge_joints(STABLE_LEGS, LOW_ARMS, {"waist_pitch_joint": 0.195, "waist_yaw_joint": -0.11, "waist_roll_joint": 0.00}),
        hammer_roll_head_pos(0.000, 0.865),
        IMPACT_QUAT,
        np.array([1.12, 1.06, 0.66]),
    ),
]


def load_model() -> tuple[mujoco.MjModel, mujoco.MjData]:
    shutil.copyfile(SCENE_TEMPLATE_PATH, SCENE_PATH)
    model = mujoco.MjModel.from_xml_path(str(SCENE_PATH))
    data = mujoco.MjData(model)
    return model, data


def interpolate_keyframes(phase: float) -> PoseKey:
    phase = phase % 1.0
    for left, right in zip(KEYS, KEYS[1:]):
        if left.phase <= phase <= right.phase:
            span = right.phase - left.phase
            u = smoothstep((phase - left.phase) / span if span > 0 else 0.0)
            joints = {
                name: left.joints.get(name, 0.0) * (1.0 - u) + right.joints.get(name, 0.0) * u
                for name in set(left.joints) | set(right.joints)
            }
            return PoseKey(
                phase,
                f"{left.name}->{right.name}",
                left.base_xy * (1.0 - u) + right.base_xy * u,
                left.base_yaw * (1.0 - u) + right.base_yaw * u,
                left.pelvis_z * (1.0 - u) + right.pelvis_z * u,
                joints,
                left.hammer_pos * (1.0 - u) + right.hammer_pos * u,
                slerp(left.hammer_quat, right.hammer_quat, u),
                left.mochi_scale * (1.0 - u) + right.mochi_scale * u,
            )
    return KEYS[0]


def joint_qpos_addresses(model: mujoco.MjModel) -> dict[str, int]:
    addresses: dict[str, int] = {}
    for joint_id in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        if name:
            addresses[name] = int(model.jnt_qposadr[joint_id])
    return addresses


def joint_dof_indices(model: mujoco.MjModel, joint_names: list[str]) -> list[tuple[int, int, int, float, float]]:
    indices: list[tuple[int, int, int, float, float]] = []
    for name in joint_names:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            continue
        qpos_adr = int(model.jnt_qposadr[joint_id])
        dof_adr = int(model.jnt_dofadr[joint_id])
        low, high = model.jnt_range[joint_id]
        if name in IK_JOINT_LIMIT_OVERRIDES:
            low, high = IK_JOINT_LIMIT_OVERRIDES[name]
        indices.append((joint_id, qpos_adr, dof_adr, float(low), float(high)))
    return indices


def actuated_joint_specs(model: mujoco.MjModel) -> list[ActuatedJointSpec]:
    specs: list[ActuatedJointSpec] = []
    for actuator_id in range(model.nu):
        actuator_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id)
        joint_id = int(model.actuator_trnid[actuator_id, 0])
        if joint_id < 0:
            continue
        joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        if not actuator_name or not joint_name:
            continue
        dof_adr = int(model.jnt_dofadr[joint_id])
        ctrl_low, ctrl_high = model.actuator_ctrlrange[actuator_id]
        torque_limit = float(max(abs(ctrl_low), abs(ctrl_high)))
        if torque_limit <= 0.0:
            continue
        specs.append(ActuatedJointSpec(actuator_name, joint_name, dof_adr, torque_limit))
    return specs


LEG_JOINT_NAMES = tuple(STABLE_LEGS)
TORSO_JOINT_NAMES = ("waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint")
ARM_JOINT_NAMES = (
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)
ARM_IK_JOINTS = list(ARM_JOINT_NAMES)
RATE_LIMITED_JOINT_NAMES = (*TORSO_JOINT_NAMES, *ARM_JOINT_NAMES)


def unit_vector(vector: np.ndarray) -> np.ndarray:
    return vector / np.linalg.norm(vector)


def grip_anchor_local(spec: GripSpec) -> np.ndarray:
    return np.array([0.0, spec.handle_y, 0.0], dtype=np.float64) + (
        unit_vector(spec.adapter_side_local) * ADAPTER_HANDLE_CLEARANCE_M
    )


def grip_handle_normal_local(spec: GripSpec) -> np.ndarray:
    return -unit_vector(spec.adapter_side_local)


def robot_left_world(pose: PoseKey) -> np.ndarray:
    return np.array([-math.sin(pose.base_yaw), math.cos(pose.base_yaw), 0.0], dtype=np.float64)


def anatomical_left_world(model: mujoco.MjModel, data: mujoco.MjData, pose: PoseKey) -> np.ndarray:
    left_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_shoulder_pitch_link")
    right_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_shoulder_pitch_link")
    if min(left_id, right_id) < 0:
        return robot_left_world(pose)
    left_axis = data.xpos[left_id] - data.xpos[right_id]
    left_axis[2] = 0.0
    norm = float(np.linalg.norm(left_axis))
    if norm < 1e-9:
        return robot_left_world(pose)
    return left_axis / norm


def robot_forward_world(pose: PoseKey) -> np.ndarray:
    return np.array([math.cos(pose.base_yaw), math.sin(pose.base_yaw), 0.0], dtype=np.float64)


def robot_local_point(pose: PoseKey, point: np.ndarray) -> np.ndarray:
    origin = np.array([pose.base_xy[0], pose.base_xy[1], 0.0], dtype=np.float64)
    delta = point - origin
    return np.array(
        [
            float(np.dot(delta, robot_forward_world(pose))),
            float(np.dot(delta, robot_left_world(pose))),
            float(point[2]),
        ],
        dtype=np.float64,
    )


def robot_frame_point(pose: PoseKey, forward_m: float, left_m: float, z_m: float) -> np.ndarray:
    origin = np.array([pose.base_xy[0], pose.base_xy[1], 0.0], dtype=np.float64)
    forward = robot_forward_world(pose)
    left = robot_left_world(pose)
    return origin + forward * forward_m + left * left_m + np.array([0.0, 0.0, z_m], dtype=np.float64)


def hammer_grip_centers(pose: PoseKey) -> dict[str, np.ndarray]:
    rotation = quat_to_matrix(pose.hammer_quat)
    return {
        spec.body_name: pose.hammer_pos + rotation @ grip_anchor_local(spec)
        for spec in GRIP_SPECS
    }


def grip_reference_pairs(pose: PoseKey, spec: GripSpec) -> list[tuple[np.ndarray, np.ndarray]]:
    rotation = quat_to_matrix(pose.hammer_quat)
    anchor_local = grip_anchor_local(spec)
    anchor = pose.hammer_pos + rotation @ anchor_local
    shaft_local = unit_vector(spec.shaft_local)
    normal_local = unit_vector(spec.normal_local)
    handle_normal_local = grip_handle_normal_local(spec)
    return [
        (spec.palm_local, anchor),
        (
            spec.palm_local + shaft_local * GRIP_AXIS_SPAN,
            anchor + rotation @ (HANDLE_AXIS_LOCAL * GRIP_AXIS_SPAN),
        ),
        (
            spec.palm_local - shaft_local * GRIP_AXIS_SPAN,
            anchor - rotation @ (HANDLE_AXIS_LOCAL * GRIP_AXIS_SPAN),
        ),
        (
            spec.palm_local + normal_local * GRIP_NORMAL_SPAN,
            anchor + rotation @ (handle_normal_local * GRIP_NORMAL_SPAN),
        ),
    ]


def hammer_grip_constraints(pose: PoseKey) -> list[GripConstraint]:
    constraints: list[GripConstraint] = []
    for spec in GRIP_SPECS:
        grip_pairs = grip_reference_pairs(pose, spec)
        constraints.extend(
            [
                GripConstraint(spec.body_name, grip_pairs[0][0], grip_pairs[0][1], 1.25),
                GripConstraint(spec.body_name, grip_pairs[1][0], grip_pairs[1][1], 0.70),
                GripConstraint(spec.body_name, grip_pairs[2][0], grip_pairs[2][1], 0.70),
                GripConstraint(spec.body_name, grip_pairs[3][0], grip_pairs[3][1], 0.80),
            ]
        )
    elbow_z = pose.pelvis_z + 0.155
    constraints.extend(
        [
            GripConstraint(
                "left_elbow_link",
                np.zeros(3, dtype=np.float64),
                robot_frame_point(pose, -0.030, 0.072, elbow_z),
                ELBOW_BRANCH_WEIGHT,
            ),
            GripConstraint(
                "right_elbow_link",
                np.zeros(3, dtype=np.float64),
                robot_frame_point(pose, -0.030, -0.084, elbow_z + 0.030),
                ELBOW_BRANCH_WEIGHT,
            ),
            GripConstraint(
                "left_wrist_yaw_link",
                np.zeros(3, dtype=np.float64),
                robot_frame_point(pose, 0.095, 0.058, pose.hammer_pos[2] - 0.055),
                WRIST_BRANCH_WEIGHT,
            ),
            GripConstraint(
                "right_wrist_yaw_link",
                np.zeros(3, dtype=np.float64),
                robot_frame_point(pose, 0.100, -0.068, pose.hammer_pos[2] + 0.015),
                WRIST_BRANCH_WEIGHT,
            ),
        ]
    )
    return constraints


def solve_arm_ik(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    constraints: list[GripConstraint],
    smoothness_weight: float = 0.0,
    reference_qpos: np.ndarray | None = None,
) -> None:
    dof_info = joint_dof_indices(model, ARM_IK_JOINTS)
    if not dof_info:
        return

    body_ids = {
        body_name: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        for body_name in {constraint.body_name for constraint in constraints}
    }
    if any(body_id < 0 for body_id in body_ids.values()):
        return

    dof_adrs = [dof_adr for _, _, dof_adr, _, _ in dof_info]
    qpos_adrs = [qpos_adr for _, qpos_adr, _, _, _ in dof_info]
    posture_weights = np.array(
        [
            IK_POSTURE_WEIGHTS.get(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id) or "", 0.0)
            for joint_id, _, _, _, _ in dof_info
        ],
        dtype=np.float64,
    )
    posture_reference_qpos = data.qpos[qpos_adrs].copy()
    if reference_qpos is None:
        smooth_reference_qpos = posture_reference_qpos
    else:
        smooth_reference_qpos = reference_qpos[qpos_adrs].copy()
    damping = 0.040
    max_step = 0.030
    for _ in range(120):
        mujoco.mj_forward(model, data)
        errors: list[np.ndarray] = []
        jac_rows: list[np.ndarray] = []
        for constraint in constraints:
            body_id = body_ids[constraint.body_name]
            body_rotation = np.array(data.xmat[body_id]).reshape(3, 3)
            point = data.xpos[body_id] + body_rotation @ constraint.hand_local
            errors.append((constraint.target - point) * constraint.weight)
            jacp = np.zeros((3, model.nv), dtype=np.float64)
            jacr = np.zeros((3, model.nv), dtype=np.float64)
            mujoco.mj_jac(model, data, jacp, jacr, point, body_id)
            jac_rows.append(jacp[:, dof_adrs] * constraint.weight)

        error = np.concatenate(errors)
        if float(np.linalg.norm(error)) < 0.014:
            break

        jacobian = np.vstack(jac_rows)
        qpos_now = data.qpos[qpos_adrs]
        smoothness = smoothness_weight * smoothness_weight
        posture = posture_weights * posture_weights
        regularization = smoothness + posture
        lhs = jacobian.T @ jacobian + np.diag(damping * damping + regularization)
        rhs = (
            jacobian.T @ error
            + posture * (posture_reference_qpos - qpos_now)
            + smoothness * (smooth_reference_qpos - qpos_now)
        )
        step = np.linalg.solve(lhs, rhs)
        step = np.clip(step, -max_step, max_step)

        for delta, (_, qpos_adr, _, low, high) in zip(step, dof_info):
            data.qpos[qpos_adr] = np.clip(data.qpos[qpos_adr] + delta, low, high)


def grip_metrics(model: mujoco.MjModel, data: mujoco.MjData, pose: PoseKey) -> dict[str, float]:
    hammer_axis = quat_to_matrix(pose.hammer_quat) @ HANDLE_AXIS_LOCAL
    hammer_axis = hammer_axis / np.linalg.norm(hammer_axis)
    hammer_rotation = quat_to_matrix(pose.hammer_quat)
    robot_left = anatomical_left_world(model, data, pose)
    metrics: dict[str, float] = {}
    for spec in GRIP_SPECS:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, spec.body_name)
        if body_id < 0:
            continue
        hammer_normal = hammer_rotation @ grip_handle_normal_local(spec)
        hammer_normal = hammer_normal / np.linalg.norm(hammer_normal)
        body_rotation = np.array(data.xmat[body_id]).reshape(3, 3)
        palm = data.xpos[body_id] + body_rotation @ spec.palm_local
        reference_errors = [
            float(np.linalg.norm(target - (data.xpos[body_id] + body_rotation @ hand_local)))
            for hand_local, target in grip_reference_pairs(pose, spec)
        ]
        center = hammer_grip_centers(pose)[spec.body_name]
        hand_axis = body_rotation @ (spec.shaft_local / np.linalg.norm(spec.shaft_local))
        axis_alignment = float(abs(np.dot(hand_axis, hammer_axis)))
        hand_normal = body_rotation @ (spec.normal_local / np.linalg.norm(spec.normal_local))
        normal_alignment = float(abs(np.dot(hand_normal, hammer_normal)))
        handle_center = pose.hammer_pos + hammer_rotation @ np.array([0.0, spec.handle_y, 0.0])
        handle_gap = float(np.dot(handle_center - palm, hammer_normal))
        robot_left_offset = float(np.dot(center - handle_center, robot_left))
        prefix = spec.adapter_name
        metrics[f"{prefix}_center_error"] = float(np.linalg.norm(center - palm))
        metrics[f"{prefix}_reference_error"] = max(reference_errors) if reference_errors else 0.0
        metrics[f"{prefix}_axis_alignment"] = axis_alignment
        metrics[f"{prefix}_normal_alignment"] = normal_alignment
        metrics[f"{prefix}_handle_gap"] = handle_gap
        metrics[f"{prefix}_robot_left_offset"] = robot_left_offset
    return metrics


def stability_metrics(model: mujoco.MjModel, data: mujoco.MjData) -> dict[str, float]:
    pelvis_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    left_foot_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_ankle_roll_link")
    right_foot_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_ankle_roll_link")
    if min(pelvis_id, left_foot_id, right_foot_id) < 0:
        return {}

    com_xy = data.subtree_com[pelvis_id][:2]
    foot_mid_xy = (data.xpos[left_foot_id][:2] + data.xpos[right_foot_id][:2]) * 0.5
    foot_separation = float(np.linalg.norm(data.xpos[left_foot_id][:2] - data.xpos[right_foot_id][:2]))
    return {
        "com_foot_mid_offset": float(np.linalg.norm(com_xy - foot_mid_xy)),
        "foot_separation": foot_separation,
    }


def arm_branch_metrics(model: mujoco.MjModel, data: mujoco.MjData, pose: PoseKey) -> dict[str, float]:
    values: dict[str, float] = {}
    for side in ("left", "right"):
        for part in ("wrist_yaw", "elbow"):
            body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{side}_{part}_link")
            if body_id < 0:
                continue
            local = robot_local_point(pose, data.xpos[body_id])
            values[f"{side}_{part}_forward_offset"] = float(local[0])
            values[f"{side}_{part}_left_offset"] = float(local[1])
    return values


def ergonomic_metrics(model: mujoco.MjModel, data: mujoco.MjData) -> dict[str, float]:
    qadr = joint_qpos_addresses(model)
    values: dict[str, float] = {}
    forearm_hand_alignments: list[float] = []
    wrist_abs_values: list[float] = []
    wrist_norms: list[float] = []
    elbow_flexions: list[float] = []

    for side in ("left", "right"):
        elbow_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{side}_elbow_link")
        wrist_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{side}_wrist_yaw_link")
        if min(elbow_id, wrist_id) >= 0:
            forearm = data.xpos[wrist_id] - data.xpos[elbow_id]
            norm = float(np.linalg.norm(forearm))
            if norm > 1e-9:
                forearm /= norm
                hand_forward = np.array(data.xmat[wrist_id]).reshape(3, 3)[:, 0]
                alignment = float(abs(np.dot(forearm, hand_forward)))
                forearm_hand_alignments.append(alignment)
                values[f"{side}_forearm_hand_alignment"] = alignment

        wrist_values: list[float] = []
        for wrist_axis in ("roll", "pitch", "yaw"):
            joint_name = f"{side}_wrist_{wrist_axis}_joint"
            if joint_name not in qadr:
                continue
            value = float(data.qpos[qadr[joint_name]])
            wrist_values.append(value)
            wrist_abs_values.append(abs(value))
        if wrist_values:
            wrist_norm = float(np.linalg.norm(wrist_values))
            wrist_norms.append(wrist_norm)
            values[f"{side}_wrist_norm"] = wrist_norm

        elbow_joint = f"{side}_elbow_joint"
        if elbow_joint in qadr:
            elbow_value = float(data.qpos[qadr[elbow_joint]])
            elbow_flexions.append(elbow_value)
            values[f"{side}_elbow_flexion"] = elbow_value

    if forearm_hand_alignments:
        values["min_forearm_hand_alignment"] = float(np.min(forearm_hand_alignments))
    if wrist_abs_values:
        values["max_wrist_joint_abs"] = float(np.max(wrist_abs_values))
    if wrist_norms:
        values["max_wrist_norm"] = float(np.max(wrist_norms))
    if elbow_flexions:
        values["min_elbow_flexion"] = float(np.min(elbow_flexions))
    return values


def collision_watch_metrics(
    model: mujoco.MjModel,
    cycle_seconds: float,
    sample_count: int = COLLISION_SCAN_SAMPLE_COUNT,
) -> dict[str, float]:
    data = mujoco.MjData(model)
    watched_pairs: dict[tuple[str, str], float] = {}
    torso_wrist_pairs: dict[tuple[str, str], float] = {}

    def contact_label(geom_id: int) -> str:
        geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(geom_id)) or f"geom#{int(geom_id)}"
        body_id = int(model.geom_bodyid[int(geom_id)])
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or "body?"
        return f"{geom_name}@{body_name}"

    for sample_idx in range(sample_count):
        pose = interpolate_keyframes(sample_idx / sample_count)
        apply_pose(
            model,
            data,
            pose,
            warm_start=sample_idx > 0,
            control_dt=cycle_seconds / sample_count,
        )
        mujoco.mj_forward(model, data)
        for contact_idx in range(data.ncon):
            contact = data.contact[contact_idx]
            if contact.dist >= 0.0:
                continue
            a = contact_label(int(contact.geom1))
            b = contact_label(int(contact.geom2))
            key = tuple(sorted((a, b)))
            joined = a + b
            if any(keyword in joined for keyword in WATCHED_COLLISION_KEYWORDS):
                watched_pairs[key] = min(watched_pairs.get(key, 0.0), float(contact.dist))
            if "torso" in joined and "wrist" in joined:
                torso_wrist_pairs[key] = min(torso_wrist_pairs.get(key, 0.0), float(contact.dist))

    return {
        "watched_negative_contact_pairs": float(len(watched_pairs)),
        "torso_wrist_negative_contact_pairs": float(len(torso_wrist_pairs)),
        "min_watched_contact_distance": min(watched_pairs.values(), default=0.0),
        "min_torso_wrist_contact_distance": min(torso_wrist_pairs.values(), default=0.0),
    }


def trajectory_safety_metrics(model: mujoco.MjModel, cycle_seconds: float) -> dict[str, float | str]:
    sample_count = max(32, int(round(cycle_seconds * TRAJECTORY_SAMPLE_RATE_HZ)))
    dt = cycle_seconds / sample_count

    data = mujoco.MjData(model)
    total_count = sample_count * 2 + 2
    all_qpos = np.zeros((total_count, model.nq), dtype=np.float64)
    all_hammer_com = np.zeros((total_count, 3), dtype=np.float64)
    all_foot_z = np.zeros((total_count, 2), dtype=np.float64)
    all_grip_center_error = np.zeros(total_count, dtype=np.float64)
    all_grip_reference_error = np.zeros(total_count, dtype=np.float64)
    all_grip_axis_alignment = np.ones(total_count, dtype=np.float64)
    all_grip_normal_alignment = np.ones(total_count, dtype=np.float64)
    all_adapter_handle_gap = np.ones(total_count, dtype=np.float64) * ADAPTER_HANDLE_CLEARANCE_M
    all_left_grip_left_offset = np.zeros(total_count, dtype=np.float64)
    all_right_grip_left_offset = np.zeros(total_count, dtype=np.float64)
    all_absolute_grip_side_separation = np.zeros(total_count, dtype=np.float64)
    all_handle_lateral_alignment = np.ones(total_count, dtype=np.float64)
    all_forearm_hand_alignment = np.ones(total_count, dtype=np.float64)
    all_wrist_joint_abs = np.zeros(total_count, dtype=np.float64)
    all_wrist_norm = np.zeros(total_count, dtype=np.float64)
    all_elbow_flexion = np.zeros(total_count, dtype=np.float64)
    all_left_wrist_left_offset = np.zeros(total_count, dtype=np.float64)
    all_right_wrist_left_offset = np.zeros(total_count, dtype=np.float64)
    all_left_elbow_left_offset = np.zeros(total_count, dtype=np.float64)
    all_right_elbow_left_offset = np.zeros(total_count, dtype=np.float64)
    all_left_elbow_forward_offset = np.zeros(total_count, dtype=np.float64)
    all_right_elbow_forward_offset = np.zeros(total_count, dtype=np.float64)

    hammer_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "kine_hammer")
    left_foot_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_ankle_roll_link")
    right_foot_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_ankle_roll_link")
    if min(hammer_id, left_foot_id, right_foot_id) < 0:
        return {}

    for frame_idx in range(total_count):
        pose = interpolate_keyframes((frame_idx % sample_count) / sample_count)
        apply_pose(
            model,
            data,
            pose,
            warm_start=frame_idx > 0,
            control_dt=dt,
        )
        all_qpos[frame_idx] = data.qpos
        all_hammer_com[frame_idx] = data.subtree_com[hammer_id]
        all_foot_z[frame_idx, 0] = data.xpos[left_foot_id, 2]
        all_foot_z[frame_idx, 1] = data.xpos[right_foot_id, 2]
        grip = grip_metrics(model, data, pose)
        centers = hammer_grip_centers(pose)
        robot_left = anatomical_left_world(model, data, pose)
        handle_axis = quat_to_matrix(pose.hammer_quat) @ HANDLE_AXIS_LOCAL
        handle_axis = handle_axis / np.linalg.norm(handle_axis)
        center_errors = [value for name, value in grip.items() if name.endswith("_center_error")]
        reference_errors = [value for name, value in grip.items() if name.endswith("_reference_error")]
        axis_alignments = [value for name, value in grip.items() if name.endswith("_axis_alignment")]
        normal_alignments = [value for name, value in grip.items() if name.endswith("_normal_alignment")]
        handle_gaps = [value for name, value in grip.items() if name.endswith("_handle_gap")]
        all_grip_center_error[frame_idx] = max(center_errors) if center_errors else 0.0
        all_grip_reference_error[frame_idx] = max(reference_errors) if reference_errors else 0.0
        all_grip_axis_alignment[frame_idx] = min(axis_alignments) if axis_alignments else 1.0
        all_grip_normal_alignment[frame_idx] = min(normal_alignments) if normal_alignments else 1.0
        all_adapter_handle_gap[frame_idx] = min(handle_gaps) if handle_gaps else ADAPTER_HANDLE_CLEARANCE_M
        all_left_grip_left_offset[frame_idx] = float(grip.get("lower_adapter_robot_left_offset", 0.0))
        all_right_grip_left_offset[frame_idx] = float(grip.get("upper_adapter_robot_left_offset", 0.0))
        all_absolute_grip_side_separation[frame_idx] = float(
            np.dot(centers["left_wrist_yaw_link"] - centers["right_wrist_yaw_link"], robot_left)
        )
        all_handle_lateral_alignment[frame_idx] = abs(float(np.dot(handle_axis, robot_left)))
        ergonomic = ergonomic_metrics(model, data)
        all_forearm_hand_alignment[frame_idx] = float(ergonomic.get("min_forearm_hand_alignment", 1.0))
        all_wrist_joint_abs[frame_idx] = float(ergonomic.get("max_wrist_joint_abs", 0.0))
        all_wrist_norm[frame_idx] = float(ergonomic.get("max_wrist_norm", 0.0))
        all_elbow_flexion[frame_idx] = float(ergonomic.get("min_elbow_flexion", 0.0))
        branch = arm_branch_metrics(model, data, pose)
        all_left_wrist_left_offset[frame_idx] = float(branch.get("left_wrist_yaw_left_offset", 0.0))
        all_right_wrist_left_offset[frame_idx] = float(branch.get("right_wrist_yaw_left_offset", 0.0))
        all_left_elbow_left_offset[frame_idx] = float(branch.get("left_elbow_left_offset", 0.0))
        all_right_elbow_left_offset[frame_idx] = float(branch.get("right_elbow_left_offset", 0.0))
        all_left_elbow_forward_offset[frame_idx] = float(branch.get("left_elbow_forward_offset", 0.0))
        all_right_elbow_forward_offset[frame_idx] = float(branch.get("right_elbow_forward_offset", 0.0))

    start = sample_count
    qpos_seq = all_qpos[start : start + sample_count + 2]
    hammer_com_seq = all_hammer_com[start : start + sample_count + 2]
    foot_z_seq = all_foot_z[start + 1 : start + sample_count + 1]
    grip_center_error_seq = all_grip_center_error[start + 1 : start + sample_count + 1]
    grip_reference_error_seq = all_grip_reference_error[start + 1 : start + sample_count + 1]
    grip_axis_alignment_seq = all_grip_axis_alignment[start + 1 : start + sample_count + 1]
    grip_normal_alignment_seq = all_grip_normal_alignment[start + 1 : start + sample_count + 1]
    adapter_handle_gap_seq = all_adapter_handle_gap[start + 1 : start + sample_count + 1]
    left_grip_left_offset_seq = all_left_grip_left_offset[start + 1 : start + sample_count + 1]
    right_grip_left_offset_seq = all_right_grip_left_offset[start + 1 : start + sample_count + 1]
    absolute_grip_side_separation_seq = all_absolute_grip_side_separation[start + 1 : start + sample_count + 1]
    handle_lateral_alignment_seq = all_handle_lateral_alignment[start + 1 : start + sample_count + 1]
    forearm_hand_alignment_seq = all_forearm_hand_alignment[start + 1 : start + sample_count + 1]
    wrist_joint_abs_seq = all_wrist_joint_abs[start + 1 : start + sample_count + 1]
    wrist_norm_seq = all_wrist_norm[start + 1 : start + sample_count + 1]
    elbow_flexion_seq = all_elbow_flexion[start + 1 : start + sample_count + 1]
    left_wrist_left_offset_seq = all_left_wrist_left_offset[start + 1 : start + sample_count + 1]
    right_wrist_left_offset_seq = all_right_wrist_left_offset[start + 1 : start + sample_count + 1]
    left_elbow_left_offset_seq = all_left_elbow_left_offset[start + 1 : start + sample_count + 1]
    right_elbow_left_offset_seq = all_right_elbow_left_offset[start + 1 : start + sample_count + 1]
    left_elbow_forward_offset_seq = all_left_elbow_forward_offset[start + 1 : start + sample_count + 1]
    right_elbow_forward_offset_seq = all_right_elbow_forward_offset[start + 1 : start + sample_count + 1]

    qvel_seq = np.zeros((sample_count + 1, model.nv), dtype=np.float64)
    qacc_seq = np.zeros((sample_count, model.nv), dtype=np.float64)
    for frame_idx in range(sample_count + 1):
        mujoco.mj_differentiatePos(model, qvel_seq[frame_idx], dt, qpos_seq[frame_idx], qpos_seq[frame_idx + 1])
    for frame_idx in range(sample_count):
        qacc_seq[frame_idx] = (qvel_seq[frame_idx + 1] - qvel_seq[frame_idx]) / dt

    hammer_acc_seq = np.zeros((sample_count, 3), dtype=np.float64)
    for frame_idx in range(sample_count):
        hammer_acc_seq[frame_idx] = (
            hammer_com_seq[frame_idx + 2] - 2.0 * hammer_com_seq[frame_idx + 1] + hammer_com_seq[frame_idx]
        ) / (dt * dt)

    specs = actuated_joint_specs(model)
    eval_data = mujoco.MjData(model)
    max_speed = 0.0
    max_accel = 0.0
    max_torque_ratio = 0.0
    worst_speed_joint = ""
    worst_accel_joint = ""
    worst_torque_joint = ""
    gravity = np.array(model.opt.gravity, dtype=np.float64)

    for frame_idx in range(sample_count):
        eval_data.qpos[:] = qpos_seq[frame_idx + 1]
        eval_data.qvel[:] = qvel_seq[frame_idx]
        mujoco.mj_forward(model, eval_data)
        estimated_torque = np.zeros(model.nv, dtype=np.float64)

        hammer_force = KINE_MASS_KG * (hammer_acc_seq[frame_idx] - gravity)
        hammer_force *= 1.25 / len(GRIP_SPECS)
        for spec in GRIP_SPECS:
            body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, spec.body_name)
            if body_id < 0:
                continue
            body_rotation = np.array(eval_data.xmat[body_id]).reshape(3, 3)
            point = eval_data.xpos[body_id] + body_rotation @ spec.palm_local
            jacp = np.zeros((3, model.nv), dtype=np.float64)
            jacr = np.zeros((3, model.nv), dtype=np.float64)
            mujoco.mj_jac(model, eval_data, jacp, jacr, point, body_id)
            estimated_torque += jacp.T @ hammer_force

        for spec in specs:
            speed = abs(float(qvel_seq[frame_idx, spec.dof_adr]))
            accel = abs(float(qacc_seq[frame_idx, spec.dof_adr]))
            torque_ratio = abs(float(estimated_torque[spec.dof_adr])) / spec.torque_limit
            if speed > max_speed:
                max_speed = speed
                worst_speed_joint = spec.joint_name
            if accel > max_accel:
                max_accel = accel
                worst_accel_joint = spec.joint_name
            if torque_ratio > max_torque_ratio:
                max_torque_ratio = torque_ratio
                worst_torque_joint = spec.joint_name

    foot_z_drift = float(np.max(np.ptp(foot_z_seq, axis=0)))
    return {
        "sample_count": float(sample_count),
        "max_trajectory_grip_center_error": float(np.max(grip_center_error_seq)),
        "max_trajectory_grip_reference_error": float(np.max(grip_reference_error_seq)),
        "min_trajectory_grip_axis_alignment": float(np.min(grip_axis_alignment_seq)),
        "min_trajectory_grip_normal_alignment": float(np.min(grip_normal_alignment_seq)),
        "min_trajectory_adapter_handle_gap": float(np.min(adapter_handle_gap_seq)),
        "min_trajectory_left_grip_left_offset": float(np.min(left_grip_left_offset_seq)),
        "max_trajectory_right_grip_left_offset": float(np.max(right_grip_left_offset_seq)),
        "min_trajectory_absolute_grip_side_separation": float(np.min(absolute_grip_side_separation_seq)),
        "min_trajectory_handle_lateral_alignment": float(np.min(handle_lateral_alignment_seq)),
        "min_trajectory_forearm_hand_alignment": float(np.min(forearm_hand_alignment_seq)),
        "max_trajectory_wrist_joint_abs": float(np.max(wrist_joint_abs_seq)),
        "max_trajectory_wrist_norm": float(np.max(wrist_norm_seq)),
        "min_trajectory_elbow_flexion": float(np.min(elbow_flexion_seq)),
        "min_trajectory_left_wrist_left_offset": float(np.min(left_wrist_left_offset_seq)),
        "max_trajectory_right_wrist_left_offset": float(np.max(right_wrist_left_offset_seq)),
        "min_trajectory_left_elbow_left_offset": float(np.min(left_elbow_left_offset_seq)),
        "max_trajectory_right_elbow_left_offset": float(np.max(right_elbow_left_offset_seq)),
        "min_trajectory_left_elbow_forward_offset": float(np.min(left_elbow_forward_offset_seq)),
        "min_trajectory_right_elbow_forward_offset": float(np.min(right_elbow_forward_offset_seq)),
        "max_foot_z_drift": foot_z_drift,
        "max_actuated_joint_speed": max_speed,
        "worst_speed_joint": worst_speed_joint,
        "max_actuated_joint_accel": max_accel,
        "worst_accel_joint": worst_accel_joint,
        "max_estimated_torque_ratio": max_torque_ratio,
        "worst_torque_joint": worst_torque_joint,
    }


def validate_pose_limits(model: mujoco.MjModel) -> list[str]:
    warnings: list[str] = []
    for key in KEYS:
        for name, value in key.joints.items():
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if joint_id < 0:
                warnings.append(f"{key.name}: missing joint {name}")
                continue
            if model.jnt_limited[joint_id]:
                low, high = model.jnt_range[joint_id]
                if value < low or value > high:
                    warnings.append(f"{key.name}: {name}={value:.3f} outside [{low:.3f}, {high:.3f}]")
    kine_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "kine_hammer")
    if kine_body >= 0 and abs(float(model.body_mass[kine_body]) - KINE_MASS_KG) > 0.05:
        warnings.append(f"kine_hammer mass={model.body_mass[kine_body]:.3f}kg, expected about {KINE_MASS_KG:.1f}kg")
    return warnings


def apply_pose(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    pose: PoseKey,
    warm_start: bool = False,
    control_dt: float | None = None,
) -> None:
    rate_limited = warm_start and control_dt is not None
    start_qpos = data.qpos.copy() if rate_limited else None
    if not warm_start:
        data.qpos[:] = model.qpos0
    if not rate_limited:
        data.qvel[:] = 0.0

    base_quat = quat_from_euler_xyz(0.0, 0.0, pose.base_yaw)
    data.qpos[0:7] = np.array([pose.base_xy[0], pose.base_xy[1], pose.pelvis_z, *base_quat])

    qadr = joint_qpos_addresses(model)
    for joint_name, value in pose.joints.items():
        if warm_start and joint_name in ARM_JOINT_NAMES:
            data.qpos[qadr[joint_name]] = (1.0 - ARM_NOMINAL_BLEND) * data.qpos[qadr[joint_name]] + (
                ARM_NOMINAL_BLEND * value
            )
        else:
            data.qpos[qadr[joint_name]] = value

    kine_joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "kine_free")
    kine_adr = int(model.jnt_qposadr[kine_joint])
    data.qpos[kine_adr : kine_adr + 3] = pose.hammer_pos
    data.qpos[kine_adr + 3 : kine_adr + 7] = pose.hammer_quat / np.linalg.norm(pose.hammer_quat)

    mochi_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "mochi_lump")
    splash_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "impact_splash")
    ring_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "strike_target_ring")
    if mochi_id >= 0:
        model.geom_size[mochi_id] = np.array([0.100, 0.084, 0.047]) * pose.mochi_scale
    if splash_id >= 0:
        squash = 1.0 - min(0.42, pose.mochi_scale[2])
        model.geom_size[splash_id] = np.array([0.070 + 0.082 * squash, 0.054 + 0.056 * squash, 0.010 + 0.006 * squash])
    if ring_id >= 0:
        model.geom_rgba[ring_id, 3] = 0.0

    mujoco.mj_forward(model, data)
    solve_arm_ik(
        model,
        data,
        hammer_grip_constraints(pose),
        smoothness_weight=IK_SMOOTHNESS_WEIGHT if warm_start else 0.0,
        reference_qpos=start_qpos if start_qpos is not None else None,
    )
    if start_qpos is not None:
        mujoco.mj_differentiatePos(model, data.qvel, control_dt, start_qpos, data.qpos)
    mujoco.mj_forward(model, data)


def render_frame(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    renderer: mujoco.Renderer,
    camera: str = "mochitsuki_camera",
) -> np.ndarray:
    renderer.update_scene(data, camera=camera)
    return renderer.render()


def write_contact_sheet(saved_frames: list[tuple[str, Path]], output: Path) -> None:
    if not saved_frames:
        return

    thumbnails: list[np.ndarray] = []
    thumb_width = 320
    thumb_height = 180
    for label, frame_path in saved_frames:
        image = cv2.imread(str(frame_path))
        if image is None:
            continue
        image = cv2.resize(image, (thumb_width, thumb_height))
        cv2.putText(image, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 20, 20), 3, cv2.LINE_AA)
        cv2.putText(image, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1, cv2.LINE_AA)
        thumbnails.append(image)

    if not thumbnails:
        return

    cols = min(4, len(thumbnails))
    rows = int(math.ceil(len(thumbnails) / cols))
    canvas = np.full((rows * thumb_height, cols * thumb_width, 3), 245, dtype=np.uint8)
    for idx, image in enumerate(thumbnails):
        row = idx // cols
        col = idx % cols
        y0 = row * thumb_height
        x0 = col * thumb_width
        canvas[y0 : y0 + thumb_height, x0 : x0 + thumb_width] = image
    cv2.imwrite(str(output), canvas)


def draw_camera_label(rgb: np.ndarray, label: str) -> np.ndarray:
    image = rgb.copy()
    text = CAMERA_LABELS.get(label, label)
    cv2.putText(image, text, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (15, 15, 15), 4, cv2.LINE_AA)
    cv2.putText(image, text, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 1, cv2.LINE_AA)
    return image


def camera_grid_shape(camera_count: int) -> tuple[int, int]:
    if camera_count <= 1:
        return (1, 1)
    if camera_count <= 4:
        return (2, 2)
    return (3, int(math.ceil(camera_count / 3)))


def compose_camera_grid(frames: list[tuple[str, np.ndarray]], width: int, height: int) -> np.ndarray:
    cols, rows = camera_grid_shape(len(frames))
    canvas = np.full((height, width, 3), 235, dtype=np.uint8)
    for idx, (camera, rgb) in enumerate(frames):
        row = idx // cols
        col = idx % cols
        x0 = int(round(col * width / cols))
        x1 = int(round((col + 1) * width / cols))
        y0 = int(round(row * height / rows))
        y1 = int(round((row + 1) * height / rows))
        tile = cv2.resize(draw_camera_label(rgb, camera), (x1 - x0, y1 - y0))
        canvas[y0:y1, x0:x1] = tile
    return canvas


def write_video(
    output: Path,
    duration: float,
    fps: int,
    width: int,
    height: int,
    save_keyframes: bool,
    cycle_seconds: float,
    cameras: tuple[str, ...],
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    model, data = load_model()
    warnings = validate_pose_limits(model)
    if warnings:
        raise RuntimeError("pose validation failed:\n" + "\n".join(warnings))

    if not cameras:
        cameras = ("mochitsuki_camera",)
    grid_cols, grid_rows = camera_grid_shape(len(cameras))
    render_width = width if len(cameras) == 1 else int(math.ceil(width / grid_cols))
    render_height = height if len(cameras) == 1 else int(math.ceil(height / grid_rows))
    renderer = mujoco.Renderer(model, height=render_height, width=render_width)
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"failed to open video writer: {output}")

    total_frames = int(duration * fps)
    keyframe_dir = output.parent / "keyframes_latest"
    camera_keyframe_dir = output.parent / "camera_keyframes_latest"
    if save_keyframes:
        keyframe_dir.mkdir(parents=True, exist_ok=True)
        camera_keyframe_dir.mkdir(parents=True, exist_ok=True)
        for stale_frame in keyframe_dir.glob("*.png"):
            stale_frame.unlink()
        for camera in cameras:
            camera_dir = camera_keyframe_dir / camera
            camera_dir.mkdir(parents=True, exist_ok=True)
            for stale_frame in camera_dir.glob("*.png"):
                stale_frame.unlink()

    keyframe_indices: dict[int, str] = {}
    saved_frames: list[tuple[str, Path]] = []
    saved_by_camera: dict[str, list[tuple[str, Path]]] = {camera: [] for camera in cameras}
    if save_keyframes:
        cycle_frames = int(cycle_seconds * fps)
        for key in KEYS[:-1]:
            keyframe_indices[int(round(key.phase * cycle_frames))] = key.name

    prewarm_frames = max(1, int(round(cycle_seconds * fps)))
    for frame_idx in range(prewarm_frames):
        pose = interpolate_keyframes((frame_idx % prewarm_frames) / prewarm_frames)
        apply_pose(
            model,
            data,
            pose,
            warm_start=frame_idx > 0,
            control_dt=1.0 / fps,
        )

    for frame_idx in range(total_frames):
        t = frame_idx / fps
        pose = interpolate_keyframes((t % cycle_seconds) / cycle_seconds)
        apply_pose(model, data, pose, warm_start=True, control_dt=1.0 / fps)
        camera_frames = [(camera, render_frame(model, data, renderer, camera=camera)) for camera in cameras]
        rgb = camera_frames[0][1] if len(camera_frames) == 1 else compose_camera_grid(camera_frames, width, height)
        writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

        if save_keyframes and frame_idx in keyframe_indices:
            name = keyframe_indices[frame_idx]
            frame_path = keyframe_dir / f"{frame_idx:04d}_{name}.png"
            cv2.imwrite(str(frame_path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            saved_frames.append((name, frame_path))
            for camera, camera_rgb in camera_frames:
                camera_frame_path = camera_keyframe_dir / camera / f"{frame_idx:04d}_{name}.png"
                cv2.imwrite(str(camera_frame_path), cv2.cvtColor(draw_camera_label(camera_rgb, camera), cv2.COLOR_RGB2BGR))
                saved_by_camera[camera].append((name, camera_frame_path))

    writer.release()
    renderer.close()
    if save_keyframes:
        contact_sheet = output.parent / "contact_sheet.png"
        write_contact_sheet(saved_frames, contact_sheet)
        camera_sheet_dir = output.parent / "camera_contact_sheets"
        camera_sheet_dir.mkdir(parents=True, exist_ok=True)
        for stale_sheet in camera_sheet_dir.glob("*.png"):
            stale_sheet.unlink()
        for camera, camera_frames in saved_by_camera.items():
            sheet_path = camera_sheet_dir / f"{camera}.png"
            write_contact_sheet(camera_frames, sheet_path)
            if camera == "grip_camera":
                write_contact_sheet(camera_frames, output.parent / "grip_contact_sheet.png")
    print(f"wrote video: {output}")
    if save_keyframes:
        print(f"wrote keyframes: {keyframe_dir}")
        print(f"wrote camera keyframes: {camera_keyframe_dir}")
        print(f"wrote contact sheet: {output.parent / 'contact_sheet.png'}")
        print(f"wrote camera contact sheets: {output.parent / 'camera_contact_sheets'}")


def write_diagnostic_sheets(
    output_dir: Path,
    cycle_seconds: float,
    fps: int,
    width: int,
    height: int,
    samples: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    model, data = load_model()
    warnings = validate_pose_limits(model)
    if warnings:
        raise RuntimeError("pose validation failed:\n" + "\n".join(warnings))

    renderer = mujoco.Renderer(model, height=height, width=width)
    saved_by_camera: dict[str, list[tuple[str, Path]]] = {camera: [] for camera in DIAGNOSTIC_CAMERAS}
    for camera in DIAGNOSTIC_CAMERAS:
        camera_dir = output_dir / camera
        camera_dir.mkdir(parents=True, exist_ok=True)
        for stale_frame in camera_dir.glob("*.png"):
            stale_frame.unlink()

    prewarm_frames = max(1, int(round(cycle_seconds * fps)))
    for frame_idx in range(prewarm_frames):
        pose = interpolate_keyframes((frame_idx % prewarm_frames) / prewarm_frames)
        apply_pose(
            model,
            data,
            pose,
            warm_start=frame_idx > 0,
            control_dt=1.0 / fps,
        )

    control_dt = cycle_seconds / max(1, samples)
    for sample_idx in range(samples):
        phase = sample_idx / samples
        pose = interpolate_keyframes(phase)
        apply_pose(model, data, pose, warm_start=True, control_dt=control_dt)
        label = f"{sample_idx:02d} p={phase:.2f}"
        for camera in DIAGNOSTIC_CAMERAS:
            rgb = render_frame(model, data, renderer, camera=camera)
            frame_path = output_dir / camera / f"{sample_idx:02d}_phase_{phase:.3f}.png"
            cv2.imwrite(str(frame_path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            saved_by_camera[camera].append((label, frame_path))

    renderer.close()
    for camera, saved_frames in saved_by_camera.items():
        sheet_path = output_dir / f"{camera}_dense.png"
        write_contact_sheet(saved_frames, sheet_path)
        print(f"wrote diagnostic sheet: {sheet_path}")


def run_viewer(duration: float, fps: int, cycle_seconds: float) -> None:
    import time
    import mujoco.viewer

    model, data = load_model()
    warnings = validate_pose_limits(model)
    if warnings:
        raise RuntimeError("pose validation failed:\n" + "\n".join(warnings))

    with mujoco.viewer.launch_passive(model, data) as viewer:
        prewarm_frames = max(1, int(round(cycle_seconds * fps)))
        for frame_idx in range(prewarm_frames):
            pose = interpolate_keyframes((frame_idx % prewarm_frames) / prewarm_frames)
            apply_pose(
                model,
                data,
                pose,
                warm_start=frame_idx > 0,
                control_dt=1.0 / fps,
            )
        start = time.perf_counter()
        frame_idx = 0
        while viewer.is_running() and time.perf_counter() - start < duration:
            elapsed = time.perf_counter() - start
            pose = interpolate_keyframes((elapsed % cycle_seconds) / cycle_seconds)
            apply_pose(model, data, pose, warm_start=frame_idx > 0, control_dt=1.0 / fps)
            viewer.sync()
            frame_idx += 1
            time.sleep(1.0 / fps)


def run_check(render_smoke: bool, cycle_seconds: float) -> None:
    model, data = load_model()
    warnings = validate_pose_limits(model)
    if warnings:
        raise RuntimeError("pose validation failed:\n" + "\n".join(warnings))

    max_grip_center_error = 0.0
    max_grip_reference_error = 0.0
    min_grip_axis_alignment = 1.0
    min_grip_normal_alignment = 1.0
    min_adapter_handle_gap = float("inf")
    min_left_palm_robot_left_offset = float("inf")
    max_right_palm_robot_left_offset = -float("inf")
    min_palm_side_separation = float("inf")
    min_absolute_grip_side_separation = float("inf")
    min_handle_lateral_alignment = 1.0
    max_com_foot_mid_offset = 0.0
    min_foot_separation = float("inf")
    min_forearm_hand_alignment = 1.0
    max_wrist_joint_abs = 0.0
    max_wrist_norm = 0.0
    min_elbow_flexion = float("inf")
    min_left_wrist_left_offset = float("inf")
    max_right_wrist_left_offset = -float("inf")
    min_left_elbow_left_offset = float("inf")
    max_right_elbow_left_offset = -float("inf")
    min_left_elbow_forward_offset = float("inf")
    min_right_elbow_forward_offset = float("inf")
    check_phases = np.linspace(0.0, 0.99, 40)
    check_dt = cycle_seconds / max(1, len(check_phases))
    for phase_idx, phase in enumerate(check_phases):
        pose = interpolate_keyframes(float(phase))
        apply_pose(model, data, pose, warm_start=phase_idx > 0, control_dt=check_dt)
        if not np.all(np.isfinite(data.qpos)):
            raise RuntimeError(f"non-finite qpos at phase={phase:.3f}")
        if not np.all(np.isfinite(data.xpos)):
            raise RuntimeError(f"non-finite xpos at phase={phase:.3f}")
        metrics = grip_metrics(model, data, pose)
        centers = hammer_grip_centers(pose)
        robot_left = anatomical_left_world(model, data, pose)
        handle_axis = quat_to_matrix(pose.hammer_quat) @ HANDLE_AXIS_LOCAL
        handle_axis = handle_axis / np.linalg.norm(handle_axis)
        center_errors = [value for name, value in metrics.items() if name.endswith("_center_error")]
        reference_errors = [value for name, value in metrics.items() if name.endswith("_reference_error")]
        axis_alignments = [value for name, value in metrics.items() if name.endswith("_axis_alignment")]
        normal_alignments = [value for name, value in metrics.items() if name.endswith("_normal_alignment")]
        handle_gaps = [value for name, value in metrics.items() if name.endswith("_handle_gap")]
        left_side_offset = float(metrics.get("lower_adapter_robot_left_offset", 0.0))
        right_side_offset = float(metrics.get("upper_adapter_robot_left_offset", 0.0))
        max_grip_center_error = max(max_grip_center_error, *center_errors)
        max_grip_reference_error = max(max_grip_reference_error, *reference_errors)
        min_grip_axis_alignment = min(min_grip_axis_alignment, *axis_alignments)
        min_grip_normal_alignment = min(min_grip_normal_alignment, *normal_alignments)
        min_adapter_handle_gap = min(min_adapter_handle_gap, *handle_gaps)
        min_left_palm_robot_left_offset = min(min_left_palm_robot_left_offset, left_side_offset)
        max_right_palm_robot_left_offset = max(max_right_palm_robot_left_offset, right_side_offset)
        min_palm_side_separation = min(min_palm_side_separation, left_side_offset - right_side_offset)
        min_absolute_grip_side_separation = min(
            min_absolute_grip_side_separation,
            float(np.dot(centers["left_wrist_yaw_link"] - centers["right_wrist_yaw_link"], robot_left)),
        )
        min_handle_lateral_alignment = min(min_handle_lateral_alignment, abs(float(np.dot(handle_axis, robot_left))))
        stable = stability_metrics(model, data)
        if stable:
            max_com_foot_mid_offset = max(max_com_foot_mid_offset, stable["com_foot_mid_offset"])
            min_foot_separation = min(min_foot_separation, stable["foot_separation"])
        ergonomic = ergonomic_metrics(model, data)
        min_forearm_hand_alignment = min(
            min_forearm_hand_alignment,
            float(ergonomic.get("min_forearm_hand_alignment", 1.0)),
        )
        max_wrist_joint_abs = max(max_wrist_joint_abs, float(ergonomic.get("max_wrist_joint_abs", 0.0)))
        max_wrist_norm = max(max_wrist_norm, float(ergonomic.get("max_wrist_norm", 0.0)))
        min_elbow_flexion = min(min_elbow_flexion, float(ergonomic.get("min_elbow_flexion", 0.0)))
        branch = arm_branch_metrics(model, data, pose)
        min_left_wrist_left_offset = min(
            min_left_wrist_left_offset,
            float(branch.get("left_wrist_yaw_left_offset", 1.0)),
        )
        max_right_wrist_left_offset = max(
            max_right_wrist_left_offset,
            float(branch.get("right_wrist_yaw_left_offset", -1.0)),
        )
        min_left_elbow_left_offset = min(
            min_left_elbow_left_offset,
            float(branch.get("left_elbow_left_offset", 1.0)),
        )
        max_right_elbow_left_offset = max(
            max_right_elbow_left_offset,
            float(branch.get("right_elbow_left_offset", -1.0)),
        )
        min_left_elbow_forward_offset = min(
            min_left_elbow_forward_offset,
            float(branch.get("left_elbow_forward_offset", 1.0)),
        )
        min_right_elbow_forward_offset = min(
            min_right_elbow_forward_offset,
            float(branch.get("right_elbow_forward_offset", 1.0)),
        )

    if max_grip_center_error > MAX_GRIP_CENTER_ERROR_M:
        raise RuntimeError(
            f"grip center error too high: {max_grip_center_error:.3f}m > {MAX_GRIP_CENTER_ERROR_M:.3f}m"
        )
    if max_grip_reference_error > MAX_GRIP_REFERENCE_ERROR_M:
        raise RuntimeError(
            f"grip rigid reference error too high: "
            f"{max_grip_reference_error:.3f}m > {MAX_GRIP_REFERENCE_ERROR_M:.3f}m"
        )
    if min_grip_axis_alignment < MIN_GRIP_AXIS_ALIGNMENT:
        raise RuntimeError(
            f"grip axis alignment too low: {min_grip_axis_alignment:.3f} < {MIN_GRIP_AXIS_ALIGNMENT:.3f}"
        )
    if min_grip_normal_alignment < MIN_GRIP_NORMAL_ALIGNMENT:
        raise RuntimeError(
            f"grip normal alignment too low: {min_grip_normal_alignment:.3f} < {MIN_GRIP_NORMAL_ALIGNMENT:.3f}"
        )
    if min_adapter_handle_gap < MIN_ADAPTER_HANDLE_GAP_M:
        raise RuntimeError(
            f"adapter handle gap too low: {min_adapter_handle_gap:.3f}m < {MIN_ADAPTER_HANDLE_GAP_M:.3f}m"
        )
    if min_absolute_grip_side_separation < MIN_HAND_SIDE_SEPARATION_M:
        raise RuntimeError(
            f"absolute left/right grip separation too low: {min_absolute_grip_side_separation:.3f}m < "
            f"{MIN_HAND_SIDE_SEPARATION_M:.3f}m"
        )
    if min_handle_lateral_alignment < MIN_HANDLE_LATERAL_ALIGNMENT:
        raise RuntimeError(
            f"hammer handle is not lateral enough: "
            f"{min_handle_lateral_alignment:.3f} < {MIN_HANDLE_LATERAL_ALIGNMENT:.3f}"
        )
    if min_left_wrist_left_offset < MIN_LEFT_WRIST_LEFT_OFFSET_M:
        raise RuntimeError(
            f"left wrist crossed to wrong side: {min_left_wrist_left_offset:.3f}m < "
            f"{MIN_LEFT_WRIST_LEFT_OFFSET_M:.3f}m"
        )
    if max_right_wrist_left_offset > MAX_RIGHT_WRIST_LEFT_OFFSET_M:
        raise RuntimeError(
            f"right wrist crossed to wrong side: {max_right_wrist_left_offset:.3f}m > "
            f"{MAX_RIGHT_WRIST_LEFT_OFFSET_M:.3f}m"
        )
    if min_left_elbow_left_offset < MIN_LEFT_ELBOW_LEFT_OFFSET_M:
        raise RuntimeError(
            f"left elbow crossed to wrong side: {min_left_elbow_left_offset:.3f}m < "
            f"{MIN_LEFT_ELBOW_LEFT_OFFSET_M:.3f}m"
        )
    if max_right_elbow_left_offset > MAX_RIGHT_ELBOW_LEFT_OFFSET_M:
        raise RuntimeError(
            f"right elbow crossed to wrong side: {max_right_elbow_left_offset:.3f}m > "
            f"{MAX_RIGHT_ELBOW_LEFT_OFFSET_M:.3f}m"
        )
    if min_left_elbow_forward_offset < MIN_ELBOW_FORWARD_OFFSET_M:
        raise RuntimeError(
            f"left elbow moved behind torso: {min_left_elbow_forward_offset:.3f}m < "
            f"{MIN_ELBOW_FORWARD_OFFSET_M:.3f}m"
        )
    if min_right_elbow_forward_offset < MIN_ELBOW_FORWARD_OFFSET_M:
        raise RuntimeError(
            f"right elbow moved behind torso: {min_right_elbow_forward_offset:.3f}m < "
            f"{MIN_ELBOW_FORWARD_OFFSET_M:.3f}m"
        )
    if max_com_foot_mid_offset > MAX_COM_FOOT_MID_OFFSET_M:
        raise RuntimeError(
            f"COM projection offset too high: {max_com_foot_mid_offset:.3f}m > {MAX_COM_FOOT_MID_OFFSET_M:.3f}m"
        )
    if min_forearm_hand_alignment < MIN_FOREARM_HAND_ALIGNMENT:
        raise RuntimeError(
            f"forearm-hand alignment too low: "
            f"{min_forearm_hand_alignment:.3f} < {MIN_FOREARM_HAND_ALIGNMENT:.3f}"
        )
    if max_wrist_joint_abs > MAX_WRIST_JOINT_ABS_RAD:
        raise RuntimeError(
            f"wrist joint angle too high: {max_wrist_joint_abs:.3f}rad > {MAX_WRIST_JOINT_ABS_RAD:.3f}rad"
        )
    if max_wrist_norm > MAX_WRIST_NORM_RAD:
        raise RuntimeError(f"wrist norm too high: {max_wrist_norm:.3f}rad > {MAX_WRIST_NORM_RAD:.3f}rad")
    if min_elbow_flexion < MIN_ELBOW_FLEXION_WARNING_RAD:
        raise RuntimeError(
            f"elbow hyperextension too high: "
            f"{min_elbow_flexion:.3f}rad < {MIN_ELBOW_FLEXION_WARNING_RAD:.3f}rad"
        )

    trajectory = trajectory_safety_metrics(model, cycle_seconds)
    max_foot_z_drift = float(trajectory.get("max_foot_z_drift", 0.0))
    max_trajectory_grip_center_error = float(trajectory.get("max_trajectory_grip_center_error", 0.0))
    max_trajectory_grip_reference_error = float(trajectory.get("max_trajectory_grip_reference_error", 0.0))
    min_trajectory_grip_axis_alignment = float(trajectory.get("min_trajectory_grip_axis_alignment", 1.0))
    min_trajectory_grip_normal_alignment = float(trajectory.get("min_trajectory_grip_normal_alignment", 1.0))
    min_trajectory_adapter_handle_gap = float(trajectory.get("min_trajectory_adapter_handle_gap", ADAPTER_HANDLE_CLEARANCE_M))
    min_trajectory_left_grip_left_offset = float(trajectory.get("min_trajectory_left_grip_left_offset", 1.0))
    max_trajectory_right_grip_left_offset = float(trajectory.get("max_trajectory_right_grip_left_offset", -1.0))
    min_trajectory_absolute_grip_side_separation = float(
        trajectory.get("min_trajectory_absolute_grip_side_separation", 1.0)
    )
    min_trajectory_handle_lateral_alignment = float(trajectory.get("min_trajectory_handle_lateral_alignment", 1.0))
    min_trajectory_forearm_hand_alignment = float(trajectory.get("min_trajectory_forearm_hand_alignment", 1.0))
    max_trajectory_wrist_joint_abs = float(trajectory.get("max_trajectory_wrist_joint_abs", 0.0))
    max_trajectory_wrist_norm = float(trajectory.get("max_trajectory_wrist_norm", 0.0))
    min_trajectory_elbow_flexion = float(trajectory.get("min_trajectory_elbow_flexion", 0.0))
    min_trajectory_left_wrist_left_offset = float(trajectory.get("min_trajectory_left_wrist_left_offset", 1.0))
    max_trajectory_right_wrist_left_offset = float(trajectory.get("max_trajectory_right_wrist_left_offset", -1.0))
    min_trajectory_left_elbow_left_offset = float(trajectory.get("min_trajectory_left_elbow_left_offset", 1.0))
    max_trajectory_right_elbow_left_offset = float(trajectory.get("max_trajectory_right_elbow_left_offset", -1.0))
    min_trajectory_left_elbow_forward_offset = float(trajectory.get("min_trajectory_left_elbow_forward_offset", 1.0))
    min_trajectory_right_elbow_forward_offset = float(trajectory.get("min_trajectory_right_elbow_forward_offset", 1.0))
    max_speed = float(trajectory.get("max_actuated_joint_speed", 0.0))
    max_accel = float(trajectory.get("max_actuated_joint_accel", 0.0))
    max_torque_ratio = float(trajectory.get("max_estimated_torque_ratio", 0.0))
    if max_trajectory_grip_center_error > MAX_TRAJECTORY_GRIP_CENTER_ERROR_M:
        raise RuntimeError(
            f"trajectory grip center error too high: "
            f"{max_trajectory_grip_center_error:.3f}m > {MAX_TRAJECTORY_GRIP_CENTER_ERROR_M:.3f}m"
        )
    if max_trajectory_grip_reference_error > MAX_TRAJECTORY_GRIP_REFERENCE_ERROR_M:
        raise RuntimeError(
            f"trajectory grip rigid reference error too high: "
            f"{max_trajectory_grip_reference_error:.3f}m > {MAX_TRAJECTORY_GRIP_REFERENCE_ERROR_M:.3f}m"
        )
    if min_trajectory_grip_axis_alignment < MIN_GRIP_AXIS_ALIGNMENT:
        raise RuntimeError(
            f"trajectory grip axis alignment too low: {min_trajectory_grip_axis_alignment:.3f} < {MIN_GRIP_AXIS_ALIGNMENT:.3f}"
        )
    if min_trajectory_grip_normal_alignment < MIN_GRIP_NORMAL_ALIGNMENT:
        raise RuntimeError(
            f"trajectory grip normal alignment too low: "
            f"{min_trajectory_grip_normal_alignment:.3f} < {MIN_GRIP_NORMAL_ALIGNMENT:.3f}"
        )
    if min_trajectory_adapter_handle_gap < MIN_ADAPTER_HANDLE_GAP_M:
        raise RuntimeError(
            f"trajectory adapter handle gap too low: "
            f"{min_trajectory_adapter_handle_gap:.3f}m < {MIN_ADAPTER_HANDLE_GAP_M:.3f}m"
        )
    if min_trajectory_absolute_grip_side_separation < MIN_HAND_SIDE_SEPARATION_M:
        raise RuntimeError(
            f"trajectory absolute left/right grip separation too low: "
            f"{min_trajectory_absolute_grip_side_separation:.3f}m < {MIN_HAND_SIDE_SEPARATION_M:.3f}m"
        )
    if min_trajectory_handle_lateral_alignment < MIN_HANDLE_LATERAL_ALIGNMENT:
        raise RuntimeError(
            f"trajectory hammer handle is not lateral enough: "
            f"{min_trajectory_handle_lateral_alignment:.3f} < {MIN_HANDLE_LATERAL_ALIGNMENT:.3f}"
        )
    if min_trajectory_forearm_hand_alignment < MIN_FOREARM_HAND_ALIGNMENT:
        raise RuntimeError(
            f"trajectory forearm-hand alignment too low: "
            f"{min_trajectory_forearm_hand_alignment:.3f} < {MIN_FOREARM_HAND_ALIGNMENT:.3f}"
        )
    if max_trajectory_wrist_joint_abs > MAX_WRIST_JOINT_ABS_RAD:
        raise RuntimeError(
            f"trajectory wrist joint angle too high: "
            f"{max_trajectory_wrist_joint_abs:.3f}rad > {MAX_WRIST_JOINT_ABS_RAD:.3f}rad"
        )
    if max_trajectory_wrist_norm > MAX_WRIST_NORM_RAD:
        raise RuntimeError(
            f"trajectory wrist norm too high: "
            f"{max_trajectory_wrist_norm:.3f}rad > {MAX_WRIST_NORM_RAD:.3f}rad"
        )
    if min_trajectory_left_wrist_left_offset < MIN_LEFT_WRIST_LEFT_OFFSET_M:
        raise RuntimeError(
            f"trajectory left wrist crossed to wrong side: "
            f"{min_trajectory_left_wrist_left_offset:.3f}m < {MIN_LEFT_WRIST_LEFT_OFFSET_M:.3f}m"
        )
    if max_trajectory_right_wrist_left_offset > MAX_RIGHT_WRIST_LEFT_OFFSET_M:
        raise RuntimeError(
            f"trajectory right wrist crossed to wrong side: "
            f"{max_trajectory_right_wrist_left_offset:.3f}m > {MAX_RIGHT_WRIST_LEFT_OFFSET_M:.3f}m"
        )
    if min_trajectory_left_elbow_left_offset < MIN_LEFT_ELBOW_LEFT_OFFSET_M:
        raise RuntimeError(
            f"trajectory left elbow crossed to wrong side: "
            f"{min_trajectory_left_elbow_left_offset:.3f}m < {MIN_LEFT_ELBOW_LEFT_OFFSET_M:.3f}m"
        )
    if max_trajectory_right_elbow_left_offset > MAX_RIGHT_ELBOW_LEFT_OFFSET_M:
        raise RuntimeError(
            f"trajectory right elbow crossed to wrong side: "
            f"{max_trajectory_right_elbow_left_offset:.3f}m > {MAX_RIGHT_ELBOW_LEFT_OFFSET_M:.3f}m"
        )
    if min_trajectory_left_elbow_forward_offset < MIN_ELBOW_FORWARD_OFFSET_M:
        raise RuntimeError(
            f"trajectory left elbow moved behind torso: "
            f"{min_trajectory_left_elbow_forward_offset:.3f}m < {MIN_ELBOW_FORWARD_OFFSET_M:.3f}m"
        )
    if min_trajectory_right_elbow_forward_offset < MIN_ELBOW_FORWARD_OFFSET_M:
        raise RuntimeError(
            f"trajectory right elbow moved behind torso: "
            f"{min_trajectory_right_elbow_forward_offset:.3f}m < {MIN_ELBOW_FORWARD_OFFSET_M:.3f}m"
        )
    if min_trajectory_elbow_flexion < MIN_ELBOW_FLEXION_WARNING_RAD:
        raise RuntimeError(
            f"trajectory elbow hyperextension too high: "
            f"{min_trajectory_elbow_flexion:.3f}rad < {MIN_ELBOW_FLEXION_WARNING_RAD:.3f}rad"
        )
    collision = collision_watch_metrics(model, cycle_seconds)
    watched_negative_contact_pairs = int(collision["watched_negative_contact_pairs"])
    torso_wrist_negative_contact_pairs = int(collision["torso_wrist_negative_contact_pairs"])
    min_watched_contact_distance = float(collision["min_watched_contact_distance"])
    min_torso_wrist_contact_distance = float(collision["min_torso_wrist_contact_distance"])
    if watched_negative_contact_pairs:
        raise RuntimeError(
            f"watched collision penetration detected: {watched_negative_contact_pairs} pairs, "
            f"min_dist={min_watched_contact_distance:.5f}m"
        )
    if torso_wrist_negative_contact_pairs:
        raise RuntimeError(
            f"torso-wrist penetration detected: {torso_wrist_negative_contact_pairs} pairs, "
            f"min_dist={min_torso_wrist_contact_distance:.5f}m"
        )
    if max_foot_z_drift > MAX_FOOT_Z_DRIFT_M:
        raise RuntimeError(f"foot z drift too high: {max_foot_z_drift:.3f}m > {MAX_FOOT_Z_DRIFT_M:.3f}m")
    if max_speed > MAX_ACTUATED_JOINT_SPEED_RAD_S + 1e-6:
        print(
            f"WARNING joint speed high: {max_speed:.3f}rad/s > {MAX_ACTUATED_JOINT_SPEED_RAD_S:.3f}rad/s "
            f"at {trajectory.get('worst_speed_joint', '')}"
        )
    if max_accel > MAX_ACTUATED_JOINT_ACCEL_RAD_S2 + 1e-6:
        print(
            f"WARNING joint accel high: {max_accel:.3f}rad/s^2 > {MAX_ACTUATED_JOINT_ACCEL_RAD_S2:.3f}rad/s^2 "
            f"at {trajectory.get('worst_accel_joint', '')}"
        )
    if max_torque_ratio > MAX_ESTIMATED_TORQUE_RATIO:
        print(
            f"WARNING estimated torque ratio high: {max_torque_ratio:.3f} > {MAX_ESTIMATED_TORQUE_RATIO:.3f} "
            f"at {trajectory.get('worst_torque_joint', '')}"
        )

    if render_smoke:
        renderer = mujoco.Renderer(model, height=360, width=640)
        apply_pose(model, data, interpolate_keyframes(0.48))
        rendered = {camera: render_frame(model, data, renderer, camera=camera) for camera in DIAGNOSTIC_CAMERAS}
        renderer.close()
        for camera, rgb in rendered.items():
            if rgb.shape != (360, 640, 3):
                raise RuntimeError(f"unexpected {camera} render shape: {rgb.shape}")
            if float(rgb.std()) < 2.0:
                raise RuntimeError(f"{camera} render appears blank")

    print("mochitsuki demo check OK")
    print(f"scene: {SCENE_PATH}")
    print(f"nq={model.nq}, nv={model.nv}, nu={model.nu}, bodies={model.nbody}")
    target_local = robot_local_point(KEYS[0], np.array([HAMMER_TARGET_WORLD[0], HAMMER_TARGET_WORLD[1], 0.0]))
    print(f"target_local_forward={target_local[0]:.3f}m")
    print(f"target_local_left={target_local[1]:.3f}m")
    print(f"max_grip_center_error={max_grip_center_error:.3f}m")
    print(f"max_grip_reference_error={max_grip_reference_error:.3f}m")
    print(f"min_grip_axis_alignment={min_grip_axis_alignment:.3f}")
    print(f"min_grip_normal_alignment={min_grip_normal_alignment:.3f}")
    print(f"min_adapter_handle_gap={min_adapter_handle_gap:.3f}m")
    print(f"min_left_grip_anatomical_left_offset={min_left_palm_robot_left_offset:.3f}m")
    print(f"max_right_grip_anatomical_left_offset={max_right_palm_robot_left_offset:.3f}m")
    print(f"min_adapter_radial_side_separation={min_palm_side_separation:.3f}m")
    print(f"min_absolute_grip_side_separation={min_absolute_grip_side_separation:.3f}m")
    print(f"min_handle_lateral_alignment={min_handle_lateral_alignment:.3f}")
    print(f"min_left_wrist_left_offset={min_left_wrist_left_offset:.3f}m")
    print(f"max_right_wrist_left_offset={max_right_wrist_left_offset:.3f}m")
    print(f"min_left_elbow_left_offset={min_left_elbow_left_offset:.3f}m")
    print(f"max_right_elbow_left_offset={max_right_elbow_left_offset:.3f}m")
    print(f"min_left_elbow_forward_offset={min_left_elbow_forward_offset:.3f}m")
    print(f"min_right_elbow_forward_offset={min_right_elbow_forward_offset:.3f}m")
    print(f"min_forearm_hand_alignment={min_forearm_hand_alignment:.3f}")
    print(f"max_wrist_joint_abs={max_wrist_joint_abs:.3f}rad")
    print(f"max_wrist_norm={max_wrist_norm:.3f}rad")
    print(f"min_elbow_flexion={min_elbow_flexion:.3f}rad")
    print(f"max_com_foot_mid_offset={max_com_foot_mid_offset:.3f}m")
    print(f"min_foot_separation={min_foot_separation:.3f}m")
    print(f"max_trajectory_grip_center_error={max_trajectory_grip_center_error:.3f}m")
    print(f"max_trajectory_grip_reference_error={max_trajectory_grip_reference_error:.3f}m")
    print(f"min_trajectory_grip_axis_alignment={min_trajectory_grip_axis_alignment:.3f}")
    print(f"min_trajectory_grip_normal_alignment={min_trajectory_grip_normal_alignment:.3f}")
    print(f"min_trajectory_adapter_handle_gap={min_trajectory_adapter_handle_gap:.3f}m")
    print(f"min_trajectory_left_grip_anatomical_left_offset={min_trajectory_left_grip_left_offset:.3f}m")
    print(f"max_trajectory_right_grip_anatomical_left_offset={max_trajectory_right_grip_left_offset:.3f}m")
    print(f"min_trajectory_absolute_grip_side_separation={min_trajectory_absolute_grip_side_separation:.3f}m")
    print(f"min_trajectory_handle_lateral_alignment={min_trajectory_handle_lateral_alignment:.3f}")
    print(f"min_trajectory_forearm_hand_alignment={min_trajectory_forearm_hand_alignment:.3f}")
    print(f"max_trajectory_wrist_joint_abs={max_trajectory_wrist_joint_abs:.3f}rad")
    print(f"max_trajectory_wrist_norm={max_trajectory_wrist_norm:.3f}rad")
    print(f"min_trajectory_elbow_flexion={min_trajectory_elbow_flexion:.3f}rad")
    print(f"min_trajectory_left_wrist_left_offset={min_trajectory_left_wrist_left_offset:.3f}m")
    print(f"max_trajectory_right_wrist_left_offset={max_trajectory_right_wrist_left_offset:.3f}m")
    print(f"min_trajectory_left_elbow_left_offset={min_trajectory_left_elbow_left_offset:.3f}m")
    print(f"max_trajectory_right_elbow_left_offset={max_trajectory_right_elbow_left_offset:.3f}m")
    print(f"min_trajectory_left_elbow_forward_offset={min_trajectory_left_elbow_forward_offset:.3f}m")
    print(f"min_trajectory_right_elbow_forward_offset={min_trajectory_right_elbow_forward_offset:.3f}m")
    print(f"max_foot_z_drift={max_foot_z_drift:.3f}m")
    print(f"max_actuated_joint_speed={max_speed:.3f}rad/s at {trajectory.get('worst_speed_joint', '')}")
    print(f"max_actuated_joint_accel={max_accel:.3f}rad/s^2 at {trajectory.get('worst_accel_joint', '')}")
    print(f"max_estimated_torque_ratio={max_torque_ratio:.3f} at {trajectory.get('worst_torque_joint', '')}")
    print(f"watched_negative_contact_pairs={watched_negative_contact_pairs}")
    print(f"torso_wrist_negative_contact_pairs={torso_wrist_negative_contact_pairs}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unitree G1 mochitsuki MuJoCo demo")
    parser.add_argument("--mode", choices=["check", "render", "viewer", "diagnose"], default="check")
    parser.add_argument("--output", type=Path, default=DEFAULT_RENDER_DIR / "g1_mochitsuki_demo.mp4")
    parser.add_argument("--diagnostic-dir", type=Path, default=DEFAULT_RENDER_DIR / "diagnostics_latest")
    parser.add_argument("--diagnostic-samples", type=int, default=52)
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--cycle-seconds", type=float, default=CYCLE_SECONDS)
    parser.add_argument(
        "--camera-layout",
        choices=["multi", "single"],
        default="multi",
        help="multi renders a grid of all validation cameras; single renders one camera only",
    )
    parser.add_argument(
        "--camera",
        choices=list(DIAGNOSTIC_CAMERAS),
        default="mochitsuki_camera",
        help="camera used when --camera-layout single",
    )
    parser.add_argument("--save-keyframes", action="store_true")
    parser.add_argument("--render-smoke", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "check":
        run_check(render_smoke=args.render_smoke, cycle_seconds=args.cycle_seconds)
    elif args.mode == "render":
        cameras = VIDEO_CAMERAS if args.camera_layout == "multi" else (args.camera,)
        write_video(
            args.output,
            args.duration,
            args.fps,
            args.width,
            args.height,
            args.save_keyframes,
            args.cycle_seconds,
            cameras,
        )
    elif args.mode == "viewer":
        run_viewer(args.duration, args.fps, args.cycle_seconds)
    elif args.mode == "diagnose":
        write_diagnostic_sheets(
            args.diagnostic_dir,
            args.cycle_seconds,
            args.fps,
            args.width,
            args.height,
            args.diagnostic_samples,
        )


if __name__ == "__main__":
    main()
