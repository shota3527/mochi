#!/usr/bin/env python3
"""Replay a configured joint trajectory through DDS for sim inspection."""

from __future__ import annotations

import argparse
import json
import math
import select
import sys
import termios
import time
import tty
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import mujoco
import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backends.sdk2_python_backend import G1Sdk2Backend
from core.safety_filter import clamp_joint_limits
from sim.run_sim_controller import UNITREE_G1_29DOF_XML

GAIN_PROFILES_CONFIG = PROJECT_ROOT / "configs" / "gain_profiles.yaml"
SHUTDOWN_COMMAND_REPEATS = 20
ARM_SDK_JOINT_INDICES = tuple(range(12, 29))
WAIST_JOINT_INDICES = (12, 13, 14)
RELEASE_POSE_NAME = "real_release_standby_20260531_094252"
INITIAL_RAMP_S = 2.0
RESAMPLED_TRAJECTORY_WAYPOINTS = 1000
WAIST_BIAS_JOINTS = (
    "waist_yaw",
    "waist_roll",
    "waist_pitch",
)
IMPACT_GAIN_JOINT_NAMES = (
    "left_shoulder_pitch",
    "left_shoulder_roll",
    "left_shoulder_yaw",
    "left_elbow",
    "left_wrist_roll",
    "left_wrist_pitch",
    "left_wrist_yaw",
    "right_shoulder_pitch",
    "right_shoulder_roll",
    "right_shoulder_yaw",
    "right_elbow",
    "right_wrist_roll",
    "right_wrist_pitch",
    "right_wrist_yaw",
)
DEFAULT_IMPACT_PHASE_S = 0.2


class StopTrajectory(Exception):
    """Raised when the operator asks a looped trajectory to stop in place."""

    def __init__(self, q_current: np.ndarray):
        super().__init__("trajectory stopped by operator")
        self.q_current = q_current


class KeyStopper:
    def __init__(self, prompt: str, keys: set[str]):
        self.prompt = prompt
        self.keys = keys
        self.fd = None
        self.old_settings = None

    def __enter__(self):
        print(self.prompt, flush=True)
        if sys.stdin.isatty():
            self.fd = sys.stdin.fileno()
            self.old_settings = termios.tcgetattr(self.fd)
            tty.setcbreak(self.fd)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.fd is not None and self.old_settings is not None:
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old_settings)

    def pressed(self) -> bool:
        if self.fd is None:
            return False
        readable, _, _ = select.select([sys.stdin], [], [], 0.0)
        if not readable:
            return False
        return sys.stdin.read(1) in self.keys


class ReplayState(Enum):
    START = "start"
    STANDBY = "standby"
    HAMMERING = "hammering"
    STOPPED = "stopped"
    RELEASE = "release"
    HOLD_EXIT = "hold_exit"
    DONE = "done"


ALLOWED_TRANSITIONS = {
    ReplayState.START: {ReplayState.STANDBY, ReplayState.DONE},
    ReplayState.STANDBY: {ReplayState.HAMMERING, ReplayState.RELEASE, ReplayState.HOLD_EXIT},
    ReplayState.HAMMERING: {ReplayState.STOPPED},
    ReplayState.STOPPED: {ReplayState.STANDBY, ReplayState.RELEASE},
    ReplayState.RELEASE: {ReplayState.DONE},
    ReplayState.HOLD_EXIT: {ReplayState.DONE},
}


def transition_state(current: ReplayState, next_state: ReplayState) -> ReplayState:
    allowed = ALLOWED_TRANSITIONS.get(current, set())
    if next_state not in allowed:
        raise RuntimeError(f"Invalid replay state transition: {current.value} -> {next_state.value}")
    print(f"state transition: {current.value} -> {next_state.value}", flush=True)
    return next_state


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def merge_dicts(base: dict, override: dict) -> dict:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


def infer_profile(path: Path, config: dict) -> str:
    if "profile" in config:
        return str(config["profile"])
    if "real" in path.stem:
        return "real"
    return "sim"


def config_reference_path(config_path: Path, reference: str) -> Path:
    referenced = Path(reference)
    if referenced.is_absolute():
        return referenced
    candidate = config_path.parent / referenced
    if candidate.exists():
        return candidate
    return PROJECT_ROOT / referenced


def load_run_config_map(path: Path, seen: set[Path] | None = None) -> dict:
    config_path = path if path.is_absolute() else PROJECT_ROOT / path
    config_path = config_path.resolve()
    seen = set() if seen is None else seen
    if config_path in seen:
        chain = " -> ".join(str(item) for item in [*seen, config_path])
        raise SystemExit(f"Run config base_config cycle detected: {chain}")
    seen.add(config_path)

    config = load_yaml(config_path)
    if not isinstance(config, dict):
        raise SystemExit(f"{config_path} must contain a YAML map.")

    base_ref = config.get("base_config")
    if not base_ref:
        return config
    base_config = load_run_config_map(config_reference_path(config_path, str(base_ref)), seen)
    override = {key: value for key, value in config.items() if key != "base_config"}
    return merge_dicts(base_config, override)


def select_run_profile(config_path: Path, config: dict) -> dict:
    profile = infer_profile(config_path, config)
    if "basic" in config:
        basic = config.get("basic") or {}
        override = config.get(profile) or {}
        merged = merge_dicts(basic, override)
    else:
        merged = {key: value for key, value in config.items() if key != "base_config"}
    merged["profile"] = profile
    return merged


def load_run_config(path: Path) -> dict:
    config_path = path if path.is_absolute() else PROJECT_ROOT / path
    config = load_run_config_map(config_path)
    return select_run_profile(config_path, config)


def smoothstep(x: float) -> float:
    x = min(1.0, max(0.0, x))
    return x * x * (3.0 - 2.0 * x)


def trapezoid_phase(x: float, accel_fraction: float) -> float:
    x = min(1.0, max(0.0, float(x)))
    accel = min(0.49, max(0.0, float(accel_fraction)))
    if accel <= 0.0:
        return x

    cruise = 1.0 - 2.0 * accel
    peak_velocity = 1.0 / (accel + cruise)
    if x < accel:
        return 0.5 * peak_velocity * x * x / accel
    if x <= accel + cruise:
        return 0.5 * peak_velocity * accel + peak_velocity * (x - accel)

    remaining = 1.0 - x
    return 1.0 - 0.5 * peak_velocity * remaining * remaining / accel


def time_plan_phase(x: float, config: dict | None) -> float:
    config = config or {}
    plan_type = str(config.get("type", "acceleration")).lower()
    x = min(1.0, max(0.0, float(x)))
    if plan_type in ("linear", "uniform"):
        return x
    if plan_type in ("acceleration", "constant_acceleration"):
        return x * x
    if plan_type == "trapezoid":
        return trapezoid_phase(x, float(config.get("accel_fraction", 0.2)))
    raise SystemExit(f"Unknown time_plan.type {plan_type!r}. Use linear, acceleration, or trapezoid.")


def joint_key_to_config_name(key: str) -> str:
    return key.removesuffix("_joint")


def apply_joint_block(q_base: np.ndarray, joints_rad: dict, name_to_index: dict[str, int]) -> np.ndarray:
    q = q_base.copy()
    for joint_key, value in joints_rad.items():
        name = joint_key_to_config_name(joint_key)
        if name not in name_to_index:
            raise SystemExit(f"Trajectory joint {joint_key!r} does not match joint config.")
        q[name_to_index[name]] = float(value)
    return q


def resample_waypoints(q_waypoints: list[np.ndarray], count: int = RESAMPLED_TRAJECTORY_WAYPOINTS) -> list[np.ndarray]:
    if len(q_waypoints) < 2:
        raise SystemExit("Trajectory needs at least two waypoints.")
    if count < 2:
        raise ValueError("count must be at least 2.")

    source = np.asarray(q_waypoints, dtype=float)
    positions = np.linspace(0.0, len(source) - 1, int(count))
    resampled = []
    for pos in positions:
        lower = int(math.floor(float(pos)))
        upper = min(lower + 1, len(source) - 1)
        alpha = float(pos - lower)
        resampled.append(source[lower] + alpha * (source[upper] - source[lower]))
    return resampled


def with_joint_value(q_waypoints: list[np.ndarray], joint_index: int, value: float) -> list[np.ndarray]:
    shifted = []
    for q in q_waypoints:
        q_next = np.asarray(q, dtype=float).copy()
        q_next[joint_index] = float(value)
        shifted.append(q_next)
    return shifted


def sample_waypoint_path(q_waypoints: list[np.ndarray], phase: float) -> np.ndarray:
    if len(q_waypoints) < 2:
        raise ValueError("q_waypoints must contain at least two points.")
    phase = min(1.0, max(0.0, float(phase)))
    pos = phase * (len(q_waypoints) - 1)
    lower = int(math.floor(pos))
    upper = min(lower + 1, len(q_waypoints) - 1)
    alpha = pos - lower
    return q_waypoints[lower] + alpha * (q_waypoints[upper] - q_waypoints[lower])


def trim_waypoint_path(q_waypoints: list[np.ndarray], replay_length: float) -> list[np.ndarray]:
    replay_length = min(1.0, max(0.01, float(replay_length)))
    if replay_length >= 1.0:
        return [np.asarray(q, dtype=float).copy() for q in q_waypoints]
    count = max(2, int(round(len(q_waypoints) * replay_length)))
    trimmed = [np.asarray(q, dtype=float).copy() for q in q_waypoints[:count]]
    trimmed[-1] = sample_waypoint_path(q_waypoints, replay_length)
    return trimmed


def velocity_feedforward(ctx: ReplayContext, q_safe: np.ndarray, q_prev: np.ndarray) -> np.ndarray:
    return float(ctx.velocity_ff_gain) * (np.asarray(q_safe, dtype=float) - np.asarray(q_prev, dtype=float)) / ctx.dt


def latest_q_or_fallback(ctx: ReplayContext, fallback: np.ndarray) -> np.ndarray:
    latest = ctx.backend.latest_state()
    return latest.q.copy() if latest is not None else np.asarray(fallback, dtype=float).copy()


def trajectory_base_q(
    current_q: np.ndarray,
    trajectory: dict,
    poses: dict,
    name_to_index: dict[str, int],
) -> np.ndarray:
    q_base = current_q.copy()
    base_pose_name = trajectory.get("base_pose")
    if not base_pose_name:
        return q_base
    if base_pose_name not in poses:
        raise SystemExit(f"Trajectory base_pose {base_pose_name!r} is missing from configs/poses.yaml.")
    return apply_joint_block(q_base, poses[base_pose_name].get("joints_rad", {}), name_to_index)


def load_gain_profile_vectors(joints: list[dict], profile_name: str) -> tuple[np.ndarray, np.ndarray]:
    config = load_yaml(GAIN_PROFILES_CONFIG)
    profiles = config.get("profiles", {})
    if profile_name not in profiles:
        available = ", ".join(sorted(profiles))
        raise SystemExit(f"Gain profile {profile_name!r} is missing from {GAIN_PROFILES_CONFIG}. Available: {available}")

    profile = profiles[profile_name] or {}
    joint_gains = profile.get("joints", {})
    if not isinstance(joint_gains, dict):
        raise SystemExit(f"Gain profile {profile_name!r} must contain a joints map.")

    joint_names = {joint["name"] for joint in joints}
    missing = sorted(joint_names - set(joint_gains))
    unknown = sorted(set(joint_gains) - joint_names)
    if missing:
        raise SystemExit(f"Gain profile {profile_name!r} is missing joint(s): {', '.join(missing)}")
    if unknown:
        raise SystemExit(f"Gain profile {profile_name!r} has unknown joint(s): {', '.join(unknown)}")

    kp = np.zeros(len(joints), dtype=float)
    kd = np.zeros(len(joints), dtype=float)
    for joint in joints:
        name = joint["name"]
        gains = joint_gains[name]
        try:
            kp[joint["index"]] = float(gains["kp"])
            kd[joint["index"]] = float(gains["kd"])
        except (KeyError, TypeError, ValueError) as exc:
            raise SystemExit(f"Gain profile {profile_name!r} has invalid gain for joint {name!r}: {gains}") from exc

    return kp, kd


def merge_impact_gains(
    *,
    joints: list[dict],
    base_kp: np.ndarray,
    base_kd: np.ndarray,
    impact_profile_name: str = "impact",
) -> tuple[np.ndarray, np.ndarray]:
    impact_kp, impact_kd = load_gain_profile_vectors(joints, impact_profile_name)
    name_to_index = {joint["name"]: joint["index"] for joint in joints}
    kp = np.asarray(base_kp, dtype=float).copy()
    kd = np.asarray(base_kd, dtype=float).copy()
    for name in IMPACT_GAIN_JOINT_NAMES:
        idx = name_to_index[name]
        kp[idx] = impact_kp[idx]
        kd[idx] = impact_kd[idx]
    return kp, kd


def max_error_index(err: np.ndarray, indices: np.ndarray) -> int:
    selected = np.asarray(err, dtype=float)[indices]
    return int(indices[int(np.argmax(selected))])


class ModelSelectedJointGravityFeedforward:
    """Quasi-static gravity feedforward from the official G1 model bias torque."""

    def __init__(
        self,
        joints: list[dict],
        scale: float,
        max_tau_nm: float,
    ):
        self.model = mujoco.MjModel.from_xml_path(str(UNITREE_G1_29DOF_XML))
        self.data = mujoco.MjData(self.model)
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        self.nominal_qpos = self.data.qpos.copy()

        self.num_motors = len(joints)
        self.mask = self._selected_joint_mask(joints)
        self.qpos_addrs = []
        self.dof_addrs = []
        for joint in joints:
            joint_id = self.model.joint(f"{joint['name']}_joint").id
            self.qpos_addrs.append(int(self.model.jnt_qposadr[joint_id]))
            self.dof_addrs.append(int(self.model.jnt_dofadr[joint_id]))
        self.qpos_addrs = np.asarray(self.qpos_addrs, dtype=int)
        self.dof_addrs = np.asarray(self.dof_addrs, dtype=int)
        self.ctrl_min = self.model.actuator_ctrlrange[: self.num_motors, 0].copy()
        self.ctrl_max = self.model.actuator_ctrlrange[: self.num_motors, 1].copy()
        self.scale = float(scale)
        self.max_tau_nm = float(max_tau_nm)

    def _selected_joint_mask(self, joints: list[dict]) -> np.ndarray:
        mask = np.zeros(len(joints), dtype=bool)
        for joint in joints:
            name = joint["name"]
            selected = (
                name.startswith("waist_")
                or name.startswith(("left_shoulder_", "right_shoulder_"))
                or name in ("left_elbow", "right_elbow")
            )
            mask[joint["index"]] = selected
        return mask

    def torque(self, q_motor: np.ndarray) -> np.ndarray:
        self.data.qpos[:] = self.nominal_qpos
        self.data.qpos[self.qpos_addrs] = np.asarray(q_motor, dtype=float)
        self.data.qvel[:] = 0.0
        self.data.qacc[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

        tau = self.scale * self.data.qfrc_bias[self.dof_addrs]
        tau = np.where(self.mask, tau, 0.0)
        tau = np.clip(tau, -self.max_tau_nm, self.max_tau_nm)
        tau = np.clip(tau, self.ctrl_min, self.ctrl_max)
        return tau


class WaistAdaptiveTauBias:
    """Slow waist-only torque bias integrator for steady tracking error."""

    def __init__(
        self,
        joints: list[dict],
        q_min: np.ndarray,
        q_max: np.ndarray,
        config: dict,
    ):
        self.enabled = bool(config.get("enabled", True))
        self.num_motors = len(joints)
        self.index_to_name = {joint["index"]: joint["name"] for joint in joints}
        self.name_to_index = {joint["name"]: joint["index"] for joint in joints}
        self.indices = np.asarray([self.name_to_index[name] for name in WAIST_BIAS_JOINTS], dtype=int)
        self.q_min = np.asarray(q_min, dtype=float)
        self.q_max = np.asarray(q_max, dtype=float)
        self.ki = np.zeros(self.num_motors, dtype=float)
        ki = float(config.get("ki_nm_per_rad_s", 0.4))
        for name in WAIST_BIAS_JOINTS:
            self.ki[self.name_to_index[name]] = ki
        self.max_tau_nm = abs(float(config.get("max_tau_nm", 8.0)))
        self.integrate_error_max_rad = 0.20
        self.near_limit_margin_rad = 0.08
        self.leak_time_s = 3.0
        self.bias = np.zeros(self.num_motors, dtype=float)

    def reset(self) -> None:
        self.bias[:] = 0.0

    def decay(self, dt: float) -> None:
        if self.leak_time_s > 0.0:
            self.bias *= math.exp(-float(dt) / self.leak_time_s)

    def update(
        self,
        q_cmd: np.ndarray,
        q_meas: np.ndarray | None,
        dt: float,
        *,
        integrate: bool = True,
    ) -> np.ndarray:
        self.decay(dt)
        if not self.enabled or q_meas is None or not integrate:
            return self.bias.copy()

        q_cmd = np.asarray(q_cmd, dtype=float)
        q_meas = np.asarray(q_meas, dtype=float)
        err = q_cmd - q_meas
        for idx in self.indices:
            near_limit = (
                q_cmd[idx] < self.q_min[idx] + self.near_limit_margin_rad
                or q_cmd[idx] > self.q_max[idx] - self.near_limit_margin_rad
            )
            if near_limit or abs(float(err[idx])) > self.integrate_error_max_rad:
                continue
            self.bias[idx] += self.ki[idx] * err[idx] * float(dt)

        self.bias[self.indices] = np.clip(self.bias[self.indices], -self.max_tau_nm, self.max_tau_nm)
        non_waist = np.ones(self.num_motors, dtype=bool)
        non_waist[self.indices] = False
        self.bias[non_waist] = 0.0
        return self.bias.copy()

@dataclass
class ReplayContext:
    backend: G1Sdk2Backend
    q_min: np.ndarray
    q_max: np.ndarray
    kp: np.ndarray
    kd: np.ndarray
    mode_machine: int | None
    limit_margin_rad: float
    rate_hz: float
    velocity_ff_gain: float
    index_to_name: dict[int, str]
    error_indices: np.ndarray
    log: list[dict]
    gravity_comp: ModelSelectedJointGravityFeedforward | None
    adaptive_tau_bias: WaistAdaptiveTauBias | None
    impact_kp: np.ndarray | None = None
    impact_kd: np.ndarray | None = None
    last_q_cmd: np.ndarray | None = None
    last_tau_gravity: np.ndarray | None = None
    last_tau_bias: np.ndarray | None = None
    report_step: int = 0

    @property
    def dt(self) -> float:
        return 1.0 / float(self.rate_hz)

    @property
    def report_interval_steps(self) -> int:
        return max(1, int(self.rate_hz))


def run_waypoint_path(
    *,
    ctx: ReplayContext,
    q_waypoints: list[np.ndarray],
    q_prev: np.ndarray,
    duration_s: float,
    label_prefix: str,
    time_plan: dict | None,
    stop_checker=None,
    integrate_bias: bool = True,
    kp: np.ndarray | None = None,
    kd: np.ndarray | None = None,
) -> np.ndarray:
    steps = max(1, int(round(float(duration_s) / ctx.dt)))
    for step in range(steps + 1):
        phase = time_plan_phase(step / steps, time_plan)
        q_des = sample_waypoint_path(q_waypoints, phase)
        q_safe, tau = prepare_command(ctx, q_des, q_prev, integrate_bias=integrate_bias)
        dq_des = velocity_feedforward(ctx, q_safe, q_prev)
        publish_and_report(
            ctx,
            q_safe,
            tau,
            label=f"{label_prefix} phase={phase:.3f}",
            force_report=step == steps,
            kp=kp,
            kd=kd,
            dq_des=dq_des,
        )
        q_prev = q_safe
        time.sleep(ctx.dt)
        if stop_checker is not None and stop_checker():
            print(f"{label_prefix}: stop key received; stopping at current command.", flush=True)
            raise StopTrajectory(q_prev.copy())
    return q_prev


def run_impact_phase(
    *,
    ctx: ReplayContext,
    q_prev: np.ndarray,
    duration_s: float,
) -> np.ndarray:
    print(f"impact phase: hold {duration_s:.2f}s with arm impact gains", flush=True)
    steps = max(1, int(round(float(duration_s) / ctx.dt)))
    kp = ctx.impact_kp if ctx.impact_kp is not None else ctx.kp
    kd = ctx.impact_kd if ctx.impact_kd is not None else ctx.kd
    q_hold = q_prev.copy()
    dq_hold = np.zeros_like(q_hold)
    for step in range(steps):
        q_safe, tau = prepare_command(ctx, q_hold, q_prev)
        publish_and_report(
            ctx,
            q_safe,
            tau,
            label="impact_hold",
            force_report=step == steps - 1,
            kp=kp,
            kd=kd,
            dq_des=dq_hold,
        )
        q_prev = q_safe
        time.sleep(ctx.dt)
    return q_prev


def run_post_impact_transition(
    *,
    ctx: ReplayContext,
    q_impact: np.ndarray,
    duration_s: float,
    stop_checker=None,
) -> np.ndarray:
    if duration_s <= 0.0:
        return q_impact

    latest = ctx.backend.latest_state()
    q_start = latest.q.copy() if latest is not None else q_impact.copy()
    q_prev = q_start.copy()
    steps = max(1, int(round(float(duration_s) / ctx.dt)))
    print(f"post-impact transition: interpolate current joints back to impact point over {duration_s:.2f}s", flush=True)
    for step in range(steps + 1):
        alpha = step / steps
        q_des = q_start + alpha * (q_impact - q_start)
        q_safe, tau = prepare_command(ctx, q_des, q_prev)
        dq_des = velocity_feedforward(ctx, q_safe, q_prev)
        publish_and_report(
            ctx,
            q_safe,
            tau,
            label=f"post_impact_transition alpha={alpha:.3f}",
            force_report=step == steps,
            dq_des=dq_des,
        )
        q_prev = q_safe
        time.sleep(ctx.dt)
        if stop_checker is not None and stop_checker():
            print("post-impact transition: stop key received; stopping at current command.", flush=True)
            raise StopTrajectory(q_prev.copy())
    return q_prev


def run_hammer_loop(
    *,
    ctx: ReplayContext,
    q_prev: np.ndarray,
    q_waypoints: list[np.ndarray],
    duration_s: float,
    return_duration_s: float,
    time_plan: dict | None,
    return_time_plan: dict | None,
    impact_phase_s: float,
    transition_phase_s: float,
    cycles: int,
) -> np.ndarray:
    cycles_done = 0
    with KeyStopper("State 2/4 hammering loop: press [s] to stop at the current command.", {"s"}) as stopper:
        try:
            while cycles <= 0 or cycles_done < cycles:
                cycles_done += 1
                print(f"cycle={cycles_done} forward")
                q_prev = run_waypoint_path(
                    ctx=ctx,
                    q_waypoints=q_waypoints,
                    q_prev=q_prev,
                    duration_s=duration_s,
                    label_prefix=f"cycle={cycles_done} forward",
                    time_plan=time_plan,
                    stop_checker=stopper.pressed,
                )
                q_prev = run_impact_phase(ctx=ctx, q_prev=q_prev, duration_s=impact_phase_s)
                q_prev = run_post_impact_transition(
                    ctx=ctx,
                    q_impact=q_prev,
                    duration_s=transition_phase_s,
                    stop_checker=stopper.pressed,
                )
                print(f"cycle={cycles_done} backward")
                q_prev = run_waypoint_path(
                    ctx=ctx,
                    q_waypoints=list(reversed(q_waypoints)),
                    q_prev=q_prev,
                    duration_s=return_duration_s,
                    label_prefix=f"cycle={cycles_done} backward",
                    time_plan=return_time_plan,
                    stop_checker=stopper.pressed,
                )
        except StopTrajectory as exc:
            q_prev = exc.q_current
    return q_prev


def publish_initial_ramp(
    *,
    ctx: ReplayContext,
    q_start: np.ndarray,
    q_target: np.ndarray,
    min_duration_s: float,
) -> np.ndarray:
    q_prev = q_start.copy()
    ramp_steps = max(1, int(math.ceil(float(min_duration_s) / ctx.dt)))

    def publish(q_des: np.ndarray, step: int, alpha: float) -> np.ndarray:
        q_safe, tau = prepare_command(ctx, q_des, q_prev, tau_scale=alpha)
        publish_and_report(ctx, q_safe, tau, label="initial_ramp", dq_des=velocity_feedforward(ctx, q_safe, q_prev))
        time.sleep(ctx.dt)
        return q_safe

    for step in range(ramp_steps):
        alpha = smoothstep((step + 1) / ramp_steps)
        q_prev = publish(q_start + alpha * (q_target - q_start), step, alpha)

    extra_step = ramp_steps
    while float(np.max(np.abs(q_target - q_prev))) > 1e-4:
        q_prev = publish(q_target, extra_step, alpha=1.0)
        extra_step += 1

    return q_prev


def reverse_to_standby(
    *,
    ctx: ReplayContext,
    q_prev: np.ndarray,
    q_waypoints: list[np.ndarray],
    return_duration_s: float,
) -> np.ndarray:
    print("manual reverse: returning to hammer standby along trajectory waypoints", flush=True)
    nearest_index = int(np.argmin([np.max(np.abs(q_prev - q)) for q in q_waypoints]))
    reverse_waypoints = [q_prev] + list(reversed(q_waypoints[: nearest_index + 1]))
    return run_waypoint_path(
        ctx=ctx,
        q_prev=q_prev,
        q_waypoints=reverse_waypoints,
        duration_s=return_duration_s,
        label_prefix="return_to_hammer_standby",
        time_plan={"type": "linear"},
        integrate_bias=False,
    )


def release_to_pose_and_disable(
    *,
    ctx: ReplayContext,
    q_prev: np.ndarray,
    q_release: np.ndarray,
    duration_s: float,
) -> np.ndarray:
    """Return to the fixed release pose before disabling the command interface."""
    print(f"State 4/4 release: returning to {RELEASE_POSE_NAME} before disabling command", flush=True)
    q_prev = run_waypoint_path(
        ctx=ctx,
        q_waypoints=[q_prev, q_release],
        q_prev=q_prev,
        duration_s=duration_s,
        label_prefix=f"return_to_{RELEASE_POSE_NAME}",
        time_plan={"type": "linear"},
        integrate_bias=False,
    )
    graceful_release(ctx=ctx, disable_repeats=SHUTDOWN_COMMAND_REPEATS)
    return q_prev


def graceful_release(
    ctx: ReplayContext,
    disable_repeats: int,
) -> None:
    """Disable the active command interface."""
    print("disabling active command interface", flush=True)
    for _ in range(max(1, int(disable_repeats))):
        ctx.backend.publish_arm_sdk_disable()
        time.sleep(ctx.dt)
    if ctx.adaptive_tau_bias is not None:
        ctx.adaptive_tau_bias.reset()


def send_arm_zero_gain_keep_waist_command(
    ctx: ReplayContext,
    q_ref: np.ndarray,
    repeats: int,
) -> None:
    """Zero shoulder/elbow/wrist gains while keeping waist joints actively held."""
    print("sending arm_sdk zero-gain command for arms; waist stays held", flush=True)
    q_ref = np.asarray(q_ref, dtype=float)
    zeros = np.zeros_like(q_ref)
    kp = zeros.copy()
    kd = zeros.copy()
    for idx in WAIST_JOINT_INDICES:
        kp[idx] = ctx.kp[idx]
        kd[idx] = ctx.kd[idx]
    for _ in range(max(1, int(repeats))):
        ctx.backend.publish_arm_sdk_command(
            q_ref,
            kp=kp,
            kd=kd,
            tau=zeros,
            dq_des=zeros,
            weight=1.0,
            joint_indices=ARM_SDK_JOINT_INDICES,
        )
        time.sleep(ctx.dt)
    if ctx.adaptive_tau_bias is not None:
        ctx.adaptive_tau_bias.reset()


def handle_keyboard_interrupt(
    *,
    ctx: ReplayContext,
    q_prev: np.ndarray,
    command_started: bool,
) -> None:
    if not command_started:
        print("Ctrl+C caught before commands started; exiting without command/release", flush=True)
        return

    fallback = ctx.last_q_cmd.copy() if ctx.last_q_cmd is not None else q_prev.copy()
    q_zero_gain = latest_q_or_fallback(ctx, fallback)
    print("Ctrl+C caught; sending arm zero-gain command with waist held and exiting", flush=True)
    send_arm_zero_gain_keep_waist_command(ctx=ctx, q_ref=q_zero_gain, repeats=SHUTDOWN_COMMAND_REPEATS)


def prepare_command(
    ctx: ReplayContext,
    q_des: np.ndarray,
    q_prev: np.ndarray,
    tau_scale: float = 1.0,
    integrate_bias: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    q_safe = clamp_joint_limits(q_des, ctx.q_min, ctx.q_max, margin=ctx.limit_margin_rad)
    tau_gravity = ctx.gravity_comp.torque(q_safe) if ctx.gravity_comp is not None else np.zeros_like(q_safe)

    latest = ctx.backend.latest_state()
    q_meas = latest.q if latest is not None else None
    tau_bias = (
        ctx.adaptive_tau_bias.update(q_safe, q_meas, ctx.dt, integrate=integrate_bias)
        if ctx.adaptive_tau_bias is not None
        else np.zeros_like(q_safe)
    )

    scale = float(tau_scale)
    tau_gravity = tau_gravity * scale
    tau_bias = tau_bias * scale
    tau = tau_gravity + tau_bias
    ctx.last_tau_gravity = tau_gravity
    ctx.last_tau_bias = tau_bias
    return q_safe, tau


def publish_and_report(
    ctx: ReplayContext,
    q_safe: np.ndarray,
    tau: np.ndarray,
    *,
    label: str,
    force_report: bool = False,
    kp: np.ndarray | None = None,
    kd: np.ndarray | None = None,
    dq_des: np.ndarray | None = None,
) -> None:
    publish_command(
        ctx,
        q_safe,
        ctx.kp if kp is None else kp,
        ctx.kd if kd is None else kd,
        tau,
        dq_des=dq_des,
    )
    report_step = ctx.report_step
    ctx.report_step += 1
    if not force_report and report_step % ctx.report_interval_steps != 0:
        return
    latest = ctx.backend.latest_state()
    if latest is None:
        return
    err = np.abs(q_safe - latest.q)
    idx = max_error_index(err, ctx.error_indices)
    tau_gravity = ctx.last_tau_gravity if ctx.last_tau_gravity is not None else np.zeros_like(q_safe)
    tau_bias = ctx.last_tau_bias if ctx.last_tau_bias is not None else np.zeros_like(q_safe)
    bias_indices = ctx.adaptive_tau_bias.indices if ctx.adaptive_tau_bias is not None else np.arange(len(tau_bias))
    max_bias_idx = int(bias_indices[int(np.argmax(np.abs(tau_bias[bias_indices])))] if len(bias_indices) else 0)
    print(
        f"{label} t={report_step * ctx.dt:.1f}s max_cmd_error={float(err[idx]):.4f}({ctx.index_to_name[idx]}) "
        f"max_abs_tau_bias={float(np.max(np.abs(tau_bias))):.3f}({ctx.index_to_name[max_bias_idx]})"
    )
    ctx.log.append(
        {
            "stamp": time.time(),
            "label": label,
            "max_cmd_error": float(err[idx]),
            "max_cmd_error_joint": ctx.index_to_name[idx],
            "max_abs_tau_ff": float(np.max(np.abs(tau_gravity))),
            "max_abs_tau_bias": float(np.max(np.abs(tau_bias))),
            "max_tau_bias_joint": ctx.index_to_name[max_bias_idx],
            "max_abs_tau_total": float(np.max(np.abs(tau))),
        }
    )


def publish_command(
    ctx: ReplayContext,
    q_des: np.ndarray,
    kp: np.ndarray,
    kd: np.ndarray,
    tau,
    dq_des=None,
) -> None:
    ctx.backend.publish_arm_sdk_command(
        q_des,
        kp=kp,
        kd=kd,
        tau=tau,
        dq_des=dq_des,
        weight=1.0,
        joint_indices=ARM_SDK_JOINT_INDICES,
    )
    ctx.last_q_cmd = np.asarray(q_des, dtype=float).copy()


def arm_sdk_to_current_pose(
    *,
    ctx: ReplayContext,
    q_current: np.ndarray,
    duration_s: float,
) -> np.ndarray:
    q_hold = latest_q_or_fallback(ctx, q_current)
    zero = np.zeros_like(q_hold)
    steps = max(1, int(round(float(duration_s) / ctx.dt)))
    zero_gain_steps = max(1, steps // 4)
    print(f"arming arm_sdk at current pose over {duration_s:.2f}s", flush=True)

    for _ in range(zero_gain_steps):
        ctx.backend.publish_arm_sdk_command(
            q_hold,
            kp=zero,
            kd=zero,
            tau=zero,
            dq_des=zero,
            weight=1.0,
            joint_indices=ARM_SDK_JOINT_INDICES,
        )
        ctx.last_q_cmd = q_hold.copy()
        time.sleep(ctx.dt)

    ramp_steps = max(1, steps - zero_gain_steps)
    for step in range(ramp_steps):
        alpha = smoothstep((step + 1) / ramp_steps)
        kp = ctx.kp * alpha
        kd = ctx.kd * alpha
        ctx.backend.publish_arm_sdk_command(
            q_hold,
            kp=kp,
            kd=kd,
            tau=zero,
            dq_des=zero,
            weight=1.0,
            joint_indices=ARM_SDK_JOINT_INDICES,
        )
        ctx.last_q_cmd = q_hold.copy()
        time.sleep(ctx.dt)
    return q_hold


def wait_menu_choice(prompt: str, choices: dict[str, str]) -> str:
    menu = " ".join(f"[{key}] {text}" for key, text in choices.items())
    print(f"{prompt}\n{menu}", flush=True)
    if not sys.stdin.isatty():
        value = input("stdin is not a TTY; enter choice: ").strip().lower()
        return value[:1]

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        while True:
            readable, _, _ = select.select([sys.stdin], [], [], 0.1)
            if not readable:
                continue
            char = sys.stdin.read(1)
            if char in choices:
                print(choices[char], flush=True)
                return char
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


def hold_menu_choice(
    *,
    ctx: ReplayContext,
    q_hold: np.ndarray,
    prompt: str,
    choices: dict[str, str],
) -> tuple[str, np.ndarray]:
    menu = " ".join(f"[{key}] {text}" for key, text in choices.items())
    print(f"{prompt}\n{menu}", flush=True)
    q_prev = q_hold.copy()
    if not sys.stdin.isatty():
        input("stdin is not a TTY; press Enter to continue.")
        return next(iter(choices)), q_prev

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        step = 0
        while True:
            q_prev, tau = prepare_command(ctx, q_hold, q_prev)
            publish_command(ctx, q_prev, ctx.kp, ctx.kd, tau)
            step += 1
            readable, _, _ = select.select([sys.stdin], [], [], ctx.dt)
            if not readable:
                continue
            char = sys.stdin.read(1)
            if char in choices:
                print(choices[char], flush=True)
                return char, q_prev
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path, help="Run profile YAML, e.g. configs/run_sim.yaml.")
    args = parser.parse_args()
    run_config = load_run_config(args.config)

    joint_config = load_yaml(PROJECT_ROOT / "configs" / "g1_29dof_joints.yaml")
    trajectory_config = load_yaml(PROJECT_ROOT / "configs" / "trajectory.yaml")
    poses_config = load_yaml(PROJECT_ROOT / "configs" / "poses.yaml")
    trajectory_name = str(run_config.get("trajectory", "standing_double_v2"))
    if trajectory_name not in trajectory_config:
        raise SystemExit(f"Trajectory {trajectory_name!r} is missing from configs/trajectory.yaml.")
    trajectory = trajectory_config[trajectory_name]
    gain_profile_name = str(run_config.get("gain_profile") or trajectory.get("gain_profile", "double_hand"))
    waypoints = trajectory["waypoints"]
    if len(waypoints) < 2:
        raise SystemExit("Trajectory needs at least two waypoints.")

    joints = joint_config["joints"]
    name_to_index = {joint["name"]: joint["index"] for joint in joints}
    index_to_name = {joint["index"]: joint["name"] for joint in joints}
    q_min = np.array([joint["q_min"] for joint in joints], dtype=float)
    q_max = np.array([joint["q_max"] for joint in joints], dtype=float)
    kp, kd = load_gain_profile_vectors(joints, gain_profile_name)
    impact_gain_profile = str(run_config.get("impact_gain_profile", "impact"))
    impact_kp, impact_kd = merge_impact_gains(
        joints=joints,
        base_kp=kp,
        base_kd=kd,
        impact_profile_name=impact_gain_profile,
    )
    error_indices = np.asarray(ARM_SDK_JOINT_INDICES, dtype=int)
    gravity_cfg = run_config.get("gravity_comp") or {}
    gravity_comp = (
        ModelSelectedJointGravityFeedforward(
            joints=joints,
            scale=float(gravity_cfg.get("scale", 1.0)),
            max_tau_nm=float(gravity_cfg.get("max_tau_nm", 8.0)),
        )
        if bool(gravity_cfg.get("enabled", False))
        else None
    )
    adaptive_tau_bias_cfg = run_config.get("adaptive_tau_bias") or {}
    adaptive_tau_bias = (
        WaistAdaptiveTauBias(joints=joints, q_min=q_min, q_max=q_max, config=adaptive_tau_bias_cfg)
        if bool(adaptive_tau_bias_cfg.get("enabled", True))
        else None
    )
    if adaptive_tau_bias is not None:
        adaptive_tau_bias.reset()

    interface = str(run_config.get("interface", "eth3"))
    domain_id = int(run_config.get("domain_id", 1))
    backend = G1Sdk2Backend(domain_id=domain_id, interface=interface)
    backend.initialize(
        enable_commands=False,
        enable_arm_sdk=True,
        release_motion_mode=False,
    )
    try:
        sample = backend.wait_for_state(timeout_s=float(run_config.get("timeout", 5.0)))
    except TimeoutError as exc:
        raise SystemExit(f"{exc}\nStart sim/run_sim_controller.py first.") from exc

    q_current = sample.q.copy()
    if RELEASE_POSE_NAME not in poses_config:
        raise SystemExit(f"Release pose {RELEASE_POSE_NAME!r} is missing from configs/poses.yaml.")
    q_release_target = apply_joint_block(
        q_current,
        poses_config[RELEASE_POSE_NAME].get("joints_rad", {}),
        name_to_index,
    )
    q_base = trajectory_base_q(q_current, trajectory, poses_config, name_to_index)
    sparse_q_waypoints = [apply_joint_block(q_base, wp["joints_rad"], name_to_index) for wp in waypoints]
    limit_margin_rad = float(run_config.get("limit_margin_rad", 0.0))
    q_release_target = clamp_joint_limits(q_release_target, q_min, q_max, margin=limit_margin_rad)
    sparse_q_waypoints = [clamp_joint_limits(q, q_min, q_max, margin=limit_margin_rad) for q in sparse_q_waypoints]
    replay_length = float(run_config.get("replay_length", 0.9))
    full_forward_q_waypoints = resample_waypoints(sparse_q_waypoints, RESAMPLED_TRAJECTORY_WAYPOINTS)
    forward_q_waypoints = trim_waypoint_path(full_forward_q_waypoints, replay_length)
    q_waypoints = forward_q_waypoints
    waist_yaw_index = name_to_index["waist_yaw"]
    standby_yaw_cfg = run_config.get("standby_waist_yaw") or {}
    side_waist_yaw = math.radians(float(standby_yaw_cfg.get("right_deg", -90.0)))
    side_waist_yaw = float(np.clip(side_waist_yaw, q_min[waist_yaw_index] + limit_margin_rad, q_max[waist_yaw_index] - limit_margin_rad))
    side_q_waypoints = with_joint_value(forward_q_waypoints, waist_yaw_index, side_waist_yaw)
    side_ready = False

    initial_error = np.abs(q_waypoints[0] - q_current)
    initial_error_index = max_error_index(initial_error, error_indices)
    initial_error_value = float(initial_error[initial_error_index])

    rate_hz = float(run_config.get("rate_hz", 100.0))
    dt = 1.0 / rate_hz
    total_duration = float(run_config.get("duration_s", trajectory.get("duration_s", 4.0)))
    return_duration_s = float(run_config.get("return_duration_s", total_duration * 2.0))
    impact_phase_s = float(run_config.get("impact_phase_s", DEFAULT_IMPACT_PHASE_S))
    transition_phase_s = float(run_config.get("transition_phase_s", 0.2))
    standby_yaw_switch_s = float(standby_yaw_cfg.get("switch_duration_s", INITIAL_RAMP_S))
    arm_sdk_arming_s = float(run_config.get("arm_sdk_arming_s", 1.0))
    velocity_ff_gain = float(run_config.get("velocity_ff_gain", 0.5))
    time_plan = run_config.get("time_plan") or trajectory.get("time_plan") or {"type": "acceleration"}
    return_time_plan = run_config.get("return_time_plan") or trajectory.get("return_time_plan") or {"type": "trapezoid", "accel_fraction": 0.2}
    playback_steps = max(1, int(round(total_duration / dt)))
    q_prev = q_current.copy()
    log = []
    ctx = ReplayContext(
        backend=backend,
        q_min=q_min,
        q_max=q_max,
        kp=kp,
        kd=kd,
        mode_machine=sample.mode_machine,
        limit_margin_rad=limit_margin_rad,
        rate_hz=rate_hz,
        velocity_ff_gain=velocity_ff_gain,
        index_to_name=index_to_name,
        error_indices=error_indices,
        log=log,
        gravity_comp=gravity_comp,
        adaptive_tau_bias=adaptive_tau_bias,
        impact_kp=impact_kp,
        impact_kd=impact_kd,
    )

    print(f"Replaying {trajectory_name} on DDS interface {interface}. arm_sdk=True.")
    print(
        f"waypoints={len(q_waypoints)} source_waypoints={len(sparse_q_waypoints)} duration_s={total_duration:.2f} "
        f"actual_duration_s={playback_steps * dt:.2f} arm_sdk_arming_s={arm_sdk_arming_s:.2f} "
        f"initial_ramp_s={INITIAL_RAMP_S:.2f} "
        f"replay_length={replay_length:.2f} "
        f"time_plan={time_plan.get('type', 'acceleration')} "
        f"return_time_plan={return_time_plan.get('type', 'trapezoid')} "
        f"velocity_ff_gain={velocity_ff_gain:.2f} "
        f"gain_profile={gain_profile_name} "
        f"impact_phase_s={impact_phase_s:.2f} transition_phase_s={transition_phase_s:.2f} "
        f"impact_gain_profile={impact_gain_profile} "
        f"side_ready_waist_yaw={side_waist_yaw:.3f}rad switch_s={standby_yaw_switch_s:.2f} "
        f"gravity_comp={bool(gravity_cfg.get('enabled', False))} gravity_comp_scale={float(gravity_cfg.get('scale', 1.0)):.2f} "
        f"gravity_comp_max_tau_nm={float(gravity_cfg.get('max_tau_nm', 8.0)):.1f} "
        f"gravity_comp_initial_max_tau_nm="
        f"{(float(np.max(np.abs(gravity_comp.torque(q_waypoints[0])))) if gravity_comp is not None else 0.0):.2f} "
        f"adaptive_tau_bias={adaptive_tau_bias is not None} "
        f"adaptive_tau_bias_max_tau_nm={float(adaptive_tau_bias_cfg.get('max_tau_nm', 8.0)):.2f} "
        f"initial_error={initial_error_value:.3f}({index_to_name[initial_error_index]}) "
        f"loop=True cycles={int(run_config.get('cycles', 0))} "
        f"return_duration_s={return_duration_s:.2f}"
    )
    interrupted = False
    release_requested = False
    hold_exit_requested = False
    command_started = False
    try:
        state = ReplayState.START
        while state is not ReplayState.DONE:
            if state is ReplayState.START:
                choice = wait_menu_choice(
                    "State 0/4 start: choose next transition.",
                    {"p": "Prepare hammer standby.", "q": "Quit before sending commands."},
                )
                if choice == "q":
                    print("quit requested before sending commands; no command/release was sent", flush=True)
                    state = transition_state(state, ReplayState.DONE)
                    continue

                command_started = True
                q_prev = arm_sdk_to_current_pose(
                    ctx=ctx,
                    q_current=q_current,
                    duration_s=arm_sdk_arming_s,
                )
                q_prev = publish_initial_ramp(
                    ctx=ctx,
                    q_start=q_prev,
                    q_target=q_waypoints[0],
                    min_duration_s=INITIAL_RAMP_S,
                )
                state = transition_state(state, ReplayState.STANDBY)
                continue

            if state is ReplayState.STANDBY:
                choice, q_prev = hold_menu_choice(
                    ctx=ctx,
                    q_hold=q_prev,
                    prompt="State 1/4 hammer standby: holding the first waypoint.",
                    choices={
                        "h": "Starting hammer trajectory.",
                        "r": "Toggle side-ready waist yaw.",
                        "x": f"Return to {RELEASE_POSE_NAME} and release.",
                        "q": "Quit now while leaving the current hold command active.",
                    },
                )
                if choice == "q":
                    hold_exit_requested = True
                    print("standby hold-exit requested: exiting without release/disable", flush=True)
                    state = transition_state(state, ReplayState.HOLD_EXIT)
                    continue
                if choice == "x":
                    state = transition_state(state, ReplayState.RELEASE)
                    continue
                if choice == "r":
                    side_ready = not side_ready
                    q_waypoints = side_q_waypoints if side_ready else forward_q_waypoints
                    target_name = "side-ready right waist yaw" if side_ready else "forward-ready waist yaw"
                    print(f"switching standby to {target_name}", flush=True)
                    q_prev = publish_initial_ramp(
                        ctx=ctx,
                        q_start=q_prev,
                        q_target=q_waypoints[0],
                        min_duration_s=standby_yaw_switch_s,
                    )
                    continue
                state = transition_state(state, ReplayState.HAMMERING)
                continue

            if state is ReplayState.HAMMERING:
                q_prev = run_hammer_loop(
                    ctx=ctx,
                    q_prev=q_prev,
                    q_waypoints=q_waypoints,
                    duration_s=total_duration,
                    return_duration_s=return_duration_s,
                    time_plan=time_plan,
                    return_time_plan=return_time_plan,
                    impact_phase_s=impact_phase_s,
                    transition_phase_s=transition_phase_s,
                    cycles=int(run_config.get("cycles", 0)),
                )
                state = transition_state(state, ReplayState.STOPPED)
                continue

            if state is ReplayState.STOPPED:
                choice, q_prev = hold_menu_choice(
                    ctx=ctx,
                    q_hold=q_prev,
                    prompt="State 3/4 hammer stopped: holding current/final command.",
                    choices={
                        "b": "Reverse back to hammer standby.",
                        "x": f"Return to {RELEASE_POSE_NAME} and release.",
                    },
                )
                if choice == "b":
                    q_prev = reverse_to_standby(
                        ctx=ctx,
                        q_prev=q_prev,
                        q_waypoints=q_waypoints,
                        return_duration_s=return_duration_s,
                    )
                    state = transition_state(state, ReplayState.STANDBY)
                    continue
                if choice == "x":
                    state = transition_state(state, ReplayState.RELEASE)
                    continue

            if state is ReplayState.RELEASE:
                release_requested = True
                state = transition_state(state, ReplayState.DONE)
                continue

            if state is ReplayState.HOLD_EXIT:
                state = transition_state(state, ReplayState.DONE)
                continue

            raise RuntimeError(f"Unhandled replay state: {state}")
    except KeyboardInterrupt:
        interrupted = True
        handle_keyboard_interrupt(ctx=ctx, q_prev=q_prev, command_started=command_started)
    finally:
        if hold_exit_requested:
            print("exiting from hammer standby without release; last hold command was left active", flush=True)
        elif command_started and release_requested and not interrupted:
            try:
                q_prev = release_to_pose_and_disable(
                    ctx=ctx,
                    q_prev=q_prev,
                    q_release=q_release_target,
                    duration_s=return_duration_s,
                )
            except KeyboardInterrupt:
                interrupted = True
                handle_keyboard_interrupt(ctx=ctx, q_prev=q_prev, command_started=command_started)

    log_dir = PROJECT_ROOT / "logs"
    log_dir.mkdir(exist_ok=True)
    path = log_dir / f"trajectory_{trajectory_name}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(log, f, indent=2)
        f.write("\n")
    print(f"saved {path}")
    if hold_exit_requested:
        print("exited from hammer standby without release")
    elif interrupted:
        print("interrupted; arm zero-gain command sent with waist held")
    elif release_requested:
        print(f"returned to {RELEASE_POSE_NAME} and released")
    else:
        print("exited without release")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
