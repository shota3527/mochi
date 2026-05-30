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
from pathlib import Path

import mujoco
import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backends.sdk2_python_backend import G1Sdk2Backend
from core.safety_filter import clamp_joint_limits, rate_limit
from sim.run_sim_controller import UNITREE_G1_29DOF_XML

GAIN_PROFILES_CONFIG = PROJECT_ROOT / "configs" / "gain_profiles.yaml"
SHUTDOWN_HOLD_S = 2.0
SHUTDOWN_HOLD_KP = 20.0
SHUTDOWN_HOLD_KD = 1.0
SHUTDOWN_DISABLE_REPEATS = 20
ARM_SDK_JOINT_INDICES = tuple(range(12, 29))


class StopTrajectory(Exception):
    """Raised when the operator asks a looped trajectory to stop in place."""

    def __init__(self, q_current: np.ndarray):
        super().__init__("trajectory stopped by operator")
        self.q_current = q_current


class SpaceStopper:
    def __init__(self, prompt: str):
        self.prompt = prompt
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
        return sys.stdin.read(1) == " "


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def smoothstep(x: float) -> float:
    x = min(1.0, max(0.0, x))
    return x * x * (3.0 - 2.0 * x)


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


def publish_segment(
    *,
    backend: G1Sdk2Backend,
    q0: np.ndarray,
    q1: np.ndarray,
    q_prev: np.ndarray,
    segment_steps: int,
    dt: float,
    q_min: np.ndarray,
    q_max: np.ndarray,
    kp: np.ndarray,
    kd: np.ndarray,
    mode_machine,
    max_step_rad: float,
    limit_margin_rad: float,
    rate_hz: float,
    index_to_name: dict[int, str],
    log: list[dict],
    label: str,
    gravity_comp: ModelSelectedJointGravityFeedforward | None,
    arm_sdk: bool,
    stop_checker=None,
) -> np.ndarray:
    for step in range(segment_steps):
        alpha = smoothstep(step / segment_steps)
        q_des = q0 + alpha * (q1 - q0)
        q_safe = clamp_joint_limits(q_des, q_min, q_max, margin=limit_margin_rad)
        q_safe = rate_limit(q_safe, q_prev, max_step=max_step_rad)
        tau_ff = gravity_comp.torque(q_safe) if gravity_comp is not None else None
        publish_command(backend, q_safe, mode_machine, kp, kd, tau_ff, arm_sdk=arm_sdk)
        if step % max(1, int(rate_hz)) == 0 or step == segment_steps - 1:
            latest = backend.latest_state()
            if latest is not None:
                err = np.abs(q_safe - latest.q)
                idx = int(np.argmax(err))
                print(
                    f"{label} t={step * dt:.1f}s "
                    f"max_cmd_error={float(err[idx]):.4f}({index_to_name[idx]})"
                )
                log.append(
                    {
                        "stamp": time.time(),
                        "label": label,
                        "max_cmd_error": float(err[idx]),
                        "max_cmd_error_joint": index_to_name[idx],
                        "max_abs_tau_ff": float(np.max(np.abs(tau_ff))) if tau_ff is not None else 0.0,
                    }
                )
        q_prev = q_safe
        time.sleep(dt)
        if stop_checker is not None and stop_checker():
            print(f"{label}: SPACE received; stopping at current command.", flush=True)
            raise StopTrajectory(q_prev.copy())
    return q_prev


def publish_initial_ramp(
    *,
    backend: G1Sdk2Backend,
    q_start: np.ndarray,
    q_target: np.ndarray,
    q_min: np.ndarray,
    q_max: np.ndarray,
    kp: np.ndarray,
    kd: np.ndarray,
    mode_machine,
    max_step_rad: float,
    limit_margin_rad: float,
    rate_hz: float,
    min_duration_s: float,
    index_to_name: dict[int, str],
    gravity_comp: ModelSelectedJointGravityFeedforward | None,
    arm_sdk: bool,
) -> np.ndarray:
    dt = 1.0 / float(rate_hz)
    q_prev = q_start.copy()
    ramp_steps = max(1, int(math.ceil(float(min_duration_s) / dt)))

    def publish(q_des: np.ndarray, step: int, alpha: float) -> None:
        nonlocal q_prev
        q_safe = clamp_joint_limits(q_des, q_min, q_max, margin=limit_margin_rad)
        q_safe = rate_limit(q_safe, q_prev, max_step=max_step_rad)
        tau_ff = gravity_comp.torque(q_safe) if gravity_comp is not None else None
        if tau_ff is not None:
            tau_ff = tau_ff * alpha
        publish_command(backend, q_safe, mode_machine, kp, kd, tau_ff, arm_sdk=arm_sdk)
        q_prev = q_safe

        if step % max(1, int(rate_hz)) == 0:
            latest = backend.latest_state()
            if latest is not None:
                err = np.abs(q_safe - latest.q)
                idx = int(np.argmax(err))
                print(
                    f"initial_ramp t={step * dt:.1f}s "
                    f"max_cmd_error={float(err[idx]):.4f}({index_to_name[idx]})"
                )
        time.sleep(dt)

    for step in range(ramp_steps):
        alpha = smoothstep((step + 1) / ramp_steps)
        publish(q_start + alpha * (q_target - q_start), step, alpha)

    extra_step = ramp_steps
    while float(np.max(np.abs(q_target - q_prev))) > 1e-4:
        alpha = 1.0
        publish(q_target, extra_step, alpha)
        extra_step += 1

    return q_prev


def graceful_release(
    backend: G1Sdk2Backend,
    q_hold: np.ndarray,
    mode_machine,
    rate_hz: float,
    hold_s: float,
    hold_kp: float,
    hold_kd: float,
    disable_repeats: int,
    arm_sdk: bool,
) -> None:
    """Briefly hold the last pose, then disable low-level commands."""
    dt = 1.0 / float(rate_hz)
    kp = np.full_like(q_hold, float(hold_kp), dtype=float)
    kd = np.full_like(q_hold, float(hold_kd), dtype=float)
    tau = np.zeros_like(q_hold, dtype=float)
    hold_steps = max(0, int(math.ceil(float(hold_s) / dt)))
    for _ in range(hold_steps):
        publish_command(backend, q_hold, mode_machine, kp, kd, tau, arm_sdk=arm_sdk)
        time.sleep(dt)
    for _ in range(max(1, int(disable_repeats))):
        if arm_sdk:
            backend.publish_arm_sdk_disable()
        else:
            backend.publish_disable_command(mode_machine=mode_machine)
        time.sleep(dt)


def publish_command(
    backend: G1Sdk2Backend,
    q_des: np.ndarray,
    mode_machine,
    kp: np.ndarray,
    kd: np.ndarray,
    tau,
    *,
    arm_sdk: bool,
) -> None:
    if arm_sdk:
        backend.publish_arm_sdk_command(
            q_des,
            kp=kp,
            kd=kd,
            tau=tau,
            weight=1.0,
            joint_indices=ARM_SDK_JOINT_INDICES,
        )
    else:
        backend.publish_position_command(q_des, mode_machine=mode_machine, kp=kp, kd=kd, tau=tau)


def wait_for_space(prompt: str, confirm_message: str) -> None:
    print(prompt, flush=True)
    if not sys.stdin.isatty():
        input("stdin is not a TTY; press Enter to continue.")
        return

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        while True:
            readable, _, _ = select.select([sys.stdin], [], [], 0.1)
            if not readable:
                continue
            char = sys.stdin.read(1)
            if char == " ":
                print(confirm_message, flush=True)
                return
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


def hold_command_until_space(
    *,
    backend: G1Sdk2Backend,
    q_hold: np.ndarray,
    mode_machine,
    kp: np.ndarray,
    kd: np.ndarray,
    tau: np.ndarray,
    arm_sdk: bool,
    rate_hz: float,
    prompt: str,
    confirm_message: str,
) -> None:
    print(prompt, flush=True)
    dt = 1.0 / float(rate_hz)
    if not sys.stdin.isatty():
        input("stdin is not a TTY; press Enter to continue.")
        return

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        while True:
            publish_command(backend, q_hold, mode_machine, kp, kd, tau, arm_sdk=arm_sdk)
            readable, _, _ = select.select([sys.stdin], [], [], dt)
            if not readable:
                continue
            char = sys.stdin.read(1)
            if char == " ":
                print(confirm_message, flush=True)
                return
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", default="knee_double_v0")
    parser.add_argument(
        "--gain-profile",
        default=None,
        help="Gain profile from configs/gain_profiles.yaml. Defaults to trajectory gain_profile, then double_hand.",
    )
    parser.add_argument("--interface", default="eth3")
    parser.add_argument("--domain-id", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument("--rate-hz", type=float, default=100.0)
    parser.add_argument("--duration-s", type=float, default=None, help="Override trajectory duration from config.")
    parser.add_argument("--max-step-rad", type=float, default=0.012)
    parser.add_argument("--limit-margin-rad", type=float, default=0.0)
    parser.add_argument("--settle-s", type=float, default=3.0, help="Minimum initial ramp time before trajectory replay.")
    parser.add_argument("--gravity-comp", action="store_true", help="Add official-model gravity feedforward for waist, shoulders, and elbows.")
    parser.add_argument("--gravity-comp-scale", type=float, default=1.0)
    parser.add_argument("--gravity-comp-max-tau-nm", type=float, default=8.0)
    parser.add_argument("--loop", action="store_true", help="Replay forward and backward repeatedly until SPACE stops at the current command.")
    parser.add_argument("--cycles", type=int, default=0, help="Loop cycles to run; 0 means infinite when --loop is set.")
    parser.add_argument("--arm-sdk", action="store_true", help="Publish upper-body commands on rt/arm_sdk instead of full-body rt/lowcmd.")
    parser.add_argument(
        "--return-to-start-s",
        type=float,
        default=6.0,
        help="On normal completion, ramp back to the startup state before releasing commands.",
    )
    args = parser.parse_args()

    joint_config = load_yaml(PROJECT_ROOT / "configs" / "g1_29dof_joints.yaml")
    trajectory_config = load_yaml(PROJECT_ROOT / "configs" / "trajectory.yaml")
    poses_config = load_yaml(PROJECT_ROOT / "configs" / "poses.yaml")
    if args.trajectory not in trajectory_config:
        raise SystemExit(f"Trajectory {args.trajectory!r} is missing from configs/trajectory.yaml.")
    trajectory = trajectory_config[args.trajectory]
    gain_profile_name = args.gain_profile or trajectory.get("gain_profile", "double_hand")
    waypoints = trajectory["waypoints"]
    if len(waypoints) < 2:
        raise SystemExit("Trajectory needs at least two waypoints.")

    joints = joint_config["joints"]
    name_to_index = {joint["name"]: joint["index"] for joint in joints}
    index_to_name = {joint["index"]: joint["name"] for joint in joints}
    q_min = np.array([joint["q_min"] for joint in joints], dtype=float)
    q_max = np.array([joint["q_max"] for joint in joints], dtype=float)
    kp, kd = load_gain_profile_vectors(joints, gain_profile_name)
    gravity_comp = (
        ModelSelectedJointGravityFeedforward(
            joints=joints,
            scale=args.gravity_comp_scale,
            max_tau_nm=args.gravity_comp_max_tau_nm,
        )
        if args.gravity_comp
        else None
    )

    backend = G1Sdk2Backend(domain_id=args.domain_id, interface=args.interface)
    backend.initialize(
        enable_commands=not args.arm_sdk,
        enable_arm_sdk=args.arm_sdk,
        release_motion_mode=False,
    )
    try:
        sample = backend.wait_for_state(timeout_s=args.timeout)
    except TimeoutError as exc:
        raise SystemExit(f"{exc}\nStart sim/run_sim_controller.py first.") from exc

    q_current = sample.q.copy()
    q_release_target = q_current.copy()
    q_base = trajectory_base_q(q_current, trajectory, poses_config, name_to_index)
    q_waypoints = [apply_joint_block(q_base, wp["joints_rad"], name_to_index) for wp in waypoints]
    q_waypoints = [clamp_joint_limits(q, q_min, q_max, margin=args.limit_margin_rad) for q in q_waypoints]

    initial_error = np.abs(q_waypoints[0] - q_current)
    initial_error_index = int(np.argmax(initial_error))
    initial_error_value = float(initial_error[initial_error_index])

    rate_hz = float(args.rate_hz)
    dt = 1.0 / rate_hz
    total_duration = float(args.duration_s if args.duration_s is not None else trajectory.get("duration_s", 4.0))
    segment_duration = total_duration / (len(q_waypoints) - 1)
    segment_steps = max(1, int(math.ceil(segment_duration / dt)))
    q_prev = q_current.copy()
    log = []

    print(f"Replaying {args.trajectory} on DDS interface {args.interface}. arm_sdk={args.arm_sdk}.")
    print(
        f"waypoints={len(q_waypoints)} duration_s={total_duration:.2f} settle_s={args.settle_s:.2f} "
        f"max_step_rad={args.max_step_rad} gain_profile={gain_profile_name} "
        f"gravity_comp={args.gravity_comp} gravity_comp_scale={args.gravity_comp_scale:.2f} "
        f"gravity_comp_max_tau_nm={args.gravity_comp_max_tau_nm:.1f} "
        f"gravity_comp_initial_max_tau_nm="
        f"{(float(np.max(np.abs(gravity_comp.torque(q_waypoints[0])))) if gravity_comp is not None else 0.0):.2f} "
        f"initial_error={initial_error_value:.3f}({index_to_name[initial_error_index]}) "
        f"loop={args.loop} cycles={args.cycles}"
    )

    forward_pairs = list(zip(q_waypoints[:-1], q_waypoints[1:]))
    interrupted = False
    completed = False
    stopped_in_loop = False
    command_started = False
    try:
        wait_for_space(
            "Stage 1/4: Press SPACE to ramp to the first waypoint, or Ctrl+C to abort.",
            "Ramping to first waypoint.",
        )

        q_prev = publish_initial_ramp(
            backend=backend,
            q_start=q_current,
            q_target=q_waypoints[0],
            q_min=q_min,
            q_max=q_max,
            kp=kp,
            kd=kd,
            mode_machine=sample.mode_machine,
            max_step_rad=args.max_step_rad,
            limit_margin_rad=args.limit_margin_rad,
            rate_hz=rate_hz,
            min_duration_s=args.settle_s,
            index_to_name=index_to_name,
            gravity_comp=gravity_comp,
            arm_sdk=args.arm_sdk,
        )
        command_started = True

        tau_hold = gravity_comp.torque(q_prev) if gravity_comp is not None else np.zeros_like(q_prev)
        hold_command_until_space(
            backend=backend,
            q_hold=q_prev,
            mode_machine=sample.mode_machine,
            kp=kp,
            kd=kd,
            tau=tau_hold,
            arm_sdk=args.arm_sdk,
            rate_hz=rate_hz,
            prompt="Stage 2/4: At first waypoint and holding. Press SPACE to start trajectory.",
            confirm_message="Starting trajectory.",
        )

        if args.loop:
            reverse_pairs = list(zip(q_waypoints[:0:-1], q_waypoints[-2::-1]))
            cycles_done = 0
            with SpaceStopper("Stage 3/4: Loop running. Press SPACE to stop at the current command for stick removal.") as stopper:
                try:
                    while args.cycles <= 0 or cycles_done < args.cycles:
                        cycles_done += 1
                        print(f"cycle={cycles_done} forward")
                        for segment_index, (q0, q1) in enumerate(forward_pairs):
                            q_prev = publish_segment(
                                backend=backend,
                                q0=q0,
                                q1=q1,
                                q_prev=q_prev,
                                segment_steps=segment_steps,
                                dt=dt,
                                q_min=q_min,
                                q_max=q_max,
                                kp=kp,
                                kd=kd,
                                mode_machine=sample.mode_machine,
                                max_step_rad=args.max_step_rad,
                                limit_margin_rad=args.limit_margin_rad,
                                rate_hz=rate_hz,
                                index_to_name=index_to_name,
                                log=log,
                                label=f"cycle={cycles_done} forward segment={segment_index}",
                                gravity_comp=gravity_comp,
                                arm_sdk=args.arm_sdk,
                                stop_checker=stopper.pressed,
                            )
                        print(f"cycle={cycles_done} backward")
                        for segment_index, (q0, q1) in enumerate(reverse_pairs):
                            q_prev = publish_segment(
                                backend=backend,
                                q0=q0,
                                q1=q1,
                                q_prev=q_prev,
                                segment_steps=segment_steps,
                                dt=dt,
                                q_min=q_min,
                                q_max=q_max,
                                kp=kp,
                                kd=kd,
                                mode_machine=sample.mode_machine,
                                max_step_rad=args.max_step_rad,
                                limit_margin_rad=args.limit_margin_rad,
                                rate_hz=rate_hz,
                                index_to_name=index_to_name,
                                log=log,
                                label=f"cycle={cycles_done} backward segment={segment_index}",
                                gravity_comp=gravity_comp,
                                arm_sdk=args.arm_sdk,
                                stop_checker=stopper.pressed,
                            )
                    completed = True
                except StopTrajectory as exc:
                    q_prev = exc.q_current
                    stopped_in_loop = True
        else:
            for segment_index, (q0, q1) in enumerate(forward_pairs):
                q_prev = publish_segment(
                    backend=backend,
                    q0=q0,
                    q1=q1,
                    q_prev=q_prev,
                    segment_steps=segment_steps,
                    dt=dt,
                    q_min=q_min,
                    q_max=q_max,
                    kp=kp,
                    kd=kd,
                    mode_machine=sample.mode_machine,
                    max_step_rad=args.max_step_rad,
                    limit_margin_rad=args.limit_margin_rad,
                    rate_hz=rate_hz,
                    index_to_name=index_to_name,
                    log=log,
                    label=f"segment={segment_index}",
                    gravity_comp=gravity_comp,
                    arm_sdk=args.arm_sdk,
                )
            completed = True
    except KeyboardInterrupt:
        interrupted = True
        print("interrupted; holding briefly, disabling command, then releasing")
    finally:
        if (completed or stopped_in_loop) and args.return_to_start_s > 0.0:
            tau_hold = gravity_comp.torque(q_prev) if gravity_comp is not None else np.zeros_like(q_prev)
            if completed and not args.loop:
                hold_command_until_space(
                    backend=backend,
                    q_hold=q_prev,
                    mode_machine=sample.mode_machine,
                    kp=kp,
                    kd=kd,
                    tau=tau_hold,
                    arm_sdk=args.arm_sdk,
                    rate_hz=rate_hz,
                    prompt="Stage 3/4: Trajectory complete; holding final command for stick removal. Press SPACE to confirm current-position stop.",
                    confirm_message="Current-position stop confirmed.",
                )
            hold_command_until_space(
                backend=backend,
                q_hold=q_prev,
                mode_machine=sample.mode_machine,
                kp=kp,
                kd=kd,
                tau=tau_hold,
                arm_sdk=args.arm_sdk,
                rate_hz=rate_hz,
                prompt="Stage 4/4: Holding current command. Press SPACE to return to startup state and release.",
                confirm_message="Returning to startup state before release.",
            )
            print("returning to startup state before release", flush=True)
            return_steps = max(1, int(math.ceil(float(args.return_to_start_s) * rate_hz)))
            q_prev = publish_segment(
                backend=backend,
                q0=q_prev,
                q1=q_release_target,
                q_prev=q_prev,
                segment_steps=return_steps,
                dt=dt,
                q_min=q_min,
                q_max=q_max,
                kp=kp,
                kd=kd,
                mode_machine=sample.mode_machine,
                max_step_rad=args.max_step_rad,
                limit_margin_rad=args.limit_margin_rad,
                rate_hz=rate_hz,
                index_to_name=index_to_name,
                log=log,
                label="return_to_startup_state",
                gravity_comp=gravity_comp,
                arm_sdk=args.arm_sdk,
            )
        if command_started:
            graceful_release(
                backend=backend,
                q_hold=q_prev,
                mode_machine=sample.mode_machine,
                rate_hz=rate_hz,
                hold_s=SHUTDOWN_HOLD_S,
                hold_kp=SHUTDOWN_HOLD_KP,
                hold_kd=SHUTDOWN_HOLD_KD,
                disable_repeats=SHUTDOWN_DISABLE_REPEATS,
                arm_sdk=args.arm_sdk,
            )

    log_dir = PROJECT_ROOT / "logs"
    log_dir.mkdir(exist_ok=True)
    path = log_dir / f"trajectory_{args.trajectory}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(log, f, indent=2)
        f.write("\n")
    print(f"saved {path}")
    if interrupted:
        print("released after interrupt")
    else:
        print("returned to startup state and released after trajectory")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
