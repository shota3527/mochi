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


@dataclass
class ReplayContext:
    backend: G1Sdk2Backend
    q_min: np.ndarray
    q_max: np.ndarray
    kp: np.ndarray
    kd: np.ndarray
    mode_machine: int | None
    max_step_rad: float
    limit_margin_rad: float
    rate_hz: float
    index_to_name: dict[int, str]
    error_indices: np.ndarray
    log: list[dict]
    gravity_comp: ModelSelectedJointGravityFeedforward | None
    arm_sdk: bool

    @property
    def dt(self) -> float:
        return 1.0 / float(self.rate_hz)

    @property
    def report_interval_steps(self) -> int:
        return max(1, int(self.rate_hz))


def publish_segment(
    *,
    ctx: ReplayContext,
    q0: np.ndarray,
    q1: np.ndarray,
    q_prev: np.ndarray,
    segment_steps: int,
    label: str,
    stop_checker=None,
) -> np.ndarray:
    for step in range(segment_steps):
        alpha = smoothstep(step / segment_steps)
        q_des = q0 + alpha * (q1 - q0)
        q_safe, tau_ff = prepare_command(ctx, q_des, q_prev)
        publish_and_report(ctx, q_safe, tau_ff, label=label, step=step, force_report=step == segment_steps - 1)
        q_prev = q_safe
        time.sleep(ctx.dt)
        if stop_checker is not None and stop_checker():
            print(f"{label}: SPACE received; stopping at current command.", flush=True)
            raise StopTrajectory(q_prev.copy())
    return q_prev


def run_segments(
    ctx: ReplayContext,
    pairs: list[tuple[np.ndarray, np.ndarray]],
    q_prev: np.ndarray,
    segment_steps: int,
    label_prefix: str,
    stop_checker=None,
) -> np.ndarray:
    for segment_index, (q0, q1) in enumerate(pairs):
        label = f"{label_prefix} segment={segment_index}" if label_prefix else f"segment={segment_index}"
        q_prev = publish_segment(
            ctx=ctx,
            q0=q0,
            q1=q1,
            q_prev=q_prev,
            segment_steps=segment_steps,
            label=label,
            stop_checker=stop_checker,
        )
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
        q_safe, tau_ff = prepare_command(ctx, q_des, q_prev, tau_scale=alpha)
        publish_and_report(ctx, q_safe, tau_ff, label="initial_ramp", step=step)
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


def graceful_release(
    ctx: ReplayContext,
    disable_repeats: int,
) -> None:
    """Disable the active command interface."""
    for _ in range(max(1, int(disable_repeats))):
        if ctx.arm_sdk:
            ctx.backend.publish_arm_sdk_disable()
        else:
            ctx.backend.publish_disable_command(mode_machine=ctx.mode_machine)
        time.sleep(ctx.dt)


def prepare_command(
    ctx: ReplayContext,
    q_des: np.ndarray,
    q_prev: np.ndarray,
    tau_scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray | None]:
    q_safe = clamp_joint_limits(q_des, ctx.q_min, ctx.q_max, margin=ctx.limit_margin_rad)
    q_safe = rate_limit(q_safe, q_prev, max_step=ctx.max_step_rad)
    tau_ff = ctx.gravity_comp.torque(q_safe) if ctx.gravity_comp is not None else None
    if tau_ff is not None and tau_scale != 1.0:
        tau_ff = tau_ff * float(tau_scale)
    return q_safe, tau_ff


def publish_and_report(
    ctx: ReplayContext,
    q_safe: np.ndarray,
    tau_ff: np.ndarray | None,
    *,
    label: str,
    step: int,
    force_report: bool = False,
) -> None:
    publish_command(ctx, q_safe, ctx.kp, ctx.kd, tau_ff)
    if not force_report and step % ctx.report_interval_steps != 0:
        return
    latest = ctx.backend.latest_state()
    if latest is None:
        return
    err = np.abs(q_safe - latest.q)
    idx = max_error_index(err, ctx.error_indices)
    print(f"{label} t={step * ctx.dt:.1f}s max_cmd_error={float(err[idx]):.4f}({ctx.index_to_name[idx]})")
    ctx.log.append(
        {
            "stamp": time.time(),
            "label": label,
            "max_cmd_error": float(err[idx]),
            "max_cmd_error_joint": ctx.index_to_name[idx],
            "max_abs_tau_ff": float(np.max(np.abs(tau_ff))) if tau_ff is not None else 0.0,
        }
    )


def publish_command(
    ctx: ReplayContext,
    q_des: np.ndarray,
    kp: np.ndarray,
    kd: np.ndarray,
    tau,
) -> None:
    if ctx.arm_sdk:
        ctx.backend.publish_arm_sdk_command(
            q_des,
            kp=kp,
            kd=kd,
            tau=tau,
            weight=1.0,
            joint_indices=ARM_SDK_JOINT_INDICES,
        )
    else:
        ctx.backend.publish_position_command(q_des, mode_machine=ctx.mode_machine, kp=kp, kd=kd, tau=tau)


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
    ctx: ReplayContext,
    q_hold: np.ndarray,
    tau: np.ndarray,
    prompt: str,
    confirm_message: str,
) -> None:
    print(prompt, flush=True)
    if not sys.stdin.isatty():
        input("stdin is not a TTY; press Enter to continue.")
        return

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        while True:
            publish_command(ctx, q_hold, ctx.kp, ctx.kd, tau)
            readable, _, _ = select.select([sys.stdin], [], [], ctx.dt)
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
        "--no-release-motion-mode",
        action="store_true",
        help="For full-body rt/lowcmd only: skip releasing the active Unitree high-level motion mode.",
    )
    parser.add_argument(
        "--return-time-multiplier",
        type=float,
        default=2.0,
        help="Return-to-start duration multiplier relative to --duration-s. Default 2.0 means half speed.",
    )
    args = parser.parse_args()
    if args.return_time_multiplier <= 0.0:
        raise SystemExit("--return-time-multiplier must be > 0.")

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
    error_indices = np.asarray(
        ARM_SDK_JOINT_INDICES if args.arm_sdk else tuple(range(len(joints))),
        dtype=int,
    )
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
    release_motion_mode = (not args.arm_sdk) and (not args.no_release_motion_mode)
    backend.initialize(
        enable_commands=not args.arm_sdk,
        enable_arm_sdk=args.arm_sdk,
        release_motion_mode=release_motion_mode,
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
    initial_error_index = max_error_index(initial_error, error_indices)
    initial_error_value = float(initial_error[initial_error_index])

    rate_hz = float(args.rate_hz)
    dt = 1.0 / rate_hz
    total_duration = float(args.duration_s if args.duration_s is not None else trajectory.get("duration_s", 4.0))
    segment_duration = total_duration / (len(q_waypoints) - 1)
    segment_steps = max(1, int(math.ceil(segment_duration / dt)))
    q_prev = q_current.copy()
    log = []
    ctx = ReplayContext(
        backend=backend,
        q_min=q_min,
        q_max=q_max,
        kp=kp,
        kd=kd,
        mode_machine=sample.mode_machine,
        max_step_rad=args.max_step_rad,
        limit_margin_rad=args.limit_margin_rad,
        rate_hz=rate_hz,
        index_to_name=index_to_name,
        error_indices=error_indices,
        log=log,
        gravity_comp=gravity_comp,
        arm_sdk=args.arm_sdk,
    )

    print(f"Replaying {args.trajectory} on DDS interface {args.interface}. arm_sdk={args.arm_sdk}.")
    print(
        f"waypoints={len(q_waypoints)} duration_s={total_duration:.2f} settle_s={args.settle_s:.2f} "
        f"max_step_rad={args.max_step_rad} max_velocity_rad_s={args.max_step_rad * rate_hz:.3f} "
        f"gain_profile={gain_profile_name} release_motion_mode={release_motion_mode} "
        f"gravity_comp={args.gravity_comp} gravity_comp_scale={args.gravity_comp_scale:.2f} "
        f"gravity_comp_max_tau_nm={args.gravity_comp_max_tau_nm:.1f} "
        f"gravity_comp_initial_max_tau_nm="
        f"{(float(np.max(np.abs(gravity_comp.torque(q_waypoints[0])))) if gravity_comp is not None else 0.0):.2f} "
        f"initial_error={initial_error_value:.3f}({index_to_name[initial_error_index]}) "
        f"loop={args.loop} cycles={args.cycles} return_time_multiplier={args.return_time_multiplier:.2f}"
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
            ctx=ctx,
            q_start=q_current,
            q_target=q_waypoints[0],
            min_duration_s=args.settle_s,
        )
        command_started = True

        tau_hold = gravity_comp.torque(q_prev) if gravity_comp is not None else np.zeros_like(q_prev)
        hold_command_until_space(
            ctx=ctx,
            q_hold=q_prev,
            tau=tau_hold,
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
                        q_prev = run_segments(
                            ctx,
                            forward_pairs,
                            q_prev,
                            segment_steps,
                            label_prefix=f"cycle={cycles_done} forward",
                            stop_checker=stopper.pressed,
                        )
                        print(f"cycle={cycles_done} backward")
                        q_prev = run_segments(
                            ctx,
                            reverse_pairs,
                            q_prev,
                            segment_steps,
                            label_prefix=f"cycle={cycles_done} backward",
                            stop_checker=stopper.pressed,
                        )
                    completed = True
                except StopTrajectory as exc:
                    q_prev = exc.q_current
                    stopped_in_loop = True
        else:
            q_prev = run_segments(ctx, forward_pairs, q_prev, segment_steps, label_prefix="")
            completed = True
    except KeyboardInterrupt:
        interrupted = True
        print("interrupted; holding briefly, disabling command, then releasing")
    finally:
        if completed or stopped_in_loop:
            tau_hold = gravity_comp.torque(q_prev) if gravity_comp is not None else np.zeros_like(q_prev)
            if completed and not args.loop:
                hold_command_until_space(
                    ctx=ctx,
                    q_hold=q_prev,
                    tau=tau_hold,
                    prompt="Stage 3/4: Trajectory complete; holding final command for stick removal. Press SPACE to confirm current-position stop.",
                    confirm_message="Current-position stop confirmed.",
                )
            hold_command_until_space(
                ctx=ctx,
                q_hold=q_prev,
                tau=tau_hold,
                prompt="Stage 4/4: Holding current command. Press SPACE to return to startup state and release.",
                confirm_message="Returning to startup state before release.",
            )
            print("returning to startup state before release", flush=True)
            return_duration_s = total_duration * float(args.return_time_multiplier)
            return_steps = max(1, int(math.ceil(return_duration_s * rate_hz)))
            q_prev = publish_segment(
                ctx=ctx,
                q0=q_prev,
                q1=q_release_target,
                q_prev=q_prev,
                segment_steps=return_steps,
                label="return_to_startup_state",
            )
        if command_started:
            graceful_release(
                ctx=ctx,
                disable_repeats=SHUTDOWN_DISABLE_REPEATS,
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
