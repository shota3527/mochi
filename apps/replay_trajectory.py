#!/usr/bin/env python3
"""Replay a configured joint trajectory through DDS for sim inspection."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import mujoco
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backends.sdk2_python_backend import G1Sdk2Backend
from core.safety_filter import clamp_joint_limits, rate_limit
from sim.run_sim_controller import POSES_CONFIG, build_initial_qpos, pose_grip_roll_phase, prepare_scene_path


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


def is_arm_joint(name: str) -> bool:
    return (
        name.startswith(("left_shoulder_", "right_shoulder_"))
        or name in ("left_elbow", "right_elbow")
        or name.startswith(("left_wrist_", "right_wrist_"))
    )


def gravity_comp_mask(joints: list[dict], groups_text: str) -> np.ndarray:
    groups = {item.strip() for item in groups_text.split(",") if item.strip()}
    if not groups:
        return np.zeros(len(joints), dtype=bool)
    if "all" in groups:
        return np.ones(len(joints), dtype=bool)

    mask = np.zeros(len(joints), dtype=bool)
    for joint in joints:
        name = joint["name"]
        index = joint["index"]
        if "lower" in groups and (
            name.startswith(("left_hip_", "right_hip_"))
            or name in ("left_knee", "right_knee")
            or name.startswith(("left_ankle_", "right_ankle_"))
        ):
            mask[index] = True
        if "waist" in groups and name.startswith("waist_"):
            mask[index] = True
        if "shoulder_elbow" in groups and (
            name.startswith(("left_shoulder_", "right_shoulder_")) or name in ("left_elbow", "right_elbow")
        ):
            mask[index] = True
        if "wrist" in groups and name.startswith(("left_wrist_", "right_wrist_")):
            mask[index] = True
    unknown = groups - {"all", "lower", "waist", "shoulder_elbow", "wrist"}
    if unknown:
        raise SystemExit(f"Unknown gravity-comp group(s): {', '.join(sorted(unknown))}")
    return mask


def gain_vectors(joints: list[dict], kp_scale: float, kd_scale: float) -> tuple[np.ndarray, np.ndarray]:
    kp = np.full(len(joints), 40.0)
    kd = np.full(len(joints), 1.0)
    for joint in joints:
        name = joint["name"]
        index = joint["index"]

        # Official G1 low-level examples use this body profile:
        # hips 60/1, knees 100/2, ankles 40/1, waist yaw 60/1,
        # waist roll/pitch 40/1, arms 40/1.
        if name.startswith(("left_hip_", "right_hip_")):
            kp[index] = 60.0
            kd[index] = 1.0
        if name in ("left_knee", "right_knee"):
            kp[index] = 100.0
            kd[index] = 2.0
        if name in ("waist_yaw",):
            kp[index] = 60.0
            kd[index] = 1.0
        if name.startswith(("left_ankle_", "right_ankle_")) or name in ("waist_roll", "waist_pitch"):
            kp[index] = 40.0
            kd[index] = 1.0

        if is_arm_joint(name):
            # Official G1 arm SDK examples use one arm-wide profile: 60/1.5.
            kp[index] = 60.0
            kd[index] = 1.5
    return kp * float(kp_scale), kd * float(kd_scale)


class GravityCompensator:
    """MuJoCo quasi-static torque feedforward for sim-only trajectory checks."""

    def __init__(self, pose_name: str, scale: float, sign: float, joints: list[dict], mask: np.ndarray):
        poses = load_yaml(POSES_CONFIG)
        if pose_name not in poses:
            raise SystemExit(f"Gravity-comp pose {pose_name!r} is missing from configs/poses.yaml.")
        pose = poses[pose_name]
        qpos = build_initial_qpos(pose_name, pose)
        scene_path = prepare_scene_path(qpos, grip_roll_phase_deg=pose_grip_roll_phase(pose))
        self.model = mujoco.MjModel.from_xml_path(str(scene_path))
        self.data = mujoco.MjData(self.model)
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        mujoco.mj_forward(self.model, self.data)
        self.nominal_qpos = self.data.qpos.copy()
        self.scale = float(scale)
        self.sign = float(sign)
        self.mask = np.asarray(mask, dtype=bool)
        self.qpos_addrs = []
        self.dof_addrs = []
        for joint in joints:
            joint_id = self.model.joint(f"{joint['name']}_joint").id
            self.qpos_addrs.append(int(self.model.jnt_qposadr[joint_id]))
            self.dof_addrs.append(int(self.model.jnt_dofadr[joint_id]))
        self.qpos_addrs = np.asarray(self.qpos_addrs, dtype=int)
        self.dof_addrs = np.asarray(self.dof_addrs, dtype=int)
        self.ctrl_min = self.model.actuator_ctrlrange[: len(joints), 0].copy()
        self.ctrl_max = self.model.actuator_ctrlrange[: len(joints), 1].copy()

    def torque(self, q_motor: np.ndarray) -> np.ndarray:
        self.data.qpos[:] = self.nominal_qpos
        self.data.qpos[self.qpos_addrs] = np.asarray(q_motor, dtype=float)
        self.data.qvel[:] = 0.0
        self.data.qacc[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        tau = self.data.qfrc_bias[self.dof_addrs] * self.scale * self.sign
        tau = np.where(self.mask, tau, 0.0)
        return np.clip(tau, self.ctrl_min, self.ctrl_max)


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
    gravity_comp: GravityCompensator | None,
) -> np.ndarray:
    for step in range(segment_steps):
        alpha = smoothstep(step / segment_steps)
        q_des = q0 + alpha * (q1 - q0)
        q_safe = clamp_joint_limits(q_des, q_min, q_max, margin=limit_margin_rad)
        q_safe = rate_limit(q_safe, q_prev, max_step=max_step_rad)
        tau_ff = gravity_comp.torque(q_safe) if gravity_comp is not None else None
        backend.publish_position_command(q_safe, mode_machine=mode_machine, kp=kp, kd=kd, tau=tau_ff)
        if step % max(1, int(rate_hz)) == 0:
            latest = backend.latest_state()
            if latest is not None:
                err = np.abs(q_safe - latest.q)
                idx = int(np.argmax(err))
                print(f"{label} t={step * dt:.1f}s max_cmd_error={float(err[idx]):.4f}({index_to_name[idx]})")
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
    return q_prev


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", default="kneel_dual_hold_swing_v0")
    parser.add_argument("--interface", default="eth3")
    parser.add_argument("--domain-id", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument("--rate-hz", type=float, default=100.0)
    parser.add_argument("--duration-s", type=float, default=None, help="Override trajectory duration from config.")
    parser.add_argument("--max-step-rad", type=float, default=0.012)
    parser.add_argument("--limit-margin-rad", type=float, default=0.0)
    parser.add_argument("--settle-s", type=float, default=1.0)
    parser.add_argument("--kp-scale", type=float, default=1.0)
    parser.add_argument("--kd-scale", type=float, default=1.0)
    parser.add_argument("--gravity-comp", action="store_true", help="Add sim-only MuJoCo gravity/bias torque feedforward.")
    parser.add_argument("--gravity-comp-scale", type=float, default=1.0)
    parser.add_argument("--gravity-comp-sign", type=float, default=-1.0)
    parser.add_argument("--gravity-comp-groups", default="waist")
    parser.add_argument("--gravity-comp-pose", default="kneel_hammer_ready_v0")
    parser.add_argument("--loop", action="store_true", help="Replay forward and backward repeatedly until Ctrl+C.")
    parser.add_argument("--cycles", type=int, default=0, help="Loop cycles to run; 0 means infinite when --loop is set.")
    parser.add_argument("--enable-command", action="store_true")
    args = parser.parse_args()

    if not args.enable_command:
        raise SystemExit("Refusing to publish trajectory commands. Re-run with --enable-command for sim-only testing.")

    joint_config = load_yaml(PROJECT_ROOT / "configs" / "g1_29dof_joints.yaml")
    trajectory_config = load_yaml(PROJECT_ROOT / "configs" / "trajectory.yaml")
    if args.trajectory not in trajectory_config:
        raise SystemExit(f"Trajectory {args.trajectory!r} is missing from configs/trajectory.yaml.")
    trajectory = trajectory_config[args.trajectory]
    waypoints = trajectory["waypoints"]
    if len(waypoints) < 2:
        raise SystemExit("Trajectory needs at least two waypoints.")

    joints = joint_config["joints"]
    name_to_index = {joint["name"]: joint["index"] for joint in joints}
    index_to_name = {joint["index"]: joint["name"] for joint in joints}
    q_min = np.array([joint["q_min"] for joint in joints], dtype=float)
    q_max = np.array([joint["q_max"] for joint in joints], dtype=float)
    kp, kd = gain_vectors(joints, kp_scale=args.kp_scale, kd_scale=args.kd_scale)
    comp_mask = gravity_comp_mask(joints, args.gravity_comp_groups)
    gravity_comp = (
        GravityCompensator(args.gravity_comp_pose, args.gravity_comp_scale, args.gravity_comp_sign, joints, comp_mask)
        if args.gravity_comp
        else None
    )

    backend = G1Sdk2Backend(domain_id=args.domain_id, interface=args.interface)
    backend.initialize(enable_commands=True)
    try:
        sample = backend.wait_for_state(timeout_s=args.timeout)
    except TimeoutError as exc:
        raise SystemExit(f"{exc}\nStart sim/run_sim_controller.py first.") from exc

    q_current = sample.q.copy()
    q_waypoints = [apply_joint_block(q_current, wp["joints_rad"], name_to_index) for wp in waypoints]
    q_waypoints = [clamp_joint_limits(q, q_min, q_max, margin=args.limit_margin_rad) for q in q_waypoints]

    rate_hz = float(args.rate_hz)
    dt = 1.0 / rate_hz
    total_duration = float(args.duration_s if args.duration_s is not None else trajectory.get("duration_s", 4.0))
    segment_duration = total_duration / (len(q_waypoints) - 1)
    segment_steps = max(1, int(math.ceil(segment_duration / dt)))
    q_prev = q_current.copy()
    log = []

    print(f"Replaying {args.trajectory} on DDS interface {args.interface}. Sim-only.")
    print(
        f"waypoints={len(q_waypoints)} duration_s={total_duration:.2f} settle_s={args.settle_s:.2f} "
        f"max_step_rad={args.max_step_rad} gain_profile=lower_official_upper_arm_sdk "
        f"kp_scale={args.kp_scale:.2f} kd_scale={args.kd_scale:.2f} "
        f"gravity_comp={args.gravity_comp} gravity_comp_scale={args.gravity_comp_scale:.2f} "
        f"gravity_comp_sign={args.gravity_comp_sign:.1f} gravity_comp_groups={args.gravity_comp_groups} "
        f"loop={args.loop} cycles={args.cycles}"
    )

    settle_steps = max(0, int(math.ceil(float(args.settle_s) / dt)))
    for step in range(settle_steps):
        q_safe = rate_limit(q_waypoints[0], q_prev, max_step=args.max_step_rad)
        tau_ff = gravity_comp.torque(q_safe) if gravity_comp is not None else None
        backend.publish_position_command(q_safe, mode_machine=sample.mode_machine, kp=kp, kd=kd, tau=tau_ff)
        q_prev = q_safe
        if step % max(1, int(rate_hz)) == 0:
            latest = backend.latest_state()
            if latest is not None:
                err = np.abs(q_safe - latest.q)
                idx = int(np.argmax(err))
                print(f"settle t={step * dt:.1f}s max_cmd_error={float(err[idx]):.4f}({index_to_name[idx]})")
        time.sleep(dt)

    forward_pairs = list(zip(q_waypoints[:-1], q_waypoints[1:]))
    if args.loop:
        reverse_pairs = list(zip(q_waypoints[:0:-1], q_waypoints[-2::-1]))
        cycles_done = 0
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
                    )
        except KeyboardInterrupt:
            print("stopped loop")
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
            )

    log_dir = PROJECT_ROOT / "logs"
    log_dir.mkdir(exist_ok=True)
    path = log_dir / f"trajectory_{args.trajectory}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(log, f, indent=2)
        f.write("\n")
    print(f"saved {path}")
    if not args.loop:
        print("holding final command; press Ctrl+C to stop")
        try:
            while True:
                tau_ff = gravity_comp.torque(q_waypoints[-1]) if gravity_comp is not None else None
                backend.publish_position_command(
                    q_waypoints[-1], mode_machine=sample.mode_machine, kp=kp, kd=kd, tau=tau_ff
                )
                time.sleep(dt)
        except KeyboardInterrupt:
            print("stopped final hold")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
