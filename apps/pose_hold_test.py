#!/usr/bin/env python3
"""Sim-only joint-position hold test configured by configs/poses.yaml."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backends.sdk2_python_backend import G1Sdk2Backend
from core.safety_filter import clamp_joint_limits, rate_limit


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def smoothstep(x: float) -> float:
    x = min(1.0, max(0.0, x))
    return x * x * (3.0 - 2.0 * x)


def joint_key_to_config_name(key: str) -> str:
    return key.removesuffix("_joint")


def build_pose_target(q_start: np.ndarray, pose: dict, name_to_index: dict[str, int]) -> np.ndarray:
    q_target = q_start.copy()
    for joint_key, q in pose.get("joints_rad", {}).items():
        config_name = joint_key_to_config_name(joint_key)
        if config_name not in name_to_index:
            raise SystemExit(f"Pose joint {joint_key!r} does not match joint config.")
        q_target[name_to_index[config_name]] = float(q)
    return q_target


def build_gain_vectors(test_config: dict, joints: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    gains = test_config["gains"]
    num_joints = len(joints)
    kp = np.full(num_joints, float(gains["default"]["kp"]))
    kd = np.full(num_joints, float(gains["default"]["kd"]))

    for joint in joints:
        name = joint["name"]
        index = joint["index"]
        if (
            name.startswith(("left_hip_", "right_hip_", "waist_"))
            or "knee" in name
            or "ankle" in name
        ):
            kp[index] = float(gains["lower_body"]["kp"])
            kd[index] = float(gains["lower_body"]["kd"])
        if name in ("left_knee", "right_knee"):
            kp[index] = float(gains["knees"]["kp"])
            kd[index] = float(gains["knees"]["kd"])

    return kp, kd


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain-id", type=int, default=1)
    parser.add_argument("--interface", default="eth3")
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument("--pose", help="Pose name from configs/poses.yaml. Defaults to pose_hold_test.pose.")
    parser.add_argument("--enable-command", action="store_true")
    args = parser.parse_args()

    if not args.enable_command:
        raise SystemExit("Refusing to publish pose commands. Re-run with --enable-command for sim-only testing.")

    joint_config = load_yaml(PROJECT_ROOT / "configs" / "g1_29dof_joints.yaml")
    poses = load_yaml(PROJECT_ROOT / "configs" / "poses.yaml")
    test_config = poses["pose_hold_test"]
    pose_name = args.pose or test_config["pose"]
    if pose_name not in poses:
        raise SystemExit(f"Configured pose {pose_name!r} is missing from configs/poses.yaml.")
    pose = poses[pose_name]

    joints = joint_config["joints"]
    name_to_index = {joint["name"]: joint["index"] for joint in joints}
    index_to_name = {joint["index"]: joint["name"] for joint in joints}
    q_min = np.array([joint["q_min"] for joint in joints], dtype=float)
    q_max = np.array([joint["q_max"] for joint in joints], dtype=float)

    backend = G1Sdk2Backend(domain_id=args.domain_id, interface=args.interface)
    backend.initialize(enable_commands=True)
    try:
        sample = backend.wait_for_state(timeout_s=args.timeout)
    except TimeoutError as exc:
        raise SystemExit(f"{exc}\nStart sim/run_sim_controller.py first.") from exc

    q_start = sample.q.copy()
    q_target = build_pose_target(q_start, pose, name_to_index)

    limit_margin = float(test_config["limit_margin_rad"])
    q_target = clamp_joint_limits(q_target, q_min, q_max, margin=limit_margin)

    kp, kd = build_gain_vectors(test_config, joints)

    rate_hz = float(test_config["rate_hz"])
    max_step = float(test_config["max_step_rad"])
    dt = 1.0 / rate_hz
    ramp_steps = max(1, int(math.ceil(float(test_config["ramp_time_s"]) / dt)))
    hold_steps = max(1, int(math.ceil(float(test_config["hold_time_s"]) / dt)))
    q_prev = q_start.copy()
    log = []

    controlled_joints = sorted(pose.get("joints_rad", {}))
    print(f"Holding pose {pose_name} from configs/poses.yaml on DDS interface {args.interface}. Sim-only.")
    print(f"Pose base_z_m={pose.get('base_z_m', 'unset')} is simulator initialization only; this app commands joints.")
    print(f"Configured pose joints: {', '.join(controlled_joints)}")
    for step in range(ramp_steps + hold_steps):
        if step < ramp_steps:
            alpha = smoothstep(step / ramp_steps)
            q_des = q_start + alpha * (q_target - q_start)
            phase = "ramp"
        else:
            q_des = q_target
            phase = "hold"

        q_safe = clamp_joint_limits(q_des, q_min, q_max, margin=limit_margin)
        q_safe = rate_limit(q_safe, q_prev, max_step=max_step)
        backend.publish_position_command(q_safe, mode_machine=sample.mode_machine, kp=kp, kd=kd)
        latest = backend.latest_state()
        if latest is not None and step % max(1, int(rate_hz)) == 0:
            cmd_abs_error = np.abs(q_safe - latest.q)
            target_abs_error = np.abs(q_target - latest.q)
            cmd_error_index = int(np.argmax(cmd_abs_error))
            target_error_index = int(np.argmax(target_abs_error))
            err = float(cmd_abs_error[cmd_error_index])
            target_err = float(target_abs_error[target_error_index])
            print(
                f"{phase} t={step * dt:.1f}s "
                f"max_cmd_error={err:.4f}({index_to_name[cmd_error_index]}) "
                f"max_target_error={target_err:.4f}({index_to_name[target_error_index]})"
            )
            log.append(
                {
                    "stamp": time.time(),
                    "phase": phase,
                    "max_cmd_error": err,
                    "max_cmd_error_joint": index_to_name[cmd_error_index],
                    "max_target_error": target_err,
                    "max_target_error_joint": index_to_name[target_error_index],
                }
            )
        q_prev = q_safe
        time.sleep(dt)

    log_dir = PROJECT_ROOT / "logs"
    log_dir.mkdir(exist_ok=True)
    path = log_dir / f"pose_hold_{pose_name}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(log, f, indent=2)
        f.write("\n")
    print(f"saved {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
