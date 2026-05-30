#!/usr/bin/env python3
"""Tiny one-joint upper-body motion test gated by state reading and safety."""

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
from core.safety_filter import filter_position_command, joint_margin, near_joint_limit


UPPER_BODY_SAFE_START = {
    "left_shoulder_roll": 16,
    "right_shoulder_roll": 23,
}


def load_yaml(name: str) -> dict:
    with (PROJECT_ROOT / "configs" / name).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def smoothstep(x: float) -> float:
    x = min(1.0, max(0.0, x))
    return x * x * (3.0 - 2.0 * x)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain-id", type=int, default=1)
    parser.add_argument("--interface", default="lo")
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument("--joint", default="left_shoulder_roll", choices=sorted(UPPER_BODY_SAFE_START))
    parser.add_argument("--delta", type=float, default=0.03)
    parser.add_argument("--duration", type=float, default=2.0)
    parser.add_argument("--rate-hz", type=float, default=100.0)
    parser.add_argument("--enable-command", action="store_true")
    args = parser.parse_args()

    if not args.enable_command:
        raise SystemExit("Refusing to publish motion commands. Re-run with --enable-command after dump_state.py works.")
    if not 0.0 < abs(args.delta) <= 0.05:
        raise SystemExit("--delta must be nonzero and no larger than 0.05 rad.")

    joint_config = load_yaml("g1_29dof_joints.yaml")
    safety_config = load_yaml("safety.yaml")
    joints = joint_config["joints"]
    q_min = np.array([joint["q_min"] for joint in joints], dtype=float)
    q_max = np.array([joint["q_max"] for joint in joints], dtype=float)
    joint_index = UPPER_BODY_SAFE_START[args.joint]

    backend = G1Sdk2Backend(domain_id=args.domain_id, interface=args.interface)
    backend.initialize(enable_commands=True)
    try:
        sample = backend.wait_for_state(timeout_s=args.timeout)
    except TimeoutError as exc:
        raise SystemExit(
            f"{exc}\nRun apps/dump_state.py successfully before enabling this motion test."
        ) from exc
    q0 = sample.q.copy()

    margins = joint_margin(q0, q_min, q_max)
    if margins[joint_index] < safety_config["danger_margin_rad"]:
        raise SystemExit(
            f"{args.joint} starts with only {margins[joint_index]:.4f} rad margin; dangerous."
        )
    if near_joint_limit(q0, q_min, q_max, margin=safety_config["near_limit_margin_rad"])[joint_index]:
        raise SystemExit(f"{args.joint} is near its joint limit; refusing motion.")

    target = q0.copy()
    target[joint_index] += args.delta
    dt = 1.0 / args.rate_hz
    max_step = safety_config["max_command_step_rad"]
    limit_margin = safety_config["command_limit_margin_rad"]

    log = []
    q_prev = q0.copy()
    total_steps = max(2, int(math.ceil(args.duration / dt)))
    print(f"Moving {args.joint} index={joint_index} by {args.delta:.4f} rad, then returning.")

    phases = [(q0, target), (target, q0)]
    for phase_name, (start, end) in zip(("out", "return"), phases):
        for step in range(total_steps + 1):
            alpha = smoothstep(step / total_steps)
            q_des = start + alpha * (end - start)
            q_safe = filter_position_command(
                q_des,
                q_prev,
                q_min,
                q_max,
                limit_margin=limit_margin,
                max_step=max_step,
            )
            backend.publish_single_joint_position(
                joint_index,
                q_safe[joint_index],
                mode_machine=sample.mode_machine,
                kp=safety_config["upper_body_kp"],
                kd=safety_config["upper_body_kd"],
            )
            log.append(
                {
                    "stamp": time.time(),
                    "phase": phase_name,
                    "joint_index": joint_index,
                    "q_cmd": float(q_safe[joint_index]),
                    "q_start": float(q0[joint_index]),
                }
            )
            q_prev = q_safe
            time.sleep(dt)

    log_dir = PROJECT_ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / f"small_motion_{time.strftime('%Y%m%d_%H%M%S')}.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(log, f, indent=2)
        f.write("\n")
    print(f"saved {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
