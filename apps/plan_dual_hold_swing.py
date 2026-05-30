#!/usr/bin/env python3
"""Plan a strict two-hand stick swing inspection trajectory."""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import mujoco
import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.stick_ik import (
    LEFT_ARM_JOINTS,
    RIGHT_ARM_JOINTS,
    grips_from_stick,
    joint_qpos_addrs,
    make_centered_stick_target,
    solve_dual_hold_ik,
)
from sim.run_sim_controller import POSES_CONFIG, build_initial_qpos, prepare_scene_path


ARM_JOINTS = LEFT_ARM_JOINTS + RIGHT_ARM_JOINTS


def torso_contact_pairs(model, data) -> Counter:
    pairs = []
    for i in range(data.ncon):
        contact = data.contact[i]
        if contact.dist > 0.002:
            continue
        body1 = model.geom_bodyid[contact.geom1]
        body2 = model.geom_bodyid[contact.geom2]
        name1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body1) or str(body1)
        name2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body2) or str(body2)
        if name1 == "torso_link" or name2 == "torso_link":
            pairs.append(tuple(sorted((name1, name2))))
    return Counter(pairs)


def joint_margin(model, joint_names, q_values) -> float:
    margins = []
    for name, value in zip(joint_names, q_values):
        joint_id = model.joint(name).id
        lower, upper = model.jnt_range[joint_id]
        margins.append(min(float(value) - lower, upper - float(value)))
    return float(min(margins))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-pose", default="hammer_mounted_elbow_65")
    parser.add_argument("--root", nargs=3, type=float, default=(0.12, 0.0, 0.77), metavar=("X", "Y", "Z"))
    parser.add_argument("--start-pitch-deg", type=float, default=60.0)
    parser.add_argument("--end-pitch-deg", type=float, default=35.0)
    parser.add_argument("--samples", type=int, default=6)
    parser.add_argument("--right-distance", type=float, default=0.0)
    parser.add_argument("--left-distance", type=float, default=0.20)
    parser.add_argument("--clamp-center-offset", type=float, default=0.0)
    parser.add_argument("--max-grip-error-m", type=float, default=0.002)
    parser.add_argument("--max-axis-error", type=float, default=0.005)
    parser.add_argument("--min-joint-margin-rad", type=float, default=0.10)
    args = parser.parse_args()

    poses = yaml.safe_load(POSES_CONFIG.read_text(encoding="utf-8"))
    base_pose = poses[args.base_pose]
    initial_qpos = build_initial_qpos(args.base_pose, base_pose)
    scene_path = prepare_scene_path(initial_qpos)
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)

    joint_addrs = joint_qpos_addrs(model, ARM_JOINTS)
    previous_solution = None
    previous_q = None
    waypoints = []
    max_grip_error = 0.0
    max_axis_error = 0.0
    min_margin = float("inf")
    max_neighbor_step = 0.0
    max_torso_contacts = 0

    for pitch_deg in np.linspace(args.start_pitch_deg, args.end_pitch_deg, args.samples):
        stick = make_centered_stick_target(
            root_m=args.root,
            pitch_rad=np.deg2rad(float(pitch_deg)),
            length_m=0.60,
        )
        targets = grips_from_stick(
            stick,
            right_grip_distance_m=args.right_distance,
            left_grip_distance_m=args.left_distance,
            clamp_center_offset_m=args.clamp_center_offset,
        )
        result = solve_dual_hold_ik(
            model,
            data,
            targets,
            elbow_clearance_y_m=0.18,
            elbow_clearance_weight=20.0,
            regularization_weight=0.01,
            right_grip_weight=100.0,
            left_grip_weight=100.0,
            right_axis_weight=30.0,
            left_axis_weight=30.0,
            initial_joint_values=previous_solution,
            use_extra_starts=previous_solution is None,
            max_nfev=900,
        )
        previous_solution = result.joint_values

        q = data.qpos[joint_addrs].copy()
        margin = joint_margin(model, ARM_JOINTS, q)
        torso_contacts = torso_contact_pairs(model, data)
        max_grip_error = max(max_grip_error, result.right_grip_error_m, result.left_grip_error_m)
        max_axis_error = max(max_axis_error, result.right_axis_error, result.left_axis_error)
        min_margin = min(min_margin, margin)
        max_torso_contacts = max(max_torso_contacts, sum(torso_contacts.values()))
        if previous_q is not None:
            max_neighbor_step = max(max_neighbor_step, float(np.max(np.abs(q - previous_q))))
        previous_q = q

        waypoint = {
            "pitch_deg": float(pitch_deg),
            "right_grip_error_m": result.right_grip_error_m,
            "left_grip_error_m": result.left_grip_error_m,
            "right_axis_error": result.right_axis_error,
            "left_axis_error": result.left_axis_error,
            "min_joint_margin_rad": margin,
            "torso_contacts": dict(torso_contacts),
            "joints_rad": result.joint_values,
        }
        waypoints.append(waypoint)
        print(
            f"pitch={pitch_deg:5.1f} "
            f"grip_mm=({result.right_grip_error_m * 1000.0:.3f},{result.left_grip_error_m * 1000.0:.3f}) "
            f"axis=({result.right_axis_error:.4f},{result.left_axis_error:.4f}) "
            f"margin={margin:.3f} torso_contacts={sum(torso_contacts.values())}",
            flush=True,
        )

    passed = (
        max_grip_error <= args.max_grip_error_m
        and max_axis_error <= args.max_axis_error
        and min_margin >= args.min_joint_margin_rad
        and max_torso_contacts == 0
    )

    print(
        "\nsummary: "
        f"passed={passed} "
        f"max_grip_error_m={max_grip_error:.6f} "
        f"max_axis_error={max_axis_error:.6f} "
        f"min_joint_margin_rad={min_margin:.6f} "
        f"max_neighbor_joint_step_rad={max_neighbor_step:.6f} "
        f"max_torso_contacts={max_torso_contacts}",
        flush=True,
    )

    print("\nyaml:")
    print(
        yaml.safe_dump(
            {
                "dual_hold_swing_v0": {
                    "memo": "Strict two-hand deadlocked stick swing candidate; sim inspection only.",
                    "base_pose": args.base_pose,
                    "root_world_m": [float(v) for v in args.root],
                    "pitch_start_deg": args.start_pitch_deg,
                    "pitch_end_deg": args.end_pitch_deg,
                    "right_grip_distance_m": args.right_distance,
                    "left_grip_distance_m": args.left_distance,
                    "clamp_center_offset_m": args.clamp_center_offset,
                    "validation": {
                        "passed": passed,
                        "max_grip_error_m": max_grip_error,
                        "max_axis_error": max_axis_error,
                        "min_joint_margin_rad": min_margin,
                        "max_neighbor_joint_step_rad": max_neighbor_step,
                        "max_torso_contacts": max_torso_contacts,
                    },
                    "waypoints": waypoints,
                }
            },
            sort_keys=False,
        )
    )
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
