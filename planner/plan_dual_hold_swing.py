#!/usr/bin/env python3
"""Plan a free-root two-hand hammer-down inspection trajectory."""

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
    body_axis_and_grip,
    joint_qpos_addrs,
    normalize,
    solve_dual_hold_closed_loop_ik,
)
from sim.run_sim_controller import POSES_CONFIG, build_initial_qpos, pose_grip_roll_phase, prepare_scene_path


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


def head_axis_target_from_down_dot(down_dot: float, lateral: float) -> np.ndarray:
    down_dot = float(np.clip(down_dot, -1.0, 1.0))
    lateral = float(np.clip(lateral, -0.5, 0.5))
    if down_dot * down_dot + lateral * lateral >= 1.0:
        lateral = 0.0
    forward = float(np.sqrt(max(0.0, 1.0 - down_dot * down_dot - lateral * lateral)))
    return normalize([forward, lateral, -down_dot])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-pose", default="knee_double_v0_start")
    parser.add_argument("--start-head-down-dot", type=float, default=0.50)
    parser.add_argument("--end-head-down-dot", type=float, default=0.85)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--left-distance", type=float, default=0.20)
    parser.add_argument("--clamp-center-offset", type=float, default=0.0)
    parser.add_argument("--max-grip-error-m", type=float, default=0.002)
    parser.add_argument("--max-axis-error", type=float, default=0.010)
    parser.add_argument("--max-head-axis-error", type=float, default=0.35)
    parser.add_argument("--loop-grip-weight", type=float, default=250.0)
    parser.add_argument("--loop-axis-weight", type=float, default=80.0)
    parser.add_argument("--head-axis-weight", type=float, default=4.0)
    parser.add_argument("--min-joint-margin-rad", type=float, default=0.10)
    args = parser.parse_args()

    poses = yaml.safe_load(POSES_CONFIG.read_text(encoding="utf-8"))
    base_pose = poses[args.base_pose]
    initial_qpos = build_initial_qpos(args.base_pose, base_pose)
    scene_path = prepare_scene_path(
        initial_qpos,
        grip_roll_phase_deg=pose_grip_roll_phase(base_pose),
        left_weld_distance_m=args.left_distance,
    )
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    initial_head_lateral = float(data.xmat[model.body("right_hammer_grip").id].reshape(3, 3)[:, 0][1])

    joint_addrs = joint_qpos_addrs(model, ARM_JOINTS)
    previous_solution = None
    previous_q = None
    waypoints = []
    max_grip_error = 0.0
    max_axis_error = 0.0
    max_head_axis_error = 0.0
    min_margin = float("inf")
    max_neighbor_step = 0.0
    max_torso_contacts = 0

    head_down_dots = list(np.linspace(args.start_head_down_dot, args.end_head_down_dot, args.samples))
    for i, head_down_dot in enumerate(head_down_dots):
        head_axis_target = head_axis_target_from_down_dot(head_down_dot, initial_head_lateral)
        if i == 0:
            right_axis, right_grip = body_axis_and_grip(model, data, "right_hammer_tool", args.clamp_center_offset)
            left_axis, left_grip = body_axis_and_grip(model, data, "left_hammer_clamp", args.clamp_center_offset)
            head_axis = data.xmat[model.body("right_hammer_grip").id].reshape(3, 3)[:, 0].copy()

            class CurrentPoseResult:
                pass

            result = CurrentPoseResult()
            result.joint_values = {name: float(data.qpos[joint_addrs[j]]) for j, name in enumerate(ARM_JOINTS)}
            result.loop_grip_error_m = float(np.linalg.norm(left_grip - (right_grip + right_axis * args.left_distance)))
            result.loop_axis_error = float(np.linalg.norm(left_axis - right_axis))
            result.head_axis_error = float(np.linalg.norm(head_axis - head_axis_target))
            result.hand_center_y_m = float(0.5 * (right_grip[1] + left_grip[1]))
            result.right_grip_m = right_grip.copy()
            result.left_grip_m = left_grip.copy()
            result.right_axis_m = right_axis.copy()
            result.head_axis_m = head_axis
        else:
            result = solve_dual_hold_closed_loop_ik(
                model,
                data,
                left_grip_distance_m=args.left_distance,
                clamp_center_offset_m=args.clamp_center_offset,
                head_axis_target_m=head_axis_target,
                elbow_clearance_y_m=0.18,
                elbow_clearance_weight=20.0,
                regularization_weight=0.02,
                loop_grip_weight=args.loop_grip_weight,
                loop_axis_weight=args.loop_axis_weight,
                head_axis_weight=args.head_axis_weight,
                initial_joint_values=previous_solution,
                use_extra_starts=False,
                max_nfev=900,
            )
        result.hand_center_y_m = float(0.5 * (result.right_grip_m[1] + result.left_grip_m[1]))
        previous_solution = result.joint_values

        q = data.qpos[joint_addrs].copy()
        margin = joint_margin(model, ARM_JOINTS, q)
        torso_contacts = torso_contact_pairs(model, data)
        max_grip_error = max(max_grip_error, result.loop_grip_error_m)
        max_axis_error = max(max_axis_error, result.loop_axis_error)
        max_head_axis_error = max(max_head_axis_error, result.head_axis_error)
        min_margin = min(min_margin, margin)
        max_torso_contacts = max(max_torso_contacts, sum(torso_contacts.values()))
        if previous_q is not None:
            max_neighbor_step = max(max_neighbor_step, float(np.max(np.abs(q - previous_q))))
        previous_q = q

        waypoint = {
            "head_down_dot": float(head_down_dot),
            "head_axis_target_world": [float(v) for v in head_axis_target],
            "right_grip_world_m": [float(v) for v in result.right_grip_m],
            "left_grip_world_m": [float(v) for v in result.left_grip_m],
            "right_axis_world": [float(v) for v in result.right_axis_m],
            "head_axis_world": [float(v) for v in result.head_axis_m],
            "loop_grip_error_m": result.loop_grip_error_m,
            "loop_axis_error": result.loop_axis_error,
            "head_axis_error": result.head_axis_error,
            "hand_center_y_m": result.hand_center_y_m,
            "min_joint_margin_rad": margin,
            "torso_contacts": dict(torso_contacts),
            "joints_rad": result.joint_values,
        }
        waypoints.append(waypoint)
        print(
            f"head_down_dot={head_down_dot:.3f} "
            f"loop_grip_mm={result.loop_grip_error_m * 1000.0:.3f} "
            f"loop_axis={result.loop_axis_error:.4f} "
            f"head_axis={result.head_axis_error:.4f} "
            f"hand_center_y_mm={result.hand_center_y_m * 1000.0:.1f} "
            f"margin={margin:.3f} torso_contacts={sum(torso_contacts.values())}",
            flush=True,
        )

    passed = (
        max_grip_error <= args.max_grip_error_m
        and max_axis_error <= args.max_axis_error
        and max_head_axis_error <= args.max_head_axis_error
        and min_margin >= args.min_joint_margin_rad
        and max_torso_contacts == 0
    )

    print(
        "\nsummary: "
        f"passed={passed} "
        f"max_grip_error_m={max_grip_error:.6f} "
        f"max_axis_error={max_axis_error:.6f} "
        f"max_head_axis_error={max_head_axis_error:.6f} "
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
                    "memo": "Free-root two-hand hammer-down swing candidate; sim inspection only.",
                    "base_pose": args.base_pose,
                    "start_head_down_dot": args.start_head_down_dot,
                    "end_head_down_dot": args.end_head_down_dot,
                    "left_grip_distance_m": args.left_distance,
                    "clamp_center_offset_m": args.clamp_center_offset,
                    "validation": {
                        "passed": passed,
                        "max_grip_error_m": max_grip_error,
                        "max_axis_error": max_axis_error,
                        "max_head_axis_error": max_head_axis_error,
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
