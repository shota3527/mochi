#!/usr/bin/env python3
"""Plan a height-targeted two-hand hammer swing and validate it in MuJoCo."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from planner.sweep_dual_hold_endpoints import EndpointSolution, EndpointSweeper


TRAJECTORY_CONFIG = PROJECT_ROOT / "configs" / "trajectory.yaml"


def rounded_list(values, ndigits: int = 6) -> list[float]:
    return [round(float(value), ndigits) for value in values]


def waypoint_from_solution(
    solution: EndpointSolution,
    *,
    alpha: float,
    target_head_z_m: float,
    head_down_dot: float,
) -> dict:
    return {
        "alpha": round(float(alpha), 6),
        "target_head_z_m": round(float(target_head_z_m), 6),
        "actual_head_z_m": round(float(solution.head_z_m), 6),
        "head_down_dot": round(float(head_down_dot), 6),
        "head_axis_y": round(float(solution.head_axis_y), 6),
        "hand_center_y_mm": round(float(solution.hand_center_y_mm), 3),
        "max_hand_grip_y_abs_mm": round(float(solution.max_hand_grip_y_abs_mm), 3),
        "loop_grip_mm": round(float(solution.loop_grip_mm), 6),
        "loop_axis_error": round(float(solution.loop_axis_error), 8),
        "min_joint_margin_rad": round(float(solution.min_joint_margin_rad), 6),
        "torso_contacts": int(solution.torso_contacts),
        "joints_rad": {name: float(value) for name, value in solution.joints_rad.items()},
    }


def replace_or_append_block(path: Path, key: str, block_text: str) -> None:
    text = path.read_text(encoding="utf-8")
    marker = f"\n{key}:"
    start = text.find(marker)
    if start < 0:
        path.write_text(text.rstrip() + "\n\n" + block_text, encoding="utf-8")
        return

    start += 1
    next_start = text.find("\n", start + len(key) + 1)
    end = len(text)
    search_pos = next_start + 1
    while search_pos < len(text):
        newline = text.find("\n", search_pos)
        if newline < 0:
            break
        line_start = newline + 1
        if line_start < len(text) and text[line_start] not in (" ", "\n", "#"):
            end = line_start
            break
        search_pos = line_start
    path.write_text(text[:start] + block_text.rstrip() + "\n\n" + text[end:].lstrip("\n"), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory-name", default="kneel_dual_hold_swing_v2")
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--duration-s", type=float, default=4.0)
    parser.add_argument("--start-head-z", type=float, default=0.86)
    parser.add_argument("--end-head-z", type=float, default=0.60)
    parser.add_argument("--start-head-down-dot", type=float, default=0.50)
    parser.add_argument("--end-head-down-dot", type=float, default=0.85)
    parser.add_argument("--head-axis-lateral-target", type=float, default=None)
    parser.add_argument("--write-config", action="store_true")
    EndpointSweeper.add_arguments(parser) if hasattr(EndpointSweeper, "add_arguments") else None

    # Keep these in sync with sweep_dual_hold_endpoints defaults while allowing
    # this planner to run without shelling out to the sweep script.
    parser.add_argument("--base-pose", default="knee_double_v0_start")
    parser.add_argument("--left-distance", type=float, default=0.20)
    parser.add_argument("--clamp-center-offset", type=float, default=0.0)
    parser.add_argument("--hand-grip-y-target", type=float, default=0.0)
    parser.add_argument("--right-tool-body", default="right_hammer_tool")
    parser.add_argument("--left-tool-body", default="left_hammer_clamp")
    parser.add_argument("--hammer-grip-body", default="right_hammer_grip")
    parser.add_argument("--head-geom", default="right_hammer_head")
    parser.add_argument("--left-elbow-body", default="left_elbow_link")
    parser.add_argument("--right-elbow-body", default="right_elbow_link")
    parser.add_argument("--left-wrist-body", default="left_wrist_yaw_link")
    parser.add_argument("--right-wrist-body", default="right_wrist_yaw_link")
    parser.add_argument("--loop-grip-weight", type=float, default=300.0)
    parser.add_argument("--loop-axis-weight", type=float, default=100.0)
    parser.add_argument("--head-z-weight", type=float, default=100.0)
    parser.add_argument("--head-axis-weight", type=float, default=2.5)
    parser.add_argument("--hand-grip-y-weight", type=float, default=0.25)
    parser.add_argument("--grip-center-x-target", type=float, default=None)
    parser.add_argument("--grip-center-x-weight", type=float, default=0.0)
    parser.add_argument("--wrist-center-x-target", type=float, default=None)
    parser.add_argument("--wrist-center-x-weight", type=float, default=0.0)
    parser.add_argument("--minimize-grip-center-x-weight", type=float, default=0.0)
    parser.add_argument("--minimize-wrist-center-x-weight", type=float, default=0.0)
    parser.add_argument("--elbow-clearance-y", type=float, default=0.18)
    parser.add_argument("--elbow-clearance-weight", type=float, default=20.0)
    parser.add_argument("--regularization-weight", type=float, default=0.025)
    parser.add_argument("--warm-blend", type=float, default=0.7)
    parser.add_argument("--max-nfev", type=int, default=1200)
    parser.add_argument("--max-loop-grip-mm", type=float, default=1.0)
    parser.add_argument("--max-loop-axis-error", type=float, default=0.002)
    parser.add_argument("--max-head-z-error", type=float, default=0.015)
    parser.add_argument("--max-torso-contacts", type=int, default=0)
    parser.add_argument("--min-joint-margin-rad", type=float, default=0.10)
    parser.add_argument("--max-neighbor-joint-step-rad", type=float, default=0.16)
    args = parser.parse_args()

    if args.samples < 2:
        raise SystemExit("--samples must be at least 2.")

    sweeper = EndpointSweeper(args)

    # Solve low endpoint first to select the installation/IK branch, then solve
    # high endpoint from that branch.
    end_solution = sweeper.solve_pose(
        target_head_z_m=args.end_head_z,
        head_down_dot=args.end_head_down_dot,
        warm_joints=sweeper.base_joints,
    )
    start_solution = sweeper.solve_pose(
        target_head_z_m=args.start_head_z,
        head_down_dot=args.start_head_down_dot,
        warm_joints=end_solution.joints_rad,
    )

    waypoints = []
    previous_solution = start_solution
    previous_q = start_solution.q
    max_neighbor_step = 0.0
    max_loop_grip_mm = 0.0
    max_loop_axis_error = 0.0
    max_head_z_error = 0.0
    max_head_axis_y_abs = 0.0
    max_torso_contacts = 0
    max_hand_center_y_mm = 0.0
    max_hand_grip_y_abs_mm = 0.0
    min_margin = float("inf")

    for i, alpha in enumerate(np.linspace(0.0, 1.0, args.samples)):
        target_head_z = (1.0 - alpha) * args.start_head_z + alpha * args.end_head_z
        head_down_dot = (1.0 - alpha) * args.start_head_down_dot + alpha * args.end_head_down_dot
        if i == 0:
            solution = start_solution
        elif i == args.samples - 1:
            solution = sweeper.solve_pose(
                target_head_z_m=target_head_z,
                head_down_dot=head_down_dot,
                warm_joints=previous_solution.joints_rad,
            )
        else:
            solution = sweeper.solve_pose(
                target_head_z_m=target_head_z,
                head_down_dot=head_down_dot,
                warm_joints=previous_solution.joints_rad,
            )

        q_step = float(np.max(np.abs(solution.q - previous_q))) if i > 0 else 0.0
        max_neighbor_step = max(max_neighbor_step, q_step)
        previous_solution = solution
        previous_q = solution.q

        max_loop_grip_mm = max(max_loop_grip_mm, solution.loop_grip_mm)
        max_loop_axis_error = max(max_loop_axis_error, solution.loop_axis_error)
        max_head_z_error = max(max_head_z_error, abs(solution.head_z_error_m))
        max_head_axis_y_abs = max(max_head_axis_y_abs, abs(solution.head_axis_y))
        max_torso_contacts = max(max_torso_contacts, solution.torso_contacts)
        max_hand_center_y_mm = max(max_hand_center_y_mm, abs(solution.hand_center_y_mm))
        max_hand_grip_y_abs_mm = max(max_hand_grip_y_abs_mm, solution.max_hand_grip_y_abs_mm)
        min_margin = min(min_margin, solution.min_joint_margin_rad)
        waypoints.append(
            waypoint_from_solution(
                solution,
                alpha=alpha,
                target_head_z_m=target_head_z,
                head_down_dot=head_down_dot,
            )
        )
        print(
            f"wp={i:02d} alpha={alpha:.3f} "
            f"head_z={solution.head_z_m:.3f}/{target_head_z:.3f} "
            f"loop_mm={solution.loop_grip_mm:.3f} axis={solution.loop_axis_error:.5f} "
            f"axis_y={solution.head_axis_y:.4f} "
            f"hand_center_y_mm={solution.hand_center_y_mm:.1f} "
            f"max_grip_y_mm={solution.max_hand_grip_y_abs_mm:.1f} "
            f"margin={solution.min_joint_margin_rad:.3f} "
            f"step={q_step:.3f} contacts={solution.torso_contacts}",
            flush=True,
        )

    passed = (
        max_loop_grip_mm <= args.max_loop_grip_mm
        and max_loop_axis_error <= args.max_loop_axis_error
        and max_head_z_error <= args.max_head_z_error
        and max_torso_contacts <= args.max_torso_contacts
        and min_margin >= args.min_joint_margin_rad
        and max_neighbor_step <= args.max_neighbor_joint_step_rad
    )
    amplitude = waypoints[0]["actual_head_z_m"] - waypoints[-1]["actual_head_z_m"]
    trajectory = {
        args.trajectory_name: {
            "memo": "Height-targeted kneeling two-hand hammer swing; free stick root with hard closed-loop hand geometry.",
            "base_pose": args.base_pose,
            "start_head_z_m": args.start_head_z,
            "end_head_z_m": args.end_head_z,
            "actual_head_z_amplitude_m": round(float(amplitude), 6),
            "duration_s": args.duration_s,
            "left_grip_distance_m": args.left_distance,
            "clamp_center_offset_m": args.clamp_center_offset,
            "grip_roll_phase_deg": -120.0,
            "validation": {
                "passed": passed,
                "closed_loop_only": True,
                "full_pose_locked": False,
                "max_loop_grip_mm": round(float(max_loop_grip_mm), 6),
                "max_loop_axis_error": round(float(max_loop_axis_error), 8),
                "max_head_z_error_m": round(float(max_head_z_error), 6),
                "max_head_axis_y_abs": round(float(max_head_axis_y_abs), 6),
                "max_hand_center_y_mm": round(float(max_hand_center_y_mm), 3),
                "max_hand_grip_y_abs_mm": round(float(max_hand_grip_y_abs_mm), 3),
                "min_arm_joint_margin_rad": round(float(min_margin), 6),
                "max_neighbor_joint_step_rad": round(float(max_neighbor_step), 6),
                "max_torso_contacts": int(max_torso_contacts),
            },
            "waypoints": waypoints,
        }
    }

    print(
        "\nsummary: "
        f"passed={passed} amplitude_m={amplitude:.3f} "
        f"max_loop_grip_mm={max_loop_grip_mm:.6f} "
        f"max_loop_axis_error={max_loop_axis_error:.8f} "
        f"max_head_z_error_m={max_head_z_error:.6f} "
        f"max_head_axis_y_abs={max_head_axis_y_abs:.6f} "
        f"max_hand_center_y_mm={max_hand_center_y_mm:.3f} "
        f"max_hand_grip_y_abs_mm={max_hand_grip_y_abs_mm:.3f} "
        f"min_margin={min_margin:.6f} "
        f"max_neighbor_step={max_neighbor_step:.6f} "
        f"max_torso_contacts={max_torso_contacts}",
        flush=True,
    )
    print("\nyaml:")
    block_text = yaml.safe_dump(trajectory, sort_keys=False, width=1000)
    print(block_text)

    if args.write_config:
        replace_or_append_block(TRAJECTORY_CONFIG, args.trajectory_name, block_text)
        print(f"wrote {args.trajectory_name} to {TRAJECTORY_CONFIG}")

    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
