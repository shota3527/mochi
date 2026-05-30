#!/usr/bin/env python3
"""Sweep two-hand hammer endpoint poses with fixed closed-loop grip geometry."""

from __future__ import annotations

import argparse
import itertools
import sys
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
import yaml
from scipy.optimize import least_squares

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from planner.plan_dual_hold_swing import joint_margin, torso_contact_pairs
from core.stick_ik import (
    LEFT_ARM_JOINTS,
    RIGHT_ARM_JOINTS,
    body_axis_and_grip,
    joint_qpos_addrs,
    joint_ranges,
    normalize,
)
from sim.run_sim_controller import POSES_CONFIG, build_initial_qpos, pose_grip_roll_phase, prepare_scene_path


ARM_JOINTS = LEFT_ARM_JOINTS + RIGHT_ARM_JOINTS


@dataclass(frozen=True)
class EndpointSolution:
    ok: bool
    q: np.ndarray
    cost: float
    loop_grip_mm: float
    loop_axis_error: float
    loop_orientation_error: float
    hand_center_y_mm: float
    max_hand_grip_y_abs_mm: float
    head_x_m: float
    head_x_error_m: float
    head_z_m: float
    head_z_error_m: float
    head_axis_error: float
    head_axis_y: float
    min_joint_margin_rad: float
    torso_contacts: int
    max_warm_step_rad: float
    joints_rad: dict[str, float]


def parse_float_list(text: str) -> list[float]:
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def make_head_axis_target(down_dot: float, lateral: float) -> np.ndarray:
    down_dot = float(np.clip(down_dot, -1.0, 1.0))
    lateral = float(np.clip(lateral, -0.5, 0.5))
    if down_dot * down_dot + lateral * lateral >= 1.0:
        lateral = 0.0
    forward = float(np.sqrt(max(0.0, 1.0 - down_dot * down_dot - lateral * lateral)))
    return normalize([forward, lateral, -down_dot])


class EndpointSweeper:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        poses = yaml.safe_load(POSES_CONFIG.read_text(encoding="utf-8"))
        if args.base_pose not in poses:
            raise SystemExit(f"Base pose {args.base_pose!r} is missing from configs/poses.yaml.")

        self.base_pose = poses[args.base_pose]
        qpos = build_initial_qpos(args.base_pose, self.base_pose)
        scene_path = prepare_scene_path(
            qpos,
            grip_roll_phase_deg=pose_grip_roll_phase(self.base_pose),
            left_weld_distance_m=args.left_distance,
        )
        self.model = mujoco.MjModel.from_xml_path(str(scene_path))
        self.data = mujoco.MjData(self.model)
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        mujoco.mj_forward(self.model, self.data)

        self.addrs = joint_qpos_addrs(self.model, ARM_JOINTS)
        self.lower, self.upper = joint_ranges(self.model, ARM_JOINTS)
        self.base_q = self.data.qpos[self.addrs].copy()
        self.base_joints = {name: float(self.data.qpos[addr]) for name, addr in zip(ARM_JOINTS, self.addrs)}
        self.head_geom_id = self.model.geom(args.head_geom).id
        self.hammer_grip_body_id = self.model.body(args.hammer_grip_body).id
        self.left_elbow_body_id = self.model.body(args.left_elbow_body).id
        self.right_elbow_body_id = self.model.body(args.right_elbow_body).id
        self.left_wrist_body_id = self.model.body(args.left_wrist_body).id
        self.right_wrist_body_id = self.model.body(args.right_wrist_body).id
        self.right_loop_site_id = self.model.site(args.right_loop_site).id
        self.left_loop_site_id = self.model.site(args.left_loop_site).id

        initial_head_axis = self.data.xmat[self.hammer_grip_body_id].reshape(3, 3)[:, 0].copy()
        self.initial_head_lateral = float(initial_head_axis[1])

    def solve_pose(
        self,
        *,
        target_head_z_m: float,
        head_down_dot: float,
        warm_joints: dict[str, float],
        target_head_x_m: float | None = None,
    ) -> EndpointSolution:
        target_lateral = (
            self.initial_head_lateral
            if self.args.head_axis_lateral_target is None
            else self.args.head_axis_lateral_target
        )
        target_axis = make_head_axis_target(head_down_dot, target_lateral)
        warm = np.array([warm_joints.get(name, self.base_joints[name]) for name in ARM_JOINTS], dtype=float)
        regularization_center = warm.copy()

        def apply(x: np.ndarray) -> None:
            self.data.qpos[self.addrs] = x
            mujoco.mj_forward(self.model, self.data)

        def residual(x: np.ndarray) -> np.ndarray:
            apply(x)
            right_axis, right_grip = body_axis_and_grip(
                self.model,
                self.data,
                self.args.right_tool_body,
                self.args.clamp_center_offset,
            )
            left_axis, left_grip = body_axis_and_grip(
                self.model,
                self.data,
                self.args.left_tool_body,
                self.args.clamp_center_offset,
            )
            right_loop_x = self.data.site_xmat[self.right_loop_site_id].reshape(3, 3)[:, 0]
            left_loop_x = self.data.site_xmat[self.left_loop_site_id].reshape(3, 3)[:, 0]
            right_left_grip = right_grip + right_axis * self.args.left_distance
            head_axis = self.data.xmat[self.hammer_grip_body_id].reshape(3, 3)[:, 0]
            head_x = self.data.geom_xpos[self.head_geom_id][0]
            head_z = self.data.geom_xpos[self.head_geom_id][2]
            hand_grip_y = np.array([right_grip[1], left_grip[1]])
            grip_center_x = 0.5 * (right_grip[0] + left_grip[0])
            wrist_center_x = 0.5 * (
                self.data.xpos[self.left_wrist_body_id][0] + self.data.xpos[self.right_wrist_body_id][0]
            )
            grip_center_x_residual = (
                np.array([grip_center_x - self.args.grip_center_x_target]) * self.args.grip_center_x_weight
                if self.args.grip_center_x_target is not None and self.args.grip_center_x_weight > 0.0
                else np.empty(0)
            )
            wrist_center_x_residual = (
                np.array([wrist_center_x - self.args.wrist_center_x_target]) * self.args.wrist_center_x_weight
                if self.args.wrist_center_x_target is not None and self.args.wrist_center_x_weight > 0.0
                else np.empty(0)
            )
            grip_center_x_minimize = (
                np.array([grip_center_x]) * self.args.minimize_grip_center_x_weight
                if self.args.minimize_grip_center_x_weight > 0.0
                else np.empty(0)
            )
            wrist_center_x_minimize = (
                np.array([wrist_center_x]) * self.args.minimize_wrist_center_x_weight
                if self.args.minimize_wrist_center_x_weight > 0.0
                else np.empty(0)
            )
            head_x_residual = (
                np.array([head_x - float(target_head_x_m)]) * self.args.head_x_weight
                if target_head_x_m is not None and self.args.head_x_weight > 0.0
                else np.empty(0)
            )
            return np.concatenate(
                [
                    (left_grip - right_left_grip) * self.args.loop_grip_weight,
                    (left_axis - right_axis) * self.args.loop_axis_weight,
                    (left_loop_x - right_loop_x) * self.args.loop_orientation_weight,
                    np.array([head_z - target_head_z_m]) * self.args.head_z_weight,
                    head_x_residual,
                    (head_axis - target_axis) * self.args.head_axis_weight,
                    (hand_grip_y - self.args.hand_grip_y_target) * self.args.hand_grip_y_weight,
                    grip_center_x_residual,
                    wrist_center_x_residual,
                    grip_center_x_minimize,
                    wrist_center_x_minimize,
                    np.array(
                        [
                            max(0.0, self.args.elbow_clearance_y - self.data.xpos[self.left_elbow_body_id][1]),
                            max(0.0, self.args.elbow_clearance_y + self.data.xpos[self.right_elbow_body_id][1]),
                        ]
                    )
                    * self.args.elbow_clearance_weight,
                    (x - regularization_center) * self.args.regularization_weight,
                ]
            )

        starts = [
            warm,
            self.args.warm_blend * warm + (1.0 - self.args.warm_blend) * self.base_q,
            self.base_q,
        ]
        best = None
        for start in starts:
            result = least_squares(
                residual,
                np.clip(start, self.lower, self.upper),
                bounds=(self.lower, self.upper),
                max_nfev=self.args.max_nfev,
                xtol=1e-10,
                ftol=1e-10,
                gtol=1e-10,
            )
            if best is None or result.cost < best.cost:
                best = result

        assert best is not None
        apply(best.x)
        right_axis, right_grip = body_axis_and_grip(
            self.model,
            self.data,
            self.args.right_tool_body,
            self.args.clamp_center_offset,
        )
        left_axis, left_grip = body_axis_and_grip(
            self.model,
            self.data,
            self.args.left_tool_body,
            self.args.clamp_center_offset,
        )
        right_left_grip = right_grip + right_axis * self.args.left_distance
        right_loop_x = self.data.site_xmat[self.right_loop_site_id].reshape(3, 3)[:, 0]
        left_loop_x = self.data.site_xmat[self.left_loop_site_id].reshape(3, 3)[:, 0]
        head_axis = self.data.xmat[self.hammer_grip_body_id].reshape(3, 3)[:, 0].copy()
        head_x = float(self.data.geom_xpos[self.head_geom_id][0])
        head_z = float(self.data.geom_xpos[self.head_geom_id][2])
        hand_center_y = float(0.5 * (right_grip[1] + left_grip[1]))
        max_hand_grip_y_abs_mm = float(max(abs(right_grip[1]), abs(left_grip[1])) * 1000.0)
        torso_contacts = sum(torso_contact_pairs(self.model, self.data).values())
        margin = joint_margin(self.model, ARM_JOINTS, best.x)
        loop_grip_mm = float(np.linalg.norm(left_grip - right_left_grip) * 1000.0)
        loop_axis_error = float(np.linalg.norm(left_axis - right_axis))
        loop_orientation_error = float(np.linalg.norm(left_loop_x - right_loop_x))
        head_x_error = 0.0 if target_head_x_m is None else float(head_x - target_head_x_m)
        head_z_error = float(head_z - target_head_z_m)
        head_axis_error = float(np.linalg.norm(head_axis - target_axis))
        joints_rad = {name: float(value) for name, value in zip(ARM_JOINTS, best.x)}
        ok = (
            loop_grip_mm <= self.args.max_loop_grip_mm
            and loop_axis_error <= self.args.max_loop_axis_error
            and abs(head_z_error) <= self.args.max_head_z_error
            and torso_contacts <= self.args.max_torso_contacts
            and margin >= self.args.min_joint_margin_rad
        )
        return EndpointSolution(
            ok=ok,
            q=best.x.copy(),
            cost=float(best.cost),
            loop_grip_mm=loop_grip_mm,
            loop_axis_error=loop_axis_error,
            loop_orientation_error=loop_orientation_error,
            hand_center_y_mm=hand_center_y * 1000.0,
            max_hand_grip_y_abs_mm=max_hand_grip_y_abs_mm,
            head_x_m=head_x,
            head_x_error_m=head_x_error,
            head_z_m=head_z,
            head_z_error_m=head_z_error,
            head_axis_error=head_axis_error,
            head_axis_y=float(head_axis[1]),
            min_joint_margin_rad=margin,
            torso_contacts=torso_contacts,
            max_warm_step_rad=float(np.max(np.abs(best.x - warm))),
            joints_rad=joints_rad,
        )


def print_joints(title: str, solution: EndpointSolution) -> None:
    print(title)
    for name, value in solution.joints_rad.items():
        print(f"  {name}: {value:.11f}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-pose", default="knee_double_v0_start")
    parser.add_argument("--start-head-z", default="0.78,0.82,0.86,0.90")
    parser.add_argument("--end-head-z", default="0.60,0.62,0.64,0.66")
    parser.add_argument("--start-head-down-dot", type=float, default=0.50)
    parser.add_argument("--end-head-down-dot", type=float, default=0.85)
    parser.add_argument(
        "--head-axis-lateral-target",
        type=float,
        default=None,
        help="Target y component for the hammer/clamp axis. Default keeps the base-pose lateral component.",
    )
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
    parser.add_argument("--right-loop-site", default="right_hammer_left_grip_site")
    parser.add_argument("--left-loop-site", default="left_hammer_clamp_center")
    parser.add_argument("--loop-grip-weight", type=float, default=300.0)
    parser.add_argument("--loop-axis-weight", type=float, default=100.0)
    parser.add_argument("--loop-orientation-weight", type=float, default=80.0)
    parser.add_argument("--head-z-weight", type=float, default=100.0)
    parser.add_argument("--head-x-target", type=float, default=None)
    parser.add_argument("--head-x-weight", type=float, default=0.0)
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
    args = parser.parse_args()

    sweeper = EndpointSweeper(args)
    start_targets = parse_float_list(args.start_head_z)
    end_targets = parse_float_list(args.end_head_z)
    results = []
    for end_z, start_z in itertools.product(end_targets, start_targets):
        end_solution = sweeper.solve_pose(
            target_head_z_m=end_z,
            head_down_dot=args.end_head_down_dot,
            warm_joints=sweeper.base_joints,
            target_head_x_m=args.head_x_target,
        )
        start_solution = sweeper.solve_pose(
            target_head_z_m=start_z,
            head_down_dot=args.start_head_down_dot,
            warm_joints=end_solution.joints_rad,
            target_head_x_m=args.head_x_target,
        )
        amplitude = start_solution.head_z_m - end_solution.head_z_m
        pair_ok = end_solution.ok and start_solution.ok and amplitude > 0.10
        results.append((pair_ok, amplitude, start_z, end_z, start_solution, end_solution))

    results.sort(key=lambda item: (not item[0], -item[1], item[2], item[3]))
    print(
        "ok amp_m start_target end_target start_z end_z "
        "start_y_mm end_y_mm start_margin end_margin "
        "start_contacts end_contacts start_loop_mm end_loop_mm"
    )
    for pair_ok, amplitude, start_z, end_z, start_solution, end_solution in results:
        print(
            f"{pair_ok} {amplitude:.3f} {start_z:.2f} {end_z:.2f} "
            f"{start_solution.head_z_m:.3f} {end_solution.head_z_m:.3f} "
            f"{start_solution.hand_center_y_mm:.1f} {end_solution.hand_center_y_mm:.1f} "
            f"{start_solution.min_joint_margin_rad:.3f} {end_solution.min_joint_margin_rad:.3f} "
            f"{start_solution.torso_contacts} {end_solution.torso_contacts} "
            f"{start_solution.loop_grip_mm:.3f} {end_solution.loop_grip_mm:.3f}"
        )

    best = next((result for result in results if result[0]), results[0])
    _, amplitude, start_target, end_target, start_solution, end_solution = best
    print("\nBEST")
    print(f"start_target_z={start_target:.2f} end_target_z={end_target:.2f} actual_amp={amplitude:.3f}")
    print_joints("END_JOINTS", end_solution)
    print_joints("START_JOINTS", start_solution)
    return 0 if best[0] else 2


if __name__ == "__main__":
    raise SystemExit(main())
