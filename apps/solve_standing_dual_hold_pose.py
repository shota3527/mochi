#!/usr/bin/env python3
"""Solve and render a centered standing two-hand stick pose."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mujoco
import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.stick_ik import grips_from_stick, make_centered_stick_target, solve_dual_hold_ik
from sim.run_sim_controller import POSES_CONFIG, build_initial_qpos, prepare_scene_path


def render_views(model, data, out_dir: Path) -> None:
    try:
        from PIL import Image
    except ImportError:
        print("PIL is not installed; skipping renders.")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    views = [
        ("side", 90, -8, 1.55, [0.32, 0.0, 0.85]),
        ("front", 180, -10, 1.55, [0.32, 0.0, 0.85]),
        ("top", 180, -70, 1.45, [0.32, 0.0, 0.82]),
    ]
    for name, azimuth, elevation, distance, lookat in views:
        renderer = mujoco.Renderer(model, height=480, width=640)
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = lookat
        cam.distance = distance
        cam.azimuth = azimuth
        cam.elevation = elevation
        renderer.update_scene(data, camera=cam)
        path = out_dir / f"{name}.png"
        Image.fromarray(renderer.render()).save(path)
        renderer.close()
        print(path)


def print_pose_block(
    name: str,
    base_z: float,
    result,
    stick,
    right_distance: float,
    left_distance: float,
    clamp_center_offset: float,
    elbow_clearance_y: float,
) -> None:
    print(f"\n{name}:")
    print("  memo: Standing centered two-hand stick-hold inspection pose solved from stick target and dual-arm IK.")
    print(f"  base_z_m: {base_z:.3f}")
    print("  joints_rad:")
    for joint_name, value in result.joint_values.items():
        print(f"    {joint_name}: {value:.11f}")
    print("  dual_hold_geometry:")
    print("    source: apps/solve_standing_dual_hold_pose.py")
    print(f"    stick_root_world_m: [{stick.root_m[0]:.4f}, {stick.root_m[1]:.4f}, {stick.root_m[2]:.4f}]")
    print(f"    stick_axis_world: [{stick.axis_m[0]:.4f}, {stick.axis_m[1]:.4f}, {stick.axis_m[2]:.4f}]")
    print(f"    right_grip_distance_m: {right_distance:.3f}")
    print(f"    left_grip_distance_m: {left_distance:.3f}")
    print(f"    clamp_center_offset_m: {clamp_center_offset:.3f}")
    print(f"    elbow_clearance_y_target_m: {elbow_clearance_y:.3f}")
    print(f"    ik_right_tool_root_error_m: {result.right_tool_error_m:.4f}")
    print(f"    ik_right_grip_error_m: {result.right_grip_error_m:.4f}")
    print(f"    ik_left_grip_error_m: {result.left_grip_error_m:.4f}")
    print(f"    ik_right_axis_error: {result.right_axis_error:.4f}")
    print("  notes:")
    print("    - Visual inspection pose only; not a trajectory.")
    print("    - Right and left hands use matching clamp center targets on the stick centerline.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-pose", default="hammer_mounted_elbow_65")
    parser.add_argument("--pose-name", default="standing_dual_hold_v0")
    parser.add_argument("--root", nargs=3, type=float, default=(0.12, 0.0, 0.69), metavar=("X", "Y", "Z"))
    parser.add_argument("--pitch-deg", type=float, default=45.0)
    parser.add_argument("--right-distance", type=float, default=0.0)
    parser.add_argument("--left-distance", type=float, default=0.20)
    parser.add_argument("--clamp-center-offset", type=float, default=0.0)
    parser.add_argument("--elbow-clearance-y", type=float, default=0.14)
    parser.add_argument("--render-dir", default="/tmp/mochi_standing_dual_hold_v1")
    args = parser.parse_args()

    poses = yaml.safe_load(POSES_CONFIG.read_text(encoding="utf-8"))
    base_pose = poses[args.base_pose]
    qpos = build_initial_qpos(args.base_pose, base_pose)
    scene_path = prepare_scene_path(qpos)
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)

    stick = make_centered_stick_target(
        root_m=args.root,
        pitch_rad=np.deg2rad(args.pitch_deg),
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
        elbow_clearance_y_m=args.elbow_clearance_y,
    )

    print(f"cost={result.cost:.6f}")
    print(f"right_tool_root_error_m={result.right_tool_error_m:.4f}")
    print(f"right_grip_error_m={result.right_grip_error_m:.4f}")
    print(f"left_grip_error_m={result.left_grip_error_m:.4f}")
    print(f"right_axis_error={result.right_axis_error:.4f}")
    print_pose_block(
        args.pose_name,
        float(base_pose.get("base_z_m", 0.793)),
        result,
        stick,
        args.right_distance,
        args.left_distance,
        args.clamp_center_offset,
        args.elbow_clearance_y,
    )
    render_views(model, data, Path(args.render_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
