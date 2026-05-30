# Mochi Hammer Planning Notes

This file is for another engineer or coding agent taking over the offline hammer motion planning flow.

## Directory Split

- `planner/`: offline planning, IK sweeps, trajectory generation, validation. These scripts may overwrite YAML configs when `--write-config` is used.
- `apps/`: realtime control or DDS replay. Do not put offline planners here.
- `sim/`: MuJoCo simulator and scene generation.
- `configs/poses.yaml`: named simulator/robot start and end poses.
- `configs/trajectory.yaml`: named trajectories for replay.

## Current Naming

- `knee_double_v0`: regenerated coarse kneeling two-hand swing, 5 waypoints, height-targeted closed-loop grip geometry.
- `knee_double_v1`: regenerated finer kneeling two-hand swing, 20 waypoints, height-targeted closed-loop grip geometry.
- `knee_double_v1_start`: start pose for simulator/replay.
- `knee_double_v1_end`: end pose for simulator/replay.

The `knee_double_v1` lower body is intentionally copied from `knee_double_v0`. Both generated trajectories store arm joints only, and their kneeling lower-body pose comes from `knee_double_v0_start` / `knee_double_v0_end`.

## Main Planner

Use:

```bash
PYTHONPYCACHEPREFIX=/tmp/mochi_pycache .venv/bin/python \
  planner/plan_dual_hold_height_swing.py \
  --trajectory-name knee_double_v1 \
  --base-pose knee_double_v0_start \
  --samples 20 \
  --head-axis-lateral-target 0.0 \
  --minimize-wrist-center-x-weight 0.01 \
  --write-config
```

This writes `knee_double_v1` into `configs/trajectory.yaml`. To regenerate the coarse v0 trajectory after a tool-link geometry change, use the same command with `--trajectory-name knee_double_v0 --samples 5`.

After regenerating the trajectory, sync the start/end poses from the first and last waypoint:

```bash
PYTHONPYCACHEPREFIX=/tmp/mochi_pycache .venv/bin/python - <<'PY'
from pathlib import Path
import yaml

pose_path = Path("configs/poses.yaml")
traj_path = Path("configs/trajectory.yaml")
poses = yaml.safe_load(pose_path.read_text(encoding="utf-8"))
traj = yaml.safe_load(traj_path.read_text(encoding="utf-8"))["knee_double_v1"]

for suffix, wp, lower_src in [
    ("start", traj["waypoints"][0], poses["knee_double_v0_start"]),
    ("end", traj["waypoints"][-1], poses["knee_double_v0_end"]),
]:
    name = f"knee_double_v1_{suffix}"
    lower_joints = {
        k: v
        for k, v in lower_src["joints_rad"].items()
        if not ("shoulder" in k or "elbow" in k or "wrist" in k)
    }
    joints = dict(lower_joints)
    joints.update(wp["joints_rad"])
    poses[name] = {
        "memo": (
            f"{suffix.capitalize()} pose for knee_double_v1 height-targeted kneeling swing; "
            f"lower body copied from knee_double_v0_{suffix}."
        ),
        "base_z_m": lower_src.get("base_z_m"),
        "base_pitch_deg": lower_src.get("base_pitch_deg"),
        "joints_rad": joints,
        "dual_hold_geometry": {
            "source": "configs/trajectory.yaml:knee_double_v1",
            "left_grip_distance_m": traj["left_grip_distance_m"],
            "clamp_center_offset_m": traj["clamp_center_offset_m"],
            "grip_roll_phase_deg": traj["grip_roll_phase_deg"],
            "closed_loop_only": True,
            "full_pose_locked": False,
        },
    }

pose_path.write_text(yaml.safe_dump(poses, sort_keys=False, width=1000), encoding="utf-8")
print("updated knee_double_v1_start/end")
PY
```

## Constraint Semantics

Hard-ish constraints in the least-squares residual:

- Dual-hand loop: left clamp point equals right clamp point plus the fixed stick-axis distance.
- Clamp axes aligned: left and right clamp axes match.
- Head height: `target_head_z_m` follows the start-to-end height schedule.

Soft objectives:

- `--head-axis-lateral-target 0.0`: keeps the hammer/clamp axis in the robot `xz` plane by pushing axis `y` toward zero.
- `--hand-grip-y-weight`: pushes each clamp/grip point's `y` coordinate toward `--hand-grip-y-target`. This is per hand, not the average of both hands.
- `--minimize-wrist-center-x-weight`: targetless loss that pulls the wrist center closer to the body in forward `x`.
- `--elbow-clearance-*`: keeps elbows away from the torso side.
- `--regularization-weight`: keeps the IK close to the warm-start branch.

Important: do not reintroduce `hand-center-y-target` or `hand-center-y-weight`. The old average-hand-center objective can hide one hand being far from the center plane.

## Current Useful Defaults

- `--samples 20`: enough resolution for dual-hand closed-loop replay.
- `--start-head-z 0.86`
- `--end-head-z 0.60`
- `--start-head-down-dot 0.50`
- `--end-head-down-dot 0.85`
- `--head-axis-lateral-target 0.0`
- `--hand-grip-y-weight 0.25`
- `--minimize-wrist-center-x-weight 0.01`

Do not blindly increase `--hand-grip-y-weight`. In the latest sweep:

- `0.25` passed with `min_arm_joint_margin_rad` about `0.126`.
- `0.5` made the clamp points closer to center, but dropped margin to about `0.069` and failed the safety threshold.
- Larger values pushed the start pose near joint limits.

## Validation Checklist

The planner prints a summary and writes validation into `configs/trajectory.yaml`.

Treat this as the minimum acceptable offline result:

- `passed: true`
- `max_torso_contacts: 0`
- `min_arm_joint_margin_rad >= 0.10`
- `max_neighbor_joint_step_rad <= 0.16`
- `max_loop_grip_mm <= 1.0`
- `max_loop_axis_error <= 0.002`
- `max_head_z_error_m <= 0.015`

Run an interpolated replay-style validation after writing YAML:

```bash
PYTHONPYCACHEPREFIX=/tmp/mochi_pycache .venv/bin/python - <<'PY'
import yaml, mujoco, numpy as np
from sim.run_sim_controller import build_initial_qpos, prepare_scene_path, pose_grip_roll_phase, pose_left_weld_distance
from core.stick_ik import LEFT_ARM_JOINTS, RIGHT_ARM_JOINTS, joint_qpos_addrs, joint_ranges, body_axis_and_grip
from planner.plan_dual_hold_swing import torso_contact_pairs, joint_margin
from apps.replay_trajectory import smoothstep

poses = yaml.safe_load(open("configs/poses.yaml"))
traj = yaml.safe_load(open("configs/trajectory.yaml"))["knee_double_v1"]
pose = poses["knee_double_v1_start"]
qpos = build_initial_qpos("knee_double_v1_start", pose)
model = mujoco.MjModel.from_xml_path(str(prepare_scene_path(
    qpos,
    grip_roll_phase_deg=pose_grip_roll_phase(pose),
    left_weld_distance_m=pose_left_weld_distance(pose),
)))
data = mujoco.MjData(model)
mujoco.mj_resetDataKeyframe(model, data, 0)
mujoco.mj_forward(model, data)

arm = LEFT_ARM_JOINTS + RIGHT_ARM_JOINTS
addrs = joint_qpos_addrs(model, arm)
lower, upper = joint_ranges(model, arm)
qwps = [np.array([wp["joints_rad"][name] for name in arm], dtype=float) for wp in traj["waypoints"]]
body_id = model.body("right_hammer_grip").id

max_contacts = 0
min_margin = 1e9
max_loop = 0.0
max_axis = 0.0
max_axis_y = 0.0
max_grip_y = 0.0
samples = 0

for q0, q1 in zip(qwps[:-1], qwps[1:]):
    for s in range(21):
        a = smoothstep(s / 20)
        q = q0 + a * (q1 - q0)
        data.qpos[addrs] = q
        mujoco.mj_forward(model, data)
        right_axis, right_grip = body_axis_and_grip(model, data, "right_hammer_tool", traj["clamp_center_offset_m"])
        left_axis, left_grip = body_axis_and_grip(model, data, "left_hammer_clamp", traj["clamp_center_offset_m"])
        head_axis = data.xmat[body_id].reshape(3, 3)[:, 0]
        max_axis_y = max(max_axis_y, abs(float(head_axis[1])))
        max_grip_y = max(max_grip_y, abs(float(right_grip[1])), abs(float(left_grip[1])))
        max_loop = max(max_loop, float(np.linalg.norm(left_grip - (right_grip + right_axis * traj["left_grip_distance_m"])) * 1000))
        max_axis = max(max_axis, float(np.linalg.norm(left_axis - right_axis)))
        max_contacts = max(max_contacts, sum(torso_contact_pairs(model, data).values()))
        min_margin = min(min_margin, joint_margin(model, arm, q))
        if np.any(q < lower) or np.any(q > upper):
            raise SystemExit("joint limit violation")
        samples += 1

print(
    "interpolated_validate",
    "samples", samples,
    "max_loop_mm", round(max_loop, 6),
    "max_axis", round(max_axis, 8),
    "max_axis_y_abs", round(max_axis_y, 6),
    "max_grip_y_abs_mm", round(max_grip_y * 1000, 3),
    "min_margin", round(min_margin, 6),
    "max_contacts", max_contacts,
)
PY
```

## Simulator Replay

Start simulator at the matching pose:

```bash
python sim/run_sim_controller.py --interface eth3 --pose knee_double_v1_start
```

Replay once before looping:

```bash
python apps/replay_trajectory.py \
  --interface eth3 \
  --trajectory knee_double_v1 \
  --gravity-comp \
  --cycles 1 \
  --enable-command
```

For real hardware, do not start with `--loop`. First run one cycle, reduce gravity compensation scale if torque or tracking looks bad, and verify the robot starts close to `knee_double_v1_start`.
