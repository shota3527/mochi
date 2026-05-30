# PhysicalAI Mochitsuki Version Snapshots

This directory carries three PhysicalAI offline mochitsuki trajectory/demo
snapshots for sharing with the `shota3527/mochi` workflow.

The trajectories are embedded in each `mochitsuki_demo.py` as keyframes and IK
constraints. They have not yet been converted into `configs/trajectory.yaml`
joint-waypoint format.

## Versions

```text
v2/       Git tag v2, commit a6992b0. Older standing mochitsuki visual demo.
v3/       Frozen v3 snapshot from PhysicalAI. Multi-camera validated baseline.
latest/   Current best PhysicalAI mainline with shota adapter geometry mapping.
```

## Run Through the Colleague Repo Shape

From the `shota3527_mochi` root:

```bash
python apps/render_physicalai_mochitsuki_snapshot.py latest -- --mode check --render-smoke
python apps/render_physicalai_mochitsuki_snapshot.py v3 -- --mode render --camera-layout multi --save-keyframes
python apps/render_physicalai_mochitsuki_snapshot.py v2 -- --mode diagnose --diagnostic-samples 52
```

If Unitree MuJoCo is not in the default upstream path, pass:

```bash
python apps/render_physicalai_mochitsuki_snapshot.py latest \
  --unitree-g1-dir /path/to/unitree_mujoco/unitree_robots/g1 \
  -- --mode check --render-smoke
```

Generated videos and keyframes go under `outputs/physicalai_mochitsuki/` and
should not be committed by default.
