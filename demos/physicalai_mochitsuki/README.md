# PhysicalAI Mochitsuki Demo Staging

This directory carries render-only documentation and versioned PhysicalAI
mochitsuki demo snapshots while porting the work into `shota3527/mochi`.

Do not place runtime control logic here. Use these upstream-style locations:

- `apps/` for runnable scripts.
- `assets/mujoco/` for MuJoCo scenes and adapter assets.
- `configs/trajectory.yaml` for named trajectories.
- `configs/poses.yaml` for named poses.
- `core/` for reusable IK, validation, and safety helpers.

Generated videos, keyframes, and diagnostic sheets should stay out of Git by
default.

Version snapshots are under `versions/`:

- `v2`: PhysicalAI Git tag `v2`.
- `v3`: frozen PhysicalAI v3 baseline.
- `latest`: current best PhysicalAI mainline with shota adapter geometry mapping.

Use `apps/render_physicalai_mochitsuki_snapshot.py` from the repo root to run a
snapshot without moving files into the original PhysicalAI directory layout.
