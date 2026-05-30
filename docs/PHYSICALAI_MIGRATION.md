# PhysicalAI Mochitsuki Migration Plan

This staging tree is organized with `shota3527/mochi` as the primary repository
shape. PhysicalAI files should be ported into that shape instead of keeping a
parallel project layout.

Source repository: https://github.com/shota3527/mochi
Imported commit: `92d6a7de94988426666b1b16d73aefc457e463ba`

## Current Upstream Layout

```text
apps/                  CLI tools and runnable control/demo entrypoints
assets/mujoco/         MuJoCo XML scenes and primitive adapter model
backends/              DDS / SDK / future simulator backend adapters
configs/               Joint limits, safety, hammer, poses, trajectories
core/                  IK, safety filtering, trajectory and hammer helpers
demos/                 PhysicalAI visual/offline demo staging area
sim/                   MuJoCo viewer plus DDS lowstate/lowcmd simulator
docs/                  Integration and migration notes
```

## Included PhysicalAI Snapshots

Three PhysicalAI offline trajectory/demo snapshots are already staged under:

```text
demos/physicalai_mochitsuki/versions/v2/
demos/physicalai_mochitsuki/versions/v3/
demos/physicalai_mochitsuki/versions/latest/
```

Each snapshot includes its own `mochitsuki_demo.py` and `scene_mochitsuki.xml`.
The trajectory data is still embedded in Python keyframes and IK constraints.
The version manifest is `configs/physicalai_mochitsuki_versions.yaml`.

Run a snapshot from the repo root with:

```bash
python apps/render_physicalai_mochitsuki_snapshot.py latest -- --mode check --render-smoke
```

## Where PhysicalAI Work Should Land

Use these target locations when moving code from `/Users/eric/Projects/PhysicalAI` into the colleague repo.

```text
PhysicalAI source                               shota3527/mochi target
--------------------------------------------------------------------------------
demos/mochitsuki/mochitsuki_demo.py             apps/render_mochitsuki_demo.py
demos/mochitsuki/scene_mochitsuki.xml           assets/mujoco/physicalai_mochitsuki_scene.xml
validated joint waypoints / keyframes           configs/trajectory.yaml
motion pose names / initial postures            configs/poses.yaml
adapter / hammer dimensions                     configs/hammer.yaml and assets/mujoco/
trajectory checks and metrics                   core/trajectory_validation.py
render-only documentation                       demos/physicalai_mochitsuki/README.md
generated videos / keyframes                    outputs/ or logs/ only; do not commit by default
```

## Naming Rules

- Prefix imported PhysicalAI trajectories with `physicalai_` until they are accepted as upstream defaults.
- Prefer explicit motion names, for example `physicalai_mochitsuki_roll_v0`.
- Keep real robot command scripts gated behind `--enable-command`.
- Keep render-only scripts separate from DDS command scripts.

## Integration Order

1. Keep the colleague primitive adapter as the base hammer model.
2. Port the PhysicalAI kinematic trajectory into `configs/trajectory.yaml` as joint waypoints.
3. Add a render/inspection app under `apps/` that reads the trajectory config instead of hardcoding paths.
4. Move safety and visual validation metrics into `core/trajectory_validation.py`.
5. Only after simulation checks pass, connect the trajectory to `apps/replay_trajectory.py`.

## Safety Boundary

The current PhysicalAI mochitsuki motion is still a simulation and visual
prototype. It must not be treated as a real robot low-level trajectory until it
has velocity limits, acceleration limits, torque limits, collision checks,
foot/COM stability checks, and an emergency stop path in the colleague repo.
