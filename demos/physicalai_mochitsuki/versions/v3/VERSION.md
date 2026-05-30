# PhysicalAI Mochitsuki v3

Source: PhysicalAI frozen v3 snapshot.

Files:

- `mochitsuki_demo.py`: frozen v3 offline kinematic trajectory and validation script.
- `scene_mochitsuki.xml`: frozen v3 MuJoCo scene template.
- `render_mochitsuki_demo.sh`: original PhysicalAI v3 render helper.

Notes:

- This is the frozen baseline before the latest shota adapter integration.
- It uses the PhysicalAI multi-camera validation workflow.
- The trajectory is embedded in Python keyframes, not in `configs/trajectory.yaml`.
- Keep this version immutable unless explicitly creating a new named version.

Local wrapper check status:

- `apps/render_physicalai_mochitsuki_snapshot.py v3 -- --mode check` passes.
- It prints high joint speed, acceleration, and estimated torque ratio warnings, so it is visual/simulation reference only.
