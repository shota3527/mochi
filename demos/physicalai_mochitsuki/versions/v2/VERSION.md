# PhysicalAI Mochitsuki v2

Source: PhysicalAI Git tag `v2`, commit `a6992b03107cb69ef125e9c21365c69bad34c6d4`.

Files:

- `mochitsuki_demo.py`: offline kinematic trajectory, IK, render/check/diagnose entrypoint.
- `scene_mochitsuki.xml`: MuJoCo scene template used by the v2 demo.

Notes:

- This is an older visual demo snapshot.
- It does not include the shota adapter geometry mapping.
- The trajectory is embedded in Python keyframes, not in `configs/trajectory.yaml`.
- Use it as historical reference, not as a real robot low-level trajectory.

Local wrapper check status:

- `apps/render_physicalai_mochitsuki_snapshot.py v2 -- --mode check` passes.
- It prints high joint speed and acceleration warnings, so it is visual/simulation reference only.
