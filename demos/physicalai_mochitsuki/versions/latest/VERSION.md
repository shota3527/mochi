# PhysicalAI Mochitsuki Latest

Source: current PhysicalAI mainline at the time this staging tree was prepared.

Files:

- `mochitsuki_demo.py`: current best offline kinematic trajectory and validation script.
- `scene_mochitsuki.xml`: current scene with `shota3527/mochi` adapter clamp/pin geometry mapping.
- `render_mochitsuki_demo.sh`: current PhysicalAI render helper.

Notes:

- This is the preferred version to port forward.
- The shota primitive adapter is used visually in the scene, while the PhysicalAI
  offline IK still drives the G1 rubber hand links.
- The trajectory is embedded in Python keyframes, not in `configs/trajectory.yaml`.
- Before connecting it to DDS replay, convert the motion into joint waypoints and
  re-run velocity, acceleration, torque, collision, foot/COM, and emergency-stop checks.

Local wrapper check status:

- `apps/render_physicalai_mochitsuki_snapshot.py latest -- --mode check` passes without dynamic warnings in the current PhysicalAI environment.
