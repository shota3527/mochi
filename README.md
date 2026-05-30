# Mochi G1 Hammer

Safe simulation and control workflow for a Unitree G1 EDU 29DOF robot with a custom hammer adapter replacing the original hand.

Current scope:

- Run the G1 29DOF MuJoCo simulator.
- Read DDS low-level joint state.
- Confirm joint order and joint limits.
- Apply project-level safety filtering before any command.
- Run tiny single upper-body joint motion tests.
- Keep a simplified primitive hammer model.

Out of scope for now:

- RL
- Isaac Lab
- vision
- realistic impact/contact modeling
- real hammer swing trajectories

## Paths

Project root:

```bash
~/workspace/mochi
```

Python venv:

```bash
~/workspace/mochi/.venv
```

Official Unitree repos:

```bash
~/dev/unitree/unitree_sdk2
~/dev/unitree/unitree_sdk2_python
~/dev/unitree/unitree_mujoco
~/dev/unitree/install
```

## Setup Check

```bash
cd ~/workspace/mochi
source .venv/bin/activate

python - <<'PY'
from unitree_sdk2py.core.channel import ChannelFactoryInitialize
print("unitree_sdk2py OK")

import mujoco
print("mujoco OK")
PY
```

## DDS Interface

Unitree SDK2 uses DDS, and DDS must bind to a network interface.

List interfaces:

```bash
ip -o link show
ip -o addr show
```

On this machine, `eth3` worked for simulator DDS. `lo` did not discover peers reliably because CycloneDDS reported loopback multicast issues.

For simulator:

```bash
--interface eth3
```

For the real robot, use the network interface physically connected to the G1 control network, for example `eth0`, `enp...`, or `enx...`.

Successful real G1 wired setup on this machine:

```text
Windows Ethernet adapter index: 16
WSL/Linux robot interface: eth0
Host IP: 192.168.123.222/24
G1 IP observed by ping: 192.168.123.161
DDS domain: 0
```

After replugging the robot cable, Windows may reset the Ethernet network back
to `Public`. In Administrator PowerShell, set it back to `Private`:

```powershell
Set-NetConnectionProfile -InterfaceIndex 16 -NetworkCategory Private
```

Then in WSL, make sure DDS multicast goes through the robot cable:

```bash
sudo ip route del 224.0.0.0/4 2>/dev/null || true
sudo ip route add 224.0.0.0/4 dev eth0
ip route get 239.255.0.1
```

Expected route:

```text
multicast 239.255.0.1 dev eth0 src 192.168.123.222
```

Do not guess for real robot control. First verify read-only state:

```bash
cd ~/workspace/mochi
source .venv/bin/activate

python apps/dump_state.py \
  --interface eth0 \
  --domain-id 0 \
  --timeout 10
```

If this fails after replugging the cable, follow [connect.md](connect.md).

## Run First-Frame G1 Viewer

This viewer is for pose and hammer geometry inspection. It publishes DDS `rt/lowstate`, subscribes to DDS `rt/lowcmd`, and starts paused by default. Press Space to pause/resume, or pass `--run` to start unpaused. The viewer also auto-starts once when the first DDS low command arrives from a sim command app.

Default hammer memo pose:

```bash
cd ~/workspace/mochi
source .venv/bin/activate

python sim/run_sim_controller.py --interface eth3
```

Hammer variants are selected with `--hammer-variant`:

```bash
python sim/run_sim_controller.py --interface eth3 --hammer-variant full_600_300
python sim/run_sim_controller.py --interface eth3 --hammer-variant easy_450_150
python sim/run_sim_controller.py --interface eth3 --hammer-variant compact_450_300
python sim/run_sim_controller.py --interface eth3 --hammer-variant handle_only_450
```

Then read state from a second terminal:

```bash
python apps/dump_state.py --interface eth3 --timeout 5
```

To inspect the current kneel candidate:

```bash
python sim/run_sim_controller.py --interface eth3 --pose kneel_static_v0
```

`kneel` is based on the knee/ankle limit-fold pose, with both knees opened 8 deg from the hard knee limit to reduce interference. It is the baseline for stable hammering inspection.

If the viewer prints Mesa/Zink errors such as:

```text
MESA: error: ZINK: failed to choose pdev
glx: failed to create drisw screen
```

force Mesa software rendering for the visual simulator:

```bash
LIBGL_ALWAYS_SOFTWARE=1 GALLIUM_DRIVER=llvmpipe \
python sim/run_sim_controller.py --interface eth3 --pose kneel_static_v0
```

To replay the locked kneeling hammer trajectory:

Terminal 1 must start the simulator at the matching first waypoint pose:

```bash
python sim/run_sim_controller.py --interface eth3 --pose knee_double_v0_start
```

Terminal 2:

```bash
python apps/replay_trajectory.py \
  --interface eth3 \
  --trajectory knee_double_v0 \
  --gravity-comp \
  --no-release-motion-mode
```

`replay_trajectory.py` first ramps from the current joint state to the first
waypoint with the configured `--max-step-rad` limit, then starts the hammer
trajectory. This lets the simulator recover from a mismatched initial pose
without a hard start.

To watch the hammer swing forward and backward repeatedly:

```bash
python apps/replay_trajectory.py \
  --interface eth3 \
  --trajectory knee_double_v0 \
  --loop \
  --no-release-motion-mode
```

Trajectory replay always uses a staged SPACE-key workflow:

1. Press SPACE to ramp to the first waypoint.
2. Press SPACE to start the trajectory.
3. By default, normal completion releases the active command interface from the
   current command.

Set `--return-to-start-s 6` when you explicitly want the old slow return to the
startup state before release.

There are two command paths:

`--arm-sdk` publishes upper-body commands on `rt/arm_sdk`, matching Unitree's G1
arm SDK example. It sets `motor_cmd[29].q = 1` while active and disables it on
exit. The built-in high-level motion mode stays running, so this is the path for
our first G1 workflow test.

Without `--arm-sdk`, the app publishes full-body low-level commands on
`rt/lowcmd` and releases the active Unitree high-level motion mode by default.
Use `--no-release-motion-mode` for simulator runs or for environments without
MotionSwitcher. Full-body lowcmd is not the path for the first real standing
test.

```bash
python apps/replay_trajectory.py \
  --interface eth3 \
  --trajectory real_link_hammer_mounted_elbow_v0 \
  --arm-sdk
```

Optional gravity feedforward is deliberately scoped for tuning. It sends torque
only to waist joints, shoulder joints, and elbows. Hips, knees, ankles, and
wrists still receive `0 Nm` feedforward.

```text
tau_ff[selected_joints] = scale * qfrc_bias[selected_dofs]
```

It uses the official Unitree `g1_29dof.xml` model, updates the configured motor
joint angles before each torque calculation, and reads MuJoCo `qfrc_bias` at
zero velocity. It does not load the mochi scene and never includes
hammer/clamp/tool bodies, so the term represents robot body support, not tool
payload. Replay limits each feedforward joint to `8 Nm` by default.

The MuJoCo calculation model is fixed to the official default G1 model for this
replay app. The hammering pose still comes from `configs/poses.yaml` and the
trajectory config; only the feedforward model stays tool-free.

Replay gains live in `configs/gain_profiles.yaml`. The `official` profile is
copied from Unitree's G1 low-level example. The default project profile is
`double_hand`: hip pitch and waist gains are 1.5x, shoulder/elbow gains use
the stable two-hand swing profile, hip roll/yaw, knees, and ankles keep the original baseline, and wrist gains stay at the original arm-SDK values.
Trajectories default to `double_hand`.

```bash
python apps/replay_trajectory.py \
  --interface eth3 \
  --trajectory knee_double_v1 \
  --loop \
  --gravity-comp
```

The default scene includes a primitive mochi-pounding target in front of the robot:

```text
assets/mujoco/mochi_g1_scene.xml
```

The fixed wooden base and mochi lump have ordinary MuJoCo collision/contact enabled. This is only for early collision checks and visual scene setup. It is not a realistic impact model, and the controller does not use contact force feedback yet.

The simulator also patches the included G1 model at runtime so the right rubber hand is replaced by a primitive hammer clamp. The forearm and wrist module stay intact; the tool attaches downstream of `right_wrist_yaw_link`.

## MuJoCo Initial Pose Rule

Initial robot poses must be written into `data.qpos`, not into `model.qpos0`.

For this project, the safe startup path is:

```python
mujoco.mj_resetDataKeyframe(model, data, 0)
mujoco.mj_forward(model, data)
viewer.sync()
```

Do not do this after loading the model:

```python
model.qpos0[:] = data.qpos
```

`data.qpos` is the live simulator state. `model.qpos0` is MuJoCo's compiled default/reference state. Changing `model.qpos0` after model load can make the viewer/reset/reference behavior misleading even when printed `data.qpos` values look correct.

If a reset-to-pose feature is needed, use the generated MJCF keyframe and call `mj_resetDataKeyframe()`, or explicitly rewrite `data.qpos` and call `mj_forward()`. Do not use `model.qpos0` as the project pose store.

## Dump Joint State

Terminal 2:

```bash
cd ~/workspace/mochi
source .venv/bin/activate

python apps/dump_state.py --interface eth3 --timeout 5
```

This prints:

- joint index
- joint name
- position `q`
- velocity `dq`

It also saves one JSON sample under:

```text
logs/
```

Use this before sending any motion commands.

## Safety Contract

All future commands must follow:

```text
trajectory -> safety_filter -> SDK/simulator command
```

Do not rely only on Unitree SDK internal safety. This project has a custom hammer load, so command limits must also exist at the project level.

Current filter:

- joint limit clamp
- command rate limit
- joint margin calculation
- near-joint-limit check

Config:

```text
configs/safety.yaml
configs/g1_29dof_joints.yaml
```

If joint margin is below `0.10 rad`, mark the pose dangerous.

## Hammer Model

Simplified primitive model only:

```text
sim/run_sim_controller.py
configs/hammer.yaml
```

Primitive parts:

- wrist-compatible short adapter: box
- split clamp: boxes plus anti-rotation pin
- wooden handle: capsule
- wooden head: cylinder, `0.06 m` diameter when enabled

The handle/head angle is fixed hammer geometry: the handle axis is tool `+Z`, the wooden head axis is tool `+X`, and the two stay at `90 deg` inside one rigid MuJoCo body. Do not tune that angle with robot joint poses.

The handle and head use hardwood density `700 kg/m^3` for mass estimates. Current variants in `configs/hammer.yaml`:

- `full_600_300`: `0.60 m` handle + `0.30 m` head, total about `1.072 kg`
- `easy_450_150`: `0.45 m` handle + `0.15 m` head, total about `0.711 kg`
- `compact_450_300`: `0.45 m` handle + `0.30 m` head, total about `1.008 kg`
- `handle_only_450`: `0.45 m` handle, no head, total about `0.414 kg`

The tool is mounted on `right_wrist_yaw_link`, local position `[0.0735, 0.0, 0.0]`. The wooden head is fixed at the front of the handle.

Static wrist moment estimate:

```text
M_wrist ~= hammer_mass * 9.81 * wrist_to_com_distance
```

No CAD mesh collision yet.

## Project Layout

```text
apps/
  dump_state.py
  replay_trajectory.py

sim/
  run_sim_controller.py

backends/
  sdk2_python_backend.py

core/
  safety_filter.py
  stick_ik.py

configs/
  g1_29dof_joints.yaml
  safety.yaml
  hammer.yaml
  trajectory.yaml

assets/mujoco/
logs/
```

## Real Robot Status

Real read-only DDS state has been connected successfully with `eth0`,
`192.168.123.222/24`, and DDS domain `0`.

Current real motion workflow uses Unitree `rt/arm_sdk` for upper-body commands,
leaving the built-in lower-body controller active. Always confirm
`dump_state.py` works before sending commands.

```bash
cd ~/workspace/mochi
source .venv/bin/activate

python apps/replay_trajectory.py \
  --interface eth0 \
  --domain-id 0 \
  --trajectory dual_hold_swing_v0 \
  --arm-sdk \
  --max-step-rad 0.003
```

SPACE workflow:

```text
SPACE 1: ramp from current state to first waypoint
SPACE 2: start trajectory
SPACE 3: hold current/final command for stick removal
SPACE 4: return to startup state, then release
```

Use the robot-side physical emergency stop if anything looks wrong. Terminal
`Ctrl+C` is only a software-level stop.
