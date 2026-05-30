#!/usr/bin/env python3
"""Visual MuJoCo simulator for Unitree G1 29DOF with mochi hammer."""

import argparse
import math
import threading
import time
import tempfile
from pathlib import Path
from threading import Thread

import mujoco
import mujoco.viewer
import yaml
from unitree_sdk2py.core.channel import (
    ChannelFactoryInitialize,
    ChannelPublisher,
    ChannelSubscriber,
)
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowState_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_


UNITREE_G1_DIR = Path("/home/shota/dev/unitree/unitree_mujoco/unitree_robots/g1")
UNITREE_G1_29DOF_XML = UNITREE_G1_DIR / "g1_29dof.xml"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MOCHI_G1_SCENE = PROJECT_ROOT / "assets" / "mujoco" / "mochi_g1_scene.xml"
POSES_CONFIG = PROJECT_ROOT / "configs" / "poses.yaml"
HAMMER_CONFIG = PROJECT_ROOT / "configs" / "hammer.yaml"
TOPIC_LOWCMD = "rt/lowcmd"
TOPIC_LOWSTATE = "rt/lowstate"
TOPIC_ARMSDK = "rt/arm_sdk"
SIM_DT = 0.005
VIEWER_DT = 0.02
LOWER_BODY_JOINTS = tuple(range(12))
ARM_SDK_JOINTS = tuple(range(12, 29))
ARM_SDK_ENABLE_INDEX = 29


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pose", default="hammer_mounted_elbow_65")
    parser.add_argument("--interface", default="eth3")
    parser.add_argument("--domain-id", type=int, default=1)
    parser.add_argument("--run", action="store_true", help="Start unpaused.")
    parser.add_argument("--zero-gravity", action="store_true", help="Disable gravity for motor-command plumbing checks.")
    args = parser.parse_args()

    poses = yaml.safe_load(POSES_CONFIG.read_text(encoding="utf-8"))
    pose = poses[args.pose]
    initial_qpos = build_initial_qpos(args.pose, pose)
    scene_path = prepare_scene_path(
        initial_qpos,
        grip_roll_phase_deg=pose_grip_roll_phase(pose),
        left_weld_distance_m=pose_left_weld_distance(pose),
    )
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    if args.zero_gravity:
        model.opt.gravity[:] = 0.0
    data = mujoco.MjData(model)

    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)

    lock = threading.Lock()
    sim_state = {
        "paused": not args.run,
        "auto_started_from_lowcmd": args.run,
        "arm_sdk_active": False,
        "butt_fixture_active": False,
        "butt_fixture_qpos": data.qpos.copy(),
    }
    right_elbow_id = model.joint("right_elbow_joint").id
    right_elbow_q = data.qpos[model.jnt_qposadr[right_elbow_id]]

    print(
        f"G1 first-frame viewer: pose={args.pose} scene={scene_path} "
        f"domain_id={args.domain_id} paused={sim_state['paused']} "
        f"right_elbow={right_elbow_q:.6f} gravity={model.opt.gravity}",
        flush=True,
    )

    def key_callback(keycode: int) -> None:
        if keycode == 32:
            sim_state["paused"] = not sim_state["paused"]
            sim_state["auto_started_from_lowcmd"] = True
            print(f"paused={sim_state['paused']}", flush=True)

    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        model.opt.timestep = SIM_DT
        viewer.sync()

        time.sleep(0.2)

        viewer_thread = Thread(
            target=physics_viewer_thread,
            args=(viewer, lock),
            daemon=True,
        )
        sim_thread = Thread(
            target=simulation_thread,
            args=(viewer, model, data, lock, sim_state, args),
            daemon=True,
        )

        viewer_thread.start()
        sim_thread.start()

        while viewer.is_running():
            time.sleep(0.1)

    return 0


def simulation_thread(viewer, model, data, lock: threading.Lock, sim_state: dict, args) -> None:
    ChannelFactoryInitialize(args.domain_id, args.interface)
    low_state = unitree_hg_msg_dds__LowState_()
    low_state_publisher = ChannelPublisher(TOPIC_LOWSTATE, LowState_)
    low_state_publisher.Init()

    low_cmd_subscriber = ChannelSubscriber(TOPIC_LOWCMD, LowCmd_)
    low_cmd_subscriber.Init(lambda msg: apply_low_cmd(msg, data, model.nu, lock, sim_state), 10)
    arm_sdk_subscriber = ChannelSubscriber(TOPIC_ARMSDK, LowCmd_)
    arm_sdk_subscriber.Init(lambda msg: apply_arm_sdk_cmd(msg, data, model.nu, lock, sim_state), 10)
    print(f"DDS initialized: interface={args.interface}", flush=True)

    while viewer.is_running():
        step_start = time.perf_counter()
        with lock:
            apply_sim_butt_fixture(data, sim_state)
            if sim_state["paused"]:
                mujoco.mj_forward(model, data)
            else:
                mujoco.mj_step(model, data)
                apply_sim_butt_fixture(data, sim_state)
                mujoco.mj_forward(model, data)
            publish_low_state(data, model.nu, low_state, low_state_publisher)
        sleep_time = model.opt.timestep - (time.perf_counter() - step_start)
        if sleep_time > 0:
            time.sleep(sleep_time)


def physics_viewer_thread(viewer, lock: threading.Lock) -> None:
    while viewer.is_running():
        with lock:
            viewer.sync()
        time.sleep(VIEWER_DT)


def apply_low_cmd(msg, data, num_motors: int, lock: threading.Lock, sim_state: dict) -> None:
    with lock:
        if sim_state["paused"] and not sim_state["auto_started_from_lowcmd"]:
            sim_state["paused"] = False
            sim_state["auto_started_from_lowcmd"] = True
            print("paused=False (lowcmd received)", flush=True)
        q = data.sensordata[:num_motors]
        dq = data.sensordata[num_motors : 2 * num_motors]
        for i in range(num_motors):
            data.ctrl[i] = (
                msg.motor_cmd[i].tau
                + msg.motor_cmd[i].kp * (msg.motor_cmd[i].q - q[i])
                + msg.motor_cmd[i].kd * (msg.motor_cmd[i].dq - dq[i])
            )


def apply_arm_sdk_cmd(msg, data, num_motors: int, lock: threading.Lock, sim_state: dict) -> None:
    with lock:
        weight = float(max(0.0, min(1.0, msg.motor_cmd[ARM_SDK_ENABLE_INDEX].q)))
        sim_state["arm_sdk_active"] = weight > 0.0
        if sim_state["arm_sdk_active"]:
            sim_state["butt_fixture_active"] = True
            if sim_state["paused"] and not sim_state["auto_started_from_lowcmd"]:
                sim_state["paused"] = False
                sim_state["auto_started_from_lowcmd"] = True
                print("paused=False (arm_sdk received)", flush=True)

        q = data.sensordata[:num_motors]
        dq = data.sensordata[num_motors : 2 * num_motors]
        for i in ARM_SDK_JOINTS:
            data.ctrl[i] = weight * (
                msg.motor_cmd[i].tau
                + msg.motor_cmd[i].kp * (msg.motor_cmd[i].q - q[i])
                + msg.motor_cmd[i].kd * (msg.motor_cmd[i].dq - dq[i])
            )


def apply_sim_butt_fixture(data, sim_state: dict) -> None:
    if not sim_state.get("butt_fixture_active"):
        return
    fixture_qpos = sim_state.get("butt_fixture_qpos")
    if fixture_qpos is None:
        return
    data.qpos[:7] = fixture_qpos[:7]
    data.qvel[:6] = 0.0
    for i in LOWER_BODY_JOINTS:
        data.qpos[7 + i] = fixture_qpos[7 + i]
        data.qvel[6 + i] = 0.0


def publish_low_state(data, num_motors: int, msg, publisher) -> None:
    q = data.sensordata[:num_motors]
    dq = data.sensordata[num_motors : 2 * num_motors]
    tau = data.sensordata[2 * num_motors : 3 * num_motors]
    for i in range(num_motors):
        msg.motor_state[i].q = float(q[i])
        msg.motor_state[i].dq = float(dq[i])
        msg.motor_state[i].tau_est = float(tau[i])
    if hasattr(msg, "mode_machine"):
        msg.mode_machine = 0
    publisher.Write(msg)


def build_initial_qpos(pose_name: str, pose: dict):
    scene_path = prepare_scene_path(
        grip_roll_phase_deg=pose_grip_roll_phase(pose),
        left_weld_distance_m=pose_left_weld_distance(pose),
    )
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    qpos = model.qpos0.copy()
    if "base_z_m" in pose:
        qpos[2] = float(pose["base_z_m"])
    if "base_pitch_deg" in pose:
        half_pitch = math.radians(float(pose["base_pitch_deg"])) * 0.5
        qpos[3:7] = [math.cos(half_pitch), 0.0, math.sin(half_pitch), 0.0]
    for joint_name, q in pose.get("joints_rad", {}).items():
        joint_id = model.joint(joint_name).id
        qpos[model.jnt_qposadr[joint_id]] = float(q)
    print(f"Initial pose written before viewer launch: {pose_name}", flush=True)
    return qpos


def hammer_default_grip_roll_phase() -> float:
    hammer_config = yaml.safe_load(HAMMER_CONFIG.read_text(encoding="utf-8"))
    return float(hammer_config.get("default_grip_roll_phase_deg", 0.0))


def pose_grip_roll_phase(pose: dict) -> float:
    return float(pose.get("dual_hold_geometry", {}).get("grip_roll_phase_deg", hammer_default_grip_roll_phase()))


def pose_left_weld_distance(pose: dict) -> float | None:
    geometry = pose.get("dual_hold_geometry", {})
    if not geometry:
        return None
    return float(geometry.get("left_grip_distance_m", 0.20))


def prepare_scene_path(
    initial_qpos=None,
    grip_roll_phase_deg: float | None = None,
    left_weld_distance_m: float | None = None,
) -> Path:
    xml = MOCHI_G1_SCENE.read_text(encoding="utf-8")
    robot_xml = patch_g1_right_hand_to_hammer(
        grip_roll_phase_deg=grip_roll_phase_deg,
        left_weld_distance_m=left_weld_distance_m,
    )
    xml = xml.replace(
        '<include file="/home/shota/dev/unitree/unitree_mujoco/unitree_robots/g1/g1_29dof.xml"/>',
        f'<include file="{robot_xml.name}"/>',
    )
    if initial_qpos is not None:
        qpos_text = " ".join(f"{float(q):.10g}" for q in initial_qpos)
        keyframe = f"""

  <keyframe>
    <key name="initial_pose" qpos="{qpos_text}"/>
  </keyframe>
"""
        xml = xml.replace("\n</mujoco>", f"{keyframe}</mujoco>")
    if left_weld_distance_m is not None:
        equality = """

  <equality>
    <weld name="left_hand_stick_weld"
      site1="right_hammer_left_grip_site"
      site2="left_hammer_clamp_center"
      solref="0.005 1"
      solimp="0.95 0.99 0.001"/>
  </equality>
"""
        xml = xml.replace("\n</mujoco>", f"{equality}</mujoco>")
    generated = Path(tempfile.gettempdir()) / "mochi_g1_scene.xml"
    generated.write_text(xml, encoding="utf-8")

    # MuJoCo resolves meshes relative to the generated robot XML.
    meshes_link = Path(tempfile.gettempdir()) / "meshes"
    if not meshes_link.exists():
        meshes_link.symlink_to(UNITREE_G1_DIR / "meshes", target_is_directory=True)
    return generated


def patch_g1_right_hand_to_hammer(
    grip_roll_phase_deg: float | None = None,
    left_weld_distance_m: float | None = None,
) -> Path:
    """Generate a G1 XML with both rubber hands replaced by matching clamps."""
    original = UNITREE_G1_29DOF_XML.read_text(encoding="utf-8")
    hammer_config = yaml.safe_load(HAMMER_CONFIG.read_text(encoding="utf-8"))
    if grip_roll_phase_deg is None:
        grip_roll_phase_deg = hammer_default_grip_roll_phase()
    half_phase = math.radians(grip_roll_phase_deg) * 0.5
    grip_roll_quat = f"{math.cos(half_phase):.7f} 0 0 {math.sin(half_phase):.7f}"
    handle = hammer_config["handle"]
    head = hammer_config["head"]
    handle_length_m = float(handle["length_m"])
    handle_radius_m = float(handle["radius_m"])
    handle_mass_kg = float(handle["mass_kg"])
    head_length_m = float(head["length_m"])
    head_radius_m = float(head["diameter_m"]) * 0.5
    head_mass_kg = float(head["mass_kg"])
    head_enabled = bool(hammer_config.get("head", {}).get("enabled", True))
    left_grip_site = ""
    if left_weld_distance_m is not None:
        left_grip_site = f"""                              <site name="right_hammer_left_grip_site"
                                pos="0 0 {left_weld_distance_m:.6f}"
                                quat="0.5 0 0 -0.8660254"
                                size="0.010" rgba="0.1 0.5 1 1"/>
"""
    head_geom = (
        f"""                              <geom name="right_hammer_head" type="cylinder"
                                fromto="0 0 {handle_length_m:.6f} {head_length_m:.6f} 0 {handle_length_m:.6f}"
                                size="{head_radius_m:.6f}"
                                rgba="0.58 0.34 0.14 1" mass="{head_mass_kg:.4f}"
                                contype="1" conaffinity="1"
                                friction="1.0 0.02 0.002" solref="0.02 1" solimp="0.9 0.95 0.001"/>
"""
        if head_enabled
        else ""
    )
    left_rubber_hand_geom = """                          <geom pos="0.0415 0.003 0" quat="1 0 0 0" type="mesh" contype="0"
                            conaffinity="0" group="1" density="0" rgba="0.7 0.7 0.7 1"
                            mesh="left_rubber_hand" />"""
    right_rubber_hand_geom = """                          <geom pos="0.0415 -0.003 0" quat="1 0 0 0" type="mesh" contype="0"
                            conaffinity="0" group="1" density="0" rgba="0.7 0.7 0.7 1"
                            mesh="right_rubber_hand" />"""
    left_clamp = """                          <body name="left_hammer_clamp" pos="0.0415 0.003 0">
                            <!-- Same clamp geometry as the right side. The clamp center
                                 for IK and grip bookkeeping is the tool root. -->
                            <geom name="left_hammer_adapter_upper" type="box" pos="0 0.024 0.030"
                              size="0.020 0.006 0.020" rgba="0.25 0.35 0.42 0.35"
                              mass="0.04"
                              contype="0" conaffinity="0" friction="0.8 0.02 0.002"/>
                            <geom name="left_hammer_adapter_lower" type="box" pos="0 -0.024 0.030"
                              size="0.020 0.006 0.020" rgba="0.25 0.35 0.42 0.35"
                              mass="0.04"
                              contype="0" conaffinity="0" friction="0.8 0.02 0.002"/>
                            <geom name="left_hammer_clamp_upper" type="box" pos="0 0.024 0"
                              size="0.022 0.006 0.052" rgba="0.25 0.35 0.42 0.38"
                              mass="0.06"
                              contype="0" conaffinity="0" friction="0.8 0.02 0.002"/>
                            <geom name="left_hammer_clamp_lower" type="box" pos="0 -0.024 0"
                              size="0.022 0.006 0.052" rgba="0.25 0.35 0.42 0.38"
                              mass="0.06"
                              contype="0" conaffinity="0" friction="0.8 0.02 0.002"/>
                            <geom name="left_hammer_antirotation_pin" type="cylinder"
                              pos="0 0 0" quat="0.7071068 0.7071068 0 0"
                              size="0.004 0.030" rgba="0.10 0.12 0.13 0.50"
                              mass="0.02"
                              contype="0" conaffinity="0" friction="0.8 0.02 0.002"/>
                            <site name="left_hammer_clamp_center" pos="0 0 0"
                              size="0.010" rgba="0.1 0.5 1 1"/>
                          </body>"""
    hammer_tool = f"""                          <body name="right_hammer_tool" pos="0.0415 -0.003 0">
                            <!-- Same clamp geometry as the left side. The wood handle
                                 rear end and clamp center are at this wrist/tool root. -->
                            <geom name="right_hammer_adapter_upper" type="box" pos="0 0.024 0.030"
                              size="0.020 0.006 0.020" rgba="0.25 0.35 0.42 0.35"
                              mass="0.04"
                              contype="0" conaffinity="0" friction="0.8 0.02 0.002"/>
                            <geom name="right_hammer_adapter_lower" type="box" pos="0 -0.024 0.030"
                              size="0.020 0.006 0.020" rgba="0.25 0.35 0.42 0.35"
                              mass="0.04"
                              contype="0" conaffinity="0" friction="0.8 0.02 0.002"/>
                            <geom name="right_hammer_clamp_upper" type="box" pos="0 0.024 0"
                              size="0.022 0.006 0.052" rgba="0.25 0.35 0.42 0.38"
                              mass="0.06"
                              contype="0" conaffinity="0" friction="0.8 0.02 0.002"/>
                            <geom name="right_hammer_clamp_lower" type="box" pos="0 -0.024 0"
                              size="0.022 0.006 0.052" rgba="0.25 0.35 0.42 0.38"
                              mass="0.06"
                              contype="0" conaffinity="0" friction="0.8 0.02 0.002"/>
                            <geom name="right_hammer_antirotation_pin" type="cylinder"
                              pos="0 0 0" quat="0.7071068 0.7071068 0 0"
                              size="0.004 0.030" rgba="0.10 0.12 0.13 0.50"
                              mass="0.02"
                              contype="0" conaffinity="0" friction="0.8 0.02 0.002"/>
                            <site name="right_hammer_clamp_center" pos="0 0 0"
                              size="0.010" rgba="0.1 0.5 1 1"/>

                            <!-- Grip phase: rotate the wood-stick/hammer assembly
                                 around the stick centerline at the wrist/tool root.
                                 The hammer shape itself remains handle local +Z and
                                 head local +X, fixed at 90 deg. -->
                            <body name="right_hammer_grip" pos="0 0 0" quat="{grip_roll_quat}">
                              <geom name="right_hammer_handle" type="capsule"
                                fromto="0 0 0 0 0 {handle_length_m:.6f}" size="{handle_radius_m:.6f}"
                                rgba="0.62 0.38 0.16 1" mass="{handle_mass_kg:.4f}"
                                contype="1" conaffinity="1"
                                friction="1.0 0.02 0.002" solref="0.02 1" solimp="0.9 0.95 0.001"/>
{left_grip_site.rstrip()}
{head_geom.rstrip()}
                            </body>
                          </body>"""
    if left_rubber_hand_geom not in original:
        raise RuntimeError("Could not find left rubber hand geom to replace.")
    if right_rubber_hand_geom not in original:
        raise RuntimeError("Could not find right rubber hand geom to replace.")

    patched = original.replace(left_rubber_hand_geom, left_clamp)
    patched = patched.replace(right_rubber_hand_geom, hammer_tool)
    generated = Path(tempfile.gettempdir()) / "g1_29dof_mochi_hammer.xml"
    generated.write_text(patched, encoding="utf-8")
    return generated


if __name__ == "__main__":
    raise SystemExit(main())
