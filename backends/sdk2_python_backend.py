"""Thin SDK2/DDS backend shared by simulator and real robot apps."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass

import numpy as np
from unitree_sdk2py.core.channel import (
    ChannelFactoryInitialize,
    ChannelPublisher,
    ChannelSubscriber,
)
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
from unitree_sdk2py.utils.crc import CRC


TOPIC_LOWCMD = "rt/lowcmd"
TOPIC_LOWSTATE = "rt/lowstate"
G1_29DOF_NUM_MOTORS = 29


@dataclass
class JointStateSample:
    q: np.ndarray
    dq: np.ndarray
    tau_est: np.ndarray
    mode_machine: int | None
    stamp: float


class G1Sdk2Backend:
    """Subscribe to G1 low state and optionally publish low commands."""

    def __init__(
        self,
        domain_id: int = 1,
        interface: str | None = "lo",
        num_motors: int = G1_29DOF_NUM_MOTORS,
    ):
        self.domain_id = domain_id
        self.interface = interface
        self.num_motors = num_motors
        self._sample: JointStateSample | None = None
        self._sample_event = threading.Event()
        self._crc = CRC()
        self._initialized = False
        self._low_state_subscriber = None
        self._low_cmd_publisher = None

    def initialize(self, enable_commands: bool = False) -> None:
        if self.interface:
            ChannelFactoryInitialize(self.domain_id, self.interface)
        else:
            ChannelFactoryInitialize(self.domain_id)

        self._low_state_subscriber = ChannelSubscriber(TOPIC_LOWSTATE, LowState_)
        self._low_state_subscriber.Init(self._low_state_handler, 10)

        if enable_commands:
            self._low_cmd_publisher = ChannelPublisher(TOPIC_LOWCMD, LowCmd_)
            self._low_cmd_publisher.Init()

        self._initialized = True

    def _low_state_handler(self, msg: LowState_) -> None:
        q = np.array([msg.motor_state[i].q for i in range(self.num_motors)], dtype=float)
        dq = np.array([msg.motor_state[i].dq for i in range(self.num_motors)], dtype=float)
        tau = np.array(
            [msg.motor_state[i].tau_est for i in range(self.num_motors)], dtype=float
        )
        mode_machine = getattr(msg, "mode_machine", None)
        self._sample = JointStateSample(q=q, dq=dq, tau_est=tau, mode_machine=mode_machine, stamp=time.time())
        self._sample_event.set()

    def wait_for_state(self, timeout_s: float = 5.0) -> JointStateSample:
        if not self._initialized:
            raise RuntimeError("Backend must be initialized before waiting for state.")
        if not self._sample_event.wait(timeout_s):
            raise TimeoutError(f"No {TOPIC_LOWSTATE} sample received within {timeout_s:.1f}s.")
        assert self._sample is not None
        return self._sample

    def latest_state(self) -> JointStateSample | None:
        return self._sample

    def publish_single_joint_position(
        self,
        joint_index: int,
        q_des: float,
        mode_machine: int | None,
        kp: float = 20.0,
        kd: float = 1.0,
    ) -> None:
        if self._low_cmd_publisher is None:
            raise RuntimeError("Command publisher was not enabled.")
        if not 0 <= joint_index < self.num_motors:
            raise ValueError(f"joint_index must be in [0, {self.num_motors - 1}]")

        cmd = unitree_hg_msg_dds__LowCmd_()
        if hasattr(cmd, "mode_pr"):
            cmd.mode_pr = 0
        if mode_machine is not None and hasattr(cmd, "mode_machine"):
            cmd.mode_machine = mode_machine

        for i in range(self.num_motors):
            cmd.motor_cmd[i].mode = 0
            cmd.motor_cmd[i].tau = 0.0
            cmd.motor_cmd[i].q = 0.0
            cmd.motor_cmd[i].dq = 0.0
            cmd.motor_cmd[i].kp = 0.0
            cmd.motor_cmd[i].kd = 0.0

        cmd.motor_cmd[joint_index].mode = 1
        cmd.motor_cmd[joint_index].tau = 0.0
        cmd.motor_cmd[joint_index].q = float(q_des)
        cmd.motor_cmd[joint_index].dq = 0.0
        cmd.motor_cmd[joint_index].kp = float(kp)
        cmd.motor_cmd[joint_index].kd = float(kd)
        cmd.crc = self._crc.Crc(cmd)
        self._low_cmd_publisher.Write(cmd)

    def publish_position_command(
        self,
        q_des,
        mode_machine: int | None,
        kp,
        kd,
        tau=None,
    ) -> None:
        if self._low_cmd_publisher is None:
            raise RuntimeError("Command publisher was not enabled.")

        q_des = np.asarray(q_des, dtype=float)
        kp = np.asarray(kp, dtype=float)
        kd = np.asarray(kd, dtype=float)
        if tau is None:
            tau = np.zeros(self.num_motors, dtype=float)
        tau = np.asarray(tau, dtype=float)
        if q_des.shape != (self.num_motors,):
            raise ValueError(f"q_des must have shape ({self.num_motors},)")
        if kp.shape != (self.num_motors,) or kd.shape != (self.num_motors,):
            raise ValueError(f"kp and kd must have shape ({self.num_motors},)")
        if tau.shape != (self.num_motors,):
            raise ValueError(f"tau must have shape ({self.num_motors},)")

        cmd = unitree_hg_msg_dds__LowCmd_()
        if hasattr(cmd, "mode_pr"):
            cmd.mode_pr = 0
        if mode_machine is not None and hasattr(cmd, "mode_machine"):
            cmd.mode_machine = mode_machine

        for i in range(self.num_motors):
            cmd.motor_cmd[i].mode = 1
            cmd.motor_cmd[i].tau = float(tau[i])
            cmd.motor_cmd[i].q = float(q_des[i])
            cmd.motor_cmd[i].dq = 0.0
            cmd.motor_cmd[i].kp = float(kp[i])
            cmd.motor_cmd[i].kd = float(kd[i])

        cmd.crc = self._crc.Crc(cmd)
        self._low_cmd_publisher.Write(cmd)
