"""Piper 起始位姿控制。

采集适配器保持严格只读。本模块仅由遥操会话在开始前显式调用，控制完成或失败后
立即断开 CAN 连接，避免与遥操 ROS 控制器并发下发指令。
"""

from __future__ import annotations

import math
import time
from typing import Any, Callable

import numpy as np

from ..config import InitialPoseConfig, RobotConfig
from ..errors import DeviceError, HardwareDependencyError
from ..models import RobotState
from .piper import euler_xyz_to_xyzw, read_piper_robot_state, rotate_vector_xyzw


_RAD_TO_MILLI_DEG = 180000.0 / math.pi
_COMMAND_INTERVAL_S = 0.05


class PiperInitialPoseController:
    """通过 Piper SDK 将机械臂移动到 YAML 明确指定的起始位姿。"""

    def __init__(
        self,
        robot_config: RobotConfig,
        initial_pose: InitialPoseConfig,
        *,
        interface_factory: Callable[[str | None, int], Any] | None = None,
        sleep_fn: Callable[[float], None] = time.sleep,
        monotonic_fn: Callable[[], float] = time.monotonic,
    ) -> None:
        self._robot_config = robot_config
        self._initial_pose = initial_pose
        self._interface_factory = interface_factory
        self._sleep = sleep_fn
        self._monotonic = monotonic_fn
        self._interface: Any | None = None

    def move(self) -> RobotState:
        """使能后低速发送目标，直到反馈进入容差；超时后切换为待机模式。"""

        if not self._initial_pose.enabled:
            raise DeviceError("robot.initial_pose.enabled=false，拒绝发送起始位姿控制指令。")
        target = self._target()
        self._connect()
        target_sent = False
        try:
            self._wait_until_enabled()
            deadline = self._monotonic() + self._initial_pose.timeout_s
            while True:
                self._send_target(target)
                target_sent = True
                state = self._read_state()
                if self._reached_target(state, target):
                    return state
                if self._monotonic() >= deadline:
                    self._enter_standby()
                    raise DeviceError(
                        f"Piper 未在 {self._initial_pose.timeout_s:.1f} 秒内到达配置的起始位姿，已切换为待机模式。"
                    )
                self._sleep(_COMMAND_INTERVAL_S)
        except DeviceError:
            if target_sent:
                self._enter_standby()
            raise
        except Exception as error:
            if target_sent:
                self._enter_standby()
            raise DeviceError(f"Piper 起始位姿控制失败：{error}") from error
        finally:
            self.stop()

    def stop(self) -> None:
        """断开主动控制连接；失败时不掩盖此前的控制错误。"""

        if self._interface is not None:
            try:
                self._interface.DisconnectPort()
            finally:
                self._interface = None

    def _connect(self) -> None:
        try:
            if self._interface_factory is not None:
                interface = self._interface_factory(self._robot_config.can_name, self._robot_config.dh_is_offset)
            else:
                from piper_sdk import C_PiperInterface_V2

                interface = C_PiperInterface_V2(
                    can_name=self._robot_config.can_name,
                    dh_is_offset=self._robot_config.dh_is_offset,
                )
            # 主动运动需要 SDK 初始化和使能；与只读采集适配器刻意分离。
            interface.ConnectPort()
            self._interface = interface
        except ImportError as error:
            raise HardwareDependencyError("缺少 piper_sdk；请执行 uv sync 安装项目依赖。") from error
        except Exception as error:
            self.stop()
            raise DeviceError(f"Piper CAN 接口 {self._robot_config.can_name} 控制连接失败：{error}") from error

    def _wait_until_enabled(self) -> None:
        assert self._interface is not None
        deadline = self._monotonic() + min(self._initial_pose.timeout_s, 10.0)
        while self._monotonic() < deadline:
            if bool(self._interface.EnablePiper()):
                return
            self._sleep(_COMMAND_INTERVAL_S)
        raise DeviceError("Piper 使能超时，未发送起始位姿目标。")

    def _target(self) -> np.ndarray:
        if self._initial_pose.mode == "joint":
            values = self._initial_pose.joint_positions_rad
            if values is None:
                raise DeviceError("缺少 robot.initial_pose.joint_positions_rad。")
            # Piper SDK 示例中的固定软件限位与现场固件反馈并不总是一致，且 SDK 默认
            # 也不启用该限位。不能因陈旧的静态表拒绝“回到当前实测姿态”；最终限位由
            # 机械臂固件执行。本模块仍保留低速、超时和反馈到位检查。
            return np.asarray(values, dtype=np.float64)
        values = self._initial_pose.tcp_pose
        if values is None:
            raise DeviceError("缺少 robot.initial_pose.tcp_pose。")
        return np.asarray(values, dtype=np.float64)

    def _send_target(self, target: np.ndarray) -> None:
        assert self._interface is not None
        if self._initial_pose.mode == "joint":
            self._interface.MotionCtrl_2(0x01, 0x01, self._initial_pose.speed_percent, 0x00)
            self._interface.JointCtrl(*(round(float(value) * _RAD_TO_MILLI_DEG) for value in target))
            return

        # 配置与采集写入的均为物理 TCP；SDK EndPoseCtrl 接收法兰末端，因此反算工具偏移。
        orientation = target[3:]
        offset = rotate_vector_xyzw(euler_xyz_to_xyzw(orientation), np.asarray(self._robot_config.tool_offset_m))
        flange_position = target[:3] - offset
        command = [
            *(round(float(value) * 1e6) for value in flange_position),
            *(round(float(value) * _RAD_TO_MILLI_DEG) for value in orientation),
        ]
        self._interface.MotionCtrl_2(0x01, 0x00, self._initial_pose.speed_percent, 0x00)
        self._interface.EndPoseCtrl(*command)

    def _read_state(self) -> RobotState:
        assert self._interface is not None
        return read_piper_robot_state(self._interface, self._robot_config, "xyz_rxryrz")

    def _reached_target(self, state: RobotState, target: np.ndarray) -> bool:
        if self._initial_pose.mode == "joint":
            return bool(np.max(np.abs(state.joint_positions - target)) <= self._initial_pose.joint_tolerance_rad)
        assert state.tcp_pose is not None
        position_error = float(np.linalg.norm(state.tcp_pose[:3] - target[:3]))
        orientation_error = _wrapped_angle_error(state.tcp_pose[3:] - target[3:])
        return (
            position_error <= self._initial_pose.position_tolerance_m
            and orientation_error <= self._initial_pose.orientation_tolerance_rad
        )

    def _enter_standby(self) -> None:
        if self._interface is None:
            return
        try:
            # 退出 CAN 指令控制模式，阻止本模块继续驱动；不替代现场硬件急停。
            self._interface.MotionCtrl_2(0x00, 0x00, 0, 0x00)
        except Exception:
            pass


def _wrapped_angle_error(delta: np.ndarray) -> float:
    """计算欧拉角各轴按 2π 回绕后的最大误差。"""

    wrapped = (np.asarray(delta, dtype=np.float64) + math.pi) % (2.0 * math.pi) - math.pi
    return float(np.max(np.abs(wrapped)))
