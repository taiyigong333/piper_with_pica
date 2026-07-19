"""Piper 只读反馈适配器。

SDK 的反馈值带有设备单位；本模块是唯一负责转换为数据集标准单位的边界。
"""

from __future__ import annotations

import math
import time
from typing import Any

import numpy as np

from ..config import RobotConfig
from ..errors import DeviceError, HardwareDependencyError
from ..models import RobotState
from .base import GripperDevice, RobotDevice


def euler_xyz_to_xyzw(euler_rad: np.ndarray) -> np.ndarray:
    """按 Piper SDK TCP 示例的 `Rz * Ry * Rx` 约定将 XYZ 欧拉角转四元数。"""

    rx, ry, rz = (float(value) for value in euler_rad)
    cx, sx = math.cos(rx / 2), math.sin(rx / 2)
    cy, sy = math.cos(ry / 2), math.sin(ry / 2)
    cz, sz = math.cos(rz / 2), math.sin(rz / 2)
    return np.asarray(
        [
            sx * cy * cz - cx * sy * sz,
            cx * sy * cz + sx * cy * sz,
            cx * cy * sz - sx * sy * cz,
            cx * cy * cz + sx * sy * sz,
        ],
        dtype=np.float64,
    )


def rotate_vector_xyzw(quaternion: np.ndarray, vector: np.ndarray) -> np.ndarray:
    """旋转工具偏移；使用矩阵形式避免引入额外姿态库。"""

    x, y, z, w = (float(value) for value in quaternion)
    rotation = np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )
    return rotation @ vector


def gripper_stroke_to_m(stroke_in_micrometers: int | float) -> float:
    """将 Piper SDK 的夹爪行程（0.001 mm）转换为标准单位米。"""

    stroke_m = float(stroke_in_micrometers) * 1e-6
    if not math.isfinite(stroke_m):
        raise DeviceError("Piper 夹爪反馈返回 NaN 或 Inf。")
    return stroke_m


class PiperRobot(RobotDevice):
    """通过 `piper_sdk` 读取 CAN 反馈，严格不下发运动或初始化查询指令。"""

    def __init__(self, config: RobotConfig, pose_representation: str) -> None:
        if pose_representation not in {"xyz_xyzw", "xyz_rxryrz"}:
            raise DeviceError("Piper 仅支持 xyz_xyzw 或 xyz_rxryrz。")
        self._config = config
        self._pose_representation = pose_representation
        self._interface: Any | None = None

    def start(self) -> None:
        if self._interface is not None:
            return
        try:
            from piper_sdk import C_PiperInterface_V2
        except ImportError as error:
            raise HardwareDependencyError("缺少 piper_sdk；请执行 uv sync 以安装本地子模块。") from error
        try:
            interface = C_PiperInterface_V2(
                can_name=self._config.can_name,
                dh_is_offset=self._config.dh_is_offset,
            )
            # piper_init=False 至关重要：禁止 PiperInit 发送任何查询或控制帧。
            interface.ConnectPort(piper_init=False)
            self._interface = interface
        except Exception as error:
            self.stop()
            raise DeviceError(f"Piper CAN 接口 {self._config.can_name} 启动失败：{error}") from error

    def read(self) -> RobotState:
        if self._interface is None:
            raise DeviceError("Piper 尚未启动。")
        try:
            joints_message = self._interface.GetArmJointMsgs().joint_state
            end_pose = self._interface.GetArmEndPoseMsgs().end_pose
            joint_values = [getattr(joints_message, f"joint_{index}") for index in range(1, 7)]
            joints_rad = np.asarray(joint_values, dtype=np.float64) * (math.pi / 180000.0)
            position_m = np.asarray([end_pose.X_axis, end_pose.Y_axis, end_pose.Z_axis], dtype=np.float64) * 1e-6
            euler_rad = np.asarray([end_pose.RX_axis, end_pose.RY_axis, end_pose.RZ_axis], dtype=np.float64) * (math.pi / 180000.0)
            quaternion = np.asarray(euler_xyz_to_xyzw(euler_rad), dtype=np.float64)
            position_m += rotate_vector_xyzw(quaternion, np.asarray(self._config.tool_offset_m, dtype=np.float64))
            # 工具偏移需要旋转矩阵，但落盘表示必须遵守现场配置。
            orientation = euler_rad if self._pose_representation == "xyz_rxryrz" else quaternion
            tcp_pose = np.concatenate((position_m, orientation))
            if not np.isfinite(joints_rad).all() or not np.isfinite(tcp_pose).all():
                raise DeviceError("Piper 返回 NaN 或 Inf。")
            return RobotState(timestamp=time.time(), joint_positions=joints_rad, tcp_pose=tcp_pose)
        except DeviceError:
            raise
        except Exception as error:
            raise DeviceError(f"Piper 反馈读取失败：{error}") from error

    def read_gripper_position(self) -> float:
        """读取 Piper 夹爪行程反馈，不发送任何 CAN 控制帧。"""

        if self._interface is None:
            raise DeviceError("Piper 尚未启动。")
        try:
            # SDK 的 grippers_angle 是 0.001 mm；采集格式统一使用米。
            message = self._interface.GetArmGripperMsgs()
            if float(message.time_stamp) <= 0:
                raise DeviceError("尚未收到 Piper 夹爪 0x2A8 反馈，不能使用 SDK 默认的零行程。")
            return gripper_stroke_to_m(message.gripper_state.grippers_angle)
        except DeviceError:
            raise
        except Exception as error:
            raise DeviceError(f"Piper 夹爪反馈读取失败：{error}") from error

    def wait_for_gripper_feedback(self, timeout_s: float = 1.0) -> float:
        """等待首个有效夹爪反馈，避免启动瞬间把 SDK 默认零值写入数据集。"""

        deadline = time.monotonic() + timeout_s
        last_error: DeviceError | None = None
        while time.monotonic() < deadline:
            try:
                return self.read_gripper_position()
            except DeviceError as error:
                last_error = error
                time.sleep(0.005)
        raise DeviceError(f"等待 Piper 夹爪 0x2A8 反馈超时（{timeout_s:.1f} 秒）。") from last_error

    def stop(self) -> None:
        if self._interface is not None:
            try:
                self._interface.DisconnectPort()
            finally:
                self._interface = None


class PiperGripper(GripperDevice):
    """复用 `PiperRobot` 的只读 CAN 连接读取夹爪行程。"""

    def __init__(self, robot: PiperRobot) -> None:
        self._robot = robot
        self._started = False

    def start(self) -> None:
        # 连接的生命周期由 PiperRobot 负责，避免为夹爪重复打开同一 CAN 口。
        self._robot.wait_for_gripper_feedback()
        self._started = True

    def read_position(self) -> float:
        if not self._started:
            raise DeviceError("Piper 夹爪尚未启动。")
        return self._robot.read_gripper_position()

    def stop(self) -> None:
        self._started = False
