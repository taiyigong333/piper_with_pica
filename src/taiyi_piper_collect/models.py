"""在采集线程和写入进程之间传递的稳定数据模型。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CameraCalibration:
    """相机运行时内参；外参由配置提供并随轨迹写入。"""

    matrix: np.ndarray | None = None
    dist_coeffs: np.ndarray | None = None


@dataclass(frozen=True)
class CameraFrame:
    """设备返回的原始图像，颜色通道顺序由相机配置声明。"""

    color: np.ndarray | None
    depth: np.ndarray | None


@dataclass(frozen=True)
class EncodedCameraFrame:
    """便于跨进程传输的已编码图像。"""

    color: bytes | None
    depth: bytes | None


@dataclass(frozen=True)
class RobotState:
    """机器人真实反馈；所有数组使用标准单位，绝不存控制目标。"""

    timestamp: float
    joint_positions: np.ndarray | None = None
    tcp_pose: np.ndarray | None = None
    gripper_position: np.ndarray | None = None
    is_intervene: bool = False


@dataclass(frozen=True)
class ObservationBundle:
    """一次多相机组帧和在该时刻可用的机器人状态快照。"""

    timestamp: float
    frames: dict[str, EncodedCameraFrame]
    aligned_robot_state: RobotState
