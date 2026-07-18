"""不依赖硬件的确定性模拟设备，用于数据链路回归测试。"""

from __future__ import annotations

import math
import time

import numpy as np

from ..config import CameraConfig, GripperConfig, RobotConfig
from ..models import CameraCalibration, CameraFrame, RobotState
from .base import CameraDevice, GripperDevice, RobotDevice


class MockCamera(CameraDevice):
    def __init__(self, config: CameraConfig) -> None:
        self._config = config
        self._capture_depth = False
        self._frame_index = 0
        self._started = False

    def start(self, capture_depth: bool) -> None:
        self._capture_depth = capture_depth
        self._started = True

    def read(self) -> CameraFrame:
        if not self._started:
            raise RuntimeError(f"模拟相机 {self._config.name} 尚未启动。")
        self._frame_index += 1
        height, width = self._config.height, self._config.width
        x = np.arange(width, dtype=np.uint16)[None, :]
        y = np.arange(height, dtype=np.uint16)[:, None]
        phase = self._frame_index % 256
        # 帧号参与像素值，避免质量检查把模拟图像误判为重复帧。
        color = np.empty((height, width, 3), dtype=np.uint8)
        color[..., 0] = ((x + phase) % 256).astype(np.uint8)
        color[..., 1] = ((y + 2 * phase) % 256).astype(np.uint8)
        color[..., 2] = ((x // 2 + y // 3 + 3 * phase) % 256).astype(np.uint8)
        depth = None
        if self._capture_depth:
            depth = (300 + ((x + y + phase) % 1200)).astype(np.uint16)
        return CameraFrame(color=color, depth=depth)

    def calibration(self) -> CameraCalibration:
        return CameraCalibration(
            matrix=np.array(
                [[600.0, 0.0, self._config.width / 2], [0.0, 600.0, self._config.height / 2], [0.0, 0.0, 1.0]],
                dtype=np.float64,
            ),
            dist_coeffs=np.zeros(5, dtype=np.float64),
        )

    def stop(self) -> None:
        self._started = False

class MockRobot(RobotDevice):
    def __init__(self, config: RobotConfig, pose_representation: str) -> None:
        self._config = config
        self._pose_representation = pose_representation
        self._started = False
        self._started_at = 0.0

    def start(self) -> None:
        self._started = True
        self._started_at = time.monotonic()

    def read(self) -> RobotState:
        if not self._started:
            raise RuntimeError("模拟机器人尚未启动。")
        elapsed = time.monotonic() - self._started_at
        joints = np.asarray([0.25 * math.sin(elapsed + index * 0.2) for index in range(self._config.joint_count)], dtype=np.float64)
        if self._pose_representation == "xyz_xyzw":
            pose = np.asarray([0.35, 0.05 * math.sin(elapsed), 0.25, 0.0, 0.0, 0.0, 1.0], dtype=np.float64)
        else:
            pose = np.asarray([0.35, 0.05 * math.sin(elapsed), 0.25, 0.0, 0.0, 0.0], dtype=np.float64)
        return RobotState(timestamp=time.time(), joint_positions=joints, tcp_pose=pose)

    def stop(self) -> None:
        self._started = False


class MockGripper(GripperDevice):
    def __init__(self, _: GripperConfig) -> None:
        self._started = False
        self._started_at = 0.0

    def start(self) -> None:
        self._started = True
        self._started_at = time.monotonic()

    def read_position(self) -> float:
        if not self._started:
            raise RuntimeError("模拟夹爪尚未启动。")
        return 0.5 + 0.1 * math.sin(time.monotonic() - self._started_at)

    def stop(self) -> None:
        self._started = False
