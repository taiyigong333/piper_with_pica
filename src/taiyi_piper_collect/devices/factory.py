"""设备注册表；新增硬件只影响这里及对应适配器。"""

from __future__ import annotations

from ..config import CameraConfig, GripperConfig, RobotConfig
from ..errors import ConfigurationError
from .base import CameraDevice, GripperDevice, RobotDevice
from .mock import MockCamera, MockGripper, MockRobot
from .piper import PiperRobot
from .realsense import RealSenseCamera


def create_camera(config: CameraConfig) -> CameraDevice:
    if config.driver == "realsense":
        return RealSenseCamera(config)
    if config.driver == "mock":
        return MockCamera(config)
    raise ConfigurationError(f"不支持的相机驱动：{config.driver}（{config.name}）。")


def create_robot(config: RobotConfig, pose_representation: str) -> RobotDevice:
    if config.driver == "piper":
        return PiperRobot(config, pose_representation)
    if config.driver == "mock":
        return MockRobot(config, pose_representation)
    raise ConfigurationError(f"不支持的机器人驱动：{config.driver}。")


def create_gripper(config: GripperConfig) -> GripperDevice | None:
    if not config.enabled:
        return None
    if config.driver == "mock":
        return MockGripper(config)
    raise ConfigurationError(f"不支持的夹爪驱动：{config.driver}。")
