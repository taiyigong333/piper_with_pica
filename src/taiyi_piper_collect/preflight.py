"""真实硬件发现与只读预检。"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np

from .config import CollectConfig
from .devices import create_camera, create_gripper, create_robot
from .devices.base import CameraDevice, GripperDevice, RobotDevice
from .errors import DeviceError, HardwareDependencyError


def discover_realsense() -> dict[str, Any]:
    """枚举当前可见的 RealSense，不启动数据流。"""

    try:
        import pyrealsense2 as rs
    except ImportError as error:
        raise HardwareDependencyError("缺少 pyrealsense2；请执行 uv sync --extra realsense。") from error
    report: dict[str, Any] = {"checked_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "device_count": 0, "devices": [], "errors": []}
    try:
        context = rs.context()
        for device in context.query_devices():
            def info(key: Any) -> str:
                return device.get_info(key) if device.supports(key) else ""

            report["devices"].append(
                {
                    "name": info(rs.camera_info.name),
                    "serial_number": info(rs.camera_info.serial_number),
                    "firmware_version": info(rs.camera_info.firmware_version),
                    "product_line": info(rs.camera_info.product_line),
                }
            )
        report["device_count"] = len(report["devices"])
    except Exception as error:
        report["errors"].append(f"{type(error).__name__}: {error}")
    return report


def preflight(config: CollectConfig) -> dict[str, Any]:
    """启动设备读取一帧/一条反馈；不创建数据文件，也不下发运动指令。"""

    cameras: dict[str, CameraDevice] = {}
    robot: RobotDevice | None = None
    gripper: GripperDevice | None = None
    report: dict[str, Any] = {
        "checked_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "safe_read_only": True,
        "result": "fail",
        "devices": {"cameras": [], "robot": {}, "gripper": None},
        "errors": [],
        "warnings": [],
    }
    try:
        for camera_config in config.enabled_cameras:
            camera = create_camera(camera_config)
            cameras[camera_config.name] = camera
            camera.start(capture_depth=config.modalities.depth)
            frame = camera.read()
            calibration = camera.calibration()
            if config.modalities.rgb and frame.color is None:
                raise DeviceError(f"{camera_config.name} 未返回 RGB。")
            if config.modalities.depth and frame.depth is None:
                raise DeviceError(f"{camera_config.name} 未返回深度。")
            report["devices"]["cameras"].append(
                {
                    "name": camera_config.name,
                    "driver": camera_config.driver,
                    "model": camera_config.model,
                    "rgb_shape": list(frame.color.shape) if frame.color is not None else None,
                    "depth_shape": list(frame.depth.shape) if frame.depth is not None else None,
                    "intrinsics_available": calibration.matrix is not None,
                    "extrinsics_configured": camera_config.base_to_camera is not None,
                }
            )
            if camera_config.base_to_camera is None:
                report["warnings"].append(f"{camera_config.name} 未配置 base_to_camera 外参。")

        robot = create_robot(config.robot, config.session.pose_representation)
        robot.start()
        state = robot.read()
        _validate_robot_state(state, config)
        report["devices"]["robot"] = {
            "name": config.robot.name,
            "driver": config.robot.driver,
            "joint_shape": list(state.joint_positions.shape) if state.joint_positions is not None else None,
            "tcp_shape": list(state.tcp_pose.shape) if state.tcp_pose is not None else None,
            "pose_representation": config.session.pose_representation,
        }
        if config.robot.base_to_robot is None:
            report["warnings"].append("未配置 base_to_robot 变换。")

        if config.modalities.gripper_position:
            gripper = create_gripper(config.gripper, robot)
            if gripper is None:
                raise DeviceError("已启用夹爪位置但未创建夹爪设备。")
            gripper.start()
            position = float(gripper.read_position())
            if not np.isfinite(position):
                raise DeviceError("夹爪返回非有限位置。")
            report["devices"]["gripper"] = {"driver": config.gripper.driver, "position": position}
        report["result"] = "pass"
    except Exception as error:
        report["errors"].append(f"{type(error).__name__}: {error}")
    finally:
        if gripper is not None:
            gripper.stop()
        if robot is not None:
            robot.stop()
        for camera in cameras.values():
            camera.stop()
    return report


def _validate_robot_state(state: Any, config: CollectConfig) -> None:
    if config.modalities.arm_joint_positions:
        if state.joint_positions is None or state.joint_positions.shape != (config.robot.joint_count,):
            raise DeviceError(f"关节状态形状应为 ({config.robot.joint_count},)。")
        if not np.isfinite(state.joint_positions).all():
            raise DeviceError("关节状态存在 NaN 或 Inf。")
    if config.modalities.tcp_pose:
        width = 7 if config.session.pose_representation == "xyz_xyzw" else 6
        if state.tcp_pose is None or state.tcp_pose.shape != (width,):
            raise DeviceError(f"TCP 位姿形状应为 ({width},)。")
        if not np.isfinite(state.tcp_pose).all():
            raise DeviceError("TCP 位姿存在 NaN 或 Inf。")
