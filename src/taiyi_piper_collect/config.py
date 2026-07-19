"""YAML 配置加载与启动前校验。

配置是采集行为的唯一入口：频率、模态、硬件和坐标变换不应散落在运行代码中。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml

from .errors import ConfigurationError

PoseRepresentation = Literal["xyz_xyzw", "xyz_rxryrz"]


@dataclass(frozen=True)
class ModalityConfig:
    rgb: bool = True
    depth: bool = False
    arm_joint_positions: bool = True
    tcp_pose: bool = True
    gripper_position: bool = False


@dataclass(frozen=True)
class Hdf5Config:
    queue_size: int = 256
    batch_size: int = 16
    flush_every_batches: int = 10
    queue_put_timeout_s: float = 2.0
    jpeg_quality: int = 95


@dataclass(frozen=True)
class SessionConfig:
    output_root: Path
    data_type: Literal["real", "sim", "synthetic"]
    language_instruction: str
    collector_hash: str
    format_version: str = "1.0.0"
    trajectory_id: str | None = None
    duration_s: float | None = None
    sim_assets: str | None = None
    pose_representation: PoseRepresentation = "xyz_xyzw"
    hdf5: Hdf5Config = field(default_factory=Hdf5Config)


@dataclass(frozen=True)
class AcquisitionConfig:
    robot_hz: float = 60.0
    camera_rig_hz: float = 30.0
    max_alignment_age_ms: float = 100.0


@dataclass(frozen=True)
class CameraConfig:
    name: str
    driver: str
    model: str
    width: int
    height: int
    fps: float
    serial_number: str | None = None
    color_order: Literal["rgb", "bgr"] = "bgr"
    enabled: bool = True
    align_depth_to_color: bool = True
    depth_width: int | None = None
    depth_height: int | None = None
    base_to_camera: tuple[tuple[float, ...], ...] | None = None


@dataclass(frozen=True)
class RobotConfig:
    name: str
    driver: str
    joint_count: int = 6
    can_name: str | None = None
    dh_is_offset: int = 1
    tool_offset_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    base_to_robot: tuple[tuple[float, ...], ...] | None = None


@dataclass(frozen=True)
class GripperConfig:
    enabled: bool = False
    driver: str = "none"


@dataclass(frozen=True)
class CollectConfig:
    session: SessionConfig
    modalities: ModalityConfig
    acquisition: AcquisitionConfig
    cameras: tuple[CameraConfig, ...]
    robot: RobotConfig
    gripper: GripperConfig
    source_path: Path

    @property
    def enabled_cameras(self) -> tuple[CameraConfig, ...]:
        return tuple(camera for camera in self.cameras if camera.enabled)


def _mapping(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ConfigurationError(f"{path} 必须是对象。")
    return value


def _required(mapping: dict[str, Any], key: str, path: str) -> Any:
    if key not in mapping:
        raise ConfigurationError(f"缺少必填配置：{path}.{key}")
    return mapping[key]


def _positive_number(value: Any, path: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ConfigurationError(f"{path} 必须为正数。") from error
    if number <= 0:
        raise ConfigurationError(f"{path} 必须为正数。")
    return number


def _matrix(value: Any, path: str) -> tuple[tuple[float, ...], ...] | None:
    if value is None:
        return None
    if not isinstance(value, list) or len(value) != 4:
        raise ConfigurationError(f"{path} 必须为 4×4 数组或 null。")
    try:
        rows = tuple(tuple(float(item) for item in row) for row in value)
    except (TypeError, ValueError) as error:
        raise ConfigurationError(f"{path} 必须为数值 4×4 数组。") from error
    if any(len(row) != 4 for row in rows):
        raise ConfigurationError(f"{path} 必须为 4×4 数组。")
    return rows


def _camera(value: Any, index: int) -> CameraConfig:
    path = f"cameras[{index}]"
    raw = _mapping(value, path)
    name = str(_required(raw, "name", path))
    if not name.startswith("camera_"):
        raise ConfigurationError(f"{path}.name 必须以 camera_ 开头。")
    color_order = str(raw.get("color_order", "bgr"))
    if color_order not in {"rgb", "bgr"}:
        raise ConfigurationError(f"{path}.color_order 只支持 rgb 或 bgr。")
    return CameraConfig(
        name=name,
        driver=str(_required(raw, "driver", path)),
        model=str(_required(raw, "model", path)),
        width=int(_positive_number(_required(raw, "width", path), f"{path}.width")),
        height=int(_positive_number(_required(raw, "height", path), f"{path}.height")),
        fps=_positive_number(_required(raw, "fps", path), f"{path}.fps"),
        serial_number=str(raw["serial_number"]) if raw.get("serial_number") else None,
        color_order=color_order,  # type: ignore[arg-type]
        enabled=bool(raw.get("enabled", True)),
        align_depth_to_color=bool(raw.get("align_depth_to_color", True)),
        depth_width=int(raw["depth_width"]) if raw.get("depth_width") is not None else None,
        depth_height=int(raw["depth_height"]) if raw.get("depth_height") is not None else None,
        base_to_camera=_matrix(raw.get("base_to_camera"), f"{path}.base_to_camera"),
    )


def load_config(path: str | Path) -> CollectConfig:
    """读取 YAML，并在接触真实设备前拒绝不完整或不安全配置。"""

    source_path = Path(path).expanduser().resolve()
    try:
        with source_path.open("r", encoding="utf-8") as file:
            raw = _mapping(yaml.safe_load(file), "根配置")
    except OSError as error:
        raise ConfigurationError(f"无法读取配置 {source_path}：{error}") from error

    session_raw = _mapping(_required(raw, "session", "根配置"), "session")
    hdf5_raw = _mapping(session_raw.get("hdf5", {}), "session.hdf5")
    data_type = str(_required(session_raw, "data_type", "session"))
    if data_type not in {"real", "sim", "synthetic"}:
        raise ConfigurationError("session.data_type 只支持 real、sim 或 synthetic。")
    pose_representation = str(session_raw.get("pose_representation", "xyz_xyzw"))
    if pose_representation not in {"xyz_xyzw", "xyz_rxryrz"}:
        raise ConfigurationError("session.pose_representation 只支持 xyz_xyzw 或 xyz_rxryrz。")
    output_root = Path(str(_required(session_raw, "output_root", "session")))
    if not output_root.is_absolute():
        output_root = (source_path.parent / output_root).resolve()
    session = SessionConfig(
        output_root=output_root,
        data_type=data_type,  # type: ignore[arg-type]
        language_instruction=str(_required(session_raw, "language_instruction", "session")),
        collector_hash=str(_required(session_raw, "collector_hash", "session")),
        format_version=str(session_raw.get("format_version", "1.0.0")),
        trajectory_id=str(session_raw["trajectory_id"]) if session_raw.get("trajectory_id") else None,
        duration_s=_positive_number(session_raw["duration_s"], "session.duration_s") if session_raw.get("duration_s") is not None else None,
        sim_assets=str(session_raw["sim_assets"]) if session_raw.get("sim_assets") else None,
        pose_representation=pose_representation,  # type: ignore[arg-type]
        hdf5=Hdf5Config(
            queue_size=int(_positive_number(hdf5_raw.get("queue_size", 256), "session.hdf5.queue_size")),
            batch_size=int(_positive_number(hdf5_raw.get("batch_size", 16), "session.hdf5.batch_size")),
            flush_every_batches=int(_positive_number(hdf5_raw.get("flush_every_batches", 10), "session.hdf5.flush_every_batches")),
            queue_put_timeout_s=_positive_number(hdf5_raw.get("queue_put_timeout_s", 2.0), "session.hdf5.queue_put_timeout_s"),
            jpeg_quality=int(_positive_number(hdf5_raw.get("jpeg_quality", 95), "session.hdf5.jpeg_quality")),
        ),
    )
    if session.data_type == "sim" and not session.sim_assets:
        raise ConfigurationError("仿真数据必须配置 session.sim_assets。")
    if not 1 <= session.hdf5.jpeg_quality <= 100:
        raise ConfigurationError("session.hdf5.jpeg_quality 必须在 1 到 100 之间。")

    modalities_raw = _mapping(raw.get("modalities", {}), "modalities")
    modalities = ModalityConfig(
        **{key: bool(modalities_raw.get(key, getattr(ModalityConfig(), key))) for key in ModalityConfig.__dataclass_fields__}
    )
    if not modalities.rgb and not modalities.depth:
        raise ConfigurationError("采集系统以相机时间轴对齐，至少需要启用 RGB 或深度模态。")

    acquisition_raw = _mapping(raw.get("acquisition", {}), "acquisition")
    acquisition = AcquisitionConfig(
        robot_hz=_positive_number(acquisition_raw.get("robot_hz", 60.0), "acquisition.robot_hz"),
        camera_rig_hz=_positive_number(acquisition_raw.get("camera_rig_hz", 30.0), "acquisition.camera_rig_hz"),
        max_alignment_age_ms=_positive_number(
            acquisition_raw.get("max_alignment_age_ms", 100.0), "acquisition.max_alignment_age_ms"
        ),
    )

    cameras_raw = raw.get("cameras", [])
    if not isinstance(cameras_raw, list):
        raise ConfigurationError("cameras 必须为列表。")
    cameras = tuple(_camera(value, index) for index, value in enumerate(cameras_raw))
    enabled_cameras = tuple(camera for camera in cameras if camera.enabled)
    if not enabled_cameras:
        raise ConfigurationError("启用图像模态时至少需要一台 enabled 相机。")
    names = [camera.name for camera in cameras]
    if len(names) != len(set(names)):
        raise ConfigurationError("相机名称不能重复。")
    slow = [camera.name for camera in enabled_cameras if camera.fps < acquisition.camera_rig_hz]
    if slow:
        raise ConfigurationError(
            f"相机标称 fps 低于 acquisition.camera_rig_hz：{', '.join(slow)}；会产生不可控重帧。"
        )

    robot_raw = _mapping(_required(raw, "robot", "根配置"), "robot")
    tool_offset_raw = robot_raw.get("tool_offset_m", [0.0, 0.0, 0.0])
    if not isinstance(tool_offset_raw, list) or len(tool_offset_raw) != 3:
        raise ConfigurationError("robot.tool_offset_m 必须为三个数值（m）。")
    try:
        tool_offset = tuple(float(value) for value in tool_offset_raw)
    except (TypeError, ValueError) as error:
        raise ConfigurationError("robot.tool_offset_m 必须为三个数值（m）。") from error
    robot = RobotConfig(
        name=str(_required(robot_raw, "name", "robot")),
        driver=str(_required(robot_raw, "driver", "robot")),
        joint_count=int(_positive_number(robot_raw.get("joint_count", 6), "robot.joint_count")),
        can_name=str(robot_raw["can_name"]) if robot_raw.get("can_name") else None,
        dh_is_offset=int(robot_raw.get("dh_is_offset", 1)),
        tool_offset_m=tool_offset,  # type: ignore[arg-type]
        base_to_robot=_matrix(robot_raw.get("base_to_robot"), "robot.base_to_robot"),
    )
    if robot.driver == "piper":
        if robot.joint_count != 6:
            raise ConfigurationError("Piper 当前必须配置 robot.joint_count=6。")
        if not robot.can_name:
            raise ConfigurationError("Piper 需要配置 robot.can_name。")
        if robot.dh_is_offset not in {0, 1}:
            raise ConfigurationError("robot.dh_is_offset 只支持 0 或 1。")
        if session.pose_representation != "xyz_xyzw":
            raise ConfigurationError("Piper 原生返回欧拉角；为避免语义错误，必须使用 xyz_xyzw。")

    gripper_raw = _mapping(raw.get("gripper", {}), "gripper")
    gripper = GripperConfig(enabled=bool(gripper_raw.get("enabled", False)), driver=str(gripper_raw.get("driver", "none")))
    if modalities.gripper_position and not gripper.enabled:
        raise ConfigurationError("启用 gripper_position 时必须配置 gripper.enabled=true。")
    if gripper.enabled and gripper.driver == "none":
        raise ConfigurationError("启用夹爪时必须指定 gripper.driver。")
    if gripper.driver == "piper" and robot.driver != "piper":
        raise ConfigurationError("gripper.driver=piper 必须与 robot.driver=piper 一起使用。")

    return CollectConfig(session, modalities, acquisition, cameras, robot, gripper, source_path)
