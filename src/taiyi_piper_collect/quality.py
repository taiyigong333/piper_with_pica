"""HDF5 自动质检与包级留存文件。"""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import h5py
import numpy as np

from .config import CollectConfig

if TYPE_CHECKING:
    from .collector import CollectionStats


def _text(value: Any) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def _is_monotonic(timestamps: np.ndarray) -> bool:
    return bool(len(timestamps) <= 1 or np.all(np.diff(timestamps) >= 0))


def _decode_image(data: np.ndarray, is_depth: bool) -> tuple[str | None, np.ndarray | None]:
    try:
        import cv2
    except ImportError:
        return "无法解码图像：当前环境未安装 OpenCV。", None
    image = cv2.imdecode(np.asarray(data, dtype=np.uint8), cv2.IMREAD_UNCHANGED if is_depth else cv2.IMREAD_COLOR)
    return (None, image) if image is not None else ("图像解码失败。", None)


def _sample_indices(length: int) -> list[int]:
    return sorted(index for index in {0, length // 2, length - 1} if 0 <= index < length)


def validate_hdf5(path: str | Path, config: CollectConfig | None = None) -> dict[str, Any]:
    """只读校验结构、时间轴、数据形状、图像可解码性和重复帧比例。"""

    file_path = Path(path)
    errors: list[str] = []
    warnings: list[str] = []
    metrics: dict[str, Any] = {}
    required_metadata = {
        "language_instruction",
        "data_type",
        "data_format_version",
        "collection_time",
        "collector",
        "trajectory_length",
        "pose_representation",
    }
    try:
        with h5py.File(file_path, "r") as file:
            if "metadata" not in file:
                errors.append("缺少 /metadata。")
                return _report(errors, warnings, metrics)
            metadata = file["metadata"]
            missing = sorted(required_metadata - set(metadata.keys()))
            if missing:
                errors.append(f"metadata 缺少字段：{', '.join(missing)}。")
            data_type = _text(metadata["data_type"][()]) if "data_type" in metadata else ""
            if data_type not in {"real", "sim", "synthetic"}:
                errors.append(f"data_type 非法：{data_type!r}。")
            if data_type == "sim" and "sim_assets" not in metadata:
                errors.append("仿真数据缺少 sim_assets。")
            pose_representation = _text(metadata["pose_representation"][()]) if "pose_representation" in metadata else ""
            if pose_representation not in {"xyz_xyzw", "xyz_rxryrz"}:
                errors.append(f"pose_representation 非法：{pose_representation!r}。")

            if "camera_observations" not in file:
                errors.append("缺少 /camera_observations。")
                return _report(errors, warnings, metrics)
            observations = file["camera_observations"]
            if not {"timestamp", "is_intervene"}.issubset(observations.keys()):
                errors.append("camera_observations 缺少 timestamp 或 is_intervene。")
                return _report(errors, warnings, metrics)
            timestamps = observations["timestamp"][:]
            trajectory_length = int(metadata["trajectory_length"][()]) if "trajectory_length" in metadata else -1
            metrics["trajectory_length"] = trajectory_length
            metrics["camera_frames"] = int(len(timestamps))
            if trajectory_length != len(timestamps):
                errors.append("trajectory_length 与 camera_observations/timestamp 长度不一致。")
            if len(observations["is_intervene"]) != len(timestamps):
                errors.append("camera_observations/is_intervene 长度与时间轴不一致。")
            if not _is_monotonic(timestamps):
                errors.append("相机时间戳不是单调递增。")
            if len(timestamps) == 0:
                errors.append("轨迹没有相机帧。")

            _validate_camera_data(file, observations, timestamps, config, errors)
            _validate_robot_data(file, trajectory_length, pose_representation, config, errors)
            _check_repetition(observations, config, warnings, metrics)
            _check_static_configuration(file, config, warnings)
    except OSError as error:
        errors.append(f"HDF5 无法打开：{error}")
    return _report(errors, warnings, metrics)


def _report(errors: list[str], warnings: list[str], metrics: dict[str, Any]) -> dict[str, Any]:
    return {"result": "pass" if not errors else "fail", "errors": errors, "warnings": warnings, "metrics": metrics}


def _validate_camera_data(
    file: h5py.File,
    observations: h5py.Group,
    timestamps: np.ndarray,
    config: CollectConfig | None,
    errors: list[str],
) -> None:
    if config is None:
        return
    for modality, group_name, is_depth in (("rgb", "color_images", False), ("depth", "depth_images", True)):
        if not getattr(config.modalities, modality):
            continue
        if group_name not in observations:
            errors.append(f"配置启用了 {modality}，但 HDF5 缺少 {group_name}。")
            continue
        group = observations[group_name]
        for camera in config.enabled_cameras:
            if camera.name not in group:
                errors.append(f"缺少 {camera.name} 的 {modality} 数据集。")
                continue
            images = group[camera.name]
            if len(images) != len(timestamps):
                errors.append(f"{camera.name} {modality} 长度与相机时间轴不一致。")
                continue
            for index in _sample_indices(len(images)):
                decode_error, image = _decode_image(images[index], is_depth=is_depth)
                if decode_error:
                    errors.append(f"{camera.name} 第 {index} 帧：{decode_error}")
                    break
                if is_depth and image is not None and image.dtype != np.uint16:
                    errors.append(f"{camera.name} 深度第 {index} 帧解码后不是 uint16。")
                    break


def _validate_robot_data(
    file: h5py.File,
    trajectory_length: int,
    pose_representation: str,
    config: CollectConfig | None,
    errors: list[str],
) -> None:
    if "puppet" not in file:
        errors.append("缺少 /puppet。")
        return
    if config is None:
        expected = [("arm_single_position", 6), ("end_effector_single_pose", 7 if pose_representation == "xyz_xyzw" else 6)]
    else:
        expected: list[tuple[str, int]] = []
        if config.modalities.arm_joint_positions:
            expected.append(("arm_single_position", config.robot.joint_count))
        if config.modalities.tcp_pose:
            expected.append(("end_effector_single_pose", 7 if pose_representation == "xyz_xyzw" else 6))
        if config.modalities.gripper_position:
            expected.append(("end_effector_single_position", 1))
    puppet = file["puppet"]
    for prefix, width in expected:
        for suffix in ("raw", "align"):
            name = f"{prefix}_{suffix}"
            if name not in puppet:
                errors.append(f"缺少 /puppet/{name}。")
                continue
            group = puppet[name]
            if not {"timestamp", "is_intervene", "data"}.issubset(group.keys()):
                errors.append(f"/puppet/{name} 缺少时序字段。")
                continue
            time_values = group["timestamp"][:]
            data = group["data"]
            if len(time_values) != len(group["is_intervene"]) or len(time_values) != len(data):
                errors.append(f"/puppet/{name} 时序字段长度不一致。")
            if suffix == "align" and len(time_values) != trajectory_length:
                errors.append(f"/puppet/{name} 没有与相机时间轴一一对齐。")
            if data.ndim != 2 or data.shape[1] != width:
                errors.append(f"/puppet/{name}/data 形状应为 (m, {width})。")
            if not _is_monotonic(time_values):
                errors.append(f"/puppet/{name} 时间戳不是单调递增。")
            if len(data) and not np.isfinite(data[:]).all():
                errors.append(f"/puppet/{name} 存在 NaN 或 Inf。")


def _check_repetition(
    observations: h5py.Group, config: CollectConfig | None, warnings: list[str], metrics: dict[str, Any]
) -> None:
    if config is None or not config.modalities.rgb or "color_images" not in observations:
        return
    ratios: dict[str, float] = {}
    for camera in config.enabled_cameras:
        images = observations["color_images"][camera.name]
        if len(images) <= 1:
            ratio = 0.0
        else:
            repeat_count = sum(bytes(images[index]) == bytes(images[index - 1]) for index in range(1, len(images)))
            ratio = repeat_count / (len(images) - 1)
        ratios[camera.name] = ratio
        if ratio > 0.05:
            warnings.append(f"{camera.name} 相邻 RGB 重复帧比例 {ratio:.2%}，高于 5%。")
    metrics["camera_repetition_ratio"] = ratios


def _check_static_configuration(file: h5py.File, config: CollectConfig | None, warnings: list[str]) -> None:
    if config is None:
        return
    missing_extrinsics = [camera.name for camera in config.enabled_cameras if camera.base_to_camera is None]
    if missing_extrinsics:
        warnings.append(f"未配置相机外参：{', '.join(missing_extrinsics)}；无法直接进行跨坐标系变换。")
    missing_intrinsics = [camera.name for camera in config.enabled_cameras if camera.name not in file["camera_intrinsics"]]
    if missing_intrinsics:
        warnings.append(f"HDF5 未写入运行时相机内参：{', '.join(missing_intrinsics)}。")
    if config.robot.base_to_robot is None:
        warnings.append("未配置 base_to_robot；无法直接进行采集参考 base 到机器人基座的转换。")


def write_quality_report(path: str | Path, config: CollectConfig, stats: CollectionStats) -> Path:
    trajectory_path = Path(path)
    report = validate_hdf5(trajectory_path, config)
    report.update(
        {
            "quality_version": "1.0.0",
            "checked_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "checker": "system",
            "collection_stats": asdict(stats),
            "manual_review": {"status": "pending", "file": "manual_quality.json"},
        }
    )
    output_path = trajectory_path.with_name("quality.json")
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return output_path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def create_manifest(trajectory_path: str | Path, quality_path: str | Path, trajectory_id: str, config: CollectConfig) -> Path:
    """生成不包含自身校验和的 manifest 与交付文件校验表。"""

    trajectory = Path(trajectory_path)
    quality = Path(quality_path)
    files = [
        {"name": trajectory.name, "sha256": _sha256(trajectory), "kind": "hdf5"},
        {"name": quality.name, "sha256": _sha256(quality), "kind": "quality"},
    ]
    quality_result = json.loads(quality.read_text(encoding="utf-8"))["result"]
    manifest = {
        "trajectory_id": trajectory_id,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "data_format_version": config.session.format_version,
        "collection_status": "completed",
        "quality_status": quality_result,
        "devices": {
            "robot": {"name": config.robot.name, "driver": config.robot.driver},
            "cameras": [{"name": camera.name, "model": camera.model, "driver": camera.driver} for camera in config.enabled_cameras],
        },
        "files": files,
    }
    output_path = trajectory.with_name("manifest.json")
    output_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    checksums = "".join(f"{item['sha256']}  {item['name']}\n" for item in files)
    trajectory.with_name("checksums.sha256").write_text(checksums, encoding="utf-8")
    return output_path
