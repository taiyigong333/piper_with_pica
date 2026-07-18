"""异步 HDF5 写入器。

写进程是唯一接触 HDF5 的执行单元，避免磁盘 I/O 反压采集与控制线程。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
import os
from pathlib import Path
from queue import Empty
from typing import Any

import h5py
import numpy as np

from .config import CollectConfig
from .models import CameraCalibration, ObservationBundle, RobotState


@dataclass(frozen=True)
class WriterSchema:
    config: CollectConfig
    trajectory_id: str
    output_path: Path
    camera_calibrations: dict[str, CameraCalibration]


@dataclass(frozen=True)
class WriterReport:
    output_path: Path
    trajectory_length: int
    raw_robot_states: int
    written_bundles: int


def _string_dataset(parent: h5py.Group, name: str, value: str) -> None:
    parent.create_dataset(name, data=value, dtype=h5py.string_dtype(encoding="utf-8"))


def _time_series_group(parent: h5py.Group, name: str, width: int) -> h5py.Group:
    group = parent.create_group(name)
    group.create_dataset("timestamp", shape=(0,), maxshape=(None,), dtype=np.float64, chunks=True)
    group.create_dataset("is_intervene", shape=(0,), maxshape=(None,), dtype=np.bool_, chunks=True)
    group.create_dataset("data", shape=(0, width), maxshape=(None, width), dtype=np.float64, chunks=True)
    return group


def _extend(dataset: h5py.Dataset, value: Any) -> None:
    index = len(dataset)
    dataset.resize(index + 1, axis=0)
    dataset[index] = value


def _append_state(group: h5py.Group, timestamp: float, is_intervene: bool, data: np.ndarray, width: int) -> None:
    array = np.asarray(data, dtype=np.float64)
    if array.shape != (width,):
        raise ValueError(f"{group.name}/data 期望形状 ({width},)，实际为 {array.shape}。")
    if not np.isfinite(array).all():
        raise ValueError(f"{group.name}/data 存在 NaN 或 Inf。")
    _extend(group["timestamp"], float(timestamp))
    _extend(group["is_intervene"], bool(is_intervene))
    _extend(group["data"], array)


class Hdf5TrajectoryWriter:
    """将队列事件写入一个 `.partial` 文件，成功后原子发布。"""

    def __init__(self, schema: WriterSchema) -> None:
        self.schema = schema
        self.partial_path = schema.output_path.with_name(f"{schema.output_path.name}.partial")
        self._file: h5py.File | None = None
        self._camera_length = 0
        self._raw_robot_states = 0
        self._written_bundles = 0

    @property
    def file(self) -> h5py.File:
        if self._file is None:
            raise RuntimeError("HDF5 文件尚未打开。")
        return self._file

    def open(self) -> None:
        self.schema.output_path.parent.mkdir(parents=True, exist_ok=True)
        if self.partial_path.exists():
            raise FileExistsError(f"发现未完成文件，拒绝覆盖：{self.partial_path}")
        if self.schema.output_path.exists():
            raise FileExistsError(f"目标轨迹已存在，拒绝覆盖：{self.schema.output_path}")
        self._file = h5py.File(self.partial_path, "w")
        self._create_static_schema()
        self._create_dynamic_schema()

    def _create_static_schema(self) -> None:
        config = self.schema.config
        session = config.session
        metadata = self.file.create_group("metadata")
        _string_dataset(metadata, "language_instruction", session.language_instruction)
        _string_dataset(metadata, "data_type", session.data_type)
        _string_dataset(metadata, "data_format_version", session.format_version)
        _string_dataset(metadata, "collection_time", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        _string_dataset(metadata, "collector", session.collector_hash)
        _string_dataset(metadata, "pose_representation", session.pose_representation)
        if session.sim_assets:
            _string_dataset(metadata, "sim_assets", session.sim_assets)
        metadata.create_dataset("trajectory_length", data=np.int64(0))

        robot_model = self.file.create_group("robot_model")
        _string_dataset(robot_model, "robot_single", config.robot.name)
        if config.robot.base_to_robot is not None:
            transforms = self.file.create_group("base_to_robot_transformation")
            transforms.create_dataset("base_to_robot_single", data=np.asarray(config.robot.base_to_robot, dtype=np.float64))

        models = self.file.create_group("camera_model")
        color_resolution = self.file.create_group("camera_color_resolution")
        color_channel = self.file.create_group("camera_color_channel")
        depth_resolution = self.file.create_group("camera_depth_resolution")
        intrinsics = self.file.create_group("camera_intrinsics")
        extrinsics = self.file.create_group("camera_extrinsics")
        for camera in config.enabled_cameras:
            _string_dataset(models, camera.name, camera.model)
            color_resolution.create_dataset(camera.name, data=np.asarray([camera.width, camera.height], dtype=np.float64))
            _string_dataset(color_channel, camera.name, camera.color_order)
            if config.modalities.depth:
                depth_resolution.create_dataset(
                    camera.name,
                    data=np.asarray([camera.depth_width or camera.width, camera.depth_height or camera.height], dtype=np.float64),
                )
            calibration = self.schema.camera_calibrations.get(camera.name, CameraCalibration())
            if calibration.matrix is not None:
                camera_intrinsics = intrinsics.create_group(camera.name)
                camera_intrinsics.create_dataset("matrix", data=np.asarray(calibration.matrix, dtype=np.float64))
                if calibration.dist_coeffs is not None:
                    camera_intrinsics.create_dataset("dist_coeffs", data=np.asarray(calibration.dist_coeffs, dtype=np.float64))
            if camera.base_to_camera is not None:
                extrinsics.create_dataset(camera.name, data=np.asarray(camera.base_to_camera, dtype=np.float64))

        acquisition = self.file.create_group("acquisition_config")
        _string_dataset(acquisition, "source_config", config.source_path.name)
        _string_dataset(
            acquisition,
            "enabled_modalities",
            ",".join(name for name, enabled in asdict(config.modalities).items() if enabled),
        )
        _string_dataset(acquisition, "alignment_strategy", "latest_robot_state_with_max_age")
        acquisition.create_dataset("robot_state_hz", data=np.float64(config.acquisition.robot_hz))
        acquisition.create_dataset("camera_rig_hz", data=np.float64(config.acquisition.camera_rig_hz))
        acquisition.create_dataset("max_alignment_age_ms", data=np.float64(config.acquisition.max_alignment_age_ms))
        acquisition.create_dataset("hdf5_batch_size", data=np.int64(session.hdf5.batch_size))

        if config.gripper.enabled:
            end_effector = self.file.create_group("end_effector_model")
            _string_dataset(end_effector, "end_effector_single", config.gripper.driver)

    def _create_dynamic_schema(self) -> None:
        config = self.schema.config
        observations = self.file.create_group("camera_observations")
        observations.create_dataset("timestamp", shape=(0,), maxshape=(None,), dtype=np.float64, chunks=True)
        observations.create_dataset("is_intervene", shape=(0,), maxshape=(None,), dtype=np.bool_, chunks=True)
        vlen_uint8 = h5py.vlen_dtype(np.dtype("uint8"))
        if config.modalities.rgb:
            colors = observations.create_group("color_images")
            for camera in config.enabled_cameras:
                colors.create_dataset(camera.name, shape=(0,), maxshape=(None,), dtype=vlen_uint8, chunks=True)
        if config.modalities.depth:
            depths = observations.create_group("depth_images")
            for camera in config.enabled_cameras:
                depths.create_dataset(camera.name, shape=(0,), maxshape=(None,), dtype=vlen_uint8, chunks=True)

        puppet = self.file.create_group("puppet")
        if config.modalities.arm_joint_positions:
            _time_series_group(puppet, "arm_single_position_raw", config.robot.joint_count)
            _time_series_group(puppet, "arm_single_position_align", config.robot.joint_count)
        if config.modalities.tcp_pose:
            width = 7 if config.session.pose_representation == "xyz_xyzw" else 6
            _time_series_group(puppet, "end_effector_single_pose_raw", width)
            _time_series_group(puppet, "end_effector_single_pose_align", width)
        if config.modalities.gripper_position:
            _time_series_group(puppet, "end_effector_single_position_raw", 1)
            _time_series_group(puppet, "end_effector_single_position_align", 1)

    def append_raw_robot_state(self, state: RobotState) -> None:
        self._append_robot_state(state, "raw")
        self._raw_robot_states += 1

    def append_observation_bundle(self, bundle: ObservationBundle) -> None:
        config = self.schema.config
        observations = self.file["camera_observations"]
        _extend(observations["timestamp"], bundle.timestamp)
        _extend(observations["is_intervene"], bundle.aligned_robot_state.is_intervene)
        for camera in config.enabled_cameras:
            frame = bundle.frames.get(camera.name)
            if frame is None:
                raise ValueError(f"相机组帧缺少 {camera.name}。")
            if config.modalities.rgb:
                if frame.color is None:
                    raise ValueError(f"相机 {camera.name} 缺少 RGB 数据。")
                _extend(observations["color_images"][camera.name], np.frombuffer(frame.color, dtype=np.uint8))
            if config.modalities.depth:
                if frame.depth is None:
                    raise ValueError(f"相机 {camera.name} 缺少深度数据。")
                _extend(observations["depth_images"][camera.name], np.frombuffer(frame.depth, dtype=np.uint8))
        aligned = RobotState(
            timestamp=bundle.timestamp,
            joint_positions=bundle.aligned_robot_state.joint_positions,
            tcp_pose=bundle.aligned_robot_state.tcp_pose,
            gripper_position=bundle.aligned_robot_state.gripper_position,
            is_intervene=bundle.aligned_robot_state.is_intervene,
        )
        self._append_robot_state(aligned, "align")
        self._camera_length += 1
        self._written_bundles += 1

    def _append_robot_state(self, state: RobotState, suffix: str) -> None:
        config = self.schema.config
        puppet = self.file["puppet"]
        if config.modalities.arm_joint_positions:
            if state.joint_positions is None:
                raise ValueError("已启用关节采集，但机器人未提供关节状态。")
            _append_state(
                puppet[f"arm_single_position_{suffix}"],
                state.timestamp,
                state.is_intervene,
                state.joint_positions,
                config.robot.joint_count,
            )
        if config.modalities.tcp_pose:
            if state.tcp_pose is None:
                raise ValueError("已启用 TCP 采集，但机器人未提供 TCP 位姿。")
            width = 7 if config.session.pose_representation == "xyz_xyzw" else 6
            _append_state(puppet[f"end_effector_single_pose_{suffix}"], state.timestamp, state.is_intervene, state.tcp_pose, width)
        if config.modalities.gripper_position:
            if state.gripper_position is None:
                raise ValueError("已启用夹爪采集，但机器人未提供夹爪位置。")
            _append_state(
                puppet[f"end_effector_single_position_{suffix}"], state.timestamp, state.is_intervene, state.gripper_position, 1
            )

    def close_successfully(self) -> WriterReport:
        self.file["metadata/trajectory_length"][()] = np.int64(self._camera_length)
        self.file.flush()
        self.file.close()
        self._file = None
        os.replace(self.partial_path, self.schema.output_path)
        return WriterReport(
            output_path=self.schema.output_path,
            trajectory_length=self._camera_length,
            raw_robot_states=self._raw_robot_states,
            written_bundles=self._written_bundles,
        )

    def close_with_error(self) -> None:
        if self._file is not None:
            self._file.flush()
            self._file.close()
            self._file = None


def writer_worker(schema: WriterSchema, data_queue: Any, result_queue: Any) -> None:
    """写进程入口。异常只保留 `.partial`，绝不伪装成完整 HDF5。"""

    writer = Hdf5TrajectoryWriter(schema)
    try:
        writer.open()
        batches_since_flush = 0
        while True:
            item = data_queue.get()
            if item == "__STOP__":
                result_queue.put(("ok", writer.close_successfully()))
                return
            batch = [item]
            stop_requested = False
            for _ in range(schema.config.session.hdf5.batch_size - 1):
                try:
                    candidate = data_queue.get_nowait()
                except Empty:
                    break
                if candidate == "__STOP__":
                    stop_requested = True
                    break
                batch.append(candidate)
            for event_type, payload in batch:
                if event_type == "raw_robot_state":
                    writer.append_raw_robot_state(payload)
                elif event_type == "observation_bundle":
                    writer.append_observation_bundle(payload)
                else:
                    raise ValueError(f"未知写入事件：{event_type}")
            batches_since_flush += 1
            if batches_since_flush >= schema.config.session.hdf5.flush_every_batches:
                writer.file.flush()
                batches_since_flush = 0
            if stop_requested:
                result_queue.put(("ok", writer.close_successfully()))
                return
    except Exception as error:
        writer.close_with_error()
        result_queue.put(("error", f"{type(error).__name__}: {error}"))
