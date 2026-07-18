"""采集协调器：固定周期读取设备，并将事件交给独立 HDF5 写进程。"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime
import multiprocessing as mp
from pathlib import Path
from queue import Empty, Full
import threading
import time
from typing import Any, Callable

import numpy as np

from .config import CollectConfig
from .devices import create_camera, create_gripper, create_robot
from .devices.base import CameraDevice, GripperDevice, RobotDevice
from .encoding import encode_frame
from .errors import CollectionError
from .models import ObservationBundle, RobotState
from .quality import create_manifest, write_quality_report
from .writer import WriterReport, WriterSchema, writer_worker


@dataclass
class CollectionStats:
    raw_robot_states: int = 0
    camera_bundles: int = 0
    skipped_camera_bundles_without_robot_state: int = 0
    skipped_camera_bundles_stale_robot_state: int = 0
    queue_timeouts: int = 0
    robot_read_errors: int = 0
    camera_read_errors: int = 0


@dataclass(frozen=True)
class CollectionResult:
    trajectory_id: str
    trajectory_path: Path
    quality_path: Path
    manifest_path: Path
    stats: CollectionStats
    writer_report: WriterReport


class LatestRobotState:
    """受锁保护的状态快照，避免对齐线程拿到设备适配器可变引用。"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._value: RobotState | None = None

    def update(self, state: RobotState) -> None:
        with self._lock:
            self._value = state

    def get(self) -> RobotState | None:
        with self._lock:
            if self._value is None:
                return None
            value = self._value
            return RobotState(
                timestamp=value.timestamp,
                joint_positions=value.joint_positions.copy() if value.joint_positions is not None else None,
                tcp_pose=value.tcp_pose.copy() if value.tcp_pose is not None else None,
                gripper_position=value.gripper_position.copy() if value.gripper_position is not None else None,
                is_intervene=value.is_intervene,
            )


class PeriodicWorker(threading.Thread):
    """固定周期线程。任何异常都停止整条轨迹，防止静默不完整数据。"""

    def __init__(self, name: str, hz: float, stop_event: threading.Event, callback: Callable[[], None]) -> None:
        super().__init__(name=name, daemon=True)
        self._period_s = 1.0 / hz
        self._stop_event = stop_event
        self._callback = callback
        self.error: BaseException | None = None

    def run(self) -> None:
        next_tick = time.perf_counter()
        while not self._stop_event.is_set():
            try:
                self._callback()
            except BaseException as error:
                self.error = error
                self._stop_event.set()
                return
            next_tick += self._period_s
            wait_s = next_tick - time.perf_counter()
            if wait_s < 0:
                # 避免积压的旧采样在恢复后占满队列，落后时直接从当前时刻重新计时。
                next_tick = time.perf_counter()
                continue
            self._stop_event.wait(wait_s)


class DataCollector:
    def __init__(self, config: CollectConfig) -> None:
        self.config = config

    def run(self, duration_s: float | None = None) -> CollectionResult:
        """执行一次有限时长采集，成功时生成可交付的数据包。"""

        duration = duration_s if duration_s is not None else self.config.session.duration_s
        if duration is None:
            raise CollectionError("未设置采集时长；请提供 --duration 或 session.duration_s。")
        if duration <= 0:
            raise CollectionError("采集时长必须为正数。")
        trajectory_id = self.config.session.trajectory_id or self._make_trajectory_id()
        trajectory_dir = (
            self.config.session.output_root
            / self.config.session.data_type
            / datetime.now().strftime("%Y%m%d")
            / trajectory_id
        )
        trajectory_path = trajectory_dir / "trajectory.hdf5"
        stop_event = threading.Event()
        latest_state = LatestRobotState()
        stats = CollectionStats()
        stats_lock = threading.Lock()
        # 写入进程必须不继承已经打开的 RealSense/CAN 句柄；spawn 在各平台上语义一致。
        mp_context = mp.get_context("spawn")
        data_queue: Any = mp_context.Queue(maxsize=self.config.session.hdf5.queue_size)
        result_queue: Any = mp_context.Queue(maxsize=1)
        cameras: dict[str, CameraDevice] = {}
        robot: RobotDevice | None = None
        gripper: GripperDevice | None = None
        workers: list[PeriodicWorker] = []
        writer_process: mp.Process | None = None
        writer_report: WriterReport | None = None
        primary_error: BaseException | None = None

        def increment(attribute: str) -> None:
            with stats_lock:
                setattr(stats, attribute, getattr(stats, attribute) + 1)

        def enqueue(event: tuple[str, Any] | str) -> None:
            try:
                data_queue.put(event, timeout=self.config.session.hdf5.queue_put_timeout_s)
            except Full as error:
                increment("queue_timeouts")
                raise CollectionError("HDF5 队列已满，已停止采集以避免静默丢数。") from error

        try:
            cameras = {camera.name: create_camera(camera) for camera in self.config.enabled_cameras}
            robot = create_robot(self.config.robot, self.config.session.pose_representation)
            gripper = create_gripper(self.config.gripper) if self.config.modalities.gripper_position else None
            for camera in cameras.values():
                camera.start(capture_depth=self.config.modalities.depth)
            robot.start()
            if gripper is not None:
                gripper.start()

            schema = WriterSchema(
                config=self.config,
                trajectory_id=trajectory_id,
                output_path=trajectory_path,
                camera_calibrations={name: camera.calibration() for name, camera in cameras.items()},
            )
            writer_process = mp_context.Process(target=writer_worker, args=(schema, data_queue, result_queue), name="hdf5-writer")
            writer_process.start()

            def sample_robot() -> None:
                assert robot is not None
                try:
                    state = robot.read()
                    if self.config.modalities.gripper_position:
                        if gripper is None:
                            raise CollectionError("夹爪位置模态已启用，但没有可用夹爪设备。")
                        position = np.asarray([gripper.read_position()], dtype=np.float64)
                        state = replace(state, gripper_position=position)
                    latest_state.update(state)
                    enqueue(("raw_robot_state", state))
                    increment("raw_robot_states")
                except Exception:
                    increment("robot_read_errors")
                    raise

            def sample_camera_rig() -> None:
                try:
                    frames = {
                        name: encode_frame(camera.read(), self.config.session.hdf5.jpeg_quality, self.config.modalities.depth)
                        for name, camera in cameras.items()
                    }
                    # 组帧全部读取完成后才取时间戳，表示实际对外发布这个对齐快照的时刻。
                    bundle_timestamp = time.time()
                    state = latest_state.get()
                    if state is None:
                        increment("skipped_camera_bundles_without_robot_state")
                        return
                    age_ms = (bundle_timestamp - state.timestamp) * 1000.0
                    if age_ms < 0 or age_ms > self.config.acquisition.max_alignment_age_ms:
                        increment("skipped_camera_bundles_stale_robot_state")
                        return
                    enqueue(("observation_bundle", ObservationBundle(bundle_timestamp, frames, state)))
                    increment("camera_bundles")
                except Exception:
                    increment("camera_read_errors")
                    raise

            robot_worker = PeriodicWorker("robot-state-reader", self.config.acquisition.robot_hz, stop_event, sample_robot)
            camera_worker = PeriodicWorker("camera-rig-reader", self.config.acquisition.camera_rig_hz, stop_event, sample_camera_rig)
            workers = [robot_worker, camera_worker]
            robot_worker.start()
            warmup_deadline = time.monotonic() + min(2.0, 2.0 / self.config.acquisition.robot_hz + 0.5)
            while latest_state.get() is None and not stop_event.is_set() and time.monotonic() < warmup_deadline:
                time.sleep(0.005)
            if latest_state.get() is None:
                raise CollectionError("机器人反馈预热超时，未开始相机采集。")
            camera_worker.start()
            deadline = time.monotonic() + duration
            while not stop_event.is_set() and time.monotonic() < deadline:
                time.sleep(min(0.05, max(0.0, deadline - time.monotonic())))
            for worker in workers:
                if worker.error is not None:
                    raise CollectionError(f"{worker.name} 失败：{type(worker.error).__name__}: {worker.error}")
            if stop_event.is_set() and time.monotonic() < deadline:
                raise CollectionError("采集在达到目标时长前被停止。")
        except BaseException as error:
            primary_error = error
        finally:
            stop_event.set()
            for worker in workers:
                worker.join(timeout=2.0)
            if gripper is not None:
                try:
                    gripper.stop()
                except Exception:
                    pass
            if robot is not None:
                try:
                    robot.stop()
                except Exception:
                    pass
            for camera in cameras.values():
                try:
                    camera.stop()
                except Exception:
                    pass

            if writer_process is not None:
                if primary_error is None:
                    try:
                        enqueue("__STOP__")
                        status, payload = result_queue.get(timeout=20.0)
                        writer_process.join(timeout=20.0)
                        if writer_process.is_alive():
                            writer_process.terminate()
                            writer_process.join(timeout=2.0)
                            raise CollectionError("HDF5 写入进程未能正常退出。")
                        if status != "ok":
                            raise CollectionError(f"HDF5 写入失败：{payload}")
                        writer_report = payload
                    except BaseException as error:
                        primary_error = error
                else:
                    # 发生采集错误时不发布正式 HDF5；若写进程已建文件，保留 .partial 便于排障。
                    # start() 本身失败时进程对象尚无 _popen，不能再调用 terminate() 覆盖原始错误。
                    if writer_process.pid is not None:
                        writer_process.terminate()
                        writer_process.join(timeout=2.0)
            data_queue.close()
            result_queue.close()

        if primary_error is not None:
            if isinstance(primary_error, CollectionError):
                raise primary_error
            raise CollectionError(f"采集失败：{type(primary_error).__name__}: {primary_error}") from primary_error
        if writer_report is None:
            raise CollectionError("HDF5 写入未返回完成报告。")
        if writer_report.trajectory_length == 0:
            raise CollectionError("采集未写入任何相机组帧，拒绝生成空轨迹。")
        quality_path = write_quality_report(trajectory_path, self.config, stats)
        manifest_path = create_manifest(trajectory_path, quality_path, trajectory_id, self.config)
        return CollectionResult(trajectory_id, trajectory_path, quality_path, manifest_path, stats, writer_report)

    def _make_trajectory_id(self) -> str:
        timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
        return f"{self.config.session.data_type}_{self.config.robot.name}_{timestamp}_0001"
