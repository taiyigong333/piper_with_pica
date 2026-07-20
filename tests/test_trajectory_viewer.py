from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from taiyi_piper_collect.trajectory_viewer import TrajectoryViewer, _PAGE


def _write_trajectory(root: Path) -> Path:
    trajectory = root / "real" / "20260720" / "demo_0001" / "trajectory.hdf5"
    trajectory.parent.mkdir(parents=True)
    with h5py.File(trajectory, "w") as file:
        metadata = file.create_group("metadata")
        metadata.create_dataset("language_instruction", data="move corn", dtype=h5py.string_dtype())
        metadata.create_dataset("collection_time", data="2026-07-20 12:00:00", dtype=h5py.string_dtype())
        metadata.create_dataset("pose_representation", data="xyz_rxryrz", dtype=h5py.string_dtype())
        observations = file.create_group("camera_observations")
        observations.create_dataset("timestamp", data=np.array([10.0, 10.1], dtype=np.float64))
        observations.create_dataset("is_intervene", data=np.array([False, True], dtype=np.bool_))
        images = observations.create_group("color_images")
        encoded = images.create_dataset("camera_front", shape=(2,), dtype=h5py.vlen_dtype(np.dtype("uint8")))
        encoded[0] = np.frombuffer(b"first-jpeg", dtype=np.uint8)
        encoded[1] = np.frombuffer(b"second-jpeg", dtype=np.uint8)
        puppet = file.create_group("puppet")
        for name, values in {
            "arm_single_position_align": np.arange(12, dtype=np.float64).reshape(2, 6),
            "end_effector_single_pose_align": np.arange(12, dtype=np.float64).reshape(2, 6) / 10,
            "end_effector_single_position_align": np.array([[0.01], [0.02]], dtype=np.float64),
        }.items():
            group = puppet.create_group(name)
            group.create_dataset("timestamp", data=np.array([10.0, 10.1], dtype=np.float64))
            group.create_dataset("data", data=values)
    trajectory.with_name("quality.json").write_text(
        json.dumps({"result": "pass", "errors": [], "warnings": [], "metrics": {"camera_frames": 2}}),
        encoding="utf-8",
    )
    return trajectory


def test_viewer_lists_and_reads_completed_trajectory(tmp_path: Path) -> None:
    trajectory = _write_trajectory(tmp_path)
    viewer = TrajectoryViewer(tmp_path)

    summaries = viewer.list_trajectories()
    payload = viewer.read_trajectory(trajectory.relative_to(tmp_path).as_posix())

    assert summaries[0]["frame_count"] == 2
    assert summaries[0]["quality_result"] == "pass"
    assert payload["cameras"] == ["camera_front"]
    assert payload["is_intervene"] == [False, True]
    assert payload["joint_positions"]["values"][1] == [6.0, 7.0, 8.0, 9.0, 10.0, 11.0]
    assert payload["tcp_pose"]["width"] == 6
    assert payload["gripper_position"]["values"] == [[0.01], [0.02]]
    assert payload["quality"]["result"] == "pass"
    assert viewer.read_frame(payload["path"], "camera_front", 1) == b"second-jpeg"


def test_viewer_rejects_paths_outside_data_root(tmp_path: Path) -> None:
    viewer = TrajectoryViewer(tmp_path)

    with pytest.raises(ValueError, match="数据根目录"):
        viewer.read_trajectory("../../outside/trajectory.hdf5")


def test_viewer_page_exposes_labeled_detailed_joint_and_tcp_charts() -> None:
    assert 'id="trajectory-previous"' in _PAGE
    assert 'id="trajectory-next"' in _PAGE
    assert 'id="joint-detail-charts"' in _PAGE
    assert 'id="joint-combined-detail-chart"' in _PAGE
    assert "关节角详细时序" in _PAGE
    assert 'id="tcp-position-chart"' in _PAGE
    assert 'id="tcp-orientation-chart"' in _PAGE
    assert "TCP 位置 xyz (m)" in _PAGE
    assert "renderLegend" in _PAGE
