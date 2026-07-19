from __future__ import annotations

from pathlib import Path
import threading
import time

import h5py
import yaml

from taiyi_piper_collect.collector import DataCollector
from taiyi_piper_collect.config import load_config
from taiyi_piper_collect.preflight import preflight
from taiyi_piper_collect.quality import validate_hdf5


def _mock_config(tmp_path: Path) -> Path:
    source = Path(__file__).parents[1] / "configs" / "mock_piper.yaml"
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    raw["session"]["output_root"] = str(tmp_path / "records")
    config_path = tmp_path / "mock.yaml"
    config_path.write_text(yaml.safe_dump(raw, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return config_path


def test_mock_preflight_and_collection_create_valid_package(tmp_path: Path) -> None:
    config = load_config(_mock_config(tmp_path))
    report = preflight(config)
    assert report["result"] == "pass"

    result = DataCollector(config).run(duration_s=0.5)

    assert result.trajectory_path.name == "trajectory.hdf5"
    assert result.trajectory_path.exists()
    assert result.quality_path.exists()
    assert result.manifest_path.exists()
    assert result.writer_report.trajectory_length > 0
    assert validate_hdf5(result.trajectory_path, config)["result"] == "pass"
    with h5py.File(result.trajectory_path, "r") as file:
        length = int(file["metadata/trajectory_length"][()])
        assert length == result.writer_report.trajectory_length
        assert file["puppet/arm_single_position_align/data"].shape == (length, 6)
        assert file["puppet/end_effector_single_pose_align/data"].shape == (length, 7)
        assert file["puppet/end_effector_single_position_align/data"].shape == (length, 1)
        assert len(file["camera_observations/color_images/camera_front"]) == length
        assert len(file["camera_observations/depth_images/camera_wrist_right"]) == length


def test_collection_can_finish_after_external_manual_stop(tmp_path: Path) -> None:
    """遥操会话的人工结束信号应正常收尾，而不是遗留 partial 文件。"""

    config = load_config(_mock_config(tmp_path))
    stop_request = threading.Event()
    result_box = {}
    error_box = {}

    def collect() -> None:
        try:
            result_box["result"] = DataCollector(config).run(stop_request=stop_request, until_stopped=True)
        except BaseException as error:
            error_box["error"] = error

    thread = threading.Thread(target=collect)
    thread.start()
    time.sleep(0.4)
    stop_request.set()
    thread.join(timeout=30.0)

    assert not thread.is_alive()
    assert "error" not in error_box
    result = result_box["result"]
    assert result.trajectory_path.exists()
    assert validate_hdf5(result.trajectory_path, config)["result"] == "pass"
