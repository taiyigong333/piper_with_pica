from __future__ import annotations

from pathlib import Path

import pytest

from taiyi_piper_collect.config import load_config
from taiyi_piper_collect.errors import ConfigurationError


def test_mock_config_can_be_loaded() -> None:
    config_path = Path(__file__).parents[1] / "configs" / "mock_piper.yaml"
    config = load_config(config_path)

    assert config.robot.driver == "mock"
    assert config.acquisition.robot_hz == 60
    assert [camera.name for camera in config.enabled_cameras] == ["camera_front", "camera_wrist_right"]


def test_piper_rejects_euler_pose_schema(tmp_path: Path) -> None:
    config_path = tmp_path / "invalid.yaml"
    config_path.write_text(
        """
session:
  output_root: ./records
  data_type: real
  language_instruction: test
  collector_hash: tester
  pose_representation: xyz_rxryrz
modalities:
  rgb: true
  depth: false
  arm_joint_positions: true
  tcp_pose: true
cameras:
  - name: camera_front
    driver: mock
    model: mock
    width: 640
    height: 480
    fps: 30
robot:
  name: piper
  driver: piper
  can_name: can0
""",
        encoding="utf-8",
    )

    with pytest.raises(ConfigurationError, match="xyz_xyzw"):
        load_config(config_path)
