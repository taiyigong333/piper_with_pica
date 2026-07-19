from __future__ import annotations

from pathlib import Path

from taiyi_piper_collect.config import load_config


def test_mock_config_can_be_loaded() -> None:
    config_path = Path(__file__).parents[1] / "configs" / "mock_piper.yaml"
    config = load_config(config_path)

    assert config.robot.driver == "mock"
    assert config.acquisition.robot_hz == 60
    assert [camera.name for camera in config.enabled_cameras] == ["camera_front", "camera_wrist_right"]


def test_fully_annotated_real_example_can_be_loaded() -> None:
    config_path = Path(__file__).parents[1] / "configs" / "piper_d405_d435.example.yaml"
    config = load_config(config_path)

    assert config.robot.driver == "piper"
    assert config.session.pose_representation == "xyz_rxryrz"
    assert config.modalities.gripper_position
    assert config.gripper.driver == "piper"
    assert not config.robot.initial_pose.enabled
    assert config.robot.initial_pose.mode == "joint"


def test_piper_accepts_native_euler_pose_schema(tmp_path: Path) -> None:
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

    assert load_config(config_path).session.pose_representation == "xyz_rxryrz"


def test_enabled_initial_tcp_pose_requires_native_euler_representation(tmp_path: Path) -> None:
    config_path = tmp_path / "invalid_initial_pose.yaml"
    config_path.write_text(
        """
session:
  output_root: ./records
  data_type: real
  language_instruction: test
  collector_hash: tester
  pose_representation: xyz_xyzw
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
  initial_pose:
    enabled: true
    mode: tcp
    tcp_pose: [0.3, 0.0, 0.2, 0.0, 0.0, 0.0]
""",
        encoding="utf-8",
    )

    try:
        load_config(config_path)
    except Exception as error:
        assert "xyz_rxryrz" in str(error)
    else:
        raise AssertionError("应拒绝 quaternion TCP 配置用于 Piper 起始位姿控制。")
