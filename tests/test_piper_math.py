from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

from taiyi_piper_collect.config import GripperConfig, RobotConfig
from taiyi_piper_collect.devices.factory import create_gripper
from taiyi_piper_collect.devices.piper import PiperGripper, PiperRobot, euler_xyz_to_xyzw, gripper_stroke_to_m, rotate_vector_xyzw
from taiyi_piper_collect.errors import DeviceError


def test_euler_xyz_to_xyzw_identity() -> None:
    quaternion = euler_xyz_to_xyzw(np.zeros(3, dtype=np.float64))
    assert np.allclose(quaternion, [0.0, 0.0, 0.0, 1.0])


def test_tool_offset_follows_end_effector_orientation() -> None:
    quaternion = euler_xyz_to_xyzw(np.asarray([0.0, 0.0, math.pi / 2], dtype=np.float64))
    rotated = rotate_vector_xyzw(quaternion, np.asarray([1.0, 0.0, 0.0], dtype=np.float64))
    assert np.allclose(rotated, [0.0, 1.0, 0.0], atol=1e-12)


def test_piper_gripper_reads_sdk_stroke_in_meters() -> None:
    robot = PiperRobot(RobotConfig(name="piper", driver="piper", can_name="can0"), "xyz_xyzw")
    robot._interface = SimpleNamespace(
        GetArmGripperMsgs=lambda: SimpleNamespace(time_stamp=1.0, gripper_state=SimpleNamespace(grippers_angle=42_500))
    )

    gripper = create_gripper(GripperConfig(enabled=True, driver="piper"), robot)

    assert isinstance(gripper, PiperGripper)
    gripper.start()
    assert gripper.read_position() == pytest.approx(0.0425)
    gripper.stop()


def test_gripper_stroke_to_meters_rejects_nonfinite_value() -> None:
    assert gripper_stroke_to_m(70_000) == pytest.approx(0.07)
    with pytest.raises(DeviceError, match="NaN 或 Inf"):
        gripper_stroke_to_m(float("nan"))


def test_piper_gripper_rejects_sdk_default_zero_before_feedback() -> None:
    robot = PiperRobot(RobotConfig(name="piper", driver="piper", can_name="can0"), "xyz_xyzw")
    robot._interface = SimpleNamespace(
        GetArmGripperMsgs=lambda: SimpleNamespace(time_stamp=0.0, gripper_state=SimpleNamespace(grippers_angle=0))
    )

    with pytest.raises(DeviceError, match="0x2A8"):
        robot.read_gripper_position()
