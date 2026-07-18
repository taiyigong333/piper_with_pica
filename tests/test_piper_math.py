from __future__ import annotations

import math

import numpy as np

from taiyi_piper_collect.devices.piper import euler_xyz_to_xyzw, rotate_vector_xyzw


def test_euler_xyz_to_xyzw_identity() -> None:
    quaternion = euler_xyz_to_xyzw(np.zeros(3, dtype=np.float64))
    assert np.allclose(quaternion, [0.0, 0.0, 0.0, 1.0])


def test_tool_offset_follows_end_effector_orientation() -> None:
    quaternion = euler_xyz_to_xyzw(np.asarray([0.0, 0.0, math.pi / 2], dtype=np.float64))
    rotated = rotate_vector_xyzw(quaternion, np.asarray([1.0, 0.0, 0.0], dtype=np.float64))
    assert np.allclose(rotated, [0.0, 1.0, 0.0], atol=1e-12)
