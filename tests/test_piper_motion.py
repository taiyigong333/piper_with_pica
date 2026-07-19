from __future__ import annotations

import math
from types import SimpleNamespace

from taiyi_piper_collect.config import InitialPoseConfig, RobotConfig
from taiyi_piper_collect.devices.piper_motion import PiperInitialPoseController


class FakePiperInterface:
    def __init__(self, joints: list[float], end_pose: list[float]) -> None:
        self._joints = joints
        self._end_pose = end_pose
        self.connected = False
        self.motion_commands: list[tuple[int, int, int, int]] = []
        self.joint_commands: list[tuple[int, ...]] = []
        self.end_pose_commands: list[tuple[int, ...]] = []

    def ConnectPort(self) -> None:
        self.connected = True

    def DisconnectPort(self) -> None:
        self.connected = False

    def EnablePiper(self) -> bool:
        return True

    def MotionCtrl_2(self, *command: int) -> None:
        self.motion_commands.append(command)

    def JointCtrl(self, *command: int) -> None:
        self.joint_commands.append(command)

    def EndPoseCtrl(self, *command: int) -> None:
        self.end_pose_commands.append(command)

    def GetArmJointMsgs(self):
        return SimpleNamespace(
            joint_state=SimpleNamespace(
                **{f"joint_{index}": self._joints[index - 1] * 180000.0 / math.pi for index in range(1, 7)}
            )
        )

    def GetArmEndPoseMsgs(self):
        return SimpleNamespace(
            end_pose=SimpleNamespace(
                X_axis=round(self._end_pose[0] * 1e6),
                Y_axis=round(self._end_pose[1] * 1e6),
                Z_axis=round(self._end_pose[2] * 1e6),
                RX_axis=round(self._end_pose[3] * 180000.0 / math.pi),
                RY_axis=round(self._end_pose[4] * 180000.0 / math.pi),
                RZ_axis=round(self._end_pose[5] * 180000.0 / math.pi),
            )
        )


def test_joint_initial_pose_sends_native_units_and_disconnects() -> None:
    target = [0.1, 0.2, -0.2, 0.3, -0.2, 0.5]
    interface = FakePiperInterface(target, [0.3, 0.0, 0.2, 0.0, 0.0, 0.0])
    controller = PiperInitialPoseController(
        RobotConfig(name="piper", driver="piper", can_name="can0"),
        InitialPoseConfig(enabled=True, mode="joint", joint_positions_rad=tuple(target)),
        interface_factory=lambda *_: interface,
    )

    state = controller.move()

    assert interface.motion_commands == [(0x01, 0x01, 10, 0x00)]
    assert interface.joint_commands == [tuple(round(value * 180000.0 / math.pi) for value in target)]
    assert not interface.connected
    assert state.joint_positions.tolist() == target


def test_tcp_initial_pose_converts_physical_tcp_to_flange_command() -> None:
    target = [0.2, 0.2, 0.3, 0.0, 0.0, math.pi / 2]
    # 工具在 J6 x 轴正向偏置 0.1 m，旋转后沿世界 y 轴；法兰目标 y 应减去该偏置。
    flange_pose = [0.2, 0.1, 0.3, 0.0, 0.0, math.pi / 2]
    interface = FakePiperInterface([0.0] * 6, flange_pose)
    controller = PiperInitialPoseController(
        RobotConfig(name="piper", driver="piper", can_name="can0", tool_offset_m=(0.1, 0.0, 0.0)),
        InitialPoseConfig(enabled=True, mode="tcp", tcp_pose=tuple(target)),
        interface_factory=lambda *_: interface,
    )

    controller.move()

    assert interface.motion_commands == [(0x01, 0x00, 10, 0x00)]
    assert interface.end_pose_commands == [
        (200_000, 100_000, 300_000, 0, 0, 90_000)
    ]
