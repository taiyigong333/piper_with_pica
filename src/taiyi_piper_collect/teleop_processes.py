"""Pika 遥操残留进程的检查与显式终止。

常规会话由 ``ExternalTeleop`` 管理进程组。本模块只处理此前异常中断遗留的
Pika ROS 进程，避免下一条会话与旧控制器争抢 CAN、ROS 节点或 Sense USB。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import os
from pathlib import Path
import signal
import time
from typing import Any, Callable


_TELEOP_MARKERS = (
    "pika_remote_agx_arm teleop_single_piper.launch.py",
    "sensor_tools open_single_sensor_with_teleop.launch.py",
    "start_single_sensor_whit_teleop.bash",
    "/pika_ros/install/agx_arm_ctrl/lib/agx_arm_ctrl/agx_arm_ctrl_single",
    "/pika_ros/install/pika_locator/lib/pika_locator/pika_single_locator_node",
    "/pika_ros/install/pika_remote_agx_arm/lib/pika_remote_agx_arm/arm_ik_pose_node.py",
    "/pika_ros/install/pika_remote_agx_arm/lib/pika_remote_agx_arm/pub_delta_pose.py",
    "/pika_ros/install/sensor_tools/lib/sensor_tools/serial_gripper_imu",
    "/pika_ros/install/agx_arm_description/share/agx_arm_description/rviz/display.rviz",
)


@dataclass(frozen=True)
class TeleopProcess:
    """一个属于本机 Pika 单臂遥操链路的用户进程。"""

    pid: int
    command: str


def list_teleop_processes(proc_root: str | Path = "/proc") -> list[TeleopProcess]:
    """只列出当前用户的、命令行可明确识别的 Pika 遥操进程。"""

    root = Path(proc_root)
    current_pid = os.getpid()
    processes: list[TeleopProcess] = []
    try:
        entries = list(root.iterdir())
    except OSError:
        return processes
    for entry in entries:
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        if pid == current_pid:
            continue
        try:
            if entry.stat().st_uid != os.getuid():
                continue
            command = entry.joinpath("cmdline").read_bytes().replace(b"\0", b" ").decode("utf-8", "replace").strip()
        except OSError:
            continue
        if command and any(marker in command for marker in _TELEOP_MARKERS):
            processes.append(TeleopProcess(pid=pid, command=command))
    return sorted(processes, key=lambda process: process.pid)


def teleop_process_report(proc_root: str | Path = "/proc") -> dict[str, Any]:
    """生成不改变进程状态的残留进程报告。"""

    processes = list_teleop_processes(proc_root)
    return {"process_count": len(processes), "processes": [asdict(process) for process in processes]}


def terminate_teleop_processes(
    *,
    proc_root: str | Path = "/proc",
    list_processes_fn: Callable[[str | Path], list[TeleopProcess]] = list_teleop_processes,
    kill_fn: Callable[[int, signal.Signals], None] = os.kill,
    sleep_fn: Callable[[float], None] = time.sleep,
    monotonic_fn: Callable[[], float] = time.monotonic,
) -> dict[str, Any]:
    """按 SIGINT、SIGTERM、SIGKILL 逐级终止可明确识别的残留进程。"""

    initial = list_processes_fn(proc_root)
    signaled: list[dict[str, Any]] = []
    for signal_value, wait_s in ((signal.SIGINT, 5.0), (signal.SIGTERM, 3.0), (signal.SIGKILL, 1.0)):
        active = list_processes_fn(proc_root)
        if not active:
            break
        for process in active:
            try:
                kill_fn(process.pid, signal_value)
                signaled.append({"pid": process.pid, "signal": signal_value.name})
            except ProcessLookupError:
                pass
            except PermissionError as error:
                signaled.append({"pid": process.pid, "signal": signal_value.name, "error": str(error)})
        deadline = monotonic_fn() + wait_s
        while monotonic_fn() < deadline and list_processes_fn(proc_root):
            sleep_fn(0.1)
    remaining = list_processes_fn(proc_root)
    return {
        "initial_processes": [asdict(process) for process in initial],
        "signals": signaled,
        "remaining_processes": [asdict(process) for process in remaining],
        "result": "pass" if not remaining else "fail",
    }
