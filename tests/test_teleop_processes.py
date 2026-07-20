from __future__ import annotations

import signal

from taiyi_piper_collect.teleop_processes import TeleopProcess, list_teleop_processes, terminate_teleop_processes


def test_list_teleop_processes_only_returns_known_pika_commands(tmp_path) -> None:
    target = tmp_path / "101"
    target.mkdir()
    (target / "cmdline").write_bytes(
        b"/usr/bin/python3\0/home/ubuntu/pika_ros/install/agx_arm_ctrl/lib/agx_arm_ctrl/agx_arm_ctrl_single\0"
    )
    unrelated = tmp_path / "102"
    unrelated.mkdir()
    (unrelated / "cmdline").write_bytes(b"/usr/bin/python3\0training.py\0")

    processes = list_teleop_processes(tmp_path)

    assert processes == [TeleopProcess(pid=101, command=processes[0].command)]


def test_terminate_teleop_processes_prefers_sigint() -> None:
    process = TeleopProcess(pid=101, command="pika_remote_agx_arm teleop_single_piper.launch.py")
    calls = 0
    signals: list[tuple[int, signal.Signals]] = []

    def list_processes(_proc_root):
        nonlocal calls
        calls += 1
        return [process] if calls <= 2 else []

    report = terminate_teleop_processes(
        proc_root="/unused",
        list_processes_fn=list_processes,
        kill_fn=lambda pid, signal_value: signals.append((pid, signal_value)),
        sleep_fn=lambda _: None,
        monotonic_fn=lambda: 0.0,
    )

    assert signals == [(101, signal.SIGINT)]
    assert report["result"] == "pass"
