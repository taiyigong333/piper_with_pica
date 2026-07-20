from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import signal

from taiyi_piper_collect.collector import CollectionStats
from taiyi_piper_collect.config import load_config
from taiyi_piper_collect.writer import WriterReport
import taiyi_piper_collect.teleop_session as teleop_session


def test_teleop_example_can_be_loaded() -> None:
    config_path = Path(__file__).parents[1] / "configs" / "pika_sense_piper.example.yaml"
    config = teleop_session.load_teleop_config(config_path)

    assert config.sensor.name == "pika_sense"
    assert "survive-cli --force-calibrate" in config.force_calibration_command
    assert "teleop_single_piper.launch.py" in config.controller.command
    assert "pub_delta_pose.py" in config.controller.required_log_patterns
    assert "process has died" not in config.controller.failure_log_patterns
    assert config.controller.failure_log_regexes


def test_ros_child_environment_removes_uv_python_environment(monkeypatch) -> None:
    monkeypatch.setenv("VIRTUAL_ENV", "/tmp/project/.venv")
    monkeypatch.setenv("PYTHONHOME", "/tmp/python-home")
    monkeypatch.setenv("PYTHONPATH", "/tmp/project/src")
    monkeypatch.setenv("PATH", "/tmp/project/.venv/bin:/usr/bin")

    environment = teleop_session._ros_child_environment()

    assert "VIRTUAL_ENV" not in environment
    assert "PYTHONHOME" not in environment
    assert "PYTHONPATH" not in environment
    assert environment["PATH"] == "/usr/bin"


def test_external_teleop_starts_in_order_and_stops_process_groups(tmp_path: Path, monkeypatch) -> None:
    config_path = Path(__file__).parents[1] / "configs" / "pika_sense_piper.example.yaml"
    config = teleop_session.load_teleop_config(config_path)
    config = replace(
        config,
        sensor=replace(config.sensor, required_log_patterns=()),
        controller=replace(config.controller, required_log_patterns=()),
    )
    started: list[FakeProcess] = []
    commands: list[str] = []
    killed: list[tuple[int, signal.Signals]] = []

    def fake_popen(args, **kwargs):
        assert args[:2] == ["bash", "-lc"]
        assert kwargs["stdin"] is None
        assert kwargs["preexec_fn"] is teleop_session.os.setpgrp
        assert "start_new_session" not in kwargs
        commands.append(args[2])
        process = FakeProcess(pid=100 + len(started))
        started.append(process)
        return process

    def fake_killpg(pid: int, sig: signal.Signals) -> None:
        process = next(process for process in started if process.pid == pid)
        if sig == 0:
            if process.return_code is None:
                return
            raise ProcessLookupError
        killed.append((pid, sig))
        process.return_code = 0

    monkeypatch.setattr(teleop_session.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(teleop_session.os, "killpg", fake_killpg)
    monkeypatch.setattr(teleop_session.time, "sleep", lambda _: None)

    external = teleop_session.ExternalTeleop(config)
    external.start(tmp_path / "logs")
    external.stop()

    assert len(started) == 2
    assert "start_single_sensor_whit_teleop.bash" in commands[0]
    assert "teleop_single_piper.launch.py" in commands[1]
    assert killed == [(101, signal.SIGINT), (100, signal.SIGINT)]
    assert (tmp_path / "logs" / "pika_sense.log").exists()
    assert (tmp_path / "logs" / "piper_controller.log").exists()


def test_controller_log_ignores_rviz_exit_but_detects_core_node_exit(tmp_path: Path) -> None:
    config_path = Path(__file__).parents[1] / "configs" / "pika_sense_piper.example.yaml"
    config = teleop_session.load_teleop_config(config_path)
    log_path = tmp_path / "piper_controller.log"
    with log_path.open("w+", encoding="utf-8") as log_file:
        managed = teleop_session._ManagedProcess(
            command=config.controller,
            process=FakeProcess(pid=100),
            log_file=log_file,
            log_path=log_path,
        )
        log_file.write("[ERROR] [rviz2-3]: process has died [pid 1, exit code -6].\n")
        log_file.flush()
        external = teleop_session.ExternalTeleop(config)
        assert external._log_failure(managed) is None

        log_file.write("[ERROR] [arm_ik_pose_node.py-5]: process has died [pid 2, exit code 1].\n")
        log_file.flush()
        failure = external._log_failure(managed)
        assert failure is not None
        assert "arm_ik_pose_node.py" in failure


def test_repeat_session_restarts_independent_trajectory(monkeypatch) -> None:
    reports = iter(
        [
            {"result": "pass", "trajectory_id": "first"},
            {"result": "pass", "trajectory_id": "second"},
        ]
    )
    calls: list[str] = []

    def fake_run_session(*args, **kwargs):
        calls.append("run")
        return next(reports)

    monkeypatch.setattr(teleop_session, "run_session", fake_run_session)
    answers = iter([" ", "q"])
    report = teleop_session.run_sessions(
        "collect.yaml",
        "teleop.yaml",
        repeat=True,
        input_fn=lambda _: next(answers),
    )

    assert calls == ["run", "run"]
    assert report["trajectory_count"] == 2
    assert [item["trajectory_id"] for item in report["sessions"]] == ["first", "second"]


def test_session_starts_collection_only_after_teleop_confirmation(tmp_path: Path, monkeypatch) -> None:
    config_path = Path(__file__).parents[1] / "configs" / "mock_piper.yaml"
    config = load_config(config_path)
    config = replace(config, session=replace(config.session, output_root=tmp_path / "records"))
    events: list[str] = []

    class FakeTeleop:
        def __init__(self, _config) -> None:
            pass

        def run_pre_start(self) -> None:
            events.append("pre-start")

        def start(self, _log_directory: Path) -> None:
            events.append("teleop-start")

        def return_codes(self) -> dict[str, int]:
            return {}

        def health_errors(self) -> dict[str, str]:
            return {}

        def stop(self) -> None:
            events.append("teleop-stop")

    class FakeCollector:
        def __init__(self, _config) -> None:
            pass

        def run(self, *, stop_request, **_kwargs):
            events.append("collect-start")
            assert stop_request.wait(timeout=1.0)
            events.append("collect-finished")
            trajectory_path = tmp_path / "records" / "synthetic" / "20260718" / "mock" / "trajectory.hdf5"
            return teleop_session.CollectionResult(
                trajectory_id="mock",
                trajectory_path=trajectory_path,
                quality_path=trajectory_path.with_name("quality.json"),
                manifest_path=trajectory_path.with_name("manifest.json"),
                stats=CollectionStats(),
                writer_report=WriterReport(trajectory_path, 1, 1, 1),
            )

    monkeypatch.setattr(teleop_session, "load_config", lambda _: config)
    monkeypatch.setattr(teleop_session, "preflight", lambda _: {"result": "pass", "errors": []})
    monkeypatch.setattr(teleop_session, "ExternalTeleop", FakeTeleop)
    monkeypatch.setattr(teleop_session, "DataCollector", FakeCollector)
    monkeypatch.setattr(teleop_session, "load_teleop_config", lambda _: object())
    answers = iter([" ", " ", " ", " ", " "])

    report = teleop_session.run_session(
        "collect.yaml",
        "teleop.yaml",
        input_fn=lambda _: next(answers),
        output_fn=lambda _: None,
    )

    assert report["action"] == "save"
    assert events.index("teleop-start") < events.index("collect-start") < events.index("teleop-stop")
    assert events.index("collect-finished") < events.index("teleop-stop")


class FakeProcess:
    def __init__(self, pid: int) -> None:
        self.pid = pid
        self.return_code: int | None = None

    def poll(self) -> int | None:
        return self.return_code

    def wait(self, timeout: float | None = None) -> int:
        assert self.return_code is not None
        return self.return_code
