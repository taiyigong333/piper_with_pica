"""Pika Sense 遥操与数据采集的会话编排。

本模块只管理外部进程和采集器的启动顺序，不导入或修改 ROS 遥操工程。Sense
夹爪的双击状态没有可用的稳定机器接口，因此必须由现场操作员显式确认后才开始采集。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
import os
from pathlib import Path
import re
import select
import shutil
import signal
import subprocess
import sys
import termios
import threading
import time
import tty
from typing import Any, Callable

import yaml

from .collector import CollectionResult, DataCollector
from .config import CollectConfig, load_config
from .devices.piper_motion import PiperInitialPoseController
from .errors import CollectionError, ConfigurationError, DeviceError, HardwareDependencyError
from .preflight import preflight


@dataclass(frozen=True)
class TeleopCommand:
    """一个保持运行的外部 ROS 启动命令。"""

    name: str
    command: str
    startup_wait_s: float = 3.0
    required_log_patterns: tuple[str, ...] = ()
    failure_log_patterns: tuple[str, ...] = ()
    failure_log_regexes: tuple[str, ...] = ()


@dataclass(frozen=True)
class TeleopConfig:
    """与采集 YAML 分离的遥操进程配置。"""

    source_path: Path
    pre_start_command: str | None
    force_calibration_command: str
    diagnose_calibration_command: str
    sensor: TeleopCommand
    controller: TeleopCommand


@dataclass
class _ManagedProcess:
    command: TeleopCommand
    process: Any
    log_file: Any
    log_path: Path


def _mapping(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ConfigurationError(f"{path} 必须是对象。")
    return value


def _required(mapping: dict[str, Any], key: str, path: str) -> Any:
    if key not in mapping:
        raise ConfigurationError(f"缺少必填配置：{path}.{key}")
    return mapping[key]


def _command(value: Any, path: str, default_name: str) -> TeleopCommand:
    raw = _mapping(value, path)
    command = str(_required(raw, "command", path)).strip()
    if not command:
        raise ConfigurationError(f"{path}.command 不能为空。")
    try:
        startup_wait_s = float(raw.get("startup_wait_s", 3.0))
    except (TypeError, ValueError) as error:
        raise ConfigurationError(f"{path}.startup_wait_s 必须为非负数。") from error
    if startup_wait_s < 0:
        raise ConfigurationError(f"{path}.startup_wait_s 必须为非负数。")
    return TeleopCommand(
        name=str(raw.get("name", default_name)),
        command=command,
        startup_wait_s=startup_wait_s,
        required_log_patterns=_log_patterns(raw.get("required_log_patterns", []), f"{path}.required_log_patterns"),
        failure_log_patterns=_log_patterns(raw.get("failure_log_patterns", []), f"{path}.failure_log_patterns"),
        failure_log_regexes=_log_regexes(raw.get("failure_log_regexes", []), f"{path}.failure_log_regexes"),
    )


def _log_patterns(value: Any, path: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(item, str) and item.strip() for item in value):
        raise ConfigurationError(f"{path} 必须是非空字符串列表。")
    return tuple(item.strip() for item in value)


def _log_regexes(value: Any, path: str) -> tuple[str, ...]:
    patterns = _log_patterns(value, path)
    for pattern in patterns:
        try:
            re.compile(pattern)
        except re.error as error:
            raise ConfigurationError(f"{path} 包含无效正则表达式 {pattern!r}：{error}") from error
    return patterns


def load_teleop_config(path: str | Path) -> TeleopConfig:
    """读取遥操编排配置，不接触任何硬件。"""

    source_path = Path(path).expanduser().resolve()
    try:
        with source_path.open("r", encoding="utf-8") as file:
            raw = _mapping(yaml.safe_load(file), "遥操根配置")
    except OSError as error:
        raise ConfigurationError(f"无法读取遥操配置 {source_path}：{error}") from error

    calibration = _mapping(_required(raw, "calibration", "遥操根配置"), "calibration")
    force_command = str(_required(calibration, "force_command", "calibration")).strip()
    diagnose_command = str(_required(calibration, "diagnose_command", "calibration")).strip()
    if not force_command or not diagnose_command:
        raise ConfigurationError("calibration 中的命令不能为空。")
    pre_start = raw.get("pre_start_command")
    if pre_start is not None and not str(pre_start).strip():
        raise ConfigurationError("pre_start_command 不能是空字符串；不需要时请设为 null。")
    return TeleopConfig(
        source_path=source_path,
        pre_start_command=str(pre_start).strip() if pre_start is not None else None,
        force_calibration_command=force_command,
        diagnose_calibration_command=diagnose_command,
        sensor=_command(_required(raw, "sensor", "遥操根配置"), "sensor", "pika-sense"),
        controller=_command(_required(raw, "controller", "遥操根配置"), "controller", "piper-controller"),
    )


class ExternalTeleop:
    """启动、监视并按进程组关闭既有的遥操命令。"""

    def __init__(self, config: TeleopConfig) -> None:
        self.config = config
        self._processes: list[_ManagedProcess] = []

    def run_pre_start(self) -> None:
        """在主终端执行必要的前置动作，例如缓存 sudo 凭据。"""

        if self.config.pre_start_command is None:
            return
        try:
            subprocess.run(["bash", "-lc", self.config.pre_start_command], check=True)
        except subprocess.CalledProcessError as error:
            raise CollectionError(f"遥操前置命令失败（退出码 {error.returncode}）。") from error

    def start(self, log_directory: Path) -> None:
        """按传感器、控制器顺序启动，并确认每个进程没有立即退出。"""

        log_directory.mkdir(parents=True, exist_ok=True)
        try:
            for command in (self.config.sensor, self.config.controller):
                self._start_command(command, log_directory)
        except BaseException:
            self.stop()
            raise

    def _start_command(self, command: TeleopCommand, log_directory: Path) -> None:
        log_path = log_directory / f"{command.name}.log"
        log_file = log_path.open("a", encoding="utf-8")
        try:
            process = subprocess.Popen(
                ["bash", "-lc", command.command],
                # 外部 Pika 脚本包含 sudo。保留启动终端可复用 sudo -v 的 tty 凭据；
                # 标准输出仍写日志，正常 ROS 节点不会读取终端输入。
                stdin=None,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                # 只创建新进程组而不创建新会话：既可 killpg 回收 ROS 子进程，又不会
                # 丢失 sudo 所需的控制终端。
                preexec_fn=os.setpgrp,
                env=_ros_child_environment(),
            )
        except BaseException:
            log_file.close()
            raise
        managed = _ManagedProcess(command=command, process=process, log_file=log_file, log_path=log_path)
        self._processes.append(managed)
        if command.startup_wait_s:
            time.sleep(command.startup_wait_s)
        return_code = process.poll()
        if return_code is not None:
            log_file.flush()
            raise CollectionError(
                f"遥操进程 {command.name} 在启动后退出（退出码 {return_code}）。请查看日志：{log_path}"
            )
        self._raise_for_log_health(managed)

    def return_codes(self) -> dict[str, int]:
        """返回异常退出的子进程及其退出码。"""

        return_codes: dict[str, int] = {}
        for managed in self._processes:
            return_code = managed.process.poll()
            if return_code is not None:
                return_codes[managed.command.name] = int(return_code)
        return return_codes

    def health_errors(self) -> dict[str, str]:
        """返回仍在运行但其日志已显示关键节点失败的遥操进程。"""

        errors: dict[str, str] = {}
        for managed in self._processes:
            failure = self._log_failure(managed)
            if failure is not None:
                errors[managed.command.name] = failure
        return errors

    def stop(self, timeout_s: float = 8.0) -> None:
        """并行关闭 ROS 进程组，避免控制器和传感器收尾时间串行叠加。"""

        managed_processes = list(reversed(self._processes))
        try:
            self._stop_process_groups([managed.process for managed in managed_processes], timeout_s)
        finally:
            for managed in managed_processes:
                managed.log_file.close()
        self._processes.clear()

    @staticmethod
    def _stop_process_groups(processes: list[Any], timeout_s: float) -> None:
        """每一轮同时发信号，再共同等待全部独立 ROS 进程组退出。"""

        active = [process for process in processes if ExternalTeleop._process_group_alive(process.pid)]
        for signal_value, wait_s in (
            (signal.SIGINT, timeout_s),
            (signal.SIGTERM, 3.0),
            (signal.SIGKILL, 2.0),
        ):
            if not active:
                break
            for process in active:
                try:
                    os.killpg(process.pid, signal_value)
                except ProcessLookupError:
                    pass
            active = ExternalTeleop._wait_for_process_groups_exit(active, wait_s)
        for process in active:
            if process.poll() is not None:
                continue
            try:
                process.wait(timeout=0.5)
            except subprocess.TimeoutExpired:
                pass

    @staticmethod
    def _process_group_alive(process_group: int) -> bool:
        try:
            os.killpg(process_group, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True

    @staticmethod
    def _wait_for_process_groups_exit(processes: list[Any], timeout_s: float) -> list[Any]:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            active = [process for process in processes if ExternalTeleop._process_group_alive(process.pid)]
            if not active:
                return []
            time.sleep(0.05)
        return [process for process in processes if ExternalTeleop._process_group_alive(process.pid)]

    def _raise_for_log_health(self, managed: _ManagedProcess) -> None:
        failure = self._log_failure(managed)
        if failure is not None:
            raise CollectionError(
                f"遥操进程 {managed.command.name} 启动失败，日志出现“{failure}”。请查看：{managed.log_path}"
            )
        content = self._read_log(managed)
        missing = [pattern for pattern in managed.command.required_log_patterns if pattern not in content]
        if missing:
            raise CollectionError(
                f"遥操进程 {managed.command.name} 未在 {managed.command.startup_wait_s:.1f} 秒内就绪，"
                f"日志缺少“{missing[0]}”。请查看：{managed.log_path}"
            )

    def _log_failure(self, managed: _ManagedProcess) -> str | None:
        content = self._read_log(managed)
        literal_failure = next((pattern for pattern in managed.command.failure_log_patterns if pattern in content), None)
        if literal_failure is not None:
            return literal_failure
        regex_failure = next(
            (pattern for pattern in managed.command.failure_log_regexes if re.search(pattern, content)), None
        )
        return f"正则 {regex_failure}" if regex_failure is not None else None

    @staticmethod
    def _read_log(managed: _ManagedProcess) -> str:
        managed.log_file.flush()
        try:
            return managed.log_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return ""


def _ros_child_environment() -> dict[str, str]:
    """隔离 uv 虚拟环境，令 ROS 子进程由其配置的 Python 3.10 环境启动。"""

    environment = os.environ.copy()
    virtual_env = environment.pop("VIRTUAL_ENV", None)
    environment.pop("PYTHONHOME", None)
    environment.pop("PYTHONPATH", None)
    # OpenCV 在首条采集后会把 .venv 的 Qt xcb 插件目录写进父进程环境；若
    # 继承给第二条 ROS 的 RViz，会加载 ABI 不匹配的插件并以 exit code -6 退出。
    environment.pop("QT_QPA_PLATFORM_PLUGIN_PATH", None)
    environment.pop("QT_PLUGIN_PATH", None)
    if virtual_env:
        virtual_bin = str(Path(virtual_env) / "bin")
        path_parts = environment.get("PATH", "").split(os.pathsep)
        environment["PATH"] = os.pathsep.join(part for part in path_parts if part != virtual_bin)
    return environment


def run_calibration(teleop_config: TeleopConfig, mode: str) -> None:
    """执行用户明确选择的基站校准/诊断命令。"""

    if mode == "force":
        command = teleop_config.force_calibration_command
    elif mode == "diagnose":
        command = teleop_config.diagnose_calibration_command
    else:
        raise ConfigurationError("校准模式只支持 force 或 diagnose。")
    try:
        subprocess.run(["bash", "-lc", command], check=True)
    except subprocess.CalledProcessError as error:
        raise CollectionError(f"基站{mode}命令失败（退出码 {error.returncode}）。") from error


def _wait_for_space(
    message: str,
    *,
    input_fn: Callable[[str], str],
    output_fn: Callable[[str], None],
    health_check: Callable[[], None] | None = None,
    allow_quit: bool = False,
) -> bool:
    """前台以单个空格推进会话，避免输入 yes/no 或再按 Enter。"""

    output_fn(f"{message} 按空格继续；按 q 取消。")
    if input_fn is not input:
        try:
            key = input_fn("")
        except EOFError as error:
            raise CollectionError("未收到空格确认，已取消会话。") from error
        if key == " ":
            return True
        if allow_quit and key.lower() == "q":
            return False
        raise CollectionError("仅支持空格继续或 q 取消。")
    if not sys.stdin.isatty():
        raise CollectionError("遥操会话需要交互式终端接收空格确认。")

    file_descriptor = sys.stdin.fileno()
    old_settings = termios.tcgetattr(file_descriptor)
    try:
        tty.setraw(file_descriptor)
        while True:
            if health_check is not None:
                health_check()
            ready, _, _ = select.select([file_descriptor], [], [], 0.1)
            if not ready:
                continue
            key = sys.stdin.read(1)
            if key == " ":
                output_fn("")
                return True
            if key.lower() == "q":
                output_fn("")
                if allow_quit:
                    return False
                raise CollectionError("操作员取消会话；未继续下一阶段。")
            if key == "\x03":
                raise KeyboardInterrupt
    finally:
        termios.tcsetattr(file_descriptor, termios.TCSADRAIN, old_settings)


def _preflight_or_raise(config: CollectConfig) -> None:
    report = preflight(config)
    if report["result"] != "pass":
        errors = "; ".join(report["errors"])
        raise CollectionError(f"采集预检失败：{errors}")


def _delete_result(result: CollectionResult, config: CollectConfig) -> None:
    trajectory_dir = result.trajectory_path.parent.resolve()
    output_root = config.session.output_root.resolve()
    if not trajectory_dir.is_relative_to(output_root):
        raise CollectionError(f"拒绝删除输出根目录之外的轨迹：{trajectory_dir}")
    shutil.rmtree(trajectory_dir)


def _completion_action(
    result: CollectionResult,
    config: CollectConfig,
    preset: str | None,
) -> str:
    action = preset or "save"
    if action == "delete":
        _delete_result(result, config)
    return action


def _teleop_health_or_raise(teleop: ExternalTeleop) -> None:
    return_codes = teleop.return_codes()
    if return_codes:
        raise CollectionError(f"遥操进程异常退出：{return_codes}")
    health_errors = teleop.health_errors()
    if health_errors:
        details = "; ".join(f"{name}: {pattern}" for name, pattern in health_errors.items())
        raise CollectionError(f"遥操关键节点异常：{details}")


def _collection_health_or_raise(teleop: ExternalTeleop, error_box: dict[str, BaseException]) -> None:
    _teleop_health_or_raise(teleop)
    _raise_collection_error(error_box)


def _raise_collection_error(error_box: dict[str, BaseException]) -> None:
    error = error_box.get("error")
    if error is None:
        return
    if isinstance(error, CollectionError):
        raise error
    raise CollectionError(f"采集失败：{type(error).__name__}: {error}") from error


def _move_to_initial_pose(config: CollectConfig, output_fn: Callable[[str], None]) -> None:
    initial_pose = config.robot.initial_pose
    if not initial_pose.enabled:
        output_fn("robot.initial_pose.enabled=false，跳过本项目的起始位姿控制。")
        return
    try:
        state = PiperInitialPoseController(config.robot, initial_pose).move()
    except (DeviceError, HardwareDependencyError) as error:
        raise CollectionError(f"Piper 起始位姿未完成：{error}") from error
    output_fn(
        "Piper 已到达起始位姿："
        + ("关节控制" if initial_pose.mode == "joint" else "TCP 控制")
        + f"，反馈时间戳 {state.timestamp:.3f}。"
    )


def run_session(
    config_path: str | Path,
    teleop_config_path: str | Path,
    *,
    duration_s: float | None = None,
    on_complete: str | None = None,
    input_fn: Callable[[str], str] = input,
    output_fn: Callable[[str], None] = print,
    show_calibration_reminder: bool = True,
) -> dict[str, Any]:
    """执行一条人工遥操轨迹，并返回保存或删除后的会话摘要。"""

    config = load_config(config_path)
    teleop_config = load_teleop_config(teleop_config_path)
    if on_complete not in {None, "save", "delete"}:
        raise ConfigurationError("on_complete 只支持 save、delete 或 null。")

    if show_calibration_reminder:
        _wait_for_space(
            "已按现场情况完成基站校准（首次或基站、频道变更必须 force 校准）。",
            input_fn=input_fn,
            output_fn=output_fn,
        )
    _preflight_or_raise(config)
    _wait_for_space(
        "预检通过，已确认工作区无人且路径安全；下一步将由本项目移动 Piper 至配置的起始位姿。",
        input_fn=input_fn,
        output_fn=output_fn,
    )
    _move_to_initial_pose(config, output_fn)

    log_directory = config.session.output_root / "teleop_logs" / datetime.now().strftime("%Y%m%dT%H%M%S")
    teleop = ExternalTeleop(teleop_config)
    stop_request = threading.Event()
    capture_stopped = threading.Event()
    result_box: dict[str, CollectionResult] = {}
    error_box: dict[str, BaseException] = {}

    def collect() -> None:
        try:
            result_box["result"] = DataCollector(config).run(
                duration_s=duration_s,
                stop_request=stop_request,
                capture_stopped=capture_stopped,
                until_stopped=duration_s is None,
            )
        except BaseException as error:
            error_box["error"] = error

    collector_thread: threading.Thread | None = None
    try:
        teleop.run_pre_start()
        teleop.start(log_directory)
        _teleop_health_or_raise(teleop)
        _wait_for_space(
            "遥操节点已就绪。双击 Sense 夹爪启用遥操并确认 Piper 已跟随动作后，",
            input_fn=input_fn,
            output_fn=output_fn,
            health_check=lambda: _teleop_health_or_raise(teleop),
        )
        output_fn(f"遥操日志目录：{log_directory}")
        collector_thread = threading.Thread(target=collect, name="teleop-data-collector", daemon=True)
        collector_thread.start()
        _wait_for_space(
            "本条遥操内容已完成；保持 Sense 遥操运行并按空格结束数据采集，",
            input_fn=input_fn,
            output_fn=output_fn,
            health_check=lambda: _collection_health_or_raise(teleop, error_box),
        )
        stop_request.set()
        output_fn("已收到停止请求：已禁止新帧进入采集队列，正在关闭 HDF5 并生成质检文件。")
        if not capture_stopped.wait(timeout=3.0):
            raise CollectionError("采集线程未在 3 秒内停止取帧；已中止会话收尾。")
        output_fn("采集已停止，正在完成写盘。")
        collector_thread.join(timeout=30.0)
        if collector_thread.is_alive():
            raise CollectionError("采集器未在 30 秒内完成收尾。")
        _raise_collection_error(error_box)
        _wait_for_space(
            "数据采集和写盘已完成。现在双击 Sense 夹爪停止遥操后，",
            input_fn=input_fn,
            output_fn=output_fn,
            health_check=lambda: _teleop_health_or_raise(teleop),
        )
    finally:
        stop_request.set()
        if collector_thread is not None and collector_thread.is_alive():
            collector_thread.join(timeout=30.0)
        output_fn("正在并行关闭 Pika ROS 进程，请等待。")
        teleop.stop()

    result = result_box.get("result")
    if result is None:
        raise CollectionError("采集未返回结果。")
    action = _completion_action(result, config, on_complete)
    return {
        "result": "pass",
        "action": action,
        "trajectory_id": result.trajectory_id,
        "trajectory_path": str(result.trajectory_path),
        "quality_path": str(result.quality_path),
        "manifest_path": str(result.manifest_path),
        "trajectory_length": result.writer_report.trajectory_length,
        "teleop_log_directory": str(log_directory),
    }


def run_sessions(
    config_path: str | Path,
    teleop_config_path: str | Path,
    *,
    duration_s: float | None = None,
    on_complete: str | None = None,
    repeat: bool = False,
    input_fn: Callable[[str], str] = input,
    output_fn: Callable[[str], None] = print,
) -> dict[str, Any]:
    """顺序执行一条或多条独立遥操轨迹。"""

    reports: list[dict[str, Any]] = []
    while True:
        reports.append(
            run_session(
                config_path,
                teleop_config_path,
                duration_s=duration_s,
                on_complete=on_complete,
                input_fn=input_fn,
                output_fn=output_fn,
                show_calibration_reminder=not repeat or not reports,
            )
        )
        if not repeat:
            return reports[0]
        if not _wait_for_space(
            "本条数据已保存。准备下一条独立轨迹时，",
            input_fn=input_fn,
            output_fn=output_fn,
            allow_quit=True,
        ):
            return {"result": "pass", "trajectory_count": len(reports), "sessions": reports}


def _parser() -> Any:
    import argparse

    parser = argparse.ArgumentParser(description="Pika Sense 遥操与 Piper 数据采集会话编排")
    subparsers = parser.add_subparsers(dest="command", required=True)
    calibrate = subparsers.add_parser("calibrate", help="执行基站校准或定位诊断")
    calibrate.add_argument("--teleop-config", required=True, help="遥操 YAML 配置路径")
    calibrate.add_argument("--mode", choices=("force", "diagnose"), required=True, help="force 用于首次/硬件或频道变更")
    session = subparsers.add_parser("run", help="按安全顺序执行一条遥操-采集会话")
    session.add_argument("--config", required=True, help="采集 YAML 配置路径")
    session.add_argument("--teleop-config", required=True, help="遥操 YAML 配置路径")
    session.add_argument("--duration", type=float, help="可选采集上限（秒）；未设时由结束遥操确认收尾")
    session.add_argument("--on-complete", choices=("save", "delete"), help="完成后保存（默认）或删除轨迹")
    session.add_argument("--repeat", action="store_true", help="本条保存后按空格开始下一条独立轨迹")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        teleop_config = load_teleop_config(args.teleop_config)
        if args.command == "calibrate":
            run_calibration(teleop_config, args.mode)
            print(json.dumps({"result": "pass", "mode": args.mode}, ensure_ascii=False))
            return 0
        report = run_sessions(
            args.config,
            args.teleop_config,
            duration_s=args.duration,
            on_complete=args.on_complete,
            repeat=args.repeat,
        )
        print(json.dumps(report, ensure_ascii=False))
        return 0
    except (CollectionError, ConfigurationError) as error:
        print(json.dumps({"result": "fail", "error": f"{type(error).__name__}: {error}"}, ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
