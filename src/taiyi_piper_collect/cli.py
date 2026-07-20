"""命令行入口。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .collector import DataCollector
from .config import load_config
from .errors import CollectionError, ConfigurationError, DeviceError, HardwareDependencyError
from .preflight import discover_realsense, preflight
from .piper_state import read_piper_state
from .quality import validate_hdf5
from .teleop_processes import teleop_process_report, terminate_teleop_processes
from .teleop_session import load_teleop_config, run_calibration, run_sessions


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Piper 多模态数据采集与质检")
    subparsers = parser.add_subparsers(dest="command", required=True)
    collect = subparsers.add_parser("collect", help="按 YAML 配置采集并生成数据包")
    collect.add_argument("--config", required=True, help="YAML 配置路径")
    collect.add_argument("--duration", type=float, help="覆盖 session.duration_s（秒）")
    validate = subparsers.add_parser("validate", help="只读验证既有 trajectory.hdf5")
    validate.add_argument("hdf5_path", help="trajectory.hdf5 路径")
    validate.add_argument("--config", help="可选：使用此 YAML 校验启用模态")
    preflight_parser = subparsers.add_parser("preflight", help="只读启动设备并读取一帧/一条反馈")
    preflight_parser.add_argument("--config", required=True, help="YAML 配置路径")
    discover = subparsers.add_parser("discover-realsense", help="列出可见的 RealSense 设备")
    discover.add_argument("--require-device", action="store_true", help="未发现设备时返回失败码")
    read_state = subparsers.add_parser("read-piper-state", help="只读输出 Piper 关节角和 TCP 位姿")
    read_state.add_argument("--config", required=True, help="YAML 配置路径")
    read_state.add_argument("--wait-s", type=float, default=0.2, help="连接后等待反馈稳定的秒数")
    calibrate = subparsers.add_parser("calibrate-base", help="执行 Pika 基站校准或定位诊断")
    calibrate.add_argument("--teleop-config", required=True, help="遥操 YAML 配置路径")
    calibrate.add_argument("--mode", choices=("force", "diagnose"), required=True, help="force 用于首次、硬件或频道变更")
    teleop_session = subparsers.add_parser("teleop-session", help="按安全顺序执行一条遥操-采集会话")
    teleop_session.add_argument("--config", required=True, help="采集 YAML 配置路径")
    teleop_session.add_argument("--teleop-config", required=True, help="遥操 YAML 配置路径")
    teleop_session.add_argument("--duration", type=float, help="可选采集上限（秒）；未设时由结束遥操确认收尾")
    teleop_session.add_argument("--on-complete", choices=("save", "delete"), help="完成后保存（默认）或删除轨迹")
    teleop_session.add_argument("--repeat", action="store_true", help="本条保存后按空格开始下一条独立轨迹")
    teleop_status = subparsers.add_parser("teleop-status", help="列出当前用户残留的 Pika 遥操进程")
    teleop_stop = subparsers.add_parser("teleop-stop", help="显式终止残留的 Pika 遥操进程")
    teleop_stop.add_argument("--terminate", action="store_true", help="执行终止；未提供时仅输出进程列表")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "collect":
            result = DataCollector(load_config(args.config)).run(args.duration)
            print(
                json.dumps(
                    {
                        "trajectory_id": result.trajectory_id,
                        "trajectory_path": str(result.trajectory_path),
                        "quality_path": str(result.quality_path),
                        "manifest_path": str(result.manifest_path),
                        "trajectory_length": result.writer_report.trajectory_length,
                    },
                    ensure_ascii=False,
                )
            )
            return 0
        if args.command == "validate":
            config = load_config(args.config) if args.config else None
            report = validate_hdf5(Path(args.hdf5_path), config)
            print(json.dumps(report, ensure_ascii=False, indent=2))
            return 0 if report["result"] == "pass" else 1
        if args.command == "preflight":
            report = preflight(load_config(args.config))
            print(json.dumps(report, ensure_ascii=False, indent=2))
            return 0 if report["result"] == "pass" else 1
        if args.command == "read-piper-state":
            print(json.dumps(read_piper_state(args.config, wait_s=args.wait_s), ensure_ascii=False, indent=2))
            return 0
        if args.command == "calibrate-base":
            run_calibration(load_teleop_config(args.teleop_config), args.mode)
            print(json.dumps({"result": "pass", "mode": args.mode}, ensure_ascii=False))
            return 0
        if args.command == "teleop-status":
            print(json.dumps(teleop_process_report(), ensure_ascii=False, indent=2))
            return 0
        if args.command == "teleop-stop":
            report = terminate_teleop_processes() if args.terminate else teleop_process_report()
            print(json.dumps(report, ensure_ascii=False, indent=2))
            return 0 if report.get("result", "pass") == "pass" else 1
        if args.command == "teleop-session":
            report = run_sessions(
                args.config,
                args.teleop_config,
                duration_s=args.duration,
                on_complete=args.on_complete,
                repeat=args.repeat,
            )
            print(json.dumps(report, ensure_ascii=False))
            return 0
        report = discover_realsense()
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if report["device_count"] or not args.require_device else 1
    except (CollectionError, ConfigurationError, DeviceError, HardwareDependencyError) as error:
        print(json.dumps({"result": "fail", "error": f"{type(error).__name__}: {error}"}, ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
