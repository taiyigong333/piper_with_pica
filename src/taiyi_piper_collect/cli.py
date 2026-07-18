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
from .quality import validate_hdf5


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
        report = discover_realsense()
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if report["device_count"] or not args.require_device else 1
    except (CollectionError, ConfigurationError, DeviceError, HardwareDependencyError) as error:
        print(json.dumps({"result": "fail", "error": f"{type(error).__name__}: {error}"}, ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
