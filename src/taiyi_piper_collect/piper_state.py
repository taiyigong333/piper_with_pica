"""Piper 关节角与 TCP 的一次性只读输出。"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from .config import load_config
from .devices.factory import create_robot
from .errors import ConfigurationError, DeviceError, HardwareDependencyError


def read_piper_state(config_path: str | Path, *, wait_s: float = 0.2) -> dict[str, Any]:
    """读取一帧 Piper 反馈，不发送使能、查询或运动控制帧。"""

    if wait_s < 0:
        raise ConfigurationError("--wait-s 必须为非负数。")
    config = load_config(config_path)
    if config.robot.driver != "piper":
        raise ConfigurationError("读取 Piper 状态需要 robot.driver=piper。")
    robot = create_robot(config.robot, config.session.pose_representation)
    robot.start()
    try:
        # Piper 持续广播状态；短暂等待避免连接刚建立时读取 SDK 的初始缓存。
        if wait_s:
            time.sleep(wait_s)
        state = robot.read()
    finally:
        robot.stop()
    return {
        "joint_positions_rad": [float(value) for value in state.joint_positions],
        "tcp_pose": [float(value) for value in state.tcp_pose] if state.tcp_pose is not None else None,
        "tcp_pose_representation": config.session.pose_representation,
        "timestamp": state.timestamp,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="只读输出 Piper 六关节角和 TCP 位姿")
    parser.add_argument("--config", required=True, help="采集 YAML 配置路径")
    parser.add_argument("--wait-s", type=float, default=0.2, help="连接后等待反馈稳定的秒数")
    args = parser.parse_args(argv)
    try:
        print(json.dumps(read_piper_state(args.config, wait_s=args.wait_s), ensure_ascii=False, indent=2))
        return 0
    except (ConfigurationError, DeviceError, HardwareDependencyError) as error:
        print(json.dumps({"result": "fail", "error": f"{type(error).__name__}: {error}"}, ensure_ascii=False))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
