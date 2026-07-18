# Piper 多模态数据采集

本工程面向当前 Piper 单臂、RealSense D405（腕部）与 D435（前方）采集，同时以设备接口和 YAML 配置隔离硬件细节，便于后续接入其他机械臂、相机和可选模态。

当前采集的核心观测为：Piper 6 关节角（`rad`）、TCP 位姿（`m + xyzw`）、多个相机的 RGB，深度和夹爪位置均由配置开关控制。数据按《数据采集与留存规范》写入 HDF5，并同时生成 `quality.json`、`manifest.json` 与 `checksums.sha256`。

## 环境与安装

项目使用 `uv` 管理虚拟环境和依赖。当前机器已经可使用 `uv`；在工程目录执行：

```bash
uv venv --python 3.10
uv sync --extra dev --extra realsense
```

`piper_sdk` 以 Git 子模块固定在本工程内。克隆工程时请使用：

```bash
git clone --recurse-submodules <repository-url>
cd code_piper
uv sync --extra dev --extra realsense
```

若已克隆但未初始化子模块：

```bash
git submodule update --init --recursive
```

## 快速验证（无硬件）

```bash
uv run piper-collect collect --config configs/mock_piper.yaml
uv run piper-collect validate configs/records/synthetic/<日期>/<轨迹ID>/trajectory.hdf5 \
  --config configs/mock_piper.yaml
uv run pytest
```

## 真实设备采集

1. 将 `configs/piper_d405_d435.yaml` 复制到不提交的现场配置文件，填入采集员哈希、任务、相机序列号、CAN 名称、标定矩阵和工具偏移。
2. 激活并确认 Piper 所在 SocketCAN 接口后，执行 `uv run piper-collect discover-realsense` 确认序列号。
3. 执行只读预检：`uv run piper-collect preflight --config <现场配置>`。
4. 确认机械臂工作空间安全后，执行 `uv run piper-collect collect --config <现场配置>`。

Piper 适配器调用 `ConnectPort(piper_init=False)`，只接收 CAN 反馈，不发送运动、使能或 SDK 初始化查询指令。预检和采集均不会下发运动控制命令。CAN 接口初始化和设备权限仍须由现场人员按 Piper 官方文档处理。

## 配置与扩展

- `modalities` 控制 RGB、深度、关节、TCP、夹爪位置是否采集与落盘。
- `acquisition.robot_hz` 和 `acquisition.camera_rig_hz` 分别控制原始状态采样及相机对齐时间轴。
- `cameras` 是可扩展列表；每台相机都通过 `driver` 注册到设备工厂。
- `robot.driver` 目前支持 `piper` 和 `mock`。接入新机械臂时实现 `RobotDevice`，并保证向核心层提供标准单位的 `RobotState`。
- 保持 `session.format_version`、字段名、单位或编码发生不兼容变化时，必须提升格式版本，并同步更新 `quality.py`。

详细边界、数据流和现场接入项见 [docs/architecture.md](docs/architecture.md) 和 [docs/project_handover.md](docs/project_handover.md)。
