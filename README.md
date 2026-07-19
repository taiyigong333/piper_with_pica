# Piper 多模态数据采集

本工程面向当前 Piper 单臂、RealSense D405（腕部）与 D435（前方）采集，同时以设备接口和 YAML 配置隔离硬件细节，便于后续接入其他机械臂、相机和可选模态。

当前采集的核心观测为：Piper 6 关节角（`rad`）、TCP 位姿（`m + xyzw`）、多个相机的 RGB 和 Piper 夹爪开闭行程（`m`）；深度采集由配置开关控制。数据按《数据采集与留存规范》写入 HDF5，并同时生成 `quality.json`、`manifest.json` 与 `checksums.sha256`。

## 环境与安装

项目使用 `uv` 管理依赖。锁文件要求 Python `>=3.10`；当前环境统一使用 uv 管理的 Python 3.11 虚拟环境，不使用 `pip` 或 Conda 安装项目依赖。安装 Git 和 `uv` 后，按以下顺序执行：

```bash
git clone --recurse-submodules git@github.com:taiyigong333/piper_with_pica.git
cd piper_with_pica
UV_CACHE_DIR=/tmp/uv-cache uv venv --python 3.11
UV_CACHE_DIR=/tmp/uv-cache uv sync --extra dev --extra realsense
```

`UV_CACHE_DIR=/tmp/uv-cache` 仅将 uv 的下载缓存放到可写的临时目录；虚拟环境仍位于工程的 `.venv/`。若仓库已经存在，只需在根目录执行后两条命令。若子模块尚未初始化或切换了子模块提交，再执行：

```bash
git submodule update --init --recursive
UV_CACHE_DIR=/tmp/uv-cache uv sync --extra dev --extra realsense
```

依赖安装完成后，先确认命令行入口可用：

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run piper-collect --help
```

## 快速验证（无硬件）

下面的命令只使用 Mock 设备，不会访问 CAN、相机或遥操进程。第一条命令输出的 `trajectory_path` 即后续校验所需的 HDF5 文件路径：

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run piper-collect collect \
  --config configs/mock_piper.yaml --duration 1
UV_CACHE_DIR=/tmp/uv-cache uv run pytest
```

使用上一条命令输出的路径执行只读校验：

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run piper-collect validate <trajectory_path>/trajectory.hdf5 \
  --config configs/mock_piper.yaml
```

## 真实设备采集

1. 复制并填写现场采集配置。`configs/现场_piper.yaml` 已被 Git 忽略，不会提交设备序列号、标定和采集员信息：

   ```bash
   cp configs/piper_d405_d435.example.yaml configs/现场_piper.yaml
   ```

   必填项为两台相机的 `serial_number`、`base_to_robot`、每台相机的 `base_to_camera`、`tool_offset_m`、任务指令和采集员哈希。未知的标定量必须保留 `null`，不能填单位矩阵。
2. 由现场人员启用 Piper 的 SocketCAN 接口，并用 `ip link show can0` 核对名称与 `robot.can_name` 一致。本工程不配置 CAN，也不发送任何运动、使能或复位指令。
3. 连接相机后，执行 `UV_CACHE_DIR=/tmp/uv-cache uv run piper-collect discover-realsense --require-device`，将输出序列号填入现场配置。
4. 在机械臂静止且工作空间安全时执行只读预检：

   ```bash
   UV_CACHE_DIR=/tmp/uv-cache uv run piper-collect preflight --config configs/现场_piper.yaml
   ```

   仅当输出的 `result` 为 `pass`，且关节、TCP、夹爪行程和相机分辨率均符合现场预期时才继续。Piper 夹爪读取 `GetArmGripperMsgs().gripper_state.grippers_angle`，SDK 原始单位为 `0.001 mm`，本工程写入 HDF5 前转换为米；预检会等待最多 1 秒的 `0x2A8` 首帧，未收到时明确失败，不会将 SDK 默认的 `0.0` 当作真实行程。该值是夹爪行程，非标准夹爪机构的指尖距离需另行标定。
5. 采集与校验：

   ```bash
   UV_CACHE_DIR=/tmp/uv-cache uv run piper-collect collect --config configs/现场_piper.yaml
   UV_CACHE_DIR=/tmp/uv-cache uv run piper-collect validate <trajectory_path>/trajectory.hdf5 \
     --config configs/现场_piper.yaml
   ```

Piper 适配器调用 `ConnectPort(piper_init=False)`，只接收 CAN 反馈，不发送运动、使能或 SDK 初始化查询指令。预检和采集均不会下发运动控制命令。CAN 接口初始化和设备权限仍须由现场人员按 Piper 官方文档处理。

## Pika Sense 遥操采集

遥操与采集保持独立，通过单独的会话编排层组合。先复制并填写被 Git 忽略的遥操配置，再校准和开始会话：

```bash
cp configs/pika_sense_piper.example.yaml configs/现场_pika_sense.yaml
UV_CACHE_DIR=/tmp/uv-cache uv run piper-collect calibrate-base \
  --teleop-config configs/现场_pika_sense.yaml --mode force
UV_CACHE_DIR=/tmp/uv-cache uv run piper-collect teleop-session \
  --config configs/现场_piper.yaml \
  --teleop-config configs/现场_pika_sense.yaml \
  --repeat
```

会话会在操作员确认 Sense 夹爪已双击启用遥操之后，才启动数据采集。结束时先在实体设备侧停止遥操，再按 Enter，并选择保存或删除该条轨迹。完整的安全流程、配置对应关系和已有 ROS 代码的边界见 [docs/2026-07-18_02_Pika_Sense遥操采集流程.md](docs/2026-07-18_02_Pika_Sense遥操采集流程.md)。

## 配置与扩展

- `modalities` 控制 RGB、深度、关节、TCP、夹爪位置是否采集与落盘。
- `acquisition.robot_hz` 和 `acquisition.camera_rig_hz` 分别控制原始状态采样及相机对齐时间轴。
- `cameras` 是可扩展列表；每台相机都通过 `driver` 注册到设备工厂。
- `robot.driver` 目前支持 `piper` 和 `mock`。接入新机械臂时实现 `RobotDevice`，并保证向核心层提供标准单位的 `RobotState`。
- 保持 `session.format_version`、字段名、单位或编码发生不兼容变化时，必须提升格式版本，并同步更新 `quality.py`。

文档按日期和同日序号排序；交接文档不参与排序。详细边界、数据流和现场接入项见 [docs/2026-07-18_01_架构与数据契约.md](docs/2026-07-18_01_架构与数据契约.md)、[docs/2026-07-18_02_Pika_Sense遥操采集流程.md](docs/2026-07-18_02_Pika_Sense遥操采集流程.md) 和 [docs/项目交接.md](docs/项目交接.md)。
