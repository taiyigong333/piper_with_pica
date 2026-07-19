# Piper 多模态数据采集

本工程面向当前 Piper 单臂、RealSense D405（腕部）与 D435（前方）采集，同时以设备接口和 YAML 配置隔离硬件细节，便于后续接入其他机械臂、相机和可选模态。

当前采集的核心观测为：Piper 6 关节角（`rad`）、TCP 位姿（默认 `m + rxryrz`，欧拉角为 `rad`）、多个相机的 RGB 和 Piper 夹爪开闭行程（`m`）；深度采集由配置开关控制。数据按《数据采集与留存规范》写入 HDF5，并同时生成 `quality.json`、`manifest.json` 与 `checksums.sha256`。

## 环境与安装

项目使用 `uv` 管理依赖，并固定使用 Python 3.10；不使用 `pip` 或 Conda 安装项目依赖。该约束与 ROS Humble/Pika 所需的 Python 3.10 一致，但项目 `.venv/` 仍与系统 Python、`pika` Conda 环境完全隔离。安装 Git 和 `uv` 后，按以下顺序执行：

```bash
# 克隆工程，并将 Piper SDK 子模块一并下载。
git clone --recurse-submodules git@github.com:taiyigong333/piper_with_pica.git
# 进入刚克隆的工程根目录。
cd piper_with_pica
# 创建项目专用的 Python 3.10 虚拟环境，不安装依赖；不会修改系统 Python。
UV_CACHE_DIR=/tmp/uv-cache uv venv --python 3.10
# 根据 uv.lock 安装运行、测试和 RealSense 依赖。
UV_CACHE_DIR=/tmp/uv-cache uv sync --extra dev --extra realsense
```

`UV_CACHE_DIR=/tmp/uv-cache` 仅将 uv 的下载缓存放到可写的临时目录；虚拟环境仍位于工程的 `.venv/`。若仓库已经存在，只需在根目录执行后两条命令。若子模块尚未初始化或切换了子模块提交，再执行：

```bash
# 初始化或更新 Git 子模块，使本地 Piper SDK 与主仓库记录的提交一致。
git submodule update --init --recursive
# 在子模块更新后重新同步依赖与本地 editable Piper SDK。
UV_CACHE_DIR=/tmp/uv-cache uv sync --extra dev --extra realsense
```

依赖安装完成后，先确认命令行入口可用：

```bash
# 列出所有采集 CLI 子命令及其参数；不访问硬件，也不生成数据。
UV_CACHE_DIR=/tmp/uv-cache uv run piper-collect --help
```

## 快速验证（无硬件）

下面的命令只使用 Mock 设备，不会访问 CAN、相机或遥操进程。第一条命令输出的 `trajectory_path` 即后续校验所需的 HDF5 文件路径：

```bash
# 使用 Mock 机器人和 Mock 相机采集 1 秒测试轨迹，并输出 trajectory_path。
UV_CACHE_DIR=/tmp/uv-cache uv run piper-collect collect \
  --config configs/mock_piper.yaml --duration 1
# 运行全部单元和集成测试；不会访问真实机器人或相机。
UV_CACHE_DIR=/tmp/uv-cache uv run pytest
```

使用上一条命令输出的路径执行只读校验：

```bash
# 只读检查指定 HDF5 的结构、时间轴和已启用模态；不修改轨迹文件。
UV_CACHE_DIR=/tmp/uv-cache uv run piper-collect validate <trajectory_path>/trajectory.hdf5 \
  --config configs/mock_piper.yaml
```

## 真实设备采集

1. 复制并填写现场采集配置。`configs/现场_piper.yaml` 已被 Git 忽略，不会提交设备序列号、标定和采集员信息：

   ```bash
   # 从全注释模板生成本机现场配置；新文件已被 Git 忽略。
   cp configs/piper_d405_d435.example.yaml configs/现场_piper.yaml
   ```

   必填项为两台相机的 `serial_number`、`base_to_robot`、每台相机的 `base_to_camera`、`tool_offset_m`、任务指令和采集员哈希。未知的标定量必须保留 `null`，不能填单位矩阵。若要让遥操会话自动移动到起始位姿，填写 `robot.initial_pose` 的目标并设为 `enabled: true`；`joint_positions_rad` 是六关节 rad，`tcp_pose` 是 `[x,y,z,rx,ry,rz]` 的 m + rad，且只填写 `mode` 对应的一项。
2. 由现场人员启用 Piper 的 SocketCAN 接口，并执行下列只读命令核对名称与 `robot.can_name` 一致。本工程不配置 CAN，也不发送任何运动、使能或复位指令：

   ```bash
   # 显示 can0 的链路状态、位速率和名称；不配置或改动 CAN。
   ip link show can0
   ```

3. 连接相机后执行设备发现，将输出序列号填入现场配置：

   ```bash
   # 列出可见 RealSense；未发现相机时以非零状态退出；不创建采集文件。
   UV_CACHE_DIR=/tmp/uv-cache uv run piper-collect discover-realsense --require-device
   ```

4. 在机械臂静止且工作空间安全时执行只读预检：

   ```bash
   # 读取一次相机、Piper 和夹爪反馈，检查配置与设备匹配；不创建轨迹或发送控制帧。
   UV_CACHE_DIR=/tmp/uv-cache uv run piper-collect preflight --config configs/现场_piper.yaml
   ```

   仅当输出的 `result` 为 `pass`，且关节、TCP、夹爪行程和相机分辨率均符合现场预期时才继续。Piper 夹爪读取 `GetArmGripperMsgs().gripper_state.grippers_angle`，SDK 原始单位为 `0.001 mm`，本工程写入 HDF5 前转换为米；预检会等待最多 1 秒的 `0x2A8` 首帧，未收到时明确失败，不会将 SDK 默认的 `0.0` 当作真实行程。该值是夹爪行程，非标准夹爪机构的指尖距离需另行标定。

   起始位姿可先用以下只读命令记录当前关节角和 TCP，输出单位与 YAML 完全一致：

   ```bash
   # 只读输出六关节 rad 与当前配置表示的 TCP；不使能、不移动 Piper。
   UV_CACHE_DIR=/tmp/uv-cache uv run piper-collect read-piper-state \
     --config configs/现场_piper.yaml
   ```
5. 采集与校验：

   ```bash
   # 按现场 YAML 采集一条真实轨迹，写入 HDF5、quality.json、manifest.json 和校验和。
   UV_CACHE_DIR=/tmp/uv-cache uv run piper-collect collect --config configs/现场_piper.yaml
   # 只读校验刚完成的轨迹；使用 collect 输出的 trajectory_path 替换占位符。
   UV_CACHE_DIR=/tmp/uv-cache uv run piper-collect validate <trajectory_path>/trajectory.hdf5 \
     --config configs/现场_piper.yaml
   ```

Piper 适配器调用 `ConnectPort(piper_init=False)`，只接收 CAN 反馈，不发送运动、使能或 SDK 初始化查询指令。预检和采集均不会下发运动控制命令。CAN 接口初始化和设备权限仍须由现场人员按 Piper 官方文档处理。

## Pika Sense 遥操采集

遥操与采集保持独立，通过单独的会话编排层组合。先复制并填写被 Git 忽略的遥操配置，再校准和开始会话：

```bash
# 从模板生成本机 Pika Sense 遥操配置；新文件已被 Git 忽略。
cp configs/pika_sense_piper.example.yaml configs/现场_pika_sense.yaml
# 执行 Pika 基站强制校准；该命令会调用现场 survive-cli，不采集 HDF5。
UV_CACHE_DIR=/tmp/uv-cache uv run piper-collect calibrate-base \
  --teleop-config configs/现场_pika_sense.yaml --mode force
# 启动传感器、已有 ROS 遥操和采集会话；每条保存后按空格开始下一条。
UV_CACHE_DIR=/tmp/uv-cache uv run piper-collect teleop-session \
  --config configs/现场_piper.yaml \
  --teleop-config configs/现场_pika_sense.yaml \
  --repeat
```

会话所有阶段均按单个空格推进：校准完成、预检后的安全确认、Sense 双击并实际跟随、以及实体设备停止遥操之后。按 `q` 取消，不再输入 `yes/no`。每条轨迹默认保存，只有显式追加 `--on-complete delete` 才会删除。编排层启动 ROS 前会移除 uv 的 Python 环境变量，并检查关键 ROS 节点日志；`pika_ros` 的代码和 launch 文件不会被修改。完整流程见 [docs/2026-07-18_02_Pika_Sense遥操采集流程.md](docs/2026-07-18_02_Pika_Sense遥操采集流程.md)。

## 配置与扩展

- `modalities` 控制 RGB、深度、关节、TCP、夹爪位置是否采集与落盘。
- `session.pose_representation` 控制 TCP 落盘格式。Piper 现场配置使用原生 `xyz_rxryrz`（`m + rad`）；`xyz_xyzw` 仅保留给已有数据和其他设备的兼容场景。
- `acquisition.robot_hz` 和 `acquisition.camera_rig_hz` 分别控制原始状态采样及相机对齐时间轴。
- `cameras` 是可扩展列表；每台相机都通过 `driver` 注册到设备工厂。
- `robot.driver` 目前支持 `piper` 和 `mock`。接入新机械臂时实现 `RobotDevice`，并保证向核心层提供标准单位的 `RobotState`。
- `robot.initial_pose` 仅供 `teleop-session` 在已有 ROS 遥操启动前使用；采集与 `preflight` 仍严格只读。TCP 目标使用与采集一致的物理 TCP，控制时会反算 `tool_offset_m` 后下发 Piper 原生末端坐标。
- 保持 `session.format_version`、字段名、单位或编码发生不兼容变化时，必须提升格式版本，并同步更新 `quality.py`。

文档按日期和同日序号排序；交接文档不参与排序。详细边界、数据流和现场接入项见 [docs/2026-07-18_01_架构与数据契约.md](docs/2026-07-18_01_架构与数据契约.md)、[docs/2026-07-18_02_Pika_Sense遥操采集流程.md](docs/2026-07-18_02_Pika_Sense遥操采集流程.md) 和 [docs/项目交接.md](docs/项目交接.md)。
