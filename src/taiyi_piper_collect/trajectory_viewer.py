"""本地只读轨迹查看器。

不额外引入 Web 框架：现场环境只需已有的 ``h5py`` 和 ``numpy``，即可在
浏览器中检查相机帧、对齐后的机器人状态和质量报告。
"""

from __future__ import annotations

from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import h5py
import numpy as np

from .errors import CollectionError


_MAX_CHART_POINTS = 2_000


@dataclass(frozen=True)
class TrajectoryViewer:
    """限制在一个数据根目录内读取已完成的轨迹。"""

    root: str | Path

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", Path(self.root).expanduser().resolve())
        if not self.root.is_dir():
            raise ValueError(f"轨迹数据根目录不存在或不是目录：{self.root}")

    def list_trajectories(self) -> list[dict[str, Any]]:
        """返回可打开轨迹，按最近修改时间倒序，且忽略未完成的 partial 文件。"""

        paths = sorted(
            (path for path in self.root.rglob("trajectory.hdf5") if path.is_file()),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        return [self._trajectory_summary(path) for path in paths]

    def read_trajectory(self, relative_path: str) -> dict[str, Any]:
        """读取界面所需的对齐时序数据，不对原 HDF5 作任何写入。"""

        path = self._resolve_trajectory(relative_path)
        with h5py.File(path, "r") as file:
            metadata = self._metadata(file)
            observations = _require_group(file, "camera_observations")
            timestamps = _float_list(observations["timestamp"][:])
            is_intervene = _bool_list(observations["is_intervene"][:]) if "is_intervene" in observations else []
            cameras = sorted(observations.get("color_images", {}).keys())
            puppet = file.get("puppet")
            joints = _series_data(puppet, "arm_single_position_align")
            tcp = _series_data(puppet, "end_effector_single_pose_align")
            gripper = _series_data(puppet, "end_effector_single_position_align")

        return {
            "path": path.relative_to(self.root).as_posix(),
            "metadata": metadata,
            "frame_count": len(timestamps),
            "timestamps": timestamps,
            "cameras": cameras,
            "is_intervene": is_intervene,
            "joint_positions": joints,
            "tcp_pose": tcp,
            "gripper_position": gripper,
            "quality": _read_json(path.with_name("quality.json")),
        }

    def read_frame(self, relative_path: str, camera: str, index: int) -> bytes:
        """读取一帧 JPEG 原始字节，避免把整段视频传到浏览器。"""

        path = self._resolve_trajectory(relative_path)
        with h5py.File(path, "r") as file:
            observations = _require_group(file, "camera_observations")
            colors = _require_group(observations, "color_images")
            if camera not in colors:
                raise ValueError(f"轨迹中不存在 RGB 相机：{camera}")
            images = colors[camera]
            if index < 0 or index >= len(images):
                raise ValueError(f"帧序号超出范围：{index}，有效范围为 0 到 {max(0, len(images) - 1)}。")
            return bytes(np.asarray(images[index], dtype=np.uint8))

    def _resolve_trajectory(self, relative_path: str) -> Path:
        candidate = (self.root / relative_path).resolve()
        try:
            candidate.relative_to(self.root)
        except ValueError as error:
            raise ValueError("轨迹路径必须位于指定数据根目录内。") from error
        if candidate.name != "trajectory.hdf5" or not candidate.is_file():
            raise ValueError("未找到已完成的 trajectory.hdf5。")
        return candidate

    def _trajectory_summary(self, path: Path) -> dict[str, Any]:
        metadata: dict[str, str] = {}
        frame_count = 0
        try:
            with h5py.File(path, "r") as file:
                metadata = self._metadata(file)
                frame_count = int(file["camera_observations/timestamp"].shape[0])
        except (KeyError, OSError, ValueError) as error:
            return {
                "path": path.relative_to(self.root).as_posix(),
                "frame_count": 0,
                "modified_at": int(path.stat().st_mtime),
                "error": f"无法读取 HDF5：{error}",
            }
        quality = _read_json(path.with_name("quality.json"))
        return {
            "path": path.relative_to(self.root).as_posix(),
            "frame_count": frame_count,
            "modified_at": int(path.stat().st_mtime),
            "instruction": metadata.get("language_instruction", ""),
            "collection_time": metadata.get("collection_time", ""),
            "quality_result": quality.get("result") if isinstance(quality, dict) else None,
        }

    @staticmethod
    def _metadata(file: h5py.File) -> dict[str, str]:
        metadata = file.get("metadata")
        if metadata is None:
            return {}
        return {
            name: _text(dataset[()])
            for name, dataset in metadata.items()
            if isinstance(dataset, h5py.Dataset) and dataset.ndim == 0
        }


def run_trajectory_viewer(root: str | Path, host: str = "127.0.0.1", port: int = 8765) -> None:
    """启动本机只读 HTTP 服务，直到操作者在终端按 Ctrl-C。"""

    try:
        viewer = TrajectoryViewer(Path(root))
        server = ThreadingHTTPServer((host, port), _handler_for(viewer))
    except (OSError, ValueError) as error:
        raise CollectionError(f"无法启动轨迹查看器：{error}") from error
    server.daemon_threads = True
    address = f"http://{host}:{server.server_port}"
    print(f"轨迹查看器已启动（只读）：{address}")
    print(f"数据根目录：{viewer.root}")
    print("在浏览器打开上述地址；按 Ctrl-C 停止服务。")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n轨迹查看器已停止。")
    finally:
        server.server_close()


def _handler_for(viewer: TrajectoryViewer) -> type[BaseHTTPRequestHandler]:
    class TrajectoryViewerHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
            parsed = urlparse(self.path)
            try:
                if parsed.path == "/":
                    self._send_bytes(_PAGE.encode("utf-8"), "text/html; charset=utf-8")
                    return
                query = parse_qs(parsed.query)
                if parsed.path == "/api/trajectories":
                    self._send_json({"root": str(viewer.root), "trajectories": viewer.list_trajectories()})
                    return
                if parsed.path == "/api/trajectory":
                    self._send_json(viewer.read_trajectory(_query_value(query, "path")))
                    return
                if parsed.path == "/api/frame":
                    image = viewer.read_frame(
                        _query_value(query, "path"),
                        _query_value(query, "camera"),
                        _query_index(query),
                    )
                    self._send_bytes(image, "image/jpeg", cache_control="no-store")
                    return
                self.send_error(HTTPStatus.NOT_FOUND, "不存在的路径")
            except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
                self._send_json({"error": str(error)}, status=HTTPStatus.BAD_REQUEST)

        def log_message(self, _format: str, *_args: object) -> None:
            """浏览器拖动时间轴会频繁请求图像，默认访问日志没有操作价值。"""

        def _send_json(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
            self._send_bytes(
                json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8"),
                "application/json; charset=utf-8",
                status=status,
            )

        def _send_bytes(
            self,
            body: bytes,
            content_type: str,
            status: HTTPStatus = HTTPStatus.OK,
            cache_control: str = "no-cache",
        ) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", cache_control)
            self.end_headers()
            self.wfile.write(body)

    return TrajectoryViewerHandler


def _require_group(parent: h5py.Group | h5py.File, name: str) -> h5py.Group:
    group = parent.get(name)
    if not isinstance(group, h5py.Group):
        raise ValueError(f"HDF5 缺少 /{name}。")
    return group


def _series_data(puppet: h5py.Group | None, name: str) -> dict[str, Any] | None:
    if puppet is None or name not in puppet:
        return None
    group = puppet[name]
    if not isinstance(group, h5py.Group) or "data" not in group:
        return None
    data = np.asarray(group["data"][:], dtype=np.float64)
    timestamps = np.asarray(group["timestamp"][:], dtype=np.float64) if "timestamp" in group else np.array([])
    if data.ndim != 2:
        return None
    indices = _chart_indices(len(data))
    return {
        # 全量 values 供滑块准确显示当前帧；曲线单独抽样以避免长轨迹卡顿。
        "timestamps": _float_list(timestamps),
        "values": data.tolist(),
        "chart_values": data[indices].tolist(),
        "width": int(data.shape[1]),
        "sample_count": int(len(data)),
    }


def _chart_indices(length: int) -> np.ndarray:
    if length <= _MAX_CHART_POINTS:
        return np.arange(length, dtype=np.int64)
    return np.unique(np.linspace(0, length - 1, _MAX_CHART_POINTS, dtype=np.int64))


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _query_value(query: dict[str, list[str]], name: str) -> str:
    values = query.get(name, [])
    if len(values) != 1 or not values[0]:
        raise ValueError(f"缺少参数：{name}")
    return values[0]


def _query_index(query: dict[str, list[str]]) -> int:
    raw = _query_value(query, "index")
    try:
        return int(raw)
    except ValueError as error:
        raise ValueError(f"帧序号必须是整数：{raw}") from error


def _text(value: Any) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def _float_list(values: np.ndarray) -> list[float]:
    return [float(value) for value in values]


def _bool_list(values: np.ndarray) -> list[bool]:
    return [bool(value) for value in values]


_PAGE = r"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Piper 轨迹查看器</title>
<style>
:root { color-scheme: light; font-family: Arial, "Noto Sans CJK SC", sans-serif; color: #18212b; background: #e8eef2; }
* { box-sizing: border-box; }
body { margin: 0; min-width: 320px; }
button, select, input { font: inherit; }
button { cursor: pointer; }
header { background: #18212b; color: #ffffff; padding: 16px clamp(16px, 3vw, 40px); display: flex; align-items: center; justify-content: space-between; gap: 16px; }
h1 { margin: 0; font-size: 20px; font-weight: 700; }
.quiet { color: #60707d; }
header .quiet { color: #c9d5df; font-size: 13px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
main { max-width: 1580px; margin: 0 auto; padding: 20px clamp(16px, 3vw, 40px) 40px; }
.toolbar { display: grid; grid-template-columns: minmax(0, 1fr) auto; gap: 12px; align-items: center; margin-bottom: 16px; }
select { width: 100%; min-width: 0; height: 38px; background: #ffffff; border: 1px solid #aebbc5; border-radius: 4px; padding: 0 10px; color: #18212b; }
button { min-width: 38px; min-height: 38px; border: 1px solid #256e79; border-radius: 4px; padding: 0 13px; background: #0e7c86; color: #ffffff; }
button:hover { background: #086773; }
button:disabled { cursor: wait; opacity: .65; }
.status { min-height: 20px; margin: 0 0 16px; color: #a53737; font-size: 14px; }
.summary { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); border: 1px solid #c5d0d8; background: #ffffff; margin-bottom: 16px; }
.metric { min-width: 0; padding: 12px 14px; border-right: 1px solid #d8e0e5; }
.metric:last-child { border-right: 0; }
.metric-label { display: block; color: #60707d; font-size: 12px; margin-bottom: 5px; }
.metric-value { display: block; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; font-size: 15px; font-weight: 700; }
.workspace { display: grid; grid-template-columns: minmax(360px, 1.25fr) minmax(360px, .75fr); gap: 16px; }
.panel { min-width: 0; background: #ffffff; border: 1px solid #c5d0d8; }
.panel-header { min-height: 48px; padding: 10px 14px; border-bottom: 1px solid #d8e0e5; display: flex; justify-content: space-between; align-items: center; gap: 12px; }
h2 { margin: 0; font-size: 16px; }
.camera-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 1px; background: #c5d0d8; }
.camera-view { min-width: 0; background: #ffffff; }
.camera-name { padding: 8px 12px; color: #384b58; font-size: 13px; font-weight: 700; }
.image-stage { aspect-ratio: 4 / 3; background: #101720; display: grid; place-items: center; }
.image-stage img { display: block; width: 100%; height: 100%; object-fit: contain; }
.frame-controls { display: grid; grid-template-columns: auto minmax(120px, 1fr) auto; align-items: center; gap: 12px; padding: 14px; }
input[type="range"] { width: 100%; accent-color: #0e7c86; }
.frame-label { min-width: 110px; text-align: right; font-variant-numeric: tabular-nums; font-size: 13px; }
.values { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 0; border-top: 1px solid #d8e0e5; }
.value-group { padding: 13px 14px; min-width: 0; border-right: 1px solid #d8e0e5; }
.value-group:last-child { border-right: 0; }
.value-title { color: #60707d; font-size: 12px; margin-bottom: 8px; }
.numeric { margin: 0; line-height: 1.7; font-family: "SFMono-Regular", Consolas, monospace; font-size: 12px; overflow-wrap: anywhere; }
.right-column { display: grid; gap: 16px; align-content: start; }
.chart-wrap { padding: 12px 14px 14px; }
.chart-legend { display: flex; flex-wrap: wrap; gap: 7px 14px; min-height: 20px; margin: 0 0 9px; color: #384b58; font-size: 12px; }
.legend-item { display: inline-flex; align-items: center; gap: 5px; white-space: nowrap; }
.legend-swatch { display: inline-block; width: 16px; height: 3px; flex: 0 0 16px; }
canvas { display: block; width: 100%; height: 190px; background: #fbfcfd; border: 1px solid #d8e0e5; }
.detail-workspace { display: grid; grid-template-columns: 1fr; gap: 16px; margin-top: 16px; }
.detail-chart-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 12px; padding: 12px 14px 14px; }
.detail-chart { min-width: 0; margin: 0; }
.detail-chart figcaption { display: flex; align-items: center; gap: 6px; min-height: 24px; color: #384b58; font-size: 13px; font-weight: 700; }
.detail-chart canvas { height: 230px; }
.tcp-detail-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 12px; padding: 12px 14px 14px; }
.tcp-detail-grid .detail-chart canvas { height: 270px; }
.quality { padding: 12px 14px 14px; font-size: 14px; }
.quality-state { font-weight: 700; margin-bottom: 10px; }
.pass { color: #19723b; }
.fail { color: #a53737; }
.pending { color: #97680b; }
.quality ul { margin: 8px 0 0; padding-left: 20px; color: #384b58; }
.quality li { margin: 5px 0; }
.empty { padding: 44px 16px; text-align: center; color: #60707d; }
@media (max-width: 980px) { .workspace { grid-template-columns: 1fr; } }
@media (max-width: 620px) { header { align-items: flex-start; flex-direction: column; } .toolbar { grid-template-columns: 1fr; } .toolbar button { width: 100%; } .summary { grid-template-columns: repeat(2, minmax(0, 1fr)); } .metric:nth-child(2) { border-right: 0; } .metric:nth-child(-n+2) { border-bottom: 1px solid #d8e0e5; } .camera-grid, .detail-chart-grid, .tcp-detail-grid { grid-template-columns: 1fr; } .values { grid-template-columns: 1fr; } .value-group { border-right: 0; border-bottom: 1px solid #d8e0e5; } .value-group:last-child { border-bottom: 0; } .frame-controls { grid-template-columns: 1fr; } .frame-label { text-align: left; } }
</style>
</head>
<body>
<header><h1>Piper 轨迹查看器</h1><span id="root" class="quiet"></span></header>
<main>
  <div class="toolbar"><select id="trajectory" aria-label="选择轨迹"></select><button id="refresh" type="button">刷新列表</button></div>
  <p id="status" class="status" role="status"></p>
  <section id="content" hidden>
    <div class="summary">
      <div class="metric"><span class="metric-label">任务</span><span id="instruction" class="metric-value">-</span></div>
      <div class="metric"><span class="metric-label">采集时间</span><span id="collection-time" class="metric-value">-</span></div>
      <div class="metric"><span class="metric-label">帧数</span><span id="frame-count" class="metric-value">-</span></div>
      <div class="metric"><span class="metric-label">位姿格式</span><span id="pose-format" class="metric-value">-</span></div>
    </div>
    <div class="workspace">
      <section class="panel">
        <div class="panel-header"><h2>相机帧</h2></div>
        <div id="camera-views" class="camera-grid"></div>
        <div class="frame-controls"><button id="previous" type="button" title="上一帧" aria-label="上一帧">上一帧</button><input id="timeline" type="range" min="0" value="0"><button id="next" type="button" title="下一帧" aria-label="下一帧">下一帧</button><span id="frame-label" class="frame-label">-</span></div>
        <div class="values">
          <div class="value-group"><div class="value-title">关节角 (rad)</div><p id="joints" class="numeric">-</p></div>
          <div class="value-group"><div id="tcp-title" class="value-title">TCP</div><p id="tcp" class="numeric">-</p><div class="value-title">夹爪行程 (m)</div><p id="gripper" class="numeric">-</p></div>
        </div>
      </section>
      <aside class="right-column">
        <section class="panel"><div class="panel-header"><h2>关节角时序</h2></div><div class="chart-wrap"><div id="joint-chart-legend" class="chart-legend"></div><canvas id="joint-chart" aria-label="关节角时序图"></canvas></div></section>
        <section class="panel"><div class="panel-header"><h2>TCP 时序</h2></div><div class="chart-wrap"><div id="tcp-chart-legend" class="chart-legend"></div><canvas id="tcp-chart" aria-label="TCP 时序图"></canvas></div></section>
        <section class="panel"><div class="panel-header"><h2>质量报告</h2></div><div id="quality" class="quality"></div></section>
      </aside>
    </div>
    <div class="detail-workspace">
      <section class="panel"><div class="panel-header"><h2>关节角详细时序</h2><span class="quiet">每个关节独立纵轴</span></div><div id="joint-detail-charts" class="detail-chart-grid"></div></section>
      <section class="panel"><div class="panel-header"><h2>TCP 详细时序</h2><span class="quiet">位置与姿态分开显示</span></div><div class="tcp-detail-grid"><figure class="detail-chart"><figcaption>TCP 位置 xyz (m)</figcaption><div id="tcp-position-legend" class="chart-legend"></div><canvas id="tcp-position-chart" data-series="tcp_pose" data-columns="0,1,2" data-unit="m" aria-label="TCP 位置 xyz 时序图"></canvas></figure><figure class="detail-chart"><figcaption id="tcp-orientation-title">TCP 姿态 rx ry rz (rad)</figcaption><div id="tcp-orientation-legend" class="chart-legend"></div><canvas id="tcp-orientation-chart" data-series="tcp_pose" data-columns="3,4,5" aria-label="TCP 姿态时序图"></canvas></figure></div></section>
    </div>
  </section>
  <section id="empty" class="panel empty" hidden>数据根目录内没有已完成的 trajectory.hdf5。</section>
</main>
<script>
const state = { list: [], trajectory: null, index: 0 };
const colors = ["#0e7c86", "#ca5a29", "#447bc4", "#8a4ea6", "#71813a", "#bb3f63", "#677784"];
const jointLabels = ["J1 (rad)", "J2 (rad)", "J3 (rad)", "J4 (rad)", "J5 (rad)", "J6 (rad)"];
const byId = (id) => document.getElementById(id);

async function request(url) {
  const response = await fetch(url, {cache: "no-store"});
  if (!response.ok) { const body = await response.json().catch(() => ({})); throw new Error(body.error || `请求失败 (${response.status})`); }
  return response.json();
}
function setStatus(message = "") { byId("status").textContent = message; }
function format(value, digits = 4) { return Number.isFinite(value) ? value.toFixed(digits) : "-"; }
function elapsedAt(index) { const values = state.trajectory.timestamps; return values.length ? values[index] - values[0] : 0; }
function seriesValue(series, index) {
  if (!series || !series.values.length) return null;
  const count = state.trajectory.frame_count;
  const position = count <= 1 ? 0 : Math.round(index * (series.values.length - 1) / (count - 1));
  return series.values[Math.max(0, Math.min(series.values.length - 1, position))];
}
function tcpLabels(metadata) {
  return metadata.pose_representation === "xyz_rxryrz"
    ? ["x (m)", "y (m)", "z (m)", "rx (rad)", "ry (rad)", "rz (rad)"]
    : ["x (m)", "y (m)", "z (m)", "qx", "qy", "qz", "qw"];
}
function tcpValueLabels(metadata) { return tcpLabels(metadata).map((label) => label.split(" ")[0]); }
function renderLegend(id, labels, columns) {
  const node = byId(id); node.replaceChildren();
  for (const [index, column] of columns.entries()) {
    const item = document.createElement("span"); item.className = "legend-item";
    const swatch = document.createElement("span"); swatch.className = "legend-swatch"; swatch.style.background = colors[column % colors.length];
    const label = document.createElement("span"); label.textContent = labels[index] || `value ${column}`;
    item.append(swatch, label); node.append(item);
  }
}

async function refreshList() {
  const button = byId("refresh"); button.disabled = true; setStatus("");
  try {
    const payload = await request("/api/trajectories");
    state.list = payload.trajectories; byId("root").textContent = payload.root;
    const select = byId("trajectory"); select.replaceChildren();
    for (const item of state.list) {
      const option = document.createElement("option"); option.value = item.path;
      const quality = item.quality_result ? ` | 质检 ${item.quality_result}` : "";
      option.textContent = `${item.collection_time || item.path} | ${item.frame_count} 帧${quality}`;
      select.append(option);
    }
    byId("empty").hidden = state.list.length > 0; byId("content").hidden = state.list.length === 0;
    if (state.list.length) await loadTrajectory(state.list[0].path);
  } catch (error) { setStatus(error.message); }
  finally { button.disabled = false; }
}

async function loadTrajectory(path) {
  setStatus("正在读取轨迹...");
  try {
    state.trajectory = await request(`/api/trajectory?path=${encodeURIComponent(path)}`);
    state.index = 0;
    byId("trajectory").value = path; byId("content").hidden = false; byId("empty").hidden = true;
    renderTrajectory(); setStatus("");
  } catch (error) { setStatus(error.message); }
}

function renderTrajectory() {
  const data = state.trajectory; const metadata = data.metadata;
  byId("instruction").textContent = metadata.language_instruction || "-";
  byId("collection-time").textContent = metadata.collection_time || "-";
  byId("frame-count").textContent = String(data.frame_count);
  byId("pose-format").textContent = metadata.pose_representation || "-";
  byId("tcp-title").textContent = `TCP ${metadata.pose_representation === "xyz_rxryrz" ? "(m, rad)" : "(m, xyzw)"}`;
  const slider = byId("timeline"); slider.max = String(Math.max(0, data.frame_count - 1)); slider.value = String(state.index);
  renderCameraViews(); renderQuality(data.quality); renderCharts();
  renderFrame();
}

function renderCharts() {
  const data = state.trajectory; const labels = tcpLabels(data.metadata); const jointColumns = [0, 1, 2, 3, 4, 5]; const tcpColumns = labels.map((_, index) => index);
  renderLegend("joint-chart-legend", jointLabels, jointColumns); renderLegend("tcp-chart-legend", labels, tcpColumns);
  drawChart(byId("joint-chart"), data.joint_positions, "rad", jointColumns);
  drawChart(byId("tcp-chart"), data.tcp_pose, data.metadata.pose_representation === "xyz_rxryrz" ? "m / rad" : "m / xyzw", tcpColumns);
  renderJointDetailCharts(data.joint_positions); renderTcpDetailCharts(data.tcp_pose, data.metadata);
}

function renderJointDetailCharts(series) {
  const host = byId("joint-detail-charts"); host.replaceChildren();
  if (!series || !series.values.length) { host.textContent = "未采集关节角。"; return; }
  for (const [index, label] of jointLabels.entries()) {
    const figure = document.createElement("figure"); figure.className = "detail-chart";
    const caption = document.createElement("figcaption"); const swatch = document.createElement("span"); swatch.className = "legend-swatch"; swatch.style.background = colors[index % colors.length]; caption.append(swatch, document.createTextNode(label));
    const canvas = document.createElement("canvas"); canvas.dataset.series = "joint_positions"; canvas.dataset.columns = String(index); canvas.dataset.unit = "rad"; canvas.setAttribute("aria-label", `${label} 时序图`);
    figure.append(caption, canvas); host.append(figure); drawChart(canvas, series, "rad", [index]);
  }
}

function renderTcpDetailCharts(series, metadata) {
  const labels = tcpLabels(metadata); const orientationColumns = labels.slice(3).map((_, index) => index + 3);
  byId("tcp-orientation-title").textContent = metadata.pose_representation === "xyz_rxryrz" ? "TCP 姿态 rx ry rz (rad)" : "TCP 姿态 xyzw";
  renderLegend("tcp-position-legend", labels.slice(0, 3), [0, 1, 2]); renderLegend("tcp-orientation-legend", labels.slice(3), orientationColumns);
  const orientationCanvas = byId("tcp-orientation-chart"); orientationCanvas.dataset.columns = orientationColumns.join(","); orientationCanvas.dataset.unit = metadata.pose_representation === "xyz_rxryrz" ? "rad" : "xyzw";
  drawChart(byId("tcp-position-chart"), series, "m", [0, 1, 2]); drawChart(orientationCanvas, series, orientationCanvas.dataset.unit, orientationColumns);
}

function renderCameraViews() {
  const views = byId("camera-views"); views.replaceChildren();
  for (const camera of state.trajectory.cameras) {
    const view = document.createElement("section"); view.className = "camera-view";
    const name = document.createElement("div"); name.className = "camera-name"; name.textContent = camera;
    const stage = document.createElement("div"); stage.className = "image-stage";
    const image = document.createElement("img"); image.className = "camera-image"; image.dataset.camera = camera; image.alt = `${camera} 当前帧`;
    stage.append(image); view.append(name, stage); views.append(view);
  }
}

function renderFrame() {
  const data = state.trajectory; if (!data) return;
  const index = state.index; const joint = seriesValue(data.joint_positions, index); const tcp = seriesValue(data.tcp_pose, index); const gripper = seriesValue(data.gripper_position, index);
  byId("timeline").value = String(index);
  byId("frame-label").textContent = `${index + 1} / ${data.frame_count} | ${format(elapsedAt(index), 3)} s${data.is_intervene[index] ? " | intervene" : ""}`;
  byId("joints").textContent = joint ? joint.map((value, i) => `J${i + 1}: ${format(value)}`).join("\n") : "未采集";
  byId("tcp").textContent = tcp ? tcp.map((value, i) => `${tcpValueLabels(data.metadata)[i]}: ${format(value)}`).join("\n") : "未采集";
  byId("gripper").textContent = gripper ? format(gripper[0], 6) : "未采集";
  for (const image of document.querySelectorAll(".camera-image")) {
    const camera = image.dataset.camera;
    if (!camera || !data.frame_count) { image.removeAttribute("src"); continue; }
    image.src = `/api/frame?path=${encodeURIComponent(data.path)}&camera=${encodeURIComponent(camera)}&index=${index}&v=${Date.now()}`;
  }
}

function renderQuality(quality) {
  const node = byId("quality"); node.replaceChildren();
  if (!quality) { node.textContent = "未找到 quality.json。"; return; }
  const stateNode = document.createElement("div"); const result = quality.result || "unknown";
  stateNode.className = `quality-state ${result === "pass" ? "pass" : result === "fail" ? "fail" : "pending"}`; stateNode.textContent = `自动检查：${result}`; node.append(stateNode);
  const metrics = quality.metrics || {}; const metric = document.createElement("div"); metric.className = "quiet";
  metric.textContent = `轨迹长度 ${metrics.trajectory_length ?? "-"}，相机帧 ${metrics.camera_frames ?? "-"}`; node.append(metric);
  for (const [title, values] of [["错误", quality.errors || []], ["告警", quality.warnings || []]]) {
    if (!values.length) continue;
    const heading = document.createElement("div"); heading.style.marginTop = "10px"; heading.textContent = title; node.append(heading);
    const list = document.createElement("ul"); for (const value of values) { const item = document.createElement("li"); item.textContent = value; list.append(item); } node.append(list);
  }
}

function drawChart(canvas, series, unit, columns) {
  const context = canvas.getContext("2d"); const width = Math.max(1, Math.round(canvas.clientWidth * devicePixelRatio)); const height = Math.max(1, Math.round(canvas.clientHeight * devicePixelRatio));
  if (canvas.width !== width || canvas.height !== height) { canvas.width = width; canvas.height = height; }
  context.setTransform(devicePixelRatio, 0, 0, devicePixelRatio, 0, 0); const cssWidth = canvas.clientWidth; const cssHeight = canvas.clientHeight;
  context.clearRect(0, 0, cssWidth, cssHeight); context.fillStyle = "#fbfcfd"; context.fillRect(0, 0, cssWidth, cssHeight);
  if (!series || !series.values.length) { context.fillStyle = "#60707d"; context.font = "13px Arial"; context.fillText("未采集", 12, 24); return; }
  const padding = {left: 46, right: 12, top: 20, bottom: 28}; const plotWidth = cssWidth - padding.left - padding.right; const plotHeight = cssHeight - padding.top - padding.bottom;
  const values = series.chart_values || series.values; const visibleColumns = columns || Array.from({length: series.width}, (_, index) => index); const flat = values.flatMap((row) => visibleColumns.map((column) => row[column])); let minimum = Math.min(...flat); let maximum = Math.max(...flat);
  if (minimum === maximum) { minimum -= 1; maximum += 1; } const margin = (maximum - minimum) * .08; minimum -= margin; maximum += margin;
  context.strokeStyle = "#d8e0e5"; context.lineWidth = 1; context.fillStyle = "#60707d"; context.font = "11px Arial";
  for (let step = 0; step <= 4; step += 1) { const y = padding.top + plotHeight * step / 4; context.beginPath(); context.moveTo(padding.left, y); context.lineTo(cssWidth - padding.right, y); context.stroke(); const value = maximum - (maximum - minimum) * step / 4; context.fillText(format(value, 2), 3, y + 4); }
  context.fillText("0 s", padding.left, cssHeight - 8); context.textAlign = "right"; const duration = series.timestamps?.length ? series.timestamps[series.timestamps.length - 1] - series.timestamps[0] : 0; context.fillText(`${format(duration, 2)} s | ${unit}`, cssWidth - padding.right, cssHeight - 8); context.textAlign = "left";
  for (const column of visibleColumns) {
    context.strokeStyle = colors[column % colors.length]; context.lineWidth = 1.3; context.beginPath();
    values.forEach((row, index) => { const x = padding.left + plotWidth * (values.length <= 1 ? 0 : index / (values.length - 1)); const y = padding.top + (maximum - row[column]) / (maximum - minimum) * plotHeight; if (index === 0) context.moveTo(x, y); else context.lineTo(x, y); }); context.stroke();
  }
}

byId("refresh").addEventListener("click", refreshList);
byId("trajectory").addEventListener("change", (event) => loadTrajectory(event.target.value));
byId("timeline").addEventListener("input", (event) => { state.index = Number(event.target.value); renderFrame(); });
byId("previous").addEventListener("click", () => { state.index = Math.max(0, state.index - 1); renderFrame(); });
byId("next").addEventListener("click", () => { state.index = Math.min(state.trajectory.frame_count - 1, state.index + 1); renderFrame(); });
window.addEventListener("resize", () => { if (!state.trajectory) return; renderCharts(); for (const canvas of document.querySelectorAll("canvas[data-series]")) { const columns = canvas.dataset.columns.split(",").map(Number); drawChart(canvas, state.trajectory[canvas.dataset.series], canvas.dataset.unit, columns); } });
window.addEventListener("keydown", (event) => { if (!state.trajectory || ["INPUT", "SELECT"].includes(document.activeElement.tagName)) return; if (event.key === "ArrowLeft") { event.preventDefault(); state.index = Math.max(0, state.index - 1); renderFrame(); } if (event.key === "ArrowRight") { event.preventDefault(); state.index = Math.min(state.trajectory.frame_count - 1, state.index + 1); renderFrame(); } });
refreshList();
</script>
</body>
</html>"""
