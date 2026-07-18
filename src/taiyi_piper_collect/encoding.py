"""图像编码：在采集进程完成编码，降低跨进程队列的内存占用。"""

from __future__ import annotations

import numpy as np

from .errors import CollectionError, HardwareDependencyError
from .models import CameraFrame, EncodedCameraFrame


def _cv2():
    try:
        import cv2
    except ImportError as error:
        raise HardwareDependencyError("缺少 OpenCV；请执行 uv sync。") from error
    return cv2


def encode_frame(frame: CameraFrame, jpeg_quality: int, capture_depth: bool) -> EncodedCameraFrame:
    """RGB/BGR 编为 JPEG；深度截断后以 uint16 PNG 无损编码。"""

    cv2 = _cv2()
    color_encoded: bytes | None = None
    if frame.color is not None:
        success, encoded = cv2.imencode(".jpg", frame.color, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
        if not success:
            raise CollectionError("RGB JPEG 编码失败。")
        color_encoded = encoded.tobytes()
    depth_encoded: bytes | None = None
    if capture_depth and frame.depth is not None:
        depth_uint16 = np.clip(frame.depth, 0, 65535).astype(np.uint16, copy=False)
        success, encoded = cv2.imencode(".png", depth_uint16)
        if not success:
            raise CollectionError("深度 PNG 编码失败。")
        depth_encoded = encoded.tobytes()
    return EncodedCameraFrame(color=color_encoded, depth=depth_encoded)
