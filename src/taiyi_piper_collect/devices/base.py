"""设备边界：核心采集器只依赖这些抽象接口。"""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..models import CameraCalibration, CameraFrame, RobotState


class CameraDevice(ABC):
    @abstractmethod
    def start(self, capture_depth: bool) -> None: ...

    @abstractmethod
    def read(self) -> CameraFrame: ...

    @abstractmethod
    def calibration(self) -> CameraCalibration: ...

    @abstractmethod
    def stop(self) -> None: ...

class RobotDevice(ABC):
    @abstractmethod
    def start(self) -> None: ...

    @abstractmethod
    def read(self) -> RobotState: ...

    @abstractmethod
    def stop(self) -> None: ...


class GripperDevice(ABC):
    @abstractmethod
    def start(self) -> None: ...

    @abstractmethod
    def read_position(self) -> float: ...

    @abstractmethod
    def stop(self) -> None: ...
