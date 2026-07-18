"""采集系统的可识别错误类型。"""


class CollectionError(RuntimeError):
    """采集期间发生的错误，当前轨迹不能被当作完整数据使用。"""


class ConfigurationError(ValueError):
    """YAML 配置不完整、不合法或存在不安全组合。"""


class DeviceError(RuntimeError):
    """硬件连接、读数或单位约定异常。"""


class HardwareDependencyError(ImportError):
    """真实硬件所需的可选 Python 依赖尚未安装。"""
