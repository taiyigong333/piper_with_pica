#!/usr/bin/env python3
"""兼容直接运行的 Piper 状态读取脚本。"""

from taiyi_piper_collect.piper_state import main


if __name__ == "__main__":
    raise SystemExit(main())
