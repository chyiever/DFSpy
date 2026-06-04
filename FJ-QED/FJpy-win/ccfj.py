"""FJpy-win 对外入口模块。

该文件保持与参考项目一致的 `import ccfj` 使用方式，
便于 notebook、脚本和上层调用代码在 Windows 环境中直接替换运行。
"""

# 统一从核心实现导出全部公开函数，保持原项目接口风格。
from src.ccfj_core import *  # noqa: F401,F403
