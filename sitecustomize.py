"""确保在任何脚本启动时 distutils.version 已被导入，避免新版本 setuptools 清理后的属性缺失。"""

import importlib
import logging

try:
    import distutils
except Exception:  # pragma: no cover - 启动时导入失败则忽略
    distutils = None

if distutils is not None and not hasattr(distutils, "version"):
    try:
        importlib.import_module("distutils.version")
    except Exception as exc:  # pragma: no cover - 若导入失败，仅记录一次日志
        logging.getLogger(__name__).debug(
            "无法预加载 distutils.version，原始异常：%s", exc
        )
