"""确保在任何脚本启动时 distutils.version 已被导入，避免新版本 setuptools 清理后的属性缺失。"""

import importlib
import logging
import sys

try:
    import distutils
except Exception:  # pragma: no cover - 启动时导入失败则忽略
    distutils = None

if distutils is not None and not hasattr(distutils, "version"):
    try:
        version_mod = importlib.import_module("distutils.version")
        distutils.version = version_mod  # pragma: no cover
    except Exception:
        try:
            version_mod = importlib.import_module("setuptools._distutils.version")
            sys.modules["distutils.version"] = version_mod
            distutils.version = version_mod  # pragma: no cover
        except Exception as exc:  # pragma: no cover - 若导入失败，仅记录一次日志
            logging.getLogger(__name__).debug(
                "无法预加载 distutils.version，原始异常：%s", exc
            )

# ---------------------------------------------------------------------- #
# 兼容 numpy 2.0 生成的 pickle：为旧版本 numpy 补上 numpy._core 别名
# ---------------------------------------------------------------------- #
try:  # pragma: no cover - 启动阶段的兼容处理
    import numpy as _np  # noqa: F401
    if not hasattr(_np, "_core") and hasattr(_np, "core"):
        import types
        import sys

        sys.modules.setdefault("numpy._core", _np.core)
        _np._core = _np.core
except Exception:  # noqa: E722 - 兼容性兜底
    pass
