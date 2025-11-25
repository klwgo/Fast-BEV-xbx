"""为旧版依赖补齐 Python 3.12 移除的接口，避免 mmcv/mmdet 导入失败。"""

import importlib
import logging
import sys


def _ensure_distutils_version():
    try:
        import distutils  # noqa: F401
    except Exception:  # pragma: no cover - 启动时导入失败则忽略
        distutils = None
    else:
        if hasattr(distutils, "version"):
            return
    try:
        version_mod = importlib.import_module("distutils.version")
    except Exception:
        try:
            version_mod = importlib.import_module("setuptools._distutils.version")
        except Exception as exc:  # pragma: no cover - 若导入失败，仅记录一次日志
            logging.getLogger(__name__).debug(
                "无法预加载 distutils.version，原始异常：%s", exc
            )
            return
        sys.modules.setdefault("distutils.version", version_mod)
    import distutils  # noqa: E402

    distutils.version = version_mod  # pragma: no cover


def _ensure_pkgutil_legacy_attrs():
    try:
        import pkgutil
    except Exception:  # pragma: no cover - 若 pkgutil 缺失则忽略
        return

    def _define(name):
        if hasattr(pkgutil, name):
            return

        class _Legacy:  # noqa: D401 - 仅用于兼容旧接口
            """占位符，满足 pkg_resources 对旧接口的引用。"""

            pass

        setattr(pkgutil, name, _Legacy)

    for attr in ("ImpImporter", "ImpLoader", "ImpFinder"):
        _define(attr)


def _patch_filefinder():
    try:
        from importlib.machinery import FileFinder
    except Exception:  # pragma: no cover - 运行环境不存在该类型
        return

    if hasattr(FileFinder, "find_module"):
        return

    def _legacy_find_module(self, fullname):
        spec = self.find_spec(fullname)
        return None if spec is None else spec.loader

    FileFinder.find_module = _legacy_find_module  # type: ignore[attr-defined]


def _ensure_numpy_core_alias():
    try:  # pragma: no cover - 启动阶段的兼容处理
        import numpy as _np  # noqa: F401
    except Exception:  # noqa: E722 - 兼容性兜底
        return
    if hasattr(_np, "_core") or not hasattr(_np, "core"):
        return
    import types

    sys.modules.setdefault("numpy._core", _np.core)
    _np._core = _np.core
    for attr in dir(_np.core):
        obj = getattr(_np.core, attr)
        if isinstance(obj, types.ModuleType) and obj.__package__ and obj.__package__.startswith("numpy.core"):
            sys.modules[f"numpy._core.{attr}"] = obj


_ensure_distutils_version()
_ensure_pkgutil_legacy_attrs()
_patch_filefinder()
_ensure_numpy_core_alias()
