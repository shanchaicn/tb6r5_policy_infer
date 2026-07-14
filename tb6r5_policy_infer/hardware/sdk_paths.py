"""Resolve bundled TB6-R5 SDK paths for policy inference.

The compiled extension modules (rpc.so / topic.so) are ABI-specific per CPython
minor version, so we ship one vendor tree per supported Python version and pick
the right one at runtime:

  - Python 3.10  -> vendor/          (send_commend_py + get_status_py)
  - Python 3.12  -> vendor_py312/    (pybind11 rebuild)

Everything else in the trees (pure-Python helpers, libzmq/libprotobuf) is
identical; only rpc.so / topic.so differ. Both trees ship x86-64 and aarch64
(ARM/Jetson Orin) binaries. Override the whole tree with TB6R5_DEPS_ROOT.
"""

from __future__ import annotations

import os
import platform
import sys
from pathlib import Path

_PACKAGE_ROOT = Path(__file__).resolve().parent.parent
_VENDOR_PY310 = _PACKAGE_ROOT / "vendor"
_VENDOR_PY312 = _PACKAGE_ROOT / "vendor_py312"


def python_tag() -> str:
    return f"py{sys.version_info.major}{sys.version_info.minor}"


def _preferred_vendor() -> Path:
    """Pick the bundled vendor tree matching the running CPython version."""
    if sys.version_info[:2] == (3, 12):
        return _VENDOR_PY312
    # 3.10 is the default/reference build; other versions fall back to it and
    # will surface a clear ABI error at import time if incompatible.
    return _VENDOR_PY310


def _platform_subdir() -> str:
    system = platform.system().lower()
    if system == "linux":
        machine = platform.machine().lower()
        return "arm" if machine in ("aarch64", "arm64") else "x86"
    if system == "windows":
        return "win"
    raise RuntimeError(f"Unsupported OS: {system}")


def _is_valid_deps_root(root: Path) -> bool:
    return (root / "send_commend_py/rpc_py_all").is_dir() and (root / "get_status_py/topic_all_py").is_dir()


def dependencies_root() -> Path:
    """Directory containing send_commend_py/ and get_status_py/ SDK trees."""
    env = os.environ.get("TB6R5_DEPS_ROOT")
    if env:
        root = Path(env).expanduser().resolve()
        if not root.is_dir():
            raise RuntimeError(f"TB6R5_DEPS_ROOT is not a directory: {root}")
        if not _is_valid_deps_root(root):
            raise RuntimeError(f"TB6R5_DEPS_ROOT missing send_commend_py/ or get_status_py/: {root}")
        return root

    preferred = _preferred_vendor()
    if _is_valid_deps_root(preferred):
        return preferred
    # Fall back to the other bundled tree if the preferred one is absent.
    for alt in (_VENDOR_PY310, _VENDOR_PY312):
        if _is_valid_deps_root(alt):
            return alt

    for parent in _PACKAGE_ROOT.parents:
        legacy = parent / "dependencies"
        if (legacy / "send_commend_py-master/rpc_py_all").is_dir() and (legacy / "get_status_py/topic_all_py").is_dir():
            return legacy
        if _is_valid_deps_root(legacy):
            return legacy

    raise RuntimeError(
        "Cannot find TB6-R5 SDK (send_commend_py, get_status_py).\n"
        f"Bundled vendor missing at {preferred}.\n"
        "Set TB6R5_DEPS_ROOT to override, or reinstall tb6r5_policy_infer."
    )


def rpc_py_all_root() -> str:
    root = dependencies_root()
    bundled = root / "send_commend_py/rpc_py_all"
    if bundled.is_dir():
        return str(bundled)
    legacy = root / "send_commend_py-master/rpc_py_all"
    if legacy.is_dir():
        return str(legacy)
    raise RuntimeError(f"rpc_py_all not found under {root}")


def rpc_lib_dir() -> str:
    return str(Path(rpc_py_all_root()) / "lib" / "linux" / _platform_subdir())


def rpc_module_path() -> str:
    return str(Path(rpc_lib_dir()) / "rpc.so")


def topic_lib_dir() -> str:
    return str(dependencies_root() / "get_status_py/topic_all_py/lib" / _platform_subdir())


def topic_all_py_root() -> str:
    return str(dependencies_root() / "get_status_py/topic_all_py")


def require_module(so_path: str, kind: str) -> None:
    """Raise a clear error if a per-Python-version SDK module is missing."""
    if not os.path.isfile(so_path):
        raise RuntimeError(
            f"Missing {kind} module for {_platform_subdir()} / {python_tag()}:\n"
            f"  {so_path}\n"
            f"The bundled SDK ships this for aarch64 (ARM/Jetson Orin). If you are on "
            f"x86-64 with Python 3.12, that build is pending — run on Python 3.10, "
            f"or drop the x86-64 3.12 {kind}.so into the path above."
        )
