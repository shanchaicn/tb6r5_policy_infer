# platform_loader.py
import sys
import os
import platform
import ctypes
import shutil
import subprocess

_loaded = False


def _elf_machine(path: str):
    try:
        out = subprocess.check_output(["file", "-b", path], text=True, stderr=subprocess.DEVNULL)
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    if "x86-64" in out:
        return "x86-64"
    if "aarch64" in out or "ARM" in out:
        return "aarch64"
    return None


def _host_elf_machine() -> str:
    machine = platform.machine().lower()
    if machine in ("x86_64", "amd64"):
        return "x86-64"
    if machine in ("aarch64", "arm64"):
        return "aarch64"
    return machine


def _ensure_protobuf_lib(target_dir: str) -> str:
    """Use host-arch libprotobuf (upstream x86/ may ship ARM binaries)."""
    pb = os.path.join(target_dir, "libprotobuf.so")
    pb32 = os.path.join(target_dir, "libprotobuf.so.32")
    host = _host_elf_machine()

    if os.path.isfile(pb) and _elf_machine(pb) == host:
        if not os.path.isfile(pb32) or _elf_machine(pb32) != host:
            shutil.copy2(pb, pb32)
        return pb32

    if os.path.isfile(pb32) and _elf_machine(pb32) == host:
        return pb32

    raise RuntimeError(
        f"No compatible libprotobuf for {host} in {target_dir}. " "Place the correct libprotobuf.so in lib/<platform>/."
    )


def _preload_library(path: str):
    ctypes.CDLL(os.path.abspath(path), mode=ctypes.RTLD_GLOBAL)


def get_topic_module():
    """自动检测平台、配置动态库路径，返回 topic 模块"""
    global _loaded
    if not _loaded:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        system = platform.system().lower()
        machine = platform.machine().lower()

        # 确定平台子目录
        if system == "windows":
            subdir = "win"
        elif system == "linux":
            if machine in ("x86_64", "amd64", "i386", "i686"):
                subdir = "x86"
            elif machine in ("armv7l", "aarch64", "arm64"):
                subdir = "arm"
            else:
                raise RuntimeError(f"Unsupported Linux architecture: {machine}")
        else:
            raise RuntimeError(f"Unsupported OS: {system}")

        target_dir = os.path.join(base_dir, "lib", subdir)
        if not os.path.isdir(target_dir):
            raise RuntimeError(f"Platform directory not found: {target_dir}")

        # 添加模块搜索路径
        if target_dir not in sys.path:
            sys.path.insert(0, target_dir)

        # 处理动态库依赖路径
        if system == "windows":
            os.environ["PATH"] = target_dir + ";" + base_dir + ";" + os.environ.get("PATH", "")
            if hasattr(os, "add_dll_directory"):
                os.add_dll_directory(target_dir)
                os.add_dll_directory(base_dir)
        elif system == "linux":
            ld_path = target_dir + ":" + base_dir
            current_ld = os.environ.get("LD_LIBRARY_PATH", "")
            if current_ld:
                ld_path = ld_path + ":" + current_ld
            os.environ["LD_LIBRARY_PATH"] = ld_path
            pb32 = _ensure_protobuf_lib(target_dir)
            os.environ["LD_PRELOAD"] = pb32
            _preload_library(pb32)

        _loaded = True

    import topic

    return topic
