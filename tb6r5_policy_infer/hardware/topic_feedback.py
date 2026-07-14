"""Topic feedback (joint state, gripper) for TB6-R5 policy inference."""

from __future__ import annotations

import ctypes
import os
import platform
import shutil
import subprocess
import sys
import threading
import time
from typing import Optional

import numpy as np

from .sdk_paths import require_module, topic_all_py_root, topic_lib_dir

GRIPPER_YS_STATUS_FORMAT = "<d"


def _host_elf_machine() -> str:
    machine = platform.machine().lower()
    if machine in ("x86_64", "amd64"):
        return "x86-64"
    if machine in ("aarch64", "arm64"):
        return "aarch64"
    return machine


def _elf_machine(path: str) -> Optional[str]:
    try:
        # -L follows symlinks (system libs are usually symlinks to versioned files).
        out = subprocess.check_output(["file", "-bL", path], text=True, stderr=subprocess.DEVNULL)
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    if "x86-64" in out:
        return "x86-64"
    if "aarch64" in out or "ARM" in out:
        return "aarch64"
    return None


def _preload_library(path: str) -> bool:
    if not os.path.isfile(path):
        return False
    try:
        ctypes.CDLL(os.path.abspath(path), mode=ctypes.RTLD_GLOBAL)
        return True
    except OSError:
        return False


def _prepend_ld_library_path(*paths: str) -> None:
    existing = os.environ.get("LD_LIBRARY_PATH", "")
    merged = ":".join(path for path in paths if path)
    if existing:
        merged = merged + ":" + existing
    os.environ["LD_LIBRARY_PATH"] = merged


def _ensure_topic_protobuf_lib(target_dir: str) -> str:
    pb = os.path.join(target_dir, "libprotobuf.so")
    pb32 = os.path.join(target_dir, "libprotobuf.so.32")
    host = _host_elf_machine()

    if os.path.isfile(pb) and _elf_machine(pb) == host:
        if not os.path.isfile(pb32) or _elf_machine(pb32) != host:
            shutil.copy2(pb, pb32)
        return pb32

    if os.path.isfile(pb32) and _elf_machine(pb32) == host:
        return pb32

    raise RuntimeError(f"No compatible libprotobuf for {host} in {target_dir}.")


def _find_system_library(sonames: tuple[str, ...]) -> Optional[str]:
    """Locate a host-arch shared library from the system (ldconfig / common dirs)."""
    host = _host_elf_machine()
    try:
        out = subprocess.check_output(["ldconfig", "-p"], text=True, stderr=subprocess.DEVNULL)
    except (subprocess.CalledProcessError, FileNotFoundError):
        out = ""
    for line in out.splitlines():
        if "=>" not in line:
            continue
        for soname in sonames:
            if soname in line:
                path = line.split("=>", 1)[1].strip()
                if os.path.isfile(path) and _elf_machine(path) == host:
                    return path
    common_dirs = (
        "/usr/lib/x86_64-linux-gnu",
        "/lib/x86_64-linux-gnu",
        "/usr/lib/aarch64-linux-gnu",
        "/lib/aarch64-linux-gnu",
        "/usr/lib",
        "/usr/local/lib",
    )
    for directory in common_dirs:
        for soname in sonames:
            path = os.path.join(directory, soname)
            if os.path.isfile(path) and _elf_machine(path) == host:
                return path
    return None


def _resolve_zmq_lib(target_dir: str) -> str:
    """Return a host-arch libzmq: bundled if it matches, else the system copy.

    The vendor bundle ships a self-contained ARM libzmq; on x86 (and any host
    where the bundled file is the wrong arch) we fall back to the system libzmq.
    """
    host = _host_elf_machine()
    for name in ("libzmq.so.5", "libzmq.so"):
        path = os.path.join(target_dir, name)
        if os.path.isfile(path) and _elf_machine(path) == host:
            return path
    system = _find_system_library(("libzmq.so.5", "libzmq.so"))
    if system:
        return system
    raise RuntimeError(
        f"No compatible libzmq (host {host}) bundled in {target_dir} nor on the system.\n"
        "Install the system ZeroMQ runtime, e.g. Debian/Ubuntu/Jetson:\n"
        "  sudo apt-get update && sudo apt-get install -y libzmq5"
    )


def setup_topic_import() -> None:
    topic_root = topic_all_py_root()
    target_dir = topic_lib_dir()
    if target_dir not in sys.path:
        sys.path.insert(0, target_dir)
    pb32 = _ensure_topic_protobuf_lib(target_dir)
    zmq_path = _resolve_zmq_lib(target_dir)
    _prepend_ld_library_path(target_dir, topic_root)
    # Preload both deps so topic.so's NEEDED sonames resolve to the correct-arch
    # files regardless of what sits in LD_LIBRARY_PATH.
    os.environ["LD_PRELOAD"] = ":".join(p for p in (pb32, zmq_path) if p)
    _preload_library(zmq_path)
    _preload_library(pb32)


def validate_topic_sdk() -> None:
    host = _host_elf_machine()
    topic_dir = topic_lib_dir()
    topic_so = os.path.join(topic_dir, "topic.so")
    require_module(topic_so, "topic")
    topic_elf = _elf_machine(topic_so)
    if topic_elf and topic_elf != host:
        raise RuntimeError(f"topic.so architecture mismatch: file is {topic_elf}, host is {host}")
    _ensure_topic_protobuf_lib(topic_dir)
    _resolve_zmq_lib(topic_dir)


class TopicFeedback:
    """Background reader for joint positions and YS gripper feedback."""

    def __init__(self, ip: str, joint_count: int = 6, poll_hz: float = 30.0):
        self.ip = ip
        self.joint_count = max(int(joint_count), 1)
        self.poll_dt = 1.0 / max(float(poll_hz), 1.0)
        self._topic = None
        self._topic_all_py_root = topic_all_py_root()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._healthy = False
        self._q = np.zeros(self.joint_count)
        self._dq = np.zeros(self.joint_count)
        self._gripper_mm: float | None = None
        self._gripper_healthy = False

    @staticmethod
    def _parse_robottarget_value(rt_value):
        try:
            if rt_value is None or not hasattr(rt_value, "__len__"):
                return np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0]), False
            vals = [float(v) for v in rt_value]
            if len(vals) < 7:
                return np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0]), False
            xyz = np.array(vals[:3], dtype=float)
            qx, qy, qz, qw = vals[3], vals[4], vals[5], vals[6]
            quat = np.array([qw, qx, qy, qz], dtype=float)
            return xyz, quat, True
        except (TypeError, ValueError):
            return np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0]), False

    def start(self, wait_timeout_s: float = 5.0) -> None:
        validate_topic_sdk()
        setup_topic_import()
        import topic

        self._topic = topic
        topic.start_subscriber(self.ip)
        time.sleep(0.5)
        self._thread = threading.Thread(target=self._loop, name="tb6r5_topic_reader", daemon=True)
        self._thread.start()
        deadline = time.monotonic() + max(float(wait_timeout_s), 0.0)
        while time.monotonic() < deadline:
            if self.is_healthy():
                return
            time.sleep(0.05)
        raise ConnectionError(f"TB6-R5 topic feedback not available from {self.ip} within {wait_timeout_s:.1f}s.")

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        self._healthy = False

    def is_healthy(self) -> bool:
        with self._lock:
            return self._healthy

    def get_joint_positions(self) -> np.ndarray:
        with self._lock:
            return self._q.copy()

    def get_joint_velocities(self) -> np.ndarray:
        with self._lock:
            return self._dq.copy()

    def get_gripper_distance_mm(self) -> float | None:
        with self._lock:
            return None if self._gripper_mm is None else float(self._gripper_mm)

    def is_gripper_feedback_healthy(self) -> bool:
        with self._lock:
            return self._gripper_healthy

    def _read_gripper_mm(self) -> tuple[float | None, bool]:
        if self._topic is None:
            return None, False
        try:
            if self._topic_all_py_root not in sys.path:
                sys.path.insert(0, self._topic_all_py_root)
            from system_state_reader import (
                get_subsystem_count,
                get_subsystem_data_size,
                get_subsystem_name,
                has_nrt_data,
                parse_subsystem_data,
            )

            if not has_nrt_data():
                return None, False
            for idx in range(get_subsystem_count()):
                name = get_subsystem_name(idx)
                if "Gripper" not in name and "gripper" not in name:
                    continue
                if get_subsystem_data_size(idx) < 8:
                    continue
                actual_pos = float(parse_subsystem_data(idx, GRIPPER_YS_STATUS_FORMAT)[0])
                return actual_pos, True
            return None, False
        except Exception:
            return None, False

    def _read_joints(self) -> tuple[np.ndarray, np.ndarray, bool]:
        if self._topic is None:
            return self._q.copy(), self._dq.copy(), False
        try:
            state = self._topic.get_system_state()
            if not state.has_rt():
                return self._q.copy(), self._dq.copy(), False
            rt = state.get_rt()
            if not rt.models:
                return self._q.copy(), self._dq.copy(), False
            model = rt.models[0]
            start = model.joint_start_idx
            q = np.zeros(self.joint_count)
            dq = np.zeros(self.joint_count)
            for j in range(start, min(start + model.joint_count, start + self.joint_count)):
                joint = rt.models_joints[j]
                q[j - start] = joint.position
                dq[j - start] = joint.velocity
            return q, dq, True
        except Exception:
            return self._q.copy(), self._dq.copy(), False

    def _loop(self) -> None:
        while not self._stop.is_set():
            q, dq, ok = self._read_joints()
            gripper_mm, gripper_ok = self._read_gripper_mm()
            with self._lock:
                if ok:
                    self._q = q
                    self._dq = dq
                    self._healthy = True
                if gripper_ok:
                    self._gripper_mm = gripper_mm
                    self._gripper_healthy = True
            time.sleep(self.poll_dt)
