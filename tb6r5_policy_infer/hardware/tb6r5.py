"""TB6-R5 hardware interface for LeRobot policy inference (send_commend_py RPC + topic feedback)."""

from __future__ import annotations

import os
import platform
import subprocess
import threading
import time
from contextlib import contextmanager
from typing import Callable, Optional

import numpy as np

from .rpc_transport import RpcSession, _load_rpc_module
from .sdk_paths import rpc_lib_dir
from .topic_feedback import TopicFeedback, validate_topic_sdk

# Keep in sync with constants.INFER_LOG_PREFIX (avoid circular import via constants).
_RPC_LOG_PREFIX = "INFER"

# Conservative JogAnyJ defaults for policy inference (slow & steady for first
# hardware runs). Override on the CLI with --joint-vel/--joint-acc/--joint-dec/--zone-ratio.
DEFAULT_ZONE_RATIO = 0.0
DEFAULT_JOG_ANY_JOINT_VEL = 1.0
DEFAULT_JOG_ANY_JOINT_ACC = 1.0
DEFAULT_JOG_ANY_JOINT_DEC = 1.0
DEFAULT_GRIPPER_MAX_D = 70.0
DEFAULT_GRIPPER_MIN_D = 0.0
DEFAULT_TWO_FINGERS_GRIPPER_INTERVAL = 25.0
DEFAULT_GRIPPER_CMD_DELTA_MM = 0.5
DEFAULT_JOG_ASYNC_TIMEOUT_MS = 5_000_000
DEFAULT_SUBLOOP1_EXEC_TIMEOUT_MS = 6_000_000
DEFAULT_SUBLOOP1_EXIT_TIMEOUT_MS = 120_000
DEFAULT_JOG_ANY_J_LAST_COUNT = 500
SUBLOOP1_CMD = "SubLoop1"
NOT_RUN_EXECUTE = "NotRunExecute"

# JogAnyJ RPC dialect (controller / Codeit version).
# 44: include --zone_ratio and --clear_buffer (current).
# 45: omit --zone_ratio and --clear_buffer.
DEFAULT_CD_VERSION = 44
CD_VERSIONS = (44, 45)

# g_model: 2=MoveTwoFingersGripper (legacy), 3=JogAnyJ j1 meters (new gripper)
DEFAULT_G_MODEL = 2
# subloop: 1=SubLoop1 --exec nesting (default); 0=direct {arm||grip}, no SubLoop1 exit
DEFAULT_SUBLOOP = 1
SUBLOOP_MODES = (0, 1)
# New gripper (g_model=3): j1 0.000–0.080 m ↔ 0–80mm (collection range).
GRIPPER_JOG_MM_FULL_SCALE = 80.0
GRIPPER_JOG_JOINT_MIN = 0.0
GRIPPER_JOG_JOINT_MAX = 0.080
GRIPPER_JOG_JOINT_VEL = 0.5
GRIPPER_JOG_JOINT_ACC = 0.5
GRIPPER_JOG_JOINT_DEC = 0.5
DEFAULT_GRIPPER_G3_MAX_D = 80.0


def _platform_subdir() -> str:
    machine = platform.machine().lower()
    return "arm" if machine in ("aarch64", "arm64") else "x86"


def _host_elf_machine() -> str:
    machine = platform.machine().lower()
    if machine in ("x86_64", "amd64"):
        return "x86-64"
    if machine in ("aarch64", "arm64"):
        return "aarch64"
    return machine


def _elf_machine(path: str) -> Optional[str]:
    try:
        out = subprocess.check_output(["file", "-b", path], text=True, stderr=subprocess.DEVNULL)
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    if "x86-64" in out:
        return "x86-64"
    if "aarch64" in out or "ARM" in out:
        return "aarch64"
    return None


def validate_robot_sdk(*, require_topic: bool = True) -> None:
    """Fail fast when RPC/topic binaries are missing or built for the wrong arch."""
    import sys

    subdir = _platform_subdir()
    host = _host_elf_machine()
    py_ver = f"{sys.version_info.major}.{sys.version_info.minor}"

    rpc_so = os.path.join(rpc_lib_dir(), "rpc.so")
    if not os.path.isfile(rpc_so):
        raise RuntimeError(f"Missing RPC library for linux/{subdir} ({host}): {rpc_so}")
    rpc_elf = _elf_machine(rpc_so)
    if rpc_elf and rpc_elf != host:
        raise RuntimeError(f"rpc.so architecture mismatch: file is {rpc_elf}, host is {host} ({rpc_so}).")

    _load_rpc_module()

    if subdir == "arm" and py_ver != "3.10":
        print(
            f"[TB6R5] WARNING: ARM RPC/topic .so files are built for Python 3.10; "
            f"current interpreter is {py_ver}. Prefer `python3.10` on ARM if import fails."
        )

    if require_topic:
        validate_topic_sdk()


def _should_drop_jog_any_j_rpc_log(line: str) -> bool:
    return "JogAnyJ" in line and ("[async] msg:" in line or "[await] msg:" in line)


@contextmanager
def _filter_stdout_lines(should_drop: Callable[[str], bool]):
    read_fd, write_fd = os.pipe()
    saved_stdout = os.dup(1)
    try:
        os.dup2(write_fd, 1)
        os.close(write_fd)
        yield
    finally:
        os.dup2(saved_stdout, 1)
        captured = b""
        while True:
            chunk = os.read(read_fd, 65536)
            if not chunk:
                break
            captured += chunk
        os.close(read_fd)
        if captured:
            text = captured.decode("utf-8", errors="replace")
            if not text.endswith("\n"):
                text += "\n"
            for line in text.splitlines(keepends=True):
                if should_drop(line):
                    continue
                os.write(saved_stdout, line.encode("utf-8", errors="replace"))
        os.close(saved_stdout)


class TB6R5Interface:
    """Policy-inference robot bridge: SubLoop1 JogAnyJ + gripper via send_commend_py RPC."""

    def __init__(
        self,
        ip: str = "192.168.11.11",
        rpc_port: int = 5868,
        enable_topic: bool = True,
        joint_count: int = 6,
        rpc_cmd_rate_hz: float = 30.0,
        zone_ratio: float = DEFAULT_ZONE_RATIO,
        joint_vel: float = DEFAULT_JOG_ANY_JOINT_VEL,
        joint_acc: float = DEFAULT_JOG_ANY_JOINT_ACC,
        joint_dec: float = DEFAULT_JOG_ANY_JOINT_DEC,
        subloop1_immediate: bool = False,
        print_rpc: bool = False,
        g_model: int = DEFAULT_G_MODEL,
        cd_version: int = DEFAULT_CD_VERSION,
        subloop: int = DEFAULT_SUBLOOP,
        gripper_jog_joint_vel: float = GRIPPER_JOG_JOINT_VEL,
        gripper_jog_joint_acc: float = GRIPPER_JOG_JOINT_ACC,
        gripper_jog_joint_dec: float = GRIPPER_JOG_JOINT_DEC,
        gripper_jog_joint_min: float = GRIPPER_JOG_JOINT_MIN,
        gripper_jog_joint_max: float = GRIPPER_JOG_JOINT_MAX,
        gripper_jog_mm_full_scale: float = GRIPPER_JOG_MM_FULL_SCALE,
    ):
        self.ip = ip
        self.rpc_port = int(rpc_port)
        self.enable_topic = bool(enable_topic)
        self.joint_count = max(int(joint_count), 1)
        self.rpc_cmd_rate_hz = max(float(rpc_cmd_rate_hz), 1.0)
        self.zone_ratio = max(float(zone_ratio), 0.0)
        self.joint_vel = max(float(joint_vel), 0.0)
        self.joint_acc = max(float(joint_acc), 0.0)
        self.joint_dec = max(float(joint_dec), 0.0)
        self.subloop1_immediate = bool(subloop1_immediate)
        self.print_rpc = bool(print_rpc)
        self.g_model = int(g_model)
        if self.g_model not in (2, 3):
            raise ValueError(f"g_model must be 2 or 3, got {self.g_model}")
        self.cd_version = int(cd_version)
        if self.cd_version not in CD_VERSIONS:
            raise ValueError(f"cd_version must be one of {CD_VERSIONS}, got {self.cd_version}")
        self.subloop = int(subloop)
        if self.subloop not in SUBLOOP_MODES:
            raise ValueError(f"subloop must be one of {SUBLOOP_MODES}, got {self.subloop}")
        self.gripper_jog_joint_vel = max(float(gripper_jog_joint_vel), 0.0)
        self.gripper_jog_joint_acc = max(float(gripper_jog_joint_acc), 0.0)
        self.gripper_jog_joint_dec = max(float(gripper_jog_joint_dec), 0.0)
        self.gripper_jog_joint_min = float(gripper_jog_joint_min)
        self.gripper_jog_joint_max = float(gripper_jog_joint_max)
        self.gripper_jog_mm_full_scale = max(float(gripper_jog_mm_full_scale), 1e-6)
        self.jog_async_timeout_ms = DEFAULT_JOG_ASYNC_TIMEOUT_MS

        self._rpc: RpcSession | None = None
        self._topic: TopicFeedback | None = None
        self._rpc_ready = False
        self._server_in_error = False
        self._last_rpc_error: str | None = None
        self._rpc_sync_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._joint_stream_count = 0
        self._cartesian_stream_count = 0
        self._gripper_stream_count = 0
        self._jog_async_pending = 0
        self._subloop1_stream_pending = 0
        self._subloop1_active = False
        self._subloop1_exiting = False
        # Invalidate late first-exec callbacks after exit/abort (JogAnyJ uses a multi-minute
        # async timeout; abort→exit leaves a stale ErrorInfo that must not poison Recover).
        self._subloop1_session_gen = 0
        self._last_gripper_distance_sent: float | None = None
        self._gripper_cmd_delta_mm = DEFAULT_GRIPPER_CMD_DELTA_MM
        self._last_cmd_q: np.ndarray | None = None
        self._last_cmd_xyz: np.ndarray | None = None
        self._last_cmd_quat_wxyz: np.ndarray | None = None
        self.cartesian_vel: float | None = None
        self.cartesian_acc: float | None = None
        self.cartesian_dec: float | None = None

    @property
    def is_connected(self) -> bool:
        if self._rpc is None or not self._rpc_ready:
            return False
        if self.enable_topic and (self._topic is None or not self._topic.is_healthy()):
            return False
        return self._rpc.is_connected()

    def connect(self, topic_wait_timeout_s: float = 5.0) -> None:
        self._rpc_ready = False
        try:
            validate_robot_sdk(require_topic=self.enable_topic)
            print(f"Connecting to TB6-R5 at {self.ip}:{self.rpc_port} ...")
            self._rpc = RpcSession(self.ip, port=self.rpc_port, connect_timeout_ms=5000)
            if not self._rpc.is_connected():
                raise ConnectionError(f"TB6-R5 RPC connection failed: {self._rpc.error_info()}")

            if self.enable_topic:
                self._topic = TopicFeedback(
                    self.ip,
                    joint_count=self.joint_count,
                    poll_hz=self.rpc_cmd_rate_hz,
                    g_model=self.g_model,
                    gripper_jog_joint_min=self.gripper_jog_joint_min,
                    gripper_jog_joint_max=self.gripper_jog_joint_max,
                    gripper_jog_mm_full_scale=self.gripper_jog_mm_full_scale,
                )
                self._topic.start(wait_timeout_s=topic_wait_timeout_s)

            if not self._send_init_commands():
                raise ConnectionError(f"TB6-R5 RPC init failed at {self.ip}:{self.rpc_port}")

            self._rpc_ready = True
            print(f"TB6-R5 connected and verified at {self.ip}:{self.rpc_port}.")
        except Exception:
            self.disconnect()
            raise

    def disconnect(self) -> None:
        self._rpc_ready = False
        if self._rpc is not None:
            try:
                self.exit_subloop1_if_active(timeout_ms=3000, blocking_exit=True)
                self._send_rpc_sync(
                    "{Disable}", timeout_ms=3000, ignore_subcmd_errors=True, log_kind="disconnect Disable"
                )
            except Exception:
                pass
            self._rpc = None
        if self._topic is not None:
            self._topic.stop()
            self._topic = None
        print("TB6-R5 disconnected.")

    def disable(self) -> None:
        # Aborting SubLoop1 Jog mid-stream often leaves "server in error"; Clear first.
        if self._server_in_error:
            self.clear_and_recover(log_prefix="disable-prep")
        if not self._send_rpc_sync("{Disable}", timeout_ms=5000, log_kind="disable"):
            # One more attempt after Clear if Disable itself reported server_in_error.
            self.clear_and_recover(log_prefix="disable-retry")
            self._send_rpc_sync("{Disable}", timeout_ms=5000, log_kind="disable")

    def clear_and_recover(self, *, log_prefix: str = "clear-recover") -> bool:
        """Clear controller error + Recover so a fresh SubLoop1 / Disable can proceed."""
        self._server_in_error = False
        ok_clear = self._send_rpc_sync(
            "{Clear}",
            timeout_ms=5000,
            sleep_s=0.05,
            ignore_subcmd_errors=True,
            log_kind=f"{log_prefix} Clear",
        )
        self._server_in_error = False
        ok_rec = self._send_rpc_sync(
            "{Recover}",
            timeout_ms=5000,
            sleep_s=0.1,
            ignore_subcmd_errors=True,
            log_kind=f"{log_prefix} Recover",
        )
        self._server_in_error = False
        return bool(ok_clear and ok_rec)

    def get_joint_positions(self) -> np.ndarray:
        if self._topic is None:
            return np.zeros(self.joint_count)
        return self._topic.get_joint_positions()

    def get_joint_velocities(self) -> np.ndarray:
        if self._topic is None:
            return np.zeros(self.joint_count)
        return self._topic.get_joint_velocities()

    def get_robottarget(self) -> tuple[np.ndarray, np.ndarray, bool]:
        """TCP pose from Topic: xyz (m), quat_xyzw, healthy.

        Uses vendor ``models_current_points[0].robottarget`` / ``get_current_robottarget``.
        """
        if self._topic is None:
            return np.zeros(3), np.array([0.0, 0.0, 0.0, 1.0]), False
        return self._topic.get_robottarget()

    def is_robottarget_healthy(self) -> bool:
        if self._topic is None:
            return False
        return self._topic.is_robottarget_healthy()

    def get_gripper_distance_mm(self) -> float | None:
        if self._topic is None:
            return None
        return self._topic.get_gripper_distance_mm()

    def get_gripper_distance_m(self) -> float | None:
        if self._topic is None:
            return None
        return self._topic.get_gripper_distance_m()

    def go_home(
        self,
        q: np.ndarray | None = None,
        *,
        gripper_distance: float | None = None,
        interval: float | None = None,
        max_distance: float | None = None,
        min_distance: float | None = None,
        move_timeout_ms: int = DEFAULT_SUBLOOP1_EXIT_TIMEOUT_MS,
        settle_timeout_s: float = 15.0,
    ) -> bool:
        """Home via exit SubLoop1, then dual-model (not SubLoop1) RPC::

            {NotRunExecute||gripper open ...}   # g_model=2 MoveTwoFingers / g_model=3 JogAnyJ j1
            {MoveAbsJ ...||NotRunExecute}       # arm home

        Splitting arm/gripper avoids dual SubLoop1 ``subsystem is not enough`` /
        packing MoveAbsJ+gripper into one SubLoop1 exec.
        """
        if q is None:
            q = np.zeros(self.joint_count)
        q = np.asarray(q, dtype=float).ravel()
        if interval is None:
            interval = DEFAULT_TWO_FINGERS_GRIPPER_INTERVAL

        was_streaming = bool(self._subloop1_active or self._subloop1_exiting)
        self.exit_subloop1_if_active(timeout_ms=move_timeout_ms, blocking_exit=True)
        if was_streaming:
            self.clear_and_recover(log_prefix="go_home post-exit")
            time.sleep(0.15)

        if gripper_distance is not None:
            grip_inner = self._format_gripper_inner(
                gripper_distance, interval, max_distance, min_distance, clear_buffer=0
            )
            # Gripper-only dual-model: {NotRunExecute||gripper ...}
            if not self.send_dual_model(
                NOT_RUN_EXECUTE,
                grip_inner,
                timeout_ms=min(int(move_timeout_ms), 60_000),
                sleep_s=0.05,
                log_kind="go_home gripper",
            ):
                print(f"[{_RPC_LOG_PREFIX}][RPC] go_home: gripper open FAILED", flush=True)
                return False
            self._last_gripper_distance_sent = self._clamp_gripper_distance(
                gripper_distance, max_distance, min_distance
            )
            if self.g_model == 3:
                self._gripper_stream_count = 1

        arm_inner = self._format_move_abs_j_inner(q)
        # Arm-only dual-model: {MoveAbsJ ...||NotRunExecute}
        if not self.send_dual_model(
            arm_inner,
            NOT_RUN_EXECUTE,
            timeout_ms=int(move_timeout_ms),
            sleep_s=0.05,
            log_kind="go_home MoveAbsJ",
        ):
            print(f"[{_RPC_LOG_PREFIX}][RPC] go_home: MoveAbsJ FAILED", flush=True)
            return False

        settled = self._wait_motion_settled(
            settle_timeout_s, target_q=q[: self.joint_count]
        )
        q_now = self.get_joint_positions()
        q_deg = np.rad2deg(q_now)
        target_deg = np.rad2deg(q[: self.joint_count])
        err_deg = q_deg[: self.joint_count] - target_deg
        print(
            f"[{_RPC_LOG_PREFIX}][RPC] go_home settle: settled={settled} "
            f"q_deg={tuple(np.round(q_deg[: self.joint_count], 2))} "
            f"target_deg={tuple(np.round(target_deg, 2))} "
            f"err_deg={tuple(np.round(err_deg, 2))}",
            flush=True,
        )
        if not settled:
            print(
                f"[{_RPC_LOG_PREFIX}][RPC] WARNING: MoveAbsJ did not reach target within "
                f"{settle_timeout_s:.1f}s",
                flush=True,
            )
            return False
        return True

    def set_joint_positions_with_gripper(
        self,
        q: np.ndarray,
        gripper_distance: float,
        force: bool = False,
        clear_buffer: int | None = None,
        interval: float | None = None,
        max_distance: float | None = None,
        min_distance: float | None = None,
        cmd_delta: float | None = None,
    ) -> bool:
        if not self._ensure_command_channel():
            return False

        q_cmd = np.asarray(q, dtype=float).ravel()[: self.joint_count].copy()
        gripper_distance = self._clamp_gripper_distance(gripper_distance, max_distance, min_distance)
        gripper_changed = self._should_send_gripper(
            gripper_distance,
            force=force,
            cmd_delta=cmd_delta,
            max_distance=max_distance,
            min_distance=min_distance,
        )

        clear_buffer = self._resolve_stream_clear_buffer(self._joint_stream_count, clear_buffer)
        arm_inner = self._strip_cmd_braces(self._format_jog_any_j_cmd(q_cmd, clear_buffer=clear_buffer))
        grip_arg = gripper_distance if gripper_changed else None
        grip_clear = None
        if grip_arg is not None and self.g_model == 3:
            grip_clear = 0 if self._gripper_stream_count == 0 else 1
        ok = self._send_subloop1(
            arm_inner,
            grip_arg,
            interval=interval,
            max_distance=max_distance,
            min_distance=min_distance,
            gripper_clear_buffer=grip_clear,
        )
        if not ok:
            return False
        self._last_cmd_q = q_cmd
        self._joint_stream_count += 1
        return True

    @staticmethod
    def quat_xyzw_to_wxyz(quat_xyzw: np.ndarray) -> np.ndarray:
        q = np.asarray(quat_xyzw, dtype=float).ravel()
        if len(q) < 4:
            return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
        qx, qy, qz, qw = float(q[0]), float(q[1]), float(q[2]), float(q[3])
        return np.array([qw, qx, qy, qz], dtype=float)

    def _format_robottarget_value(self, xyz: np.ndarray, quat_wxyz: np.ndarray) -> str:
        xyz = np.asarray(xyz, dtype=float).ravel()[:3]
        quat = np.asarray(quat_wxyz, dtype=float).ravel()
        if len(quat) < 4:
            quat = np.array([1.0, 0.0, 0.0, 0.0])
        qw, qx, qy, qz = float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])
        return "{" + ",".join(f"{v:.6f}" for v in (xyz[0], xyz[1], xyz[2], qx, qy, qz, qw)) + "}"

    def _format_jog_any_c_cmd(
        self,
        xyz: np.ndarray,
        quat_wxyz: np.ndarray,
        clear_buffer: int = 0,
        zone_ratio: float | None = None,
        last_count: int = DEFAULT_JOG_ANY_J_LAST_COUNT,
    ) -> str:
        val_str = self._format_robottarget_value(xyz, quat_wxyz)
        cmd = (
            "{JogAnyC --robottarget_value="
            + val_str
            + self._jog_any_j_zone_clear_suffix(clear_buffer, zone_ratio)
            + f" --last_count={int(last_count)}"
        )
        if self.cartesian_vel is not None:
            cmd += f" --cartesian_vel={float(self.cartesian_vel):.4f}"
        if self.cartesian_acc is not None:
            cmd += f" --cartesian_acc={float(self.cartesian_acc):.4f}"
        if self.cartesian_dec is not None:
            cmd += f" --cartesian_dec={float(self.cartesian_dec):.4f}"
        return cmd + "}"

    def set_cartesian_target_with_gripper(
        self,
        xyz: np.ndarray,
        quat_xyzw: np.ndarray,
        gripper_distance: float,
        force: bool = False,
        clear_buffer: int | None = None,
        interval: float | None = None,
        max_distance: float | None = None,
        min_distance: float | None = None,
        cmd_delta: float | None = None,
    ) -> bool:
        """Send JogAnyC + gripper. ``quat_xyzw`` is policy order; RPC uses {x,y,z,qx,qy,qz,qw}."""
        if not self._ensure_command_channel():
            return False

        xyz_cmd = np.asarray(xyz, dtype=float).ravel()[:3].copy()
        quat_wxyz = self.quat_xyzw_to_wxyz(quat_xyzw)
        gripper_distance = self._clamp_gripper_distance(gripper_distance, max_distance, min_distance)
        gripper_changed = self._should_send_gripper(
            gripper_distance,
            force=force,
            cmd_delta=cmd_delta,
            max_distance=max_distance,
            min_distance=min_distance,
        )

        clear_buffer = self._resolve_stream_clear_buffer(self._cartesian_stream_count, clear_buffer)
        arm_inner = self._strip_cmd_braces(self._format_jog_any_c_cmd(xyz_cmd, quat_wxyz, clear_buffer=clear_buffer))
        grip_arg = gripper_distance if gripper_changed else None
        grip_clear = None
        if grip_arg is not None and self.g_model == 3:
            grip_clear = 0 if self._gripper_stream_count == 0 else 1
        ok = self._send_subloop1(
            arm_inner,
            grip_arg,
            interval=interval,
            max_distance=max_distance,
            min_distance=min_distance,
            gripper_clear_buffer=grip_clear,
        )
        if not ok:
            return False
        self._last_cmd_xyz = xyz_cmd
        self._last_cmd_quat_wxyz = quat_wxyz
        self._cartesian_stream_count += 1
        return True

    def build_subloop1_stream_cmd(
        self,
        q: np.ndarray,
        gripper_distance: float | None,
        clear_buffer: int,
        interval: float | None = None,
        max_distance: float | None = None,
        min_distance: float | None = None,
        immediate: bool | None = None,
    ) -> str:
        """Build the motion RPC command string without sending it (for --print-rpc/dry-run).

        ``gripper_distance=None`` means the gripper slot is NotRunExecute (arm-only step).
        ``subloop=1`` → SubLoop1 nested exec; ``subloop=0`` → direct ``{arm||grip}``.
        """
        if immediate is None:
            immediate = self.subloop1_immediate
        if interval is None:
            interval = DEFAULT_TWO_FINGERS_GRIPPER_INTERVAL
        q_cmd = np.asarray(q, dtype=float).ravel()[: self.joint_count]
        arm_inner = self._strip_cmd_braces(self._format_jog_any_j_cmd(q_cmd, clear_buffer=int(clear_buffer)))
        if gripper_distance is None:
            grip_inner = NOT_RUN_EXECUTE
        else:
            grip_clear = 0 if clear_buffer == 0 else 1
            grip_inner = self._format_gripper_inner(
                gripper_distance,
                interval,
                max_distance,
                min_distance,
                clear_buffer=grip_clear,
            )
        if not self.use_subloop1:
            return self.format_dual_model_cmd(arm_inner, grip_inner)
        return self.format_subloop1_exec_cmd(arm_inner, grip_inner, immediate=immediate)

    def build_cartesian_stream_cmd(
        self,
        xyz: np.ndarray,
        quat_xyzw: np.ndarray,
        gripper_distance: float | None,
        clear_buffer: int,
        interval: float | None = None,
        max_distance: float | None = None,
        min_distance: float | None = None,
        immediate: bool | None = None,
    ) -> str:
        """Build JogAnyC (+ gripper) command string without sending (dry-run / --print-rpc)."""
        if immediate is None:
            immediate = self.subloop1_immediate
        if interval is None:
            interval = DEFAULT_TWO_FINGERS_GRIPPER_INTERVAL
        xyz_cmd = np.asarray(xyz, dtype=float).ravel()[:3]
        quat_wxyz = self.quat_xyzw_to_wxyz(quat_xyzw)
        arm_inner = self._strip_cmd_braces(
            self._format_jog_any_c_cmd(xyz_cmd, quat_wxyz, clear_buffer=int(clear_buffer))
        )
        if gripper_distance is None:
            grip_inner = NOT_RUN_EXECUTE
        else:
            grip_clear = 0 if clear_buffer == 0 else 1
            grip_inner = self._format_gripper_inner(
                gripper_distance,
                interval,
                max_distance,
                min_distance,
                clear_buffer=grip_clear,
            )
        if not self.use_subloop1:
            return self.format_dual_model_cmd(arm_inner, grip_inner)
        return self.format_subloop1_exec_cmd(arm_inner, grip_inner, immediate=immediate)

    def _should_send_gripper(
        self,
        gripper_distance: float,
        force: bool = False,
        cmd_delta: float | None = None,
        max_distance: float | None = None,
        min_distance: float | None = None,
    ) -> bool:
        if force:
            return True
        if cmd_delta is None:
            cmd_delta = self._gripper_cmd_delta_mm
        gripper_distance = self._clamp_gripper_distance(gripper_distance, max_distance, min_distance)
        if self._last_gripper_distance_sent is None:
            return True
        return abs(gripper_distance - self._last_gripper_distance_sent) >= cmd_delta

    def _send_init_commands(self) -> bool:
        print(f"[{_RPC_LOG_PREFIX}][RPC] ---- init sequence begin ----", flush=True)
        init_cmds = [
            "{Clear}",
            "{Disable}",
            "{Mode}",
            "{SetMaxToq}",
            "{Recover}",
            "{SetRate}",
            "{Var --clear}",
            "{Recover}",
            "{SetUsingSP --state=on}",
            "{Var --type=jointtarget --name=teleop --value={0,0,0,0,0,0,0,0,0,0}}",
        ]
        for cmd in init_cmds:
            if not self._send_rpc_sync(cmd, timeout_ms=5000, sleep_s=0.1, log_kind="init"):
                return False
        # g_model=2: enable arm only; g_model=3 (new gripper): Enable||Enable + Start||Start
        if self.g_model == 3:
            enable_right = "Enable"
            start_right = "Start"
        else:
            enable_right = NOT_RUN_EXECUTE
            start_right = NOT_RUN_EXECUTE
        if not self.send_dual_model(
            "Enable", enable_right, timeout_ms=5000, sleep_s=0.1, log_kind="init Enable"
        ):
            return False
        ok = self.send_dual_model(
            "Start",
            start_right,
            timeout_ms=5000,
            sleep_s=0.1,
            ignore_subcmd_errors=True,
            log_kind="init Start",
        )
        print(f"[{_RPC_LOG_PREFIX}][RPC] ---- init sequence end (ok={ok}) ----", flush=True)
        return ok

    def _ensure_command_channel(self) -> bool:
        if not self.is_connected:
            return False
        with self._state_lock:
            return not self._server_in_error

    def _log_rpc_send(self, kind: str, cmd: str, *, always: bool = False) -> None:
        """Print exact RPC payload. Init/home/exit/disable always; stream only with --print-rpc."""
        if always or self.print_rpc:
            print(f"[{_RPC_LOG_PREFIX}][RPC] {kind}: {cmd}", flush=True)

    def _send_rpc_sync(
        self,
        cmd: str,
        timeout_ms: int = 5000,
        sleep_s: float = 0.0,
        ignore_subcmd_errors: bool = False,
        *,
        log_kind: str = "sync send",
    ) -> bool:
        if self._rpc is None:
            return False
        self._log_rpc_send(log_kind, cmd, always=True)
        with self._rpc_sync_lock:
            status, resp_list = self._rpc.call_await(cmd, timeout_ms)
        if status != 0:
            print(f"[TB6R5] RPC sync failed: {cmd} (status={status})", flush=True)
            return False
        self._last_rpc_error = None
        for r in resp_list or []:
            if r.code < 0 and not ignore_subcmd_errors:
                self._last_rpc_error = r.message
                self._server_in_error = True
                print(f"[TB6R5] RPC error: {cmd} -> {r.message}", flush=True)
                return False
        self._server_in_error = False
        if sleep_s > 0:
            time.sleep(sleep_s)
        return True

    @staticmethod
    def _resolve_stream_clear_buffer(stream_count: int, clear_buffer: int | None = None) -> int:
        if clear_buffer is not None:
            return int(clear_buffer)
        return 0 if stream_count == 0 else 1

    def _wait_motion_settled(
        self,
        timeout_s: float,
        target_q: np.ndarray | None = None,
        vel_eps: float = 0.02,
        pos_tol: float = 0.02,
        settle_count: int = 3,
    ) -> bool:
        if self._topic is None or not self._topic.is_healthy():
            # No feedback: open-loop wait. Previously slept <=0.5s then returned False,
            # which let go_home call SubLoop1 exit and abort MoveAbsJ almost immediately.
            wait_s = max(float(timeout_s), 0.0)
            print(
                f"[{_RPC_LOG_PREFIX}][RPC] settle: topic unhealthy/missing; "
                f"open-loop wait {wait_s:.1f}s",
                flush=True,
            )
            if wait_s > 0:
                time.sleep(wait_s)
            return True
        deadline = time.time() + max(timeout_s, 0.0)
        stable = 0
        while time.time() < deadline:
            dq = self.get_joint_velocities()
            moving = bool(np.any(np.abs(dq) > vel_eps))
            reached = True
            if target_q is not None:
                q = self.get_joint_positions()
                reached = bool(np.all(np.abs(q - target_q) < pos_tol))
            if (not moving) and reached:
                stable += 1
                if stable >= settle_count:
                    return True
            else:
                stable = 0
            time.sleep(0.02)
        return False

    def _format_jointtarget_value(self, q: np.ndarray) -> str:
        q = np.asarray(q, dtype=float).ravel()
        values = [0.0] * 10
        for i in range(min(len(q), self.joint_count)):
            values[i] = float(q[i])
        return "{" + ",".join(f"{v:.6f}" for v in values) + "}"

    def _jog_any_j_zone_clear_suffix(
        self,
        clear_buffer: int = 0,
        zone_ratio: float | None = None,
    ) -> str:
        """cd-version 44: --zone_ratio/--clear_buffer; 45: omit both."""
        if self.cd_version >= 45:
            return ""
        if zone_ratio is None:
            zone_ratio = self.zone_ratio
        return f" --zone_ratio={float(zone_ratio):.4f} --clear_buffer={int(clear_buffer)}"

    def _format_jog_any_j_cmd(
        self,
        q: np.ndarray,
        clear_buffer: int = 0,
        zone_ratio: float | None = None,
        last_count: int = DEFAULT_JOG_ANY_J_LAST_COUNT,
    ) -> str:
        val_str = self._format_jointtarget_value(q)
        return (
            "{JogAnyJ --jointtarget_value="
            + val_str
            + self._jog_any_j_zone_clear_suffix(clear_buffer, zone_ratio)
            + f" --last_count={int(last_count)}"
            + f" --joint_vel={self.joint_vel:.4f} --joint_acc={self.joint_acc:.4f} --joint_dec={self.joint_dec:.4f}"
            + "}"
        )

    def _format_move_abs_j_inner(self, q: np.ndarray) -> str:
        return f"MoveAbsJ --jointtarget_value={self._format_jointtarget_value(q)}"

    @staticmethod
    def _strip_cmd_braces(cmd: str) -> str:
        cmd = cmd.strip()
        if cmd.startswith("{") and cmd.endswith("}"):
            return cmd[1:-1]
        return cmd

    def _clamp_gripper_distance(
        self,
        distance: float,
        max_distance: float | None = None,
        min_distance: float | None = None,
    ) -> float:
        lo = float(DEFAULT_GRIPPER_MIN_D if min_distance is None else min_distance)
        if max_distance is None:
            hi = float(DEFAULT_GRIPPER_G3_MAX_D if self.g_model == 3 else DEFAULT_GRIPPER_MAX_D)
        else:
            hi = float(max_distance)
        if lo > hi:
            lo, hi = hi, lo
        return max(lo, min(float(distance), hi))

    def _gripper_mm_to_joint(self, distance_mm: float) -> float:
        """Map opening mm → gripper JogAnyJ j1 (meters). 0–full_scale ↔ joint_min–joint_max."""
        scale = self.gripper_jog_mm_full_scale
        mm = max(0.0, min(float(distance_mm), scale))
        j_lo = self.gripper_jog_joint_min
        j_hi = self.gripper_jog_joint_max
        return j_lo + (mm / scale) * (j_hi - j_lo)

    def _format_gripper_jog_any_j_inner(
        self,
        distance: float,
        max_distance: float | None = None,
        min_distance: float | None = None,
        clear_buffer: int | None = None,
        last_count: int = DEFAULT_JOG_ANY_J_LAST_COUNT,
    ) -> str:
        """New gripper (g_model=3): right slot JogAnyJ on j1 only (meters)."""
        distance = self._clamp_gripper_distance(distance, max_distance, min_distance)
        j1 = self._gripper_mm_to_joint(distance)
        if clear_buffer is None:
            clear_buffer = 0 if self._gripper_stream_count == 0 else 1
        val_str = self._format_jointtarget_value(np.array([j1], dtype=float))
        return (
            f"JogAnyJ --jointtarget_value={val_str}"
            f"{self._jog_any_j_zone_clear_suffix(clear_buffer)}"
            f" --last_count={int(last_count)}"
            f" --joint_vel={self.gripper_jog_joint_vel:.4f}"
            f" --joint_acc={self.gripper_jog_joint_acc:.4f}"
            f" --joint_dec={self.gripper_jog_joint_dec:.4f}"
        )

    def _format_gripper_inner(
        self,
        distance: float,
        interval: float,
        max_distance: float | None = None,
        min_distance: float | None = None,
        clear_buffer: int | None = None,
    ) -> str:
        if self.g_model == 3:
            return self._format_gripper_jog_any_j_inner(
                distance,
                max_distance=max_distance,
                min_distance=min_distance,
                clear_buffer=clear_buffer,
            )
        distance = self._clamp_gripper_distance(distance, max_distance, min_distance)
        interval = max(0.0, float(interval))
        return f"MoveTwoFingersGripper --distance={distance:.4f} --interval={interval:.4f}"

    def format_dual_model_cmd(self, arm_inner: str, grip_inner: str) -> str:
        arm_inner = (arm_inner or NOT_RUN_EXECUTE).strip()
        grip_inner = (grip_inner or NOT_RUN_EXECUTE).strip()
        return f"{{{arm_inner}||{grip_inner}}}"

    def send_dual_model(
        self,
        arm_inner: str,
        grip_inner: str,
        timeout_ms: int = 5000,
        sleep_s: float = 0.0,
        ignore_subcmd_errors: bool = False,
        *,
        log_kind: str = "dual-model send",
    ) -> bool:
        return self._send_rpc_sync(
            self.format_dual_model_cmd(arm_inner, grip_inner),
            timeout_ms=timeout_ms,
            sleep_s=sleep_s,
            ignore_subcmd_errors=ignore_subcmd_errors,
            log_kind=log_kind,
        )

    def format_subloop1_exec_cmd(self, arm_inner: str, grip_inner: str, immediate: bool = False) -> str:
        arm_inner = (arm_inner or NOT_RUN_EXECUTE).strip()
        grip_inner = (grip_inner or NOT_RUN_EXECUTE).strip()
        immediate_suffix = " --immediate=true" if immediate else ""
        return (
            f"{{{SUBLOOP1_CMD} --exec={{{arm_inner}}}{immediate_suffix}"
            f"||{SUBLOOP1_CMD} --exec={{{grip_inner}}}{immediate_suffix}}}"
        )

    def format_subloop1_exit_cmd(self) -> str:
        return f"{{{SUBLOOP1_CMD} --exec={{exit}}||{SUBLOOP1_CMD} --exec={{exit}}}}"

    def _send_subloop1_first_async(self, cmd: str) -> bool:
        if self._rpc is None:
            return False

        gen = self._subloop1_session_gen

        def _on_response(status: int, resp_list):
            with self._state_lock:
                self._jog_async_pending = max(0, self._jog_async_pending - 1)
                stale = gen != self._subloop1_session_gen
            if stale:
                # Intentionally aborted session (exit/home); do not mark server_in_error.
                return
            if status < 0:
                self._server_in_error = True
                self._last_rpc_error = f"SubLoop1 first exec async timeout (status={status})"
                print(f"[TB6R5] {self._last_rpc_error}")
                return
            for r in resp_list or []:
                if r.code < 0:
                    self._server_in_error = True
                    self._last_rpc_error = r.message
                    print(f"[TB6R5] SubLoop1 first exec error: {r.message}")
                    return
            self._server_in_error = False
            self._last_rpc_error = None

        ok = self._rpc.call_async(cmd, self.jog_async_timeout_ms, _on_response)
        if ok:
            with self._state_lock:
                self._jog_async_pending += 1
            self._subloop1_active = True
        return bool(ok)

    def _send_subloop1_stream_async(self, cmd: str) -> bool:
        """Subsequent SubLoop1 exec: fire-and-forget (expect_resp=False).

        High-rate JogAnyJ must not register per-frame pending responses: on Ctrl+C/exit
        hundreds of CallAsync callbacks would otherwise time out as status=-3 and spam
        ``No valid pending response for seqID``.
        """
        if self._rpc is None:
            return False

        def _on_response(status: int, resp_list):
            # With expect_resp=False the SDK usually will not call this; keep a no-op
            # for bindings that still invoke it.
            return

        with _filter_stdout_lines(_should_drop_jog_any_j_rpc_log):
            ok = self._rpc.call_async(
                cmd, DEFAULT_SUBLOOP1_EXEC_TIMEOUT_MS, _on_response, expect_resp=False
            )
        return bool(ok)

    def drain_async_pending(self, timeout_s: float = 2.0) -> int:
        """Wait until tracked first-exec / stream pending counters reach 0. Returns leftover."""
        deadline = time.monotonic() + max(float(timeout_s), 0.0)
        leftover = 0
        while time.monotonic() < deadline:
            with self._state_lock:
                leftover = self._jog_async_pending + self._subloop1_stream_pending
            if leftover == 0:
                return 0
            time.sleep(0.01)
        return leftover

    def _finalize_subloop1_session(self) -> None:
        self._subloop1_active = False
        self._subloop1_exiting = False
        with self._state_lock:
            self._jog_async_pending = 0
            self._subloop1_stream_pending = 0
            self._subloop1_session_gen += 1
        # Next SubLoop1 session should start with clear_buffer=0 for arm/gripper streams.
        self._joint_stream_count = 0
        self._cartesian_stream_count = 0
        self._gripper_stream_count = 0

    def send_subloop1_exit(self, timeout_ms: int = DEFAULT_SUBLOOP1_EXIT_TIMEOUT_MS, blocking: bool = False) -> bool:
        if not self.use_subloop1:
            return True
        if not self._subloop1_active and not self._subloop1_exiting:
            return True
        if self._rpc is None:
            return False
        if blocking and not self._subloop1_active:
            return True

        cmd = self.format_subloop1_exit_cmd()
        done = threading.Event()
        result = {"ok": True}

        def _on_response(status: int, resp_list):
            with self._state_lock:
                self._jog_async_pending = max(0, self._jog_async_pending - 1)
            if status < 0:
                result["ok"] = False
                print(
                    f"[TB6R5] SubLoop1 exit async failed (status={status})",
                    flush=True,
                )
            self._finalize_subloop1_session()
            done.set()

        kind = "SubLoop1 exit (async, wait)" if blocking else "SubLoop1 exit (async)"
        self._log_rpc_send(kind, cmd, always=True)
        self._subloop1_exiting = True
        self._subloop1_active = False
        with self._state_lock:
            self._subloop1_stream_pending = 0
            # Invalidate JogAnyJ first-exec immediately so late ErrorInfo cannot poison homing.
            self._subloop1_session_gen += 1

        ok = self._rpc.call_async(cmd, timeout_ms, _on_response)
        if not ok:
            self._subloop1_exiting = False
            done.set()
            return False

        with self._state_lock:
            self._jog_async_pending += 1

        if not blocking:
            return True

        wait_s = max(float(timeout_ms), 0.0) / 1000.0
        if not done.wait(timeout=wait_s):
            print(
                f"[{_RPC_LOG_PREFIX}][RPC] WARNING: SubLoop1 exit async wait timed out "
                f"after {wait_s:.1f}s",
                flush=True,
            )
            self._finalize_subloop1_session()
            return False
        return bool(result["ok"])

    def exit_subloop1_if_active(
        self,
        timeout_ms: int | None = None,
        settle_timeout_s: float = 2.0,
        blocking_exit: bool = False,
        drain_timeout_s: float = 2.0,
    ) -> bool:
        if not self.use_subloop1:
            return True
        if not self._subloop1_active and not self._subloop1_exiting:
            return True
        if timeout_ms is None:
            timeout_ms = DEFAULT_SUBLOOP1_EXIT_TIMEOUT_MS
        # Stop-stream hygiene: drain tracked async first, then settle, then exit.
        leftover = self.drain_async_pending(drain_timeout_s)
        if leftover:
            print(
                f"[{_RPC_LOG_PREFIX}][RPC] WARNING: {leftover} async RPC still pending "
                f"before SubLoop1 exit (will exit anyway)",
                flush=True,
            )
        if blocking_exit and self._subloop1_active:
            self._wait_motion_settled(settle_timeout_s)
        return self.send_subloop1_exit(timeout_ms=timeout_ms, blocking=blocking_exit)

    @property
    def use_subloop1(self) -> bool:
        """True when Jog/gripper are wrapped in ``SubLoop1 --exec={...}`` (``subloop=1``)."""
        return self.subloop != 0

    def _send_subloop1(
        self,
        arm_inner: str,
        gripper_distance: float | None,
        interval: float | None = None,
        max_distance: float | None = None,
        min_distance: float | None = None,
        immediate: bool | None = None,
        gripper_clear_buffer: int | None = None,
    ) -> bool:
        """Send arm+gripper exec.

        ``subloop=1`` (default): ``{SubLoop1 --exec={arm}||SubLoop1 --exec={grip}}``.
        ``subloop=0``: direct ``{arm||grip}`` (no SubLoop1 nesting / no exit).
        """
        if not self._ensure_command_channel():
            return False
        if immediate is None:
            immediate = self.subloop1_immediate
        if interval is None:
            interval = DEFAULT_TWO_FINGERS_GRIPPER_INTERVAL
        arm_inner = (arm_inner or NOT_RUN_EXECUTE).strip()
        if gripper_distance is None:
            grip_inner = NOT_RUN_EXECUTE
        else:
            if gripper_clear_buffer is None:
                gripper_clear_buffer = 0 if self._gripper_stream_count == 0 else 1
            grip_inner = self._format_gripper_inner(
                gripper_distance,
                interval,
                max_distance,
                min_distance,
                clear_buffer=gripper_clear_buffer,
            )
        if not self.use_subloop1:
            cmd = self.format_dual_model_cmd(arm_inner, grip_inner)
            self._log_rpc_send("dual-model stream send", cmd, always=False)
            ok = self._send_subloop1_stream_async(cmd)
        else:
            cmd = self.format_subloop1_exec_cmd(arm_inner, grip_inner, immediate=immediate)
            if self._subloop1_exiting:
                return False
            slot = "first" if not self._subloop1_active else "stream"
            self._log_rpc_send(f"SubLoop1 {slot} send", cmd, always=False)
            if not self._subloop1_active:
                ok = self._send_subloop1_first_async(cmd)
            else:
                ok = self._send_subloop1_stream_async(cmd)
        if ok and gripper_distance is not None:
            self._last_gripper_distance_sent = self._clamp_gripper_distance(
                gripper_distance, max_distance, min_distance
            )
            self._gripper_stream_count += 1
        return ok

    def _send_subloop1_blocking(
        self,
        arm_inner: str,
        grip_inner: str,
        timeout_ms: int = DEFAULT_SUBLOOP1_EXIT_TIMEOUT_MS,
        immediate: bool = False,
        settle_target_q: np.ndarray | None = None,
        settle_timeout_s: float = 15.0,
    ) -> bool:
        was_streaming = bool(self._subloop1_active or self._subloop1_exiting)
        self.exit_subloop1_if_active(timeout_ms=timeout_ms, blocking_exit=True)
        # After killing a high-rate JogAnyJ session, controller often needs Clear+Recover
        # before MoveAbsJ in a new SubLoop1 will actually move (else settle stays at stream pose).
        if was_streaming:
            self.clear_and_recover(log_prefix="post-jog-exit")
            time.sleep(0.15)
        cmd = self.format_subloop1_exec_cmd(arm_inner, grip_inner, immediate=immediate)
        self._log_rpc_send("SubLoop1 blocking send", cmd, always=True)
        if not self._send_subloop1_first_async(cmd):
            print(f"[{_RPC_LOG_PREFIX}][RPC] SubLoop1 blocking: first async send FAILED", flush=True)
            return False
        settled = self._wait_motion_settled(settle_timeout_s, target_q=settle_target_q)
        q_now = self.get_joint_positions()
        q_deg = np.rad2deg(q_now)
        if settle_target_q is not None:
            target_deg = np.rad2deg(np.asarray(settle_target_q, dtype=float).ravel()[: self.joint_count])
            err_deg = q_deg[: self.joint_count] - target_deg
            print(
                f"[{_RPC_LOG_PREFIX}][RPC] homing settle: settled={settled} "
                f"q_deg={tuple(np.round(q_deg[: self.joint_count], 2))} "
                f"target_deg={tuple(np.round(target_deg, 2))} "
                f"err_deg={tuple(np.round(err_deg, 2))}",
                flush=True,
            )
        else:
            print(
                f"[{_RPC_LOG_PREFIX}][RPC] homing settle: settled={settled} "
                f"q_deg={tuple(np.round(q_deg[: self.joint_count], 2))}",
                flush=True,
            )
        exit_ok = self.send_subloop1_exit(timeout_ms=timeout_ms, blocking=True)
        if not settled:
            print(
                f"[{_RPC_LOG_PREFIX}][RPC] WARNING: MoveAbsJ did not reach target within "
                f"{settle_timeout_s:.1f}s (exit_ok={exit_ok})",
                flush=True,
            )
            return False
        return bool(exit_ok)
