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
DEFAULT_SUBLOOP1_EXEC_TIMEOUT_MS = 5000
DEFAULT_SUBLOOP1_EXIT_TIMEOUT_MS = 120_000
DEFAULT_JOG_ANY_J_LAST_COUNT = 500
SUBLOOP1_CMD = "SubLoop1"
NOT_RUN_EXECUTE = "NotRunExecute"


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
        self.jog_async_timeout_ms = DEFAULT_JOG_ASYNC_TIMEOUT_MS

        self._rpc: RpcSession | None = None
        self._topic: TopicFeedback | None = None
        self._rpc_ready = False
        self._server_in_error = False
        self._last_rpc_error: str | None = None
        self._rpc_sync_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._joint_stream_count = 0
        self._jog_async_pending = 0
        self._subloop1_stream_pending = 0
        self._subloop1_active = False
        self._subloop1_exiting = False
        self._last_gripper_distance_sent: float | None = None
        self._gripper_cmd_delta_mm = DEFAULT_GRIPPER_CMD_DELTA_MM
        self._last_cmd_q: np.ndarray | None = None

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
                self._topic = TopicFeedback(self.ip, joint_count=self.joint_count, poll_hz=self.rpc_cmd_rate_hz)
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
        self._send_rpc_sync("{Disable}", timeout_ms=5000, log_kind="disable")

    def get_joint_positions(self) -> np.ndarray:
        if self._topic is None:
            return np.zeros(self.joint_count)
        return self._topic.get_joint_positions()

    def get_joint_velocities(self) -> np.ndarray:
        if self._topic is None:
            return np.zeros(self.joint_count)
        return self._topic.get_joint_velocities()

    def get_gripper_distance_mm(self) -> float | None:
        if self._topic is None:
            return None
        return self._topic.get_gripper_distance_mm()

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
        if q is None:
            q = np.zeros(self.joint_count)
        q = np.asarray(q, dtype=float).ravel()
        arm_inner = self._format_move_abs_j_inner(q)
        if interval is None:
            interval = DEFAULT_TWO_FINGERS_GRIPPER_INTERVAL
        if gripper_distance is None:
            grip_inner = NOT_RUN_EXECUTE
        else:
            grip_inner = self._format_gripper_inner(gripper_distance, interval, max_distance, min_distance)
        ok = self._send_subloop1_blocking(
            arm_inner,
            grip_inner,
            timeout_ms=move_timeout_ms,
            settle_target_q=q[: self.joint_count],
            settle_timeout_s=settle_timeout_s,
        )
        if ok and gripper_distance is not None:
            self._last_gripper_distance_sent = self._clamp_gripper_distance(
                gripper_distance, max_distance, min_distance
            )
        return ok

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
        ok = self._send_subloop1(
            arm_inner,
            grip_arg,
            interval=interval,
            max_distance=max_distance,
            min_distance=min_distance,
        )
        if not ok:
            return False
        self._last_cmd_q = q_cmd
        self._joint_stream_count += 1
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
        """Build the SubLoop1 exec command string without sending it (for --print-rpc/dry-run).

        ``gripper_distance=None`` means the gripper slot is NotRunExecute (arm-only step).
        """
        if immediate is None:
            immediate = self.subloop1_immediate
        if interval is None:
            interval = DEFAULT_TWO_FINGERS_GRIPPER_INTERVAL
        q_cmd = np.asarray(q, dtype=float).ravel()[: self.joint_count]
        arm_inner = self._strip_cmd_braces(self._format_jog_any_j_cmd(q_cmd, clear_buffer=int(clear_buffer)))
        grip_inner = (
            NOT_RUN_EXECUTE
            if gripper_distance is None
            else self._format_gripper_inner(gripper_distance, interval, max_distance, min_distance)
        )
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
        if not self.send_dual_model(
            "Enable", NOT_RUN_EXECUTE, timeout_ms=5000, sleep_s=0.1, log_kind="init Enable"
        ):
            return False
        ok = self.send_dual_model(
            "Start",
            NOT_RUN_EXECUTE,
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

    def _format_jog_any_j_cmd(
        self,
        q: np.ndarray,
        clear_buffer: int = 0,
        zone_ratio: float | None = None,
        last_count: int = DEFAULT_JOG_ANY_J_LAST_COUNT,
    ) -> str:
        if zone_ratio is None:
            zone_ratio = self.zone_ratio
        val_str = self._format_jointtarget_value(q)
        return (
            "{JogAnyJ --jointtarget_value="
            + val_str
            + f" --zone_ratio={float(zone_ratio):.4f} --clear_buffer={int(clear_buffer)} --last_count={int(last_count)}"
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
        hi = float(DEFAULT_GRIPPER_MAX_D if max_distance is None else max_distance)
        if lo > hi:
            lo, hi = hi, lo
        return max(lo, min(float(distance), hi))

    def _format_gripper_inner(
        self,
        distance: float,
        interval: float,
        max_distance: float | None = None,
        min_distance: float | None = None,
    ) -> str:
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

        def _on_response(status: int, resp_list):
            with self._state_lock:
                self._jog_async_pending = max(0, self._jog_async_pending - 1)
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
        if self._rpc is None:
            return False

        def _on_response(status: int, resp_list):
            with self._state_lock:
                self._subloop1_stream_pending = max(0, self._subloop1_stream_pending - 1)
            if status < 0:
                print(f"[TB6R5] SubLoop1 stream async failed (status={status}): {cmd[:120]}...")

        with _filter_stdout_lines(_should_drop_jog_any_j_rpc_log):
            ok = self._rpc.call_async(cmd, DEFAULT_SUBLOOP1_EXEC_TIMEOUT_MS, _on_response)
        if ok:
            with self._state_lock:
                self._subloop1_stream_pending += 1
        return bool(ok)

    def _finalize_subloop1_session(self) -> None:
        self._subloop1_active = False
        self._subloop1_exiting = False
        with self._state_lock:
            self._jog_async_pending = 0
            self._subloop1_stream_pending = 0

    def send_subloop1_exit(self, timeout_ms: int = DEFAULT_SUBLOOP1_EXIT_TIMEOUT_MS, blocking: bool = False) -> bool:
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
    ) -> bool:
        if not self._subloop1_active and not self._subloop1_exiting:
            return True
        if timeout_ms is None:
            timeout_ms = DEFAULT_SUBLOOP1_EXIT_TIMEOUT_MS
        if blocking_exit and self._subloop1_active:
            self._wait_motion_settled(settle_timeout_s)
        return self.send_subloop1_exit(timeout_ms=timeout_ms, blocking=blocking_exit)

    def _send_subloop1(
        self,
        arm_inner: str,
        gripper_distance: float | None,
        interval: float | None = None,
        max_distance: float | None = None,
        min_distance: float | None = None,
        immediate: bool | None = None,
    ) -> bool:
        if not self._ensure_command_channel():
            return False
        if immediate is None:
            immediate = self.subloop1_immediate
        if interval is None:
            interval = DEFAULT_TWO_FINGERS_GRIPPER_INTERVAL
        grip_inner = (
            NOT_RUN_EXECUTE
            if gripper_distance is None
            else self._format_gripper_inner(gripper_distance, interval, max_distance, min_distance)
        )
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
        self.exit_subloop1_if_active(timeout_ms=timeout_ms, blocking_exit=True)
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
