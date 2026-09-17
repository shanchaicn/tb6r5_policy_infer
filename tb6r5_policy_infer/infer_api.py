"""HTTP API for starting/stopping ``tb6r5-policy-infer`` (LeRobot env compatible).

Does not import lerobot/torch. The subprocess uses the same conda env's
``tb6r5-policy-infer`` entry point.

Run::

    tb6r5-infer-api --host 0.0.0.0 --port 8005
    python -m tb6r5_policy_infer.infer_api
    python infer_api.py
"""

from __future__ import annotations

import argparse
import os
import shlex
import shutil
import signal
import subprocess
import threading
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncIterator, TextIO

import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import PlainTextResponse


# Repo root (this file lives in tb6r5_policy_infer/tb6r5_policy_infer/).
PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = PROJECT_DIR / "configs" / "act.yaml"
DEFAULT_LOG_FILE = PROJECT_DIR / "logs" / "inference.log"

INFER_COMMAND = os.environ.get("TB6R5_INFER_COMMAND", "tb6r5-policy-infer")
CONFIG_FILE = Path(os.environ.get("TB6R5_INFER_CONFIG", str(DEFAULT_CONFIG))).resolve()
LOG_FILE = Path(os.environ.get("TB6R5_INFER_LOG", str(DEFAULT_LOG_FILE))).resolve()
STOP_TIMEOUT_SECONDS = float(os.environ.get("TB6R5_INFER_STOP_TIMEOUT", "10"))

_lock = threading.RLock()
_process: subprocess.Popen[str] | None = None
_log_handle: TextIO | None = None
_started_at: str | None = None
_stopped_at: str | None = None
_last_pid: int | None = None
_last_returncode: int | None = None


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _command() -> list[str]:
    """Build an argv list without invoking a shell."""
    command = shlex.split(INFER_COMMAND)
    if not command:
        raise RuntimeError("TB6R5_INFER_COMMAND cannot be empty")
    return [*command, "--config", str(CONFIG_FILE)]


def _executable_exists(command: list[str]) -> bool:
    executable = command[0]
    if os.sep in executable:
        path = Path(executable)
        return path.is_file() and os.access(path, os.X_OK)
    return shutil.which(executable) is not None


def _finalize_locked(process: subprocess.Popen[str]) -> None:
    """Store the exit result and release the log descriptor."""
    global _process, _log_handle, _stopped_at, _last_returncode

    if _process is not process:
        return
    returncode = process.poll()
    if returncode is None:
        return
    _last_returncode = returncode
    _stopped_at = _utc_now()
    _process = None
    if _log_handle is not None:
        _log_handle.close()
        _log_handle = None


def _watch_process(process: subprocess.Popen[str]) -> None:
    process.wait()
    with _lock:
        _finalize_locked(process)


def _refresh_locked() -> None:
    if _process is not None and _process.poll() is not None:
        _finalize_locked(_process)


def _status_locked() -> dict[str, Any]:
    _refresh_locked()
    running = _process is not None
    return {
        "status": "running" if running else ("exited" if _last_pid is not None else "idle"),
        "running": running,
        "pid": _process.pid if running else _last_pid,
        "returncode": None if running else _last_returncode,
        "started_at": _started_at,
        "stopped_at": None if running else _stopped_at,
        "command": _command(),
        "config": str(CONFIG_FILE),
        "log_file": str(LOG_FILE),
    }


def start_inference() -> dict[str, Any]:
    global _process, _log_handle, _started_at, _stopped_at
    global _last_pid, _last_returncode

    with _lock:
        _refresh_locked()
        if _process is not None:
            raise HTTPException(status_code=409, detail="Inference is already running")
        if not CONFIG_FILE.is_file():
            raise HTTPException(status_code=500, detail=f"Config file not found: {CONFIG_FILE}")

        command = _command()
        if not _executable_exists(command):
            raise HTTPException(
                status_code=500,
                detail=f"Inference executable not found or not executable: {command[0]}",
            )

        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        log_handle = LOG_FILE.open("a", encoding="utf-8", buffering=1)
        started_at = _utc_now()
        log_handle.write(
            f"\n[{started_at}] Starting inference\n"
            f"cwd={PROJECT_DIR}\n"
            f"command={shlex.join(command)}\n"
        )

        environment = os.environ.copy()
        environment.setdefault("PYTHONUNBUFFERED", "1")
        try:
            process = subprocess.Popen(
                command,
                cwd=PROJECT_DIR,
                env=environment,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )
        except OSError as exc:
            log_handle.write(f"[{_utc_now()}] Failed to start: {exc}\n")
            log_handle.close()
            raise HTTPException(status_code=500, detail=f"Failed to start inference: {exc}") from exc

        _process = process
        _log_handle = log_handle
        _started_at = started_at
        _stopped_at = None
        _last_pid = process.pid
        _last_returncode = None
        threading.Thread(target=_watch_process, args=(process,), daemon=True).start()
        return _status_locked()


def _signal_process_group(process: subprocess.Popen[str], sig: signal.Signals) -> None:
    try:
        os.killpg(process.pid, sig)
    except ProcessLookupError:
        pass


def stop_inference() -> dict[str, Any]:
    with _lock:
        _refresh_locked()
        if _process is None:
            raise HTTPException(status_code=409, detail="Inference is not running")

        process = _process
        _signal_process_group(process, signal.SIGINT)
        try:
            process.wait(timeout=STOP_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            _signal_process_group(process, signal.SIGTERM)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                _signal_process_group(process, signal.SIGKILL)
                process.wait(timeout=5)

        _finalize_locked(process)
        return _status_locked()


def _tail_lines(path: Path, count: int) -> str:
    """Read the last *count* text lines without loading an unbounded log."""
    if not path.exists():
        return ""

    block_size = 8192
    data = bytearray()
    with path.open("rb") as stream:
        stream.seek(0, os.SEEK_END)
        position = stream.tell()
        while position > 0 and data.count(b"\n") <= count:
            size = min(block_size, position)
            position -= size
            stream.seek(position)
            data[:0] = stream.read(size)
    return b"\n".join(bytes(data).splitlines()[-count:]).decode("utf-8", errors="replace")


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    yield
    # Let the inference runner handle Ctrl+C and perform its normal robot cleanup.
    with _lock:
        running = _process is not None and _process.poll() is None
    if running:
        try:
            stop_inference()
        except Exception:
            pass


app = FastAPI(
    title="TB6-R5 Policy Inference API",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/act/inference/start", status_code=201, include_in_schema=False)
@app.post("/act/inference/start", status_code=201, include_in_schema=False)
@app.post("/inference/start", status_code=201)
def inference_start() -> dict[str, Any]:
    """Start ``tb6r5-policy-infer --config configs/act.yaml``."""
    return start_inference()


@app.post("/api/act/inference/stop", include_in_schema=False)
@app.post("/act/inference/stop", include_in_schema=False)
@app.post("/inference/stop")
def inference_stop() -> dict[str, Any]:
    """Gracefully stop the active inference process."""
    return stop_inference()


@app.get("/api/act/inference/status", include_in_schema=False)
@app.get("/act/inference/status", include_in_schema=False)
@app.get("/inference/status")
def inference_status() -> dict[str, Any]:
    with _lock:
        return _status_locked()


@app.get("/api/act/inference/logs", response_class=PlainTextResponse, include_in_schema=False)
@app.get("/act/inference/logs", response_class=PlainTextResponse, include_in_schema=False)
@app.get("/inference/logs", response_class=PlainTextResponse)
def inference_logs(
    lines: int = Query(default=200, ge=1, le=5000, description="Number of trailing lines"),
) -> str:
    return _tail_lines(LOG_FILE, lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="TB6-R5 inference process API")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8005)
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
