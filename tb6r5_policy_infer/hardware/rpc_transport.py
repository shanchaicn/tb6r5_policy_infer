"""RPC transport based on send_commend_py (rpc_client.py + rpc.so)."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Callable

from .sdk_paths import require_module, rpc_module_path, rpc_py_all_root

_RPC_MODULE = None


def _load_rpc_module():
    global _RPC_MODULE
    if _RPC_MODULE is not None:
        return _RPC_MODULE

    require_module(rpc_module_path(), "rpc")

    rpc_root = rpc_py_all_root()
    if rpc_root not in sys.path:
        sys.path.insert(0, rpc_root)

    # Triggers platform lib path setup inside vendored rpc_client.py (if present).
    try:
        importlib.import_module("rpc_client")
    except ModuleNotFoundError:
        # Newer (pybind) bundles ship a self-contained rpc.so without rpc_client.py;
        # ensure the module dir is importable directly.
        rpc_dir = str(Path(rpc_module_path()).parent)
        if rpc_dir not in sys.path:
            sys.path.insert(0, rpc_dir)
    _RPC_MODULE = importlib.import_module("rpc")
    return _RPC_MODULE


class RpcSession:
    """Thin session wrapper over rpc.CPPClient (send_commend_py API)."""

    def __init__(self, ip: str, port: int = 5868, connect_timeout_ms: int = 3000):
        rpc = _load_rpc_module()
        self._rpc = rpc
        self._client = rpc.CPPClient(ip, int(port), int(connect_timeout_ms))
        self._seq_id = 0

    @property
    def inner(self):
        return self._client

    def is_connected(self) -> bool:
        return bool(self._client.IsConnected())

    def error_info(self) -> str:
        return str(self._client.GetErrorInfo())

    def new_msg(self, cmd: str):
        msg = self._rpc.Msg(cmd)
        msg.setMsgID(10001)
        self._seq_id += 1
        msg.setMsgSeqID(self._seq_id)
        return msg

    def call_await(self, cmd: str, timeout_ms: int = 5000):
        msg = self.new_msg(cmd)
        return self._client.CallAwait(msg, timeout_ms)

    def call_async(self, cmd: str, timeout_ms: int, callback: Callable, *, expect_resp: bool = True) -> bool:
        msg = self.new_msg(cmd)
        return bool(self._client.CallAsync(msg, timeout_ms, callback, expect_resp))
