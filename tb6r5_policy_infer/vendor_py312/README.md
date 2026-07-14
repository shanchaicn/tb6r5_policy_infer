# TB6-R5 vendor SDK (bundled with tb6r5_policy_infer)

Self-contained robot SDK — no separate download required.

| Path | Contents |
|------|----------|
| `send_commend_py/rpc_py_all/` | **send_commend_py** RPC SDK (`rpc_client.py`, `rpc.so` x86/arm) |
| `get_status_py/topic_all_py/` | Topic feedback (`topic.so`, libprotobuf, libzmq, Python helpers) |

## Architecture support

All binaries ship for both **x86-64** and **aarch64 (ARM64, incl. NVIDIA Jetson
Orin)** and are self-contained (only depend on standard system libs — libc,
libstdc++, libgcc_s, libpthread). The correct arch is picked automatically at
runtime from `platform.machine()`:

- `send_commend_py/rpc_py_all/lib/linux/{x86,arm}/rpc.so`
- `get_status_py/topic_all_py/lib/{x86,arm}/{topic.so,libprotobuf.so.32,libzmq.so.5}`

`libzmq` is loaded arch-aware: the bundled copy is used when its ELF arch
matches the host, otherwise the loader falls back to a system `libzmq.so.5`
(`apt-get install libzmq5`). All binaries require **Python 3.10**.

Override the whole tree with `TB6R5_DEPS_ROOT` (must contain both trees above).
