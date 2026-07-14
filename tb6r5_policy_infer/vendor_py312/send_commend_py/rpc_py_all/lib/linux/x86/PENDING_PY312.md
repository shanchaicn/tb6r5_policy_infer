# x86 (Python 3.12) rpc.so — PENDING

The Python 3.12 **x86-64** build of `rpc.so` is not bundled yet.

Only the aarch64 (ARM/Jetson Orin) 3.12 `rpc.so` is currently shipped.
Drop the x86-64 3.12 build here as `rpc.so` to enable x86 on Python 3.12:

    lib/linux/x86/rpc.so   (ELF x86-64, built for CPython 3.12)

Until then, run x86 on Python 3.10 (uses ../../../../vendor/) or use ARM/Orin.
