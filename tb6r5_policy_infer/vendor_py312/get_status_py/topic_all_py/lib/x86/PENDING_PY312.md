# x86 (Python 3.12) topic.so — PENDING

The Python 3.12 **x86-64** build of `topic.so` is not bundled yet.
The matching x86-64 `libzmq.so.5` / `libprotobuf.so.32` are already here.

Only the aarch64 (ARM/Jetson Orin) 3.12 `topic.so` is currently shipped.
Drop the x86-64 3.12 build here as `topic.so` to enable x86 Topic feedback on 3.12:

    lib/x86/topic.so   (ELF x86-64, built for CPython 3.12)

Until then, run x86 on Python 3.10 (uses ../../../../vendor/) or use ARM/Orin.
