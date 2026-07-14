# hello_demo_py — 机器人 RPC Python 控制示例

基于 `cpp_rpc` 通信框架，通过 pybind11 封装的 Python 模块实现对机器人的远程控制。

## 目录结构

```
hello_demo_py/
├── README.md
└── rpc_py_all/
    ├── main.py          ← 入口脚本，修改 IP 即可运行
    ├── rpc_client.py    ← RPC 工具模块（连接、同步/异步发送）
    └── lib/             ← 预编译的 rpc 动态库
        ├── win/
        │   └── rpc.pyd           (Windows, MSVC)
        ├── linux/
        │   ├── x86/
        │   │   └── rpc.so        (Linux x86_64, g++ 9.4)
        │   └── arm/
        │       └── rpc.so        (Linux ARM, g++ 11.4)
```

## 环境要求

| 项目   | 说明                      |
| ------ | ------------------------- |
| Python | 3.10                      |
| 系统   | Windows / Linux x86_64 / Linux ARM |

> **注意：** 动态库（`.pyd` / `.so`）基于 Python 3.10 编译，使用其他 Python 版本可能导致 `import` 失败。

## 快速开始

### 1. 修改目标机器人 IP

打开 [main.py](rpc_py_all/main.py)，修改第 10 行：

```python
ROBOT_IP = "192.168.11.11"   # ← 改为目标机器人 IP
```

### 2. 运行

```bash
cd rpc_py_all
python main.py
```

程序会自动检测当前平台并加载对应的 `rpc` 动态库，无需手动配置路径。

## 文件说明

### main.py — 入口脚本

定义了三组指令和主流程：

| 变量           | 用途                               |
| -------------- | ---------------------------------- |
| `INIT_CMDS`    | 初始化指令序列，连接后首先执行     |
| `MOTION_CMDS`  | 运动指令（示例中被注释）           |
| `SPEEDL_CMDS`  | 末端线性速度控制指令，循环执行 3 次 |
| `YOUR_CMDS`    | 用户自定义指令，预留空列表         |

### rpc_client.py — RPC 工具模块

封装了平台检测、动态库导入、连接管理和请求发送，核心组件：

#### RpcClient（客户端连接）

```python
from rpc_client import RpcClient

client = RpcClient("192.168.11.11")            # 端口固定 5868
client = RpcClient("192.168.11.11", timeout_ms=5000)  # 自定义连接超时
print(client.is_connected())                   # 是否已连接
print(client.error_info())                     # 错误信息

# 构造消息
msg = client.new_msg("{Enable}")
msg = client.new_msg("{Var --type=jointtarget --name=j0 --value={0,0,0,0,0,0,0,0,0,0}}")
```

#### send_rpcsy — 同步发送

```python
send_rpcsy(client, cmd_list, sleep_s=0.1, timeout_ms=5000)
```

逐条发送，等待每条响应后发下一条。适用于初始化等强依赖顺序的场景。

#### send_rpc_async — 异步发送

```python
send_rpc_async(client, cmd_list, wait_s=0.5, timeout_ms=10000)
```

逐条发送，不等响应即发下一条，响应通过回调打印。适用于实时控制等不依赖响应的场景。

> **注意：** 异步模式下 `timeout_ms` 是等待回调的超时（由 C++ 端超时线程处理）。由于 `PushCallback` 是 C 函数指针，pybind11 无法暴露 Python 端的 fire-and-forget 接口；如需发后不理，使用 `CallAsyncRaw(expect_resp=False)`（见下方高级用法）。

## API 参考

### RpcClient

| 方法                        | 说明                           |
| --------------------------- | ------------------------------ |
| `RpcClient(ip, timeout)`    | 构造，端口固定 5868            |
| `is_connected() -> bool`    | 返回连接状态                   |
| `error_info() -> str`       | 返回最近错误信息               |
| `new_msg(cmd) -> rpc.Msg`   | 创建消息（自动设置 ID 和序列号） |

### rpc.Msg

| 方法                       | 说明        |
| -------------------------- | ----------- |
| `Msg(cmd: str)`            | 构造消息    |
| `setMsgID(int)`            | 设置消息 ID |
| `setMsgSeqID(int)`         | 设置序列号  |

### rpc.CPPClient（底层对象，通过 `client.inner` 访问）

| 方法                                                         | 说明                   |
| ------------------------------------------------------------ | ---------------------- |
| `CallAwait(msg, timeout) -> (status, list[CommResp])`        | 同步调用，等待响应     |
| `CallAwaitRaw(msg, timeout) -> (status, str)`                | 同步调用，返回原始字符串 |
| `CallAsync(msg, timeout, callback, expect_resp) -> bool`     | 异步调用               |
| `CallAsyncRaw(msg, timeout, callback, expect_resp) -> bool`  | 异步调用原始模式       |
| `IsConnected() -> bool`                                      | 连接状态               |
| `GetErrorInfo() -> str`                                      | 错误信息               |

### rpc.CommResp

| 属性      | 类型  | 说明       |
| --------- | ----- | ---------- |
| `code`    | int   | 返回码     |
| `index`   | int   | 子命令索引 |
| `message` | str   | 返回消息   |

## 自定义指令

在 `main.py` 中编辑 `YOUR_CMDS` 列表，或在主流程中直接调用：

```python
# 方式一：添加到 YOUR_CMDS
YOUR_CMDS = [
    "{MyCmd --param=value}",
]

# 方式二：在主流程中调用
send_rpcsy(client, ["{Enable}", "{Start}"], timeout_ms=1000)
send_rpcsy(client, YOUR_CMDS, sleep_s=0.1, timeout_ms=5000)
```

## 平台支持

| 平台             | 状态        | 说明                          |
| ---------------- | ----------- | ----------------------------- |
| Windows x86_64   | ✅ 已编译   | `rpc.pyd`，基于 wepoll        |
| Linux x86_64     | ✅ 已编译   | `rpc.so`，基于 epoll          |
| Linux ARM        | ✅ 已编译   | `rpc.so`，基于 epoll          |

## 常见问题

**Q: `import rpc` 报 `ImportError`？**

检查 Python 版本是否为 3.10，以及对应平台的 `.pyd` / `.so` 文件是否存在于 `lib/<平台>/` 目录。

**Q: 连接失败？**

- 确认机器人 IP 和端口 5868 可达（`ping`、`telnet`）
- 检查防火墙设置
- 增大 `connect_timeout_ms` 参数

**Q: Windows 下报 `DLL load failed`？**

安装 [VC++ 2015-2022 运行库](https://aka.ms/vs/17/release/vc_redist.x64.exe)。
