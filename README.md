# tb6r5_policy_infer

独立于 `xrobotoolkit_teleop` 遥操作框架的 TB6-R5 **LeRobot 策略推理与离线评估**包。

支持 LeRobot 训练的 **ACT**、**Diffusion**、**SmolVLA**、**pi0 / pi0.5 (pi05) / pi0-FAST** 等策略（由 checkpoint 内 `config.json` 的 `type` 字段自动识别）。不同策略会自动套用各自的真机部署默认（控制频率、RPC 频率、动作队列处理、语言任务需求），参数取值参考 LeRobot 各策略配置。

**真机推理数据接口（观测/动作/相机/RPC）**：[docs/ACT_Policy_Infer_Interface.md](../docs/ACT_Policy_Infer_Interface.md)

---

## 快速开始（推荐流程）

```text
1. 离线评估（验证 checkpoint + 数据集对齐）
      tb6r5-policy-eval ...

2. Dry-run（验证相机 + 推理输出，不发 RPC）
      tb6r5-policy-infer ... --dry-run

3. 真机（去掉 --dry-run，首次建议 --fps 10）
      tb6r5-policy-infer ...
```

---

## 安装

本包**不依赖** `xrobotoolkit_teleop`；TB6-R5 RPC/Topic 与 RealSense 接口已内置在 `tb6r5_policy_infer.hardware`。

```bash
# 仅离线评估
pip install -e ./tb6r5_policy_infer

# 真机推理（RealSense 相机）
pip install -e "./tb6r5_policy_infer[hardware]"
```

厂商 SDK 二进制（`rpc.so` / `topic.so`）**已内置**在 `tb6r5_policy_infer/vendor/`，安装包后即可用，无需单独下载。

### 新机器最小安装

> 说明：**TER30** 是机器人随附的 ARM 边缘计算主机（板载算力单元，通常为 **Jetson Orin**）。这里“新机器”泛指任何全新环境——TER30、其它 ARM 主机，或 x86 工作站都适用。内置 SDK 的 `.so` 按 CPython 版本编译，**支持 Python 3.10 或 3.12**（运行时按版本自动选用 `vendor/` 或 `vendor_py312/`）。其中 **3.12 的 x86-64 `rpc.so`/`topic.so` 尚未提供**（目前仅 ARM/Jetson Orin），x86 上请用 3.10。

#### Jetson / TER30（实测：JetPack 6.2 + CUDA 12.6）

本机实测环境（`/etc/nv_tegra_release`）：**L4T R36.4.3（JetPack 6.2）+ CUDA 12.6 + cuDNN 9 + Python 3.10**。

**不要用** `pip install torch`（会装 PyPI / CPU 版，`cuda.is_available()` 变 `False`）。  
也**不要**用 JetPack 6.0 的 torch 2.3 wheel（依赖 `libcudnn.so.8`，JP6.2 只有 cuDNN 9，会 `ImportError`）。

##### 0）准备虚拟环境（装到大盘，避免占满用户目录）

```bash
# 可选：关掉失效代理，否则 pip/wget 会连 127.0.0.1
unset http_proxy https_proxy ftp_proxy all_proxy HTTP_PROXY HTTPS_PROXY FTP_PROXY ALL_PROXY no_proxy NO_PROXY

# 环境放在 /home/ai（大盘），不要默认装到 ~/.local
conda create -p /home/ai/condaenv/tb6r5 python=3.10 -y
conda activate /home/ai/condaenv/tb6r5   # 或: conda activate tb6r5（若已注册）

which python   # 期望: /home/ai/condaenv/tb6r5/bin/python
which pip      # 期望: /home/ai/condaenv/tb6r5/bin/pip
```

##### 1）安装 Jetson 专用 PyTorch（CUDA 可用）

把兼容 JP6 / CUDA 12.6 的 wheel 放到例如 `/home/ai/jetson_jp6_cu126/`（本机实测可用）：


| 包           | 版本     | Wheel 文件名                                          |
| ----------- | ------ | -------------------------------------------------- |
| torch       | 2.11.0 | `torch-2.11.0-cp310-cp310-linux_aarch64.whl`       |
| torchvision | 0.26.0 | `torchvision-0.26.0-cp310-cp310-linux_aarch64.whl` |
| torchaudio  | 2.10.0 | `torchaudio-2.10.0-cp310-cp310-linux_aarch64.whl`  |


```bash
cd /home/ai/jetson_jp6_cu126
pip install torch-2.11.0-cp310-cp310-linux_aarch64.whl \
            torchvision-0.26.0-cp310-cp310-linux_aarch64.whl \
            torchaudio-2.10.0-cp310-cp310-linux_aarch64.whl

python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
# 期望类似: 2.11.0 12.6 True
```

##### 2）安装推理包 + LeRobot（勿覆盖 torch）

```bash
cd /home/ai/pi0.5/chaishan   # 仓库根；按实际路径调整

# 先只注册本包（--no-deps 不会装依赖，也不会动 torch）
python -m pip install -e "./tb6r5_policy_infer[hardware]" --no-deps

# 再补依赖。lerobot 会拉 torchvision 等；装完若 torch 变 CPU 版，立刻用上面的 wheel 盖回去
python -m pip install "lerobot==0.4.4" opencv-python matplotlib tyro

# 若 torch 被换成 2.10.0+cpu / cuda=False，重新装回 Jetson wheel：
cd /home/ai/jetson_jp6_cu126
pip install --force-reinstall --no-deps \
  torch-2.11.0-cp310-cp310-linux_aarch64.whl \
  torchvision-0.26.0-cp310-cp310-linux_aarch64.whl \
  torchaudio-2.10.0-cp310-cp310-linux_aarch64.whl
```

> **说明：** `lerobot 0.4.4` 声明 `torch<2.11`，与 JP6.2 所需的 torch 2.11 会有 pip 警告，可忽略；以 `torch.cuda.is_available()` 为准。  
> 若直接 `pip install -e "...[hardware]"`（不带 `--no-deps`），同样可能把 torch 覆盖成 PyPI CPU 版，装完务必验证 CUDA。

##### 3）自检

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
python -c "import lerobot; from tb6r5_policy_infer.hardware import TB6R5Interface; print('ok')"
pip show tb6r5_policy_infer torch lerobot | grep -E '^Name:|^Version:|^Location:|^Editable'
# Location / Editable 应在 /home/ai/...，不要出现 /home/test/.local
```

**仅测策略、不连机器人/相机：**

```bash
tb6r5-policy-infer --robot-ip 192.168.11.11 --policy-path model/... \
  --dry-run --no-camera
```

连真机或开 RealSense 直接运行即可（SDK 已随包内置）。

#### 踩坑摘要（本次 TER30 配置）


| 现象                                | 原因                                  | 处理                                                                |
| --------------------------------- | ----------------------------------- | ----------------------------------------------------------------- |
| `libcudnn.so.8: ... No such file` | torch 2.3 需 cuDNN 8，JP6.2 是 cuDNN 9 | 改用适配 CUDA 12.6 的 torch 2.11 wheel                                 |
| `torch 2.10.0+cpu` / `cuda=False` | `pip install lerobot` 拉了 PyPI torch | 用 `/home/ai/jetson_jp6_cu126/*.whl` `--force-reinstall --no-deps` |
| 包装到 `/home/test/.local`           | 未激活 conda，或用了 `--user`              | `which pip` 确认指向 `/home/ai/condaenv/tb6r5/bin/pip`                |
| `nvidia.box.com` / 代理连不上          | 国内网络或失效代理                           | `unset *_proxy`；离线拷贝 wheel 再装                                     |


#### x86 工作站

```bash
cd <仓库根>
conda activate tb6r5

pip install "lerobot==0.4.4"
pip install -e "./tb6r5_policy_infer[hardware]"
pip install tyro numpy torch opencv-python
```

### 远程机器更新

```bash
cd /home/ai/pi0.5/chaishan
git pull origin YS          # 或你的分支名
conda activate /home/ai/condaenv/tb6r5
pip install -e "./tb6r5_policy_infer[hardware]" --no-deps
# 若依赖有变再补装；最后确认 CUDA 仍可用
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

模型权重（`pretrained_model/`）需单独 `rsync`，不在 git 里。

---

## 真机推理（`tb6r5-policy-infer`）

### Checkpoint 路径

`--policy-path` 须指向 `**pretrained_model` 目录**（含 `config.json`、`model.safetensors`、processor json），不是上一级 `checkpoints/` 目录：

```text
outputs/train/tb6r5_rings_P05/checkpoints/100000/pretrained_model   ✓
model/act/080000/pretrained_model                                   ✓
outputs/train/.../checkpoints/100000/                               ✗
```

一般**不需要**传 `--dataset-root` / `--repo-id`：checkpoint 已烘焙归一化统计。

### 支持的策略类型

策略类型由 checkpoint 内 `config.json` 的 `type` 字段自动识别，无需手动指定；如需断言可加 `--policy-type <type>`（类型不匹配会报错）。加载后会按下表套用**真机部署默认**（可用 CLI 覆盖，或 `--no-policy-defaults` 回退到通用 30 Hz）。


| `type`      | 默认 `--fps` | 默认 `--arm-rpc-rate-hz` | 需要 `--task` | 可调队列参数                                                                                     | 说明            |
| ----------- | ---------- | ---------------------- | ----------- | ------------------------------------------------------------------------------------------ | ------------- |
| `act`       | 30         | 30                     | 否           | `--n-action-steps`（1…chunk_size）、`--temporal-ensemble-coeff`、`--refresh-policy-every-step` | 推理快           |
| `diffusion` | 15         | 15                     | 否           | `--n-action-steps`（1…`horizon−n_obs_steps+1`）                                              | 有观测历史 + 去噪，较慢 |
| `smolvla`   | 10         | 10                     | **是**       | `--n-action-steps`（1…chunk_size）                                                           | VLA，GPU 推理慢   |
| `pi0_fast`  | 10         | 10                     | **是**       | `--n-action-steps`（1…chunk_size）                                                           | VLA（FAST 解码）  |
| `pi0`       | 8          | 8                      | **是**       | `--n-action-steps`（1…chunk_size）                                                           | VLA，很慢        |
| `pi05`      | 8          | 8                      | **是**       | `--n-action-steps`（1…chunk_size）                                                           | VLA，很慢        |


- **夹爪 RPC** 默认所有策略均为 2 Hz。任何显式 CLI 值都会覆盖上表默认。
- **ACT 专用**参数（`--temporal-ensemble-coeff` / `--refresh-policy-every-step`）用于非 ACT 策略会直接报错。
- **VLA 策略**（smolvla/pi0/pi05/pi0_fast）必须传真实的 `--task` 语言指令，占位默认值会报错。

观测/动作布局须与训练数据集一致（TB6-R5 默认 7 维 state/action + 双相机 RGB 480×640）。

### 基本命令

**Dry-run（推荐第一步）：**

```bash
tb6r5-policy-infer \
  --robot-ip 192.168.11.11 \
  --policy-path /home/ai/pi0.5/wqp/XRcopy/model/act_rings_P02/080000/pretrained_model \
  --device cuda \
  --no-camera \
  --dry-run
```

**ACT 真机（`--fps` 不传时自动用 30）：**

```bash
tb6r5-policy-infer \
  --robot-ip 192.168.11.11 \
  --policy-path /home/ai/pi0.5/wqp/XRcopy/model/act_rings_P02/080000/pretrained_model \
  --device cuda \
  --joint-step-max-rad 0.03
```

```
(/home/ai/condaenv/tb6r5) test@TER30JB3-ubuntu:/home/ai/jetson_jp6_cu126$ tb6r5-policy-infer   --robot-ip 192.168.11.11   --policy-path /home/ai/pi0.5/wqp/XRcopy/model/act_rings_P02/080000/pretrained_model   --device cuda   --camera-devices 'realsense_0=/dev/video2,realsense_1=/dev/video8'   --dry-run --no-show-camera
```
**Diffusion 真机（P05 示例，夹爪为 mm，勿加 `--gripper-normalized`；默认 fps=15）：**

```bash
tb6r5-policy-infer \
  --robot-ip 192.168.11.11 \
  --policy-path /home/ai/chaishan/model/dp-P05/checkpoints/100000/pretrained_model \
  --device cuda \
  --dry-run    # 验证通过后再去掉
```

**SmolVLA 真机（VLA，必须给 `--task`；默认 fps=10）：**

```bash
tb6r5-policy-infer   --robot-ip 192.168.11.11   --policy-path /home/ai/chaishan/model/smolvla-rings/checkpoints/020000/pretrained_model/   --device cuda   --joint-step-max-rad 0.03   --camera-urls 'realsense_0=http://192.168.11.11:8888/RsCameraSensor/1/0/color,realsense_1=http://192.168.11.11:8888/RsCameraSensor/0/0/color'   --fps 20 --task "Pick up the purple ring and place it on the white plate." --dry-run
```

**pi0.5 真机（`type=pi05`，同为 VLA，必须给 `--task`；默认 fps=8）：**

```bash
tb6r5-policy-infer \
  --robot-ip 192.168.11.11 \
  --policy-path /home/ai/pi0.5/tzh/openpi/checkpoints/pi05_tb6r5/tb6r5_lora_3k/1500 \
  --device cuda \
  --task "Pick up the purple ring and place it on the white plate." --dry-run
```

> **VLA 提示：** smolvla / pi0 / pi05 / pi0_fast 在算力有限的 GPU 上单帧推理较慢，默认已下调 `--fps`。若跟不上可进一步降低 `--fps` 或用 `--n-action-steps` 拉长动作队列。`--task` 文本应与训练时数据集的 `single_task` 一致。

也可用模块方式运行：

```bash
python -m tb6r5_policy_infer.cli --robot-ip 192.168.11.11 --policy-path ... --dry-run
```

完整参数：`tb6r5-policy-infer --help`

### 观测 / 动作约定


| 字段                                 | 形状        | 单位 / 说明                                                |
| ---------------------------------- | --------- | ------------------------------------------------------ |
| `observation.state`                | 7         | `[q0..q5 rad, gripper]`；夹爪默认 **mm** 反馈                 |
| `observation.images.realsense_0/1` | HWC uint8 | 480×640×3 RGB；逻辑名须与训练集一致                               |
| `action`                           | 7         | `[q0..q5 rad, gripper]`；下发前经 `--joint-step-max-rad` 限幅 |


训练时夹爪为 **[0,1] 归一化** 的数据集，加 `--gripper-normalized`（见下文「夹爪单位」）。

### 安全与生命周期


| 参数                                         | 默认       | 说明                             |
| ------------------------------------------ | -------- | ------------------------------ |
| `--dry-run`                                | 关        | 只推理打印，不发 RPC                   |
| `--print-rpc`                              | 关        | 每次发送时打印完整 SubLoop1 RPC 指令（dry-run 打印"本应发送"的指令），不受 `--print-every` 限流 |
| `--fps`                                    | 按策略      | 控制循环频率；默认按策略类型自动选择（见上表），显式传值覆盖 |
| `--joint-step-max-rad`                     | 0.03     | 策略输出限幅：每步最大关节变化（rad）           |
| `--home-joint-deg`                         | 遥操作 home | 启动 / Ctrl+C 复位姿态（度）            |
| `--home-settle-time`                       | 3 s      | 复位后等待                          |
| `--no-home-on-start` / `--no-home-on-exit` | 关        | 跳过启动或退出复位                      |
| `--print-every`                            | 0.5 s    | 调试打印间隔                         |


退出：`Ctrl+C` → 停相机 →（可选）复位 → `arm.disable()`。

### JogAnyJ 运动参数（RPC 层）

策略输出的关节目标经 `--joint-step-max-rad` 限幅后，通过 SubLoop1 `JogAnyJ` 下发。下列参数写入 RPC 命令的 `--joint_vel` / `--joint_acc` / `--joint_dec` / `--zone_ratio`，控制**机器人执行**时的运动学限制。**推理包默认取保守值 `1 / 1 / 1 / 0`（比遥操作更慢更稳，便于首次真机验证）**；确认安全后再逐步调大。


| 参数             | 默认  | 说明                             |
| -------------- | --- | ------------------------------ |
| `--joint-vel`  | 1.0 | `JogAnyJ` 关节速度（保守默认，首次真机稳）     |
| `--joint-acc`  | 1.0 | `JogAnyJ` 关节加速度                |
| `--joint-dec`  | 1.0 | `JogAnyJ` 关节减速度                |
| `--zone-ratio` | 0.0 | `JogAnyJ` 过渡区比例（0=逐点到位，不做圆弧过渡） |


```bash
# 默认即最慢最稳（1/1/1/0），无需额外传参

# 确认安全后适度加快执行
tb6r5-policy-infer ... \
  --joint-vel 4.0 --joint-acc 2.0 --joint-dec 2.0 \
  --zone-ratio 0.05

# 策略跟得更紧（软件层）+ 机器人跑得快（RPC 层）
tb6r5-policy-infer ... \
  --joint-step-max-rad 0.05 --fps 20 \
  --joint-vel 8.0 --joint-acc 4.0 --joint-dec 4.0
```

**调参区分：**

- 策略「算得慢 / 步幅小」→ 调 `--joint-step-max-rad`、`--fps`、ACT 的 `--n-action-steps`
- 机器人「执行拖沓 / 跟不上目标」→ 调 `--joint-vel` / `--joint-acc` / `--joint-dec`
- 运动不够顺滑 → 适当增大 `--zone-ratio`（与采集时 `--zone-ratio` 保持一致更安全）

实现位置：`runner.py` → `tb6r5_policy_infer.hardware.tb6r5`（底层 RPC 使用内置 `send_commend_py/rpc_py_all`）。

### RPC 与夹爪


| 参数                       | 默认              | 说明                                      |
| ------------------------ | --------------- | --------------------------------------- |
| `--arm-rpc-rate-hz`      | 按策略（匹配 `--fps`） | 臂 SubLoop1 下发频率                         |
| `--gripper-rpc-rate-hz`  | 2               | 夹爪 SubLoop1 下发频率                        |
| `--gripper-max-distance` | 70              | 全开距离（mm）                                |
| `--gripper-min-distance` | 30              | 全合距离（mm）                                |
| `--gripper-continuous`   | 开               | 连续 mm；`--no-gripper-continuous` 为滞回二值模式 |
| `--gripper-cmd-delta`    | 0.5             | 夹爪指令变化小于此值（mm）不重发 RPC                   |


### ACT 部署调参（无需重新训练）b

三者互斥注意：**不要**同时开 `--temporal-ensemble-coeff` 与 `--refresh-policy-every-step`。


| 模式       | 参数                               | 何时重推理                | 适用              |
| -------- | -------------------------------- | -------------------- | --------------- |
| **默认队列** | （不传）                             | 每 `n_action_steps` 步 | 与训练一致           |
| **缩短队列** | `--n-action-steps N`             | 每 N 步（1…chunk_size）  | 更灵敏             |
| **时间集成** | `--temporal-ensemble-coeff 0.01` | 每步推理 + 融合            | 更平滑（原论文常用 0.01） |
| **每步刷新** | `--refresh-policy-every-step`    | 每步 `policy.reset()`  | 最灵敏、最慢          |


```bash
# 更灵敏：每 10 步重推理（chunk_size=50 时）
tb6r5-policy-infer ... --n-action-steps 10 --fps 20

# 更平滑：Temporal Ensemble
tb6r5-policy-infer ... --temporal-ensemble-coeff 0.01 --fps 20
```

> `--temporal-ensemble-coeff` / `--refresh-policy-every-step` 为 **ACT 专用**，对其它策略传入会直接报错。

**其它策略的队列调参：**

- **Diffusion**：队列由 checkpoint 内 `n_obs_steps`、`n_action_steps`、`horizon` 控制；`--n-action-steps` 取值范围 `1…(horizon − n_obs_steps + 1)`。
- **SmolVLA / pi0 / pi05 / pi0_fast**：chunk 队列策略，`--n-action-steps` 取值范围 `1…chunk_size`；拉长队列减少推理次数（更省算力、更平滑），缩短队列更灵敏。这些策略额外需要 `--task` 语言指令。

### 相机

推理时图像写入 `observation.images.<name>`（RGB HWC `uint8`，默认 640×480）。**逻辑名**（如 `realsense_0`）须与训练数据集一致；变的只是如何打开物理设备。


| 模式                     | 参数                                              | 依赖                         |
| ---------------------- | ----------------------------------------------- | -------------------------- |
| **RealSense（默认）**      | 默认 SN 在 `constants.py`；可用 `--camera-serials` 覆盖 | `pyrealsense2`             |
| **V4L2 `/dev/video*`** | `--camera-devices`                              | `opencv-python`            |
| **HTTP URL**           | `--camera-urls`                                 | `urllib` + `opencv-python` |
| **无相机**                | `--no-camera`                                   | 喂黑图，仅测 RPC                 |


**RealSense — 指定序列号：**

```bash
tb6r5-policy-infer ... \
  --camera-serials 'realsense_0=135522071053,realsense_1=244222075136'
```

查本机 SN：

```bash
rs-enumerate-devices | grep Serial
```

**V4L2：**

```bash
tb6r5-policy-infer ... \
  --camera-devices 'realsense_0=/dev/video0,realsense_1=/dev/video4'
```

指定 `--camera-devices` 时**忽略** `--camera-serials`。

**HTTP 远程相机（RsCameraSensor 等）：**

```bash
tb6r5-policy-infer ... \
  --camera-urls 'realsense_0=http://192.168.2.42:8888/RsCameraSensor/0/0/color,realsense_1=http://192.168.2.42:8888/RsCameraSensor/1/0/color'
```

- URL 响应须为 JPEG/PNG 字节
- 指定 `--camera-urls` 时忽略 `--camera-serials` 与 `--camera-devices`

其它：`--camera-width/height/fps`（默认 640×480×30）、`--show-camera` / `--no-show-camera`、`--camera-preview-fps`。

### 夹爪单位

- **默认（mm）**：`action[6]` / `state[6]` 为 mm（0=闭合，`--gripper-max-distance` 默认 70=全开）
- **归一化训练集**：加 `--gripper-normalized`（obs: `feedback_mm/max`，action: `norm×max` 再下发）

### 机器人 SDK（已内置）

真机 RPC/Topic 依赖的 `.so` 已打包在 `tb6r5_policy_infer/vendor/`，`pip install` 后自动使用。

**支持架构（自动按 `platform.machine()` 选择，均自包含）：**

- **x86-64**（普通工作站）
- **aarch64 / ARM64**（TER30 等 ARM 主机、**NVIDIA Jetson Orin**）

RPC（`rpc.so`）、Topic（`topic.so`/`libprotobuf`/`libzmq`）两个架构都已内置且仅依赖标准系统库，开箱即用。`libzmq` 采用架构感知加载：内置版架构与主机一致时用内置，否则回退系统 `libzmq.so.5`（`apt-get install libzmq5`）。

覆盖路径（可选）：

```bash
export TB6R5_DEPS_ROOT=/path/to/dependencies   # 含 hello_demo_py/ 与 get_status_py/
```


| 用途        | 包内路径（`<vendor>` = `vendor/` 或 `vendor_py312/`） |
| --------- | ---------------------------------------------- |
| RPC 发指令   | `/send_commend_py/rpc_py_all/lib/linux/{x86    |
| Topic 读状态 | `/get_status_py/topic_all_py/lib/{x86          |


**Python 版本与 SDK 树（自动选择）：** `.so` 按 CPython 版本编译，运行时由 `sdk_paths.py` 依据 `sys.version_info` 选用：


| 运行的 Python | 使用的树            | x86-64       | aarch64（ARM/Orin） |
| ---------- | --------------- | ------------ | ----------------- |
| 3.10       | `vendor/`       | ✓            | ✓                 |
| 3.12       | `vendor_py312/` | ⏳ 待补（用 3.10） | ✓                 |


3.12 的 x86-64 `rpc.so`/`topic.so` 尚未提供（配套 `libzmq`/`libprotobuf` 已就位）；补齐后放到 `vendor_py312/.../lib/.../` 对应目录即可（见该目录下 `PENDING_PY312.md`）。其它 Python 版本（如 3.11/3.13）不受支持，导入时会报 ABI 错误。

```bash
python -c "from tb6r5_policy_infer.hardware.sdk_paths import dependencies_root; print(dependencies_root())"
python scripts/hardware/verify_tb6r5_sdk.py --robot-ip 192.168.11.11 --send-test-cmd
```

---

## 离线评估（`tb6r5-policy-eval`）

在已有 LeRobot 数据集上计算 action **MAE**，输出推理速度，并生成对比曲线图。

### 基本用法

```bash
tb6r5-policy-eval \
  --policy-path model/act/080000/pretrained_model \
  --dataset-root data/lerobot/tb6r5_yellow_yogurt_47_v3 \
  --repo-id shanchai/tb6r5_yellow_yogurt_47_v3 \
  --device cuda
```

**Diffusion 示例（P05）：**

```bash
tb6r5-policy-eval \
  --policy-path outputs/train/tb6r5_rings_P05/checkpoints/100000/pretrained_model \
  --dataset-root data/lerobot/tb6r5_rings/P05 \
  --repo-id local/P05 \
  --device cuda \
  --max-samples 50 \
  --stride 30 \
  --warmup-samples 2
```

### 采样与速度参数


| 参数                 | 默认                           | 说明                            |
| ------------------ | ---------------------------- | ----------------------------- |
| `--max-samples`    | 1000                         | 最多评估多少个采样点                    |
| `--stride`         | 5                            | 每隔 N 帧取 1 帧（越大越快、统计越稀疏）       |
| `--warmup-samples` | 10                           | 预热帧数，**不计入**末尾 FPS 统计（但仍消耗时间） |
| `--benchmark-only` | 关                            | 只测速度，不算 MAE、不画图               |
| `--bench-samples`  | 200                          | `--benchmark-only` 模式下计时的推理次数 |
| `--output-dir`     | `outputs/eval_act/<dataset>` | 曲线图输出目录                       |


采样逻辑：`range(0, n_frames, stride)[:max_samples]`。数据集 30 fps 时 `--stride 30` ≈ 每秒取 1 帧。

**Diffusion 评估较慢**：每帧都会 `policy.reset()` 并做完整去噪（默认 ~100 步），GPU 占满是正常现象。快速抽查建议 `--max-samples 50 --stride 30 --warmup-samples 2`。

**只测推理速度：**

```bash
tb6r5-policy-eval \
  --policy-path outputs/train/tb6r5_rings_P05/checkpoints/100000/pretrained_model \
  --dataset-root data/lerobot/tb6r5_rings/P05 \
  --repo-id local/P05 \
  --device cuda \
  --benchmark-only \
  --bench-samples 30 \
  --warmup-samples 3
```

输出图：`action_comparison.png`、`mae_per_dim.png`、`error_over_time.png`。

---

## 依赖版本

本包当前固定 **lerobot 0.5.1**（见 `pyproject.toml`）。SmolVLA / pi0 等策略依赖 `transformers` / `huggingface-hub`，版本须配套。

### lerobot 0.5.1（当前默认）

```bash
pip install "lerobot==0.5.1" "transformers>=5.3.0,<6.0.0" "huggingface-hub>=1.16.0,<2.0.0"
pip install -e ./tb6r5_policy_infer --no-deps
```

**Jetson / TER30**：必须先装 Jetson 专用 torch wheel，再装 lerobot；`pyproject.toml` 不含 `torch`，避免 pip 误装 PyPI 版。详见上文「Jetson / TER30」安装流程。

若环境里已有 `huggingface-hub>=1.0` 但 `transformers<5`，导入时会报版本冲突；升级 transformers 即可：

```bash
pip install "transformers>=5.3.0,<6.0.0"
```

### lerobot 0.4.4（旧版，可选回退）

```bash
pip install "lerobot==0.4.4"
pip install "transformers>=4.57.1,<5.0.0" "huggingface-hub>=0.34.2,<0.36.0"
pip install -e ./tb6r5_policy_infer --no-deps
```

注意同时改 `pyproject.toml` 中的 `lerobot==...`，并确认与当前 Jetson torch 兼容。

### 常见报错

```text
DecodingError: The fields `use_peft` are not valid for ACTConfig
```

**修复：** 升级 lerobot，或同步最新 `tb6r5_policy_infer`（`load_pretrained_config` 会自动忽略未知字段）。

```text
ImportError: huggingface-hub>=0.34.0,<1.0 is required ... but found huggingface-hub==1.22.0
```

**0.5 环境修复（当前默认）：**

```bash
pip install "transformers>=5.3.0,<6.0.0" "huggingface-hub>=1.16.0,<2.0.0"
```

```text
ImportError: cannot import name 'is_offline_mode' from 'huggingface_hub'
```

**0.4 环境修复：**

```bash
pip install "transformers>=4.57.1,<5.0.0" "huggingface-hub>=0.34.2,<0.36.0"
```

---

## 包结构


| 模块               | 说明                                                           |
| ---------------- | ------------------------------------------------------------ |
| `cli`            | 真机推理 CLI（`tb6r5-policy-infer`）                               |
| `eval_cli`       | 离线评估 CLI（`tb6r5-policy-eval`）                                |
| `runner`         | 硬件控制循环                                                       |
| `policy`         | 模型加载与各策略推理 override（ACT/Diffusion/SmolVLA/pi0/pi05/pi0_fast） |
| `deploy`         | 策略类型识别、每策略真机部署默认（fps/RPC/队列/task）与 CLI 校验                    |
| `camera`         | RealSense、V4L2、HTTP 采集                                       |
| `gripper`        | 夹爪观测与指令辅助                                                    |
| `hardware/`      | TB6-R5 RPC/Topic（`tb6r5.py`）与 RealSense 相机                   |
| `lerobot_compat` | LeRobot 版本兼容与 config 加载                                      |


真机部署只需更新本推理包；SDK 已内置在 `tb6r5_policy_infer/vendor/`（也可用 `TB6R5_DEPS_ROOT` 覆盖）。

等价的 legacy 入口：`scripts/hardware/policy_infer_tb6r5_act.py`（推荐统一使用 `tb6r5-policy-infer`）。
