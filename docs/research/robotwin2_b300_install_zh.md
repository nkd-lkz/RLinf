# RoboTwin 2.0 在 B300 服务器上的安装与验收

更新日期：2026-09-07

本文记录当前服务器上已经执行并通过验收的 RoboTwin 2.0 安装。代码、Python、CUDA 工具链、缓存、资产、日志和输出都位于 `/data/kaize/rlinf/robotwin2`。安装没有修改系统 Python，也没有把模型或资产写入根分区。

## 1. 当前安装结果

| 项目 | 已安装值 |
|---|---|
| GPU | 8 × NVIDIA B300 SXM6 AC，compute capability 10.3 |
| NVIDIA driver | 590.48.01 |
| 操作系统 | Ubuntu 24.04.4 LTS |
| RoboTwin | `96c1feab536306b50c26af200044fcdf126e8904`，`main` |
| XPolicyLab | `c37109c500be67d0dea6b36bf7337bbd26e763cd`，RoboTwin 固定的子模块版本 |
| CuRobo | `d64c4b005459db10c5dd867d8b30a87d5bda9bdb`，v0.7.8 |
| Python | 3.10.21 |
| PyTorch | 2.9.1+cu130 |
| CUDA 编译器 | 13.0.48，独立安装在 `/data` |
| SAPIEN / MPLib | 3.0.0b1 / 0.2.1 |
| NumPy | 1.26.4 |

安装目录如下：

```text
/data/kaize/rlinf/robotwin2/
├── repos/RoboTwin/                 # RoboTwin、XPolicyLab、CuRobo、assets
├── venvs/robotwin2/                # Python 3.10 环境
├── cuda-13.0/                      # 编译 CuRobo 的 CUDA toolkit
├── python/                         # uv 管理的 CPython
├── cache/{uv,pip,huggingface,...}/
├── lerobot/                        # 后续 LeRobot 数据根目录
├── logs/
├── outputs/
├── env.sh                          # 日常激活入口
├── smoke_test.py                   # adjust_bottle 场景测试
├── install-metadata.txt
└── simulation-requirements.lock.txt
```

## 2. 每次使用时先激活

```bash
source /data/kaize/rlinf/robotwin2/env.sh
cd "$ROBOTWIN_ROOT"
python --version
python -c 'import torch; print(torch.__version__, torch.cuda.get_device_name(0))'
```

`env.sh` 会设置 RoboTwin 的 `PYTHONPATH`、Python 虚拟环境、CUDA 13.0 编译器、NVRTC 动态库路径、NVIDIA Vulkan ICD，以及 uv、pip、Hugging Face、Torch、LeRobot、Weights & Biases 和临时文件目录。它没有重写 `HOME`。

如需指定一张 GPU，在启动命令前设置：

```bash
CUDA_VISIBLE_DEVICES=0 python /data/kaize/rlinf/robotwin2/smoke_test.py
```

首次运行包含左右臂 CuRobo warmup，约需一到两分钟。通过时应出现：

```text
TASK_SETUP_OK
rgb left_camera (240, 320, 3) uint8
rgb right_camera (240, 320, 3) uint8
rgb head_camera (240, 320, 3) uint8
rgb front_camera (240, 320, 3) uint8
joint_action_vector (14,) float64
success_initial False
```

## 3. 为什么 B300 不能照抄上游依赖

RoboTwin 官方当前要求 Python 3.10、推荐 CUDA 12.1，并在 `scripts/requirements.txt` 固定 `torch==2.4.1`。[官方安装说明](https://robotwin-platform.github.io/doc/usage/robotwin-install.html)

B300 是 `sm_103`。本机实测分为三步：

1. PyTorch 2.9.1+cu128 能识别 B300，也能执行普通 CUDA tensor kernel。
2. CuRobo 导入时会用 NVRTC 即时编译融合 kernel；CUDA 12.8 的 NVRTC 对 `compute_103` 报 `invalid value for --gpu-architecture`。
3. 保持 PyTorch 2.9.1 API，切换到官方 cu130 wheel，并用 CUDA 13.0 重编 CuRobo 后，NVRTC、5 个 CuRobo 扩展和 RoboTwin planner 均成功加载。

因此当前环境以官方 requirements 为基础，只对 Torch/CUDA 组合做 B300 兼容覆盖；CuRobo 会追加或更新少量传递依赖，最终版本以 `simulation-requirements.lock.txt` 为准。PyTorch 官方提供 `2.9.1 + torchvision 0.24.1` 的 cu130 wheel。[PyTorch 历史版本](https://pytorch.org/get-started/previous-versions/)

## 4. 从空目录复现本次安装

以下命令与本机成功路径一致。已有安装无需重复执行。

### 4.1 建立全部数据盘目录

```bash
export ROBOTWIN_BASE=/data/kaize/rlinf/robotwin2
mkdir -p "$ROBOTWIN_BASE"/{repos,venvs,python,cache/uv,cache/pip,cache/huggingface,cache/torch,cache/xdg,tmp,logs,outputs,lerobot}
export UV_CACHE_DIR="$ROBOTWIN_BASE/cache/uv"
export UV_PYTHON_INSTALL_DIR="$ROBOTWIN_BASE/python"
export PIP_CACHE_DIR="$ROBOTWIN_BASE/cache/pip"
export TMPDIR="$ROBOTWIN_BASE/tmp"
```

### 4.2 下载最新 RoboTwin 2.0 及固定子模块

```bash
git clone --recurse-submodules \
  https://github.com/RoboTwin-Platform/RoboTwin.git \
  "$ROBOTWIN_BASE/repos/RoboTwin"
cd "$ROBOTWIN_BASE/repos/RoboTwin"
git rev-parse HEAD
git submodule status --recursive
```

实验开始前保存输出。若未来 `main` 已前进，应记录新 SHA，不要无记录地把本教程的 SHA 与新代码混用。

### 4.3 创建 Python 3.10 环境

```bash
uv python install 3.10
uv venv --python 3.10 "$ROBOTWIN_BASE/venvs/robotwin2"
export VIRTUAL_ENV="$ROBOTWIN_BASE/venvs/robotwin2"
export PATH="$VIRTUAL_ENV/bin:$PATH"
uv pip install --python "$VIRTUAL_ENV/bin/python" pip setuptools==69.5.1
```

SAPIEN 3.0.0b1 仍导入 `pkg_resources`，所以需要保留上游安装脚本最后指定的 `setuptools==69.5.1`。

### 4.4 安装 B300 版本的 Torch 和其余依赖

```bash
uv pip install --python "$VIRTUAL_ENV/bin/python" \
  torch==2.9.1 torchvision==0.24.1 \
  --index-url https://download.pytorch.org/whl/cu130

sed -E '/^torch==/d; /^torchvision([=<>!~ ].*)?$/d' \
  scripts/requirements.txt \
  > "$ROBOTWIN_BASE/requirements-robotwin2-b300.txt"
uv pip install --python "$VIRTUAL_ENV/bin/python" \
  -r "$ROBOTWIN_BASE/requirements-robotwin2-b300.txt"
uv pip install --python "$VIRTUAL_ENV/bin/python" \
  -e "$ROBOTWIN_BASE/repos/RoboTwin/XPolicyLab"
```

第二条安装会按 RoboTwin 要求把 NumPy 固定为 1.26.4。若之后再次强制重装 Torch，需再执行：

```bash
uv pip install --python "$VIRTUAL_ENV/bin/python" numpy==1.26.4 pillow==11.3.0
```

### 4.5 应用官方 SAPIEN/MPLib 修补

上游 `_install.sh` 会修改已安装包：SAPIEN 读取 URDF/SRDF 时显式使用 UTF-8，并修正默认 SRDF 后缀；MPLib 的 screw-plan 早退条件移除 `collide`。本机执行的可重复命令如下：

```bash
sapien_dir=$(python -c 'import pathlib, sapien; print(pathlib.Path(sapien.__file__).parent)')
mplib_dir=$(python -c 'import pathlib, mplib; print(pathlib.Path(mplib.__file__).parent)')
urdf_loader="$sapien_dir/wrapper/urdf_loader.py"
planner="$mplib_dir/planner.py"
sed -i 's/with open(urdf_file, "r") as f:/with open(urdf_file, "r", encoding="utf-8") as f:/' "$urdf_loader"
sed -i 's/with open(srdf_file, "r") as f:/with open(srdf_file, "r", encoding="utf-8") as f:/' "$urdf_loader"
sed -i 's/srdf_file = urdf_file\[:-4\] + "srdf"/srdf_file = urdf_file[:-4] + ".srdf"/' "$urdf_loader"
sed -i 's/or collide or not within_joint_limit/or not within_joint_limit/' "$planner"
```

### 4.6 在 `/data` 安装 CUDA 13.0 编译器

本机没有系统 `nvcc`，因此使用 micromamba 只安装 CuRobo 构建所需组件：

```bash
mkdir -p "$ROBOTWIN_BASE/tools" "$ROBOTWIN_BASE/mamba"
curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest \
  | tar -xj -C "$ROBOTWIN_BASE/tools" bin/micromamba
export MAMBA_ROOT_PREFIX="$ROBOTWIN_BASE/mamba"
"$ROBOTWIN_BASE/tools/bin/micromamba" create -y \
  -p "$ROBOTWIN_BASE/cuda-13.0" \
  -c nvidia/label/cuda-13.0.0 -c conda-forge \
  cuda-nvcc cuda-cudart-dev cuda-cccl
"$ROBOTWIN_BASE/cuda-13.0/bin/nvcc" --list-gpu-arch | rg '^compute_103$'
```

### 4.7 构建 CuRobo v0.7.8

```bash
git clone --branch v0.7.8 --depth 1 \
  https://github.com/NVlabs/curobo.git envs/curobo
uv pip install --python "$VIRTUAL_ENV/bin/python" \
  warp-lang==1.12.0 ninja pybind11 setuptools-scm wheel

export CUDA_HOME="$ROBOTWIN_BASE/cuda-13.0"
export PATH="$CUDA_HOME/bin:$VIRTUAL_ENV/bin:$PATH"
export CPATH="$CUDA_HOME/targets/x86_64-linux/include${CPATH:+:$CPATH}"
export CPLUS_INCLUDE_PATH="$CUDA_HOME/targets/x86_64-linux/include${CPLUS_INCLUDE_PATH:+:$CPLUS_INCLUDE_PATH}"
export LIBRARY_PATH="$CUDA_HOME/targets/x86_64-linux/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
export TORCH_CUDA_ARCH_LIST=10.0
export MAX_JOBS=16
python -m pip install -e envs/curobo --no-build-isolation
```

扩展构建为 `sm_100`，可在 B300 上运行；运行时 NVRTC 来自 cu130 wheel，并原生识别 `sm_103`。还需把 PyTorch wheel 的 NVRTC builtins 加入动态库路径：

```bash
export LD_LIBRARY_PATH="$VIRTUAL_ENV/lib/python3.10/site-packages/nvidia/cu13/lib:$CUDA_HOME/targets/x86_64-linux/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

### 4.8 下载资产

官方三个包合计约 14.9 GB：纹理 10.97 GB、对象 3.74 GB、embodiments 0.22 GB。默认 Hugging Face 下载器在本机只有约 1 MB/s，使用官方加速后端：

```bash
uv pip install --python "$VIRTUAL_ENV/bin/python" hf-transfer==0.1.9
export HF_HOME="$ROBOTWIN_BASE/cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HUGGINGFACE_HUB_CACHE="$HF_HUB_CACHE"
export HF_HUB_ENABLE_HF_TRANSFER=1
bash scripts/_download_assets.sh
```

脚本会解压 `background_texture`、`objects`、`embodiments`，删除压缩包，并把 6 份 CuRobo 配置中的 `${ASSETS_PATH}` 替换为当前 RoboTwin 根目录。

### 4.9 配置 ffmpeg 与检查依赖

系统没有 ffmpeg，本次复用 `imageio-ffmpeg` 已下载的静态二进制：

```bash
ln -sfn "$(python -c 'import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())')" \
  "$VIRTUAL_ENV/bin/ffmpeg"
python -m pip check
ffmpeg -version | sed -n '1p'
python -m pip freeze > "$ROBOTWIN_BASE/simulation-requirements.lock.txt"
```

本机 `pip check` 的结果是 `No broken requirements found.`。

## 5. 已完成的验收

### 5.1 PyTorch/B300

- `torch.cuda.is_available()` 为 true。
- 设备名为 `NVIDIA B300 SXM6 AC`，capability 为 `(10, 3)`。
- CUDA tensor kernel 成功执行。
- PyTorch 为 2.9.1+cu130，运行时 CUDA 为 13.0。

### 5.2 CuRobo

`lbfgs_step_cu`、`kinematics_fused_cu`、`line_search_cu`、`tensor_step_cu`、`geom_cu` 和两个 RoboTwin planner 均已成功导入。

### 5.3 SAPIEN 和完整任务

官方 `python scripts/test_render.py` 输出 `Render Well`。`smoke_test.py` 随后完成：载入 `demo_clean` 和 ALOHA AgileX、初始化 `adjust_bottle`、完成 CuRobo warmup、生成四路 RGB、读取 14 维 joint state，并关闭场景。

日志在：

```text
/data/kaize/rlinf/robotwin2/logs/test-render.log
/data/kaize/rlinf/robotwin2/logs/adjust-bottle-smoke.log
/data/kaize/rlinf/robotwin2/logs/download-assets.log
```

## 6. 当前限制

- SAPIEN 初始化时会打印 `OIDN Error: unsupported device type: CUDA`，但官方 renderer test 仍返回 `Render Well`，完整任务也成功产生 RGB。当前可以做无窗口 RGB 仿真；若论文实验要求 OIDN 去噪后的 ray-tracing 图像，需要单独修复并验证 denoiser。
- PyTorch3D 尚未安装。RoboTwin 官方说明不使用 3D 数据时该组件失败不影响项目；当前 RGB、joint state 和任务 reset 已验证。启用 point cloud、mesh 或相关 3D pipeline 前需单独编译并测试 PyTorch3D。
- 本次只验证了 `adjust_bottle` 的 reset/observation，没有执行完整专家 `play_once()`、采集 50 个 episode 或下载演示数据。这些会产生额外计算与存储，应作为下一阶段验证。
- RLinf 当前的 `robotwin` adapter 使用旧 `RLinf_support` 分支的 `robotwin.envs.vector_env.VectorEnv`。最新版 main 没有该 API，因此这次完成的是最新模拟器安装与独立运行验收；接入 RLinf RLT 仍需实现 main backend/进程 bridge。

## 7. 下一步接入 RLinf 的最小顺序

1. 以 `/data/.../smoke_test.py` 的参数装载代码为基础，封装 `reset(seed)`。
2. 把 `get_obs()` 的 `head_camera/left_camera/right_camera`、14 维 `joint_action.vector` 映射到 RLinf 观测字典。
3. 用 `take_action(..., action_type="qpos")` 封装单步或动作 chunk，并显式处理 success、step limit、terminated 和 truncated。
4. 先用独立仿真进程调用本环境，RLinf learner 保留自己的依赖环境；IPC 只传 NumPy/张量，不传 SAPIEN 对象。
5. 单环境、`adjust_bottle`、K=1 通过 20 步 transition 对齐检查后，再接 RLT reference/actor route 和 streaming replay。

更完整的环境接口与 Physical Token 研究路线见 [RLT、RoboTwin 2.0、RoboDojo 与 Physical Token 指南](rlt_robotwin_robodojo_physical_token_guide_zh.md)。
