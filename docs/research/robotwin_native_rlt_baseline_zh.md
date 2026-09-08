# 原生 RoboTwin RLT baseline 复现记录

更新日期：2026-09-08

本文记录 RLinf 原生 `robotwin` 环境上的 Pi0.5 policy baseline、RLT Stage 1 和 RLT Stage 2 最小闭环。它用于冻结第一版可复现基线，并作为 Physical Token 与 Streaming RL 分支的共同起点。最新 RoboTwin 2.0 bridge 保存在独立 worktree，不进入本基线。

## 固定版本与目录

| 项目 | 固定值 |
|---|---|
| RLinf 基线 commit | `68feab9f0a50764d75e42f12f837dbbb74775d15` |
| RoboTwin 分支/commit | `RLinf_support` / `0008ae6800df9f75fc8de7098bacb01735fd8fd2` |
| GPU | NVIDIA B300 SXM6 AC，compute capability 10.3 |
| PyTorch / CUDA | `2.11.0+cu130` / CUDA 13.0 |
| Ray | 2.58.0 |
| SAPIEN | 3.0.1 |
| Python 环境 | `/data/kaize/rlinf/venvs/openpi-robotwin` |
| RoboTwin | `/data/kaize/rlinf/robotwin-native/repos/RoboTwin` |
| baseline worktree | `/data/kaize/rlinf/worktrees/robotwin-rlt` |

RoboTwin assets 复用 `/data/kaize/rlinf/robotwin2/repos/RoboTwin/assets`，原生 checkout 中的 `assets` 是指向该目录的符号链接。这样不会在根分区或 `/data` 重复保存约 16 GB 资源。

## 固定模型与数据

- 官方 Pi0.5 checkpoint：`RLinf/RLinf-Pi05-RoboTwin-SFT-adjust_bottle`
  - Hugging Face revision：`fa8df6ed103db0f5549c122f3a17c00ba6426c98`
  - 本地路径：`/data/kaize/rlinf/models/RLinf-Pi05-RoboTwin-SFT-adjust_bottle`
  - 为 `openpi_rlinf` 兼容性额外生成了无损合并文件 `model.safetensors`，原始 shards 保留。
- 官方 50 条示范数据：`RLinf/RoboTwin-adjust_bottle-official-demo_clean50-Pi0_processed-data`
  - Hugging Face revision：`c9944782f5473feef33d98af6cc033e23acfd875`
  - 本地路径：`/data/kaize/rlinf/datasets/robotwin-adjust-bottle-clean50`
  - 统计：50 episodes、7188 frames、50 fps、3 个相机、14 维 state/action。
  - normalization stats：`adjust_bottle/norm_stats.json`。

## 配置入口

| 阶段 | 配置 |
|---|---|
| 官方 policy smoke baseline | `evaluations/robotwin/robotwin_adjust_bottle_openpi_pi05_smoke.yaml` |
| RLT Stage 1 SFT | `examples/sft/config/robotwin_rlt_stage1_sft_openpi_pi05.yaml` |
| RLT Stage 2 AC | `examples/embodiment/config/robotwin_rlt_stage2_ac_mlp.yaml` |

所有大模型、数据、日志和 checkpoint 都放在 `/data/kaize/rlinf`。配置中的本地路径均可通过环境变量替换。

## 已验证结果

### 官方 Pi0.5 policy baseline

单环境 `adjust_bottle` 完整 150 步评测成功：

- `episode_len=150`
- `return=1.0`
- `success_once=1.0`
- rollout 约 75.82 秒；整步约 83.40 秒
- 日志目录：`/data/kaize/rlinf/worktrees/robotwin-rlt/logs/20260908-04:44:14-robotwin_adjust_bottle_openpi_pi05_smoke`

这证明 RLinf 原生 RoboTwin、官方 checkpoint、三相机输入、14 维动作和 B300 推理链路可用。

### RLT Stage 1

在官方 50 条示范数据上完成一个真实优化步，并保存可被 Stage 2 加载的 wrapper checkpoint：

- 参数量：3.35B
- `train/loss=3.76`
- `train/rlt_loss=3.64`
- `train/vla_loss=0.123`
- `train/grad_norm=5.45`
- checkpoint：`/data/kaize/rlinf/runs/robotwin-rlt-stage1-smoke-v2/robotwin_rlt_stage1_sft_openpi_pi05/checkpoints/global_step_1/actor`

该 checkpoint 同时含 VLA 和 RLT token 模块。单步 smoke checkpoint 包含 FSDP optimizer 状态，磁盘占用较大；正式实验应设置合理保存间隔并制定保留策略。

### RLT Stage 2：K=1 transition 与切换闭环

固定条件：

- 单环境、seed 0
- `num_action_chunks=1`
- 每个 rollout 20 个 simulator steps
- reference horizon 50，只执行首个动作
- replay warmup 4 transitions
- warmup 后执行 4 次 critic update 和 4 次 actor update
- actor 输出为 reference action 上的有界 residual

首次端到端检查发现 reset 的 `rlt_switch_flags` 只保存在 `env_infos`，没有传入 rollout，因此首条 transition 被跳过，20 步只写入 19 条 replay。修复 `EnvWorker.bootstrap_step()` 后结果为：

| 指标 | 第 1 回合 | 第 2 回合 |
|---|---:|---:|
| simulator steps | 20 | 20 |
| 本回合 replay transitions | 20 | 20 |
| replay cache size | 20 | 40 |
| `record_transition_rate` | 1.0 | 1.0 |
| `actor_switch_rate` | 0.0 | 1.0 |
| `ready_for_online` | 0.0 | 1.0 |
| actor / critic updates | 4 / 4 | 4 / 4 |
| success | 0 | 0 |

第 1 回合由 reference policy 采样并完成 warmup；权重同步后，第 2 回合由 actor 接管。20 步远短于 `adjust_bottle` 完整任务的 150 步评测长度，因此这里的 success=0 只说明 smoke horizon 内未完成任务，不能用于算法成功率比较。

已保存的 Stage 2 actor checkpoint：`/data/kaize/rlinf/runs/robotwin-rlt-stage2-baseline-v1/robotwin_rlt_stage2_ac_mlp/checkpoints/global_step_2/actor`。对应原始指标日志为 `/data/kaize/rlinf/runs/robotwin-rlt-stage2-baseline-v1/metrics.log`。

## 时序约束

当前原生 RoboTwin RLT 明确限制 `K=1`。环境会在 RLT 打开时拒绝 `num_action_chunks>1`，原因是 RoboTwin 的 chunk API 会一次返回 chunk 内多组 reward/done，而 reference/actor switch 和 RLT feature 是按 policy query 生成的。直接提高 K 会把以下语义混在一起：

1. chunk 中第几个动作触发 success 或 truncation；
2. terminal observation 应对应哪个动作；
3. replay 中 `s_t, a_t, r_t, s_{t+1}` 的一一对应；
4. switch 在 chunk 内生效时，reference 与 actor 动作的归属；
5. residual action 使用哪个 reference action。

在实现逐子步 observation/feature、done mask 和 action ownership 之前，不应以 K>1 数据做训练结论。

## 吞吐与成功率风险

K=1 会增加 VLA/feature model 查询次数。官方 policy baseline 使用 50 步 action chunk 时，150 个环境步只需约 3 次策略查询；Stage 2 K=1 则每个环境步都查询一次 3.35B feature model。当前受其他作业竞争的 B300 上，reference warmup 的 20 步 rollout 约 60 秒，actor 接管后的 20 步约 31 秒。这些数字只能说明功能链路，不能作为独占 GPU 吞吐 benchmark。

bridge 或新环境适配本身的 Python/RPC 开销通常不是主要瓶颈。更大的吞吐风险来自 K=1 的推理频率、SAPIEN 渲染、CPU/GPU 同步、Ray tensor copy、三路图像预处理和 actor/rollout/env 共卡竞争。后续优化顺序应为：

1. 先用 K=1 固定 transition 正确性和成功率基准；
2. 给 policy inference、env step、图像搬运、Ray channel 和训练更新分别计时；
3. 再实现 K>1 的逐子步 replay 语义；
4. 最后增加并行环境和异构 placement。

成功率掉点的主要风险不是 bridge 函数调用，而是 observation/action 约定变化，包括相机顺序、RGB 范围、state 顺序、absolute/delta action、gripper 表示、control frequency、终止条件和 normalization stats。任何吞吐优化都必须同时复测固定 seeds 的成功率。

## 已知非阻断告警

- RoboTwin planner 会尝试导入 CuRobo，并打印 `curobo.types.math` 缺失 traceback；当前配置使用 `planner_backend=mplib`，实际评测和训练均成功。
- SAPIEN/svulkan2 在 B300 上打印 OIDN `unsupported device type: CUDA` / `invalid handle`；渲染和任务执行仍成功。
- `/data` 当前使用率约 95%，仍有约 1.3 TiB 可用。Ray 会持续发出磁盘阈值告警；正式长训前应清理无用 smoke checkpoint，并把 Ray object spilling 指向容量更宽裕的目录。

## 分支边界

- `baseline/robotwin-rlt-native-v1`：本文件记录的原生 RoboTwin + RLT 可复现基线。
- `infra/robotwin2-main-bridge`：最新 RoboTwin 2.0 main bridge，暂存且不进入 baseline。
- `exp/physical-token-v1`：从 baseline commit 创建，开发 Physical Token。
- `exp/streaming-rlt-v1`：从 baseline commit 创建，开发 Streaming RL。
- `exp/physical-token-streaming`：只在两个单独方向完成消融后用于组合实验。

创建实验分支前必须先提交 baseline；否则分支只会指向旧 commit，未提交改动仍停留在一个 worktree 中，无法形成清晰实验谱系。
