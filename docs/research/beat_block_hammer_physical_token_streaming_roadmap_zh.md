# beat_block_hammer：Physical Token + Streaming RLT 执行路线

## 0. 研究范围与冻结项

首轮只使用 RLinf 原生支持的 RoboTwin：

- RoboTwin 分支：`RLinf_support`
- RoboTwin commit：`0008ae6800df9f75fc8de7098bacb01735fd8fd2`
- 任务：`beat_block_hammer`
- embodiment：`aloha-agilex`
- VLA：π0.5
- 在线动作接口：14 维 ALOHA joint-position target；正式 RLT 使用
  `C=10`，π0.5 reference horizon 为 `H=50`。`C=1` 仅用于 transition
  对齐调试和单独的 chunk-length 消融。

RoboTwin2 bridge 暂不进入首轮训练。只有原生任务完成 B0/B1 多 seed 后，才把已经隔离的 bridge 用于论文环境迁移。

需要固定并在所有主要比较中保持一致：

- 同一个任务特定 π0.5 SFT checkpoint；
- 同一个 reference policy；
- 相同的视觉、语言和 proprio 在线输入；
- 相同 token 输出维度；
- 相同环境交互步数、seed、奖励、episode horizon 与更新预算；
- privileged physics 量只作为训练 target 和评测诊断量，不能作为部署时新增 observation。

核心可检验主张：

> 在冻结同一 VLA/reference policy、固定表示维度和交互预算下，动作条件的几何、真实执行状态与接触动力学监督，使 RLT state bottleneck 更具 action consequence predictiveness，从而改善 critic 学习和动力学变化下的在线适应；进一步检验这种表示能否减小低回放或严格 streaming 带来的性能退化。

## 1. 已确认的当前实现事实

### 1.1 任务与成功条件

原生任务会抓取锤子，再把锤头移动至方块。官方 success 是：

- 锤子 functional point 与方块 functional point 的 XY 误差均小于 2 cm；
- 锤子与方块发生 contact。

因此官方 success 必须保留，但它不足以单独证明“有力、快速且稳定的敲击”。必须额外记录：首次接触时间、接触 impulse、相对接触速度、抓稳率、掉落率和最终几何误差。

### 1.2 当前 RLT 输入与监督

- actor 输入：reference action、`z_rl`、14 维 proprio；
- critic 输入：action、`z_rl`、14 维 proprio；
- token Stage 1 当前只用 MSE 重建 detached VLA prefix hidden states；
- 当前 Stage 1 配置同时优化 `L_RLT + L_VLA`，且 `train_expert_only: false`，不满足“冻结同一 VLA/reference”的假设；
- 原生 `observation.state` 是 joint drive target，不是真实执行 qpos；
- 真实 qpos、TCP pose、锤子 pose、接触点与 impulse 在模拟器中可得，但当前 wrapper 没有传给 RLinf。

由此得到两个实验约束：

1. Stage 1 必须新增显式 `freeze_vla_for_rlt` 路径，只训练 token/adapter/auxiliary heads。
2. “+proprio”不能作为 Physical Token 的有效消融，因为 B0 actor/critic 已经看到 proprio。

## 2. 分支与工作区

```text
baseline/robotwin-rlt-native-v1  @ 0dd01ac9  冻结，不再改
├── task/robotwin-native-beat-hammer-pi05          当前任务 bring-up
│   ├── exp/physical-token-v1                      B1 与 token 消融
│   ├── exp/streaming-rlt-v1                       B2
│   └── exp/physical-token-streaming-v1            B3，最后组合
├── infra/robotwin2-main-bridge      @ f3df3622     暂不进入首轮
└── upgrade/rlinf-upstream-1c9eed0                  仅做独立上游迁移
```

当前 hammer worktree：

```bash
cd /data/kaize/rlinf/worktrees/beat-block-hammer
git status --short --branch
```

不要从 `main` 或 RoboTwin2 bridge 开始 hammer 实验。Physical/streaming 分支要等任务特定 baseline 的配置、测试和 smoke commit 完成后再建立。

## 3. 总执行顺序

```text
G0 Git 冻结
  -> H0 原生 hammer 环境/专家数据 smoke
  -> H1 任务特定 π0.5 SFT + reference 画像
  -> H2 原始 RLT B0
  -> P0 privileged physics 标签管线
  -> P1 Physical Token 离线 probe 与损失消融
  -> P2 Physical Token + replay B1
  -> D0 Appearance-OOD / Dynamics-OOD
  -> S1 低回放 B2/B3
  -> S2 严格 streaming B2/B3
  -> A1 不确定性 residual / 短历史
  -> A2 多频控制、physics critic、WAM
  -> R2 RoboTwin2 迁移
```

任何阶段未通过验收门槛，不进入下一阶段。

## 4. H0：原生 hammer bring-up

### 要做的代码与配置

在 task 分支创建任务专属文件，禁止直接覆盖 adjust_bottle 配置：

- `examples/embodiment/config/robotwin_beat_block_hammer_openpi_pi05_eval.yaml`
- `examples/sft/config/robotwin_beat_block_hammer_sft_openpi_pi05.yaml`
- `examples/sft/config/robotwin_beat_block_hammer_rlt_stage1_openpi_pi05.yaml`
- `examples/embodiment/config/robotwin_beat_block_hammer_rlt_stage2_ac_mlp.yaml`

统一设置：

- task：`beat_block_hammer`；
- prompt：`grab the hammer and hit the block`；
- embodiment：`[aloha-agilex]`；
- 三路相机：head + left/right wrist；
- `config_name: pi05_aloha_robotwin`；
- action/state 14 维；
- RLT actor/env 的 `num_action_chunks: 10`，reference/feature model 保持 50；
- clean reference 评测先关闭外观随机化；
- 数据、checkpoint、norm stats 使用 hammer 专属目录与 key。

必须增加配置一致性测试，至少检查 task、prompt、embodiment、三相机、14 维、正式 `C=10`、reference `H=50` 和 norm key，防止无声复用 adjust_bottle。

### 专家数据 smoke

原生 collector 的入口是：

```bash
cd /data/kaize/rlinf/robotwin-native/repos/RoboTwin
bash collect_data.sh beat_block_hammer demo_clean 0
```

正式执行前先复制出项目专属 smoke 配置，把 `episode_num` 改为 3，不要修改共享 `demo_clean.yml`。确认 3 个成功轨迹的：

- head/wrist 图像齐全；
- state/action 都是 14 维且无 NaN；
- endpose 存在；
- action、observation、terminal 时间对齐；
- 视频中使用正确手臂、抓锤、锤头接触方块。

### H0 验收

- 3 个成功专家 episode 可重复生成；
- evaluator 可完成 1 个随机策略/未训练 π0.5 smoke，不崩溃；
- 任务专属配置测试通过；
- 容量路径全部落在 `/data/kaize/rlinf`。

## 5. H1：任务特定 π0.5 reference

当前服务器只有 adjust_bottle 的 π0.5 checkpoint 和数据，不能把它直接当 hammer reference。官方也只公开了 adjust_bottle 的 π0/π0.5 权重；hammer 的公开结果是 OpenVLA-OFT，不是 π0.5。

### 数据与 SFT

1. 用固定 native commit 收集 50 条成功 `demo_clean` 轨迹。
2. 在收集原始 HDF5 时同步保存 physics sidecar，避免以后重跑模拟器。
3. 实现或验证 HDF5 -> LeRobot v2.1 转换，输出 hammer 专属：
   - 三路 RGB；
   - `observation.state` 14 维；
   - `action` 14 维；
   - task/prompt；
   - physics sidecar 或 Parquet 扩展字段。
4. 计算 hammer 专属 action/state norm stats。
5. 从本地 `pi05_base` 做任务 SFT：

```bash
cd /data/kaize/rlinf/worktrees/beat-block-hammer
bash examples/sft/run_vla_sft.sh robotwin_beat_block_hammer_sft_openpi_pi05
```

该命令只有在专属配置和转换数据完成后才能运行。

### reference 画像，而不是只看一个 success

先用 30 个固定 eval seeds，统计：

- 官方 success；
- 到达抓取/抬起/对准/接触各阶段的比例；
- time-to-first-grasp、time-to-first-hammer-block-contact；
- 成功 episode steps；
- final hammer-head/block XY error；
- 抓稳后掉落率；
- 接触 impulse 与相对速度分布。

### H1 决策门槛

理想 reference 区间：

- 至少约 50% rollout 能到达关键接触前阶段；
- 整体 success 约 30%–80%；
- 最后对准/接触阶段仍有约 20%–50% 失败。

如果 success < 20%，先修数据/SFT，不做 RL；如果 > 85%，先保留 clean 官方结果，再引入单因素 Dynamics-OOD，而不是人为改 success 定义。

## 6. H2：冻结 VLA 的原始 RLT baseline（B0）

### 先修 Stage 1 公平性

实现 `freeze_vla_for_rlt: true`：

- 所有 VLA、vision/language backbone、action expert/projection 参数 `requires_grad=False`；
- 只训练 `rlt_module`；
- `L_VLA` 仅用于可选日志，不进入优化目标；
- 启动时打印并断言 trainable parameter names/count；
- 保存前后 hash 或逐参数比较，证明 reference 没有变化。

原始 token 目标：

```text
L_A0 = MSE(reconstruct(z_rlt), stopgrad(VLA_prefix))
```

### B0 Stage 2

- reference、VLA、RLT encoder 全部冻结；
- 只更新小型 residual actor + twin critic；
- action 是 `a_ref + scale * tanh(delta)`；
- 正式 actor chunk `C=10`，reference horizon `H=50`；`C=1` 仅保留为调试/消融；
- 训练 reward 对所有 B0–B3 完全相同；
- 成功立即终止，并报告 time-to-success；
- smoke 从 20 步扩大到完整 200 步只是管线验证，不是论文结果。

正式 B0 至少先做 3 train seeds；最终主表建议 5 train seeds，并在同一固定 eval seed set 上评估。

### H2 验收

- Stage 1 后 VLA/reference 参数逐位未变化；
- 200-step smoke transition 记录率为 1.0；
- actor/critic loss、Q 值和 residual norm 无 NaN/爆炸；
- reference-only 与 B0 都完成 30-seed 画像；
- 保存 B0 config、commit、checkpoint、seed 文件和 metrics JSON。

## 7. P0：hammer privileged physics 标签

### 只做 target，不给在线策略偷看

在 native RoboTwin task/SubEnv 中形成 `physics` dict，经 VectorEnv 和 RLinf wrapper 写入 transition/info。第一版字段：

```text
q_cmd[14]                 当前 drive target，即原 observation.state
q_real[14]                实际关节位置 + gripper
qvel_real[14]             articulation qvel；不可用维度用有效 mask
tcp_left/right_pose[7]
hammer_pose[7]
hammer_head_pose[7]
block_target_pose[7]
gripper_hammer_contact[2]
hammer_block_contact[1]
hammer_block_impulse[3]   接触点 impulse 求和
is_in_hand[2]
```

从相邻时刻派生：

```text
delta_q_real
delta_tcp_pose
delta_hammer_pose
hammer_head_to_block
delta_hammer_head_to_block
first_grasp / first_contact / slip / drop / success phase
```

必须记录 `valid_mask`。在 reset/terminal 边界禁止跨 episode 求差分。

### 对齐单元测试

构造短轨迹并验证：

- 第 t 条 transition 是 `(o_t, a_t, r_t, o_{t+1})`；
- physics target 来自 `t -> t+1`；
- auto-reset 时使用 final observation，而非下一 episode reset observation；
- contact label 对应 `t+1`，不是提前一帧；
- 所有 Δpose 对 quaternion 符号和旋转表示处理一致。

## 8. P1：Physical Token V0（先离线证明表示，再做 RL）

### 正确定义 action-conditioned

actor 在选动作前需要 token，因此不能把待选择的 `a_t` 输入 token encoder。动作条件应放在预测 head：

```text
z_phys,t = Adapter(z_rlt,t, proprio_t [, short_history])

F_q(z_phys,t, a_t)       -> delta_q_real
F_tcp(z_phys,t, a_t)     -> delta_tcp_pose
F_obj(z_phys,t, a_t)     -> delta_hammer_pose, delta_geometry
C(z_phys,t, a_t)         -> contact_{t+1}, grasp/slip/drop
```

第一版 loss：

```text
L_PT = L_RLT
     + lambda_q   * Huber(pred_delta_q, delta_q_real)
     + lambda_tcp * Huber(pred_delta_tcp, delta_tcp_pose)
     + lambda_obj * Huber(pred_delta_hammer, delta_hammer_pose)
     + lambda_geo * Huber(pred_delta_geometry, delta_geometry)
     + lambda_c   * weighted_BCE(pred_contact, contact_{t+1})
```

规则：

- VLA 完全冻结，`L_VLA` 不进入优化；
- adapter 输出仍为 2048 维；
- auxiliary heads 只在 Stage 1 使用；
- B1 第一版在 Stage 2 冻结 Physical encoder，避免“表示更好”和“在线共同训练更多参数”混为一谈；
- 各 target 用 train split 统计量归一化；
- contact 稀疏时使用 class weight/分阶段均衡采样，但不得改变 RL 交互预算。

### 第一轮最小 token 消融

| ID | 表示 | 目的 |
|---|---|---|
| A0 | 原 RLT token | baseline |
| A1 | 同构 adapter + 相同输入，仅重建/蒸馏，无 physics loss | 控制参数量和架构 |
| A2 | A1 + real joint/TCP dynamics | 低维动作后果 |
| A3 | A2 + hammer/relative geometry dynamics | task-relevant geometry |
| A4 | A3 + contact/grasp/slip/drop | 完整 V0 |

不要首轮加入解析动力学 residual、WAM、uncertainty、短历史和 constraint action penalty。

### 离线 representation probe

在完全冻结表示上训练同等容量 linear/2-layer probes，报告 held-out：

- delta state/pose Huber error；
- contact AUROC、AUPRC、F1；
- grasp/slip/drop F1；
- future hammer-head/block distance error；
- appearance nuisance probe 与 dynamics parameter probe。

只有 A2–A4 在 held-out consequence prediction 上稳定优于 A0/A1，才进入在线 B1。

## 9. P2：B1 Physical Token + replay

主比较：

| 组 | state bottleneck | learner |
|---|---|---|
| R0 | 无 actor，reference only | 无 RL |
| R1 | proprio + reference，无 token | ordinary residual replay |
| B0 | 原 RLT token | replay actor-critic |
| C1 | 同构无物理监督 adapter | replay actor-critic |
| B1 | Physical Token A4 | 同一 replay actor-critic |

固定 actor/critic architecture、初始化方案、更新次数、batch、buffer、环境步数与 seeds。

除控制效果外，必须测 critic：

- held-out one-step TD error；
- 与 Monte-Carlo return 的相关性和 calibration；
- Q ensemble disagreement；
- critic loss 达到固定阈值所需环境步数。

这样才能支撑“更适合 critic”，而不只是最终 success 偶然更高。

## 10. D0：外观 OOD 与动力学 OOD 分开

先单因素，再组合：

### Appearance-OOD

- 背景；
- 光照；
- 桌面纹理/轻度 clutter；
- 相机位姿小扰动。

### Dynamics-OOD

- hammer mass；
- hammer/table/contact friction；
- action execution delay；
- actuator/controller gain 或 execution noise。

训练范围与 held-out 测试范围必须不重叠，并把实际采样参数写入 episode metadata。官方 clean success 仍单独保留。

理想表征现象：对 appearance 变化较不敏感，同时能编码/区分会改变 action consequence 的 dynamics 变化。

## 11. S1/S2：低回放与严格 streaming

只有 B0/B1 replay 已稳定后再开始。

### S0：正式 replay reference

当前 cache=64 是 smoke 参数，不是论文 replay baseline。先设置可容纳多 episode 的正式 buffer，并固定 UTD。

### S1：低回放

在 B0、B1 上分别跑：

```text
window = 5000 / 500 / 50
UTD = 1
```

如果总交互预算较小，应同时报告 window 占总 transitions 的比例。

### S2：严格 streaming

定义：最新 transition 到达后只更新一次，随后丢弃，不从 replay 重采样。需要：

- VLA 和 token/Physical encoder 冻结；
- 只更新小 actor/critic；
- running state/reward normalization，只使用过去统计量；
- 明确是否保留 target network、target policy smoothing 和 optimizer modification；
- 每条 transition 的 `learn_count == 1` 自动审计；
- 禁止 warm replay 偷渡到 strict streaming 结果中。

第一版沿用确定性 residual actor，逐步吸收 SDAC 的稳定化组件，而不是简单把 batch size 改成 1。

### 最小 2x2 主矩阵

| 组 | Token | Learner |
|---|---|---|
| B0 | 原 RLT | replay |
| B1 | Physical | replay |
| B2 | 原 RLT | 低回放/strict streaming |
| B3 | Physical | 相同低回放/strict streaming |

关键统计不是只比较 B3 与 B2，还要比较退化量：

```text
degradation_original = performance(B0) - performance(B2)
degradation_physical = performance(B1) - performance(B3)
```

只有后者显著更小，才能支持“Physical Token 缓解 streaming 退化”。

## 12. 统一指标与日志

### 控制与吞吐

- official success rate；
- learning curve AUC；
- environment steps to X% success；
- time/steps-to-success；
- successes per 1000 env steps；
- wall-clock samples/hour、updates/hour；
- reference 调用次数和 actor takeover rate。

### 精细操作/物理

- hammer-head/block XY error；
- grasp、lift、align、contact 阶段成功率；
- first-contact step；
- contact impulse 与相对速度；
- slip/drop/robot-block unintended contact；
- executed q velocity/acceleration/jerk，而不是只算 action target 差分。

### 统计

- 开发期 3 train seeds；主结果至少 5 train seeds；
- 同一方法共享固定 eval seed set；
- 报告均值、95% bootstrap CI 与 per-seed scatter；
- 配对比较使用相同 seeds；
- 同时保存 raw episode-level JSON/Parquet，不能只保留 TensorBoard 曲线。

## 13. 后续增强的进入顺序

V0 主张成立后再逐项加入：

1. 4/8 步短历史，并加入“原 token + 相同历史”对照；
2. contact/dynamics uncertainty，用于 residual scale 或 reference switch；
3. physics-aware critic head；
4. constraint margin 与 action anchor；
5. 接触事件驱动可变 K，多频控制并正确使用真实执行长度和 `gamma^K`；
6. WAM/RepWAM 或 online world-model co-training。

PIN-WM 属于显式、可微物理 world model；PhysReflect-VLA 属于执行期可行性检查与反思；ExToken 用 token 做结构化探索；RepWAM/WAM-RL 属于 world-action modeling。当前方法应把差异写成：固定 VLA 的 critic-oriented predictive bottleneck + 低回放/streaming 在线适应，而不是泛称“加入物理 token”。

## 14. 论文停止/继续判据

### 继续 Physical Token 主线

同时满足：

- A4 held-out geometry/contact probe 优于 A0/A1；
- B1 critic calibration/TD 指标优于 B0；
- B1 在相同预算下的 learning AUC 或 sample efficiency 稳定提升；
- Dynamics-OOD 提升大于 Appearance-OOD 的随机波动。

### 转向表示学习而弱化控制 claim

若 probe 明显提升但 B1 控制不提升，检查 critic 是否已直接从 raw proprio/reference 获得足够信息，以及 z 是否被 actor/critic 忽略。此时先做 representation usage/gradient/feature ablation，不立即加更多模块。

### 暂停 strict streaming

若 B0/B1 replay 尚不稳定，或 low-replay window=50 已完全坍塌且无法复现，不进入 strict streaming。先让 streaming learner 在低维 control benchmark 和冻结 hammer feature 上通过单 transition 审计。

## 15. 现在最近的四个执行项

1. 在 task 分支实现四份 hammer 专属 config 和一致性测试。
2. 创建 3-episode `demo_clean` smoke config，运行原生专家 collector。
3. 实现 physics snapshot/sidecar 与 t->t+1 对齐测试，再正式采 50 条成功轨迹。
4. 完成 LeRobot v2.1 转换和 hammer π0.5 SFT，先做 30-seed reference 画像；通过 H1 门槛后再创建 `exp/physical-token-v1`。

## 16. 主要参考

- RL Token: https://arxiv.org/abs/2604.23073
- S2AC/SDAC: https://arxiv.org/abs/2603.08588
- PhysReflect-VLA: https://arxiv.org/abs/2606.27146
- ExToken: https://arxiv.org/abs/2607.12931
- RepWAM: https://arxiv.org/abs/2606.13674
- WAM-RL: https://arxiv.org/abs/2606.17906
- PIN-WM: https://www.roboticsproceedings.org/rss21/p153.pdf
- RLinf RoboTwin guide: https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/robotwin.html
