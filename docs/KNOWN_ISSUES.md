# 已知问题与踩坑汇总

> 记录 DinoBoard 开发过程中发现并修复的 bug，以及需要注意的设计取舍。
> 目的：避免未来开发者重复踩坑。

根据受众不同分为两大类：

- **框架层 Issues**：影响 `engine/` / `engine/runtime/` / `engine/search/` / `training/` / `bindings/` / `platform/` 的问题。修改框架代码前建议通读——这些是架构层面的坑，跨所有游戏生效。
- **游戏层 Issues**：只影响 `games/<name>/` 下具体游戏的 rules / encoder / tracker / config。开发新游戏时**应着重参考本节**——里面的模式（Splendor 不偷看、Love Letter 揭牌事件）是游戏开发者常见的踩坑点。

---

## 目录

### 通用踩坑事项

- [通用踩坑事项](#通用踩坑事项) — 9 条新游戏接入前必读的隐性陷阱

### 设计决策（DEC）

- [DEC-001] 旧 2p 标量价值头永久兼容（显式契约，不是 BUG）
- [DEC-002] Warmstart 并入 heuristic_guidance schedule
- [DEC-003] Session 不再每步 randomize_unseen + IGameRules wrapper 接管 step_count

### 框架层 Issues（搜索 / 训练 / 运行时 / 平台）

- [BUG-001] Tail Solver 转置表标志位反转
- [BUG-002] 平局 z 值赋值错误
- [BUG-003] 训练-评估动作空间不一致
- [BUG-004] Heuristic Runner 缺少 Adjudicator 支持
- [BUG-005] FilteredRulesWrapper 的 const_cast
- [BUG-006] Replay Buffer 样本利用率
- [BUG-007] pipeline.py 用初始局面特征训练所有样本
- [BUG-008] Splendor temperature_schedule 被静默忽略
- [BUG-009] pipeline.py 重写丢失三项训练改进
- [BUG-010] pipeline.py 重写丢失 Replay Buffer
- [BUG-011] ONNX 未编译导致 MCTS 使用均匀策略
- [BUG-012] model_init.onnx 导出顺序错误
- [BUG-013] ONNX 不是每步导出，selfplay 用旧模型
- [BUG-014] best model 路径指向 latest 文件被覆盖
- [BUG-015] pipeline.py 用 z_values 取值但 C++ 不总是填充
- [BUG-016] legal mask 被 filter 缩小导致 free 模式失效
- [BUG-018] Adjudicator z_values 不零和 + 标量 value head 的 3p+ 展开 bug
- [BUG-019] 多人模式全链路 2p 硬编码
- [BUG-020] Pipeline 分析使用错误的 stats key 导致 expert AI 卡死
- [BUG-021] fly 动画继承源容器尺寸，导致 Azul 砖巨大化 + 顺序播放卡顿
- [BUG-022] cancel_pipeline 误清 precompute，导致 expert 第一手 AI 响应多 8 秒
- [BUG-024] GameSession MCTS 搜索在真实状态上跑，应隔离为 AI view
- [BUG-025] pipeline.py `nopeek_enabled` off-by-one：peek_steps=0 被错误解读为"第 0 步 peek"
- [BUG-026] ISMCTS DAG hash collision → MCTS 选中非法 action 崩溃
- [BUG-027] Quoridor 手机端棋盘 UI 连环坑 —— button UA baseline 偏移 + 固定像素尺寸在小 slot 下退化
- [BUG-029] tail-solve 采纳 ProvenWin 路径未填 `root_values`，专家模式终局窗口 45s 后才显示
- [BUG-031] Web 隔离 AI 会话没有继承 `tail_solve` 配置（web AI 实际未启用 tail-solve）
- [BUG-032] Selfplay per-perspective trackers 未被初始化 → randomize_unseen 覆写当前玩家自己的手牌 (OB-011)
- [BUG-034] 体验版（casual）AI 走子后没显示对手胜率——前端 difficulty gate 把已经算好的数字扔了 (OB-013)

### 游戏层 Issues（具体游戏的规则 / 编码器 / tracker）

- [BUG-017] SplendorBeliefTracker 偷看牌堆内容
- [BUG-023] Love Letter AI 永远猜对 Guard — terminal-by-elimination 漏过 NoPeek 检测
- [BUG-028] public hash 混入不可观察随机源，导致 ISMCTS DAG 按隐藏信息分裂
- [BUG-030] Love Letter encoder 把 tracker 知识泄漏到非 perspective 玩家视角
- [BUG-033] Azul 轮末结算飞砖落地后砖消失 + 多行同结算时 +score 偏高 (OB-012 / OB-008)
- [BUG-035] Azul 轮末地板扣分用错时间点的 floor_count——actor 在结算回合扔的砖没算进去
- [BUG-037] LoveLetter `hand` 槽 hash 只 mix value 不 mix idx，dynamic-reveal 视野下两套 (idx,value) 序列哈希撞车 → DAG node legal-action mismatch

---

## 通用踩坑事项

### 1. UndoToken 的 undo_depth 必须在 push 前设置

```cpp
UndoToken token{};
token.undo_depth = static_cast<std::uint32_t>(s->undo_stack.size());  // push 之前！
// ... push UndoRecord ...
// ... 修改状态 ...
return token;
```

如果在 push 之后设置 undo_depth，undo 时会弹出错误的记录。

### 2. state_hash 必须是确定性的

对于相同的游戏状态，`state_hash()` 必须始终返回相同的哈希值。常见错误：
- 忘记哈希某个影响游戏走向的字段（如 current_player）
- 使用了内存地址或指针值
- 没有处理 `include_hidden_rng` 参数

### 3. game.json 的 feature_dim 和 action_space 必须和 encoder 一致

`config/game.json` 中的 `feature_dim` 和 `action_space` 必须精确匹配 `IFeatureEncoder` 的 `feature_dim()` 和 `action_space()` 返回值。不一致会导致：
- 训练时 PyTorch 模型输入/输出维度错误
- ONNX 推理时 tensor shape mismatch crash

### 4. 不要做棋盘旋转（canonicalize_action）

早期版本 `IFeatureEncoder` 提供了 `canonicalize_action` / `decanonicalize_action` 钩子，允许把棋盘旋转到「当前玩家视角」。实测这是陷阱：

- 格子旋转容易写对，但和格子绑定的附加结构（如 Quoridor 墙的「挡哪两条边」的语义）非常容易旋转错
- 写错时训练看上去能跑，loss 正常下降，但某一方的策略永远学不出来（因为旋转后的动作 id 对应的物理语义根本和 encoder 看到的局面不匹配）
- 这类 bug 极难被发现，Quoridor 训不动就是这么来的

正确做法：**不旋转棋盘**。视角处理只做「我的特征 / 对手的特征」的交换（把 perspective_player 放前面，对手放后面），再加一个 scalar 特征告诉网络「我是先手还是后手」（或等价的「我走哪个方向」）。网络自己会学到 P0/P1 的不对称，省下来的复杂度远超过这个 scalar 的代价。

### 5. do_action_fast 和 undo_action 必须完美逆操作

`undo_action` 必须将状态精确恢复到 `do_action_fast` 之前的状态。常见遗漏：
- 忘记恢复 `current_player`、`winner`、`terminal`
- 忘记恢复 score 数组
- 忘记清除放置的棋子/墙/牌

这个 bug 特别隐蔽：MCTS 的 tree search 重度依赖 do/undo 循环，状态恢复不完整会导致搜索树污染，表现为莫名其妙的走法。

### 6. 随机游戏的 rng_nonce() 必须在每次随机事件后变化

`default_stochastic_detector` 通过比较 `rng_nonce()` 的变化来检测随机转移。如果你的游戏有隐藏信息或随机事件（如翻牌、抽卡），`rng_nonce()` 必须在每次这类事件后返回不同的值。否则 NoPeek 系统无法正确工作。

### 7. 使用 checked_cast 而非 static_cast 做状态类型转换

引擎提供了 `board_ai::checked_cast<T>(state)` 辅助函数，它在 cast 失败时抛出 `std::invalid_argument`。在 Rules 和 Encoder 的实现中，始终使用 `checked_cast` 而非 `static_cast` 来确保类型安全。

### 8. legal_actions 在 terminal 状态必须返回空

如果 `is_terminal()` 为 true，`legal_actions()` 必须返回空 vector。否则 selfplay 循环不会正确终止，可能导致无限循环或崩溃。

### 9. 多人变体的 feature_dim 和 2p 不同

如果你的游戏支持 3p/4p 变体，**每个变体的 feature_dim 通常不相等**。因为 encoder 会为每个对手编码独立的特征通道——2p 有 1 个对手通道，3p 有 2 个，4p 有 3 个。

实测数据（单位：float 个数）：

| 游戏 | 2p | 3p | 4p |
|------|-----|-----|-----|
| Splendor | 295 | 355 | 415 |
| Azul | 163 | 235 | 307 |

**影响**：
- `game.json` 中的 `feature_dim` 只记录了 2p 的值——因为 3p/4p 变体共享同一个 `game.json`
- 多人变体的 encoder 在运行时报告正确的 `feature_dim()`，**网络创建和 ONNX 导出必须使用 encoder 报告的值**，不能从 config 文件读
- **每个人数变体需要独立训练独立的网络**——2p 模型无法用于 3p/4p 对局（tensor shape 不匹配）

**常见错误**：

```python
# 错：用 config 里的 feature_dim 创建 3p 模型
cfg = load_game_config("splendor")  # feature_dim=295, 但 3p 实际是 355
net = PVNet(cfg["feature_dim"], ...)  # shape 错误

# 对：用 encoder 报告的实际值
info = dinoboard_engine.encode_state("splendor_3p", seed=42)
net = PVNet(info["feature_dim"], ...)  # 355, 正确
```

**建议**：如果要支持多人变体训练，每个变体应有独立的训练配置（或由 pipeline 动态查询 encoder 的 feature_dim），不能假设和 2p 相同。

---

# 设计决策（DEC）

以下条目记录显式架构决策（不是 bug 修复），作用域跨越框架，未来重构时必须显式确认或迁移。

## [DEC-001] 旧 2p 标量价值头永久兼容（显式契约，不是 BUG）

### 背景

引入 N-dim value head 之前，所有 2p 游戏（tictactoe / quoridor 2p / splendor 2p / azul 2p / loveletter 2p / coup 2p）训练出的 `model_best.onnx` 都是 `[1, 1]` 标量输出 = perspective player 在 [-1, 1] 上的期望价值。这些模型已经在 web 对局、录像分析、智能提示、eval 流水线中被反复使用，重新训练成本不可忽视。

### 契约

`OnnxPolicyValueEvaluator::evaluate`（`engine/infer/onnx_policy_value_evaluator.cpp`）在 `value_len == 1 && num_players == 2` 时显式按 zero-sum 把标量 `v` 展开为 `(v_perspective, -v_opponent)` 并返回长度 2 的 values 向量。下游一切（`net_mcts.cpp` 的 leaf backup、`bindings/py_engine.cpp` 暴露的 `root_values` / `action_values`、`platform/game_service/pipeline.py` 的 winrate pill / drop-score）都是维度无关的，不需要也不应该有 scalar-aware 分支。

3p+ 的 `value_len == 1` 必须抛错——zero-sum 在 N>2 没有唯一分解，silent broadcast 违反 "No silent degradation" 原则。

### 回归保护

`tests/framework/test_scalar_value_head_compat.py` 把以下行为钉死：

- 标量 `[1, 1]` ONNX 在 `GameSession.get_ai_action` / selfplay / arena 三条路径都能跑，`root_values` 长度=2 且 zero-sum。
- `pipeline._human_wr_from_stats` / `_human_wr_for_action` 在标量模型上返回 [0, 1] 之间的胜率，且两个玩家胜率互补。
- 3p loveletter + 标量 head 必须抛 `value output length` 错误。

### 教训

旧二阶段模型（pre-N-dim）+ 新代码（N-dim 期望）是一种隐性 ABI。重构 value 解码路径时（任何对 `OnnxPolicyValueEvaluator::evaluate` value branch 的修改），必须先确认 scalar 2p 分支保留或显式迁移；一行删掉就会让所有 2p 旧模型悄无声息地失效——症状是 web 胜率/分析直接 throw `value output length 1 for 2 players` 之类的运行期错误，没有 evaluator 这一层兜底就根本走不通推理路径。

---

## [DEC-002] Warmstart 并入 heuristic_guidance schedule

**类型**：架构决策（不是 BUG）
**日期**：2026-05-08

### 决策

删除独立的 warm start 阶段（`_worker_warm_start` + 一次性收集 N 局 + 跑 M epoch + 导出 `model_warm.onnx` + 种子 replay buffer 整套约 90 行）。把它的角色完全并入 `heuristic_guidance` 三段式 schedule（hold → 线性衰减 → 0）。`heuristic_guidance_initial_ratio = 1.0` + 大 `heuristic_guidance_hold_steps` 等价于"前 N 步 100% 启发式自博弈"，样本走主 replay buffer，每步训练。

### 配置变化

废弃（出现即 `raise ValueError`）：

- `warm_start_episodes` / `warm_start_epochs` / `warm_start_heuristic` / `warm_start_temperature`

新增 / 拆分：

- `heuristic_guidance_hold_steps`：hold 期长度（前 N 步锁在 `initial_ratio`）
- `heuristic_guidance_temperature`：**selfplay** 启发式分支温度（warm 期多样性，旧 `warm_start_temperature` 的延续）
- `heuristic_temperature`：**eval vs heuristic** 时启发式对手温度（强度基准，独立于 selfplay）

旧 `warm_start_temperature` 与 `heuristic_temperature` 的语义混淆（同一温度用于两个完全不同目标）就此消除。

### 旧痛点 → 新方案对应

1. **warmstart 局数受内存限制** → 三段式每步只收集 `episodes_per_step` 局，进 deque maxlen，永远不会爆。
2. **N epoch full pass 相关性高** → hold 期每步只抽 `train_batches_per_step` 个 mini-batch（默认 3），样本随 replay buffer 自然 decorrelate。
3. **配置面冗余** → 4 + 3 = 7 键 → 5 键（去掉 4 个 warm_start_*，加 1 个 hold_steps + 1 个 temperature 拆分）。
4. **`model_warm.onnx` 特殊路径** → 删除。selfplay 在 hold 期 `heuristic_guidance_ratio = 1.0` 时，`use_heuristic` 分支结构性 short-circuit 不调 evaluator，即使读 `model_init.onnx` 也根本不评估，所以特殊导出多余（参考 `engine/runtime/selfplay_runner.cpp:93-192`）。

### 训练增强屏蔽（结构性，非配置）

hold 期 `heuristic_guidance_ratio = 1.0` 时，selfplay_runner 的 `use_heuristic` 分支在 `continue` 之前完全不触碰 dirichlet / tail_solve / training_filter / simulations。屏蔽是结构性的，不依赖把这些选项配 0；用户依然可以保留它们的全局配置，hold 期自然不生效，hold 之后 ratio 衰减时按调度概率混合走 MCTS 时这些增强才生效。

### 兼容旧 `model_warm.onnx`

外部脚本如果还引用 `models/<game>/model_warm.onnx`（例如旧的 arena 比对命令），改读 `model_init.onnx`（hold 期开始前的初始网络快照）或 `model_step_NNNNN.onnx`（按 `save_every` 存档）。

### 迁移示例

旧 `quoridor` 配置：
```
"warm_start_episodes": 800,
"warm_start_epochs": 10,
"warm_start_heuristic": true,
"warm_start_temperature": 3.0,
```

新等价：
```
"heuristic_guidance_hold_steps": 8,
"heuristic_guidance_steps": 200,
"heuristic_guidance_initial_ratio": 1.0,
"heuristic_guidance_temperature": 3.0,
```

折算依据：旧 warmstart 总有效样本数 ≈ `warm_start_episodes`（一个样本被训 `warm_start_epochs` 次，但相关性高，按一次计入）。新方案每步 `episodes_per_step = 100` 局；hold 8 步 ≈ 800 局，等量。`decay_end = 200` 让网络在 hold 期之后再有 ~200 步线性衰减平滑接管。

### 教训

- **重复 abstraction 是配置的债**。两条几乎同样目标（启发式当老师）的路径并存几个月，每加一个游戏就要在两套语义之间选——选错只会在多周训练后才暴露。统一到一条 schedule 后，hold 期 vs 衰减期 vs 0 期的过渡是一条平滑曲线，没有"warmstart 结束-MCTS 开始"那个突变点。
- **特殊路径多一个就多一个失败点**。`model_warm.onnx` 在 BUG-010（init / warm 同 hash）和 BUG-011（silent ONNX 退化）里都参与过排查，但它本身不解决任何问题——只是 warmstart 阶段为了"区分 init 和 warmed 的网络"留下来的中间产物。删掉之后这个角色由 `model_init.onnx`（已存在）+ `model_step_NNNNN.onnx`（存档机制已有）覆盖，不留特殊点。

---

## [DEC-003] Session 不再每步 randomize_unseen + IGameRules wrapper 接管 step_count

**类型**：架构决策（不是 BUG）
**日期**：2026-05-13

### 决策

两条互相独立但同步落地的契约收紧：

1. **session 上的 viz=0 槽位永远不被框架 freshen**。`apply_observation` / `advance_per_seat_states` / `advance_ai_view_` 都不再调 `tracker.randomize_unseen` 把"belief 采样"灌回 session state。session 上的 viz=0 字节是 `reset_with_seed` 写入的初始噪声 + 后续若干步的"未读残值"，**结构性不可读**——hash 走 `kHiddenHashSentinel`，encoder 走 `kPlaceholder`，MCTS sim 在 sim_tracker 克隆上调 `randomize_unseen` 重采。
2. **step_count_ bookkeeping 进 IGameRules wrapper**。`do_action_fast` / `do_action_deterministic` / `undo_action` 改成 non-virtual public wrapper，自动在 protected `*_impl` 之前 / 之后 ±1。`step_count_` 字段 protected + `friend class IGameRules`，作者既看不到也无法忘记 / 双 bump。Session 路径上仍需要在 snapshot 替换前 bump 一次，框架对外暴露 `begin_step_for_session_observe()`（命名故意冗长，劝退游戏代码）。

### 为什么

**对 (1)**：session 上的 viz=0 内容**从来就不应该被读**——四面墙（belief 接口物理拿不到 `IGameState*`、observer 不调 `do_action_fast`、决策侧三处读全替换为 placeholder/sentinel、selfplay/web/API 三条路径同栈）已经保证它不可达。每步把它"换上一份新采样"是**冗余的 hygiene**，伪装成"belief 注入 state"——任何指望它的代码都是契约违规。删掉之后：
- 多人 hidden-info 游戏每步省一次 walker + 一次 tracker.randomize_unseen
- 不再有"session viz=0 内容像是某种共享 belief"的语义错觉
- `test_public_hash_excludes_internal_rng`（60-seed 扫 4 hidden-info game）继续是结构性回归保护——两个不同 session_rng 起的 session 在 viz=0 槽位上字节会差，但 perspective hash 必须 byte-equal

**对 (2)**：之前 step_count 由游戏作者在 `do_action_fast` 第一行 `s->begin_step()`、`undo_action` 末尾 `s->end_step()` 手动维护。这是 BUG-037 同一档次的隐患——**只要忘一次 / 双 bump 一次，DAG 就破环**，但作者无法从签名读出这个义务。改成 wrapper 之后：
- protected 字段 + IGameRules friend = 作者既写不进也读不出 step_count_，**结构性无法破坏 invariant**
- public wrapper 的 wrapper 注释直接说明了 invariant 和职责
- `tests/framework/test_step_count_strict_increase.py` 跑遍所有启用游戏，assert 每动作 step_count 严格 +1，作为结构性回归

### 落地

**Plan 1 §4+§6**：
- `engine/core/game_interfaces.h`：`step_count_` protected，IGameRules friend；新增 `reset_step_count_base()` 给游戏作者用、`begin_step_for_session_observe()` 给框架内部用；IGameRules 拆 wrapper / impl，新增 protected static `invoke_*_impl` 让 sibling instance（FilteredRulesWrapper）能跨实例调 impl
- `engine/runtime/selfplay_runner.h`：`FilteredRulesWrapper` 改成只重写 `*_impl`，wrapper 转发通过 `invoke_*_impl(inner_, ...)`
- 6 个游戏 rules（tictactoe / quoridor / azul / splendor / loveletter / coup）：方法重命名 `_impl` 移到 protected，删除 `s->begin_step()` / `s->end_step()`，`do_action_fast_impl` 返回 void
- `engine/runtime/{selfplay,arena,heuristic}_runner.cpp` + `bindings/py_engine.cpp`：所有 session-snapshot 路径上的 `seat.begin_step()` 改成 `seat.begin_step_for_session_observe()`

**Plan 2**：
- `engine/runtime/{selfplay,arena,heuristic}_runner.cpp`：删除 `advance_per_seat_states` 末尾的 `tracker->randomize_unseen(seat, p, freshen_rng)`
- `bindings/py_engine.cpp`：删除 `apply_observation` / `advance_ai_view_` 末尾的同一调用
- `engine/search/net_mcts.cpp`：sim 入口的 `sim_tracker->randomize_unseen(sim_state, sim_rng)` 保留（这是唯一一处合法调用）

### 回归保护

- `tests/framework/test_step_count_strict_increase.py`：5 game × 80 ply，assert step_count 严格 +1
- `tests/framework/test_public_hash_excludes_internal_rng.py`：60-seed × 4 hidden-info game，assert 两个不同 session_rng 起的 session 在 perspective hash 上 byte-equal（守 session viz=0 不被决策侧读）
- `tests/framework/test_public_snapshot_round_trip.py`：每个 hidden-info game、每 ply：truth → masked → wire snapshot → observer apply，observer hash byte-equal truth（守 begin_step_for_session_observe 仍然在 snapshot apply 前推进 step）
- 全套 `tests/framework/`（702 通过）+ 5 个 per-game 套件（199 通过）

### 教训

- **义务藏在签名里就总会被忘**。`do_action_fast` / `undo_action` 一直在结构上要求作者手 bump step_count_，但签名上看不出，**几个月里没出过 bug 不代表它不会出 bug**——作者新增游戏时只要复制粘贴一份漏掉 begin_step 就破环。把义务从"作者必须做对"挪到"框架自动做对、作者无法搞错"，bug 就消失在结构里。
- **冗余 hygiene 是错觉的温床**。之前 session 每步 randomize_unseen 看起来像"维持 session 的 belief 状态最新"，但 session viz=0 从来不该被读——任何依赖它的代码都是契约违规。删掉之后契约从"作者别读 session viz=0"收紧为"框架不维护 session viz=0"，差一个层级的强度但好检查得多。
- **结构性测试比一次性测试值钱**。新加的 `test_step_count_strict_increase` 不是测某个 bug 的修复，而是测"框架 invariant 仍然成立"——加一个游戏就自动覆盖一个游戏，作者忘不了也无法绕过。

---
# 框架层 Issues

以下所有 issue 都位于 `engine/` / `engine/runtime/` / `engine/search/` / `training/` / `bindings/` / `platform/` ——修改框架代码前应通读，跨所有游戏生效。

## [BUG-001] Tail Solver 转置表标志位反转

**状态**：已修复
**文件**：`engine/search/tail_solver.cpp`
**严重程度**：高 — 导致残局求解结果错误

### 问题描述

Alpha-Beta 搜索的 minimizing 分支中，转置表（TT）的初始标志位和截断标志位互换了：

```
修复前（错误）：
  maximizing 分支：initial = kUpperBound, cutoff = kLowerBound  ✓ 正确
  minimizing 分支：initial = kUpperBound, cutoff = kLowerBound  ✗ 错误（和 max 一样了）

修复后（正确）：
  maximizing 分支：initial = kUpperBound, cutoff = kLowerBound  ✓
  minimizing 分支：initial = kLowerBound, cutoff = kUpperBound  ✓
```

### 根因分析

Alpha-Beta 中，TT 标志位的含义：
- **kExact**：存储的值是精确值
- **kLowerBound**：存储的值是真实值的下界（发生了 beta 截断）
- **kUpperBound**：存储的值是真实值的上界（没有提升 alpha）

对于 **maximizing** 分支：
- 初始 best_value = -∞，尚未搜索任何子节点 → 这是一个上界（kUpperBound）
- 搜索过程中提升了 alpha → 变为精确值（kExact）
- 发生 alpha >= beta 截断 → 实际值至少这么大，是下界（kLowerBound）

对于 **minimizing** 分支（关键区别）：
- 初始 best_value = +∞，尚未搜索任何子节点 → 这是一个下界（kLowerBound）
- 搜索过程中降低了 beta → 变为精确值（kExact）
- 发生 alpha >= beta 截断 → 实际值至多这么小，是上界（kUpperBound）

修复前 minimizing 分支的初始和截断标志完全反了，导致转置表在后续查询中做出错误的剪枝决策。

### 影响

Tail solver 可能错误地判断胜负，导致在残局阶段选择劣势走法。由于 tail solver 的结果会覆盖 MCTS 的选择（temperature=0 确定性选择），一旦结果错误，影响直接且严重。

### 修复代码

```cpp
// minimizing 分支 — 修复后的版本
best_value = 2.0f;
flag = TTFlag::kLowerBound;  // 修复：初始为下界（非 kUpperBound）
for (const ActionId action : ordered) {
    // ... alpha-beta 搜索 ...
    if (best_value < beta) {
        beta = best_value;
        flag = TTFlag::kExact;
    }
    if (alpha >= beta) {
        flag = TTFlag::kUpperBound;  // 修复：截断为上界（非 kLowerBound）
        break;
    }
}
```

---

## [BUG-002] 平局 z 值赋值错误

**状态**：已修复
**文件**：`engine/runtime/selfplay_runner.cpp`
**严重程度**：中 — 偏置训练数据

### 问题描述

selfplay_runner 在游戏平局时将所有样本的 z 值设为 -0.5，而非 0.0。

```
修复前：s.z = -0.5f;   // 平局被当作半输
修复后：s.z = 0.0f;    // 平局是中性结果
```

### 根因分析

z 值是训练 value head 的目标标签：+1 表示赢、-1 表示输、0 表示平局。使用 -0.5 意味着平局被当作「比输好一点但仍然是负面结果」。

### 影响

- Value head 会学到「平局不好」，导致模型在应该接受平局的位置过度冒险
- 对于平局很少的游戏（如 Quoridor）影响较小，但对于平局常见的游戏（如 TicTacToe）影响显著
- 同样的 bug 存在于 terminal 和 adjudicator 两个分支中，都已修复

---

## [BUG-003] 训练-评估动作空间不一致

**状态**：已修复
**文件**：`bindings/py_engine.cpp`, `training/pipeline.py`
**严重程度**：高 — 导致评估结果完全不可信

### 问题描述

当使用 `TrainingActionFilter`（如 Quoridor 的最短路径差约束）进行自我对弈训练时，模型只在约束后的动作空间内学习策略。但评估（eval vs heuristic）使用的是无约束的完整动作空间。

结果：模型在约束空间内已经能和 heuristic 五五开，但评估显示 0% 胜率。

### 根因分析

- 自我对弈：`run_selfplay_episode` 接收 `training_action_filter` 参数，通过 `FilteredRulesWrapper` 将约束注入 MCTS 搜索
- 评估：`run_arena_match` 不使用 `training_action_filter`，模型被迫在完整动作空间中选择，而它从未训练过这些动作
- 这本质上是一个 **distribution shift** 问题：训练和测试的动作分布不一致

### 修复方案

1. 在 `py_engine.cpp` 中添加 `run_constrained_eval_vs_heuristic` 函数：
   - 模型侧通过 `FilteredRulesWrapper` 约束动作空间
   - Heuristic 侧也通过 `FilteredRulesWrapper` 约束（公平对比）
2. 在 `pipeline.py` 中同时运行两种评估：
   - `eval vs heuristic (constrained)` — 约束对局，反映模型在训练分布内的实力
   - `eval vs heuristic (free)` — 自由对局，反映模型在完整游戏中的实力

### 教训

**当训练使用动作约束时，必须同时提供约束和非约束两种评估**。只看非约束评估会误判模型完全没有学到东西。约束评估是衡量学习进度的真实指标，非约束评估是衡量泛化能力的指标。

---

## [BUG-004] Heuristic Runner 缺少 Adjudicator 支持

**状态**：已修复
**文件**：`engine/runtime/heuristic_runner.h`, `engine/runtime/heuristic_runner.cpp`
**严重程度**：中 — warm start 训练数据标签错误

### 问题描述

`run_heuristic_episode()` 不接受 `adjudicator` 参数。当 heuristic 对局达到 `max_game_plies` 而游戏未终局时，所有样本的 z 值默认为 0.0（未知结果），而不是通过 adjudicator 判定胜负。

### 影响

- Warm start 阶段使用 heuristic 对局生成训练数据
- 对于像 Quoridor 这样经常不能在 ply 限制内结束的游戏，大量样本的 z 值为 0（既不是赢也不是输）
- 这些错误标签会误导 value head，降低 warm start 的效果

### 修复

在 `run_heuristic_episode` 的参数列表中添加 `GameAdjudicator adjudicator`。当游戏超时未终局时，调用 adjudicator 判定胜负并正确分配 z 值。

---

## [BUG-005] FilteredRulesWrapper 的 const_cast

**状态**：已修复（2026-05-08）
**文件**：`engine/core/game_registry.h`、`engine/runtime/selfplay_runner.h`、`games/quoridor/quoridor_register.cpp`
**严重程度**：低 — 历史 workaround，已清理

### 问题描述

历史上 `IGameRules::legal_actions` 接口签名是 `const IGameState&`，但 `TrainingActionFilter` 类型是 `IGameState&`（早期 quoridor filter 设计想直接 do_action + undo_action）。`FilteredRulesWrapper::legal_actions` 用 `const_cast` 把 const 强转掉：

```cpp
auto filtered = filter_(const_cast<IGameState&>(state), inner_, legal);
```

### 修复

把 `TrainingActionFilter` 第一参数改为 `const IGameState&`，filter 内部需要模拟动作时用 `state.clone_state()` 做隔离副本（quoridor 现行实现已经是这种模式）。`FilteredRulesWrapper::legal_actions` 直接 `filter_(state, inner_, legal)`，去 const_cast。

---

## [BUG-006] Replay Buffer 样本利用率

**状态**：已知行为（可接受）
**文件**：`training/pipeline.py`
**严重程度**：低 — 这是在线 RL 的正常行为

### 观察

Replay buffer 大小为 `episodes_per_step * 50 * 20`（约 100,000 个样本）。每步训练生成约 5,000 个新样本（100 episode * ~50 ply）。`train_epochs=3` 表示每步训练遍历整个 buffer 3 次。

结果：每个样本在被新数据淘汰前，平均只被训练 1-2 次。

### 是否是问题

**通常不是**。在线 RL（如 AlphaZero）中，使用每个样本少量次数可以防止对过时数据的过拟合。AlphaZero 原论文也使用类似的滑动窗口策略。

### 调参建议

- 如果训练 loss 不稳定：增大 buffer（提高 buffer multiplier）或减少 `episodes_per_step`
- 如果收敛太慢：减小 buffer 使数据更新鲜，或增加 `train_epochs`
- 如果 value head 质量差：增大 buffer 以增加训练样本多样性

---

## [BUG-007] pipeline.py 用初始局面特征训练所有样本

**状态**：已修复
**文件**：`training/pipeline.py`
**严重程度**：致命 — 模型在噪声上训练

### 问题描述

`pipeline.py` 在收集训练数据时调用 `encode_state(game_id, seed + sample["ply"])`。`encode_state` 始终创建一个新游戏（初始局面），用不同的 seed 参数——它不会重建第 N 步的棋局。结果：每个训练样本的 features 都是某个初始局面的编码，而 policy/value 标签来自实际对局中的中间局面。

### 根因分析

C++ selfplay runner 已经在每个采样点正确编码了 features（`SelfplaySample::features`），并通过 pybind11 返回给 Python。但 `pipeline.py` 忽略了这些 features，试图自己重新编码——然而 `encode_state` 的设计只是"给定 seed 创建初始局面并编码"，无法重建中间状态。

### 影响

- 所有训练样本的 features 与 policy/value 标签完全不匹配
- 模型学到的是随机噪声，无法泛化
- 这解释了 v6/v7/v8 训练中 free eval 始终为 0% 的根本原因
- constrained eval 能达到 ~42% 只是因为 training filter 大幅缩小了动作空间，使模型即使用错误特征也能碰巧选到不太差的动作

### 修复

使用 C++ 返回的 `sample["features"]` 作为训练输入，不再调用 `encode_state`。

### 教训

**永远不要在 Python 侧重新实现 C++ 已经做好的事**（详见 CLAUDE.md 的 "Training Pipeline" 原则）。features 的编码必须发生在采样点的实际游戏状态上，而 C++ selfplay runner 正是这么做的。

---

## [BUG-008] Splendor temperature_schedule 被静默忽略

**状态**：已修复
**文件**：`training/pipeline.py`
**严重程度**：中 — Splendor 训练时温度衰减完全失效

### 问题描述

`pipeline.py` 读取 `train_cfg.get("temperature_initial", -1.0)` 等平坦键来配置温度衰减。但 Splendor 的 `game.json` 使用嵌套结构：

```json
"temperature_schedule": {
    "enabled": true,
    "initial": 1.0,
    "final": 0.1,
    "decay_plies": 30
}
```

平坦键 `temperature_initial` 不存在，pipeline 得到默认值 -1.0，C++ 侧判断为"未启用"，温度衰减被静默跳过。

### 修复

添加 `_get_temperature_key(train_cfg, key, default)` 辅助函数：先查找平坦键 `temperature_{key}`，不存在则回退到 `temperature_schedule.{key}`。两种配置格式均可正确读取。平坦键优先，确保向后兼容。

### 教训

**配置读取必须有对应的测试**。pipeline 读取的每个 `game.json` 键都应该有测试验证实际值是否到达 C++ 侧。静默的默认回退是最危险的 bug 类别之一。

---

## [BUG-009] pipeline.py 重写丢失三项训练改进

**状态**：已修复
**文件**：`training/pipeline.py`
**严重程度**：高 — 三项 AlphaZero 标准训练技术被静默丢弃

### 问题描述

修复 BUG-007 时重写了 `pipeline.py` 的样本收集逻辑。重写过程中遗漏了三项此前已实现的训练改进：

1. **Legal mask masking**：`train_step()` 不再对非法动作的 policy logits 做 mask（`masked_fill(mask == 0, -1e9)`）。结果：网络可以在非法动作上分配概率，policy loss 包含无意义的梯度信号。
2. **AdamW 优化器**：优化器从 `AdamW`（带 weight decay 的 L2 正则化）退化为 `Adam`。结果：缺失正则化，训练后期容易过拟合。
3. **梯度裁剪**：`clip_grad_norm_` 调用丢失。结果：训练不稳定时梯度爆炸无保护。

### 根因分析

三项改进的代码散落在 `train_step()` 和 `run_training_loop()` 中，没有独立的单元测试保护。重写 `pipeline.py` 时注意力集中在样本收集逻辑的修复上，没有逐行比对旧代码。

### 如何发现

通过测试驱动发现：编写 `test_sample_collection.py` 时为每个训练特性写了独立断言（检查 optimizer 类型为 AdamW、train_step 参数列表包含 legal_mask、grad_clip_norm 参数被使用），这些断言立即暴露了缺失。

### 修复

```python
# 1. Legal mask masking（train_step 内）
if legal_mask is not None:
    policy_logits = policy_logits.masked_fill(legal_mask == 0, -1e9)

# 2. AdamW 优化器（run_training_loop 内）
weight_decay = train_cfg.get("weight_decay", 1e-4)
optimizer = torch.optim.AdamW(net.parameters(), lr=learning_rate, weight_decay=weight_decay)

# 3. 梯度裁剪（train_step 内）
if grad_clip_norm > 0:
    torch.nn.utils.clip_grad_norm_(net.parameters(), grad_clip_norm)
```

### 教训

**每个行为特性都需要独立的测试保护，而非只依赖集成测试**。BUG-007 的 fix 通过了"selfplay 能跑、训练 loss 能降"的集成测试，但这些粗粒度的测试无法检测到 legal mask、optimizer 类型、梯度裁剪等细节是否存在。对于训练管线，应该有：

- `train_step` 的参数完整性断言（接受 legal_mask、grad_clip_norm）
- optimizer 类型检查（`isinstance(optimizer, torch.optim.AdamW)`）
- legal mask 功能验证（非法动作概率接近 0）
- 梯度裁剪效果验证（大梯度被截断）

这些断言编写成本极低，但能防止重写时的静默退化。

---

## [BUG-010] pipeline.py 重写丢失 Replay Buffer

**状态**：已修复
**文件**：`training/pipeline.py`
**严重程度**：高 — 自博弈训练无法累积学习

### 问题描述

BUG-007 修复时重写了 `pipeline.py`，丢失了跨步累积的 replay buffer。旧代码维护一个 `deque(maxlen=100000)` 的滑动窗口 buffer，每步新 selfplay 样本追加到 buffer，训练从整个 buffer 采样。重写后变为每步只用当前步的 ~7000 个样本训练，训完就丢弃。

### 根因分析

重写注意力集中在样本收集逻辑（`encode_state` → `sample["features"]`），没有注意到 replay buffer 的存在。原代码中 buffer 是一个模块级别的 list 加手动 truncation，容易被忽略。

### 如何发现

v10 训练 150 步后对比 warm start 模型和 step 50/100/150 模型的 eval 战绩，发现完全一致（35% constrained win rate vs heuristic）。模型权重在变（ONNX 文件 hash 不同），但策略没有改善。

### 影响

- 每步只从 ~7000 个样本学习 3 次（3 个 batch × 2048），数据利用率极低
- 模型每步都在"从头学"当前步的数据，无法在多步间累积知识
- 旧代码（v6）日志显示 `samples=100000`（buffer 满载），新代码显示 `samples=6916`（仅当前步）

### 修复

```python
replay_buffer: deque[tuple] = deque(maxlen=episodes_per_step * 50 * 20)

for step in range(1, steps + 1):
    # ... selfplay ...
    for sample in ep["samples"]:
        replay_buffer.append((feats, policy, z, mask, aux))
    
    # 从 buffer 随机抽 train_batches_per_step 个 batch 训练
    buf = list(replay_buffer)
    for b in range(train_batches_per_step):
        idx = torch.randint(n, (batch_size,))
        train_step(net, optimizer, feat_tensor[idx], ...)
```

配置项：
- `train_batches_per_step`（默认 3）：每步从 buffer 抽几个 batch
- `batch_size`（默认 2048）：每个 batch 的大小

### 教训

BUG-009 的教训同样适用：**重写代码时必须逐行比对旧代码**。这已经是 pipeline 重写丢失的第四项特性（legal mask、AdamW、梯度裁剪、replay buffer）。replay buffer 尤其危险——没有它训练也能跑、loss 也能降，但模型不会真正进步。

---

## [BUG-011] ONNX 未编译导致 MCTS 使用均匀策略

**状态**：已修复
**文件**：`setup.py`, `engine/infer/onnx_policy_value_evaluator.cpp`
**严重程度**：致命 — 神经网络完全没有参与 MCTS 搜索

### 问题描述

`setup.py` 默认 `BOARD_AI_WITH_ONNX=0`，需要手动设置环境变量 `BOARD_AI_WITH_ONNX=1` 和 `BOARD_AI_ONNXRUNTIME_ROOT` 才能编译 ONNX 支持。未设置时，`OnnxPolicyValueEvaluator` 在运行时静默回退到均匀策略（`priors = uniform, values = 0`）——**不报错、不警告、返回 true**。

结果：MCTS 搜索完全不使用神经网络。selfplay 生成的是纯搜索数据（无 NN 引导），eval 中所有模型行为完全相同（因为都是均匀 MCTS）。

### 如何发现

不同训练步骤（warm, step50, step100, step150）的 constrained eval 结果逐局完全一致（winner、total_plies 完全相同）。即使一个是随机初始化模型、另一个是训练 150 步后的模型。

验证方式：
1. 检查 .so 文件中是否存在 `"onnx runtime not enabled at build time"` 字符串 → 存在
2. 检查 `"dino_onnx_eval"` 字符串（ONNX 环境名称） → 不存在
3. `otool -L` 检查动态库依赖 → 无 onnxruntime

### 根因分析

`OnnxPolicyValueEvaluator::evaluate()` 在 ONNX 未编译时，静默返回均匀策略并返回 `true`。调用者无法区分"ONNX 正常运行"和"回退到均匀"。这是一个**静默降级**设计缺陷。

### 修复

1. **`setup.py`**：添加自动检测逻辑，检查 `/opt/homebrew` 和 `/usr/local` 是否存在 onnxruntime 头文件。如果存在则自动启用 ONNX，无需手动设置环境变量。未找到时打印 WARNING。
2. **`onnx_policy_value_evaluator.cpp`**：构造函数在 ONNX 未编译时抛出 `std::runtime_error`（而非静默设置 ready=false）。`evaluate()` 所有失败路径均抛异常（不再有 `return false`）。
3. **`net_mcts.h`**：删除 `UniformPolicyValueEvaluator` 类。不再存在均匀策略回退。
4. **`bindings/py_engine.cpp`**：`run_selfplay_episode`、`run_arena_match`、`run_constrained_eval_vs_heuristic` 在 `model_path` 为空时直接 `throw std::invalid_argument`。

### 教训

**静默降级是最危险的设计模式**。当一个关键子系统不可用时，系统应该 fail-fast 而不是继续运行。以下代码模式应被视为 bug：

```cpp
// 错：静默降级为无效数据
if (!ready_) {
    priors = uniform;
    return true;  // 调用者以为一切正常
}

// 对：告诉调用者此功能不可用
if (!ready_) {
    return false;  // 调用者可以采取补救措施
}
```

---

## [BUG-012] model_init.onnx 导出顺序错误

**状态**：已修复
**文件**：`training/pipeline.py`
**严重程度**：低 — 只影响 init 模型文件的正确性，不影响训练

### 问题描述

`pipeline.py` 在 warm-start 训练完成后才导出 `model_init.onnx`。此时 `net` 已经包含 warm-start 训练后的权重，所以 `model_init.onnx` 和 `model_warm.onnx` 完全相同（同一个 md5 hash）。

```python
# 修复前（错误）：
net = create_model(...)   # 随机权重
# ... warm start training ...
export_onnx(net, "model_warm.onnx")   # warm 权重
export_onnx(net, "model_init.onnx")   # 也是 warm 权重！相同 md5
current_model_path = "model_init.onnx"

# 修复后（正确）：
net = create_model(...)   # 随机权重
export_onnx(net, "model_init.onnx")   # 真正的初始权重
# ... warm start training ...
export_onnx(net, "model_warm.onnx")   # warm 权重
current_model_path = "model_warm.onnx"
```

### 影响

- `model_init.onnx` 不代表真正的初始（随机）模型
- 这与 BUG-011 结合时，使调试更加困难——即使手动比较 init 和 warm 的输出，也看不到差异
- 训练本身不受影响（`current_model_path` 指向的模型权重是对的）

---

## [BUG-013] ONNX 不是每步导出，selfplay 用旧模型

**状态**：已修复
**文件**：`training/pipeline.py`
**严重程度**：高 — 训练数据与网络严重脱节

### 问题描述

`run_training_loop` 只在 `step % save_every == 0` 时导出 ONNX 并更新 `current_model_path`。`save_every` 默认等于 `eval_every`（默认 50）。这意味着 step 1~49 的 selfplay 全部使用同一个旧模型（warm/init），但 PyTorch 网络已经更新了 ~150 次梯度。

### 修复

每步训练后都导出到 `model_latest.onnx`，selfplay 立刻用新权重。`save_every` 只控制是否额外保存 `model_step_NNNNN.onnx` 存档。

---

## [BUG-014] best model 路径指向 latest 文件被覆盖

**状态**：已修复
**文件**：`training/pipeline.py`
**严重程度**：高 — gating 机制完全失效

### 问题描述

Gating 通过时 `best_model_path = eval_model`，而 `eval_model` 指向 `model_latest.onnx`。下一步训练重新导出 `model_latest.onnx` 后，`best_model_path` 指向的文件内容就变成了新模型。best 永远等于 latest，eval 对弈变成自己打自己。

### 修复

Gating 通过时用 `shutil.copy2()` 复制到 `model_best.onnx`，best 的内容不会被后续训练覆盖。

---

## [BUG-015] pipeline.py 用 z_values 取值但 C++ 不总是填充

**状态**：已修复（两次）
**文件**：`training/pipeline.py`
**严重程度**：致命 — 训练直接崩溃

### 问题描述

清理兜底逻辑时，将 sample 的 value target 提取改为 `z_vals = sample["z_values"]; z = z_vals[sample["player"]]`。但 C++ 侧 `z_values`（per-player 值向量）只在 `is_terminal()` 路径填充。adjudicator 判定和非终局截断路径返回空列表。

### 修复历史

**第一次修复**：统一使用标量 `sample["z"]`（+1/-1/0）。

**第二次修复（N 维 value head）**：value head 改为 N 维输出后，训练需要 per-player 向量而非标量。改为使用 `rotate_z_values(z_vals, player, num_players)`，空 `z_values`（非终局截断）降级为全零向量。adjudicator 路径的 `z_values` 同时修复为零和（BUG-018）。

### 教训

z_values 为空的路径（非终局截断，无 adjudicator）仍然存在。`rotate_z_values` 必须处理空列表情况。

---

## [BUG-016] legal mask 被 filter 缩小导致 free 模式失效

**状态**：已修复
**文件**：`engine/runtime/selfplay_runner.cpp`
**严重程度**：高 — 模型无法学会在完整动作空间下游戏

### 问题描述

`selfplay_runner.cpp` 中 `encoder->encode()` 传入的 `legal` 来自 `effective_rules.legal_actions()`——当 training filter 生效时，`legal` 是过滤后的子集。这导致训练样本的 `legal_mask` 只在 filtered 动作上为 1。

训练时 `train_step()` 用 `legal_mask == 0` 的位置填 `-1e9`，这些位置在 softmax 后概率趋近 0，不参与 cross entropy 梯度计算。被 filter 排除的合法动作从未收到任何梯度信号，logit 保持随机初始化值。

在 free 模式（不用 filter）下，这些随机 logit 参与 softmax 计算，可能产生较高概率，导致模型选择垃圾动作。表现为 heuristic_free eval 胜率 0%，gating 对局全部平局（双方都乱走到超时）。

### 根因分析

`legal_mask` 的设计本意是排除**真正不合法的动作**。被 filter 排除的动作是合法但较差的动作，不应该被 mask 掉——模型需要通过 cross entropy 梯度学到这些动作的概率应为 0。

### 修复

`encoder->encode()` 始终传入完整的 `rules.legal_actions()`（不经过 filter），只在 MCTS 搜索和动作选择时使用 filtered rules。被过滤动作在 policy target 中 visits 为 0，cross entropy 梯度自然将其概率压低。

### 教训

mask 机制有两种语义：(1) "不合法，不存在"——应该 mask 掉；(2) "合法但不好"——应该让模型学到概率为 0。Training filter 属于后者，不能复用 legal mask 通道。

---

## [BUG-018] Adjudicator z_values 不零和 + 标量 value head 的 3p+ 展开 bug

**状态**：已修复
**文件**：`engine/runtime/selfplay_runner.cpp`、`engine/infer/onnx_policy_value_evaluator.cpp`
**严重程度**：高 — 3+ 人游戏的训练和搜索都有误

### 问题描述

三个相关联的多人游戏 bug：

1. **Adjudicator z_values 不零和**：adjudicator 路径赋值 `+1/-1` 给赢家/输家，但 3+ 人游戏的零和目标应为 `+1/-1/(n-1)`。例如 3 人游戏：赢家 +1，输家应为 -0.5 而非 -1。sum 为 0 而非 -1。

2. **标量 evaluator 展开 bug**：ONNX evaluator 的标量分支展开为 `[v, -v, -v, ...]`，但 3+ 人游戏应展开为 `[v, -v/(n-1), -v/(n-1), ...]`。

3. **N 维分支缺少 perspective 旋转**：evaluator 的 N 维分支（`value_len >= num_players`）直接用 `value_ptr[p]` 当绝对顺序，但模型输出是 perspective-relative（与 encoder 旋转对齐），需要旋转回绝对顺序。

### 修复

1. Adjudicator: `adj_vals[p] = (p == winner) ? 1.0f : -1.0f / (n-1)`
2. 标量分支: `opponent_v = -v / (n-1)`
3. N 维分支: `values[(perspective + i) % n] = value_ptr[i]`
4. 两条 z_values 路径（terminal + adjudicator）加零和 assert

### 关联改动

配合 N 维 value head 实现：value head 从 `nn.Linear(prev, 1)` 改为 `nn.Linear(prev, num_players)`，训练 target 从标量 `z` 改为 `rotate_z_values(z_values, player)`。2 人游戏不受影响（`-v/(2-1) = -v`）。

---

## [BUG-019] 多人模式全链路 2p 硬编码

**状态**：已修复
**文件**：`engine/runtime/heuristic_runner.cpp`、`bindings/py_engine.cpp`、`training/pipeline.py`、`platform/game_service/sessions.py`、`platform/game_service/routes.py`、`platform/game_service/pipeline.py`、`platform/static/general/sidebar.js`、`platform/static/general/app.js`
**严重程度**：高 — 3+ 人游戏全链路无法正确运行

### 问题描述

三处独立的 2-player 硬编码阻碍了 3+ 人游戏的正确运行：

1. **heuristic_runner.cpp adjudicator z_values**：与 BUG-018 中 selfplay_runner 相同的 bug —— adjudicator 路径给输家赋 `-1.0f` 而非 `-1.0f/(n-1)`，导致 3p+ 训练数据不零和。

2. **arena match pybind 只支持 2 个模型**：`run_arena_match_py` 接受 `model_path_0`/`model_path_1` 两个参数，无法用于 3p+ gating/eval。C++ 侧的 `run_arena_match` 已支持 N 个 player config，但 pybind 接口是瓶颈。

3. **Web 前端全链路假设 2 人**：session 数据模型使用单一 `ai_player: int`；routes 中 `1 - ai_player` 计算对手；pipeline 分析中 value 取反仅对 2 人零和正确；前端 JS 用 `state.aiPlayer` 单值判断 AI 回合。

### 修复

1. **heuristic_runner.cpp**：adjudicator z_values 改为 `(winner) ? 1.0f : -1.0f/(n-1)`，加零和 assert。

2. **arena pybind**：`run_arena_match_py` 改为接受 `model_paths: list[str]` 和 `simulations_list: list[int]`，加载 N 个 evaluator，factory 用 `player % n_eval` 分配。`training/pipeline.py` 的 `_worker_arena` 和 `run_eval_batch` 同步适配——candidate 轮流坐每个座位，其余填 opponent。

3. **Web 全链路**：
   - session 数据模型：`ai_player: int` → `human_player: int` + `ai_players: list[int]`
   - routes：`CreateGameRequest` 改为 `human_player` 参数；AI 回合判定改为 `current_player in ai_players`
   - pipeline：AI 连续落子循环（while loop 直到轮到人类）；3p+ 跳过 probe analysis（value 取反不适用）
   - 前端 sidebar：新增人数选择按钮组（仅 max > 2 时显示）；多人时座位选择替代先手/后手
   - 前端 app.js：`state.aiPlayer` → `state.aiPlayers`（array）+ `state.humanPlayer`；所有 AI 判定改为 `includes()`

### 教训

在框架"支持多人"的高层设计之后，每个组件（训练数据生成、评估、pybind 接口、web session 模型、前端交互）都需要独立审计多人路径。C++ 核心正确不代表上层接口正确。

---

## [BUG-020] Pipeline 分析使用错误的 stats key 导致 expert AI 卡死

**状态**：已修复
**文件**：`platform/game_service/pipeline.py`
**严重程度**：高 — expert 难度 AI 完全无法走子

### 问题描述

Expert 难度 AI 在落子后 pipeline 立即进入 `error` 状态，表现为 AI 永远不走。根因是 `_analyze_user_move()` 中访问 `precompute_result["stats"]["best_action_value"]`，但 pybind 导出的 key 实际上是 `"best_value"`。

C++ 结构体字段名为 `NetMctsStats::best_action_value`，但 `bindings/py_engine.cpp` 第 630 行导出时做了重命名：`st["best_value"] = stats.best_action_value;`。pipeline.py 使用了 C++ 字段名而非 Python 导出名，导致 `KeyError: 'best_action_value'`。

三处均受影响（pipeline.py 第 134、148、178 行）：
- precompute 结果读取
- fallback inline MCTS 结果读取
- post-move probe 结果读取

### 修复

将 `pipeline.py` 中所有 `["stats"]["best_action_value"]` 替换为 `["stats"]["best_value"]`。

**复发（2026-05-03）**：前端 `platform/static/general/pipeline.js:37-38` 犯了**完全相同的错**——读 `st.ai_stats.best_action_value` 而不是 `best_value`。表现为 Web 对局里"对手预估胜率"永远是 "--"。已修复。这证明教训里说的"新增代码时确认实际 key"在实际开发里很容易被忘——同一个 pit 隔几周在不同文件里被踩了两次。

### 教训

pybind 导出层可能对字段名做重命名。引用 C++ 导出数据时，应以 Python 侧实际拿到的 key 为准，而非 C++ 结构体字段名。这个坑**任何消费 MCTS stats 的代码都可能踩**（后端 pipeline.py、前端 pipeline.js、未来的日志/分析工具等），两次已经验证。

**防止再次复发的查表**——`stats` dict 的 Python 侧 key（见 `bindings/py_engine.cpp` 中 `get_ai_action` 的 `st[...]`）：

| Python key | C++ struct 字段 |
|-----------|----------------|
| `simulations` | `simulations_done` |
| `best_value` | `best_action_value` ← 重命名 |
| `root_values` | `root_values` |
| `action_values` | `root_edge_values`（展开成 `{action_id: [values]}` dict） ← 重命名 + 结构变化 |
| `tail_solved` | `tail_solved` |
| `tail_solve_value` | `tail_solve_value` |
| `traversal_stops` | `traversal_stops` |

**调试姿势**：新增读 stats 的代码之前先 `print(st.keys(), st['ai_stats'].keys())` 一次，或直接查上面这张表。别猜。

---

## [BUG-021] fly 动画继承源容器尺寸，导致 Azul 砖巨大化 + 顺序播放卡顿

**状态**：已修复
**文件**：`platform/static/general/animate.js`，`games/azul/web/azul.js`
**严重程度**：中 — 前端可用性问题，不影响训练 / AI 决策，但 Azul 对局手感非常糟糕

### 问题描述

Azul 前端每次出动作后，飞行的砖动画看起来"巨大"，而且拿的砖、落到 center 的剩余砖、first-player 令牌是**一个接一个**播的，整局下来每步等 1s+ 才能继续。

两个子 bug：

1. **尺寸 bug**：`animate.js::stepFly` 用 `fromRect.width/height` 作为飞行元素的 css 尺寸。Azul `describeTransition` 的 `from` 指向整个工厂 disc（`[data-factory="0"]`，~120×120px），结果飞的砖是 120×120 的巨型 blob，而棋盘上的 `.tile` 实际只有 34×34。

2. **顺序播放 bug**：`describeTransition` 给每个颜色去向推一个 fly step，`playTransition` 逐步 `await`，每步 350ms + 60ms 间隔。如果一次出牌有 1 个目标色 + 3 个去 center 的剩余色 + FP 令牌，总耗时 ~1700ms，直观感受是"一个一个飞"。

### 修复

**框架层** (`animate.js`):
- 新增 `flyGroup` step type：`flights[]` 数组里的所有子飞行并行 `Promise.all`，总耗时 = 最慢的一条
- `fly` step 支持可选 `width`/`height` 字段，显式指定时覆盖 `fromRect` 的继承。默认行为不变（继承 rect 尺寸），Splendor 等现有游戏不受影响
- 飞行定位从"左上角对齐"改为"中心对齐 fromRect / toRect"——小 flyer 在大容器里不再跑偏到角落

**Azul 前端** (`azul.js`):
- 把原来的多个 sequential fly step 合并为一个 `flyGroup`（选中色去目标 + 剩余色去 center + FP 令牌去地板，全部并行）
- 每个 flight 明确 `width: 34, height: 34`（= 棋盘 tile 实际尺寸）

### 教训

1. **飞行动画不应该默认继承源容器的 bounding rect**——当 `from` 是一个大容器（工厂 disc、对手 board 区）而飞行的"逻辑对象"是小单位（一块砖、一个 gem）时，尺寸会错得离谱。应该以飞行元素自身的 CSS 尺寸或调用方显式指定为准。
2. **多段连续 fly 默认顺序播放对桌游来说太慢**：一个真实物理动作（从一堆砖里拿同色、剩的推中间）在桌面上是一瞬间发生，不是慢动作分解。新游戏接入动画时，**凡是一个游戏动作物理上同时发生的多个位移，应该用 `flyGroup`，不是多个 `fly` 串联**。
3. 新游戏如果 `from` 选择器指向容器而非单体元素，写 `describeTransition` 时必须显式传 `width`/`height`，否则就会撞这个坑。新游戏验收时应该实际在浏览器里走几步看动画，不能只靠单元测试通过。
4. 框架层的默认行为（继承 rect 尺寸）保留是为了向后兼容 Splendor——Splendor 的 `from` 一般就是 token 或卡牌本身，大小合适。但该默认行为是"陷阱型"默认——以后可能考虑改成必须显式传尺寸，或者把默认行为改为"从 createElement 的元素自然尺寸推导"。

---

## [BUG-022] cancel_pipeline 误清 precompute，导致 expert 第一手 AI 响应多 8 秒

**状态**：已修复
**文件**：`platform/game_service/pipeline.py`
**严重程度**：中 — 前端性能问题，不影响训练和 AI 决策质量

### 问题描述

Expert 难度下，人类作为先手走第一步后，AI 要等 10 秒左右才落子。按配置（analysis=500 sims, expert=500 sims）理论耗时应该 < 100ms。

### 根因

`signal_cancel()` 在同一个函数里做了两件事：取消 pipeline 阶段机 + 清空 precompute 缓存。而 `apply_action` 路由处理人类动作时第一步调 `cancel_pipeline()`（为了确保没有在跑的 pipeline worker 还会写 session），间接触发了 `precompute_clear()`。

踩坑流程：
1. 游戏创建时人类先手 → `schedule_precompute(初始局面)` 立刻后台跑 MCTS，~30ms 完成，结果缓存在 `sess["precompute"]["result"]`
2. 人类点击落子 → `POST /action` → `apply_action` 首行 `cancel_pipeline(sess)`
3. `cancel_pipeline → signal_cancel → precompute_clear` —— 把刚刚缓存的 precompute 结果清成 `None`
4. Pipeline worker 进入 `_analyze_user_move`，while 循环等 `pc["result"] is not None`
5. 等不到 → 撑满 8 秒 deadline → 触发 fallback inline MCTS
6. 总耗时 = 8 秒等待 + fallback MCTS + AI decision ≈ 10 秒

AI 后续每一步就正常了（因为从 pipeline 结束开始，precompute 是在 AI 回合之后重新 schedule 的）。

### 修复

`signal_cancel` 里移除 `precompute_clear` 调用。precompute 是独立于 pipeline 的后台任务（在 `PRECOMPUTE_EXECUTOR`，与 `PIPELINE_EXECUTOR` 不共享状态），pipeline 取消不应该波及它。

真正需要清 precompute 的场景（undo / state 重建）在 `step_back` handler 里原本就显式调用 `precompute_clear`，不依赖 `cancel_pipeline` 的副作用，所以删除后不会回归 undo 逻辑。

### 教训

1. **副作用打包到"看起来该一起做"的函数里是坑**：`cancel_pipeline` 在语义上只负责 pipeline，但它悄悄顺带清 precompute，调用方不看实现就猜不到。修复后 `signal_cancel` 的 docstring 明写"does NOT touch precompute — callers that need it clear must call precompute_clear() explicitly"。
2. **这个 bug 的症状（"第一手慢 10 秒"）很容易被错误归因到"AI 计算慢"**。排查时发现 10 秒 ≈ 8 秒等待 + 两次实际 MCTS 时，8 秒这个具体数字暴露了是 deadline 超时而不是计算量爆炸。调试类似问题时，**对着"奇怪的整数时长"回头查代码里的等待上限**是一个有效快捷方式。
3. precompute 和 pipeline 应该是两个解耦的任务系统。代码里分两个 ThreadPoolExecutor 是对的，但 helper 函数混在一起就没起到隔离作用。以后如果要做类似并发架构，helper 函数的命名要严格反映其副作用边界。

---

## [BUG-024] GameSession MCTS 搜索在真实状态上跑，应隔离为 AI view

> ⚠️ **历史归档**:本条目记录的修复路径(GameSessionWrapper 在 `bundle_->state` 上跑 MCTS,通过 NoPeek + sample-from-history 兜底)已在 [DEC-003] 与后续 per-seat session 重构中被替代。当前架构是:selfplay / arena / web / API 都给每个 perspective 持有独立的 `per_seat_states[p]`,MCTS root 永远是 `per_seat_states[acting_player]`,truth state 物理上不被传给 search;`IBeliefTracker` 接口签名里没有 truth 指针。本条目里"NoPeek 是事后挡板"等叙述属于演进过程,不再描述当前行为——结构性隔离已经直接消除了"在真实状态上跑 MCTS"的可能性。详见 CLAUDE.md "AI Pipeline Independence from Game State" 与 ALGORITHM_OVERVIEW §1。

**状态**：已修复（架构重构）
**文件**：`bindings/py_engine.cpp` GameSessionWrapper
**严重程度**：严重（BUG-023 的根因）
**关联**：BUG-023 是这个根因在 Love Letter 上的具体表现

### 问题描述

发现 BUG-023 后用户追问："那之前训练的时候自博弈也有这个问题吗"、"本质上模拟游戏的地方和你跑 AI 的地方根本不在一起"。深挖发现：

- **AI API 路径**（`test_ai_api_separation` 覆盖的那条）和 **GameSession 路径**（web / selfplay 用的）**共享同一份 MCTS 代码**（`NetMcts::search_root`）和同一个 `GameSessionWrapper`
- 所谓"API 模式隔离"其实没有物理隔离 —— `bundle_->state` 在两种模式下都持有完整的隐藏字段。API 模式之所以"看起来没 bug"只是因为 seed 生成的 `bundle_->state` 里对手的隐藏字段是"任意合法占位"，对 AI 而言是无信号的随机值；而 GameSession 模式下那里装的是真相
- MCTS `search_root(*bundle_->state, ...)` 直接在真相状态上展开。rules.apply 读 `d.hand[opp]`、`d.reserved[opp][i]` 等隐藏字段时拿到的是真实值
- NoPeek 是事后挡板，依赖 nonce 变化判断"这条边需要随机化"。任何走不到 nonce 变化路径的动作（BUG-023 的 terminal-by-elimination、Splendor 的 BuyReserved 不抽新牌、等等）都是潜在漏点
- 每出现一个同类 bug 修一次是打地鼠，根本做法是 **MCTS 搜索用的状态必须和游戏真相是两个对象**

### 修复（最终架构：ISMCTS-A Method 1 + 2b）

经过迭代收敛到下面这套。关键认知来自用户反复质询："搜索能看见真相本身就是越界"、"对手不同隐藏牌合法集不一样你怎么办"、"对手采样进 UCB 的 prior 是第一次到达决定的吗"。

**第一层——ai_view 架构隔离**（GameSessionWrapper 内）：

```cpp
std::unique_ptr<IGameState> bundle_->state;                      // 真相
std::vector<std::unique_ptr<IGameState>> ai_views_;              // 每个视角一份 AI view
std::vector<std::unique_ptr<IBeliefTracker>> ai_trackers_;
std::vector<std::unique_ptr<IFeatureEncoder>> ai_encoders_;
std::vector<std::unique_ptr<OnnxPolicyValueEvaluator>> ai_evaluators_;
```

- 构造时：对每个 perspective p 从真相克隆后走 `initial_observation_extractor`/`applier` 把观察者不该看到的字段掩成占位。其他 per-perspective 对象从 fresh bundle 偷 unique_ptr（不改变堆地址，encoder 内部 raw ptr 照旧有效）
- `apply_action`：真相侧推进；同时对每个 perspective 经 `public_event_extractor(truth_before, action, truth_after, p)` 提取观察事件，按 `pre → action → post` 顺序在 ai_view 上回放

**第二层——ISMCTS-A 搜索**（`engine/search/net_mcts.cpp`）：

- 每次 simulation 起点：`sim_state = clone(ai_view); randomize_unseen(sim_state, per_sim_rng)`——从 belief 采一个具体世界
- 搜索过程中 `rules.apply(sim_state, action)` 严格按规则结算，读隐藏字段时读的是**这次 sim 采样的值**。**不跳过、不 chance-random-pick**
- 节点首次扩展时请求网络对全 action space 输出 prior，UCB 用 per-sim `legal_actions(sim_state)` 过滤；不同 world 的 legal 集不同是被这个过滤吸收的
- Prior floor `0.01/|A|`：防止"首次访问那个 world 下不合法"的动作 prior 被压到 0，保证后续 sim 里一旦合法了仍有机会被探索
- Paranoid `validate_action` 断言：过滤失败时崩溃而不是 silent apply

**第三层——Observer-hash 树共享**（`IGameState::state_hash_observer(perspective)`）：

新增虚方法，返回"观察者视角下 information set"的 hash（排除观察者看不见的隐藏字段）。MCTS 用它做树节点键 + `chance_children` 分流。效果：不同采样世界的对手节点共享一个 tree node，Q/N 跨世界聚合，避免树按采样世界分裂导致有效搜索深度稀释。Love Letter 和 Splendor 各自覆盖实现。

**三条路径统一到 ISMCTS-A**：
- `GameSessionWrapper::get_ai_action`（web / 实时）
- `run_selfplay_episode_py` → `selfplay_runner`（训练数据生成）
- `run_arena_match_py` → `arena_runner`（eval / model 对比）

都在 `NetMctsConfig` 里塞 `root_belief_tracker` + `root_observer_perspective` + `full_action_space`。旧的 NoPeek `traversal_limiter` 在 ISMCTS-A 路径下置 nullptr（不删除，给没 belief_tracker 的纯物理随机游戏留着）。

**API 模式兼容**（历史描述，当时的协议）：`external_obs_mode_` flag，`apply_initial_observation` / `apply_observation` / `apply_event` 会翻 true。外部调用方通过事件协议驱动 `bundle_->state`，此模式下 bundle_->state 就是 AI view，MCTS 走老路径（测试用）。当前协议已不再有 `apply_event` 入口，observer 路径只走 `apply_observation(action, events, public_snapshot)`

同时 **revert 了 BUG-023 的 loveletter nonce bump**。新架构让那个补丁变成无用代码——即使规则里的 `apply()` 读了 `d.hand[target]`，读的也是 `ai_views_[p]` 里的占位值而不是真相。BUG-023 的黑盒测试（Guard 命中率 ~14% 而不是 76%）依然通过。

### 连带好处

1. **彻底解决一整类 bug**：Love Letter Guard/Baron/King/Prince、Splendor BuyReserved（包括 `legal_actions` 读隐藏字段的泄漏）、以及任何未来"读隐藏字段但不消耗 RNG"的动作都被架构层拦住。开发者不再需要在游戏的 rules 里做 defensive 的 `++nonce`
2. **selfplay 训练一并修复**：selfplay 也走 `GameSessionWrapper`（via `run_selfplay_episode_py` → `GameSessionWrapper` 指导 search），同样受益。之前污染的 Love Letter 训练数据建议重训一次得到干净模型
3. **web UI 的 "AI 只看到该看到的" 成为架构保证**：物理层面 MCTS 拿不到 `bundle_->state` pointer
4. **未来 Coup 的 bluff-biased 采样直接插在 `init_ai_views_` 的 `randomize_unseen` 钩子上**，与框架一致，不需要专门路径

### 回归测试

**统计层**：`tests/loveletter/test_checklist.py::TestLoveLetterGuardAccuracy::test_guard_accuracy_not_better_than_bounded_inference`——跑 60 局 vs 随机对手，无先验信息时 Guard 命中率必须 < 40%。BUG-023 时是 76%，当前架构下是 ~20%，留 2 倍以上 headroom。

**不变性层**：`tests/framework/test_api_mcts_policy_invariance.py::test_api_mcts_policy_matches_selfplay`——直击"selfplay 猛如虎 / API 变弱"的信息泄漏病症。流程：
1. 用 seed_gt 跑 selfplay，记录观察历史 + 每个 ply 的 MCTS visit distribution（在 GameSession 真相驱动路径下计算）
2. 用 seed_api（不同）起 API 会话，replay 同一观察历史
3. 在 perspective 行动的 ply 上对比两条路径各自 `get_ai_action` 的 argmax 与完整 visit 分布:argmax 偏差率要求 ≤ 65%、平均 total-variation distance ≤ 0.40(两条路径用独立的 MCTS RNG,bit-exact 不可能;阈值留宽是为了在 RNG jitter 下不假阳,真正的泄漏会让其中一路系统性偏向"凑巧好"的动作、把数字推到远超阈值)

两测合起来：统计层抓"作弊具体症状"（Guard 命中率），不变性层抓"两条路径是否真的走同一个信息"。

### 性能成本

每个 session 多持有 N 个 `IGameState` 克隆 + N 个 tracker + N 个 encoder + N 个 evaluator（N = 玩家数，2–4）。ONNX 模型 N 倍显存（对当前 1-2MB 级别的小模型可接受）。每次 `apply_action` 多做 N 次事件提取+应用。Love Letter / Splendor 级别游戏未观察到明显性能影响。如果未来上大模型或者玩家数更多，可以考虑共享 evaluator 但 per-perspective 切换 encoder。

### 教训

1. **测试通过 ≠ 架构正确**：三层测试（API 契约 / belief 等价 / encoder 不泄漏）都是必要条件，不是充分条件。MCTS 实际搜索用的状态是不是隔离的，需要单独的黑盒测试覆盖（见 `TestLoveLetterGuardAccuracy`）。
2. **"现有 API 模式测试过了，让其他地方也用 API 模式的路径"不一定对**：本次第一版方案就是这么说的，但深挖发现 API 模式自己也没物理隔离，只是"对手字段是随机占位"的偶然掩盖。正确方案是直接在 `GameSessionWrapper` 层做物理隔离。
3. **当用户反复让检测某个怀疑方向时，比起反复跑既有测试、应该主动设计一个能直接测量该怀疑的新测试**。BUG-023 本来如果更早写黑盒命中率测试，就不需要用户催五次。
4. **架构层修复优先于补丁层修复**。BUG-023 单个修的话是 5 行 nonce bump，但一眼看上去不知道是不是还有同类 bug 漏。架构修完，一整类根本问题清零。

---

## [BUG-025] pipeline.py `nopeek_enabled` off-by-one：peek_steps=0 被错误解读为"第 0 步 peek"

**分类**：框架层
**状态**：已修复
**文件**：`training/pipeline.py:439`
**严重程度**：高 — peek 模式在 ISMCTS 下让 MCTS 在 truth 上搜索，破坏 DAG hash 的 info-set 语义；训练第一步直接崩

### 问题描述

peek_steps 字段语义：**前 N 步训练用 peek（MCTS 看真相），之后切回 ISMCTS**。默认 0 表示始终 ISMCTS。

pipeline.py 里原来写的：

```python
"nopeek_enabled": step > peek_steps,
```

当 `peek_steps=0, step=0`（训练第一步）时，`0 > 0 = False` → `nopeek_enabled=False` → peek 模式被**意外开启**。意思变成了"前 1 步用 peek"。

### 根因

off-by-one：`step > N` 表达的是 "step 不在前 N+1 个里面"，但 "前 N 步" 的正确判定是 `step < N`，取反（不在前 N 步 → ISMCTS 开启）就是 `step >= N`。

### 修复

```python
"nopeek_enabled": step >= peek_steps,
```

加了注释说明 off-by-one caveat。

### 教训

"前 N 步做某事" 这类表达翻译成代码时要显式写 `step < N`，不要用 `step > N-1` 之类的等价变形——容易和边界对不上。测试应该覆盖 `peek_steps=0`（完全不 peek）的 case：训练第 0 步 nopeek_enabled 必须为 True。

---

## [BUG-026] ISMCTS DAG hash collision → MCTS 选中非法 action 崩溃

**分类**：框架层
**状态**：已兜底修复（真正的 Hasher 强化是未来工作）
**文件**：`engine/search/net_mcts.cpp::search_root`
**严重程度**：高 — 启用 Dirichlet 噪声的隐藏信息游戏训练中概率触发（Love Letter 2p 训练第 5 步就挂）

### 问题描述

ISMCTS 用 DAG 节点共享：同一 `state_hash_for_perspective(current_player)` 的 sims 复用同一节点的 edges（UCB 统计在该节点聚合）。训练中偶尔报：

```
MCTS: selected action failed validate_action — node/state inconsistency
```

诊断 dump 显示：

```
Node edges=[46(Princess), 32(Baron-self)]
Current legal_actions=[45(Countess)]
```

两个 sim 都到达这个 DAG 节点，但 sim_state 的 (hand, drawn_card) 不同：
- 第一个 sim：手牌是 `(Princess, Baron)` 或 `(Baron, Princess)` 且全部 opp protected → legal 包含 Princess 和 Baron-self（fallback 自选）。Expansion 按这套 legal 建边
- 第二个 sim：手牌是 `(Countess, King)` → must_countess 强制 → legal 只剩 Countess

两套 state 在 `Hasher::combine`（XOR + shift）下**产生了相同 hash**。DAG 复用了第一个 sim 的 edges。UCB 选中 Princess → 在第二个 sim 的 state 上非法 → 崩。

### 根因

`Hasher::combine(uint64_t v)`：`seed_ ^= v + kGoldenRatio64 + (seed_ << 6) + (seed_ >> 2)`。64 位空间下理论上可碰撞；Love Letter 私有字段组合（hand × drawn_card = 约 64 种）和公共字段某些序列相乘，碰撞概率非零。Dirichlet 噪声让 sims 探索更多分支，放大了碰撞触发概率——关噪声 + greedy 选择时观察不到。

### 修复（兜底）

把硬 `throw` 改成**重新选边**：

```cpp
if (!rules.validate_action(*sim_state, chosen_action)) {
  auto legal = rules.legal_actions(*sim_state);
  std::unordered_set<ActionId> legal_set(legal.begin(), legal.end());
  // Re-pick best UCB edge restricted to currently-legal actions.
  int fallback = -1; float fb_score = -inf;
  for (each edge) if (legal_set.count(edge.action)) {
    float s = q + u; if (s > fb_score) fallback = i;
  }
  if (fallback < 0) {
    // Zero overlap — terminate this simulation at leaf with value estimate.
    break;
  }
  best_edge = fallback;
}
```

兜底**不影响 correctness**：碰撞时 stale edges 只是这一次 sim 被忽略，visit 统计和 value 更新都在真正合法的动作上。UCT2 的 DAG 共享即使在碰撞下仍然渐进正确。

### 真正的 root-cause fix（未做，未来 work）

强化 `Hasher` 抗碰撞：
- 选项 A：`combine` 用更强的混合（如 xxhash 或 MurmurHash3 的 finalize 步骤）
- 选项 B：`state_hash_for_perspective` 在 finalize 后做一轮额外 avalanche
- 选项 C：用 128-bit hash 拼接，节点 key 用 pair

当前兜底的 cost：每次碰撞多一次 legal_actions 调用（O(legal) 的小成本），碰撞率估计 < 0.01%，实战可忽略。

### 教训

- **DAG + hash 共享节点的搜索**永远要在 edge 传播前 validate，不能假设"hash 相同 → state 行为相同"。这是 ISMCTS 设计的 invariant **要**然成立但**不能**假设 100% 无碰撞
- 隐藏信息游戏 + Dirichlet 噪声是最容易暴露 hash collision 的组合，因为噪声扩大了搜索分支、不同 sim 走到同 info-set 的机会增多
- 诊断要早一点写——第一次挂的时候信息几乎为零（`node/state inconsistency`），加 dump 之后立刻看明白是 edge set vs legal set 不匹配

### 剩余未做

- `tests/framework/test_dag_hash_collision.py`：构造已知碰撞的状态对（如果能找到）断言 fallback 路径触发且 MCTS 正常完成
- Hasher 升级到 128-bit 或更强混合

---

## [BUG-027] Quoridor 手机端棋盘 UI 连环坑 —— button UA baseline 偏移 + 固定像素尺寸在小 slot 下退化

**分类**：Web 前端
**状态**：已修复
**文件**：`games/quoridor/web/quoridor.js`、`games/quoridor/web/styles.v2.css`、`platform/app.py`
**严重程度**：中 — 手机访问者感知 legal 高亮偏离格子中心、放墙 pill 变成圆点，影响可玩性

### 问题描述

手机浏览器（以及安卓 Chromium webview 的"请求桌面版"模式）访问 Quoridor，出现两个叠加的视觉 bug：

1. legal 走子高亮的小圆点整体比格子几何中心**下移几个像素**，视觉上像"贴着 cell 底边"
2. legal 放墙 pill 在桌面上是**长条形**，手机上退化成**正方形/圆点**

桌面浏览器两个都正常。

### 调试弯路

这个 bug 吃了很多调试时间，走了好几条死路，值得完整记录：

**弯路 1：`floor()` + 重声明 fallback 误以为会降级，结果把整个棋盘压成一条线**

最早怀疑是 CSS Grid 的 subpixel 漂移（`--slot = (100vw - X) / 17` 在手机上是 19.7px 之类的分数，grid track 被 floor 到 19，但 `var(--slot) * N` 仍然按 19.7 算，累积偏移）。写了：

```css
--slot: min(34px, calc((100vh - 120px) / 17), calc(...));
--slot: floor(min(34px, calc((100vh - 120px) / 17), calc(...)));
```

以为"浏览器不认 `floor()` 就 fallback 到前一行"。**但 CSS 自定义属性的 fallback 不是这么工作的** —— 自定义属性值里可以包含任何 token，解析时不验证函数名，`--slot: floor(...)` 总是作为合法声明覆盖前一行。某些手机浏览器支持 `floor()` 语法但在 `floor(min(...))` 这种嵌套下算出 0，于是 `repeat(17, 0)` 让棋盘直接变成一条线。

**教训**：自定义属性 (`--x: ...`) 不做值验证，标准的"声明两次，浏览器挑能解析的那条"fallback 技巧对它**无效**。

**弯路 2：把 grid tracks 换成 `1fr` + `aspect-ratio: 1/1`**

第二轮假设：既然 `var(--slot)` 被独立 floor 和 `calc(var(--slot) * N)` 对不齐，干脆让 grid template 用 `1fr` 平分总宽，总宽用 `calc(var(--slot) * 17 + 28px)` 锁定，`aspect-ratio: 1/1` 保证正方形。

```css
grid-template-columns: repeat(17, 1fr);
grid-template-rows: repeat(17, 1fr);
width: min(100%, calc(var(--slot) * 17 + 28px));
aspect-ratio: 1 / 1;
```

这在桌面上看着 OK，但手机浏览器上 `aspect-ratio` 的支持在某些 webview 里有 bug —— 结果**只渲染了 8 行的高度，剩下 9 行溢出到棋盘外**，对手棋子从白色棋盘里面"掉"到页面背景下面。

**教训**：`aspect-ratio` 虽然 W3C 标准支持广，但在手机浏览器（尤其中国系 webview）的实现有坑。任何依赖它的布局必须在实机上测。

**弯路 3：`--slot` 预留宽度算错，棋盘压到 info-panel 上**

回滚到简单的 `repeat(17, var(--slot))` + `width: max-content` 之后，发现桌面模式下（viewport ~980px）棋盘右侧压住了 info-panel。原因：`--slot` 公式假设 `info-col = 280px`，但 `layout.css` 里其实已经是 `340px`，少减了 60px。**这条是**真实 bug，改成：

```css
/* 1200px 以下：sidebar 280 + info-col 340 + stage padding 32 + gap 16
 * + grid padding 28 = 696 */
--slot: min(34px, calc((100vh - 120px) / 17), calc((100vw - 696px) / 17));
```

并在每个 breakpoint 里写出显式加法，下次 layout 改动时数值对不上一眼可见。

**弯路 4：JS 测量 `--slot` 写回整数像素**

即使布局宽度算对了，手机上 legal 圆点还是偏。加了 `alignSlotToRenderedTrack(boardEl)` 在 render 后测量实际 cell 宽度、floor 回整数覆写 `--slot`。这个 fix 本身**原理正确**（`var(--slot) * N` 与 grid tracks 严格对齐），但对这个具体 bug **没效果** —— 因为 bug 的根因不是 subpixel 漂移。保留下来作为未来防御。

**弯路 5：把 legal 圆点从 `::after + position: absolute; top: 50%` 改成 flex item，再改成 `radial-gradient` background**

以为是 `<button>.board-cell` 的 flex 居中或 absolute 定位受 button UA line-height 影响。先试 flex 居中（`::after { display: block; flex: 0 0 auto }`），没用。又试 `background-image: radial-gradient(circle, ...) center center`，按理说 background 是按 padding box 画的，不经过任何 child box，该居中。**还是偏**。

**根因：`<button>` 元素本身的 UA baseline 基线偏移**

这时才意识到：即使 padding、border、margin、line-height、font、appearance 全部 reset 成 0 / inherit / none，安卓某些 Chromium webview 给 `<button>` 烘焙了一个 baseline 调整，让 button 的**内在高度略大于指定值或内部有隐含的上 padding**，把它整个推下去几个像素。

无论你把**什么**东西放在 button 里或者画在 button background 上，都会被这个偏移一起带下去。background-position: center 没问题，问题是"center 相对于被挤扁的 padding box"本身就已经不在格子几何中心了。

### 修复

把所有 `.board-cell` 和 `.edge-slot` 从 `<button>` 换成 `<div role="button" tabindex="-1">`：

```js
// 之前
const cell = document.createElement('button');
cell.type = 'button';
cell.className = 'board-cell';

// 之后
const cell = document.createElement('div');
cell.setAttribute('role', 'button');
cell.setAttribute('tabindex', '-1');
cell.className = 'board-cell';
```

`<div>` 没有 UA baseline 偏移，所有子元素（pawn、legal 圆点）落在真正的几何中心。点击事件和无障碍语义靠 `role="button"` 保留。

### 修复 2：wall-pill 固定像素高度 → 改成 slot 百分比

button → div 那一版修完后，用户发现 legal 放墙 pill 在桌面上是长条形、手机上变成了圆点。CSS 当时是：

```css
.edge-slot.edge-h::after {
  width: calc(100% - 8px);   /* 宽度按 slot 比例变 */
  height: 8px;               /* 但高度是固定 8px */
}
```

- 桌面 slot = 34px：pill 是 26×8，长宽比 ~3.2，看着是长 pill
- 手机 slot = 17~19px：pill 是 9~11×8，长宽比 ~1.1，看着像正方形/圆点

**固定像素尺寸在"设计时只照顾到一个 slot 大小"的场景下必然退化**。桌面那版看着对是因为 slot 大；手机上 slot 小了一倍，固定 8px 相对比例就翻倍，pill 宽高比塌了。

改成两个轴都用 slot 百分比：

```css
.edge-slot.edge-h::after {
  width: 85%;
  height: 30%;
}
.edge-slot.edge-v::after {
  width: 30%;
  height: 85%;
}
```

现在桌面和手机 pill 比例一致，都是长条。已放置的 wall-anchor 继承 30% 高度，手机上稍薄但还是清晰的长条形墙体。

### 顺带修的：**Cache busting**

调试期间发现服务器端代码已经更新但手机上还显示旧版，即使手动清理缓存、换浏览器都没用 —— 手机 ISP / 系统 webview 层的缓存比桌面粗暴得多，即便服务器返回 `Cache-Control: no-cache, no-store` 也会被忽略。

在 `platform/app.py` 里加了**自动 cache-busting**：serve `/games/<game>/` 时动态改写 index.html，把里面 `href="styles.css"` 和 `src="quoridor.js"` 改成 `href="styles.css?v=<mtime>"`，每次文件被改 mtime 就变，URL 就变，任何缓存层都找不到旧条目必然 miss：

```python
_ASSET_REF_RE = re.compile(
    r'(?P<attr>href|src)="(?P<url>[^"]+\.(?:css|js))"'
)

def _rewrite_asset_refs(html: str, web_dir: Path) -> str:
    def repl(m):
        url = m.group("url")
        if "?" in url or "://" in url or url.startswith("/"):
            return m.group(0)
        asset = web_dir / url
        if not asset.exists():
            return m.group(0)
        mtime = int(asset.stat().st_mtime)
        return f'{m.group("attr")}="{url}?v={mtime}"'
    return _ASSET_REF_RE.sub(repl, html)
```

注意注册自定义 handler **必须在** `StaticFiles` mount 之前，否则 mount 会 shadow 掉根路径的 GET。

### 教训

- **`<button>` 在 Web UI 组件中不是自由替换 `<div>` 的选择**。在对几何对齐要求苛刻的场景（棋盘格子、色块、坐标点），UA 给 button 烘焙的 baseline 偏移是 CSS 重置不掉的。默认用 `<div role="button">`，除非需要 native 表单语义。
- **CSS 自定义属性的 fallback 需要 `@supports`**，不能靠"声明两行"。`--x: floor(...)` 不会因为浏览器不认 `floor()` 而被跳过。
- **`aspect-ratio` 在某些手机 webview 里有 bug**，需要实机验证，不能假设"W3C 标准 + caniuse 全绿就能用"。
- **混合"百分比 + 固定像素"的尺寸公式在小 slot 下会退化**。`width: calc(100% - 8px); height: 8px` 在 34px slot 是长 pill，换到 17px slot 变成正方形。响应式 UI 里凡是要保持"视觉比例"的尺寸，两个轴都用百分比，不要一轴百分比一轴固定像素。
- **中国系手机浏览器的缓存比 `Cache-Control` 更顽固**，走纯 header 方案不可靠，必须 URL 层 cache-busting（文件名或 query）才能突破。
- **调试弯路值得完整记录**。这个 bug 改了 8 版 CSS、2 版 JS，其中 5 版都是错的方向。没有 devlog，下次遇到类似症状又要走一遍。

---

## [BUG-029] tail-solve 采纳 ProvenWin 路径未填 `root_values`，专家模式终局窗口 45s 后才显示

**分类**：框架层（search / web pipeline）
**状态**：已修复
**文件**：`engine/search/net_mcts.cpp`、`platform/static/general/pipeline.js`、`tests/framework/test_tail_solve_root_values.py`
**严重程度**：中（用户可观察）— 体感像彻底卡死

### 症状

Azul 专家模式（`difficulty=expert`，启用分析路径）打到终局，最后一回合的结算/终局动画都不渲染、终局弹窗不出现，浏览器看起来直接“冻死”。等 45 秒后弹一个 timeout。
体验版（`difficulty=casual`，无分析）能正常终局；情书 / Quoridor 等其他启用 tail-solve 的游戏没出现，因为它们的对局结构很少在“人类该走那一步”的预计算位置触发 ProvenWin 采纳。

### Root cause

`NetMcts::search_root` 中，当 `tail_solver` 返回 `kProvenWin && value >= 1.0` 时走的是“早返回”分支，把 `tail_solved=true` 等字段写好就返回 best_action。但这条路径**没有填 `stats->root_values` 也没有填 `stats->root_edge_values`**，二者保持空 vector。

下游 `pipeline.py::_human_wr_from_stats` 对 `root_values[human_player]` 做下标访问，空 vector 直接 `IndexError`。
`_pipeline_worker` 的 `except Exception` 捕获后把 `phase` 置为 `"error"`。前端 `pipeline.js` 的 poll 循环只识别 `done` / `idle` 退出，`error` 不退、继续轮询直到 45s 整体 timeout —— 看起来就是“卡住”。

体验版用 `_pipeline_worker_ai_only`，根本不调 `_analyze_user_move`，所以打不到这条空 vector 解码 → 体验版正常终局。问题是“专家模式才有的分析路径 × tail-solve 在人类下一步的预计算位置触发 × `root_values` 空”这三者凑齐才会暴露。

### 修复

1. `engine/search/net_mcts.cpp` ProvenWin 早返回前：填 `root_values` 为 `(actor=+1, others 均分 -1)` 的零和向量；填 `root_edge_values`，被采纳 action 的位置写入同样向量、其余 `0`。语义上 ProvenWin 的胜率本就是 100% / 0%，这只是把它显式表达出来，下游解码路径全部可用。
2. `platform/static/general/pipeline.js`：把 `phase === "error"` 当成终止条件直接退出 poll 循环并触发 `onError`，避免今后任何 worker 异常都让浏览器静默等满 45s。
3. `tests/framework/test_tail_solve_root_values.py`：回归测试。从 Azul 2p 局推到 tail-solve 采纳，断言 `root_values` 长度等于 num_players、零和、actor 位置等于 +1，且 `action_values` 含选中 action 且向量与 `root_values` 一致。

### 教训

- “提前返回 + 部分填字段”的优化路径必须把所有下游字段都填到“跟正常路径一样合法”的状态，否则就是另一个 BUG-011 那种 silent contract violation。Stats struct 不是只给 logger 用的，平台层 / web 层都会拿来当导航数据。
- Web pipeline 的 `phase=error` 必须有显式分支处理。任何能把 worker 推进 `except` 的 bug 都会被 45s timeout 覆盖成“假冒卡死”，让 root cause 极难被注意到。
- 难度模式（casual / expert）走不同 worker 是个值得记住的差异：casual 不跑 analysis，所以分析路径独有的 bug 不会出现在 casual 复现里——遇到“专家模式才挂”的报告，先 diff `_pipeline_worker` vs `_pipeline_worker_ai_only`。

---

## [BUG-031] Web 隔离 AI 会话没有继承 `tail_solve` 配置（web AI 实际未启用 tail-solve）

**分类**：平台层（Web / 隔离 GameSession）
**状态**：已修（2026-05-07）
**文件**：`platform/game_service/sessions.py`、`platform/game_service/pipeline.py`、`platform/game_service/routes.py`
**严重程度**：高 — silent degradation，用户和开发者都误判 AI 强度

### 问题描述

`web.json` 里的 `tail_solve` 配置（depth_limit / node_budget / time_limit_ms / margin_weight）只在创建主 `GameSession` 时被 `configure_tail_solve()` 调用。但 web AI 落子、precompute、hint fallback、analysis 这些路径都会**新建隔离 GameSession**（为了 hash scope 隔离 + 避免污染主 session 的 belief tracker）。新建的隔离会话默认 `ts_enabled_=false`，**`web.json` 的 tail_solve 配置永远不生效**。

### 影响

- 用户和开发者都误以为 web AI = expert（latest + tail-solve），**实际只是 latest**
- 所有 "web 看起来强不强"、"expert 是不是真的比 latest 强一档"、"端局收官准不准" 的主观判断全都被这条静默偏置污染
- Eval / arena 结果用 web 配置作 sanity check 时口径错位
- 属于 BUG-011（ONNX 未编译 → MCTS 走 uniform 静默几周）那一族

### 症状

Web AI 落子 stats 中 `tail_solve_attempted=0`，但 web.json 里明明配了 tail_solve depth/budget。当时没暴露这个 stat 到响应所以肉眼看不出，打开后端日志能看到 ts 走的是 disabled 路径。

### 修复

1. 把 tail-solve 配置（depth, node_budget, time_limit_ms, margin_weight）存进 `sess` 对象
2. 所有用于 AI move / precompute / analysis / hint fallback 的隔离 `GameSession` 创建后立刻 `configure_tail_solve()`。涉及 `sessions.py` / `pipeline.py` / `routes.py` 三处
3. `bindings/py_engine.cpp::get_ai_action()` stats 暴露 `tail_solve_attempted / completed / elapsed_ms`，web 响应透传，"是否真的跑了 tail-solve" 肉眼可验证
4. 加测试：`tests/web/...` 用 Quoridor 末段 8 步内必胜的局面创建 web session，断言 `r["stats"]["tail_solve_attempted"] >= 1`

### 教训

**用户配置和实际生效之间永远要有 stat 暴露**。这是框架级原则：任何 "配置 → 实际行为" 的中间环节都必须有一个可观测的 stat（通过响应 / 日志 / metric）让运维 / 开发者能一眼验证配置生效了。隔离 GameSession 的创建是看不见的"配置重置"节点，必须显式复制配置。原则写入 CLAUDE.md 或 GAME_DEVELOPMENT_GUIDE.md 的 "隔离 session" 章节。

---

## [BUG-032] Selfplay per-perspective trackers 未被初始化 → randomize_unseen 覆写当前玩家自己的手牌 (OB-011)

> **后续**：E13 (2026-05-13) 把 `initial_observation_extractor` / `applier` 这一对 per-game hook 整体删掉，改成框架层 walker。E14 (2026-05-13) 又把这一步拆成与 per-ply 同构的两段 wire：`viz::serialize_public(state, schema)` 走全 all_public 字段（与每 ply snapshot 共用同一个 walker），加上 `tracker.pack_init_payload(state, perspective)` 输出 perspective-private bootstrap（LL 是 `own_hand` + 起始玩家的 `drawn_card`，Splendor 是空）。`IBeliefTracker::init` 签名变成 `(IGameState&, int, AnyMap)`，让 tracker 直接把 perspective-private 写回 state 并 toggle viz=1。下文里所有提到 `bundle.initial_observation_extractor` / `trace_obs_extractor` / `serialize_public_for_perspective` 的代码片段都是当时的现场，今天已经不存在；assertion 那一道仍然在 LL tracker 入口保留作为契约守门。

### 背景

- BG-008 Phase 2 stage 5 引入 per-perspective trackers（每 seat 一份 tracker，观测事件累积到 game over），替代"每 ply re-init 单例 tracker"的旧路径
- selfplay_runner 里已经循环 `per_perspective_trackers[p]->init(p, init_obs)` 并在每一步给每个 tracker 喂 `observe_public_event` —— 看起来是完备的
- MCTS 搜索 acting player 的决策时：`mcts_tracker = per_perspective_trackers[current_player]`，调 `randomize_unseen` 采样一个 belief 相容的隐藏世界
- 但启用这条路径后 `test_selfplay_sample_integrity[loveletter]` 立刻炸 `DAG node legal-action mismatch`，错误里 `node_edges=[Handmaid, Princess]` 而 `current_legal=[Guard+target1, Prince+self/target1]`

### 根因

`bindings/py_engine.cpp::run_selfplay_episode_py` 的初始化顺序：

```cpp
runtime::PublicEventExtractor trace_extractor;         // 空 std::function
runtime::InitialObservationExtractor trace_obs_extractor;  // 空
if (trace_perspective >= 0) {  // 默认 -1，通常走不进
  ...
  trace_extractor = bundle.public_event_extractor;
  trace_obs_extractor = bundle.initial_observation_extractor;
}
...
run_selfplay_episode(..., trace_extractor, trace_obs_extractor);
```

`trace_extractor / trace_obs_extractor` 这两个名字误导——它们**不只是 trace 专用**。`run_selfplay_episode` 里把它们当成整个 episode 的 `public_event_extractor` 和 `initial_observation_extractor` 用：pp_trackers 的 `init()` 在游戏开始调用时，如果 `initial_observation_extractor` 为空，**直接跳过 init**（`if (per_perspective_trackers[p] && initial_observation_extractor)` 短路），tracker 的 `perspective_player_` 永久停留在默认 `-1`。

然后 MCTS 调 `per_perspective_trackers[0]->randomize_unseen(sim, rng)`：LoveLetter 的 randomize 代码里 "跳过 perspective 自己的 hand" 用的是 `if (p == perspective_player_) continue`。`perspective_player_ == -1` 意味着**没有任何 p 等于它** —— 包括真正的 current player——所以 `d.hand[0]` 被从 unseen 池里重新采样成另一张牌，`d.drawn_card` 同样被 `d.current_player != perspective_player_` 判 true 而覆写。

结果：real root expand 得到的 edges=[Handmaid, Princess]（基于真 hand），sim_state 在 randomize_unseen 后 hand=Guard drawn=Prince（被覆写），legal_actions 完全不一致，DAG 检查炸。

### 症状关键词

- `DAG node legal-action mismatch; selected action X is not legal in current state`
- `depth=0 step_count=0`（在 root 就炸，因为 randomize_unseen 在 root 前第一次调用）
- node_edges 和 current_legal 两组动作**对应完全不同的 (hand, drawn) 组合**
- 只在启用 per_perspective trackers 的 routing 后出现；legacy 单 tracker 路径 `belief_tracker->init(player, obs)` 在 selfplay_runner 里每 ply 显式调所以看不到

### 为什么 legacy 路径不炸

Legacy `belief_tracker` 路径在 selfplay_runner 的 `else if (belief_tracker)` 分支里显式调 `belief_tracker->init(player, main_init_obs)` —— 这里也用到 `initial_observation_extractor`，但它 re-init 时即使 obs 为空 map，`init()` 内部实现（以 LoveLetter 为例）是 `it_h != end() ? cast : 0` —— own_hand_ 被置 0，perspective_player_ **正确**设为 player。所以 legacy 单 tracker 是安的。

问题特有于"不在 selfplay_runner 自己 re-init、而是依赖 py_engine 之前把 extractor 传进来"的 pp_trackers 路径。

### 修复

`bindings/py_engine.cpp::run_selfplay_episode_py`：无条件从 bundle 填 extractor，不再让 `trace_perspective >= 0` 这个 tracing flag 门禁它：

```cpp
runtime::PublicEventExtractor trace_extractor = bundle.public_event_extractor;
runtime::InitialObservationExtractor trace_obs_extractor =
    bundle.initial_observation_extractor;
std::unique_ptr<GameBundle> trace_bundle;
IBeliefTracker* trace_bt = nullptr;
if (trace_perspective >= 0) {
  trace_bundle = ...;
  trace_bt = trace_bundle->belief_tracker.get();
  if (!trace_bt || !trace_extractor) throw ...;
}
```

并在 `LoveLetterBeliefTracker::randomize_unseen` 入口加一道硬 assertion：

```cpp
if (perspective_player_ < 0 || perspective_player_ >= NPlayers) {
  throw std::runtime_error(
      "LoveLetterBeliefTracker::randomize_unseen called with uninitialized "
      "perspective_player_=" + std::to_string(perspective_player_) +
      " (init() must run before search)");
}
```

这条 assertion 是**框架级契约**：任何 tracker 在被 search 调用前必须已 init。下次有人写新游戏的 belief_tracker 或重构 runner 时，如果 init 被漏掉，这里会炸而不是静默污染。

### 教训

1. **变量名带有"trace"字样但被用于通用路径，是最容易踩的坑**。`trace_extractor / trace_obs_extractor` 原本只给 trace 用，后来 BG-008 stage 5 复用它们承载 pp_trackers 的 extractor 需求，名字没改。下次同类扩展要么重命名（`public_event_extractor / initial_observation_extractor`），要么在 bundle 上单独引一对字段给 pp_trackers 用。

2. **默认值 `-1` 的 sentinel + "跳过 perspective" 的语义是个静默泄漏陷阱**。LL 的 randomize 代码用 `if (p == perspective_player_) continue` 跳过自己——perspective_player_=-1 时这个条件对任何 p 都 false，所有 p 都被覆写。改成 assertion 前，这是一个**"不崩溃、不报错、只是悄悄把当前玩家的手牌换掉"**的 bug，在 MCTS 没 DAG 检查的话永远抓不到。属于 BUG-028 / BUG-030 族的"silent correctness 漏洞"——只要有 silent 路径能让不变式被绕过，就一定要补 assertion。

3. **用 DAG 校验撞出逻辑 bug 是这套框架的强项，要珍惜**。`DAG node legal-action mismatch` 本来是为"hash scope 漏字段"设计的错误信号，这次它抓到的是"tracker 没 init"这种相邻问题——因为 tracker 错了会让 sim_state 和 real state 出现公共观测不一致，表现出来就是 hash 撞上了但 legal 对不齐。下次遇到 DAG mismatch，除了查 hash 字段覆盖，还要查"tracker 是不是也出了别的错让状态被污染"。

4. **修复不只是打补丁，还要防守**。这次修 py_engine 的 extractor wiring 是主菜，但在 LL 的 randomize_unseen 入口加"perspective 必须 init 过"的 assertion 是主菜之外的保险——未来任何新的 runner / binding 如果漏 init，这里会崩，不会再"silently 降级再让 DAG 检查去撞"。Coup 的 belief tracker 同样应该加一道类似 assertion（参见 DC 2026-05-07 笔记，建个 SM 任务跟进）。

---

## [BUG-034] 体验版（casual）AI 走子后没显示对手胜率——前端 difficulty gate 把已经算好的数字扔了 (OB-013)

### 背景

- Web 前端有"显示胜率"开关（默认开），AI 走完一步后 info panel 应该显示这一步对人类胜率的估值
- 用户反馈"体验版（casual）AI 走完没胜率"
- 第一反应是"casual 不跑 analysis pipeline，所以没胜率数据"——错

### 根因

数据其实一直到了前端：
1. `_pipeline_worker_ai_only`（casual 路径）→ `_commit_ai_move` → `pipe["ai_stats"] = ai_stats`（pipeline.py:267）。AI 自己 MCTS 跑完产生的 `root_values` 已经塞进 pipeline status
2. `/pipeline` endpoint 透出 `ai_stats`（routes.py:163）
3. poller `pipeline.js` 从 `st.ai_stats.root_values[humanPlayer]` 算出 `humanWinrate` 传给 `onDone`
4. **但** `app.js:445` 把赋值 `state.lastAiWinrate = result.humanWinrate` 关在 `if (state.difficulty === 'expert')` 里。casual 拿到 humanWinrate 立刻扔。

`sidebar.getShowWinrate()` 用户开关是另一条独立路径（在 `setWinrate` 调用处守门）；这次的 difficulty gate 是冗余的、且直接掐死了 casual 的合理用例。

### 修复

`platform/static/general/app.js`：删 difficulty gate，casual / expert 都把 humanWinrate 赋给 state。注释说明：casual 跳过的是 analysis pipeline（drop-score、smart hint），不是 AI 自己的 MCTS——AI 自己的 root_values 一直都在。

### 教训

1. **debug 跨服务流时先 trace 端到端数据流，不要靠直觉猜哪一段没产数据**。本来想在 backend casual 路径加 root_values 上报，结果 backend 早就上报了，问题在 frontend 一行 difficulty gate。直觉走的是"casual = 没数据"，实际走的是"casual = 数据被前端最后一步丢了"。下次类似 bug：先用 devtools 看 `/pipeline` 响应里有没有数据，再决定改哪一端。

2. **同一个用户开关不要在多个地方守门**。`getShowWinrate()` 在 `setWinrate` 调用处已经守了"用户不要看胜率"，`difficulty === 'expert'` 的 gate 是另一个角色（"casual 难度不该有胜率"），但这两个角色有重叠——如果用户在 casual 也想看，开关明明开着却看不见，UX 矛盾。原则：**用户开关的语义边界要单一**，每个开关只在唯一一处守门，避免"开关开了但被另一处守门掐掉"的隐性失效。

---

# 游戏层 Issues

以下所有 issue 都位于 `games/<name>/` 下——开发新游戏时**应着重参考本节**，里面的模式（Splendor 不偷看、Love Letter 揭牌事件、Azul 动画等）是游戏开发者常见的踩坑点。

## [BUG-017] SplendorBeliefTracker 偷看牌堆内容

**分类**：游戏层（Splendor）— 开发新游戏时请参考此案例避免 tracker 读 state 隐藏字段
**状态**：已修复
**文件**：`games/splendor/splendor_net_adapter.cpp`
**严重程度**：高 — AI 精确知道牌堆组成，等于作弊

### 问题描述

`SplendorBeliefTracker::randomize_unseen` 直接读取 `data.decks` 来构建 unseen pool——等于 AI 知道牌堆里有哪些牌。belief tracker 本应是"玩家的记忆"，只通过 `init` 和 `observe_action` 积累信息来推导 unseen pool。

```cpp
// 修复前（偷看）：
for (auto cid : d.decks[tier]) {
    unseen_pool.push_back(cid);  // 直接从真实牌堆读
}

// 修复后（正确）：
for (int cid = 0; cid < 90; ++cid) {
    if (seen_cards_.find(cid) == seen_cards_.end() && card_pool[cid].tier == tier) {
        unseen_pool.push_back(cid);  // 从 全卡池-seen 推导
    }
}
```

### 根因分析

初始实现把 `randomize_unseen` 当成"shuffle 已知内容"，但正确语义是"基于观察推导可能的内容并采样"。两者在单机自我对弈中看似等价（state 对自己可见），但在对外 API 场景（真实隐藏状态在别人服务器上）下完全不可行。

### 修复方案

1. `SplendorBeliefTracker` 增加 `seen_cards_: std::unordered_set<int>` 和 `initialized_: bool` 成员
2. `init(state, player)` 首次调用时扫描所有公开位置（tableau + visible reserved + 自己的 reserved）建立 seen set
3. `observe_action(before, action, after)` 增量追踪新揭示的卡：
   - BuyFaceup/ReserveFaceup → 比较 state_after 的 tableau 与 state_before，新出现的 card_id 加入 seen
   - ReserveDeck 且 actor == perspective_player → 新预留的暗牌 card_id 加入 seen
4. `randomize_unseen(state, rng)` 用 `全卡池(90) - seen_cards_` 按 tier 分组构建 unseen pool，shuffle 后回填

### 验证

新增 5 个测试（`TestSplendorBeliefTracker`）验证：
- 随机化后的牌堆组成与真实牌堆不同（证明不偷看）
- tableau 卡不出现在随机化牌堆中
- 牌堆大小不变
- 多次随机化产生不同结果
- 无重复卡牌

### 教训

**belief tracker 是玩家的记忆，不是上帝视角**。`randomize_unseen` 的正确语义是"根据我所知推测未知"，不是"重排我已知的真相"。测试验证方式：`全卡池 - seen` 与 `真实 deck 内容` 在有牌被购买后必然不同——如果每次都相同，说明在偷看。

---

## [BUG-023] Love Letter AI 永远猜对 Guard — terminal-by-elimination 漏过 NoPeek 检测

**分类**：游戏层（Love Letter）— 只在旧的 NoPeek 架构下成立，**ISMCTS 重构后不再可能**（root 采样不依赖 rng_nonce 触发）。保留作为"游戏规则中隐藏-信息-依赖动作要触发随机化"的历史教训
**状态**：已修复（后被 ISMCTS 整体架构替代）
**文件**：`games/loveletter/loveletter_rules.cpp`
**严重程度**：严重 — AI 对隐藏信息游戏直接读取真实状态决策，相当于作弊

### 问题描述

用户实测发现 Love Letter AI 打 Guard 时命中率异常高（后经量化：无 Priest/Baron 先验的情况下命中率 76%，随机基线 14.3%）。既有两层 AI API 分离测试（`test_ai_api_separation` 和 `test_api_belief_matches_selfplay`）全部通过，说明 API 契约层面没有泄漏；feature encoder 直检也确认不经由特征通道泄漏对手手牌。

### 根因

问题在 **NoPeek traversal limiter 的激活条件**。框架用"rng_nonce 是否改变"判定 stochastic 转移（`default_stochastic_detector`），只有跨越 stochastic 边界时才 `randomize_unseen` + 重新应用动作。

但 Love Letter 的 Guard 正确猜中→对手淘汰→`advance_turn` 开头 `check_end_game` 发现 2p 终局→直接 return，**不抽牌**→`draw_nonce` 不变→stochastic_detector 返回 false→NoPeek 不触发→MCTS 看到的 child 是用**真实手牌**算出来的终局 win 结果（Q=1.0）。

对比猜错分支：对手不死→`advance_turn` 抽牌→nonce 变→NoPeek 正常触发→随机化后重算 Guard→有一定概率命中→Q ≈ 1/7。

结果：MCTS 把"猜中对应的 guess"这条分支估得 Q=1.0，其他 guess 都是 ~0.14。AI 每次都精确选中真实手牌那一个 guess。本质上 MCTS 偷看了一次真实状态来做局部决策。

同类问题存在于 Baron（比大小直接淘汰到终局）、Prince（牌堆空了 draw 不到会不抽，此时无 nonce 变化）、King（交换手牌无 draw）。Priest 不受影响——其效果是 `hand_exposed[target]=1`，并不依赖隐藏信息做分支。

### 修复

`loveletter_rules.cpp::do_action_fast` 在 switch 之后、`advance_turn` 之前，对 Guard/Baron/Prince/King 无条件 `++d.draw_nonce`。这样任何读取过隐藏手牌的动作都会产生 nonce 变化，NoPeek 正常触发并随机化对手手牌。

修复前后实测（2p，80 盘，AI=player0，对手随机）：

| 指标 | 修复前 | 修复后 | 基线 |
|------|--------|--------|------|
| Guard 无先验命中率 | 76.0% | 14.0% | 14.3% |
| tracker 有先验时的命中率 | ~100% | ~100% | — |

修复后命中率严格匹配 1/7 基线；有 Priest/Baron 先验时仍然 100% 命中（合法使用公开信息）。

### 教训

1. **nonce-based stochastic detector 的语义是"存在 RNG 消耗"，而我们真正需要的语义是"转移结果依赖观察者不知道的信息"**。这两者大多数时候等价（draw 是最常见的隐藏消耗），但在 terminal-by-elimination 或 deck-empty 的边界上不等价。此类游戏中所有"读隐藏手牌"的动作都需要显式 nonce 增量。
2. **两层 API 分离测试不能抓这类 bug**。现有测试覆盖"API 契约是否携带隐藏字段"和"observation-only belief 是否等价于 selfplay belief"，但不覆盖 "MCTS 在 GameSession 路径下实际搜索的世界是否真的用了 belief 而不是 true state"。修复后应补一个"AI 决策不应显著优于无先验基线"的统计测试。
3. **用户主观"AI 太强"的反馈在隐藏信息游戏里永远是硬信号**，要优先怀疑泄漏，不要先辩护"也许是合理推断"。本次排查先假设是合理的 Priest 先验推断（对 AI 有利的解释），险些漏掉 bug；直到跑量化检查才暴露。
4. **新游戏接入 Checklist 里应加一条**：对隐藏信息读取型动作（Guard/Baron/King 类），确认 nonce 会在应用时变化。在 `docs/guide/NEW_GAME_TEST_GUIDE.md` 里加一个对应测试模板。
5. 现有 Coup 暂停开发的理由（诈唬核心游戏 uniform-sampling ISMCTS 不适用）是更抽象的 bias 问题；本 bug 是具体实现问题。两者都属于 ISMCTS 采样逻辑的潜在坑，开发隐藏信息游戏时都要盯。

---

## [BUG-028] public hash 混入不可观察随机源，导致 ISMCTS DAG 按隐藏信息分裂

> ⚠️ **历史归档**:本条目记录的修复路径(MVP-B / BG-008 Phase 2)已在 [DEC-003] 中被进一步收紧。当前架构以 DEC-003 + 当前代码为准——session 不再每步调 `randomize_unseen`、`apply_observation` 末尾不再有 truth-override patch event,viz=0 槽位永远不被框架 freshen。本条目里"BG-008 把 randomize_unseen 搬到每个 apply_observation 末尾"等叙述属于演进过程,不再描述当前行为。

**分类**：游戏层（Azul / Splendor）— 开发新游戏时必须参考此案例审计 `hash_public_fields()`
**状态**：已修复
**文件**：`games/azul/azul_state.cpp`、`games/splendor/splendor_state.cpp`
**严重程度**：高 — 破坏 information-set DAG 共享，可能让搜索行为和 encoder scope 不一致

### 问题描述

ISMCTS 的节点 key 使用 `state_hash_for_perspective(current_player)`，语义应是：

```
public fields + current player's private fields + step_count
```

这里的 `public fields` 必须是玩家从观察历史中能知道的公共信息，不能包含“谁都不知道”的随机源内部状态。

本次发现两个具体问题：

1. **Azul**：`hash_public_fields()` 把 `bag` 和 `box_lid` 的 vector 顺序逐个 hash 进去。玩家最多能知道袋子 / 盒盖中各颜色剩余数量，不能知道未来抽牌顺序。
2. **Splendor**：`hash_public_fields()` 把 `rng_salt` hash 进去。`rng_salt` 是内部 RNG 盐，不属于任何玩家的公共观察。

### 影响

- 同一玩家视角下完全相同的信息集，会因为隐藏随机源不同而产生不同 hash。
- ISMCTS DAG 节点被按不可观察信息分裂，等价于减少复用、削弱搜索深度和统计聚合。
- 更严重时，hash scope 与 encoder scope 不一致：网络看不到的信息却影响 MCTS 节点 key，调试表现会很反直觉。
- 这类问题在 API / Web 隔离架构下尤其危险，因为 ground truth 的随机源内部状态不应泄漏到 AI 决策 pipeline。

### 修复

**Azul**：

`bag` / `box_lid` 不再按 vector 顺序 hash，改为按 5 色计数 hash：

```cpp
std::array<int, kColors> bag_counts{};
for (std::int8_t t : bag) {
  if (t >= 0 && t < kColors) {
    ++bag_counts[static_cast<std::size_t>(t)];
  }
}
for (int count : bag_counts) h.add(count);
```

这样保留公开可推导的 composition，移除不可观察的 draw order。

**Splendor**：从 `SplendorState::hash_public_fields()` 移除 `h.add(rng_salt)`；`decks[t].size()` 留在 public hash（公共可推导）。deck 内容的 RNG 漂移由 BG-008（见下文）彻底收掉。

**Coup**（2026-05-07，第三轮 BUG-028 同族 bug）：

强化版 `test_public_hash_excludes_internal_rng.py`（60-seed sweep）暴露了 Coup 的三处遗漏 event。`court_deck` 是 Coup 的 face-down deck，类比 Azul `bag` / Splendor `decks[t]`：内容每个人都不知道，所以 `hash_public_fields()` 只能 hash `court_deck.size()`；`exchange_drawn[]` 是 active player 临时手牌，hash 进 `hash_private_fields(active_player)`。当 perspective 从 court_deck 抽牌、或对手公开归还卡时，AI session 的 sampled world 必须被显式同步到 truth，否则 hash 在“看不见但已观察的轴”上分裂：

1. **`self_exchange_draw`**（post）：perspective 是 Ambassador 交换的 active player，从 `kExchangeReturn1` 之外进入 `kExchangeReturn1` 时触发。perspective 看见自己抽到的 2 张牌，但 AI session 在 sampled court_deck 上抽出了不同的两张。Applier 把 AI sampled 的两张推回 court_deck，删掉 truth 对应的两张（保持 size 不变），把 `exchange_drawn` 钉到 truth。
2. **`self_influence_redraw`**（post）：`kRevealSlot0/1` 在挑战成功（`card == claimed_character`）分支中，`coup_rules.cpp` line 431 / 509 给 revealer 抽一张新的 influence card；当 revealer == perspective 时，新卡进入 `hash_private_fields(perspective)`，AI sampled 不一致。Applier 同 self_exchange_draw 套路：sampled 的旧 influence 推回 court_deck，删掉 truth 对应的一张，把 `influence[][]` 钉到 truth。
3. **`public_return_card`**（post）：每个 `kReturn*` action 在 truth 中都把 1 张牌推回 court_deck。载荷含 `expected_deck_size` + `exchange_drawn: [int, int]`。Applier 强制 `court_deck.size()` 等于 `expected_deck_size`，并把 `exchange_drawn` 钉到 truth。

**BG-008（2026-05-07，架构层收尾）**：

上面三轮都是在"per-event 修补漂移"的思路里打补丁，每遇到一条新的撞车路径就加一个 event。根治的架构是：`py_engine::apply_observation` 末尾**统一调 `tracker.randomize_unseen(state_, rng)`**，让 session state_ 的隐藏字段每 ply 都被重新采样成一个与观测史一致的新世界；`randomize_unseen` 的契约被强化为“产出世界的 `hash_public_fields` 在同一观测史下 byte-equal”。

这条路径要求每个游戏的 `do_action_fast` **不能让自己的公开输出依赖被 `randomize_unseen` 重采样的字段**。对 LoveLetter 来说，`check_end_game` 在牌堆空时会读 `d.hand[all alive players]` 比大小判胜负——对手手牌是 session 采样值，污染 winner；修复是 extractor 在 terminal 翻转时 emit `round_end` post-event 带 truth winner，applier 覆盖。Splendor / Coup / Azul 的 `do_action_fast` 公开输出都不读重采样字段（Splendor 对手暗 reserve、Coup 对手 influence/exchange_drawn、Azul bag 顺序都是 hidden-only），所以不需要额外 event。

副作用：Coup 的 `exchange_drawn{}` 默认值是 `{0, 0}`（`std::array<int8_t, 2>` 的 zero-init），而 `randomize_unseen` 用 `>= 0` 判断 "slot 有真卡"——这是 Coup 从 day-1 就存在的潜在 bug，只不过 `randomize_unseen` 原先只在 MCTS clone 上跑，从没写回 session 持久 state。BG-008 把 `randomize_unseen` 搬到每个 `apply_observation` 末尾后，这个 bug 立刻把 court_deck 偷走 2 张牌。修复点：`coup_state.h` 成员默认 `{-1, -1}`、`coup_state.cpp::reset_with_seed` 显式赋 `{-1, -1}`、`coup_rules.cpp::advance_turn` 的 `= {}` 改成 `= {-1, -1}`。

BG-008 MVP-B 落地之后：`IBeliefTracker::reconcile_state` 虚函数删除；Splendor 的 `reconcile_state` 实现删除；Coup 的 3 个 truth-sync event 继续存在（它们 pin 的是 `hash_private_fields` 里的 perspective 私有字段，不是 public hash 漂移的补丁，仍然必要）。

**BG-008 Phase 2（2026-05-07 晚，message-driven public state）**：MVP-B 仍然保留 `do_action_fast(session state_)` 这一跑步，要求开发者手动在 `public_event_extractor` 里为每个 "public 输出读 hidden" 的路径写 truth-override 补丁 event（LL round_end、Coup public_return_card expected_deck_size / exchange_drawn）。Phase 2 把这整条路径替换成：每个隐藏信息游戏注册 `public_state_applier`，`public_event_extractor` 在 `PublicEventTrace.public_snapshot` 里 dump post-action public state 全量。`apply_observation` 末尾统一调 applier 把 session state_ 的 public 字段从 truth 覆盖，`do_action_fast` 的公共输出是什么不再重要。效果：

1. **LL `round_end` post-event 删除** —— snapshot 里的 `winner`/`terminal` 把 `check_end_game` 读 hidden 产生的错误 winner 覆盖掉
2. **Coup `public_return_card` post-event 删除** —— snapshot 里的 `court_deck_size` 覆盖尺寸漂移；新增 `exchange_drawn_mask` 覆盖 shape（opp 私有 char id 不泄漏，perspective 自己的走 `self_exchange_draw`）
3. **新游戏入职心智负担下降** —— 不再需要人肉识别 "do_action_fast 里哪个公共输出读了 hidden"，snapshot 是 public 的 single source of truth
4. **回归保护新增**：`tests/framework/test_public_snapshot_round_trip.py` 逐 ply 检查 `truth → extract snapshot → blank observer → apply snapshot → hash_public byte-equal truth`，钉住 applier + extractor + hash_public_fields 三件套的 field-level 一致性
5. **OB-005 数据层修复**：selfplay_runner 改成持有 N 个 per-perspective tracker，每个 tracker init 一次后只 observe_public_event 增量更新。MCTS root 暂时仍走 legacy 单 tracker 路径（per-perspective tracker 的 narrower belief 暴露了 LL hash scope 的 latent bug，独立 audit 后再合流）  
   ⚠️ **历史归档**:此句的"MCTS root 暂时仍走 legacy 单 tracker 路径"已不再成立。当前 selfplay_runner 在 MCTS 前已把当前行动玩家的 per-perspective tracker 交给 MCTS(`mcts_cfg.root_belief_tracker = per_perspective_trackers[player]`),legacy 单 tracker 路径已删除。

完整设计见 `docs/plans/MESSAGE_DRIVEN_AI_REFACTOR.md`。

回归保护：`tests/framework/test_public_hash_excludes_internal_rng.py`（60-seed × 4 hidden-info 游戏）+ `tests/framework/test_session_hidden_fields_resampled.py`（新增，断言 session state_ 的隐藏字段在每 ply 末被重新采样），连续 5 次稳定通过。

> ⚠️ **历史归档**:`test_session_hidden_fields_resampled.py` 已在 [DEC-003] 中删除——session 不再每步 freshen hidden,断言"每 ply 末重新采样"已不是当前契约。当前的隐藏字段不变性由 `test_public_hash_excludes_internal_rng`(viz=0 槽位的真实值不进 hash)守护。

### 教训

1. **`hash_public_fields()` 不是“把 state 里的公共成员变量都 hash 进去”**，而是 hash “观察历史能唯一确定的公共信息”。无人知道的随机源不属于 public。
2. **随机源内部状态和随机结果要分开**：已经公开揭示的牌 / 砖 / 骰子结果可以 hash；未来抽牌顺序、RNG salt、deck shuffle order 不能 hash，除非它已经被观察到。
3. **hash scope 必须和 encoder scope 对齐**：如果 encoder 只看 bag counts，hash 也只能包含 bag counts；如果 hash 包含 encoder 看不到的信息，DAG 会按网络无法区分的状态分裂。
4. **发现 DAG 复用异常、搜索“很快但棋力弱”、同一观察历史 selfplay/API 策略分布不一致时，应优先审计 public/private hash scope**。
5. 对每个新游戏应加测试：改变不可观察随机源顺序或 RNG salt，在 public + own private 不变时，`state_hash_for_perspective(p)` 必须不变；同时改变公开可推导的 composition 时 hash 必须变化。
6. **架构上的预防**：`tests/framework/test_public_hash_excludes_internal_rng.py` 已经从单 seed 强化为 60 seed sweep。BUG-028 同族 bug 的触发本质是 Bernoulli——`randomize_unseen` 把某张未见牌分配到 deck 还是 opp face-down 的概率每次都不同，单 seed 配对在有 bug 的版本上经常恰好 hash 相等而通过。Sweep 把漏检率压到 `p^N`：~2% per-episode 的真实 drift rate × 60 seeds → 可靠失败（P[全过] ≈ 0.3）。任何新游戏注册时这个测试都会自动覆盖；为新隐藏信息游戏加 game_id 进 `HIDDEN_INFO_GAMES` 列表是注册流程的一部分。

---

## [BUG-030] Love Letter encoder 把 tracker 知识泄漏到非 perspective 玩家视角

**分类**：游戏层（Love Letter）
**状态**：已修（2026-05-07）
**文件**：`games/loveletter/loveletter_net_adapter.cpp`
**严重程度**：高 — 信息泄漏导致 MCTS 搜索偏置，无崩溃、不易观测

### 问题描述

MCTS 深入搜索时按当前节点的 `current_player` 视角编码特征。`LoveLetterFeatureEncoder::encode_private()` 在 encoder 使用 tracker 时只判断了 `tracker_ != nullptr`，没有检查 "当前编码的 `player` 是否等于 tracker 绑定的 perspective"。

当根玩家 P 用 Priest 偷看了对手 P1 的手牌后，tracker 记录 `known_hand_[1] = X`。搜索走到 P2 决策节点时，encoder 会对 pid=0,1,2 所有玩家依次调 `tracker_->known_hand(pid)`——即使 P2 视角下 P2 并不知道 P1 的手牌。这条知识被注入 P2 的 features → P2 的决策价值估计偏向 "假设 P2 也知道 P1 是 X" 的分布。

### 根因

Encoder 注释写明 "只有 player == tracker.perspective 才能用 tracker 知识"，但实现缺这条 guard。跨 perspective 的 tracker-derived 信息泄漏是 ISMCTS 框架里最隐性的 bug 类之一：没有 crash，没有 assertion failure，只在训练收敛曲线上体现为 "Love Letter 强度上限低于应有水平"。

### 修复

1. 给 `IBeliefTracker` 增加 `virtual int perspective_player() const` 默认返回 -1；每个游戏的 tracker 实现 override 返回自己绑定的 perspective
2. `LoveLetterFeatureEncoder::encode_private()` 增加 `if (player != tracker_->perspective_player()) skip tracker block` 的 guard
3. `tests/framework/test_encoder_respects_hash_scope.py` 扩展 LL 场景：根玩家 Priest 看到 P1 手牌后，从 P2 视角 encode_private 不应包含 P1 信息

### 教训

- encoder 在调 tracker 时**必须**按当前编码的 perspective 过滤。这不是 "优化"，而是正确性前提
- 类似 pattern 应当在 framework 层抽出通用的 `EncoderContext::tracker_knowledge_if_perspective_matches(pid)` helper，让所有游戏的 encoder 结构化地不可能犯这个 bug。未来新游戏接入时直接走这个 helper
- 隐性泄漏通过 "同一观察历史但不同 perspective 的 encoder 特征应当一致" 类型的测试断言捕捉

---

## [BUG-033] Azul 轮末结算飞砖落地后砖消失 + 多行同结算时 +score 偏高 (OB-012 / OB-008)

### 背景

- Azul `roundEndSteps`（games/azul/web/azul.js）负责轮末从 pattern lines 飞砖到 wall 的动画 + 给每砖弹 `+N` 加分 popup
- Prev render 把 wall 上没填的格子标 `.ghost`（半透 + 透明度 0.2），fly 落地后由 next-state re-render 翻成 `.filled`
- 用户两条独立报告："飞砖落地一瞬间砖消失/再 popup +score / 再 re-render 又出现"（OB-012）；"轮末分数显示和实际加分对不上、整轮一齐 pop 看不出每砖加多少"（OB-008）

### 根因（同源 / 一处代码两个症状）

1. **OB-012 — 中间状态没维护**：settle fly 只挂了 `onStart` 把源 pattern row 改成空，但**目的 wall cell 没挂 `onComplete`**。落地一瞬间 sprite 被 anim engine 移除，cell 仍是 `.ghost` 透明态，到 re-render 才翻 `.filled`——中间这段就是"砖消失"。属于 `WEB_DESIGN_PRINCIPLES §3 中间状态维护`：动画期间 DOM 必须反映动作的物理过程，不是 prev 也不是 next。源态改了空，目的态没改成已落位。

2. **OB-008 — scoring 用了"全结算后"的 wall**：`computePlacementScore(wall, row, col)` 算邻居连通时直接传 `next.players[pi].wall`——也就是这一轮**所有**砖结算之后的整面墙。Azul 真规则按 row=0..4 顺序逐行结算、每行只看截至此刻已落位的砖。如果同玩家 row 1 和 row 3 都满，row 1 结算时 row 3 还在 pattern line，不应作为邻居；但 `next.wall` 把 row 3 也算进去，导致 row 1 的连通邻居被高估，少数情况会把"+1"显示成"+3"。

### 修复

`games/azul/web/azul.js`：

1. 加 `cloneWallMatrix(wall)` 工具，5×5 deep copy
2. 重写 `roundEndSteps`：placements 按 (pi asc, row asc) 串行结算（settledPlacements 已按这个顺序产出）。维护 `incrementalWalls[pi]`（从 `prev.players[pi].wall` 拷贝起步），每砖一个串行子序列：
   - 先用 incrementalWalls[pi] 算 `score`（wall 处于本砖落位之**前**的状态）
   - 再 `incrementalWalls[pi][row][col] = 1`
   - 推一个 `fly` 步带 `onComplete`：取 `[data-wall-cell="<pi>-<row>-<col>"]`，`classList.remove('ghost'); add('filled'); style.opacity = ''`
   - 紧跟一个 `popup` 步显示 `+score`
   - 不是最后一砖加一段 `SETTLE_INTER_TILE_PAUSE`
3. 地板罚分 popup 推到所有 settle fly 之后作为 `group` 并行——不和飞砖竞争视觉

### 教训

1. **fly 是双端都要维护中间状态的步骤**。源端用 `onStart` 把容器改成空态、目的端用 `onComplete` 把容器从 ghost 改成 filled——两端都不能省。同时维护源和目的是 fly 的"完整中间态"，少一端都会有"砖消失/出现"的视觉断层。这条扩进 `WEB_DESIGN_PRINCIPLES §3` 的"中间状态维护"——明确"对每个 fly，源和目的都要有显式的中间态切换，hideFrom/onStart 处理源、onComplete 处理目的；ghost→filled 的切换不能只指望 next-state re-render"。

2. **真值结算顺序在前端动画里也要尊重**。前端不重写 C++ 规则，但**展示**结算时如果不按真规则的顺序展开，每砖的"加多少分"就会算错。同回合多行结算的"逐行 + 每行只看已结算邻居"语义在前端必须等价复刻——`incrementalWall` 就是这个等价复刻的载体。如果未来发现累计 popup 和 C++ 最终分对不上，应该考虑暴露 C++ 逐砖增量结算接口（CLAUDE.md "Never write Python fallback or reimplementation of any game logic"，前端的等效约束是同样的）；当前的 incrementalWall 是 best-effort 等价，不是规则真值。

3. **MAX_QUEUE_MS 的副作用**：4 玩家 Azul 单轮可能 20+ 砖串行结算，每砖 ~700ms，再加上 pause 和 popup 容易超过 5s。`platform/static/general/animate.js` 的 `MAX_QUEUE_MS` 从 5000 提到 30000，并加注释说明"不是为容忍长动画，而是为容忍合法长度的串行结算 + 兜底跑飞 describeTransition"。

---

## [BUG-035] Azul 轮末地板扣分用错时间点的 floor_count——actor 在结算回合扔的砖没算进去

### 背景

- Azul `roundEndSteps` 给每个有地板砖的玩家弹一个 `-N` 罚分 popup
- 用户反馈："这一把我那一轮是最后一个动的，我把一块砖放到了地板，最后这块的扣分没算"
- 走查代码：popup 用 `prev.players[pi].floor_count` 算

### 根因

`prev` 是**结算回合前**的 state 快照。actor 触发轮末结算的那一动作几乎肯定**自己**给地板加了砖：
- 显式 floor 投放（`targetLine === 5`）
- pattern row 装不下的 overflow 流到地板
- 从 center pool 取砖时附带的 first-player token

这些都属于"本回合落到地板"，C++ 引擎在 `do_action_fast` 里照常计入 floor_count 然后才 settle。但前端 popup 用的是 `prev.floor_count`——actor 的"本回合刚扔的"全没算。

特别极端情况：用户报告的就是 prev floor_count = 0、actor 这一动作直接把唯一一块砖扔地板。`floorPenaltyTotal(0) = 0`，循环里直接 `continue`——**popup 完全不出现**。罚分本身在 C++ 里照算了（next state 分数对得上），只是动画里看不到——属"显示静默漏算"。

### 修复

`games/azul/web/azul.js::roundEndSteps`：增加 `actionContext` 参数（`{actor, source, color, targetLine, isCenter}`，由 describeTransition 传入）。新建 `floorCounts[]`：

```js
floorCounts[pi] = prev.players[pi].floor_count;  // 其他玩家不变
// actor 的修正：
const tilesTaken = isCenter ? prev.center[color] : prev.factories[source][color];
let added;
if (targetLine === 5) added = tilesTaken;                       // 全部进地板
else added = max(0, tilesTaken - (capacity - prevLineLen));     // 只算 overflow
if (isCenter && prev.first_player_token_in_center) added += 1;  // FP token
floorCounts[actor] += added;
```

popup 循环改用 `floorCounts[pi]`。逻辑等价于"在 prev 上模拟 do_action 的 floor 增量、不模拟 settle 清空"，因为 settle 清空已经在 next 里发生了，C++ 那边记的罚分也是基于"清空前"的 floor_count。

### 教训

1. **`prev` 和 `next` 都不是"结算时刻"的 state**。Azul 这种"一个动作 + 自动 settle"的复合 transition 里，前端能看到的两个时间点：prev = 动作前、next = 动作 + settle 后。**结算用的 floor_count 是动作之后、settle 之前**——前端拿不到这个中间快照，必须自己模拟 actor 这一动作的 floor 增量。这是动画前端的"中间态计算"通则：不是所有需要的中间快照都能从 prev/next 直接读，部分必须靠**等价复刻动作的局部效果**得到。

2. **popup count = 0 时静默 skip 是个 bug 放大器**。如果罚分循环里 `if (floorCount <= 0) continue` 没了——退化成"用错的 count 算出 0 罚分但仍然弹一个 0"——用户会立刻看到"罚分 0"觉得不对然后报 bug。但 skip 让症状变成"什么都没显示"，看起来像"这位玩家这轮没扔过地板"，肉眼难辨。**在视觉反馈层"无显示"和"显示 0/null"是不同的失败模式**——前者掩盖 bug，后者暴露 bug。WEB_DESIGN_PRINCIPLES 可考虑加一条："计算结果 0/空 时仍然渲染（占位/灰显），不要 skip 整个 UI 元素"，避免"该有显示的地方什么都没有"被误读成正常状态。

3. **AI 价值真值与显示口径要对齐验证**。这次 bug 不影响游戏分数（C++ 引擎照算）但影响玩家对"我刚才那一动到底亏了多少"的认知。Azul 这种快节奏游戏，玩家对每动的得失有实时反馈预期，"显示和真实分数对不上"是教学信号衰减。后续在 `tests/web/...` 应该加一条"动画 popup 数字之和 = next.scores - prev.scores"的不变量校验（每帧），跨游戏适用。

---

## [BUG-036] LoveLetter `known_hand_[]` / `hand_override` 平行通道把 perspective 私人推断塞进 tracker 与 wire（§G.1 清理）

### 背景

- 落地于 2026-05-12，详见 `docs/plans/LL_LANDING.md`。
- LoveLetter `LoveLetterBeliefTracker` 历史上持 `perspective_player_` + `known_hand_[Cfg::kPlayers]`：Priest 偷看 / Baron 比较 / King 交换后把 opp 的 cid 塞进 `known_hand_[opp]`，encoder 通过 `tracker_->known_hand(perspective)` 读出来拼进特征。
- 同一对私人字段在 wire 上还有第二条通道：`loveletter_register.cpp` 的 `extract_events` 给每个 perspective 各自塞 `hand_override` / `drawn_override` 平行 event，receiver `apply_public_event` 反向写回 session state。

### 根因

CLAUDE.md 的「AI Pipeline Independence」第二条（session 公开字段被 message 重建覆盖）要求所有 perspective-private 知识活在 `state.viz` 上、由 rules 通过 `viz::reveal_slot_to` 维护，而不是绕过 viz 走 tracker 私字段或 wire 平行通道——后者会导致：

1. 同一 perspective 的"知道 opp 手牌"事实在两份地方记账（tracker `known_hand_` 与 viz reveal），更新时机一旦不同步就会让 encoder 与 hash 漂移。
2. tracker 持 `perspective_player_` 等于把 perspective 烙进 belief 内容，破坏 §B 「tracker perspective-agnostic」的不变式——两个 session 喂同一观察流应得 byte-equal belief，烙了 perspective 就做不到。
3. wire 上多一条 `hand_override` 通道，受 `test_snapshot_keys_match_schema` 之类 lint 拦截不到，是潜在的"wire 多塞了真值"漏洞面。

### 修复

- LL rules 在 Priest peek / King swap / Baron compare / 抽牌 / 弃牌 / 淘汰 全部改用 `viz::reveal_slot_to(observer)` / `viz::swap_slot_owned(a, b)` / `viz::reset_to_base(slot)`，rules 是唯一的 viz writer（I1 lint 由 `test_rules_sole_viz_writer[loveletter]` 守护）。
- `LoveLetterBeliefTracker` 退化为公开聚合 + uniform `randomize_unseen`，不再持 `perspective_player_` / `known_hand_[]`。
- `loveletter_net_adapter` encoder 改读 `MaskedState` 的 `hand[p]` 槽位——viz=1 的 opp hand 槽本来就有真值，placeholder 走零编码。
- `loveletter_register.cpp` 删 `hand_override` / `drawn_override` 平行通道，`public_state_applier` 走 `viz::apply_public` walker；私人 reveal 字段（owner-visible `hand[p]`、actor-visible `drawn_card`）走新的 partial-reveal sidecar `owner_overlay`（仿 Splendor `reserved_faceup_ids_flat`），receiver 写回真值并翻 viz bit。
- `bindings/py_engine.cpp` 把 `loveletter` 加进 `per_seat_in_scope`（per-seat session 路径），`test_selfplay_no_truth_in_ai_path` 矩阵把 LL 加进 PUBLIC_KEYS。
- `engine/core/belief_tracker.h` class doc 更新："perspective-baked private fields → state.viz" 现在只剩 Coup（待 §G.2）。

### 教训

1. **wire 上每多一条平行通道都要在 schema lint 里登记**。`hand_override` 走的是 `PublicEvent` 而不是 `public_snapshot["..."]`，所以 `test_snapshot_keys_match_schema` 对它无感——同样的盲区出现在 Coup 的 `hand_override`、未来任何"私人 patch event"。后续若再加私人事件通道，必须同步加 lint。
2. **partial-reveal sidecar 是 per-perspective 私人字段进入 walker 路径的标准做法**。基线 schema 不能表达"对 receiver 这个 perspective viz=1，其它 perspective viz=0"的字段（schema 是单一基线 + rules 动态写 viz），wire 必须额外 ship 一条 perspective-aware 的 sidecar 让 receiver 写回真值并翻 viz——否则 walker 的 `serialize_public` 永远拿不到 owner-visible 真值，receiver 就只能用 belief sample 替代，DAG 会因 hash 不一致漂掉。这条经验同时复用于 Splendor `reserved_faceup_ids_flat` 和 LL `owner_overlay`，未来的私人 reveal 字段都走这个 pattern。
3. **encoder 不再读 tracker 私字段是 §G 的硬指标**。任何"tracker 上有 perspective-private 数据"都是 §G 没收尾的信号——§G.2 Coup 收尾时同样适用。

---

## [BUG-037] LoveLetter `hand` 槽 hash 只 mix value 不 mix idx，dynamic-reveal 视野下两套 (idx,value) 序列哈希撞车 → DAG node legal-action mismatch

### 背景

- 训练运行 `runs/loveletter_4p_20260512_113903` 在 step 125 selfplay 抛 `MCTS: DAG node legal-action mismatch`：node 缓存 `node_edges=[37,38,39,40,...]`（Prince 系列）而当前 sim 世界的 `current_legal=[45,...]`（must-Countess）。两个看上去毫无相似度的合法集落进同一个 DAG node。
- 单种子复现：`BASE_SEED + 129 = 21510452`（step 125 的 episode 129）一次跑就 byte-equal 重现，`hash_pub=17913659907566068133`。
- 用户对 5000 局规模做随机 rollout 验证："不可能就是 hash 64-bit 量级的随机碰撞" — 实测 0 collision，事实站在用户这边。
- 在 `engine/search/net_mcts.cpp` 加 `DINOBOARD_DAG_DEBUG=1` 网关下的 insertion-time vs throw-time MaskedState slot dump，重跑同一个种子，拿到两条 byte-equal 的 dump：
  - INSERTION（perspective=1）：`hand[1]=5 | hand[3]=7 | drawn_card=1`
  - CURRENT（perspective=1）：`hand[0]=5 | hand[1]=7 | drawn_card=1`
  - 两者除 `hand` 槽外所有公开字段、ply、step_count 完全一致；`hand` 的"被访问到的索引集"和值不同，但 `state_hash_for_perspective(1)` 落在同一个 64-bit 桶。

### 根因

LoveLetter `hash_field_slot` 对 `hand` 槽只 mix 了 value，没 mix `idx[0]`：

```cpp
// games/loveletter/loveletter_state.cpp（修复前）
if (name == "hand") {
  h.add(d.hand[static_cast<size_t>(idx[0])] + 19); return;
}
```

`hand` 在 schema 里是 `owner_only_first_axis`，配合规则的动态 reveal（Priest peek、Baron showdown、King swap、`hand_exposed` 翻牌、淘汰后 `reset_to_base`），**不同世界 perspective=p 看到的 `hand[*]` 索引集会不同**：上面例子里世界 A 的 perspective=1 看到 `hand[1], hand[3]`，世界 B 看到 `hand[0], hand[1]`。两者都向 hash 喂一对 `(value+19)`：A 喂 `5+19, 7+19`，B 喂 `5+19, 7+19`——除了第二个值的位置互换之外字面相同；只要顺序和值相等，hash 就相等。

walker 在每个 viz=1 槽位调用 `hash_field_slot(name, idx, ...)` 时把索引送进来，正是为了让 `(name, idx, value)` 一起决定哈希。LL 的实现丢了 `idx`，等价于把"哪个玩家的手牌是 5"这个事实降级成"某玩家手牌是 5"——一旦 perspective 看到的 owner 集合在不同 sim 世界里漂移，碰撞就成必然。这跟"hash 不可观察 RNG"和 BUG-028 同源：可观察事实在 hash 里**完整**编码才能保住 DAG 节点身份。

为什么之前没爆：早期 LL 跑通的回合数较少，sim 深度浅；step 125 的 selfplay 进入 ply=9（4 人都活着 + Priest peek 已经发生 + 一次淘汰把 hand[2] 撤掉）的局面才触发。规模一上来 sim 进入此类深节点时碰撞概率就足以稳定吃到。

### 修复

`games/loveletter/loveletter_state.cpp` `hash_field_slot` 的 `hand` 分支额外 mix `idx[0]`：

```cpp
if (name == "hand") {
  h.add(static_cast<int>(idx[0]) + 71);
  h.add(d.hand[static_cast<size_t>(idx[0])] + 19);
  return;
}
```

200-episode 复现器全过；`tests/loveletter` 全过；`tests/framework`（含 `test_public_hash_excludes_internal_rng[loveletter*]` 和 `test_public_snapshot_round_trip[loveletter*]`、`test_dag_acyclic`）全过。

### 教训

1. **walker-driven hash 必须 mix `(name, idx, value)` 全三元组**。schema + viz 决定 visit 序列，但 visit 序列不是常数：动态 reveal（Priest / Baron / King / `hand_exposed` / 淘汰）让 perspective 在不同 sim 世界里看到不同的 owner 子集。任何把 idx 丢掉、只 mix value 的实现都隐含"被访问的 idx 集合是世界无关的"假设——一旦 dynamic reveal 进来这个假设就废了。规约：所有 `hash_field_slot` 实现里凡接受非空 `idx` 的字段，至少 mix 一次 `idx[k]`，无论 base viz 当前看起来"是不是固定的"。
2. **DAG 碰撞复现路径要走规则 + 网络的 byte-equal 重放**。普通 random rollout 的 5000 局测不出，因为 prior 不会真的把 sim 推进 ply=9 的稀疏节点。带 model + selfplay schedule 的 single-seed 重放把 prior 收敛固定，就能稳定吃到。"训练崩了能不能复现"的标准回答：保留模型 + episode_seed，单种子单线程跑 `run_selfplay_episode` byte-equal 重现 — 这次成立，下次也应保留这条路径。
3. **insertion-time vs throw-time MaskedState dump 是 DAG 撞车的高信噪比工具**。比起在 hash 内部加 trace（每个槽都打），存一份 node→insertion dump、撞了再 print insertion + current 两份对比，差异立刻 evidently 落在 `hand[1],hand[3]` vs `hand[0],hand[1]`。这个 pattern（gated on env var、dump 仅在 DAG 缓存命中且 legal 不一致时打）值得保留为可启用的诊断，不是常驻代码——本次修完后已下线。

### 不在范围

- Coup 的同模式问题暂不修：Coup 的 `hash_field_slot` 对 `influence` 也没 mix `idx`，但 Coup 的 owner-only viz 在当前规则下未被动态收缩（"revealed" 是 public flag，不撤 viz；hand 数永远 2），visited idx 集合稳定，未触发；此外 Coup 整体走 §G.2 重写路径，按用户指示不动。

### 已被框架层防御性修复覆盖

LL 一行修复完成后，`idx` 编码上交给 framework：`state_hash_for_perspective` 改成走 schema 全集（visible + hidden），每槽先 mix `(field_pos, idx[])` 做结构盐，然后视 viz 派发——visible 调游戏的 `hash_field_slot`（**只 mix 值**），hidden mix 一个固定常量 `kHiddenHashSentinel`（`engine/core/types.h`）。改动落在 `engine/core/types.h` / `engine/core/viz_walker.h`（新增 `for_each_slot` 全集 walker）/ `engine/core/schema_hash.h`，同时回滚 `loveletter_state.cpp` 里 `idx[0]+71` 的临时修复（变成框架行为的一部分）。

副作用：之前 LL `apply_public_state` 在 owner_overlay 全 -1 时不撤 `drawn_card` 的 viz，旧 hash 因为隐藏槽不进 hash 而掩盖了这个不一致；新的"hidden 也算 hash 结构"立刻让 `test_public_snapshot_round_trip[loveletter]` 抓到，顺带在 `loveletter_register.cpp` 把 `apply_public_state` 改成 wholesale-replace（>=0 设值+viz=1，-1 撤 viz=0）。

整套修复后："游戏忘记 mix idx" 这一类 bug 从游戏层消失（framework 接管），未来加新 reveal/swap 机制都不再需要 audit `hash_field_slot` 是否漏 idx。

### 后续：彻底删除 off-schema 后门（2026-05-12）

第一版的"框架接管 idx 编码"保留了 `IGameState::hash_extra_state_fields(perspective, h)` 钩子作为"还没迁进 schema 的字段"的临时通道——LL 用它 hash `discard_piles` / `face_up_removed` / `drawn_card`、Coup 用它 hash `revealed` / `court_deck` / `exchange_drawn`。这个口子一旦留下，新游戏作者就会无意识地把"我懒得迁 schema 的字段"塞进去；而它的语义是"绕过 schema walker"，所以 BUG-037 类的结构碰撞会再次出现。

这一轮把它彻底删掉：

1. **LL 变长字段全部转 count 数组并迁进 schema**：`std::vector<int8_t> deck` → `std::array<std::int8_t, kCardTypes+1> deck_count`（`all_hidden`，randomize_unseen 重建）；`std::vector<int8_t> discard_piles[N]` → `discard_count[N, kCardTypes+1]`（`all_public`）；`std::vector<int8_t> face_up_removed` → `face_up_count[kCardTypes+1]`（`all_public`，仅 2p 用）。视觉上还需要"按时间顺序展示弃牌堆"的 UI 从 action stream 重建，不进 hash。`drawn_card` 已经在 schema 里（base `all_hidden`），rules 在抽牌时 `reveal_slot_to(current_player)`，hash 视 viz 决定走值还是哨兵。
2. **`hash_extra_state_fields` 从 `IGameState` 删掉**：`engine/core/game_interfaces.h` 移除 virtual 声明，`engine/core/schema_hash.h` 移除调用点。Coup 的 override 留在 `games/coup/coup_state.{h,cpp}` 里但已不会被调用——按用户指示 Coup 此轮不修，索性把整块从 `games/manifest.json` 摘掉，不参与编译（CMakeLists 和 setup.py 都从 manifest 读，不需要改）。
3. **新立场，文档化**：`engine/core/game_interfaces.h` 的 `IGameState` 注释明确写"every piece of state that participates in DAG node identity MUST be a schema slot"——变长结构通过定长 count 数组表达，视觉顺序是表现层关切，从 action stream 重建。

效果：framework 不再有让"半成品 schema"通过的口子；新游戏作者写 `hash_field_slot` 时只能 mix 已声明 slot 的值，schema-外的 hash 输入物理上不存在。`tests/{loveletter,quoridor,tictactoe,azul,splendor,framework}` 全过，BUG-037 200-episode 复现器仍 0 撞车。

