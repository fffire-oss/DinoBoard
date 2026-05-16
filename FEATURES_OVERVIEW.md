# DinoBoard — 功能概览

给开发者的速查地图。详细实现见 [GAME_DEVELOPMENT_GUIDE.md](docs/guide/GAME_DEVELOPMENT_GUIDE.md)；
框架契约 + MCTS 算法见 [ALGORITHM_OVERVIEW.md](ALGORITHM_OVERVIEW.md)。

---

## 核心组件

一个完整的游戏 = `state.cpp` + `rules.cpp` + `net_adapter.cpp` +
`register.cpp` + `config/game.json` + `web/`

| 文件 | 接口 | 职责 |
|------|-----|------|
| `<game>_state.cpp` | `IGameState` | 状态、当前玩家、终局；schema-driven `hash_field_slot` / `mask_field_slot` / `read_field_slot` / `write_field_slot` 分发器；静态 `schema()` 函数声明字段（name + 数据 shape + base viz tensor），是 hash / encoder / snapshot scope 的单一事实源 |
| `<game>_rules.cpp` | `IGameRules` | 合法动作、`do_action_fast`（含 viz 维护，`reveal_slot` / `reveal_slot_to` / `reset_to_base`；不支持 undo——MCTS / selfplay / arena / web 都丢弃用完的 state）；想接 tail solver 才额外实现 `do_action_deterministic` + `undo_action`（这两个配对工作） |
| `<game>_net_adapter.cpp` | `IFeatureEncoder` + `IBeliefTracker` | encoder 入参锁 `const IGameState& masked_state`（单一 `encode_features(masked_state, perspective, tracker, out)`；非虚入口 `encode` 先 `make_masked_state` 再转发，viz=0 槽位天然是 placeholder）；tracker(有非对称隐藏信息 / 需要 sim 入口对 viz=0 槽位 determinization 的游戏才需要;纯公开物理随机由 `do_action_fast` 中的 `sim_rng` 处理,不需要 tracker) |
| `<game>_register.cpp` | `GameBundle` 工厂 + `GameRegistrar` | 组件打包注册，配变体（如 `splendor_3p`）和可选组件 |
| `config/game.json` | — | 训练超参（simulations / lr / 网络结构等） |
| `web/<game>.js` | `createApp(...)` | 玩家交互界面 |

---

## 框架架构：GT 推进 + AI session 各自独立

DinoBoard 把"推游戏"和"AI 决策"彻底分离。这是整个框架的心智模型，先
立这个再读后面所有章节。

- **GT（ground truth）端推进游戏**——持真值 state，每步调
  `do_action_fast` 推进，产出公开消息流（每步的 `events` 列表 +
  `public_snapshot`）。GT 可以是我们的 C++ 引擎，也可以是外部 API、甚至
  物理桌游——AI 不在乎 GT 是谁
- **AI session（每个 perspective 一份）只吃消息流**——内部持一份本地
  state 副本，public 字段每步从 `public_snapshot` 整张覆盖;viz=0 hidden
  槽位 session 不维护(决策侧物理上读不到——hash 看 `kHiddenHashSentinel`、
  encoder 看 placeholder、MCTS sims 在自己克隆的 sim_tracker 上才调
  `randomize_unseen`)。session state 永远不是 truth 的拷贝
- **MCTS / encoder 在 selfplay / web / API 三条路径下吃的都是 AI session
  state**（不是 truth）——这是"AI 结构性不读真值"的根。AI session 的
  接口里没有 `IGameState*` 指向 truth，物理上拿不到

让这条架构靠住的内部机制（每条都有结构性守护，详见
[ALGORITHM_OVERVIEW.md](ALGORITHM_OVERVIEW.md)）：

- **VisibilitySchema 是单一事实源**：每个字段声明 name + 数据 shape +
  base viz tensor。可见性完全由 viz 承载。RNG 由调用方持有：
  `reset_with_seed` 用一次性 mt19937 即弃，`do_action_fast` 的 rng 走入参
- **rules 是 viz 的唯一游戏侧 writer**：`do_action_fast` 在改业务字段的同
  时调 `reveal_slot` / `reveal_slot_to` / `reset_to_base` 维护
  `state.viz_`。Receiver 侧 GT 把 perspective 的整张 viz 切片连同
  `__viz__` 键一起写到 wire；框架 helper `viz::apply_full_slice`
  （由 `viz::apply_public_snapshot` 调用）把 `state.viz_[name][..., perspective]`
  按字节整张覆盖。没有第二个游戏侧 writer，没有 per-game viz 推导 hook；
  `test_rules_sole_viz_writer` 仅守游戏侧 I1（框架 helper 在
  `engine/core/` 下，不在它的扫描范围内）
- **walker 一次产出 masked clone**：`make_masked_state(state, schema,
  perspective)` 按 schema 遍历每槽——viz=1 复制真值，否则写
  `kPlaceholder` sentinel。snapshot（GT 端序列化）/ hash（sim 内每步）/
  encoder（sim 内新节点）三家共用同一份 masked clone，结构性对齐
- **接口入参锁 `const IGameState& masked_state`**：`IFeatureEncoder` /
  `IPolicyValueEvaluator` 的入参签名约定它是 walker 已经 mask 过的副本
  ——非虚入口 `IFeatureEncoder::encode` / `IBeliefFeatureExtractor::extract_from_state`
  在 forward 前先 `make_masked_state` 把 hidden 槽位换成 placeholder，
  子类 override 拿不到 raw state（没有独立的 C++ 类型阻挡误传——结构性
  屏障是这层包装 + placeholder + 回归测试）
- **AI session 端不重算 viz**：observe 时按 schema 序 walker 把
  `public_snapshot` 整张写回 state 公开字段；viz 由 schema base + 规则期间
  `reveal_slot` / `reset_to_base` 已经维护好，从消息侧来的也是同一张
  schema-driven 视图——运行时正确性 = 消息传输完整性

---

## 搜索：ISMCTS

神经网络引导的 MCTS，原生支持 2-4 人。**算法详解见
[ALGORITHM_OVERVIEW.md §9](ALGORITHM_OVERVIEW.md#9-mcts--ismcts-over-a-dag)**。
要点：

- **Root 采样 determinization**：每次 sim 持独立 RNG；root 从 belief
  采一个完整世界,descent 期间该 RNG 仍可能被 `do_action_fast` 消费处
  理物理随机(如 Azul 工厂 refill)。给定 sim 的种子,整个 sim 的展开
  完全可复现
- **DAG 而非 tree**：全局 hash 表，同一 info set 不同路径共享节点；UCT2
  修正多父路径下的 over-exploration
- **节点 key 按 acting player 视角**：
  `state_hash_for_perspective(current_player)` —— 每个决策节点是合法 info set
- **Step counter 防环**：`step_count_` 单调递增，DAG 结构性 acyclic
- **每个 descent 步只 mask 一次**：`make_masked_state` 出来的 masked clone
  喂给（hash 和）encoder，同一份副本不重复遍历
- **Encoder 对齐 hash scope**：encoder 入参锁 `const IGameState& masked_state`，viz=0
  槽位结构性是 placeholder，物理上无法读出对手 private
  （`test_encoder_respects_hash_scope` 守护）

主要配置：`simulations` / `c_puct` / `temperature`（支持 schedule）/
Dirichlet 噪声。

---

## 随机性与隐藏信息

物理随机和信息不对称在 ISMCTS 里**统一处理**：

- Root 采样吞掉所有未来随机
- 观察者**可见**的后果通过 schema 公开槽位（mask=1）的差异自然分叉成不同
  hash 节点
- 观察者**不可见**的后果在 masked clone 里全是 placeholder、对所有 sim 一
  样，自然合并到同一节点

没有显式 chance node 机制。隐藏信息游戏需注册 `belief_tracker` + 在
schema 里 declare 私人字段（owner-only viz）+ 实现
`mask_field_slot` 让 walker 物化时把 placeholder 写到位。详见
[Guide §10](docs/guide/GAME_DEVELOPMENT_GUIDE.md#10-隐藏信息与-belief-tracker含物理随机性)。

**现有游戏的组合**：

| 游戏 | 物理随机 | 信息不对称 | schema 形态 |
|------|---------|-----------|------|
| TicTacToe / Quoridor | 无 | 无 | all_one（trivial） |
| Azul | 袋中抽（对称） | 无 | all_one（袋子内容 hash 派生自公开 token 总数） |
| Splendor | 翻牌 | 暗压牌 | mixed（reserve owner-only） |
| Love Letter | 抽牌 | 手牌 | mixed（hand owner-only） |
| Coup | 抽牌 + bluff | 影响牌 + claim | mixed（influence + exchange-seen owner-only） |

### 混合均衡游戏的三板斧

Coup 这类「混合策略均衡 + 隐藏信息 + bluff」的游戏单靠 ISMCTS 是吃不下
来的——下面三件组合起来才咬得住，缺一不可：缺 belief → claim 不影响
sim；缺对手行为冻结 → bluff EV 永远是负的；缺池 → 训练震荡。Coup 当
前三件全开。

#### 板斧一：网络化 belief（避开 uniform 采样退化）

最朴素的 `randomize_unseen` 是按 `remaining[R]` 的张数 weighted 抽——
"袋里剩多少就按多少抽"。Splendor 这种"袋里有什么我都看得到"的游戏没
问题；**诈唬游戏（Coup）上是结构性 bug**：对手早期 claim Tax 的信号
在 sim 入口完全消失（被 uniform 一抹平），网络很快学会"永远
Challenge"，gating 来回震荡也不会装 / 不会基于 claim 做有信息的决策。

框架提供网络化 belief 路径解决这条：游戏注册 `belief_feature_extractor
+ belief_model_path + belief_label_extractor`，框架在 root 处跑一次独
立 belief 网络，输出空间 = 采样空间 = 对手手牌 multiset 上的
categorical（Coup alive=2 走 15 维二张 multiset 分支、alive=1 走 5 维
单张分支），按 kBeliefTemperature 做分支内 softmax 缓存到
pi_hand_[opp]。sim 入口 randomize_unseen 在 clone 出的 sim_tracker 上
读这张 cache，叠 deck remaining[R] feasibility mask 后一次 categorical
抽完整手牌——没有 marginal 拼装，也不再走 Wallenius weighted 那条路
径。把 hand-craft uniform 退化均衡换成 selfplay 自己学出来的 prior。
当前 Coup 启用，详见
[Coup BELIEF_NETWORK.md](games/coup/BELIEF_NETWORK.md) +
[ALGORITHM_OVERVIEW §8.4](ALGORITHM_OVERVIEW.md#84-可选网络化-belieflearned-posterior)。

#### 板斧二：对手行为冻结（避开对手全知化）

ISMCTS 默认对每个 opp 决策节点也跑 PUCT bandit。bluff 游戏里这条会让
sim 内对手在每个 determinization 里都挑「这个世界下最好」的动作——
**高估对手 value、低估自己 value**，bluff EV 永远是负的，网络收敛到
「永远说真话」。

框架提供 per-profile 开关 `mcts_profiles.<name>.opponent_selection ∈
{"puct", "prior"}`：切到 `"prior"` 后非根 opp 节点不再 bandit，而是
直接按策略头的冻结先验做 multinomial 采样（Smooth-UCT 思路），跨
determinization 的对手行为分布一致，bluff EV 不再被「对手总能精准戳
穿」吃掉。**根节点本人座位永远 PUCT** 不受影响。Coup `selfplay /
arena / web_expert` profile 默认开启，详见
[ALGORITHM_OVERVIEW §9](ALGORITHM_OVERVIEW.md#9-mcts--ismcts-over-a-dag) +
[CONFIG_REFERENCE mcts_profiles](docs/guide/CONFIG_REFERENCE.md#mcts_profiles)。

#### 板斧三：对手池（避开 mirror selfplay 的反向钟摆）

纯 self-play 下双方策略同步漂移、没人锚定均衡——bluff/truthful 的最
优响应一直在跟着对手跑，训练曲线会在「全说真话 ↔ 全 bluff」之间剧烈
震荡，stdev 大到 gating 完全是噪声驱动的。混合策略均衡不会被 single
策略 self-play 自然找到。

框架提供对手池机制：每步一部分 worker 用 `opponent_pool_self_ratio`
比例把对手换成历史 `model_step_*.onnx`（`_collect_pool_paths` 收集所
有 checkpoint，不按 gating 过滤），仅保留 latest 座位的样本。新策略
被迫同时对池里所有历史版本可解 → 经验平均 ≈ 混合均衡，类似 fictitious
self-play / NFSP / Deep CFR 的廉价实现。Coup 训练池开到 ~30 个
checkpoint 后，p0 第一步 Tax bluff 才从 0% 稳定到 ~45% 的混合策略附
近。详见
[CONFIG_REFERENCE opponent-pool](docs/guide/CONFIG_REFERENCE.md#opponent-pool对手池)。

### 残局求解（Tail Solve）

MCTS 前用 alpha-beta 尝试精确求解，proven win 时跳过 MCTS。多人游戏被压
成"我 max、其他所有对手联合 min"的两方博弈——给出的是 root 价值下界，
所以**算出 proven win 时一定真能赢**（对手实际不会合谋专门搞你，只会更
弱）；非 win 仍交回 MCTS。

随机/隐藏信息游戏需实现 `do_action_deterministic`（用占位符替代未知字
段，比如 Splendor 的 `forced_draw_override = -2`）。**与 `do_action_fast`
最大的两点区别**：

1. `do_action_deterministic` 必须**支持 undo**——和 `undo_action` 配对工
   作，alpha-beta 在同一个 state 上 do → recurse → undo 反复滚动；
   `do_action_fast` 不支持 undo（MCTS / selfplay / web 都丢弃用完的
   state，省掉拍快照的开销）
2. `do_action_deterministic` 必须 freeze 隐藏抽牌（让结果可重放）；
   `do_action_fast` 的随机走入参 rng

Splendor / Azul 已实现，Love Letter / Coup 未实现。详见
[Guide §3.2](docs/guide/GAME_DEVELOPMENT_GUIDE.md#32-可选实现do_action_deterministic残局求解需要)。

---

## AI API（observation-only）

`platform/ai_service/` 提供 REST API：调用方只传 `action_id` + 公开事
件，session 内部自维护 state，响应只有 `action_id` + 元数据。端点：
`POST /ai/sessions[/{id}/observe|decide]` /
`DELETE /ai/sessions/{id}`。

这是上文"GT 推进 + AI session 各自独立"架构的对外接口实例化。**双重
身份**：

1. **结构性证明 AI 不读真值**——架构层已经讲过 AI session 拿不到
   `IGameState*` 指向 truth；API 层的额外守护测试有
   `test_public_snapshot_round_trip` / `test_public_hash_excludes_internal_rng` /
   `test_api_belief_matches_selfplay`,以及一个针对 Love Letter 的
   API/selfplay MCTS 策略统计回归 `test_api_mcts_policy_invariance`(Splendor 因
   replay / `self_reserve_deck` 交错的已知问题暂不纳入,Web 路径不直接
   覆盖)
2. **接入第三方**——GT 端可以是任意来源（外部 API、物理桌游），只要实现
   "事件 → API"翻译层即可使用 AI，无需共享 state 代码或嵌入 C++ 引擎

API 文档：[AI_API.md](docs/guide/AI_API.md)。每游戏的动作编码 + 事件格式：
`docs/games/<game>_api.md`（新游戏必写）。

---

## 训练

```
selfplay → 收集样本 → 训练网络 → gating eval → 更新 best model → 循环
```

全流程 C++（selfplay / 搜索 / 求解 / eval），Python 只做训练循环和网络
训练，配置驱动。

**Value Head**：N 维向量（N = num_players），perspective-relative
（`values[0]` = 当前玩家）。MCTS backup 时旋转回绝对玩家顺序。ONNX
evaluator 在加载到 2p 标量 value head 时自动按零和展开为 2 维；3p+ 标量
直接报错。

**训练增强**（每项详见
[Guide §9](docs/guide/GAME_DEVELOPMENT_GUIDE.md#9-训练可选特性)）：

从通用 → 框架专属排（前面是任何 DL / AlphaZero 训练栈都有的旋钮，后面越来越是本框架特有的解法）：

| 机制 | 说明 |
|------|------|
| 学习率调度 | `training.lr_schedule`：constant / cosine / step；cosine 是默认推荐 |
| 梯度裁剪 | `training.grad_clip_norm` 防止 PV 早期 value head 抖出 NaN |
| 温度 schedule | 分段线性衰减（temperature_initial → temperature_final，按 plies） |
| Dirichlet 噪声 | 根节点注入，可限制前 N 步（AlphaZero 标准探索） |
| 启发式 warmstart 权重 | `--init-from <path.pt>` 从既有 PyTorch checkpoint 加载权重作为新 run 的起点（如 Splendor 13500-step legacy checkpoint），跨网络变体可加输入适配层 fine-tune |
| 启发式引导 | `heuristic_guidance_*` 三段式 schedule（hold → 衰减 → 0），早期一部分 selfplay 走 `heuristic_picker` 而非 MCTS，相当于 warmup（DEC-002 已把独立 `--warmstart-steps` 折进这条 schedule） |
| 辅助训练信号 | `auxiliary_scorer` 在 PV value 之外加一个 score head（对 Splendor / Azul 这类「快赢」游戏特别有效） |
| 动作过滤 | `training_action_filter` 裁剪垃圾动作，schedule 衰减到 0；`legal_mask` 始终为完整集；profile 级 `ai_use_action_filter` 可整体关闭 |
| 超时裁决 | `adjudicator` 在 `max_game_plies` 后判胜负，避免 selfplay 死循环 |
| 残局求解 | `tail_solve_*`（profile 开关 + depth/budget/margin），proven win 时跳过 MCTS；要游戏注册 `tail_solve_trigger` |
| `cover_root_edges` | profile 开关；分析 / precompute 用 true 让 root 每条 legal edge 至少 visit 一次（保证掉分计算覆盖全），训练 / 对战不开 |
| 对手池 | `opponent_pool_enabled` + `opponent_pool_self_ratio`：每步一部分 worker 把对手换成历史 `model_step_*.onnx`，仅 latest 座位入 replay buffer，缓解 mirror selfplay 的策略坍缩 / 混合均衡反向钟摆。详见 [CONFIG_REFERENCE](docs/guide/CONFIG_REFERENCE.md#opponent-pool对手池) |
| `opp_sel="prior"` | 非根 opp 节点用策略头先验采样替代 PUCT bandit，避免 ISMCTS strategy fusion / 对手全知化（bluff 游戏关键，详见上文「混合均衡游戏的三板斧」） |
| 网络化 belief | `belief_*` 三件（feature_extractor / model_path / label_extractor）+ `game.json::belief` 块；只对 uniform-from-pool 退化严重的游戏需要（目前只有 Coup） |

**评估**：每 `--eval-every` 步触发，含 benchmark eval
（`heuristic_constrained` / `heuristic_free` / 指定 ONNX）+ gating（latest
vs best，胜率 ≥ 阈值更新 best）。N 人游戏 candidate 轮坐每个座位以消除
偏差。

---

## Web 前端

前端通过 HTTP API 与引擎交互，读 `state_serializer` 渲染、读
`action_descriptor` 理解动作。

**交互设计原则**（详见
[WEB_DESIGN_PRINCIPLES.md](docs/guide/WEB_DESIGN_PRINCIPLES.md)）：自然交互（不
要每动作一按钮）、空间锚定（固定区域固定位置和尺寸）、动作动画（每动作
有 `describeTransition`）、视觉可辨。

**通用布局**：上方公共游戏区 + 信息栏（回合/胜率/AI 提示）+ 录像窗口；下
方玩家区域（2p 左右，3-4p 网格）。general 层提供，游戏前端只填充内容。

**多人模式**：`players.max > 2` 的游戏侧边栏显示人数 + 座位选择。AI 连
续落子直到轮到人类。

**高级操作**（general 层统一实现）：悔棋、替对手落子（隐藏信息游戏可
`disableForce: true` 关闭）、智能提示、胜率预估、对局中显示录像栏。隐藏
信息游戏须 `showWinrateDefault: false`（胜率/失误标记基于 root values，
含真实隐藏状态）。

**Web AI 配置**（`config/web.json`）：`mcts_profiles.{web_expert, web_casual, analysis}` 三个命名 profile，每个含完整 MCTS 旋钮（simulations、temperature、tail_solve、ai_use_action_filter、cover_root_edges 等）；难度 → profile 映射 `{casual → web_casual, expert → web_expert}`，录像分析 / precompute 走 `analysis`。详见
[配置参考](docs/guide/CONFIG_REFERENCE.md#webjson)。

**AI Pipeline**：每个 human-to-play 局面只跑一次 MCTS（precompute），结
果两用——智能提示返回 + 落子后读 `action_values[chosen]` 算掉分。AI 自
己的决策是另一次独立 MCTS。详见
[Web 开发指南](docs/guide/WEB_DEVELOPMENT_GUIDE.md#8-ai-pipeline-与动作分析)。

**录像回放**：仅专家难度可用，结束后进入回放，每帧附带掉分（≥5% 失误、
≥10% 严重失误）。`platform/tools/eval_model.py` 可独立生成录像。

---

## 可选组件速查

| 组件 | 何时需要 |
|------|---------|
| `tail_solver` / `tail_solve_trigger` | 想精确求解残局 / 智能触发求解 |
| `heuristic_picker` | 启发式给动作打分，混入 selfplay 加速早期学习 |
| `auxiliary_scorer` | 提供胜负外的额外学习信号 |
| `training_action_filter` | 动作空间有明显垃圾，训练时裁剪 |
| `adjudicator` | 游戏可能死循环，超时判胜负 |
| `episode_stats_extractor` | 自定义指标追踪 |
| `belief_tracker` | 有非对称隐藏信息(Love Letter / Splendor / Coup),需要在 ISMCTS sim 入口对 viz=0 槽位采样;纯公开物理随机的 Azul 不需要——其物理随机由 sim_rng 在 `do_action_fast` 里直接消费 |
| `public_event_extractor` / `applier` / `public_state_applier` | snapshot-path 游戏(隐藏信息 + Azul)在 message-driven 路径下维护 session 公开字段(每 ply truth 端 extract → observer 端 apply 覆写) |
| `belief_feature_extractor` + `belief_model_path` | 启用网络化 belief(learned posterior)代替 uniform-from-pool 退化采样;两者一起注册,框架自动 load `OnnxBeliefEvaluator`,在 root 一次推理 → tracker 按 alive 切分支做 softmax 缓存 `pi_hand_[opp]` → sim 入口在 multiset 上一次 categorical 抽完整手牌(叠 deck remaining feasibility mask)。当前 Coup 启用,详见 [Coup BELIEF_NETWORK.md](games/coup/BELIEF_NETWORK.md) + [ALGORITHM_OVERVIEW §8.4](ALGORITHM_OVERVIEW.md#84-可选网络化-belieflearned-posterior) |
| `belief_label_extractor` | GT-side(只在 selfplay runner emit 路径调,AI session / wire 拿不到)产 belief 训练 label(对手手牌 multiset id + remaining + alive_per_opp);只有训练 belief 网络的游戏需要 |

完整字段说明见 [CONFIG_REFERENCE.md](docs/guide/CONFIG_REFERENCE.md)。

---

## 新游戏开发步骤

分两个阶段：1–10 让游戏跑起来（编译 / 注册 / web 能玩 / 测试全绿），
11–12 让游戏训得好（看曲线调参；隐藏信息 + 退化严重的游戏才上
learned belief）。

### 阶段 A：跑起来

1. **定义状态结构**——继承 `CloneableState<T>`，业务字段 + viz tensor
2. **写 schema**——`<game>_state.cpp` 里 `static const
   viz::VisibilitySchema& schema()`，declare 每个字段 name + 数据 shape
   + base viz tensor（`all_public` / `owner_only_first_axis` /
   `all_hidden` 等 builder 覆盖大部分情形）。schema 是 hash / encoder
   / snapshot scope 的单一事实源
3. **实现规则**——`legal_actions` + `do_action_fast`，后者同时调
   `reveal_slot` / `reveal_slot_to` / `reset_to_base` 维护
   `state.viz_`。要接 tail solver 才额外实现 `do_action_deterministic`
   + `undo_action`（配对工作）。**`do_action_fast` 不支持 undo**：返
   回的 `UndoToken` 是和 `do_action_deterministic` 共享签名的 vestige，
   不要在 fast 路径 push undo_stack / 拍快照——MCTS / selfplay / web 都
   丢弃用完的 state，拍快照纯浪费
4. **schema-driven 分发器**——state 里实现 `hash_field_slot` /
   `mask_field_slot` / `read_field_slot` / `write_field_slot` /
   `schema_ref`，按 schema 列字段答 typed value
5. **特征编码器**——单一 `encode_features(masked_state, perspective,
   tracker, out)`，入参锁 `const IGameState& masked_state`，viz=0 槽位
   读 placeholder 分流。tracker 引用是 belief tracker 的公开聚合特征
   入口
6. **写 `register.cpp` 组装 GameBundle**——核心是把 1–5 拼起来。
   **snapshot-path 游戏**（manifest 挂 `"snapshot"` capability 的）额
   外调一行：

   ```cpp
   b.install_event_protocol(extract_events_only, schema_provider);
   ```

   `extract_events_only(state_before, action, state_after, perspective)
   → vector<PublicEvent>` 只 diff 出 tracker 需要的事件，**不**做 viz
   序列化——viz 切片由 framework 整批塞。框架据此自动派生
   `events_only_extractor`（sim 路径）/ `public_event_extractor`
   （events + framework 的 `viz::serialize_public_snapshot`，wire /
   selfplay / trace 走这条）/ `public_state_applier`（framework 的
   `viz::apply_public_snapshot`，接收侧 session 走这条）。tracker 不
   依赖 events（纯 viz 推断）就传 `viz::no_events_extractor`
7. **(隐藏信息游戏额外一步) tracker bootstrap**——tracker 实现
   `pack_init_payload(gt_state, perspective) → AnyMap` 配合 `init`，
   把 perspective 私有的初始观测灌进去（viz=1 槽位塞不下的部分，例如
   Love Letter 起始手牌）
8. **写 `game.json` 配置**——MCTS profiles / training / 网络结构等。
   字段全集见 [CONFIG_REFERENCE.md](docs/guide/CONFIG_REFERENCE.md)
9. **在 `games/manifest.json` 追加一条**——`{ id, enabled,
   framework_whitelist, capabilities, sources }`，详见下面字段说明表。
   漏了 = 编译过但 register 不上 = web 看不见 = 测试不覆盖
10. **写 Web 前端 + 跑测试**——`web/<game>.js` 用 `createApp`，详见
    [WEB_DEVELOPMENT_GUIDE](docs/guide/WEB_DEVELOPMENT_GUIDE.md)；然后
    `pytest tests/<game>/` 一次全绿（隐藏信息游戏额外要求
    `test_tracker_consistent_with_truth` /
    `test_ismcts_samples_respect_tracker` /
    `test_public_hash_excludes_internal_rng`）

### 阶段 B：训得好

11. **跑训练 / 看日志 / 调参**——主要旋钮在
    `mcts_profiles.selfplay.simulations` / `c_puct` / `temperature
    schedule` / `dirichlet`，外加可选的 `opponent_pool_*`、
    `heuristic_picker`、`tail_solver`。train.log 字段含义和触发各异常的
    阈值见 [CONFIG_REFERENCE.md](docs/guide/CONFIG_REFERENCE.md) +
    [Guide §11 训练循环可选组件](docs/guide/GAME_DEVELOPMENT_GUIDE.md)
12. **(可选) 上 learned belief**——只对 uniform-from-pool
    `randomize_unseen` 退化严重的游戏做（目前只有 Coup；PV 链路跑通且
    bluff/挑战维度仍未被 ISMCTS 解开时再上）。需要四件事：
    - (a) `<game>_net_adapter.cpp` 写 `belief_feature_extractor`（AI 侧
      输入，与 PV encoder 一样吃 masked clone）+ `belief_label_extractor`
      （**只在 GT 侧 selfplay runner emit 路径调**——AI session / wire /
      web / API 物理上拿不到，这是 belief 网络的「无 truth 泄漏」红线）
    - (b) `register.cpp` 注册三件：`b.belief_feature_extractor` /
      `b.belief_label_extractor` /
      `b.belief_model_path = "games/<g>/model/<g>_belief_<N>p.onnx"`
    - (c) `game.json` 加 `belief` 块（architecture / temperature /
      training schedule）
    - (d) **每个 N-player 变体单独 init 并训练一份 `<g>_belief_<N>p.onnx`**
      ——belief 网络的输入维度 `6 + 28×(N-1)` 跟 N 绑死，PV 网络也是这样，
      变体之间不能共用权重

   详见 [Coup BELIEF_NETWORK.md](games/coup/BELIEF_NETWORK.md)。

### `games/manifest.json` 字段说明

| 字段 | 含义 |
|------|------|
| `id` | 游戏 id；建议是 `<base>` 名字（变体 `<base>_2p` / `_3p` / `_4p` 由 GameRegistrar 在 register 时派生） |
| `enabled` | bool，默认 true。false = 完全跳过这条目（不编译、不出现在 `available_games()`、不上 web、所有 framework 测试自动忽略） |
| `framework_whitelist` | bool，默认 false。true 表示这个游戏**必须**通过 framework matrix 守的所有 invariant（无 truth 泄漏、hash perspective 正确、snapshot round-trip 等）。新游戏开发期可以先 false，跑通验收清单后再翻成 true |
| `capabilities` | 字符串列表，driving framework 测试矩阵：`"hidden_info"`（有 perspective-private 槽）/ `"snapshot"`（注册了 public_state_applier）/ `"tracker"`（注册了 belief_tracker factory）。每加一个标签，对应那条 capability 的 framework 测试就会自动覆盖这个游戏 |
| `sources` | 编译进 `_dinoboard_engine` 的 `.cpp` 文件名，相对 `games/<id>/`。CMake 和 setup.py 都按这个列表喂给编译器 |

下线 / 重新启用一个游戏的全部操作：改 manifest 的一行 `enabled` 字段，跑 `pip install -e .` 重新编译，结束。

参考实现按接入模式分类：

| 游戏 | 展示什么 |
|------|---------|
| TicTacToe | 最小闭环；schema 全 all_one，不需要任何 optional 组件 |
| Quoridor | 完全信息确定游戏；heuristic_picker / tail_solver / adjudicator / auxiliary_scorer / training_action_filter 全配齐 |
| Splendor | **端到端 walker 化参考实现**：reserve owner-only viz、`mask_field_slot` 走 COW shared_ptr 一次 detach、snapshot 走 `viz::serialize_public_snapshot` walker 路径（GT 整张 viz 切片随 wire 同传）、`do_action_deterministic` 用 `forced_draw_override = -2` 占位符 |
| Azul | 纯对称物理随机；schema 全 all_public（袋子和 box_lid 在 schema 里以 per-color counts 体现），不注册 `belief_tracker`——sim 入口没东西可 determinize，物理随机走 `do_action_fast` 里 `sim_rng` 即时抽 |
| Love Letter | 非对称隐藏 + viz reveal 槽位承载确定信息（rules 通过 `reveal_slot_to` / `swap_slot_owned` 写入），tracker stateless 只做剩余牌池均匀采样 |
| Coup | **网络化 belief 范例**：multiset categorical posterior（15 维二张 + 5 维单张分支）+ 三板斧（belief / `opp_sel="prior"` / 对手池）应对混合均衡 bluff 游戏 |

详见 [Guide §13 完整 Checklist](docs/guide/GAME_DEVELOPMENT_GUIDE.md#13-完整-checklist)。

---

## 测试

**两层架构**：

- **`tests/framework/`** —— 框架不变量，分两层：
  - 纯 framework matrix 测试（encoder / sample_collection / 守恒律
    类）跑在固定 3 游戏 matrix 上：`FRAMEWORK_GAMES = ["quoridor",
    "azul", "loveletter"]`，最小完备覆盖确定/对称随机/非对称隐藏 ×
    2p/2-4p × tail solver / belief tracker 等结构特征
  - AI / observation 协议测试（`test_selfplay_no_truth_in_ai_path` /
    `test_public_snapshot_round_trip` / `test_deployed_models_match_
    encoder` 等）按 capability 标签从 `games/manifest.json` 动态派生覆盖
    集（`enabled_games()` / `games_with_capability("hidden_info")` /
    `games_with_capability("hidden_info", "snapshot")` 等 helper 在
    `tests/conftest.py`），新游戏 manifest 一加 capability 测试矩阵自动
    跟上、不需要改测试源

  其中 `test_visibility_schema` / `test_snapshot_keys_match_schema` 守
  walker 路径：schema 改了忘了同步分发器 → CI 立刻 fail
- **`tests/<game>/`** —— 每游戏完整验收清单，包含 `TestRuleInvariants`
  用 `run_random_episode_states` 驱动随机对局并断言**该游戏自己的守恒
  律**（token / 卡 / 棋子总量、容量上限等）

新游戏 ready = `pytest tests/<game>/` 一次全绿。隐藏信息游戏额外要求
`test_tracker_consistent_with_truth` + `test_ismcts_samples_respect_tracker`
+ `test_public_hash_excludes_internal_rng`。

接入流程：从最相近的现有游戏复制 `tests/<game>/test_checklist.py`，改
`GAME = "..."`，根据测试失败迭代。详见
[新游戏验收测试指南](docs/guide/NEW_GAME_TEST_GUIDE.md)。

---

## 框架局限性

本框架基于 AlphaZero 范式（MCTS + 神经网络），按"是否能根治"分组列出
不适用场景。

### 范式硬冲突（不可解，是框架边界）

这一组场景跟 AlphaZero / ISMCTS / zero-sum self-play / god-view viz 这
些**核心范式假设**相悖，不是补 helper 或调参能解决的——撞上就换框架。

1. **需要混合策略均衡的游戏**（如德州扑克）—— AlphaZero 训练确定性策
   略，无法收敛到精确混合 Nash。推荐 CFR / DeepCFR 系算法
2. **动作空间组合爆炸**（如斗地主 27,000+ 出牌组合、万智牌等卡牌构筑
   从 N 张候选选 K 张 `C(N,K)` 爆炸）—— 框架的固定 `action_space` 既稀
   疏又低效，需分层 / 自回归动作解码（参考 DouZero）。卡牌构筑还多一
   层困难：构筑本身是独立 meta-game，本框架只解决"给定牌组怎么打"
3. **不支持合作博弈**（Hanabi / Overcooked 等需要队友配合的游戏）——
   工程项（value head 改共享 reward、gating 换分数阈值、selfplay 改组
   队结构）都改完后，真正的硬冲突是 self-play 假设「partner 跟我同
   分布」：两个 seat 会同步收敛到一个**只对训练里那份自己有效的私
   有约定**（典型如 Hanabi 里"我打第三张固定意味着你 1 号位是 1"这类
   私有 convention），换一个独立 seed 训出来的版本配不上、换人类玩家
   更配不上（zero-shot coordination / ad-hoc teamwork 问题）。根治得
   在训练目标里显式建模「partner 策略不可观测」——other-play 强制对
   称破缺、population-based training 让 partner 多样化、Bayesian
   Action Decoder 显式推 partner belief——这些都不是 AlphaZero 范式
   内的旋钮。MCTS 搜索内核本身能跑 cooperative backup（OpenSpiel /
   SPARTA），是范式不行不是搜索不行
4. **不支持单人游戏 / solitaire**（如纸牌接龙、2048）—— 自博弈学习的
   bootstrap 依赖"双方差不多菜也能有胜有负"：早期网络随机走子，对手同样
   随机，胜负信号自然产生，z_values 给出有方差的训练目标。单人游戏没有
   对手，胜利条件靠规则定义（拼出特定牌型 / 达到分数阈值），随机策略下
   触发胜利的概率几乎为零，整局都是 z=-1，网络拿不到任何"哪种走法更接
   近赢"的梯度，训练根本起不来。根治需要 reward shaping（按距离目标的
   进度给分）或课程学习（从简化局面开始），不是本框架范畴。建议改用
   intrinsic motivation / curiosity-driven 系算法
5. **viz 是公开 god-view 函数,不支持"闭眼"环节**（如狼人杀夜晚阶段、密
   写动作）—— 框架假设"哪个槽位对谁可见"是 god-view 下唯一确定的游戏规
   则:每个玩家都知道"自己看得到 X、看不到 Y、对手能看到 Z",只是看不到
   Y、Z 的内容。闭眼环节违反这条:玩家**不知道对手是否在某一时刻获得了
   信息**(狼人夜晚睁眼互认 → 平民不知道狼人之间有没有共享身份;法官私下
   告诉某玩家牌面 → 其他玩家不知道发生了这次告知)。这种"viz 嵌套在
   hidden 上"的高阶不确定性既不能写成确定的 base_viz、也不能写成
   `do_action_fast` 里的 deterministic reveal。根治需要在 belief 公式里
   建模"对手是否获取了信息"的二阶概率分布,跟当前"viz 是公开规则函数"的
   单一事实源设计冲突,不是补一个 helper 能解决的。详见
   [FRAMEWORK_DESIGN_RATIONALE.md §3.2](docs/FRAMEWORK_DESIGN_RATIONALE.md)

### 投入产出比低（暂时不打算解决）

这一组解决方案明确但当前不值得投入,大多数桌游不命中,命中也有缓解手段。

6. **稀疏奖励 + 长对局**（>200 步）—— 仅终局 z_values 提供训练信号，
   value head 需要从终局一步步往前 bootstrap，游戏越长训练越慢。症状是
   "训练 plateau / value head 长期噪声"，不是 crash。粗略分档：<30 步无
   影响；30–150 步可行（Quoridor ~50-80、Splendor ~30-60、Azul ~120 都
   训练良好）；150–200 步多半需要 `auxiliary_scorer` + `heuristic_picker`
   配合才跑得通；>200 步框架不适合。`AuxiliaryScorer` 是 reward shaping
   性质的缓解，TD(λ) bootstrapping 是更系统的方向，未实现
7. **仅 CPU 自博弈**——当前网络规模（~1M 参数，CPU 单次推理 ~200μs）
   GPU batch inference 的 IPC 固定开销没有摊销价值。网络规模提到 ~5M+
   参数（小型 ResNet / Transformer）后才值得加 GPU

**关于裸 PPO**：两条都不合适。
(1) 决策频率：回合制桌游决策频率低、分支因子有限，每步都值得花
~hundreds–thousands 个 sim 做规划——正是 MCTS 强项；裸 PPO 更适合实时
游戏（星际、Dota）那种「每秒几十步、单步规划没意义」的场景。
(2) model-free vs model-based：裸 PPO 是 model-free，靠纯环境采样
estimate gradient，sample efficiency 远低于 MCTS。本框架是 model-based
（规则就是 perfect simulator，`do_action_fast` 让 sim 几乎无代价），同
样 wallclock 下 MCTS 能 evaluate 的局面数比 PPO 高几个数量级，桌游这种
状态空间相对小、模型完美的场景特别吃这个红利。
