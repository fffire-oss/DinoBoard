# DinoBoard — 功能概览

给开发者的速查地图。详细实现见 [GAME_DEVELOPMENT_GUIDE.md](guide/GAME_DEVELOPMENT_GUIDE.md)。

---

## 核心组件

一个完整的游戏 = `state.cpp` + `rules.cpp` + `net_adapter.cpp` + `register.cpp` + `config/game.json` + `web/`

| 文件 | 接口 | 职责 |
|------|-----|------|
| `<game>_state.cpp` | `IGameState` | 状态、当前玩家、终局、`hash_public_fields` / `hash_private_fields` |
| `<game>_rules.cpp` | `IGameRules` | 合法动作、do/undo（隐藏信息游戏若要 tail solver 还需 `do_action_deterministic`） |
| `<game>_net_adapter.cpp` | `IFeatureEncoder` + `IBeliefTracker` | encoder（`encode_public` + `encode_private(p)` 拆分）；tracker（隐藏信息游戏才需要） |
| `<game>_register.cpp` | `GameBundle` 工厂 + `GameRegistrar` | 组件打包注册，配变体（如 `splendor_3p`）和可选组件 |
| `config/game.json` | — | 训练超参（simulations / lr / 网络结构等） |
| `web/<game>.js` | `createApp(...)` | 玩家交互界面 |

---

## 搜索：ISMCTS

神经网络引导的 MCTS，原生支持 2-4 人。**算法详解见 [MCTS_ALGORITHM.md](guide/MCTS_ALGORITHM.md)**。要点：

- **Root 采样 determinization**：每次 sim 从 belief 采一个完整世界，descent 纯 deterministic
- **DAG 而非 tree**：全局 hash 表，同一 info set 不同路径共享节点；UCT2 修正多父路径下的 over-exploration
- **节点 key 按 acting player 视角**：`state_hash_for_perspective(current_player)` —— 每个决策节点是合法 info set
- **Step counter 防环**：`step_count_` 单调递增，DAG 结构性 acyclic
- **Encoder 对齐 hash scope**：encoder 拆分成 `encode_public` + `encode_private(p)`，结构性禁止读其他玩家 private（`test_encoder_respects_hash_scope` 守护）

主要配置：`simulations` / `c_puct` / `temperature`（支持 schedule）/ Dirichlet 噪声。

---

## 随机性与隐藏信息

物理随机和信息不对称在 ISMCTS 里**统一处理**：

- Root 采样吞掉所有未来随机
- 观察者**可见**的后果通过 `hash_public_fields` 自然分叉成不同节点
- 观察者**不可见**的后果通过 hash 自然合并到同一节点

没有显式 chance node 机制。隐藏信息游戏需注册 `belief_tracker` + `hash_private_fields(p)` + `public_event_extractor` + `public_state_applier` + `initial_observation_extractor/applier`。详见 [Guide §10](guide/GAME_DEVELOPMENT_GUIDE.md#10-隐藏信息与-belief-tracker含物理随机性)。

**现有游戏的组合**：

| 游戏 | 物理随机 | 信息不对称 | 需要注册 |
|------|---------|-----------|------|
| TicTacToe / Quoridor | 无 | 无 | 仅 state / rules / encoder |
| Azul | 袋中抽（对称） | 无 | + belief_tracker（无 hash_private） |
| Splendor / Love Letter / Coup | 翻牌 / 抽牌 | 暗牌 / 手牌 | + belief_tracker + hash_private + events |

### 残局求解（Tail Solve）

MCTS 前用 alpha-beta 尝试精确求解，proven win 时跳过 MCTS。paranoid 假设，支持任意玩家数。随机/隐藏信息游戏需实现 `do_action_deterministic`（用占位符替代未知字段）。Splendor / Azul 已实现，Love Letter / Coup 未实现。详见 [Guide §3.2](guide/GAME_DEVELOPMENT_GUIDE.md#32-可选实现do_action_deterministic残局求解需要)。

### Peek 模式（训练增强）

前 `peek_steps` 步训练用全知状态搜索（跳过 root 采样），帮网络早期学稳；之后切回 ISMCTS。仅训练用，实战 AI 不用。

---

## AI API（observation-only）

`platform/ai_service/` 提供 REST API：调用方只传 `action_id` + 公开事件，session 内部自维护 state，响应只有 `action_id` + 元数据。端点：`POST /ai/sessions[/{id}/observe|decide]` / `DELETE /ai/sessions/{id}`。

**双重身份**：

1. **结构性证明 AI 不读真值**——session 用独立 seed 初始化，public 字段每步从消息流的 `public_snapshot` 重建（`test_public_snapshot_round_trip`），hidden 字段每步重新采样（`test_session_hidden_fields_resampled`），与 truth 数值独立但行为等价（`test_api_belief_matches_selfplay` / `test_api_mcts_policy_invariance`）
2. **接入第三方**——任何外部系统实现"事件 → API"翻译层即可使用 AI，无需共享 state 代码或嵌入 C++ 引擎

API 文档：[AI_API.md](guide/AI_API.md)。每游戏的动作编码 + 事件格式：`docs/games/<game>_api.md`（新游戏必写）。

---

## 训练

```
selfplay → 收集样本 → 训练网络 → gating eval → 更新 best model → 循环
```

全流程 C++（selfplay / 搜索 / 求解 / eval），Python 只做训练循环和网络训练，配置驱动。

**Value Head**：N 维向量（N = num_players），perspective-relative（`values[0]` = 当前玩家）。MCTS backup 时旋转回绝对玩家顺序。ONNX evaluator 兼容旧的 2p 标量 value head（自动展开为零和 2 维）；3p+ 标量直接报错。

**训练增强**（每项详见 [Guide §9](guide/GAME_DEVELOPMENT_GUIDE.md#9-训练可选特性)）：

| 机制 | 说明 |
|------|------|
| 启发式引导 | `heuristic_picker` 三段式 schedule（hold → 衰减 → 0），早期由启发式驱动 selfplay |
| 辅助训练信号 | `auxiliary_scorer` 提供胜负外的 score head |
| 动作过滤 | `training_action_filter` 裁剪垃圾动作，概率衰减到 0；`legal_mask` 始终为完整集 |
| 温度 schedule | 分段线性衰减 |
| Dirichlet 噪声 | 根节点注入，可限制前 N 步 |
| 超时裁决 | `adjudicator` 在 `max_game_plies` 后判胜负 |
| Peek | 训练早期全知状态搜索，仅训练用 |

**评估**：每 `--eval-every` 步触发，含 benchmark eval（`heuristic_constrained` / `heuristic_free` / 指定 ONNX）+ gating（latest vs best，胜率 ≥ 阈值更新 best）。N 人游戏 candidate 轮坐每个座位以消除偏差。

---

## Web 前端

前端通过 HTTP API 与引擎交互，读 `state_serializer` 渲染、读 `action_descriptor` 理解动作。

**交互设计原则**（详见 [WEB_DESIGN_PRINCIPLES.md](guide/WEB_DESIGN_PRINCIPLES.md)）：自然交互（不要每动作一按钮）、空间锚定（固定区域固定位置和尺寸）、动作动画（每动作有 `describeTransition`）、视觉可辨。

**通用布局**：上方公共游戏区 + 信息栏（回合/胜率/AI 提示）+ 录像窗口；下方玩家区域（2p 左右，3-4p 网格）。general 层提供，游戏前端只填充内容。

**多人模式**：`players.max > 2` 的游戏侧边栏显示人数 + 座位选择。AI 连续落子直到轮到人类。

**高级操作**（general 层统一实现）：悔棋、替对手落子（隐藏信息游戏可 `disableForce: true` 关闭）、智能提示、胜率预估、对局中显示录像栏。隐藏信息游戏须 `showWinrateDefault: false`（胜率/失误标记基于 root values，含真实隐藏状态）。

**Web AI 配置**（`config/web.json`，可选）：`ai_use_action_filter` / `analysis_simulations` / `difficulty_overrides` / `tail_solve`。Peek 模式仅训练用，Web 不支持。详见 [配置参考](guide/CONFIG_REFERENCE.md#webjson--web-平台配置)。

**AI Pipeline**：每个 human-to-play 局面只跑一次 MCTS（precompute），结果两用——智能提示返回 + 落子后读 `action_values[chosen]` 算掉分。AI 自己的决策是另一次独立 MCTS。详见 [Web 开发指南](guide/WEB_DEVELOPMENT_GUIDE.md#8-ai-pipeline-与动作分析)。

**录像回放**：仅专家难度可用，结束后进入回放，每帧附带掉分（≥5% 失误、≥10% 严重失误）。`platform/tools/eval_model.py` 可独立生成录像。

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
| `belief_tracker` | 有隐藏信息或物理随机（ISMCTS root 采样的来源） |
| `public_event_extractor` / `applier` / `public_state_applier` | 隐藏信息游戏的消息流双向应用 |
| `initial_observation_extractor` / `applier` | 隐藏信息游戏的开局可见信息 |

完整字段说明见 [CONFIG_REFERENCE.md](guide/CONFIG_REFERENCE.md)。

---

## 新游戏开发步骤

1. 定义状态结构（继承 `CloneableState<T>`）
2. 实现规则（`legal_actions` / `do_action_fast` / `undo_action`，可选 `do_action_deterministic`）
3. 实现特征编码器（`encode_public` + `encode_private(p)`）
4. 写 `register.cpp` 组装 GameBundle
5. 写 `game.json` 配置
6. 加入 setup.py（必须！否则编译过但 register 不上）+ CMake
7. 写 Web 前端
8. 跑 `pytest tests/<game>/` 全绿
9. 跑训练、看日志、调参

参考实现按接入模式分类：

| 游戏 | 展示什么 |
|------|---------|
| TicTacToe | 最小闭环；不需要任何 optional 组件 |
| Quoridor | 完全信息确定游戏；heuristic_picker / tail_solver / adjudicator / auxiliary_scorer / training_action_filter 全配齐 |
| Splendor | 对称随机 + 非对称隐藏；`seen_cards` 增量 + `deck_flip` / `self_reserve_deck` 事件 + `do_action_deterministic` 占位符 |
| Azul | 纯对称物理随机；belief_tracker 只驱动 `randomize_unseen`，无 `hash_private_fields` |
| Love Letter | 非对称隐藏 + tracker 精确知识；`known_hand_[]` 跟踪 Priest 偷看 / Baron / King 后的确定信息 |
| Coup | **自定义 randomize_unseen** 范例：claim/challenge 历史驱动加权联合采样，避免诈唬游戏的 uniform 退化均衡 |

详见 [Guide §13 完整 Checklist](guide/GAME_DEVELOPMENT_GUIDE.md#13-完整-checklist)。

---

## 测试

**两层架构**：

- **`tests/framework/`** —— 框架不变量，跑在固定 3 游戏 matrix（`FRAMEWORK_GAMES = ["quoridor", "azul", "loveletter"]`）上，最小完备覆盖确定/对称随机/非对称隐藏 × 2p/2-4p × tail solver / belief tracker 等结构特征
- **`tests/<game>/`** —— 每游戏完整验收清单，包含 `TestRuleInvariants` 用 `run_random_episode_states` 驱动随机对局并断言**该游戏自己的守恒律**（token / 卡 / 棋子总量、容量上限等）

新游戏 ready = `pytest tests/<game>/` 一次全绿。隐藏信息游戏额外要求 `test_tracker_consistent_with_truth` + `test_ismcts_samples_respect_tracker`。

接入流程：从最相近的现有游戏复制 `tests/<game>/test_checklist.py`，改 `GAME = "..."`，根据测试失败迭代。详见 [新游戏验收测试指南](guide/NEW_GAME_TEST_GUIDE.md)。

---

## 框架局限性

本框架基于 AlphaZero 范式（MCTS + 神经网络），以下场景不适合或天花板有限：

1. **需要混合策略均衡的游戏**（如德州扑克）—— AlphaZero 训练确定性策略，无法收敛到精确混合 Nash。推荐 CFR / DeepCFR 系算法
2. **决策依赖长历史序列**（如 Hanabi）—— 当前 encoder 是状态快照；超出 belief tracker 能编码的范围需要 RNN/Transformer 序列建模
3. **动作空间组合爆炸**（如斗地主，27,000+ 出牌组合）—— 固定 `action_space` 既稀疏又低效，需分层动作解码（参考 DouZero）
4. **卡牌构筑（deck construction）**（如万智牌）—— 框架只解决"给定牌组怎么打"，构筑是独立 meta-game
5. **稀疏奖励 + 长对局**（>200 步）—— 仅终局 z_values 提供训练信号，value head 需要从终局一步步往前 bootstrap，游戏越长训练越慢。症状是"训练 plateau / value head 长期噪声"，不是 crash。粗略分档：<30 步无影响；30–150 步可行（Quoridor ~50-80、Splendor ~30-60、Azul ~120 都训练良好）；150–200 步多半需要 `auxiliary_scorer` + `heuristic_picker` 配合才跑得通；>200 步框架不适合。`AuxiliaryScorer` 是 reward shaping 性质的缓解，TD(λ) bootstrapping 是更系统的方向，未实现
6. **ISMCTS strategy fusion**（算法层固有限制）—— 对手节点会"看到"当前玩家 tracker 锁死的已知信息，把本应跨 info set 求期望的决策当成完全信息求解。在 LL Priest+Guard / Coup challenge / Werewolf 查验后发言这类"一方确定知识 + 另一方即时推理"高频场景下，AI 强度有天花板。根治需要 nested ISMCTS / subgame resampling，会破坏 DAG 共享 + 搜索吞吐降一个量级，未做
7. **仅 CPU 自博弈**——当前网络规模（~1M 参数，CPU 单次推理 ~200μs）GPU batch inference 的 IPC 固定开销没有摊销价值。网络规模提到 ~5M+ 参数（小型 ResNet / Transformer）后才值得加 GPU

**关于裸 PPO**：回合制桌游决策频率低、分支因子有限，正是 MCTS 强项。裸 PPO 更适合实时游戏（星际、Dota），不在本框架目标范围。
