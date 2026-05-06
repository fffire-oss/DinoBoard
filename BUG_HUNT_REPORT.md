# DinoBoard Bug 排查报告

日期：2026-05-06

本文记录本次围绕 Azul AI、ISMCTS、隐藏信息建模、Web AI 配置排查后仍需要处理的问题。已经确认为设计选择的内容不再列为 bug。


## 已修复

### 1. Azul 的 public hash 包含隐藏的袋子和盒盖顺序

相关文件：

- `games/azul/azul_state.cpp`

状态：已修复。

修复内容：

- `AzulState::hash_public_fields()` 不再 hash `bag` / `box_lid` 的 vector 顺序。
- 改为按 5 色计数 hash `bag` / `box_lid`，保留公开可推导的 composition，移除无人知道的 draw order。

影响：

- 同一可见局面 + 相同袋子/盒盖 multiset 不再因为随机顺序不同而分裂 DAG 节点。
- Encoder 只编码 bag 各色计数，修复后 hash scope 和 feature scope 对齐。

### 2. Splendor 的 public hash 包含 `rng_salt`

相关文件：

- `games/splendor/splendor_state.cpp`

状态：已修复。

修复内容：

- 从 `SplendorState::hash_public_fields()` 移除 `rng_salt`。
- 隐藏 RNG 仍保留在 legacy `state_hash(include_hidden_rng=true)` 路径，用于 full-state/debug 语义。

### 3. Tail solver 采用阈值与文档口径不一致

相关文件：

- `engine/search/net_mcts.cpp`
- `engine/search/tail_solver.cpp`

状态：已修复。

问题：

- `tail_solver.cpp` 内部用 `value > 0.5f` 将结果分类为 `kProvenWin`。
- `net_mcts.cpp` 采用结果时只检查 `outcome == kProvenWin`，导致实际采用阈值也是 `value > 0.5f`。
- 文档和设计口径是 `value >= 1.0f` 才采用 tail-solve 结果。

修复内容：

- `NetMcts::search_root()` 采用 tail-solve 结果时新增 `ts.value >= 1.0f` 条件。
- `tail_solver.cpp` 的 `0.5f` 仍只作为内部 outcome 分类阈值，不再决定是否替代 MCTS。

### 4. MCTS 的 legal-action fallback 会静默掩盖 hash-scope bug

相关文件：

- `engine/search/net_mcts.cpp`

状态：已修复。

问题：

- 复用到的 DAG 节点如果含有当前 sampled world 不合法的边，旧逻辑会在当前合法边里重选。
- 如果完全没有交集，旧逻辑会截断 simulation 并用 `terminal_values()` backup；非终局状态下这通常等价于静默 0 value。

修复内容：

- expanded DAG node 的边必须在当前状态合法；发现不合法 action 时直接抛出 `MCTS: DAG node legal-action mismatch`。
- 非终局 expanded node 没有边、无法选择边、非终局无合法动作等异常状态都会硬失败，不再当作终局处理。

### 5. ONNX / MCTS 数值 fallback 会把坏输出静默均匀化

相关文件：

- `engine/search/net_mcts.cpp`
- `engine/infer/onnx_policy_value_evaluator.cpp`

状态：已修复。

问题：

- evaluator priors 全非正或 sum 为 0 时会被改成 uniform prior。
- ONNX logits 非法、legal mask 和 legal actions 不一致、mask 后没有 finite legal logits 时会被改成 uniform prior。
- leaf value 维度不足时，缺失玩家 value 会按 0 backup。
- value 输出中出现 NaN/Inf 时会被 clamp 或继续流入搜索。

修复内容：

- MCTS 对 priors 做 finite / 非负 / 正质量校验，坏输出直接抛错。
- `masked_softmax()` 对 logits、legal mask、legal action index、finite softmax 权重做硬校验，不再 uniform fallback。
- leaf values 必须覆盖所有玩家且全部 finite，否则搜索失败。
- ONNX value 输出必须覆盖 `num_players`；scalar / partial vector、NaN、Inf 都直接抛错。

### 6. 非法或未注册人数变体可能回退到 base game

相关文件：

- `platform/game_service/sessions.py`

状态：已修复。

问题：

- 请求 `{game_id}_{num_players}p` 未注册时，旧逻辑可能继续创建 base game session。
- 这会导致请求人数和实际游戏人数不一致。

修复内容：

- `create_session()` 先验证 base `game_id` 必须存在。
- 请求非 2 人变体时，目标 variant 必须已经注册，否则直接 `ValueError`。
- 创建 session 后再次校验 `gs.num_players == num_players`，不允许静默人数漂移。

### 7. `select_action_from_visits()` 在坏 stats 下会静默选择第一个合法动作

相关文件：

- `engine/search/net_mcts.cpp`
- `bindings/py_engine.cpp`

状态：已修复。

问题：

- `actions/visits` 不匹配、visits 全 0、温度采样权重全 0 时，旧逻辑会返回调用方传入的 `fallback_action`。
- Web / selfplay / arena 传入的是 `legal[0]`，会把上游 MCTS stats 异常伪装成合法但任意的第一手。

修复内容：

- `select_action_from_visits()` 现在对空 actions、size mismatch、负 visit、全 0 visit、非 finite 权重直接抛异常。
- 保留函数签名中的 `fallback_action` 只是为了兼容现有调用点；实现不再使用它。

## 严重 / 高优先级

### 1. Love Letter encoder 会把 tracker 知识泄漏到非拥有者视角

相关文件：

- `games/loveletter/loveletter_net_adapter.cpp`

`LoveLetterFeatureEncoder::encode_private()` 的注释写明：只有请求编码的 `player` 等于 tracker perspective 时，才能使用 tracker 知识。但实现只判断 `tracker_ != nullptr`，随后就对对手使用 `tracker_->known_hand(pid)`。

MCTS 深入搜索时，节点会按当前行动玩家视角编码。如果根玩家的 tracker 知道某个对手的手牌，这个知识可能被注入到另一个玩家的决策节点特征里。这属于信息泄漏，会偏置搜索。

建议修复：

- 给 tracker 暴露 `perspective_player_`，或提供查询接口。
- 只有当 `player == tracker.perspective_player()` 时才允许使用 `known_hand()`。
- 增加测试：根玩家知道某个对手手牌后，从第三方玩家视角编码，确认该知识不会出现。

### 2. Love Letter 的 hash 没包含 belief 派生的已知手牌信息

相关文件：

- `games/loveletter/loveletter_state.cpp`
- `games/loveletter/loveletter_net_adapter.cpp`

`LoveLetterState::hash_private_fields(player)` 只 hash `player` 自己的手牌和当前抽牌。但 encoder 可能把 tracker 中的 `known_hand_` 编进该玩家视角的私有特征。

这意味着两个信息状态可能拥有相同 perspective hash，但网络输入不同：

- 状态 A：玩家知道对手手牌。
- 状态 B：玩家不知道对手手牌。

如果 policy/value 输入不同，它们不应该共享同一个 DAG 节点。

建议修复：

- 要么让 tracker 已知牌参与 MCTS 使用的信息集 hash。
- 要么在框架支持 belief-aware hashing 之前，先不要把 tracker 知识编码进 features。
- 更推荐做框架级接口，让 `IBeliefTracker` 能贡献 root / player 信息集 key。
- 增加 Love Letter 已知手牌场景的 hash/feature 对齐测试。

### 3. Selfplay 和 arena 对所有视角共用一个 belief tracker

相关文件：

- `engine/runtime/selfplay_runner.cpp`
- `engine/runtime/arena_runner.cpp`
- `bindings/py_engine.cpp`

Web `GameSession` 会为每个 perspective 创建独立的 AI view / tracker / encoder / evaluator。但 selfplay 和 arena 复用一个 `belief_tracker`，并在每个 ply 调用 `init(player, obs)`。

对 belief 会长期积累的游戏来说，行动玩家切换时会清空或覆盖其他视角之前积累的知识。这会让训练/eval 的 belief 行为偏离 Web/API，也会削弱隐藏信息游戏的训练质量。

建议修复：

- selfplay 和 arena 改成每个 perspective 一个 belief tracker，和 Web `GameSession` 对齐。
- 每次动作后，把 public event stream 按各自 perspective 喂给每个 tracker。
- 当前玩家行动时，用该玩家自己的 tracker 作为 `root_belief_tracker`。

### 4. Web 隔离 AI 会话没有继承 `tail_solve` 配置

相关文件：

- `platform/game_service/sessions.py`
- `platform/game_service/pipeline.py`
- `platform/game_service/routes.py`
- `bindings/py_engine.cpp`

`create_session()` 只在主 `GameSession` 上调用 `configure_tail_solve()`。但实际 AI 落子、precompute、hint fallback 都是在新建的隔离 `GameSession` 上执行。这些隔离会话默认 `ts_enabled_ = false`，所以 `web.json` 中的 `tail_solve` 配置经常不会影响真实 AI 搜索。

建议修复：

- 把 tail-solve 设置存入 `sess`。
- 所有用于 AI move、precompute、analysis、hint fallback 的隔离 `GameSession` 都调用 `configure_tail_solve()`。
- Web 响应暴露 tail-solve attempted/completed 等统计，方便确认配置确实生效。

## 低优先级 / 诊断项

### 5. `get_ai_action()` stats 缺少有用的 tail-solve 诊断信息

相关文件：

- `bindings/py_engine.cpp`

Python stats 只暴露 `tail_solved` 和 `tail_solve_value`，没有暴露 `tail_solve_attempted`、`tail_solve_completed`、耗时等信息。另外 `simulations` 被赋值两次。

建议修复：

- 补充缺失的 tail-solve stats。
- 移除重复的 `simulations` 赋值。

## 讨论项

### A. Quoridor 使用 action filter 时，录像里的掉点如何定义

相关文件：

- `games/quoridor/config/web.json`
- `platform/game_service/pipeline.py`

`ai_use_action_filter: true` 是有意设计，AI search 可以使用训练过滤动作空间。但这里有一个需要单独定义的问题：如果人类在录像中走了未被 filter 纳入搜索树的合法动作，分析中的 drop / 掉点应该怎么计算？

需要明确的语义：

- 掉点是相对于“完整合法动作空间”的最优动作计算，还是相对于“AI 过滤后的动作空间”计算？
- 如果人类动作不在过滤空间内，是否应临时对该动作做一次单独评估？
- 录像回放中展示的 best move 是否应该标注“过滤策略下的 best”还是“完整规则下的 best”？

建议处理：

- 保留 `ai_use_action_filter` 设计。
- 为录像/分析单独定义 drop 语义，避免 filtered search 和 full legal human move 混用导致数值误导。
- 如要分析人类未过滤动作，可增加单动作评估路径，而不是直接 fallback 到 best value。

## 建议修复顺序

1. 修复 Love Letter tracker-perspective 泄漏，以及 hash/feature 对齐问题。
2. 将 selfplay/arena 改成 per-perspective belief trackers。
3. 修复 Web 隔离 AI 会话没有继承 `tail_solve` 配置的问题。
4. 明确 Quoridor action filter 下录像/分析的掉点定义。
