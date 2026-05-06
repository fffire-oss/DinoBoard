# DinoBoard Bug 排查报告

日期：2026-05-06

本文记录本次围绕 Azul AI、ISMCTS、隐藏信息建模、Web AI 配置排查后仍需要处理的问题。已经确认为设计选择的内容不再列为 bug。


## 严重 / 高优先级

### 1. Azul 的 public hash 包含隐藏的袋子和盒盖顺序

相关文件：

- `games/azul/azul_state.cpp`
- `games/azul/azul_net_adapter.cpp`

`AzulState::hash_public_fields()` 把 `bag` 和 `box_lid` 的完整有序向量写入 public hash。Azul 玩家知道剩余瓷砖组成，不知道未来抽取顺序。`AzulBeliefTracker::randomize_unseen()` 每次 MCTS 模拟都会 shuffle `bag`，因此同一个可见局面、同一组袋子计数，会生成不同 public hash。

影响：

- DAG 节点被袋子随机顺序打碎，几乎无法跨 determinization 复用。
- MCTS 会表现得异常快但深度不足，接近浅层一步搜索。
- Encoder 只编码 bag 各色计数，不编码顺序，导致 hash scope 和 feature scope 不一致。

建议修复：

- 对 `bag` 和 `box_lid` 按颜色计数 hash，不按 vector 顺序 hash。
- 明确 `box_lid` 顺序是否隐藏；如果隐藏，也要按 multiset 处理。
- 如果 legacy `state_hash(false)` 被当作可见状态 hash 使用，也应同步移除隐藏顺序。
- 增加回归测试：同一可见状态 + 相同 bag multiset + 不同 bag 顺序，必须得到相同 perspective hash 和相同 features。

### 2. Love Letter encoder 会把 tracker 知识泄漏到非拥有者视角

相关文件：

- `games/loveletter/loveletter_net_adapter.cpp`

`LoveLetterFeatureEncoder::encode_private()` 的注释写明：只有请求编码的 `player` 等于 tracker perspective 时，才能使用 tracker 知识。但实现只判断 `tracker_ != nullptr`，随后就对对手使用 `tracker_->known_hand(pid)`。

MCTS 深入搜索时，节点会按当前行动玩家视角编码。如果根玩家的 tracker 知道某个对手的手牌，这个知识可能被注入到另一个玩家的决策节点特征里。这属于信息泄漏，会偏置搜索。

建议修复：

- 给 tracker 暴露 `perspective_player_`，或提供查询接口。
- 只有当 `player == tracker.perspective_player()` 时才允许使用 `known_hand()`。
- 增加测试：根玩家知道某个对手手牌后，从第三方玩家视角编码，确认该知识不会出现。

### 3. Love Letter 的 hash 没包含 belief 派生的已知手牌信息

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

### 4. Selfplay 和 arena 对所有视角共用一个 belief tracker

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

### 5. Web 隔离 AI 会话没有继承 `tail_solve` 配置

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

## 中优先级

### 6. MCTS 的 legal-action fallback 会静默掩盖 hash-scope bug

相关文件：

- `engine/search/net_mcts.cpp`

当复用到的 DAG 节点里有当前 world 不合法的边时，MCTS 会在当前合法边中重选。如果完全没有交集，就 break 并使用 `value_model.terminal_values(*sim_state)`。对非终局状态，这通常返回 0。

这会把 hash 不完备 bug 静默变成搜索截断和错误 backup。

建议修复：

- expanded DAG node 的 legal actions 与当前状态不一致时直接抛出明确错误。
- 至少暴露硬统计 / warning，并让测试失败。
- 增加测试确保 hash scope 能决定 legal action set。

### 7. Splendor 的 public hash 包含 `rng_salt`

相关文件：

- `games/splendor/splendor_state.cpp`

`SplendorState::hash_public_fields()` 把 `rng_salt` 加进 hash。这个字段玩家不可见，encoder 也不编码。单次 search 中它大多是常量，影响通常小于 Azul，但仍违反 public/private hash 契约。

建议修复：

- 从 `hash_public_fields()` 移除 `rng_salt`。
- 隐藏 RNG 只保留在显式请求 hidden RNG 的 legacy full-state hash 路径里。

### 8. ONNX / MCTS 数值 fallback 会把坏输出静默均匀化

相关文件：

- `engine/search/net_mcts.cpp`
- `engine/infer/onnx_policy_value_evaluator.cpp`

多个异常输出会被转成 uniform prior 或 0 value：

- evaluator priors 全非正或 sum 为 0 -> uniform prior。
- ONNX logits 非法，或 mask 后没有 finite legal logits -> uniform prior。
- leaf value 维度不足 -> 缺失玩家 value 按 0 backup。

这违反项目的 no-silent-degradation 原则。

建议修复：

- invalid logits、invalid priors、value dimension 错误时直接抛异常。
- 增加模型输出验证测试。

### 9. 非法或未注册人数变体可能回退到 base game

相关文件：

- `platform/game_service/sessions.py`

如果请求的 `{game_id}_{num_players}p` 变体未注册，而 base game 存在于 `GAME_CONFIGS`，`actual_id` 可能保持 base game。这会导致实际开局人数和请求人数不一致。

建议修复：

- 用注册表和 game metadata 验证请求的 `num_players`。
- 请求变体不可用时直接失败，不要静默回退。

## 低优先级 / 诊断项

### 10. `get_ai_action()` stats 缺少有用的 tail-solve 诊断信息

相关文件：

- `bindings/py_engine.cpp`

Python stats 只暴露 `tail_solved` 和 `tail_solve_value`，没有暴露 `tail_solve_attempted`、`tail_solve_completed`、耗时等信息。另外 `simulations` 被赋值两次。

建议修复：

- 补充缺失的 tail-solve stats。
- 移除重复的 `simulations` 赋值。

### 11. `select_action_from_visits()` 在坏 stats 下会静默选择第一个合法动作

相关文件：

- `engine/search/net_mcts.cpp`
- `bindings/py_engine.cpp`

如果 visits/actions 不匹配，或 temperature 采样权重全为 0，`select_action_from_visits()` 会返回 `fallback_action`；Web 传入的是 `legal[0]`。这会隐藏上游 MCTS stats 异常。

建议修复：

- malformed visit stats 直接抛异常。
- fallback 只用于明确设计的 no-search 模式。

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

1. 修复 Azul bag/box hash 为 multiset，并增加回归测试。
2. 修复 Love Letter tracker-perspective 泄漏，以及 hash/feature 对齐问题。
3. 将 selfplay/arena 改成 per-perspective belief trackers。
4. 把 MCTS hash-scope legal-action fallback 从静默截断改为硬失败。
5. 修复 Web 隔离 AI 会话没有继承 `tail_solve` 配置的问题。
6. 从 Splendor public hash 移除 `rng_salt`。
7. 收紧 ONNX/MCTS 非法输出处理，改成抛异常而不是 uniform fallback。
8. 明确 Quoridor action filter 下录像/分析的掉点定义。
