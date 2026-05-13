# DinoBoard — 功能概览

给开发者的速查地图。详细实现见 [GAME_DEVELOPMENT_GUIDE.md](docs/guide/GAME_DEVELOPMENT_GUIDE.md)；
框架契约 + MCTS 算法见 [ALGORITHM_OVERVIEW.md](ALGORITHM_OVERVIEW.md)。

---

## 核心组件

一个完整的游戏 = `state.cpp` + `rules.cpp` + `visibility.cpp` +
`net_adapter.cpp` + `register.cpp` + `config/game.json` + `web/`

| 文件 | 接口 | 职责 |
|------|-----|------|
| `<game>_state.cpp` | `IGameState` | 状态、当前玩家、终局；schema-driven `hash_field_slot` / `mask_field_slot` / `read_field_slot` / `write_field_slot` 分发器；`schema_ref()` 返回静态 schema |
| `<game>_visibility.cpp` | `viz::VisibilitySchema` | 字段 declare（name + 数据 shape + base viz tensor），是 hash / encoder / snapshot scope 的单一事实源 |
| `<game>_rules.cpp` | `IGameRules` | 合法动作、`do_action_fast`（含 viz 维护，`reveal_slot` / `reveal_slot_to` / `reset_to_base`；不支持 undo——MCTS / selfplay / arena / web 都丢弃用完的 state）；想接 tail solver 才额外实现 `do_action_deterministic` + `undo_action`（这两个配对工作） |
| `<game>_net_adapter.cpp` | `IFeatureEncoder` + `IBeliefTracker` | encoder 入参锁 `const MaskedState&`（`encode_public` + `encode_private(p)` 拆分）；tracker(有非对称隐藏信息 / 需要 sim 入口对 viz=0 槽位 determinization 的游戏才需要;纯公开物理随机由 `do_action_fast` 中的 `sim_rng` 处理,不需要 tracker) |
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
- **rules 是 viz 的唯一 writer**：`do_action_fast` 在改业务字段的同时调
  `reveal_slot` / `reveal_slot_to` / `reset_to_base` 维护 `state.viz_`
- **walker 一次产出 MaskedState**：`make_masked_state(state, schema,
  perspective)` 按 schema 遍历每槽——viz=1 复制真值，否则写
  `kPlaceholder` sentinel。snapshot（GT 端序列化）/ hash（sim 内每步）/
  encoder（sim 内新节点）三家共用同一份 MaskedState，结构性对齐
- **接口入参锁 `const MaskedState&`**：`IFeatureEncoder` /
  `IPolicyValueEvaluator` 的入参签名钉死 MaskedState——即使有人把 truth
  state 传进来，`encode(...)` helper 也会先 `make_masked_state` 把
  hidden 槽位换成 placeholder，子类 override 拿不到 raw state
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
- **每个 descent 步只 mask 一次**：`make_masked_state` 出来的 MaskedState
  喂给（hash 和）encoder，同一份副本不重复遍历
- **Encoder 对齐 hash scope**：encoder 拆分成 `encode_public` +
  `encode_private(p)`，结构性禁止读其他玩家 private（`test_encoder_respects_hash_scope`
  守护）

主要配置：`simulations` / `c_puct` / `temperature`（支持 schedule）/
Dirichlet 噪声。

---

## 随机性与隐藏信息

物理随机和信息不对称在 ISMCTS 里**统一处理**：

- Root 采样吞掉所有未来随机
- 观察者**可见**的后果通过 schema 公开槽位（mask=1）的差异自然分叉成不同
  hash 节点
- 观察者**不可见**的后果在 MaskedState 里全是 placeholder、对所有 sim 一
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

| 机制 | 说明 |
|------|------|
| 启发式引导 | `heuristic_picker` 三段式 schedule（hold → 衰减 → 0），早期由启发式驱动 selfplay |
| 辅助训练信号 | `auxiliary_scorer` 提供胜负外的 score head |
| 动作过滤 | `training_action_filter` 裁剪垃圾动作，概率衰减到 0；`legal_mask` 始终为完整集 |
| 温度 schedule | 分段线性衰减 |
| Dirichlet 噪声 | 根节点注入，可限制前 N 步 |
| 超时裁决 | `adjudicator` 在 `max_game_plies` 后判胜负 |

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

完整字段说明见 [CONFIG_REFERENCE.md](docs/guide/CONFIG_REFERENCE.md)。

---

## 新游戏开发步骤

1. 定义状态结构（继承 `CloneableState<T>`）
2. 写 `<game>_visibility.cpp`：declare 每个字段 name + 数据 shape +
   base viz tensor（`all_public` / `owner_only_first_axis` / `all_hidden`
   等 builder 覆盖大部分情形）；schema 是 hash / encoder / snapshot scope
   的单一事实源
3. 实现规则（`legal_actions` / `do_action_fast`；`do_action_fast` **不支持 undo**，
   `UndoToken` 是和 `do_action_deterministic` 共享签名的 vestige，不要在
   `do_action_fast` 里 push undo_stack / 拍快照）；要接 tail solver 才额
   外实现 `do_action_deterministic` + `undo_action` 配对（前者 freeze 隐
   藏抽牌，后者承担恢复职责）；`do_action_fast` 里同时调
   `reveal_slot` / `reset_to_base` 维护 `state.viz_`
4. 在 state 实现 schema-driven 分发器：`hash_field_slot` /
   `mask_field_slot` / `read_field_slot` / `write_field_slot` /
   `schema_ref`（按 schema 列字段答 typed value）
5. 实现特征编码器（`encode_public` + `encode_private(p)`，入参
   `const MaskedState&`，读 placeholder 分流）
6. 写 `register.cpp` 组装 GameBundle
7. 写 `game.json` 配置
8. **在 `games/manifest.json` 追加一条**：`{ "id": ..., "enabled": true, "framework_whitelist": <bool>, "capabilities": [...], "sources": [...] }` —— 这是 **CMake 编译 / setup.py 编译 / `engine.available_games()` / web 列表 / framework 测试矩阵** 这五层的唯一事实源；漏了这一步 = 编译过但 register 不上 = web 看不见 = 测试不覆盖。临时下线一个游戏只要把 `enabled: false`（源码留在 disk 上但所有层都跳过它）
9. 写 Web 前端
10. 跑 `pytest tests/<game>/` 全绿
11. 跑训练、看日志、调参

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
| Splendor | **端到端 walker 化参考实现**：reserve owner-only viz、`mask_field_slot` 走 COW shared_ptr 一次 detach、snapshot 走 `serialize_public(MaskedState)` walker 路径、`do_action_deterministic` 用 `forced_draw_override = -2` 占位符 |
| Azul | 纯对称物理随机；schema 全 all_public（袋子和 box_lid 在 schema 里以 per-color counts 体现），不注册 `belief_tracker`——sim 入口没东西可 determinize，物理随机走 `do_action_fast` 里 `sim_rng` 即时抽 |
| Love Letter | 非对称隐藏 + viz reveal 槽位承载确定信息（rules 通过 `reveal_slot_to` / `swap_slot_owned` 写入），tracker stateless 只做剩余牌池均匀采样 |
| Coup | **自定义 randomize_unseen** 范例：claim/challenge 历史驱动加权联合采样，避免诈唬游戏的 uniform 退化均衡 |

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
2. **动作空间组合爆炸**（如斗地主，27,000+ 出牌组合）—— 固定
   `action_space` 既稀疏又低效，需分层动作解码（参考 DouZero）
3. **卡牌构筑（deck construction）**（如万智牌）—— 两层困难:
   (a) 框架只解决"给定牌组怎么打"，构筑本身是独立 meta-game;
   (b) 构筑动作空间是组合性的（从 N 张候选卡里选 K 张组成牌组,搜索空间
   `C(N, K)` 爆炸）,跟第 2 条的 action_space 限制叠加,固定动作槽位无法
   表达,需要分层 / 自回归动作解码
4. **只支持零和博弈**（如外交风云 Diplomacy 这类需要谈判 / 结盟 / 共同
   收益的游戏不适用）—— 整个训练栈在零和假设上构造:value head 强制
   `sum(values) ≈ 0`、selfplay/arena 用 candidate vs opponent 一对一对
   战、gating 只看胜率。在合作或一般和（general-sum）博弈里,"赢"没有
   单一定义,Nash 均衡可以多个、Pareto 前沿不唯一,纯 selfplay 容易收敛
   到不合作的"安全"策略。根治需要重新设计训练目标(joint value /
   coalition formation)和 evaluator(N 维独立 reward 而非 zero-sum
   constrained),不是参数微调能解决的
5. **不支持单人游戏 / solitaire**（如纸牌接龙、2048）—— 自博弈学习的
   bootstrap 依赖"双方差不多菜也能有胜有负"：早期网络随机走子，对手同样
   随机，胜负信号自然产生，z_values 给出有方差的训练目标。单人游戏没有
   对手，胜利条件靠规则定义（拼出特定牌型 / 达到分数阈值），随机策略下
   触发胜利的概率几乎为零，整局都是 z=-1，网络拿不到任何"哪种走法更接
   近赢"的梯度，训练根本起不来。根治需要 reward shaping（按距离目标的
   进度给分）或课程学习（从简化局面开始），不是本框架范畴。建议改用
   intrinsic motivation / curiosity-driven 系算法
6. **viz 是公开 god-view 函数,不支持"闭眼"环节**（如狼人杀夜晚阶段、密
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

7. **稀疏奖励 + 长对局**（>200 步）—— 仅终局 z_values 提供训练信号，
   value head 需要从终局一步步往前 bootstrap，游戏越长训练越慢。症状是
   "训练 plateau / value head 长期噪声"，不是 crash。粗略分档：<30 步无
   影响；30–150 步可行（Quoridor ~50-80、Splendor ~30-60、Azul ~120 都
   训练良好）；150–200 步多半需要 `auxiliary_scorer` + `heuristic_picker`
   配合才跑得通；>200 步框架不适合。`AuxiliaryScorer` 是 reward shaping
   性质的缓解，TD(λ) bootstrapping 是更系统的方向，未实现
8. **仅 CPU 自博弈**——当前网络规模（~1M 参数，CPU 单次推理 ~200μs）
   GPU batch inference 的 IPC 固定开销没有摊销价值。网络规模提到 ~5M+
   参数（小型 ResNet / Transformer）后才值得加 GPU

### 路径已知,未来可能补上(下一轮迭代候选)

这一组**有明确技术路线**,只是当前没做。如果未来某条被实现,记得回来删掉
对应条目。

9. **决策依赖长历史序列**（如 Hanabi）—— 当前 encoder 是状态快照；超出
   belief tracker 能编码的范围需要 RNN/Transformer 序列建模
10. **ISMCTS strategy fusion**（算法层固有限制）—— 对手节点会"看到"当前
    玩家 tracker 锁死的已知信息，把本应跨 info set 求期望的决策当成完全
    信息求解。在 LL Priest+Guard / Coup challenge / Werewolf 查验后发言
    这类"一方确定知识 + 另一方即时推理"高频场景下，AI 强度有天花板。根
    治需要 nested ISMCTS / subgame resampling，会破坏 DAG 共享 + 搜索吞
    吐降一个量级，未做。**部分缓解**：`opponent_selection="prior"`（Smooth-UCT
    风格）在非根对手节点用 policy 先验 multinomial sampling 替代 PUCT
    bandit，避免对手节点在每个 determinization 里都"贪婪最优"导致的全知
    偏置；可在 `game.json mcts_profiles.<selfplay|arena|eval>.opponent_selection`
    / `web.json mcts_profiles.<web_expert|web_casual|analysis>.opponent_selection`
    配置，默认 `"puct"` 保持向后兼容。根节点（轮到自己）始终走 PUCT。
11. **策略追逐 / 自博弈非传递循环**（如石头剪刀布的扩展型博弈）—— 当
    前 selfplay 只跟"latest vs best"对打、训练目标是击败当前 best。如
    果游戏存在"A 克 B、B 克 C、C 克 A"的非传递结构，AI 容易陷入局部循
    环：训出克制当前 best 的 A，下一轮 best 变 A、再训出克 A 的 C，循
    环往复，平均强度不上升。根治需要维护**对手池**（PSRO / fictitious
    self-play / league training）—— selfplay 同时跟历史多个 checkpoint
    对打、用 meta-solver 计算混合策略权重。当前 gating eval 只是一对一
    胜率比较，没有对手池机制，未实现

**关于裸 PPO**：回合制桌游决策频率低、分支因子有限，正是 MCTS 强项。裸
PPO 更适合实时游戏（星际、Dota），不在本框架目标范围。
