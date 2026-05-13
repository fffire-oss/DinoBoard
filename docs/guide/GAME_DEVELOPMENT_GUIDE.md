# 游戏开发指南

> 本文档事无巨细地描述了如何在 DinoBoard 平台上添加一个新游戏。
> 阅读后你将了解所有必须实现的接口、所有可选特性、配置文件格式、构建集成方式，以及常见踩坑。

---

## 目录

1. [总览](#1-总览)
2. [IGameState — 游戏状态](#2-igamestate--游戏状态)
3. [IGameRules — 游戏规则](#3-igamerules--游戏规则)
4. [IFeatureEncoder — 特征编码](#4-ifeatureencoder--特征编码)
5. [GameBundle — 组件注册](#5-gamebundle--组件注册)
6. [GameRegistrar — 注册模式](#6-gameregistrar--注册模式)
7. 配置文件已抽到独立文档：[CONFIG_REFERENCE.md](CONFIG_REFERENCE.md)（game.json + web.json）
8. [构建集成](#8-构建集成)
9. [训练可选特性](#9-训练可选特性)（Heuristic、TailSolver、Filter、AuxScorer、Adjudicator、Stats、Peek）
10. [隐藏信息与 Belief Tracker（含物理随机性）](#10-隐藏信息与-belief-tracker含物理随机性)（ISMCTS 根采样、Encoder 信息屏障、物理随机性）
11. Web 前端开发已独立成章，详见 [WEB_DEVELOPMENT_GUIDE.md](WEB_DEVELOPMENT_GUIDE.md)
12. [测试](#12-测试)（自动化测试套件、运行方式、接入新游戏）
13. [完整 Checklist](#13-完整-checklist)
14. [AI API 分离验收](#14-ai-api-分离验收--信息泄漏的唯一证明)

---

## 1. 总览

添加一个新游戏需要创建以下文件：

```
games/<your_game>/
├── <game>_state.h          # 游戏状态定义
├── <game>_state.cpp        # 游戏状态实现
├── <game>_rules.h          # 规则引擎头文件
├── <game>_rules.cpp        # 规则引擎实现
├── <game>_net_adapter.h    # 特征编码头文件
├── <game>_net_adapter.cpp  # 特征编码实现
├── <game>_register.cpp     # 注册到全局 GameRegistry
├── config/
│   └── game.json           # 训练和网络超参数
├── CMakeLists.txt          # CMake 构建文件
└── web/                    # Web 前端（玩家游玩 + 验收训练结果）
    ├── index.html
    ├── styles.css
    └── <game>.js
```

需要实现 **3 个核心类**（State、Rules、FeatureEncoder），外加 **1 个注册文件** 和 **1 个配置文件**。

所有 C++ 代码位于 `board_ai::<game_name>` 命名空间下。

---

## 2. IGameState — 游戏状态

**文件**：`engine/core/game_interfaces.h`

### 2.1 推荐继承方式

使用 CRTP 模板 `CloneableState<T>` 自动实现 `clone_state()` 和 `copy_from()`：

```cpp
#include "../../engine/core/game_interfaces.h"

namespace board_ai::mygame {

class MyGameState final : public CloneableState<MyGameState> {
 public:
  // === 必须实现的方法 ===
  StateHash64 state_hash() const override;        // 全字段 hash,tail solver / debug 用
  void hash_field_slot(Hasher& h, const std::string& name,
                       const std::vector<int>& idx) const override;
  // (隐藏信息游戏额外:read_field_slot / write_field_slot / mask_field_slot)
  int current_player() const override;
  bool is_terminal() const override;
  int num_players() const override;
  int winner() const override;
  void reset_with_seed(std::uint64_t seed) override;

  // === 可选覆盖 ===
  // int first_player() const override;       // 默认返回 0

  // === 游戏数据 ===
  int current_player_ = 0;
  int winner_ = -1;
  bool terminal = false;
  int move_count = 0;
  std::array<int, 2> scores{};
  // ... 你的棋盘数据 ...

  // === Undo 支持 ===
  struct UndoRecord { /* 存储 do_action 前的快照 */ };
  std::vector<UndoRecord> undo_stack;
};

}  // namespace board_ai::mygame
```

### 2.2 各方法详解

#### `state_hash() -> StateHash64`

返回当前状态的 64 位哈希(全字段)。**仅用于 tail solver 转置表和
调试**——ISMCTS 不用这个函数做节点 keying。节点 keying 走框架内置的
`state_hash_for_perspective(p)`(`engine/core/schema_hash.h`):按
visibility schema 序遍历每个 slot,只对 `viz[..., p] = 1` 的槽位调
游戏侧实现的 `hash_field_slot(h, name, idx)`,自动叠 `step_count`。

**`state_hash()` 要求**:
- 相同状态(含所有 hidden)必须返回相同哈希
- 内容覆盖完整,因为 tail solver 在解析路径上会同 hash 共享子树

**`hash_field_slot(Hasher& h, name, idx)` 要求**:
- 一个 slot 一次 `h.add(...)`,内容 = 该 slot 的**真值**
- schema 里声明的所有字段都要在 `if (name == "...")` 分支里覆盖
- viz=0 的 slot 走框架自动写入的 `kHiddenHashSentinel`,**不要**在
  `hash_field_slot` 里自己判 viz
- **不要** mix `idx` —— framework 在调用 `hash_field_slot` 之前
  已经把 `(field_pos, idx[])` mix 进 hash 了。重复 mix idx 不算 bug,
  但是冗余;**漏掉 idx 也不再是 bug**(BUG-037 postmortem:framework
  现在守住 idx,游戏只管值)

**示例**:
```cpp
StateHash64 state_hash() const override {
  Hasher h;
  h.add(current_player_);
  h.add(move_count);
  for (auto cell : board) h.add(cell);
  return h.digest();
}

void hash_field_slot(Hasher& h, const std::string& name,
                     const std::vector<int>& idx) const override {
  if (name == "board") { h.add(board[idx[0]]); return; }
  if (name == "current_player") { h.add(current_player_); return; }
  // ...
}
```

#### `current_player() -> int`

返回当前行动玩家的索引（0-based）。

#### `is_terminal() -> bool`

游戏是否结束。true 时 `legal_actions()` 必须返回空。

#### `is_turn_start() -> bool`（可选，默认 true）

当前位置是否是一个玩家动作序列的起点。对于有多步动作的游戏（如 Splendor 的拿币→退币→选贵族），在子动作（退币、选贵族）阶段返回 false。可用于 UI 提示玩家正在进行子操作。简单游戏（每步恰好一个动作）不需要重写。

> **注意**：悔棋和替对手落子不依赖此标志，而是通过 `last_actor`（录像帧中的行动者）判断连续回合——无论是子动作还是跨轮连续行动，统一处理。

#### `num_players() -> int`

玩家总数。通常为 2，最多支持 4。

#### `winner() -> int`

- 游戏未结束或平局：返回 -1
- 有赢家：返回赢家的索引（0-based）

#### `reset_with_seed(uint64_t seed)`

重置为初始状态。使用 `sanitize_seed(seed)` 确保种子非零。

```cpp
void reset_with_seed(std::uint64_t seed) override {
  rng_salt = sanitize_seed(seed);
  current_player_ = 0;
  winner_ = -1;
  terminal = false;
  move_count = 0;
  scores = {0, 0};
  board.fill(kEmpty);
  undo_stack.clear();
}
```

### 2.3 UndoRecord 设计模式

**只在实现 `do_action_deterministic` + `undo_action`（tail solver 路径）时才需要 UndoRecord。** `do_action_fast` 不支持 undo，不要在 fast 路径里 push `undo_stack`——MCTS / selfplay / web 都丢弃用完的 state，没有恢复需求。

`do_action_deterministic` 执行前，将所有会被修改的字段快照保存到 `UndoRecord` 并压入 `undo_stack`；`undo_action` 弹出最后一条记录并恢复状态。Tail solver 在同一 state 上 do→recurse→undo 反复滚动。

```cpp
struct UndoRecord {
  int prev_player;
  int prev_winner;
  bool prev_terminal;
  int prev_move_count;
  std::array<int, 2> prev_scores;
  // ... 动作相关的还原信息 ...
};
```

**关键**：`undo_stack` 必须用 `std::vector`（不是 `std::stack`），因为 `CloneableState<T>` 依赖拷贝构造函数进行克隆。

### 2.4 CloneableState 的工作原理

`CloneableState<T>` 通过 CRTP 自动生成：
- `clone_state()` → `make_unique<T>(*static_cast<const T*>(this))`（拷贝构造）
- `copy_from(other)` → `*static_cast<T*>(this) = checked_cast<T>(other)`（拷贝赋值）

前提：你的状态类必须是**可拷贝构造和可拷贝赋值**的。如果使用了 `unique_ptr` 等不可拷贝的字段，需要自己写拷贝构造函数。

---

## 3. IGameRules — 游戏规则

**文件**：`engine/core/game_interfaces.h`

### 3.1 必须实现的方法

```cpp
class MyGameRules final : public IGameRules {
 public:
  bool validate_action(const IGameState& state, ActionId action) const override;
  std::vector<ActionId> legal_actions(const IGameState& state) const override;
  UndoToken do_action_fast(IGameState& state, ActionId action,
                           std::mt19937_64& rng) const override;
  void undo_action(IGameState& state, const UndoToken& token) const override;
};
```

#### `validate_action(state, action) -> bool`

检查 action 在当前状态下是否合法。用于 Python bindings 的输入验证。

#### `legal_actions(state) -> vector<ActionId>`

返回当前玩家所有合法动作。**terminal 状态必须返回空 vector**。

ActionId 是 `int32_t`，你需要设计一套编码方案将游戏动作映射为连续整数。例如：

```
TicTacToe：action = cell_index (0-8)
Quoridor：action ∈ [0, 209)
  - [0, 81)   → 移动棋子到 (row, col)
  - [81, 145)  → 放置水平墙 (row, col)
  - [145, 209) → 放置垂直墙 (row, col)
```

建议提供 `encode_xxx_action()` 和 `decode_xxx_action()` 辅助函数。

#### `do_action_fast(state, action, rng) -> UndoToken`

**核心热路径方法**。游戏作者重写的是 protected `do_action_fast_impl`（不是 public 的 `do_action_fast`）；后者是框架的 non-virtual wrapper，负责在调用 impl 之前自动 `state.step_count_ += 1`，作者既看不到 step_count_ 也无法忘记。MCTS 每次模拟调用上千次。`rng` 由调用方持有（runner / session），用于消费物理随机（抽牌、翻牌）；不依赖随机的游戏忽略即可。**`do_action_fast` 不支持 undo**（MCTS / selfplay / web 都丢弃用完的 state，没有 undo_stack push 必要）；要接 tail solver 才需额外重写 §3.2 的 `do_action_deterministic_impl` + `undo_action_impl` 配对。

**实现模板**：
```cpp
 protected:
  void do_action_fast_impl(IGameState& state, ActionId action,
                           std::mt19937_64& rng) const override {
    auto* s = &checked_cast<MyGameState>(state);

    // 1. 执行动作（不要 push undo_stack —— do_action_fast 不支持 undo）
    //    用 rng 消费物理随机：例如 std::uniform_int_distribution<>(...)(rng)
    // ... 修改棋盘状态、调用 reveal_slot / reset_to_base 维护 viz_ ...

    // 2. 更新游戏元数据
    s->move_count += 1;
    // ... 检查胜负 ...
    s->current_player_ = 1 - s->current_player_;

    // step_count_ 由框架 wrapper 自动 +1，无需手动 begin_step。
  }
```

#### `undo_action(state, token)`

弹出 undo_stack 顶部记录，恢复状态。

```cpp
void undo_action(IGameState& state, const UndoToken& token) const override {
  auto* s = &checked_cast<MyGameState>(state);
  if (s->undo_stack.empty()) return;
  const auto rec = s->undo_stack.back();
  s->undo_stack.pop_back();

  // 还原所有修改过的字段
  s->current_player_ = rec.prev_player;
  s->winner_ = rec.prev_winner;
  // ...

  (void)token;  // token.undo_depth 可用于一致性检查
}
```

### 3.2 可选实现：do_action_deterministic（残局求解需要）

#### `do_action_deterministic(state, action) -> UndoToken`

执行动作的**确定化版本**——随机结果 / 依赖隐藏信息的结果用**占位符**替代。仅当启用残局求解（Tail Solver）时才需要实现。基类默认实现直接委托给 `do_action_fast`。

**什么游戏要实现**：
- **纯确定游戏**（TicTacToe、Quoridor）：不用，默认实现就对
- **有物理随机的游戏**（Splendor 翻牌、Azul 袋中抽瓷砖）：要用 tail solver 就必须实现，已实现参考 `splendor_rules.cpp` / `azul_rules.cpp`
- **有隐藏信息的游戏**（Love Letter 看对手手牌、Coup 盲牌）：**理论上也可以实现**——只要开发者能给对手手牌 / 盲牌设计一个"不会破坏规则语义"的占位符，并在 `do_action_fast`（被调用 for 确定化路径时）正确处理占位符。做不出来就不要注册 `tail_solver`，框架不会强制

**实现原则**：
- 随机抽牌 / 翻牌：用占位符牌替代。规则引擎要识别占位符并标记不可用（不能打出、不能作为资源、不触发能力）
- 隐藏信息动作：比如 Guard 猜牌这种"动作效果依赖对手手牌"的，占位符手牌要让动作结果变为"明确可判的确定分支"（比如固定一个猜错分支，或者让此动作在 deterministic 路径里变成无效动作）
- 占位符不能破坏 `legal_actions` / `is_terminal` / `winner` 的正确性。Tail solver 会在这个确定化状态上完整展开搜索树

**选择不实现**：游戏注册时不设 `tail_solver` 即可，ISMCTS 的主搜索路径（selfplay / arena / GameSession）完全不依赖 `do_action_deterministic`。

### 3.3 物理随机游戏的主搜索流程

主搜索（selfplay / arena / API / GameSession）**不使用** `do_action_deterministic`。流程：

1. `do_action_fast` 直接消费真实 state 里的 RNG（抽牌从 deck top 弹、翻牌翻 tier deck 等），正常推进
2. MCTS 每次 sim 持独立 RNG;root 步调 `belief_tracker.randomize_unseen(sim_state, observer, sim_rng)` 把 viz=0 槽位采样成具体值,descent 期间 `do_action_fast` 仍可能消费同一个 sim_rng 处理物理随机(如 Azul 工厂 refill)。给定 sim 种子整个展开可复现
3. 不同 sim 采不同的世界,observer 能分辨的后继(如 Splendor 翻出的公开
   牌)通过 `state_hash_for_perspective` 差异(viz=1 公开槽位变化)自然
   分叉,observer 不能分辨的后继(如 opp 抽的私牌)在 observer 视角下
   viz=0,hash 不变,自然在同一节点上汇聚

`step_count_` 由框架的 `IGameRules` wrapper 在 `do_action_fast_impl` / `do_action_deterministic_impl` 之前自动 +1、`undo_action_impl` 之后自动 -1，**作者既看不到也无法直接维护**(字段 protected + `IGameRules` friend);开发者重写 `do_action_fast_impl` / `undo_action_impl` 时只需正确推进游戏字段和 RNG 状态。**如果你的游戏需要 tail solver，额外实现 §3.2 的 `do_action_deterministic_impl`**。详见 [ALGORITHM_OVERVIEW.md §9.7](../../ALGORITHM_OVERVIEW.md#97-物理随机--sampled-world)。

---

## 4. IFeatureEncoder — 特征编码

**文件**：`engine/core/feature_encoder.h`

### 4.1 必须实现的方法

Encoder 接口按 hash scope 拆成 public / private 两半，**结构性约束**和 walker 驱动的 `state_hash_for_perspective(p)` 完全对齐——encoder 输入是 `MaskedState`，slots 中 `viz[..., perspective]=0` 的位置已经被 framework 替换成 `kPlaceholder*`，encoder 物理上读不到 truth：

```cpp
class MyGameFeatureEncoder final : public IFeatureEncoder {
 public:
  int action_space() const override;          // 动作空间大小
  int feature_dim() const override;           // = public + private 特征总维度
  int public_feature_dim() const override;    // 公开特征维度
  int private_feature_dim() const override;   // 一名玩家的私有特征维度（对所有玩家相同）

  void encode_public(const MaskedState& state,
                     int perspective_player,
                     const IBeliefTracker* tracker,
                     std::vector<float>* out) const override;

  void encode_private(const MaskedState& state,
                      int player,
                      const IBeliefTracker* tracker,
                      std::vector<float>* out) const override;
};
```

**硬约束**（`tests/framework/test_encoder_respects_hash_scope.py` 守护）：
- `encode_public` / `encode_private` 都从 `MaskedState` 读字段；`viz[..., perspective]=0` 的 slot 已经是 `kPlaceholder*`，encoder 必须分支处理 placeholder，**禁止查询 viz、禁止访问任何玩家的 private 字段绕过 mask**
- `encode_private(p)` 由于 MaskedState 是按 `player` 视角构建的，其它玩家的 private slot 在 `MaskedState` 中天然是 placeholder——读它们结构上就是读 placeholder，不是泄露
- `tracker` 可能为 nullptr（游戏没有注册 belief_tracker 时）；tracker 上只能读公开衍生统计（如 Coup 的 claim history、Splendor 的多重集统计），不要把"perspective 自己的 private 知识"塞到 tracker 里——那应该走 `state.viz_`
- 不需要自己实现 `encode(...)`——基类提供默认实现，会自动 `make_masked_state` 一次、按 `[encode_public, encode_private(perspective)]` 顺序拼接，并填充 `legal_mask`。游戏直接 override `encode_public` + `encode_private` 即可

#### `action_space() -> int`

返回动作空间总大小（策略头的输出维度）。**必须和 game.json 中的 `action_space` 一致**。

#### `feature_dim() / public_feature_dim() / private_feature_dim() -> int`

总维度 = public + private 之和。**必须和 game.json 中的 `feature_dim` 一致**（game.json 只记录总维度）。完全公开游戏（TicTacToe、Quoridor）`private_feature_dim()` 返回 0、`encode_private` 是空实现。

**示例**（TicTacToe，public 28 维 + private 0 维）：
```cpp
// tictactoe_net_adapter.h
class TicTacToeFeatureEncoder final : public IFeatureEncoder {
 public:
  int action_space() const override { return 9; }
  int feature_dim() const override { return 28; }
  int public_feature_dim() const override { return 28; }
  int private_feature_dim() const override { return 0; }

  void encode_public(const IGameState& state, int perspective_player,
                     std::vector<float>* out) const override;
  void encode_private(const IGameState&, int, std::vector<float>*) const override {}
};

// tictactoe_net_adapter.cpp
void TicTacToeFeatureEncoder::encode_public(
    const IGameState& state, int perspective_player,
    std::vector<float>* out) const {
  const auto& s = checked_cast<TicTacToeState>(state);
  const int opp = 1 - perspective_player;
  // 9 格 × 3 通道（是我的、是对手的、是空的）
  for (int i = 0; i < 9; ++i) {
    out->push_back(s.board[i] == perspective_player ? 1.0f : 0.0f);
    out->push_back(s.board[i] == opp ? 1.0f : 0.0f);
    out->push_back(s.board[i] == kEmpty ? 1.0f : 0.0f);
  }
  // 1 标量：是否先手
  out->push_back(perspective_player == s.first_player() ? 1.0f : 0.0f);
}
```

**隐藏信息游戏的写法**：手牌、盲压牌、对手牌的私人确定知识（如 LL Priest 偷看后看到的 opp hand）这些都通过 `MaskedState` 的 `viz=1` 槽位读到——直接 `if (mstate.is_visible({...}, p)) ...` 即可，**不要**把这种 perspective-private 知识塞到 tracker 里。`encode_private` 拼自身手牌 + 通过 viz reveal 看到的 opp 槽位；`encode_public` 拼公开弃牌区、棋盘、当前玩家标记、tracker 公开知识聚合（如 Coup claim history、Splendor 多重集统计）。两个函数被默认 `encode()` 自动按顺序拼接成单一 flat tensor 喂给网络——网络架构不变，纯粹是代码层面的强约束。

### 4.2 视角处理

**不要旋转棋盘**。只做「我 / 对手」的特征交换，加一个 scalar 标识「我是先手 / 后手」（或等价的方向标记）。

原因详见 `docs/KNOWN_ISSUES.md` 第 4 条：棋盘旋转容易把和格子绑定的结构（例如 Quoridor 里墙的「挡哪两条边」语义）旋转错，而且训练看起来能跑、但有一方的策略永远学不好，这种 bug 非常难定位。网络自己可以学 P0/P1 的不对称，不需要我们帮它「归一化」视角。

```cpp
// 正确示范（写在 encode_public 里）：把 "我" 和 "对手" 的棋子都按
// perspective_player 来选，棋盘坐标保持不变
const int me = perspective_player;
const int opp = 1 - perspective_player;
for (int i = 0; i < kCells; ++i) {
  out->push_back(s.board[i] == me ? 1.0f : 0.0f);
}
for (int i = 0; i < kCells; ++i) {
  out->push_back(s.board[i] == opp ? 1.0f : 0.0f);
}
// 告诉网络 "我" 是谁
out->push_back(perspective_player == 0 ? 1.0f : 0.0f);
```

动作输出层面 `policy_action_ids` 直接用原始 `ActionId`，不做任何旋转映射。

### 4.3 标量特征归一化

将标量特征归一化到 [0, 1] 或 [-1, 1]：
```cpp
// 好
out->push_back(static_cast<float>(walls_remaining) / kMaxWalls);
out->push_back(static_cast<float>(move_count) / kMaxPlies);

// 差
out->push_back(static_cast<float>(walls_remaining));  // 原始值 0-10
```

### 4.4 合法掩码

不需要游戏自己实现——基类 `encode()` 默认实现会用 `legal_actions` 自动填充 `legal_mask`，game-specific 代码只关心 features。

---

## 5. GameBundle — 组件注册

**文件**：`engine/core/game_registry.h`

GameBundle 是一个聚合所有游戏组件的结构体。工厂函数返回一个 GameBundle 实例。

### 5.1 全部字段

| # | 字段 | 类型 | 必须 | 说明 |
|---|------|------|------|------|
| 1 | `game_id` | `string` | 是 | 唯一标识符，如 `"quoridor"` |
| 2 | `state` | `unique_ptr<IGameState>` | 是 | 初始状态（已调用 `reset_with_seed`） |
| 3 | `rules` | `unique_ptr<IGameRules>` | 是 | 规则引擎 |
| 4 | `value_model` | `unique_ptr<IStateValueModel>` | 是 | 通常用 `DefaultStateValueModel`。`terminal_values()` 必须零和 |
| 5 | `encoder` | `unique_ptr<IFeatureEncoder>` | 是 | 特征编码器 |
| 6 | `belief_tracker` | `unique_ptr<IBeliefTracker>` | 否 | 有隐藏信息或物理随机的游戏必须注册 |
| 7 | `public_event_extractor` | `PublicEventExtractor` | 否 | `(before, action, after, perspective) → PublicEventTrace { events, public_snapshot }`；`events` 仅供 tracker 增量更新 belief，`public_snapshot` 用来在 observer 端整体覆写公开字段 |
| 8 | `public_state_applier` | `PublicStateApplier` | 否 | **隐藏信息游戏必装**。配合 extractor 的 `public_snapshot` 反向写回 state。observer 路径上 `apply_observation` 不调 `do_action_fast` —— `begin_step → public_state_applier(snapshot) → tracker.observe_public_event(events)` —— 公开字段完全由 message 重建，viz=0 槽位不动（决策侧的 hash / encoder / sim 都不读 viz=0 真值，结构性杜绝泄漏）。Round-trip 测试见 `tests/framework/test_public_snapshot_round_trip.py`，决策侧不读 viz=0 的回归 sweep 见 `test_public_hash_excludes_internal_rng.py` |
| 9 | `initial_observation_extractor` | `InitialObservationExtractor` | 否 | 提取 perspective 的开局可见信息 |
| 10 | `initial_observation_applier` | `InitialObservationApplier` | 否 | 把 initial observation 填入 state（AI API 侧用） |
| 11 | `state_serializer` | `StateSerializer` | 否 | 状态序列化为 JSON（Web 前端需要;**也用于规则不变量测试,详见 §10.6**） |
| 12 | `action_descriptor` | `ActionDescriptor` | 否 | 动作语义描述（Web 前端需要） |
| 13 | `heuristic_picker` | `HeuristicPicker` | 否 | 启发式策略（heuristic guidance + eval benchmark） |
| 14 | `tail_solver` | `unique_ptr<ITailSolver>` | 否 | 残局求解器（通常用 `AlphaBetaTailSolver`） |
| 15 | `tail_solve_trigger` | `TailSolveTrigger` | 否 | 残局求解触发条件（未注册则 fallback 到 ply 阈值） |
| 16 | `episode_stats_extractor` | `EpisodeStatsExtractor` | 否 | 每局自定义统计 |
| 17 | `adjudicator` | `GameAdjudicator` | 否 | 超时判定胜负 |
| 18 | `auxiliary_scorer` | `AuxiliaryScorer` | 否 | 辅助训练信号 |
| 19 | `training_action_filter` | `TrainingActionFilter` | 否 | 训练时约束动作空间 |

### 5.2 各类型签名

```cpp
// 状态序列化：state → key-value map
using StateSerializer = std::function<AnyMap(const IGameState& state)>;

// 动作描述：action → key-value map
using ActionDescriptor = std::function<AnyMap(ActionId action)>;

// 启发式策略：返回带分数的动作列表，框架根据 scores 选择动作
using HeuristicPicker = std::function<
    HeuristicResult(IGameState& state, const IGameRules& rules, std::uint64_t rng_seed)>;

struct HeuristicResult {
  std::vector<ActionId> actions;
  std::vector<double> scores;
};

// 每局统计
using EpisodeStatsExtractor = std::function<
    std::map<std::string, double>(
        const IGameState& final_state,
        const std::vector<SelfplaySampleView>& samples)>;

// 超时判定
using GameAdjudicator = std::function<int(const IGameState& state)>;

// 辅助分数
using AuxiliaryScorer = std::function<float(const IGameState& state, int player)>;

// 训练动作过滤
using TrainingActionFilter = std::function<std::vector<ActionId>(
    const IGameState& state, const IGameRules& rules, const std::vector<ActionId>& legal)>;
```

---

## 6. GameRegistrar — 注册模式

**文件**：`<game>_register.cpp`

使用文件作用域的静态对象，在程序启动时自动注册游戏。

### 6.1 最小示例（TicTacToe）

```cpp
#include "../../engine/core/game_registry.h"
#include "mygame_state.h"
#include "mygame_rules.h"
#include "mygame_net_adapter.h"

namespace {

board_ai::GameRegistrar reg("mygame", [](std::uint64_t seed) {
  board_ai::GameBundle b;
  b.game_id = "mygame";

  auto s = std::make_unique<board_ai::mygame::MyGameState>();
  s->reset_with_seed(seed);
  b.state = std::move(s);

  b.rules = std::make_unique<board_ai::mygame::MyGameRules>();
  b.value_model = std::make_unique<board_ai::DefaultStateValueModel>();
  b.encoder = std::make_unique<board_ai::mygame::MyGameFeatureEncoder>();

  return b;
});

}  // namespace
```

### 6.2 完整示例（带所有可选特性）

参考 `games/quoridor/quoridor_register.cpp`，它注册了以下所有可选组件：

```cpp
board_ai::GameRegistrar reg("quoridor", [](std::uint64_t seed) {
  board_ai::GameBundle b;
  b.game_id = "quoridor";

  // 必须组件
  auto s = std::make_unique<QuoridorState>();
  s->reset_with_seed(seed);
  b.state = std::move(s);
  b.rules = std::make_unique<QuoridorRules>();
  b.value_model = std::make_unique<board_ai::DefaultStateValueModel>();
  b.encoder = std::make_unique<QuoridorFeatureEncoder>();

  // 状态序列化（Web 前端用）
  b.state_serializer = serialize_quoridor;
  b.action_descriptor = describe_quoridor;

  // 启发式策略（heuristic guidance + eval benchmark）
  b.heuristic_picker = heuristic_pick_quoridor;

  // 残局求解器
  b.tail_solver = std::make_unique<board_ai::search::AlphaBetaTailSolver>();

  // 每局统计
  b.episode_stats_extractor = [](const board_ai::IGameState&,
      const std::vector<board_ai::SelfplaySampleView>& samples)
      -> std::map<std::string, double> {
    return {{"turns", static_cast<double>(samples.size()) / 2.0}};
  };

  // 辅助训练信号
  b.auxiliary_scorer = [](const board_ai::IGameState& state, int player) -> float {
    const auto& qs = board_ai::checked_cast<QuoridorState>(state);
    int me = QuoridorRules::shortest_path_distance(qs, player);
    int opp = QuoridorRules::shortest_path_distance(qs, 1 - player);
    return std::tanh(static_cast<float>(opp - me) / 8.0f);
  };

  // 超时判定
  b.adjudicator = [](const board_ai::IGameState& state) -> int {
    const auto& qs = board_ai::checked_cast<QuoridorState>(state);
    int d0 = QuoridorRules::shortest_path_distance(qs, 0);
    int d1 = QuoridorRules::shortest_path_distance(qs, 1);
    if (d0 < d1) return 0;
    if (d1 < d0) return 1;
    return -1;
  };

  // 训练动作过滤
  b.training_action_filter = [](board_ai::IGameState& state, /* ... */) {
    // ... 过滤逻辑 ...
  };

  return b;
});
```

### 6.3 多变体注册

同一个游戏可以注册多个变体（如不同玩家数），使用不同的 game_id：

```cpp
// Splendor 注册了 4 个变体
board_ai::GameRegistrar reg_2p("splendor",    factory<2>);
board_ai::GameRegistrar reg_2p_("splendor_2p", factory<2>);
board_ai::GameRegistrar reg_3p("splendor_3p", factory<3>);
board_ai::GameRegistrar reg_4p("splendor_4p", factory<4>);
```

---

## 7. 配置文件

配置字段说明已抽到独立文档：[CONFIG_REFERENCE.md](CONFIG_REFERENCE.md)。涵盖 `games/<game>/config/game.json` 的全部顶层字段、`network` / `training` / `replay` / `optimizer` / `tail_solve` / `heuristic_guidance` / `training_action_filter` / `auxiliary_score` / `mcts_profiles.{selfplay,arena,eval}` 等子段，以及 `web.json` 的 Web 平台配置(`mcts_profiles.{web_expert,web_casual,analysis}`、tail-solve、动作过滤)。MCTS 强度统一通过六个命名 profile + `resolve_profile(...)` 取用,不再有顶层 `mcts` / `mcts_schedule` 段。

训练 pipeline 自动发现 `games/<game>/config/game.json`（通过 `training/cli.py` 中的 `find_game_config()`）。

---

## 8. 构建集成

### 8.1 games/<name>/CMakeLists.txt

在 `games/<game>/` 下创建：

```cmake
add_library(game_<name> STATIC
    <name>_state.cpp
    <name>_rules.cpp
    <name>_net_adapter.cpp
    <name>_register.cpp
)
target_include_directories(game_<name> PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(game_<name> PUBLIC dinoboard_core)
```

### 8.2 games/manifest.json

在 `games/manifest.json` 的 `games` 数组里追加一项——这是 setup.py 和顶层 CMakeLists.txt 共享的唯一来源，新游戏只需要在这里登记一次：

```json
{
  "id": "<name>",
  "sources": [
    "<name>_state.cpp",
    "<name>_rules.cpp",
    "<name>_net_adapter.cpp",
    "<name>_register.cpp"
  ]
}
```

不要再去手改 `setup.py` 的 sources 列表，也不要在顶层 `CMakeLists.txt` 添加 `add_subdirectory`——两边都会自动从 manifest 读取。

### 8.3 构建验证

```bash
# 构建 C++ 扩展
pip install -e .

# 验证注册成功
python -c "import dinoboard_engine; print(dinoboard_engine.available_games())"
# 应该输出包含你的 game_id 的列表
```

---

## 9. 训练可选特性

### 9.1 HeuristicPicker — 启发式策略

**用途**：
1. Heuristic guidance：selfplay 中按三段式 schedule（`heuristic_guidance_hold_steps` + `heuristic_guidance_steps` + `heuristic_guidance_initial_ratio`）以一定概率用 heuristic 代替 MCTS 选动作，在网络太弱无法有效搜索时推动游戏前进。hold 期可设 100% 启发式自博弈，样本走主 replay buffer
2. Eval benchmark：评估模型 vs heuristic 的胜率

**签名**：`(IGameState&, const IGameRules&, uint64_t rng_seed) -> HeuristicResult`

**开发者只需给动作打分**。框架负责根据分数选择动作；selfplay 与 eval 使用同一套"分数 + 温度"规则但温度独立配置：
- Selfplay guidance：由 `heuristic_guidance_temperature` 控制。温度=0 时走贪心 argmax；温度>0 时对分数做 softmax 采样（`exp(score/T)`）。常配高温（1.5–3.0）追求多样性
- Eval benchmark：由 `heuristic_temperature` 控制（通过 `run_constrained_eval_vs_heuristic` 传入；训练 pipeline 自动从 `game.json` 读）。常配 0.0 求强度基准

不需要在 heuristic 内部实现随机逻辑或 tiebreaking。启发式步骤也生成训练样本，policy target 对应实际使用的概率分布。

> **注意**：eval 与 selfplay 必须共用同一套启发式选择规则（共享 `sample_heuristic_index`）。如果 eval 里退化成"只取第一个最大值"，对打分存在并列的游戏（如 Quoridor 的走子 vs 有效放墙都是 +1）会让启发式行为和训练时完全不同，win rate 毫无参考意义。

```cpp
HeuristicResult heuristic_pick(IGameState& state, const IGameRules& rules,
                                std::uint64_t /*rng_seed*/) {
  auto legal = rules.legal_actions(state);
  HeuristicResult result;
  result.actions = legal;
  result.scores.resize(legal.size());

  for (size_t i = 0; i < legal.size(); ++i) {
    auto clone = state.clone_state();
    rules.do_action_fast(*clone, legal[i]);
    result.scores[i] = evaluate_position(*clone);
  }

  return result;
}
```

### 9.2 ITailSolver — 残局求解器

**用途**：在 MCTS 搜索之前，尝试用 alpha-beta 精确求解。如果证明了必赢（proven win），直接使用求解结果而非 MCTS。使用 paranoid 假设（所有对手联合针对当前玩家），支持任意玩家数。仅在 proven win 时替代 MCTS——proven loss 在多人场景下过于悲观（对手之间有利益冲突），不作为放弃搜索的依据。

**触发条件**：通过 `tail_solve_trigger` 回调决定是否尝试求解。**注册 tail_solver 必须配套注册 trigger**——profile 启用 `tail_solve_enabled = true` 时，resolver 会校验 `game_metadata(...).has_tail_solve_trigger`，缺失则启动报错。

**签名**：`(const IGameState& state, int ply) -> bool`

同时接收游戏状态和轮数（因为有些游戏轮数没有编码进状态）。返回 `true` 表示应该尝试求解。

**注册方式**：
```cpp
b.tail_solver = std::make_unique<board_ai::search::AlphaBetaTailSolver>();
b.tail_solve_trigger = [](const board_ai::IGameState& state, int ply) -> bool {
  // 根据游戏局势判断是否值得尝试求解
  ...
};
```

**示例**（Splendor — 有人接近胜利时尝试）：
```cpp
b.tail_solve_trigger = [](const board_ai::IGameState& state, int /*ply*/) -> bool {
  const auto& s = board_ai::checked_cast<SplendorState<NPlayers>>(state);
  const auto& d = s.persistent.data();
  for (int p = 0; p < NPlayers; ++p) {
    // 10 分是规则上的相变点：5 分卡数量决定 10→15 必经一张大牌，
    // 搜索空间在此处显著塌缩。实测 11 分阈值会漏掉~17% 必胜手。
    if (d.player_points[static_cast<size_t>(p)] >= 10) return true;
  }
  return false;
};
```

**示例**（Quoridor — 有人接近目标时尝试）：
```cpp
b.tail_solve_trigger = [](const board_ai::IGameState& state, int ply) -> bool {
  if (ply < 20) return false;
  const auto& qs = board_ai::checked_cast<QuoridorState>(state);
  const int d0 = QuoridorRules::shortest_path_distance(qs, 0);
  const int d1 = QuoridorRules::shortest_path_distance(qs, 1);
  return d0 <= 4 || d1 <= 4;
};
```

**设计建议**：
- 用 `ply` 下限避免在开局/中局浪费预算（搜索树太大必然超 budget）
- 用游戏局势判断"可能在几步内结束"再尝试求解
- 训练日志中的 tail solve 统计可以帮助调参：成功率过低说明触发太激进，成功率接近 100% 说明可以适当放宽条件

**训练日志格式**：`tail_solve=X%(N/M, Yms)` 和 `ts=proven/completed/attempts`
- `X%`：成功率（proven / attempts）
- `N/M`：成功数 / 尝试数
- `Yms`：平均求解耗时
- `ts=proven/completed/attempts`：每步的求解统计（proven 是证明必赢/必输的次数，completed 是在预算内完成搜索的次数，attempts 是总尝试次数）

**采用条件**：求解返回的 `|value| >= 1.0`。只有证明了必赢/必输才替换 MCTS 结果；平局或搜索不完整（budget 耗尽）不会采用。

需要在 game.json 的相关 profile 中启用（每个 profile 独立设置；典型场景 selfplay/arena/eval 关闭、`web_expert` 打开）：

```json
"mcts_profiles": {
  "web_expert": {
    "...": "...",
    "tail_solve_enabled": true,
    "tail_solve_depth_limit": 10,
    "tail_solve_node_budget": 200000,
    "tail_solve_margin_weight": 0.01
  }
}
```

触发由 `tail_solve_trigger` 决定，profile 不再有 `tail_solve_start_ply` 字段（已删除）。

**分差偏好**：设置 `tail_solve_margin_weight` 可让 tail solver 在多条必赢路线中选择"赢得最多"的。终局节点的评估公式变为：

```
value = terminal_value + margin_weight × auxiliary_scorer(state, perspective)
```

`margin_weight` 需保证 `margin_weight × max(|scorer_value|) < 1.0`，否则平局可能被误判为胜利（采用条件为 `|value| >= 1.0`）。`auxiliary_scorer` 接口无返回值限制，如果 scorer 无界需相应减小 weight。需要同时注册 `auxiliary_scorer`。

**注意**：对于随机游戏，确保实现了 `do_action_deterministic()`，否则 tail solver 会包含随机分支导致结果不准确。

#### 9.2.1 调参方法论：触发条件 / 深度 / 预算如何选

新游戏接入 tail solver 时不要凭直觉拍超参数。以下是从 Splendor / Azul 调参实战总结的流程：

**Step 1 — 用游戏规则锁定触发器锚点**

触发条件应该对应**规则常量定义的"残局相变点"**，不是笼统的"差不多到后期"：

| 游戏 | 触发锚点 | 规则依据 |
|------|---------|---------|
| Splendor | 任一玩家 ≥10 分 | 5 分卡数量有限，10→15 必经一张大牌，搜索空间塌缩 |
| Azul | 某行 pattern-line ≥4 + ≥2 工厂空 | 倒数第二格 + 本轮接近结束，行动空间显著收窄 |
| Quoridor | 某玩家最短路 ≤ 4 | 距离阈值下分支因子可控 |

**反例**（要避免）：
- `ply >= N` 单独作为触发条件 —— ply 是 epiphenomenon，不是因果
- 复合 AND 条件里塞"双保险"（如 `ply>=40 && points>=10`），实战会发现某一项从不咬住，删掉等价。**不咬住 = 不存在**
- 拍脑袋的中间值（如 Splendor 选 12 分 = 离胜利只差 3 分）—— 没有规则依据，多半不是相变点

**Step 2 — 端到端实测，看四个 KPI**

跑 N=20-30 局 `run_selfplay_episode(tail_solve_enabled=True, ...)`，收集 episode stats：

| 指标 | 含义 | 健康范围 |
|------|------|---------|
| `tail_solve_attempts` | 触发次数 | 衡量触发器作用范围 |
| `completed / attempts` | 完成率（未超 budget） | **必须 ≥ 95%**，低于此说明在浪费搜索 |
| `successes / attempts` | 命中率（找到必胜的比例） | 触发器精度核心 KPI |
| `tail_solve_total_ms / attempts` | 单次平均耗时 | 网页端 < 50ms 玩家无感，>200ms 要警惕 |

**别只测一个手造局面**。让 AI 自对弈 25 局，得到的是**真实分布**下的指标。单点测试无意义。

**Step 3 — 调参顺序：一次只动一个变量**

1. **先把 budget 给慷慨**（比预期需要的大一个数量级，例如 1M），让深度自由发挥
2. **从某个 depth 起步往上推**（5 → 6 → 7 → 8 ...），观察完成率
3. **判定停止**：完成率明显掉下来才停（不是看耗时、不是看"够用了"）。Splendor 实测 6→7 完成率从 100%→99.6%、必胜手 +19%，这种程度**不算崩**，应该继续往上测 —— 因为崩塌曲线通常先慢后快，而效率/质量平衡的 sweet spot 一般在崩塌前的 1-2 档。**实际工程上还要权衡"崩了一点 vs 多吃下来的必胜手"**：如果 99% → 95% 完成率换来 +30% 必胜手，多半值得；如果换来 +5% 必胜手，那就停在 99%。
4. **再调触发阈值**（`points>=10` vs `>=11`）：往严的方向收一档对比
   - successes 掉得慢、attempts 掉得快 → 阈值收得对（精度提升，浪费减少）
   - successes 跟着掉很多 → 阈值过严，错过有价值局面，退回去
5. **回头压 budget**（如果完成率仍接近 100%）：压到刚好覆盖 95 分位耗时，给玩家更快响应

**Step 4 — 把规则知识凌驾于命中率之上**

命中率（successes/attempts）高不等于强。一个永远不触发的过严触发器命中率 100%，但毫无用处。**规则告诉你的相变点 > 命中率优化结果**。

**例**：Splendor `>=11` 命中率（30.9%）高于 `>=10`（22.1%），但 `>=10` 多找出 21% 必胜手 —— 因为规则上 10 分才是相变点，11 分时已经晚了一步。

**反原则总结**

- ❌ 没测过的设置不要相信先验直觉（"指数爆炸"在 TT + 迭代加深下不一定）
- ❌ "看起来够用就停" → 必须测到完成率真的掉
- ❌ 复合 AND 条件没单独验证每一项是否在咬住
- ❌ 只测一个手造局面就下结论
- ❌ 同时动多个变量（深度 + 阈值 + 预算）—— 失去归因能力

### 9.3 TrainingActionFilter — 训练动作过滤

**用途**：在训练时缩小动作空间，去除明显不好的动作，加速早期学习。

**签名**：`(const IGameState&, const IGameRules&, const vector<ActionId>&) -> vector<ActionId>`

**概率应用**：filter 不是永久生效的。通过 `training_filter_steps` 配置，filter 的应用概率从 `training_filter_initial_ratio`（默认 0.5）线性衰减到 0。每个 ply 独立掷骰决定是否使用 filter。这确保网络最终在全动作空间上训练，避免泛化问题。

```json
"training_filter_steps": 800,
"training_filter_initial_ratio": 1.0
```

**注意**：
- filter 接收 `const IGameState&`（不可变）；评估动作质量需要前瞻时，应在副本上 clone 后再做（不要试图在原 state 上 do/undo）
- 如果 filter 返回空 vector，框架会 **fallback 到完整合法动作集**
- filter 同时影响 selfplay MCTS 和 constrained eval
- 详见 [BUG-003](../KNOWN_ISSUES.md#bug-003-训练-评估动作空间不一致) 关于评估一致性的讨论

**legal_mask 与 filter 的关系**：filter 只影响 MCTS 搜索和动作选择的范围，训练样本的 `legal_mask` 始终使用完整合法动作集。被过滤的动作在 policy target 中 visits 为 0，通过 cross entropy 梯度，模型学到这些动作概率应该为 0。如果 `legal_mask` 也被 filter 缩小，模型不会收到任何关于被过滤动作的梯度信号，导致这些动作的 logit 保持随机值——在 free 模式下模型会错误地选择它们（见 [BUG-016](../KNOWN_ISSUES.md#bug-016-legal-mask-被-filter-缩小导致-free-模式失效)）。

### 9.4 AuxiliaryScorer — 辅助训练信号

**用途**：提供 win/loss 之外的额外训练目标。网络会多出一个 score head 来预测这个分数。

**签名**：`(const IGameState&, int player) -> float`

建议返回值在 [-1, 1] 范围内（网络使用 tanh 输出）。

**示例**（Quoridor 的位置优势分）：
```cpp
b.auxiliary_scorer = [](const IGameState& state, int player) -> float {
  const auto& qs = checked_cast<QuoridorState>(state);
  int me = QuoridorRules::shortest_path_distance(qs, player);
  int opp = QuoridorRules::shortest_path_distance(qs, 1 - player);
  return std::tanh(static_cast<float>(opp - me) / 8.0f);
};
```

### 9.5 GameAdjudicator — 超时判定

**用途**：当游戏达到 `max_game_plies` 但未终局时，判定胜负。

**签名**：`(const IGameState&) -> int`

- 返回赢家索引（0-based）
- 返回 -1 表示平局

### 9.6 EpisodeStatsExtractor — 每局统计

**用途**：从完成的对局中提取自定义指标，汇总后在训练日志中显示。

**签名**：`(const IGameState&, const vector<SelfplaySampleView>&) -> map<string, double>`

```cpp
b.episode_stats_extractor = [](const IGameState&,
    const std::vector<SelfplaySampleView>& samples) -> std::map<std::string, double> {
  return {{"turns", static_cast<double>(samples.size()) / 2.0}};
};
```

这会在训练日志中产生 `turns=7.4` 这样的条目。

## 10. 隐藏信息与 Belief Tracker（含物理随机性）

> **算法深入**：DAG 节点共享、UCT2、完整 search_root 流程、debug 指标——独立文档 [`ALGORITHM_OVERVIEW.md`](../../ALGORITHM_OVERVIEW.md)。本节讲开发者接口。

隐藏信息指玩家间的**非对称**信息——某个玩家知道、其他玩家不知道的游戏状态。如 Splendor 的盲压暗牌（执行者知道是什么牌，对手不知道）。

对称无知的随机（如 Azul 袋子未来抽取顺序）走另一条路径：所有玩家对袋子里下一张牌的"已知"是同样的，没有 viz=0 槽位需要 tracker 来采样——物理随机走 descent 中 `sim_rng` 在 `do_action_fast` 里即时抽的路径，**不需要注册 `belief_tracker`**。见 §10.1。

### 10.1 物理随机性

物理随机性指翻牌、抽卡、掷骰等改变游戏状态的随机事件。确定性游戏（TicTacToe、Quoridor）不涉及此节。

**ISMCTS 下的处理**：物理随机和信息不对称被**统一**——没有 chance node 专门机制。关键点：

- Root 采样通过 `belief_tracker->randomize_unseen(sim_state, observer, sim_rng)` 一次性固定当前 sim 的 hidden 抽样(opp 手牌、未观察到的 deck 顺序等);公开物理随机(如 Azul 工厂 refill)留到 descent 时由 `do_action_fast` 用同一个 sim_rng 即时抽
- Descent 里 `do_action_fast` 照常从状态中读取随机源（如 `d.deck.top()` 或 `splitmix64(d.draw_nonce)`），每次 sim 拿到的值由 root 采样决定
- 观察者可见的后果(Splendor 翻新卡到 tableau)自然通过
  `state_hash_for_perspective` 差异(即 `viz=1` 公开槽位变化)分化到不同
  DAG 节点
- 观察者不可见的后果(opp 抽牌)在 observer 视角下 viz=0,hash 不变,自然
  在同一 DAG 节点上汇聚

**开发者要做的**(对有物理随机的游戏):

1. `do_action_fast` 接 `mt19937_64& rng`,从 rng 即时抽取(对完全公开的
   袋子/盒盖,可以直接按 `bag_counts[color]` 抽色;对隐藏 deck,弹
   `deck.top()`)。**rng 不在 state 上**——sim 入口由 caller 提供
   `sim_rng`,GT 端提供 `gt_rng`,各自独立
2. 实现 `IBeliefTracker::randomize_unseen(state, observer, rng)`(仅
   在游戏有 asymmetric hidden info 时需要)—— 把 state 里 viz=0 的槽
   位按 belief 一次性采样填入。**约束**:产出世界中所有公开槽位
   (`viz=1` to anyone)的内容只取决于 tracker 的观察历史,不依赖输入
   state 的 hidden 内容也不依赖调用者 RNG 特定值

对**完全公开 + 物理随机** 的游戏(如 Azul):**不必注册 `belief_tracker`**——
没有 viz=0 槽位需要采样。物理随机走 descent 中由 sim_rng 在
`do_action_fast` 里即时抽的路径。

### 10.2 信息屏障：AI 链路从根源读不到真值
完整论证见 [`CLAUDE.md` 「AI Pipeline Independence from Game State」](../../CLAUDE.md#ai-pipeline-independence-from-game-state)：tracker 没有 `IGameState*`、session public 部分由 message 整张覆盖、session viz=0 槽位决策侧物理上读不到(hash/encoder/sim 三家都把 viz=0 屏蔽掉)、selfplay/web/API 走同一套 per-perspective tracker——四条结构性约束让 AI 物理上没有路径可读真值。下文 §10.3 起讲各 hook 的具体签名与实装。

### 10.3 架构原理（ISMCTS）
核心三层机制：

1. **Root 采样 determinization**：每次 MCTS simulation 持独立 RNG。Root 步调 `belief_tracker->randomize_unseen(sim_state, observer, sim_rng)`,把 viz=0 hidden 字段（opp 手牌、deck 顺序、未来 hidden 抽牌结果）采样成具体值;descent 期间该 sim_rng 仍可能被 `do_action_fast` 消费处理物理随机(如 Azul 工厂 refill 在 round-end 的公开重抽)。给定 sim 种子整个展开完全可复现
2. **Per-acting-player DAG keying**：每决策节点用 `state.state_hash_for_perspective(state.current_player())` 当 key。全局 `unordered_map<StateHash64, int>` 表让不同路径到达同一信息集共享节点
3. **UCT2 UCB**：DAG 下多父路径汇聚到同一节点时，`sqrt()` 分子底用刚经过的入边的 visit_count，避免 node global count 夸大探索预算

**无 chance node 机制**——hidden 随机被 root 采样吞掉,公开物理随机由 descent 里 sim_rng 在 `do_action_fast` 内即时抽;observer-visible 后果通过 hash 自然分叉,observer-invisible 后果通过 hash 自然合并。

**新游戏开发者要做**:
- 在 `<game>_state.cpp` 用 `viz::declare_field` 声明每个 state 字段的 name
  + data shape + base viz tensor(`all_public` / `owner_only_first_axis` /
  `all_hidden`)——schema 是 hash / encoder / snapshot 的单一事实源(§E.3)
- 实现 `IGameState::hash_field_slot(h, name, idx)`——按 name 分支 emit
  该 slot 的值。框架的 `state_hash_for_perspective(p)` 会按 schema 序遍历,
  对 viz=1 的 slot 调本函数,viz=0 的 slot 自动 emit sentinel。游戏侧
  **不要**自己判 viz、**不要**自己 hash step_count
- 隐藏信息游戏额外实现 `read_field_slot` / `write_field_slot` /
  `mask_field_slot`——对应 walker 物化 / wire 序列化 / 掩盖 placeholder
- 实现 `IBeliefTracker`(§10.4)——`init` / `observe_public_event` /
  `randomize_unseen`(仅 asymmetric hidden info 需要;Azul 这种完全公开
  +物理随机不需要)
- 实现 public-event protocol(`public_event_extractor` / `applier` /
  `initial_observation_extractor` / `applier`)
- 框架自动接管 root 采样时机、DAG 节点复用、UCT2 UCB。**不需要**在 rules
  里做任何防御性 nonce bump 或 hidden-info guard

### 10.3b 声明式 hash API:`hash_field_slot` + visibility schema
每个 `IGameState` 子类实现一个虚方法 + 一个静态 schema:

```cpp
// 静态 schema(每游戏一份):字段 name + 数据 shape + base viz
static const viz::VisibilitySchema& schema() {
  static const viz::VisibilitySchema s = [] {
    viz::VisibilitySchema schema;
    viz::declare_field(schema, "board",
                       viz::all_public({9}, /*n_players=*/2));
    viz::declare_field(schema, "current_player",
                       viz::all_public({}, 2));
    return schema;
  }();
  return s;
}

// per-slot hash dispatcher
void hash_field_slot(Hasher& h, const std::string& name,
                     const std::vector<int>& idx) const override;
```

框架自动派生节点 hash(`for_each_slot` 走 schema 全集 × row-major
idx,**framework 自己 mix `(field_pos, idx[])` 做结构盐**,然后再分发
值的 mix):

```cpp
StateHash64 state_hash_for_perspective(int player) const {
  Hasher h;
  h.add(step_count_);                 // DAG 结构性防环
  for_each_slot(*this, schema_ref(), player,
      [&](const std::string& name, const std::vector<int>& idx,
          const VizTensor&, bool visible) {
        h.add(field_pos_of(name));     // 框架 mix 槽位结构盐
        for (int x : idx) h.add(x);
        if (visible) hash_field_slot(h, name, idx);  // 游戏只 mix 真值
        else         h.combine(kHiddenHashSentinel); // 隐藏槽走哨兵
      });
  return h.finalize();
}
```

**重点**:`hash_field_slot` 现在**只负责 mix slot 的值**,不要 mix
`idx`(framework 已经在调你之前把 `(field_pos, idx[])` mix 进去了)。
BUG-037 postmortem:LL 4p step-125 DAG mismatch 的根因就是 `hash_field_slot`
对 `hand[idx]` 只 mix 了值丢了 idx,导致 `{hand[1]=5, hand[3]=7}` 与
`{hand[0]=5, hand[1]=7}` digest 相同。framework 接管 idx 编码后,这
一类"游戏忘记 mix idx"的 bug 在源头消失。

**字段归类(由 schema base viz 决定,运行时由 rules 通过 `viz::reveal_slot`
/ `reveal_slot_to` / `reset_to_base` 改写 viz)**:
- **`all_public`**:所有玩家都能看见的字段。弃牌堆、公开棋盘、分数、当前
  玩家、回合计数等。所有 perspective 都看到 viz=1
- **`owner_only_first_axis`**:第一维当 owner 轴。owner-only 手牌、影响牌
  等——`viz[hand[p], p] = 1`(p 看见自己),`viz[hand[p], q != p] = 0`
- **`all_hidden`**:对所有人都不可见的字段(初始 base)。rules 显式
  `reveal_slot` / `reveal_slot_to` 后才对部分玩家可见

**常见错误**:
- ❌ 把不归任何玩家观察的字段(内部 RNG salt、未抽 deck 顺序等)放进
  `viz=1` 路径 → 这是本框架最隐蔽的失误模式。会让本应该是同一个 DAG
  节点的信息集,按"未来抽牌的具体顺序"分裂成 N 个不同节点;网络无法
  分辨它们,搜索的统计聚合被打散。**症状只是"AI 莫名变弱 / selfplay 与
  API 路径策略不一致",从不崩溃**——所以最难抓。Azul BUG-028
  (`bag` / `box_lid` 的 vector 顺序进 public)和 Splendor 历史上的
  `rng_salt` 都属此类。修法:**RNG 不进 state**(三类 RNG 由 caller 持
  有,§3 ALGORITHM_OVERVIEW),deck 顺序属于 viz=0 的 hidden 内容
- ❌ 在 `hash_field_slot` 里读 viz 自己判要不要 emit → 框架已经判过了,
  双重判断容易写错
- ✓ 公开可推导的 multiset(袋子各色剩余数、牌堆大小)进 schema 当
  `all_public` 字段;具体顺序 / 内部 RNG 不进 state
- ✓ 写完 `hash_field_slot` 分支后通读一遍,问自己每个 `h.add(x)`:**"这
  个 slot 在所有 viz=1 的 perspective 看到的真值,在那位 perspective 的
  观察历史里能复现吗?"** 如果答案是"不能,这是引擎实现细节",这个字
  段就不该进 schema(应该改成 viz=0 的 hidden,或彻底从 state 移除)

**Step counter**:`IGameState::step_count_` 由框架管理,**游戏作者既看不到也无法
忘记**:
- `step_count_` 字段是 protected,只有 `IGameRules` 是 friend
- `IGameRules::do_action_fast` / `do_action_deterministic` / `undo_action`
  是 non-virtual wrapper,自动在 `*_impl` 前后 ±1
- `reset_with_seed` 第一行调 `reset_step_count_base()` 把 step_count_ 归零

`state_hash_for_perspective` 自动把 step_count 计入 hash,保证 DAG 结构性
acyclic。**游戏开发者不要在 `hash_field_slot` 里重复 hash step_count**。
回归测试:`tests/framework/test_step_count_strict_increase.py` 跑遍 manifest
里所有启用的游戏,assert 每动作 step_count 严格 +1。

**测试建议**:构造两个 state,自己 perspective 的 viz=1 槽位完全相同,
但 opp 私有(viz=0 to me)不同;assert
`state_hash_for_perspective(self)` 相等。这是 hash 归类正确性的基本检查,
`tests/framework/test_encoder_respects_hash_scope.py` 提供模板。

### 10.4 IBeliefTracker
**用途**：维护当前玩家的信息认知，为 ISMCTS 根采样提供 prior。

5 个必须实现的方法（观察-only 接口）：

```cpp
virtual void init(const AnyMap& initial_observation) = 0;
virtual void observe_public_event(
    int actor, ActionId action,
    const std::vector<PublicEvent>& events) = 0;
virtual void randomize_unseen(IGameState& state, int observer,
                              std::mt19937_64& rng) const = 0;
virtual std::unique_ptr<IBeliefTracker> clone() const = 0;
virtual AnyMap serialize() const = 0;
```

**关于这几个方法**：
- `init` **不接 perspective 形参**——tracker 是 perspective-agnostic 的，观察者座位由 `randomize_unseen` 的 `observer` 参数传入。
- `randomize_unseen` 必须 `const`：MCTS sim 路径上调的是 cloned `sim_tracker`，不允许写回累积状态。
- `clone()` 每次 sim determinization 都会被调用——把当前 belief 复制一份给 sim 用，主 session 的 tracker 不被 sim 写脏。
- `serialize()` 输出 canonical 字典，给 `test_api_belief_matches_selfplay` 之类的回归测试做对比。

**结构性约束（编译器层强制）**：`init` 和 `observe_public_event` 方法签名里没有 `IGameState*` —— tracker 在这两个方法里物理上拿不到 state 指针，**无法**偷看真实游戏状态。所有输入都来自游戏注册的两个 extractor：

- `init` 收 `initial_observation` → 来自 `bundle.initial_observation_extractor(state, perspective)`
- `observe_public_event` 收 `events` → 来自 `bundle.public_event_extractor(before, action, after, perspective).events`。list 内顺序就是 producer 发出的顺序；tracker 把它当事实序列消费，不需要区分动作前后

两个 extractor 是小函数（~20-40 行），只读观察者可见字段，易于审计。

`randomize_unseen(state, observer, rng)` 是采样的**写入口**——可以读 state 的公开字段 + observer 视角下 viz=1 的字段（discard_piles、自己的 hand 等），但禁止读 opp 视角下的 hidden 字段（对手手牌、deck 内容）。tracker 从自己累积的 belief 构建 unseen pool 和 known-hand 分配。

**`randomize_unseen` 契约**:返回的世界必须满足所有公共不变量——
`state_hash_for_perspective(any p)` 在 viz=1 的所有公开槽位部分,在相同
观测史下、不同 RNG 采样下 byte-equal。隐藏多重集(deck size / bag size /
court-deck size 等)必须由 tracker 的 seen 信息**推导**出来,不能保留输
入 state 里的残值(输入里的残值本身就是采样结果,不是真值的副本)。
Stale 采样(opp hidden 字段的旧 cid 现在已经被公开看到)必须从当前未
见池重采。

**`randomize_unseen` 的唯一调用点**:MCTS sim 入口的 cloned `sim_tracker` 上一次调用(per-sim determinization)。**session 永远不调** `randomize_unseen`——session 的 viz=0 槽位是 unread bytes,不进 hash(`kHiddenHashSentinel`)、不进 encoder(`MaskedState` placeholder)、不进 sim 入口(sim 自己 clone tracker 重新采)。详见 [DEC-003](../KNOWN_ISSUES.md)。

**开发者必须保证**：游戏的公开输出不依赖任何"只存在于 session state_ 里的隐藏字段"。做法是规范化的：`public_event_extractor` 把 post-action 的全部 public 字段 dump 进 `PublicEventTrace.public_snapshot`，`public_state_applier`（§5.1 项 8）把 snapshot 反向写回 session state_ 的 public 字段。**observer 路径上不调 `do_action_fast`** —— `apply_observation` 只走 `begin_step_for_session_observe → public_state_applier(snapshot) → tracker.observe_public_event(events)`,viz=0 hidden 槽位不被 freshen,公开部分完全由 message 重建。round-trip 测试 (`test_public_snapshot_round_trip`) + 60-seed drift 扫 (`test_public_hash_excludes_internal_rng`) 在 CI 里守这个契约。

**Belief tracker 追踪公开聚合信息**（剩余牌池的多重集、已 flipped tier 卡集合等）。游戏技能合法获得的"私有知识"（Priest 偷看的对手牌、King 交换后双方知道的牌、Baron 比较中暴露的双方牌）由 GT 端 rules 直接通过 `viz::reveal_slot_to(viewer)` / `swap_slot_owned` 写到 `state.viz_` 上——schema 走 walker 进 `public_snapshot`，observer apply 后那些槽位天然以 viz=1 出现在 session state 里。`randomize_unseen` 只填 viz=0 的槽位，对 viz=1 的已知槽位不动。

**实装要点**：tracker 的接口只吃 message（`init` + `observe_public_event`），没有 `IGameState*` 能指回 truth——读 state 是物理上做不到的。tracker 自身持的状态应该是 perspective-agnostic 的公开聚合（"剩余牌池"、"deck 顺序"等），而不是某玩家私密信息（"我看到 p1 是 Guard"——这种走 viz）。

参考 `games/splendor/splendor_net_adapter.h` 中的 `SplendorBeliefTracker` 实现。

### 10.5 Belief Tracker 生命周期
所有代码路径（selfplay、arena、web GameSession、AI API）走同一个生命周期：每个座位一份 tracker（`per_perspective_trackers[p]`），一局 init 一次，之后每一步每个 tracker 都接收一次 `observe_public_event`——tracker 的观测单调累积到 game over，不会被 re-init 清空。

```
游戏开始（每个座位 p = 0..num_players-1）：
  initial_obs_p = bundle.initial_observation_extractor(state, p)
  per_perspective_trackers[p]->init(initial_obs_p)  // perspective-agnostic

每一步 ply（acting player = cp）：
  1. MCTS 搜索                                                 // 见 §MCTS
       root_tracker = per_perspective_trackers[cp]
       每 sim 开头 sim_tracker = root_tracker->clone()
                  sim_tracker->randomize_unseen(sim_state, cp, sim_rng)
       descent 给定 sim_rng 完全可复现(物理随机由 do_action_fast 里同一
       sim_rng 即时抽)
  2. do_action_fast(state, chosen)                             // GT 执行动作
  3. 对每个座位 p：
       evt_p = bundle.public_event_extractor(before, action, after, p)
       per_perspective_trackers[p]->observe_public_event(
           cp, chosen, evt_p.events)

web / API / per-seat selfplay 推 observer state（不调 do_action_fast）：
  4. seat.begin_step_for_session_observe()
  5. bundle.public_state_applier(seat, evt.public_snapshot)    // public 字段由
                                                                // message 重建
                                                                // (viz=0 不动)
```

游戏开发者只需实现 `IBeliefTracker` 的 5 个方法 + `initial_observation_extractor` + `public_event_extractor`（+ `public_state_applier`，隐藏信息游戏必装）。extractor 调用封装见 `bindings/py_engine.cpp`。

### 10.6 实现示例
Belief tracker 有两种不同定位，由游戏的信息结构决定：

**定位一：维护随机来源**（Splendor）——状态无法推导出完整的"见过什么"历史，tracker 增量追踪 seen 信息，为 `randomize_unseen` 提供准确的 unseen pool。

**定位二：精确知识由 viz 承载**（Love Letter）——随机来源可从状态推导（弃牌堆是完整打出记录），不需要 tracker 维护 seen 集合；游戏技能产生的精确信息（Priest 偷看、Baron 比较、King 交换）由 rules 直接调 `reveal_slot_to` / `swap_slot_owned` 写进 `state.viz`。tracker 退化成 stateless：只为 `randomize_unseen` 提供"剩余牌池多重集"，对手已知手牌靠 session state 上 viz=1 的槽位天然存在。

两种定位可以并存。Love Letter 是本框架中对"不完美信息 + 短对局 + 频繁信息交换"类游戏的探索——这类游戏的难点不在随机来源维护，而在通过 viz schema + walker 把游戏技能产生的精确知识正确传递给 observer。

**Love Letter（精确知识走 viz，tracker stateless）**：
- `init(initial_obs)`：从 `initial_obs["face_up_removed"]` 等公开开局信息推断剩余牌池多重集
- `observe_public_event(actor, action, events)`：根据弃牌堆增量、淘汰宣告等公开事件更新剩余牌池
- `randomize_unseen(state, observer, rng)`：扣除 state 上 viz=1 槽位（自己手牌、Priest peek 后的对手槽位等）已经占用的 cid，对剩余 viz=0 槽位（牌堆、未 reveal 的对手 hand）从池子均匀采样
- encoder 直接读 MaskedState：viz=1 槽位拿到真值（Priest peek 后的对手 hand），viz=0 槽位拿 placeholder——不用查 tracker
- 随机来源与 Azul 同属"状态可推导"模式：弃牌堆是完整历史，不需要 `seen_cards` 集合

**Splendor（随机来源追踪——seen 需增量维护）**：
- 维护 `seen_cards` 集合，**绝不读取 `data.decks`**
- `init(perspective, initial_obs)`：从 `initial_obs["tableau"]` 初始化 seen（所有公开 tableau 卡）
- `observe_public_event`：处理 `deck_flip` 事件（新翻出来的卡加进 seen）和 `self_reserve_deck`（perspective 自己盲抽知道的牌）
- `randomize_unseen()`：用 `全卡池 - seen_cards` 构建 unseen pool，按 tier shuffle 后回填 deck 和对手暗牌
- 需要增量追踪的原因：牌被买走后离开 tableau，仅从当前状态无法知道历史上哪些牌曾经可见

**Coup（自定义概率化 sampling 的范例）**：

Coup 是**诈唬核心**游戏——公开声明（claim）和真实持牌可以不一致。如果用 uniform 采样，对手被建模成"随机机器人"，MCTS 会疯狂质疑，训练收敛到"永不质疑永不诈唬"的退化均衡。解决方案是在 `randomize_unseen` 里**手写启发式加权采样**，从 claim / challenge 历史推断对手可能持有的角色。

- `init(perspective, initial_obs)`：清空 `signals_[player][role]` 矩阵（5 个角色 × N 玩家）
- `observe_public_event`：增量维护 `signals_[p][R]`
  - **claim R 无人质疑**：`signals_[claimer][R] += 1`（弱正信号）
  - **claim R 被质疑 + 为真**（证实后洗回 deck）：`signals_[claimer][R] = 0`（那张牌已经不在手上）
  - **claim R 被质疑 + 为假**（bluff 戳穿）：`signals_[claimer][R] = 0`（原始 claim 是假的）
  - **玩家 q 质疑某人 claim R**：`signals_[q][R] += 1`（q 敢质疑说明他自己可能有 R）
  - **任何玩家暴露一张 X**：`signals_[revealer][X] = 0`（X 已 revealed，剩余 slot 对 X 的先验回到 baseline）
  - **Ambassador 换牌**：`signals_[exchanger][*] = 0` 全清零
- `randomize_unseen()`：**带硬约束的加权联合采样**
  1. 收集所有 unseen slot：opp 未揭露 hand + opp exchange_drawn + court deck
  2. 随机 shuffle slot 顺序（避免 opp₀ 总是优先拿稀有角色）
  3. 对每个 slot，`weight[R] = remaining[R] × (1 + 0.5 × signals_[owner][R])`；`remaining[R]` 是**硬约束**——池里没了权重直接 0，保证全局每角色总数 ≤ 3
  4. 按 weight 采样一张，从 `remaining` 扣除
  5. 所有 weight 为 0 时 fallback 到 uniform 采样（保证永不陷入不可行）

**为什么 feasible 是关键**：N-玩家 Coup 的 unseen 字段包括 N 个 opp 各 1~2 张手牌 + deck 剩余（可达 ≥8 张），全都要**联合采样**满足总牌数守恒。独立 per-player 采样会出现"两个 opp 都被采样成 Duke×2，但 pool 里总共只有 3 张 Duke"的不可行情况。用"剩余池 × 启发式权重"的顺序采样自然 enforce 约束。

参考实现：`games/coup/coup_net_adapter.cpp` 的 `CoupBeliefTracker::randomize_unseen`。这套模式适用于任何**手牌可能不匹配公开声明**的游戏——把 claim/challenge 历史转成 per-opp role prior，在池约束下联合采样。比完整的概率化 belief network（未来 work）实现成本低一个数量级。

### 10.7 Encoder 信息屏障与 Tracker 协作
Feature encoder 必须在视角层强制信息隐藏。核心规则：**每个玩家只能看到自己通过合法途径获得的信息**。即使 `randomize_unseen` 已经把采样值写入 state，encoder 对非当前视角玩家的隐藏字段仍须输出占位符——除非 tracker 确认当前玩家合法知道该信息。

Encoder 的信息源有两个，地位等价：
- **state**：公开局面信息（棋盘、弃牌堆、存活状态等）
- **belief tracker**：通过 `observe_public_event` 积累的私有知识（Priest 偷看的手牌、Baron 比较的结果等）

Encoder 直接读 `MaskedState`：viz=1 槽位拿到真值（被 Priest peek 后的对手 hand cid、King 交换后双方各自的新手牌等），viz=0 槽位拿 placeholder。这让网络直接获得游戏技能产生的信息，无需从弃牌历史自行推导，也不需要 encoder 查询 tracker——viz schema + walker 已经把"该看到什么"在数据层定下来了。

以 Splendor 双人局为例，假设双方各有一张暗牌：

| 视角玩家 | 自己的暗牌 | 对手的暗牌 |
|---------|-----------|-----------|
| 我的节点 | 编码真实牌 | **占位符** |
| 对手节点 | 编码采样牌（来自 ISMCTS 根采样） | **占位符**（我的暗牌对手看不到） |

两个方向互为镜像——我看不到对手的牌，对手也看不到我的牌。ISMCTS 采样的值只在"自己看自己"时可见，不会泄露到对方的 encoding 里。

Splendor 实现（`splendor_net_adapter.cpp`）：

```cpp
const bool is_self = (pid == perspective_player);
if (!is_self) {
    const bool visible = d.reserved_visible[pid][slot] != 0;
    if (!visible) {
        encode_hidden_reserved_placeholder(features);  // 占位符
        continue;
    }
}
// is_self=true: 编码真实（或采样的）卡牌
```

这保证了 encoder 在训练和搜索中看到的信息结构完全一致——训练时每个玩家的 sample 也是从该玩家视角编码的，看不到对手隐藏信息。

### 10.8 框架限制
ISMCTS 根采样 + encoder 信息屏障在双人游戏中完全自洽；多人游戏（3+）存在 determinization 方法的固有精度局限（无法建模"B 用 Priest 看了 C 的牌"等第三方私有知识）。完整说明见 [FEATURES_OVERVIEW.md「框架局限性」](../../FEATURES_OVERVIEW.md#框架局限性)。

### 10.9 开发者 Checklist
1. 确认游戏是否有非对称隐藏信息（玩家间知道的不一样）。**对称无知**（如 Azul 的 bag）不需要注册 `belief_tracker`——viz=0 槽位不存在，物理随机直接走 `do_action_fast` 里 `sim_rng` 即时抽。**只有非对称**才需要 tracker 来驱动 `randomize_unseen`
2. 实现 `IBeliefTracker` 的五个方法（`init` / `observe_public_event` / `randomize_unseen` / `clone` / `serialize`），遵守"接口签名根本拿不到 `IGameState*`"的结构性约束（§10.4）
3. Encoder 接受 `MaskedState`——viz=0 槽位由 framework 替换为 `kPlaceholder`，encoder 必须分支处理 placeholder（§10.7），不要查询 viz、不要绕过 mask 读 truth
4. **在 `<game>_state.cpp` 用 `viz::declare_field` 声明每个 state 字段的 name + data shape + base viz tensor**（`all_public` / `owner_only_first_axis` / `all_hidden`），实现 `hash_field_slot` / `read_field_slot` / `write_field_slot` / `mask_field_slot` 四个 per-slot dispatcher（§10.3b）。框架的 walker 按 schema 顺序遍历每个 slot——`viz[..., perspective]=1` 时调 `hash_field_slot` 把 truth mix 进 hash，`viz[..., perspective]=0` 时 mix `kPlaceholder` 哨兵——自动得到 `state_hash_for_perspective(p)` 作 DAG 节点键
5. **`reset_with_seed` 第一行调 `reset_step_count_base()` 把 step_count_ 归零**。step_count_ 由框架的 `IGameRules` wrapper 在 `do_action_fast_impl` / `do_action_deterministic_impl` 前自动 +1，在 `undo_action_impl` 后自动 -1，**作者既看不到 step_count_ 也无法忘记 / 双 bump**（字段 protected + IGameRules friend）。step_count_ 单调递增保证 DAG 结构性 acyclic（回归测试见 `tests/framework/test_step_count_strict_increase.py`）
6. **实现 message-driven snapshot 路径**：要么用 walker（`viz::serialize_public` / `viz::apply_public`，全 schema 字段自动同步），要么用 SnapshotIO（`emit_snapshot` / `apply_snapshot` 手写），把 GT 端的公开 state 序列化为 `AnyMap public_snapshot`，AI session 端 wholesale 替换。tracker 需要的私有/对手知识增量通过 `events` 列表（`PublicEventTrace.events`）增量传递——一份 list、按 producer 顺序、不区分动作前后。详见 §14 事件协议章节（或直接参考 `games/splendor/splendor_register.cpp` walker 路径，`games/loveletter/loveletter_register.cpp` SnapshotIO 路径）
7. 在 `make_<game>` 里注册 `belief_tracker` + `public_event_extractor` + `public_state_applier`（+ `initial_observation_extractor` / `applier` 如果游戏开局有公开信息）
8. **验证测试**：
   - `tests/framework/test_ai_api_separation.py::test_full_game_via_api[<game>]` 必须过（API 契约）
   - `tests/framework/test_api_belief_matches_selfplay.py::*[<game>]` 必须过（belief 等价）
   - `tests/framework/test_public_snapshot_round_trip.py::*[<game>]` 必须过（observer apply snapshot 后 `state_hash_for_perspective(own)` 与 truth bit-equal）
   - `tests/framework/test_encoder_respects_hash_scope.py` 必须过（encoder 不越界读 opp private）
   - `tests/framework/test_public_hash_excludes_internal_rng.py` 必须过（hash 不能依赖 RNG/未揭示 deck 等任何 viz=0 应该藏起来的内容）
   - 建议自己写黑盒统计测试（类似 `TestLoveLetterGuardAccuracy`）：在某个会读隐藏信息的决策点上，验证 AI 的选择分布和无先验情况下的基线一致

### 10.9a 为什么这么多 hook
ISMCTS 让开发者**不需要在游戏规则里做任何防御性代码**（没有 `++nonce`、没有 hidden-info guard、没有 MCTS 特殊路径）。所有隐藏信息处理都在框架层，代价是开发者要把"观察者视角下能看到什么"精确表达出来——这是 hook 列表看起来长的原因。每个 hook 的职责都有清晰语义：

| Hook | 说什么 |
|------|-------|
| `viz::declare_field` 的 base viz tensor | 字段的"出厂可见性"（全公开 / owner-only / 全隐藏） |
| `do_action_fast` 中的 `viz::reveal_slot` / `reset_to_base` | 规则期间显式翻牌 / 槽位换内容时如何更新 viz |
| `hash_field_slot` | 单个 slot 的 hash 贡献（被 walker 在 viz=1 时调用） |
| `read/write/mask_field_slot` | 单个 slot 的序列化 / 反序列化 / placeholder 写入（被 walker 在 snapshot/AI session 端调用） |
| `belief_tracker` | 观察者基于历次观察累积的精确知识 + `randomize_unseen`（仅非对称隐藏信息游戏需要） |
| `public_event_extractor` / `public_state_applier`（+ optional `initial_observation_*`） | message 流的事件 + 公开 state 同步通道 |
| encoder 对 placeholder 的分支 | viz=0 slot 已经被 framework 替换成 `kPlaceholder`，encoder 物理上读不到 truth |

这套"schema + per-slot dispatchers + tracker（可选）+ message protocol"加起来是"观察者视角"的完整规格。ISMCTS 的行为完全从这个规格推导——相同的 framework code 处理所有游戏，不需要 per-game 的 MCTS 特判。

---

## 11. Web 前端开发

Web 前端开发已独立成章，详见 [WEB_DEVELOPMENT_GUIDE.md](WEB_DEVELOPMENT_GUIDE.md)。该文档涵盖 StateSerializer / ActionDescriptor / 目录结构 / 交互设计原则 / `createApp(config)` 框架 API、通用布局、悔棋/替对手落子/智能提示等高级操作、AI Pipeline 与掉分分析、录像回放、统一录像格式、模型评估工具、common.js 通用功能、核心 API 与交互流程。

视觉与交互的总体原则见 [WEB_DESIGN_PRINCIPLES.md](WEB_DESIGN_PRINCIPLES.md)，新游戏前端开发**必读**。

Web 平台配置（AI 难度、tail-solve、动作过滤等）见本文档 [§7 配置文件](#7-配置文件) 中的 `web.json` 字段说明。

---


## 12. 测试

DinoBoard 采用**两层测试架构**:框架层不变量 + 每个游戏自己完整的验收清单。新游戏 ready 的标志是「`pytest tests/<新游戏>/` 一次全绿」。

### 12.1 两层架构概念

- **`tests/framework/`** — 框架不变量。在固定 3 游戏 matrix carrier(`FRAMEWORK_GAMES = ["quoridor", "azul", "loveletter"]`)上跑——这三个游戏一起最小完备覆盖了框架关心的每个结构特征(确定/对称随机/非对称隐藏、2p/2-4p、tail solver、belief tracker 有无 per-player private 字段、淘汰)。**这是项目维护者改框架时的护栏**,新游戏不需要被加到这层。
- **`tests/<game>/`** — 每个游戏自己完整的验收清单,**与框架层有意冗余**。你的工作流就是在这一层完成的。

详见 [新游戏验收测试指南 § 测试架构原则](NEW_GAME_TEST_GUIDE.md#测试架构原则两层测试)。

### 12.2 运行测试

```bash
# 全部(框架 + 所有游戏)
python -m pytest tests/ -x -q

# 只跑你的游戏(开发期最常用)
python -m pytest tests/<your_game>/ -v

# 只跑框架层
python -m pytest tests/framework/ -q
```

### 12.3 接入新游戏:写一份独立的验收清单

**不需要**修改 `tests/framework/` 或 `tests/conftest.py::FRAMEWORK_GAMES`。流程:

1. 从最相近的现有游戏复制一份模板:
   - 完全公开 → `tests/tictactoe/test_checklist.py` 或 `tests/quoridor/test_checklist.py`
   - 对称物理随机 → `tests/azul/test_checklist.py`
   - 非对称隐藏 → `tests/loveletter/test_checklist.py` 或 `tests/coup/test_checklist.py`

2. 改 `GAME = "<your_game>"`、`VARIANTS = [...]`,以及 encoder 字段偏移等游戏特定常量。

3. 创建 `tests/<your_game>/__init__.py`(空文件,pytest 同名 `test_checklist.py` 消歧需要)。

4. `pytest tests/<your_game>/ -v` 迭代到全绿。**全绿就是 ready 的明确信号**。

### 12.4 测试辅助工具

`tests/conftest.py` 严格遵守白名单原则:框架层只认识 `FRAMEWORK_GAMES = ["quoridor", "azul", "loveletter"]` 三个 carrier 游戏,**不存在「所有游戏 metadata 全局表」**。Per-game checklist 通过 `load_game_config(GAME)` 自己加载配置。

| 名称 | 类型 | 说明 |
|---|---|---|
| `FRAMEWORK_GAMES` | 常量 | 框架矩阵 carrier 白名单(quoridor + azul + loveletter)。Per-game 测试**不要**改这个 |
| `load_game_config(game_id)` | 函数 | 加载任意游戏的 game.json + C++ metadata。Per-game checklist 在文件头调用一次 |
| `get_test_model(game_id)` | 函数 | 创建/缓存随机初始化 ONNX 模型 |
| `run_short_selfplay(game_id)` | 函数 | 快速跑一局 selfplay(10 sims, 50 plies) |
| `run_short_heuristic(game_id)` | 函数 | 快速跑一局 heuristic 对局 |
| `run_random_episode_states(game_id, seed, max_plies)` | 生成器 | 随机走子驱动整局,逐步 yield `state_dict`。给 `TestRuleInvariants` 写每个游戏自己的守恒律断言用,不依赖 model |
| `assert_api_belief_matches_selfplay(game_id, public_keys, ...)` | 函数 | **隐藏信息游戏的标准三层等价断言** —— belief/public state/legal actions。在 per-game checklist 里调用一次即可,无需重复实现 |
| `game_id` / `game_config` / `model_path` | fixture | **仅供框架层使用**——参数化为 `FRAMEWORK_GAMES`。Per-game checklist 自己 hardcode `GAME = "..."`,不用这些 fixture |

### 12.5 规则不变量与 `state_serializer` 的"测试可见性"

每个 per-game checklist 都应该包含一个 `TestRuleInvariants` 类——通过 `run_random_episode_states` 驱动随机对局,逐步从 `state_dict` 上断言**这个游戏自己的守恒律**(token 总量、卡总量、容量上限、可达性等)。完整模式与例子见 [新游戏验收测试指南 § 第 11 步](NEW_GAME_TEST_GUIDE.md#第-11-步规则不变量强制)。

实现这一步时常常会发现需要让 `serialize_<game>` 暴露当前不暴露的字段——典型例子:Azul 的 `box_lid`(完成图案行后回收的瓷砖)、Love Letter 的 `set_aside_card`(开局抽走的那张卡)。这些都是隐藏字段,在 AI 决策中**不应该**被读取,但守恒律测试必须看到它们才能完成断言。

**模式**:把它们加进 `state_serializer` 的输出,**并在 C++ 注释里明确说明这是"测试可见性"**——belief tracker 和 encoder 都不能读它,因为它们只走公开 API,结构上就拿不到这些字段。这样既支持守恒律测试,又不破坏 [§AI Pipeline Independence](../CLAUDE.md) 里的隔离保证。

```cpp
// games/azul/azul_register.cpp
// Box lid contents (tiles returned from completed pattern rows / floor
// overflow). Exposed for tile-conservation invariants in test suites.
// Belief tracker / encoder must NOT read it — they go through the public
// API only.
std::vector<int> box_counts(kColors, 0);
for (auto tile : s.box_lid) {
  if (tile >= 0 && tile < kColors) box_counts[tile]++;
}
m["box_counts"] = std::any(box_counts);
```

### 12.6 ISMCTS 根采样必须尊重 tracker 已知信息

每个有 `belief_tracker` 的游戏要在 `tests/framework/test_ismcts_samples_respect_tracker.py` 里加一个 checker:`belief_tracker.randomize_unseen` 给 MCTS 仿真填充隐藏槽位时,必须只填 viz=0 的槽位（viz=1 槽位是已知约束，不能随机覆盖——例如 Love Letter Priest peek 之后那个对手 hand 槽位 viz=1，已知 cid 不能再被采样改写）。

测试通过 `dinoboard_engine.test_belief_tracker(...)` 拿到 `belief_snapshot` 和 `trial_states[t]`(每次 `randomize_unseen` 后的完整 GT-style state dict),逐次比对。具体模式见 [新游戏验收测试指南 § 8f-ter](NEW_GAME_TEST_GUIDE.md#8f-ter-ismcts-根采样必须尊重-tracker-的已知声明强制)。

---

## 13. 完整 Checklist

### 必须完成

- [ ] 创建 `games/<name>/` 目录
- [ ] 实现 `<name>_state.h/.cpp`（继承 `CloneableState<T>`，实现 6 个必须方法）
- [ ] 实现 `<name>_rules.h/.cpp`（继承 `IGameRules`，实现 4 个必须方法）
- [ ] 实现 `<name>_net_adapter.h/.cpp`（继承 `IFeatureEncoder`，实现 3 个必须方法）
- [ ] 创建 `<name>_register.cpp`（GameRegistrar + 工厂函数）
- [ ] 创建 `config/game.json`（game_id、action_space、feature_dim 必须精确）
- [ ] 创建 `games/<name>/CMakeLists.txt`
- [ ] 在 `games/manifest.json` 追加一项 `{ "id": "<name>", "sources": [...] }`
- [ ] 构建通过：`pip install -e .`
- [ ] 验证注册：`python -c "import dinoboard_engine; print(dinoboard_engine.available_games())"`

### 必须完成（续）

- [ ] 实现 `state_serializer`（Web 前端需要）
- [ ] 实现 `action_descriptor`（Web 前端需要）
- [ ] 创建 `web/` 前端（玩家游玩 + 验收训练结果的主要界面）
- [ ] **写 `tests/<name>/test_checklist.py`**——从最相近的现有游戏复制模板,改 `GAME = "..."`。这是「游戏 ready」的明确信号:`pytest tests/<name>/` 一次全绿即合格
- [ ] 创建 `tests/<name>/__init__.py`(空文件,pytest 同名 `test_checklist.py` 消歧需要)
- [ ] 通过手动验证：按 [NEW_GAME_TEST_GUIDE.md](NEW_GAME_TEST_GUIDE.md) 逐步验证
- [ ] 全量测试通过：`python -m pytest tests/ -x -q`
- [ ] **通过 AI API 分离验收**（强制——证明 AI 不从 ground truth 偷看信息的唯一机制；不过这一步即使其他测试全过也不算合格）：
  - 所有游戏：API 契约测试在 `tests/<name>/test_checklist.py` 里覆盖
  - 随机游戏额外：belief 等价(belief / public state / legal actions)在同一份 checklist 里覆盖
  - 详见 §17
- [ ] 运行首次训练：`python -m training.cli --game <name> --output runs/<name>_001`
- [ ] 部署模型：将 `runs/<name>_001/models/model_best.onnx` 复制到 `games/<name>/model/<name>_<N>p.onnx`，其中 `<N>p` 是变体标识（2p / 3p / 4p）。同一游戏的所有变体模型共用 `games/<base>/model/`，不再为每个变体建单独目录。例如 Azul 的三个模型并排放在 `games/azul/model/{azul_2p,azul_3p,azul_4p}.onnx`。
- [ ] 创建 `config/web.json`（可选，配置 Web AI 参数：难度覆盖、温度、动作过滤、残局求解）
- [ ] **写 `docs/games/<name>_api.md`**——面向第三方接入者的 AI API 说明。至少包含：动作空间编码表、公开事件格式与发放时机、一个完整 curl/python 示例。模板见 `docs/games/tictactoe_api.md`（最简）和 `docs/games/splendor_api.md`（带事件复杂例）。**这份文档是游戏的"外部接口规范"**——没它第三方无法把你的 AI 接入他们的游戏 / 网站

### 推荐完成

- [ ] 实现 `heuristic_picker`（加速训练收敛）

### 进阶可选

- [ ] 实现 `adjudicator`（长局游戏需要）
- [ ] 实现 `auxiliary_scorer`（提供额外训练信号）
- [ ] 实现 `training_action_filter`（动作空间过大时加速学习）
- [ ] 实现 `episode_stats_extractor`（自定义训练监控）
- [ ] 启用 `tail_solver`（残局求解，需要 `do_action_deterministic`）
- [ ] 实现 `IBeliefTracker`（隐藏信息游戏）

---

## 14. AI API 分离验收 —— 信息泄漏的唯一证明

> **这一节的硬性要求**：一个新游戏的 AI 实现不通过本节所有测试就不能算验收合格，哪怕 selfplay 能跑、ONNX 能导出、Web 对局能完成。
>
> **为什么**：CLAUDE.md 的核心设计原则要求 AI 链路只依赖观察历史、不读真实隐藏 state。但这个要求靠 code review 根本无法保证——一个 `checked_cast` 加一行隐藏字段读取就能悄悄泄漏，审代码时很容易漏掉。**唯一能确定性证明"AI 没有偷看"的方法**就是：把 AI 放到一个独立的 session 里，用和 ground truth **不同的 seed** 初始化（两边的内部隐藏 state 完全不同），只通过动作序列 + 公开事件同步，最后检查 AI 维护的 belief 是否与 ground truth 的自博弈 belief 逐步等价。
>
> 如果 AI 代码里有任何对真实 state 的隐藏读取，这个测试会把它暴露为 belief 发散、公开 state 发散、或 AI 给出非法动作。反之，如果三个断言全过，就在信息论层面证明了 AI 决策只依赖观察序列——这正是设计原则要求的"分离"。

框架提供一个观察驱动的 AI 推理 API（`platform/ai_service/`），把 AI 决策暴露为 HTTP 端点。外部调用者只通过动作 ID 与 AI 交互，任何 state 字段都不会跨越边界。这同时也是未来对接第三方数字化桌游团队的接口。

### 14.1 两道门槛

两层测试都通过才算合格：

**第一层**：`tests/framework/test_ai_api_separation.py::test_full_game_via_api[<game_id>]`（所有游戏）
- API 契约干净（没有 state 进，没有 state 出）——由响应字段白名单扫描保证
- 给定初始设置 + 动作序列，AI 能端到端完成对局
- AI 返回的动作永远在 ground truth 的合法动作集里

**第二层**：`tests/framework/test_api_belief_matches_selfplay.py`（随机游戏）——这是真正的分离证明
- AI 用独立 seed 从零启动
- 只通过公开事件同步
- 每步的 belief snapshot 必须与自博弈维护的 belief 完全相等
- 终局公开 state 必须完全相等
- perspective 回合的 legal actions 必须完全相等

### 14.2 新游戏需要做什么

> 注意:`tests/framework/` 在固定 3 游戏 matrix 上跑(见 §15.1),它**不会自动跑你的新游戏**。下面的 "把 game_id 加入 _PLY_BUDGET / _DETERMINISTIC_GAMES / GAMES_WITH_EVENT_PROTOCOL / _PUBLIC_KEYS" 仅当你想让框架层也用你的游戏作为 carrier 时才需要——这是**项目维护层面**的决定,通常新游戏只在 `tests/<game>/test_checklist.py` 里完成验收即可。

**确定性游戏**（无 `belief_tracker`）：
1. 在 `tests/<your_game>/test_checklist.py` 里覆盖 API 分离场景(参考 `tests/quoridor/test_checklist.py`)
2. 跑 `pytest tests/<your_game>/ -v` 全通过

**随机或信息不对称游戏**（有 `belief_tracker`）：
1. 上面 2 步
2. 实现 `IBeliefTracker::serialize()` — 输出 canonical 可对比字典（sorted set → vector）
3. 实现 public-event 协议（§14.3），在 GameBundle 注册：
   - `public_event_extractor` — GT 侧：`(state_before, action, state_after, perspective) → PublicEventTrace { events, public_snapshot }`
   - `public_state_applier` — observer 侧：把 `public_snapshot` 整体写回 session state 的公开字段
   - `initial_observation_extractor` / `initial_observation_applier` — 初始设置同步
4. 在 `tests/<your_game>/test_checklist.py` 里加一个 `TestApiBeliefEquivalence` 类(参考 `tests/loveletter/` / `tests/splendor/` / `tests/coup/test_checklist.py`),调用 `assert_api_belief_matches_selfplay(GAME, PUBLIC_KEYS)` —— 这个 helper 一次完成三层等价断言(belief snapshot 每步一致 / 公开 state 字段终局相等 / perspective 回合 legal actions 相等),不需要重新实现

### 14.3 Public-Event 协议设计

GT 端每步产出 `PublicEventTrace { events, public_snapshot }`：
- **`public_snapshot`** — 动作之后所有公开 slot 的值（按 schema field name 索引）。observer 收到后通过 `public_state_applier` **整体覆写** session state 的公开部分。observer 路径上**不调 `do_action_fast`**——公开局面完全由 snapshot 重建，不存在"AI 用错的采样隐藏算公开"这条路径。
- **`events`** — 这次 transition 里 observer 能看到的公开事实序列（按 producer 发出的顺序），**只喂给 belief tracker** 用来增量更新对隐藏信息的推断（对手手牌、deck 内容、claim 历史等）。事件不会作用到 state 上；observer 不区分动作前后，list 内顺序就是 producer 怎么发就怎么发。

**事件 shape**：`{"kind": str, "payload": dict}`，kind 和 payload 结构由每个游戏定义。

**`apply_observation(action, events, public_snapshot)`**：observer 侧统一入口，顺序：
1. `state.begin_step_for_session_observe()` — `step_count_` 单调递增，DAG 防环（这是 session 路径上唯一一处直接 bump step_count 的合法入口；rules 路径走 `IGameRules` wrapper）
2. `public_state_applier(state, public_snapshot)` — 公开字段整体覆写
3. `belief_tracker.observe_public_event(actor, action, events)` — tracker 从事件流增量更新

session 上 viz=0 槽位**不动**——决策侧的 hash 用 `kHiddenHashSentinel`、encoder 用 `kPlaceholder`、MCTS sim 入口克隆 tracker 后调 `randomize_unseen`，结构性读不到 session 的 viz=0 内容。

**可见性过滤**：ground truth 端实现 `public_event_extractor` 时决定给 perspective 看什么。比如对手盲抽一张卡，事件只传 `{"player": 1}`（不带牌面），AI 知道"发生过抽牌"但不知道具体卡。框架不做 firewall——ground truth 愿意多传也可以（对接友好）。

**初始观察**：`initial_observation_extractor` 输出 perspective 视角能看到的开局信息（比如 Love Letter 的 `my_hand`、Splendor 的 `tableau` + `nobles`）。`initial_observation_applier` 在 API session 启动时 apply。

### 14.4 参考实现

| 游戏 | 事件类型 | 关键特点 | 位置 |
|------|---------|---------|------|
| Azul | `factory_refill` | 不注册 tracker，事件仍发但无消费方;snapshot 是 observer 同步公开局面的唯一通道 | `games/azul/azul_register.cpp` |
| Splendor | `deck_flip`, `self_reserve_deck` | tracker 维护 seen_cards，盲预订时 AI 需要知道自己抽了什么 | `games/splendor/splendor_register.cpp` |
| Love Letter | 仅淘汰 / reset 性质事件，精确知识走 viz | Priest/Baron/King/Prince 的对手手牌信息由 rules 直接通过 `reveal_slot_to` / `swap_slot_owned` 写进 viz，snapshot 自带这些 cid；events 只在结构层面（座位淘汰、reveal 槽位变更通知 tracker 重算多重集）发声，tracker 不靠 events 重建 hand | `games/loveletter/loveletter_register.cpp` |

**编写事件协议时的 checklist**：
- [ ] GT 端 `do_action_fast` 每处读 / 写隐藏字段的地方（翻牌、抽牌、对手手牌揭露），`extract_events` 都要发对应事件
- [ ] `advance_turn` 之类的子流程也要审一遍
- [ ] 事件只暴露 perspective 能公开看到的信息；不可见的事件仍要发（让 AI 知道"发生过"），但 payload 不含牌面
- [ ] tracker 的 `observe_public_event` 把事件折成对手信息集的更新——observer 路径不重放规则，state 公开字段靠 `public_snapshot` 整体覆写
