# 配置文件参考

本文档汇总 DinoBoard 各 JSON 配置文件的字段说明，作为开发新游戏或调整训练参数时的速查表。

- 训练 / 自博弈 / arena / eval：`games/<game>/config/game.json` 顶层 + `mcts_profiles.{selfplay, arena, eval}` + `training`（训练循环旋钮）
- Web 平台 / REST AI API：`games/<game>/config/web.json` 顶层 + `mcts_profiles.{web_expert, web_casual, analysis}`

上层概念与流水线介绍见 [GAME_DEVELOPMENT_GUIDE.md](GAME_DEVELOPMENT_GUIDE.md)。

---

## 整体架构：六个命名 MCTS profile

历史上 MCTS 旋钮散落在五处（game.json `training.*`、arena/eval **隐式复用** training、web.json 顶层、`difficulty_overrides`、代码硬编码），每加一个旋钮都要改 4–5 处。现在统一收敛到 **六个命名 profile**，每个 profile 覆盖一组完整且自洽的 MCTS 参数：

| profile 名 | 文件位置 | 用途 |
|---|---|---|
| `selfplay` | `game.json mcts_profiles` | 自博弈生成训练数据 |
| `arena` | `game.json mcts_profiles` | latest vs best gating + cross-ONNX eval |
| `eval` | `game.json mcts_profiles` | benchmark eval（vs heuristic / vs ONNX） |
| `web_expert` | `web.json mcts_profiles` | Web 难度 = expert / REST AI API |
| `web_casual` | `web.json mcts_profiles` | Web 难度 = casual |
| `analysis` | `web.json mcts_profiles` | 录像分析 / precompute / 智能提示 |

callsite 按 profile 名索引（`resolve_profile(game_id, "arena")`），不再有"偷字段"的隐式复用。新加 MCTS 旋钮只改 `MctsProfile` dataclass 一处（`training/mcts_profile.py`）。

### Profile 共享字段集

每个 profile（不管定义在哪个文件）都包含同一组字段：

| 字段 | 类型 | 说明 |
|---|---|---|
| `simulations` | int | MCTS 单步模拟次数 |
| `c_puct` | float | PUCT 探索常数 |
| `temperature` | float | 基础温度（schedule 关闭时使用） |
| `temperature_schedule.enabled` | bool | 是否启用线性温度衰减 |
| `temperature_schedule.initial` | float | 衰减起点 |
| `temperature_schedule.final` | float | 衰减终点 |
| `temperature_schedule.decay_plies` | int | 衰减步数 |
| `dirichlet_alpha` | float | 根节点 Dirichlet 噪声 alpha |
| `dirichlet_epsilon` | float | 噪声混合比例 |
| `dirichlet_on_first_n_plies` | int | 仅前 N 步加噪 |
| `opponent_selection` | str | `"puct"` 或 `"prior"`（见下） |
| `tail_solve_enabled` | bool | 是否在终局阶段启用 alpha-beta |
| `tail_solve_depth_limit` | int | alpha-beta 搜索深度上限 |
| `tail_solve_node_budget` | int | alpha-beta 节点预算 |
| `tail_solve_margin_weight` | float | 终局评估分差加权 |
| `ai_use_action_filter` | bool | 是否遵循 `training_action_filter`（仅对注册了 filter 的游戏有意义） |
| `cover_root_edges` | bool | 是否在 PUCT 之前先把每条 root legal edge 至少 visit 一次（分析 pipeline 用 true，其它 false） |

`opponent_selection`：
- `"puct"`（默认）：所有节点（含对手节点）都走 UCB/PUCT bandit
- `"prior"`：对手节点改为按 policy 先验做 multinomial sampling（Smooth-UCT 风格）。缓解 ISMCTS strategy fusion / 对手全知偏置——尤其对 Love Letter / Coup 这类藏牌博弈有用。**根节点（自己的决策节点）始终走 PUCT**，无论此设置。

`tail_solve_enabled` 硬约束：profile `tail_solve_enabled = true` 时该 game 必须注册 `tail_solve_trigger`，否则 resolver 启动时抛 `ValueError`。

### 继承机制（`inherits`）

每个 profile 都可以 `inherits` 另一个 profile（同文件或跨文件），未指定的字段从父 profile 继承；BASE 默认值见 `training/mcts_profile.py:_BASE_DICT`。继承允许多层链，循环时抛错。跨 game.json 与 web.json 的继承允许（例如 `analysis` 可以 `inherits: "selfplay"`），但同名 profile 在两边同时出现是硬错误。

### temperature_schedule 形状

JSON 中**强制嵌套**：

```json
"temperature_schedule": { "enabled": true, "initial": 1.0, "final": 0.15, "decay_plies": 40 }
```

旧的扁平形式（`temperature_initial: 1.0`）被显式拒绝。

---

## game.json

**位置**：`games/<game>/config/game.json`

训练 pipeline 自动发现此文件（`training/cli.py find_game_config()`）。

### 顶层字段

| 字段 | 类型 | 必须 | 说明 |
|---|---|---|---|
| `game_id` | string | 是 | 必须和 GameRegistrar 注册的 id 一致 |
| `display_name` | string | 是 | 显示名称 |
| `players.min` | int | 是 | 最少玩家数 |
| `players.max` | int | 是 | 最多玩家数 |
| `action_space` | int | 是 | 动作空间大小（必须和 encoder 一致） |
| `feature_dim` | int | 是 | 特征维度（必须和 encoder 一致） |
| `max_game_plies` | int | 是 | 最大步数（超出后调用 adjudicator 或判和）。**所有 profile 共享，作为游戏属性放顶层** |
| `mcts_profiles` | object | 是 | 见上文，至少要包含 `selfplay` / `arena` / `eval` 三个 profile |
| `training` | object | 是 | 训练循环参数（不含 MCTS 旋钮） |
| `network` | object | 是 | 网络结构 |

### `training` 字段（训练循环参数，不含 MCTS）

| 字段 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `steps` | int | 1000 | 总训练步数 |
| `episodes_per_step` | int | 200 | 每步自我对弈局数 |
| `simulations_start` | int | =selfplay.simulations | 训练初始 MCTS 模拟次数，前 30% 步线性爬到 selfplay profile 的 simulations |
| `batch_size` | int | 512 | SGD mini-batch 大小 |
| `learning_rate` | float | 0.001 | AdamW 学习率 |
| `weight_decay` | float | 1e-4 | AdamW weight decay |
| `train_batches_per_step` | int | 3 | 每步从 replay buffer 采样训练的 mini-batch 数 |
| `grad_clip_norm` | float | 1.0 | 梯度裁剪范数（0 表示不裁剪） |
| `gating_accept_win_rate` | float | N 人自适应 | latest vs best 晋升阈值 |
| `eval_benchmarks` | list[str] | `null` | benchmark 列表 |
| `heuristic_temperature` | float | 0.0 | eval vs heuristic 时启发式对手的温度 |
| `free_heuristic_temperature` | float | 0.0 | eval `heuristic_free` 模式的温度 |
| `heuristic_guidance_*` | — | — | heuristic guidance schedule（见下） |
| `training_filter_*` | — | — | training action filter schedule（见下） |
| `auxiliary_score` | bool | false | 启用辅助训练头 |
| `auxiliary_score_weight` | float | 0.5 | 辅助损失权重 |

> **`eval_temperature` 字段已删除**：其语义并入 `mcts_profiles.eval.temperature`。
> **`tail_solve_start_ply` 字段已删除**：触发条件由游戏注册的 `tail_solve_trigger` 决定，不再用步数阈值兜底。

#### Heuristic Guidance schedule

| 字段 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `heuristic_guidance_hold_steps` | int | 0 | 前 N 步 ratio 锁在 `initial_ratio` |
| `heuristic_guidance_steps` | int | 0 | 衰减终点；step ≥ 此值后 ratio = 0 |
| `heuristic_guidance_initial_ratio` | float | 0.5 | 初始概率（使用 heuristic 而非 MCTS） |
| `heuristic_guidance_temperature` | float | 0.0 | selfplay 启发式分支的温度 |

三段式 schedule：`step ≤ hold_steps` → ratio = `initial_ratio`；`step ≥ heuristic_guidance_steps` → ratio = 0；中间线性衰减。`hold_steps` 必须严格小于 `heuristic_guidance_steps`。

#### Training Filter schedule

| 字段 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `training_filter_steps` | int | 0 | 动作过滤的训练步数（0 表示禁用） |
| `training_filter_initial_ratio` | float | 0.5 | 初始 ratio |
| `training_filter_hold_steps` | int | 0 | hold 期 |

每句 effective filter ratio = `schedule_ratio if profile.ai_use_action_filter else 0.0`。把 `selfplay.ai_use_action_filter` 关掉就能整体禁用 filter（保留 schedule 配置，便于 ablation）。

### `network` 字段

| 字段 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `hidden_layers` | int[] | `[256, 256]` | 隐层大小列表 |

> 激活函数硬编码 ReLU，结构固定 MLP。Value head 维度按 `num_players` 自动决定。

### 完整示例（Quoridor）

```json
{
  "game_id": "quoridor",
  "display_name": "Quoridor",
  "players": {"min": 2, "max": 2},
  "action_space": 209,
  "feature_dim": 295,
  "max_game_plies": 200,

  "mcts_profiles": {
    "selfplay": {
      "simulations": 800,
      "c_puct": 1.4,
      "temperature": 1.0,
      "temperature_schedule": {"enabled": true, "initial": 1.0, "final": 0.15, "decay_plies": 24},
      "dirichlet_alpha": 0.05,
      "dirichlet_epsilon": 0.1,
      "dirichlet_on_first_n_plies": 8,
      "opponent_selection": "puct",
      "tail_solve_enabled": false,
      "tail_solve_depth_limit": 10,
      "tail_solve_node_budget": 200000,
      "tail_solve_margin_weight": 0.01,
      "ai_use_action_filter": true,
      "cover_root_edges": false
    },
    "arena": {
      "inherits": "selfplay",
      "temperature": 0.1,
      "temperature_schedule": {"enabled": false, "initial": 0.0, "final": 0.0, "decay_plies": 0},
      "dirichlet_epsilon": 0.0,
      "ai_use_action_filter": false
    },
    "eval": { "inherits": "arena" }
  },

  "training": {
    "steps": 1500,
    "episodes_per_step": 100,
    "simulations_start": 100,
    "batch_size": 2048,
    "learning_rate": 0.001,
    "heuristic_guidance_hold_steps": 8,
    "heuristic_guidance_steps": 500,
    "heuristic_guidance_initial_ratio": 1.0,
    "heuristic_guidance_temperature": 3.0,
    "auxiliary_score": true,
    "auxiliary_score_weight": 0.5
  },

  "network": { "hidden_layers": [256, 256, 256] }
}
```

### `gating_accept_win_rate` 自适应阈值

N 人零和游戏的零假设胜率是 `1/N`，不是 0.5。框架未显式配置时自动采用：

```
threshold = 1/N + z · sqrt((1/N)·(1 - 1/N) / eval_games)
```

`z ≈ 0.632` 经校准，`eval_games=40` 时大致落点：2p ≈ 0.55、3p ≈ 0.38、4p ≈ 0.29。

### Tail Solve 注意事项

> **是否启用？** alpha-beta 在每个搜索节点调用 `legal_actions()`，开销 = `node_budget × legal_actions 单次耗时`。分支因子 < 20 且 `legal_actions` 廉价（如 TicTacToe）→ 推荐。分支因子 > 50 或 `legal_actions` 含 BFS（如 Quoridor）→ 不推荐用于 selfplay，但用于 web expert（人类等待 800ms 可接受）OK。
>
> 实测开销可用 `dinoboard_engine.tail_solve(game_id, seed, perspective_player, depth_limit, node_budget)` 的 `elapsed_ms` / `nodes_searched`。

> **隐藏信息游戏的 tail solver 硬约束**：
> 1. 必须 override `do_action_deterministic`，保证不从隐藏源抽取数据（参考 Splendor 的 `forced_draw_override`）
> 2. 在 GameBundle 中显式设置 `stochastic_tail_solve_safe = true`
>
> `GameRegistry::create_game()` 在检测到 belief_tracker + tail_solver 而没有 `stochastic_tail_solve_safe = true` 时抛异常。

---

## web.json

**位置**：`games/<game>/config/web.json`

| 字段 | 类型 | 必须 | 说明 |
|---|---|---|---|
| `mcts_profiles` | object | 是 | 至少包含 `web_expert` / `web_casual` / `analysis` 三个 profile |

### 难度 → profile 映射

| Web 难度 | profile 名 |
|---|---|
| `heuristic` | （无 profile，直接调 `get_heuristic_action()`） |
| `casual` | `web_casual` |
| `expert` | `web_expert` |
| 录像分析 / precompute | `analysis` |

### 完整示例（Quoridor）

```json
{
  "mcts_profiles": {
    "web_expert": {
      "simulations": 5000,
      "c_puct": 1.4,
      "temperature": 0.0,
      "temperature_schedule": {"enabled": false, "initial": 0.0, "final": 0.0, "decay_plies": 0},
      "dirichlet_alpha": 0.0,
      "dirichlet_epsilon": 0.0,
      "dirichlet_on_first_n_plies": 0,
      "opponent_selection": "puct",
      "tail_solve_enabled": true,
      "tail_solve_depth_limit": 10,
      "tail_solve_node_budget": 200000,
      "tail_solve_margin_weight": 0.0,
      "ai_use_action_filter": false,
      "cover_root_edges": false
    },
    "web_casual": {
      "inherits": "web_expert",
      "simulations": 10,
      "tail_solve_enabled": false
    },
    "analysis": {
      "inherits": "web_expert",
      "cover_root_edges": true
    }
  }
}
```

### 分析 pipeline 的特殊行为（与 AI 实战的区别）

录像里"掉点"那一栏走 `analysis` profile。和 AI 实战是**两份独立 session**：

1. **始终 unfiltered**：`engine.GameSession(..., False)`。原因：人类不受 filter 约束、可以走 filter 外的合法动作；如果分析 session 也带 filter，那一手不会进搜索树，`drop_score` 会**静默**报 0（看着像最佳着法）。
2. **始终 `cover_root_edges = true`**：在 PUCT 之前先把每条 root legal edge 至少 visit 一次，保证 `action_values[a]` 对所有 legal `a` 都是真值。AI 实战路径不开这个 flag，预算全给 PUCT。
3. **opponent_selection 恒走 `"puct"`**：跨难度自洽（drop-score / smart-hint 不应受 selection mode 影响）。
4. **不影响 belief / 隐藏信息处理**：用的还是同一棵 search tree、同一套 root determinization。

`analysis.cover_root_edges = true` 是这条规则在 profile 里的体现；`web_expert` / `web_casual` 都设 false，行为分开。

---

## REST AI API 与 web profile

`POST /ai/sessions` 等接口的强度（simulations / temperature / opponent_selection）由服务器从 `web_expert` profile 读出，**客户端不能传**——`platform/ai_service/sessions.py:_resolve_strength` 是唯一入口。如果客户端在 body 里塞 `simulations` / `temperature`，pydantic 会丢弃。

详细语义见 [AI_API.md](AI_API.md)。
