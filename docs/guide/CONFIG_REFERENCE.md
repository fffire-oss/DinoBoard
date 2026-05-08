# 配置文件参考

本文档汇总 DinoBoard 各 JSON 配置文件的字段说明，作为开发新游戏或调整训练参数时的速查表。

- 训练 / 自博弈 / eval / 残局求解：`games/<game>/config/game.json`
- Web 平台（AI 难度、tail-solve、动作过滤）：`games/<game>/config/web.json`

上层概念与流水线介绍见 [GAME_DEVELOPMENT_GUIDE.md](GAME_DEVELOPMENT_GUIDE.md)。

---

## game.json

**位置**：`games/<game>/config/game.json`

训练 pipeline 自动发现此文件（通过 `training/cli.py` 中的 `find_game_config()`）。

### 顶层字段
| 字段 | 类型 | 必须 | 说明 |
|------|------|------|------|
| `game_id` | string | 是 | 必须和 GameRegistrar 注册的 id 一致 |
| `display_name` | string | 是 | 显示名称 |
| `players.min` | int | 是 | 最少玩家数 |
| `players.max` | int | 是 | 最多玩家数 |
| `action_space` | int | 是 | 动作空间大小（必须和 encoder 一致） |
| `feature_dim` | int | 是 | 特征维度（必须和 encoder 一致） |

---

### training 字段
#### MCTS 搜索参数

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `simulations` | int | 200 | 最终 MCTS 模拟次数（eval 也用此值） |
| `c_puct` | float | 1.4 | PUCT 探索常数 |
| `temperature` | float | 1.0 | 基础温度（未启用 schedule 时使用） |
| `temperature_initial` | float | -1 | 温度衰减起始值（-1 表示使用 temperature） |
| `temperature_final` | float | -1 | 温度衰减终止值 |
| `temperature_decay_plies` | int | 0 | 温度从 initial 线性衰减到 final 的步数 |
| `dirichlet_alpha` | float | 0.3 | Dirichlet 噪声 alpha 参数 |
| `dirichlet_epsilon` | float | 0.25 | 根节点先验中噪声的比例 |
| `dirichlet_on_first_n_plies` | int | 30 | 只在前 N 步添加 Dirichlet 噪声 |
| `max_game_plies` | int | 500 | 最大步数（超出后调用 adjudicator 或判和） |

**温度调参建议**：
- `alpha` 的经验法则：`alpha ≈ 10 / action_space`
- 简单游戏（TicTacToe）：alpha=1.0, decay_plies=6
- 复杂游戏（Quoridor）：alpha=0.05, decay_plies=24

#### 训练循环参数

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `steps` | int | 1000 | 总训练步数 |
| `episodes_per_step` | int | 200 | 每步自我对弈局数 |
| `batch_size` | int | 512 | SGD mini-batch 大小 |
| `learning_rate` | float | 0.001 | AdamW 优化器学习率 |
| `weight_decay` | float | 1e-4 | AdamW weight decay |
| `train_batches_per_step` | int | 3 | 每步从 replay buffer 随机采样训练的 mini-batch 数 |
| `grad_clip_norm` | float | 1.0 | 梯度裁剪范数（0 表示不裁剪） |

#### MCTS Schedule 参数

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `simulations` | int | 200 | 最终 MCTS 模拟次数（也用于 eval） |
| `simulations_start` | int | =simulations | 训练初始 MCTS 模拟次数，线性递增到 `simulations` |

如果 `simulations_start` 未设置或等于 `simulations`，则模拟次数全程固定（向后兼容）。

递增公式：`sims = start + min(1.0, step / (steps * 0.3)) * (simulations - start)`

即在前 30% 步数内线性爬坡到 `simulations`，之后固定。这样训练早期的自博弈更快，能更快形成有效的 replay buffer。

**注意**：eval 对弈始终使用 `simulations`（最终值），确保评估标准一致。

#### Eval 与 Gating 参数

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `gating_accept_win_rate` | float | N 人自适应 | latest vs best 的晋升阈值。不显式设置时，框架按 `1/N` 零和基线 + 同等置信区间自动推算（见下方说明）。 |
| `eval_temperature` | float | 0.0 | gating 和 ONNX benchmark 对局的动作选择温度。0=贪心，>0 引入随机性。确定性游戏建议 0.1 |

每 `--eval-every` 步触发一轮评估，包含两部分：

**Benchmark eval**（通过 CLI `--eval-benchmark` 配置，可同时指定多个）：

| Benchmark 值 | 说明 |
|-------------|------|
| `heuristic_constrained` | 模型 vs heuristic，模型受 action filter 约束 |
| `heuristic_free` | 模型 vs heuristic，模型不受 action filter 约束 |
| ONNX 文件路径 | 模型 vs 指定 ONNX 模型 |

示例：
```bash
# Quoridor: constrained + free + gating
python3 -m training.cli --game quoridor --output runs/quoridor_v12 \
  --eval-benchmark heuristic_constrained heuristic_free

# 不传 --eval-benchmark，只跑 gating
python3 -m training.cli --game quoridor --output runs/quoridor_v12
```

**Gating eval**（固定执行，不受 `--eval-benchmark` 影响）：latest vs best 对打 `--eval-games` 局，胜率 ≥ `gating_accept_win_rate` 时 `shutil.copy2` 更新 `model_best.onnx`。

**多人游戏的默认阈值**：N 人零和游戏的零假设胜率是 `1/N`，不是 0.5。固定 0.55 对 2 人合适但对 3p/4p 过于宽松（null 已在 0.333 / 0.25）。框架未显式配置时自动采用：

```
threshold = 1/N + z · sqrt((1/N)·(1 - 1/N) / eval_games)
```

`z ≈ 0.632` 经校准使 (2p, 40 局) 回到历史上的 0.55，保证各人数下相对 null 有同等单边置信超出。`eval_games=40` 时大致落点：2p ≈ 0.55、3p ≈ 0.38、4p ≈ 0.29。除非确有更严或更松的业务理由，**不要在 3p/4p 游戏的 `game.json` 里硬写 0.55**——那是"永远通不过"的门槛。需要手工覆盖时在 `training.gating_accept_win_rate` 里写明白值即可。

训练过程中维护两个模型文件：
- **latest** (`model_latest.onnx`)：每步训练后都重新导出，selfplay 立刻使用新权重，**永不被替换或回退**
- **best** (`model_best.onnx`)：独立文件，只在 gating 通过时从 latest 复制过来

Selfplay 始终使用 latest 模型。Gating 只影响 best 模型的保存。定期存档 `model_step_NNNNN.onnx` 按 `--save-every` 间隔保存，用于事后实验，不参与训练流程。

#### Tail Solve 参数

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `tail_solve_enabled` | bool | false | 启用残局求解 |
| `tail_solve_start_ply` | int | 40 | 最早尝试求解的步数 |
| `tail_solve_depth_limit` | int | 5 | alpha-beta 最大深度 |
| `tail_solve_node_budget` | int | 10000000 | 最大搜索节点数 |
| `tail_solve_margin_weight` | float | 0.0 | 终局评估加入分差系数（需配合 auxiliary_scorer） |

> **是否启用残局求解？** alpha-beta 在每个搜索节点调用 `legal_actions()`，因此开销 = `node_budget × legal_actions 单次耗时`。启用前评估你的游戏：
> - 分支因子 < 20 且 `legal_actions` 廉价（如 TicTacToe）→ 推荐启用
> - 分支因子 > 50 或 `legal_actions` 含 BFS/连通性检查（如 Quoridor ~130 分支，~35μs/次）→ 不推荐，200k 预算下单次求解耗时可达 800+ms，远超 800 次 MCTS simulation（~28ms）
>
> 要实测你的游戏的 tail solve 开销，可用 `dinoboard_engine.tail_solve(game_id, seed, perspective_player, depth_limit, node_budget)` 在 Python 中直接调用，读取 `elapsed_ms` 和 `nodes_searched`。

> **隐藏信息游戏的 tail solver 硬约束**：如果游戏同时注册了 `belief_tracker`（有非对称隐藏信息）和 `tail_solver`，必须：
> 1. override `do_action_deterministic`，保证其 NEVER 从隐藏源（deck / 袋子 / 对手手牌）抽取数据。如果 `do_action_fast` 的回合推进会翻随机牌，deterministic 版本要用占位或冻结逻辑（参考 Splendor 的 `forced_draw_override`）
> 2. 在 GameBundle 中显式设置 `stochastic_tail_solve_safe = true`，作为你已审阅过 `do_action_deterministic` 的声明
>
> `GameRegistry::create_game()` 在检测到 belief_tracker + tail_solver 而没有 `stochastic_tail_solve_safe = true` 时会抛异常。这是防止 tail solver 偷看真实状态污染训练数据（违反 "AI 链路不读取隐藏字段" 的核心设计原则）。

#### Heuristic Guidance 参数

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `heuristic_guidance_hold_steps` | int | 0 | 前 N 步 ratio 锁在 `initial_ratio`（hold 期，≥0） |
| `heuristic_guidance_steps` | int | 0 | 衰减终点；step ≥ 此值后 ratio = 0（0 表示禁用） |
| `heuristic_guidance_initial_ratio` | float | 0.5 | 初始概率（使用 heuristic 而非 MCTS） |
| `heuristic_guidance_temperature` | float | 0.0 | **selfplay** 启发式分支的温度（高温 → 多样性） |
| `heuristic_temperature` | float | 0.0 | **eval vs heuristic** 时启发式对手的温度（强度基准） |

三段式 schedule：
- `step ≤ hold_steps` → ratio = `initial_ratio`（hold 期）
- `step ≥ heuristic_guidance_steps` → ratio = 0
- 中间 → 线性衰减

`hold_steps` 必须严格小于 `heuristic_guidance_steps`，否则 pipeline 启动报错。设 `initial_ratio = 1.0` + 较大 `hold_steps` 可以让网络还弱的前 N 步全部用启发式引导自博弈，样本进 replay buffer 滚动训练；之后线性衰减让网络平滑接管。

注意 `heuristic_guidance_temperature` 与 `heuristic_temperature` 是两个独立配置：前者控制 selfplay 启发式分支（探索为主，常配 1.5–3.0），后者控制 eval 强度基准（常配 0.0）。policy target 对应实际使用的概率分布。

#### Training Filter 参数

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `training_filter_steps` | int | 0 | 动作过滤的训练步数（0 表示禁用） |
| `training_filter_initial_ratio` | float | 0.5 | 初始概率（应用 filter） |

在 `training_filter_steps` 步内，比例从 `initial_ratio` 线性衰减到 0。之后在全动作空间训练。

#### Peek 参数

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `peek_steps` | int | 0 | 前 N 个训练步用 peek 模式（跳过 root 采样，MCTS 看真相），之后切回 ISMCTS。0 表示始终 ISMCTS |

Peek 模式下 `ismcts_enabled=False`，selfplay 调 `run_selfplay_episode` 时 belief_tracker 传 `nullptr`——MCTS 不调 `randomize_unseen`，直接在 truth state 上搜索。适合训练早期让 value head 先学到基本策略结构，再切到 ISMCTS 学习在信息不完全下决策。仅影响 selfplay，arena/eval 始终使用 ISMCTS。

#### Auxiliary Score 参数

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `auxiliary_score` | bool | false | 启用辅助训练头 |
| `auxiliary_score_weight` | float | 0.5 | 辅助损失的权重 |

### network 字段
| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `hidden_layers` | int[] | `[256, 256]` | 隐层大小列表，如 `[256, 256, 256]` |

> 激活函数当前硬编码为 ReLU，网络结构固定为 MLP。如需其他结构/激活，修改 `training/model.py`。

> **Value head 维度**：自动由 `num_players`（从 C++ `game_metadata()` 获取）决定。输出 N 维 perspective-relative value，与 encoder 旋转对齐。无需在 config 中指定。

### 完整示例（game.json）
<details>
<summary>TicTacToe（最小配置）</summary>

```json
{
  "game_id": "tictactoe",
  "display_name": "Tic-Tac-Toe",
  "players": {"min": 2, "max": 2},
  "action_space": 9,
  "feature_dim": 28,
  "training": {
    "simulations": 100,
    "c_puct": 1.4,
    "temperature": 1.0,
    "temperature_initial": 1.0,
    "temperature_final": 0.1,
    "temperature_decay_plies": 6,
    "dirichlet_alpha": 1.0,
    "dirichlet_epsilon": 0.25,
    "dirichlet_on_first_n_plies": 9,
    "max_game_plies": 9,
    "episodes_per_step": 200,
    "batch_size": 256,
    "learning_rate": 0.001,
    "steps": 500
  },
  "network": {
    "hidden_layers": [64, 64]
  }
}
```
</details>

<details>
<summary>Quoridor（完整配置）</summary>

```json
{
  "game_id": "quoridor",
  "display_name": "Quoridor",
  "players": {"min": 2, "max": 2},
  "action_space": 209,
  "feature_dim": 295,
  "training": {
    "simulations": 200,
    "c_puct": 1.4,
    "temperature_initial": 1.0,
    "temperature_final": 0.15,
    "temperature_decay_plies": 24,
    "dirichlet_alpha": 0.05,
    "dirichlet_epsilon": 0.1,
    "dirichlet_on_first_n_plies": 8,
    "max_game_plies": 200,
    "episodes_per_step": 100,
    "batch_size": 2048,
    "learning_rate": 0.001,
    "steps": 1500,
    "heuristic_guidance_hold_steps": 8,
    "heuristic_guidance_steps": 200,
    "heuristic_guidance_initial_ratio": 1.0,
    "heuristic_guidance_temperature": 3.0,
    "tail_solve_enabled": true,
    "tail_solve_start_ply": 30,
    "tail_solve_depth_limit": 10,
    "tail_solve_node_budget": 200000,
    "tail_solve_margin_weight": 0.01,
    "heuristic_guidance_steps": 500,
    "auxiliary_score": true,
    "auxiliary_score_weight": 0.5
  },
  "network": {
    "hidden_layers": [256, 256, 256]
  }
}
```
</details>

## web.json — Web 平台配置
**位置**：`games/<game>/config/web.json`（可选，不存在则全部用默认值）

Web 平台相关的 AI 参数独立于训练配置，放在 `web.json` 中管理。

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `ai_use_action_filter` | bool | false | Web 对局时 AI 搜索是否遵循 `training_action_filter`。人类永远不受约束；只控制 AI（pipeline、ai-hint、precompute）。只有注册了 `training_action_filter` 的游戏才需要设置 |
| `analysis_simulations` | int | 5000 | 录像分析和 precompute 的 MCTS 模拟次数。对局较短或决策空间较小的游戏可以调低，减少分析延迟 |
| `difficulty_overrides` | object | {} | 按难度覆盖 AI 参数。键为难度名（casual/expert），值可包含 `simulations` 和 `temperature` |
| `tail_solve` | object | {} | 残局求解配置。子字段见下 |
| `tail_solve.enabled` | bool | false | 是否在 Web 对局中启用残局求解。只有注册了 `tail_solver` 的游戏才有效 |
| `tail_solve.depth_limit` | int | 10 | 残局求解搜索深度上限 |
| `tail_solve.node_budget` | int | 200000 | 残局求解节点预算 |

**全局默认值**（未配置时）：casual=10 sim, expert=5000 sim，温度均为 0，分析 5000 sim。heuristic 难度不走 MCTS，直接调用 `get_heuristic_action()`，无需配置 simulations。

**示例**（Coup 的 `config/web.json`）：
```json
{
  "analysis_simulations": 2000,
  "difficulty_overrides": {
    "casual": {"simulations": 50, "temperature": 0.3},
    "expert": {"simulations": 2000, "temperature": 0.1}
  }
}
```

**示例**（Quoridor 的 `config/web.json`，启用动作过滤和残局求解）：
```json
{
  "ai_use_action_filter": true,
  "tail_solve": {
    "enabled": true,
    "depth_limit": 10,
    "node_budget": 200000
  }
}
```

**何时需要配置**：
- 含隐藏信息或虚张声势的游戏（Coup、Love Letter）：加非零温度，避免 AI 完全确定性
- 使用 training_action_filter 的游戏（Quoridor）：设置 `ai_use_action_filter: true`
- 注册了 tail_solver 的完全信息游戏（Quoridor）：设置 `tail_solve.enabled: true`，AI 在终局阶段会尝试精确求解

**向后兼容**：如果 `web.json` 不存在，平台会回退读取 `game.json` 中的 `web` 字段和 `ai_use_action_filter`。新游戏应使用 `web.json`。

### 分析 pipeline 的特殊行为（与 AI 实战路径的区别）

录像里"掉点"那一栏来自分析 pipeline（`platform/game_service/pipeline.py`）。它和 AI 实战走 MCTS 的 GameSession 是**两份独立 session**，配置不一样——这条要在新游戏接入时记得：

1. **始终 unfiltered**。无论 `ai_use_action_filter` 是 true 还是 false，分析 session 一律用 `engine.GameSession(..., False)`。原因：人类不受 filter 约束、可以走 filter 外的合法动作；如果分析 session 也带 filter，那一手不会进搜索树，`drop_score` 会**静默**报 0（看着像最佳着法）。AI 实战仍按 `ai_use_action_filter` 走，强度不变。

2. **始终 `cover_root_edges=True`**。`get_ai_action(sims, temperature, cover_root_edges=True)`。该 flag 在 PUCT 之前先把每条 root legal edge 至少 visit 一次，保证 `action_values[a]` 对所有 legal `a` 都是真值，而不是 visit_count=0 时的默认 0（→ 50% 胜率假象）。AI 实战路径不开这个 flag，预算全给 PUCT。

3. **不影响 belief / 隐藏信息处理**。cover 用的还是同一棵 search tree、同一套 root determinization、同一条 backup 路径——和 PUCT 自然采到的那一手语义一致。

如果你新接入一款 `ai_use_action_filter: true` 的游戏，或合法动作空间特别大（比如 Quoridor 的 209 维）、5000 sim 都覆盖不全冷门 edge 的游戏，**不需要** 在配置里特别声明——以上行为是 pipeline 内部的硬规则，对所有游戏一视同仁。
