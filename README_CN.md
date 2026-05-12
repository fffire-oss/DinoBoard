# DinoBoard

**English version:** [README.md](README.md)

> **Drop a rulebook. Ship a superhuman AI.**
>
> 给 LLM 一句话「加上 Azul」，它读规则书、写 C++规则引擎、跑全量测试、训
> 练出 AI、生成 Web 前端。你只负责验收。Azul 的 AI 就是这么诞生
> 的——并且训练出的模型棋力已经**超越当前已知最强人类玩家**。这
> 条流程能跑通，是因为框架把每个接入点都做成了**LLM 容易写对、写
> 错了 CI 立刻抓住**的形状。

一个面向**通用桌游**的 AlphaZero 风格框架，目标是把"规则引擎 +
AI 决策 + Web 对战"做成端到端闭环、并在 LLM 协助开发下保持可信。

---

## 核心机制

三条 mechanism 服务一条动机：**让规则引擎和 AI 决策路径在 LLM 协助
开发下也对人类玩家可信**。

### 1. Per-field viz tensor

每个游戏在 `<game>_visibility.cpp` 声明字段名 + 数据 shape + base viz
tensor。`viz[..., p] = 1` 表示玩家 p 现在能看见这个槽位的真值。
rules 是 viz 唯一 writer——`do_action_fast` 里改业务字段的同时调
`reveal_slot` / `reveal_slot_to` / `reset_to_base` 维护 viz。

`make_masked_state(state, schema, perspective, belief_filled)` 走
schema 派一份 MaskedState，三家消费者（snapshot / hash / encoder
tensor）共享同一对象——viz=0 槽位结构性读到 placeholder。"对哪个
玩家可见"这件事变成 **state 字段**，不是 observer 实现内部的代码逻
辑。

跟 OpenSpiel Observer API 比，这条不是 OpenSpiel 做不到的事——他们
真实游戏（Gin Rummy 等）也用 visibility bitmap——只是 OpenSpiel 不
强制这种形状，作者写错了 round-trip 测试才会抓到；DinoBoard 把它
强制成所有游戏共享的基类设施，可以集中跑一组 CI 测试覆盖所有游戏。

### 2. GT 与 AI session 物理分离

selfplay / arena / web / API 都持一份 truth 状态用来推进游戏，**额
外**给每个 perspective 一份 session state。AI 决策路径（belief
tracker / encoder / MCTS）只读 session，物理上没把 truth 指针递进
去。`IBeliefTracker::randomize_unseen(state, observer, rng)` 接口签
名里没有 truth；MCTS root state 是 `per_seat_states[acting_player]`，
不是 truth。

OpenSpiel 上做 per-seat runner 也完全可行——只是 `ResampleFromInfostate`
是 `State` 成员，作者**有能力**读 truth，是 contractual safety；
DinoBoard 的接口里不传，是 structural safety。差别在 LLM / 经验不
足的作者身上才显现，对自觉作者接近 0。

### 3. 强制 Web 前端

每款游戏接进来必须有可玩 web 前端、可视化 AI 决策、回放工具。这不
是 UI 美学，是 AI 行为可信度的最终判据——training metric（win rate /
loss curve / policy entropy）正常但 AI 在第 7 回合明显送子，这种 bug
命令行看不出来，**人类对战 30 秒能感觉到**。

研究路线用 exploitability / NashConv 做 AI 验证比 web 对战更严格更
可复现——这是 OpenSpiel 的路线；DinoBoard 的目标用户不发 paper、需
要"做出来给真人玩、看 AI 行为对不对劲"，所以选 web 闭环作为最后一
道防线。

---

## 算法层

ISMCTS over DAG：

- **Root determinization**：每个 sim 从 belief tracker 采一个完整世
  界，descent 完全 deterministic——物理随机和信息不对称在搜索里统
  一处理
- **DAG 而非 tree**：`(state hash, current_player)` keying，同一 info
  set 从不同路径到达共享节点；UCT2（Childs 2008）多入边修正避免
  over-exploration
- **PUCT 选择**：AlphaZero 风格 prior 引导，神经网络策略头作为初始
  动作权重
- **残局 tail solver**：MCTS 前用 alpha-beta 尝试精确求解，proven
  win 时跳过 MCTS

详见 [ALGORITHM_OVERVIEW.md](ALGORITHM_OVERVIEW.md)。

---

## Observation-only AI API

```
POST /ai/sessions                  → 创建 AI 会话
POST /ai/sessions/{id}/observe     → 告诉 AI 发生了什么(action_id + 公开事件)
POST /ai/sessions/{id}/decide      → 返回最优动作
DELETE /ai/sessions/{id}           → 结束会话
```

调用方不需要共享 game state 代码、不需要嵌入 C++ 引擎。把自己游戏
的事件翻译成 action_id + 公开事件即可——GT 端可以是任意来源（外部
API、物理桌游）。这是上面"GT/AI session 物理分离"架构的对外接口实
例化，selfplay / web / API 三条路径在 MCTS 行为上完全等价
（`test_api_mcts_policy_invariance` 守护）。

详见 [docs/guide/AI_API.md](docs/guide/AI_API.md)。

---

## 训练

```
selfplay → 收集样本 → 训练网络 → gating eval → 更新 best model → 循环
```

selfplay / arena / search / 求解全流程 C++，Python 只跑训练循环和网
络训练。配置驱动——所有训练超参写在 `games/<game>/config/game.json`
里，不改代码。

可选训练增强：启发式引导（三段式 schedule）、辅助分数信号、动作过
滤、温度 schedule、Dirichlet 噪声、超时裁决。详见
[FEATURES_OVERVIEW.md](FEATURES_OVERVIEW.md) §训练。

---

## 适用范围与边界

DinoBoard 基于 AlphaZero / ISMCTS 范式，撞上以下场景**不要硬接**
（详见 [FEATURES_OVERVIEW.md §框架局限性](FEATURES_OVERVIEW.md)）：

1. 需要混合策略均衡（扑克类）—— 用 OpenSpiel CFR / Deep CFR / NFSP
2. 动作空间组合爆炸（斗地主）—— 用 OpenSpiel `dou_dizhu` 或 DouZero
3. 卡牌构筑（万智牌）—— 框架接不进来
4. 非零和 / 合作博弈（外交风云、Hanabi）—— 用 OpenSpiel `hanabi` /
   `bargaining`
5. 单人游戏（纸牌接龙、2048）—— 用 intrinsic motivation 系算法
6. 闭眼环节（狼人杀夜晚、密写动作）—— viz 嵌套在 hidden 上的高阶
   不确定性，与本框架"viz 是公开规则"假设硬冲突

---

## 快速上手

### 环境要求

- **C++17 编译器** —— Mac: `xcode-select --install`；Linux:
  `apt install build-essential`；Windows: [Visual Studio 2022 Build Tools](https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2022)
  （只需勾选 "使用 C++ 的桌面开发"，不是完整 IDE）
- **Python ≥ 3.9** + `pybind11` + `torch`
- **ONNX Runtime** —— Web / selfplay / 评估都要加载 `.onnx` 模型，刚
  需。Linux x64 / Windows x64 已随仓库自带；Mac 用 `brew install
  onnxruntime`；其它平台从 [ONNX Runtime releases](https://github.com/microsoft/onnxruntime/releases)
  下载并通过 `BOARD_AI_ONNXRUNTIME_ROOT` 指定路径。

### 构建（Mac / Linux）

```bash
pip install pybind11 torch
pip install -e .

python -c "import dinoboard_engine; print(dinoboard_engine.available_games())"
```

### 构建（Windows）

打开 "x64 Native Tools Command Prompt for VS 2022"（Build Tools 装
完后开始菜单可搜到，本质是预设好 MSVC 环境变量的 cmd），`cd` 到仓
库目录：

```bat
pip install pybind11 torch
pip install -e .
python -c "import dinoboard_engine; print(dinoboard_engine.available_games())"
```

仓库自带的 ONNX Runtime DLL 会被自动复制到扩展旁边，`import` 时直
接加载。

### 训练

```bash
python -m training.cli --game tictactoe --output runs/tictactoe_001
python -m training.cli --game quoridor  --output runs/quoridor_001 \
    --workers 4 --eval-every 25 --eval-games 40 --eval-benchmark heuristic
```

### Web 对战

```bash
pip install -r requirements.txt
cd platform && python -m uvicorn app:app --host 0.0.0.0 --port 8000
open http://localhost:8000
```

6 款游戏、三档难度、多人座位选择、悔棋、智能提示、录像回放 + 掉分
分析。

---

## 新游戏接入

典型工作流：

1. 让 LLM 读 `docs/guide/GAME_DEVELOPMENT_GUIDE.md` 和
   `docs/KNOWN_ISSUES.md`，模仿最相近的现有游戏实现规则
2. 在 `tests/<新游戏>/` 放一份 `test_checklist.py`（从最相近的现有
   游戏复制，改 `GAME = "..."`）—— `pytest tests/<新游戏>/` 一次
   全绿就是"游戏 ready"的明确信号
3. 根据测试失败迭代修复
4. `python -m training.cli --game <id>` 启动训练
5. Web 上验收

这个流程能跑通，是因为框架的每个接入点都有**可机械验证的契约**
（schema 强制、签名锁 const MaskedState、CI 覆盖 viz 错位 / hash
scope / belief 等价 / 公开快照 round-trip）—— LLM 写错了立刻被测
试抓住，闭环自修。Azul 就是这么接进来的：扔规则书 + 启动 LLM，跑
完一轮迭代得到能训练能对战的实现。

---

## 文档

- [FEATURES_OVERVIEW.md](FEATURES_OVERVIEW.md) —— 框架能力速查
- [ALGORITHM_OVERVIEW.md](ALGORITHM_OVERVIEW.md) —— schema / walker /
  MaskedState / RNG / encoder / tracker / belief / ISMCTS DAG 数据流
  契约
- [docs/FRAMEWORK_DESIGN_RATIONALE.md](docs/FRAMEWORK_DESIGN_RATIONALE.md) ——
  立项动机 / 不适合谁 / 与 OpenSpiel 的取舍对比
- [docs/guide/GAME_DEVELOPMENT_GUIDE.md](docs/guide/GAME_DEVELOPMENT_GUIDE.md) ——
  添加新游戏的单一权威来源
- [docs/guide/NEW_GAME_TEST_GUIDE.md](docs/guide/NEW_GAME_TEST_GUIDE.md) ——
  11 步验收流程
- [docs/KNOWN_ISSUES.md](docs/KNOWN_ISSUES.md) —— BUG postmortem +
  设计取舍

---

## 模型状态

各变体已发布的 ONNX 模型：

| 游戏 | 2p | 3p | 4p |
|------|----|----|----|
| 井字棋 | 已训练 | — | — |
| Quoridor | 已训练 | — | — |
| 璀璨宝石 | 已训练 | **未训练（随机初始化）** | **未训练（随机初始化）** |
| 花砖物语 | 已训练 | **未训练（随机初始化）** | **未训练（随机初始化）** |
| 情书 | 已训练 | 已训练 | **未训练（随机初始化）** |
| 政变 | 已训练 | **未训练（随机初始化）** | **未训练（随机初始化）** |

标注*未训练*的变体发布的是随机初始化网络——网页能玩，但没有棋力。
