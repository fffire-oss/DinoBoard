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

### 1. State 字段级可见性标记

每个游戏在可见性声明文件里把所有数据按字段拆开，逐字段写明：字段
名、形状、以及一份和字段同形状的"谁能看见"标记（每个槽位对每
个玩家一个比特）。规则代码是这份标记的**唯一**写入者——推进游
戏状态时同步翻面、揭示给特定玩家、或重置回初始可见性。

框架按这份声明走一遍，对每个槽位决定"对当前观察者该露还是该藏"，
得到一份**遮罩后的观察视图**，由三家消费者共用：发给客户端的公开
快照、MCTS 节点哈希、神经网络输入张量。藏起来的槽位结构性读到占
位哨兵——三家都不可能编入真值。"哪个槽位对哪个玩家可见"这件事
变成 **state 字段**，不是 observer 实现内部的代码逻辑。这条形状被
强制成所有游戏共享的基类设施，可以集中跑一组 CI 测试覆盖所有游戏：
作者写错的 round-trip / hash scope / 公开快照对齐立即被抓住。

### 2. Ground truth 与 AI 物理分离

Ground truth持一份完整真相状态用来推进游戏，**额
外**给每个玩家视角一份独立的会话状态。AI 决策路径（belief 追踪、
特征编码、MCTS 搜索）只读会话状态，物理上没有任何接口能拿到真相
指针。belief 追踪器的"按观察记忆采一个完整世界"接口只接受观察者
身份和随机源、不接受真相状态；MCTS 的根节点状态是当前行动玩家自
己的会话状态，不是真相。这是 structural safety，不是 contractual safety：作者**没有能力**
读真相，不是"被规劝不要读"。差别在 LLM / 经验不足的作者身上才显
现，对自觉作者接近 0。

### 3. 强制 Web 前端

每款游戏接进来必须有可玩 web 前端、可视化 AI 决策、回放工具。这不
是 UI 美学，是 AI 行为可信度的最终判据——training metric（win rate /
loss curve / policy entropy）正常但 AI 在某个回合明显送子，这种 bug
命令行看不出来，**人类对战 30 秒能感觉到**。

---

## 算法层

ISMCTS over DAG：

- **根节点确定化**：每次仿真持一份独立的随机源；根节点从 belief 追
  踪器采一个完整世界，往下推进时同一份随机源继续驱动规则代码处理
  物理随机（如 Azul 工厂补抽）。无机会节点——物理随机和信息不对
  称在搜索里统一处理
- **DAG 而非 tree**：节点键是「状态哈希 + 当前行动玩家」，同一信
  息集从不同路径到达共享节点；用 UCT2（Childs 2008）的多入边修正
  避免过度探索
- **PUCT 选择**：AlphaZero 风格的先验引导，神经网络策略头作为初始
  动作权重
- **残局求解器**：MCTS 之前先用 alpha-beta 尝试精确求解，已证明必
  胜时直接跳过 MCTS

详见 [ALGORITHM_OVERVIEW.md](ALGORITHM_OVERVIEW.md)。

---

## 训练

```
selfplay → 收集样本 → 训练网络 → gating eval → 更新 best model → 循环
```

selfplay / arena / search / 求解全流程 C++，Python 只跑训练循环和网
络训练。配置驱动——训练循环超参写在 `games/<game>/config/game.json`；
MCTS 强度被拆成六个命名 profile（selfplay / arena / eval 在
`game.json`，web_expert / web_casual / analysis 在 `web.json`），不
改代码。

可选训练增强：启发式引导、辅助分数信号、动作过
滤、温度 schedule、Dirichlet 噪声。详见
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

### 仅观察的 AI API

```
POST /ai/sessions                  → 创建 AI 会话
POST /ai/sessions/{id}/observe     → 告诉 AI 发生了什么（动作 id + 公开事件）
POST /ai/sessions/{id}/decide      → 返回最优动作
DELETE /ai/sessions/{id}           → 结束会话
```

调用方不需要共享游戏状态代码、不需要嵌入 C++ 引擎。把自己游戏的事
件翻译成动作 id + 公开事件即可——真相端可以是任意来源（外部 API、
物理桌游）。这是上面"Ground truth 与 AI 物理分离"架构的对外接口实
例化。详见 [docs/guide/AI_API.md](docs/guide/AI_API.md)。

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

- [FEATURES_OVERVIEW.md](FEATURES_OVERVIEW.md) —— 框架能力概览
- [ALGORITHM_OVERVIEW.md](ALGORITHM_OVERVIEW.md) —— 核心算法概览
- [docs/FRAMEWORK_DESIGN_RATIONALE.md](docs/FRAMEWORK_DESIGN_RATIONALE.md) ——
  立项动机 / 与 OpenSpiel 的取舍对比
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
| 步步为营 | 已训练 | — | — |
| 璀璨宝石 | 已训练 | **未训练（随机初始化）** | **未训练（随机初始化）** |
| 花砖物语 | 已训练 | **未训练（随机初始化）** | **未训练（随机初始化）** |
| 情书 | 已训练 | 已训练 | **未训练（随机初始化）** |
| 政变 | 已训练 | **未训练（随机初始化）** | **未训练（随机初始化）** |

标注*未训练*的变体发布的是随机初始化网络——网页能玩，但没有棋力。
