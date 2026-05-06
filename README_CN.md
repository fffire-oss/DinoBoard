# DinoBoard

**English version:** [README.md](README.md)

> **Drop a rulebook. Ship a superhuman AI.**
>
> 给 Claude Code 一句话「加上 Azul」,它自己查规则书、写 C++、跑全量测试、训练出超越人类的 AI、生成 Web 前端。你只负责验收。

通用棋盘游戏 AI 引擎——**一套框架、一次工程投入、无限游戏复用**。AlphaZero 风格 MCTS + 神经网络自我对弈,支持 2-4 人游戏。

---

## 这个项目解决什么问题

数字桌游的 AI 普遍很弱,不是技术不存在,而是**每个游戏从零实现 AlphaZero 的工程成本太高**——MCTS、ONNX 集成、训练管线、隐藏信息处理、调参踩坑,每个游戏都要重走一遍。

DinoBoard 把这条路径一次性打通,变成**可复用的引擎 + 可调用的 API**:

- **核心框架 10k+ 行 C++/Python**——MCTS、belief tracker、训练、Web、分析,全套通用
- **接入一个新游戏只需 ~2000 行**——规则 + 特征编码 + JSON 配置,其余全由框架托管
- **两层测试架构**——框架不变量在 quoridor + azul + loveletter 三游戏 matrix 上跑(最小完备覆盖每个结构特征);**每个新游戏在 `tests/<game>/` 下有自己独立完整的验收清单**,「这个游戏 ready 了」=`pytest tests/<game>/` 一次全绿
- **已有 6 个游戏,覆盖 4 类范式**——完全信息、对称随机、非对称隐藏、虚张声势
- **对外的 observation-only REST API**——第三方桌游 app / 网站 / 平台直接调用,不需要共享任何 game state 代码,不需要嵌入 C++ 引擎
- **从训练到 Web 前端到外部 API 一条链路**——同一份 C++ MCTS 同时服务训练、对战、录像掉分分析、第三方接入,绝无「训练时和生产时逻辑漂移」

---

## 技术介绍

### ISMCTS:面向隐藏信息游戏的 DAG 搜索

针对隐藏信息游戏的原生 MCTS 重构。**Root-sampling determinization + per-acting-player info-set keying + UCT2**——每次 simulation 从 belief 采一个完整世界,之后 descent 完全 deterministic;同一 info set 从不同路径到达共享 DAG 节点。详见 [docs/MCTS_ALGORITHM.md](docs/MCTS_ALGORITHM.md)。

### Observation-Only AI API:训练完就能被第三方调用

框架提供 REST API,第三方系统**零集成成本**接入 AI:

```
POST /ai/sessions                         → 创建 AI 会话
POST /ai/sessions/{id}/observe            → 告诉 AI 发生了什么(动作 ID + 公开事件)
POST /ai/sessions/{id}/decide             → AI 返回最优动作
DELETE /ai/sessions/{id}                  → 结束会话
```

**调用方不需要共享游戏 state 代码、不需要嵌入 C++ 引擎、不需要知道 MCTS 存在**。只要能把自己的游戏事件翻译成动作 ID + 公开事件,就能把超人 AI 当作黑盒对手或教练使用。

这不是简化版接口——它和 selfplay 训练用的是**完全相同的 MCTS + belief tracker + ONNX 推理**。API 的 observation-only 设计是架构层面的硬约束(`IBeliefTracker::observe_public_event()` 的签名里没有 `IGameState*` 参数),意味着:

- AI 决策只依赖观察历史,永不读游戏真实 state → **结构性禁止作弊**
- 同一个 AI 既服务 selfplay 训练、也服务 Web 对战、也服务第三方 API → **一份训练投入,三个部署场景**
- 独立 seed 的 belief 等价测试做信息论层面的分离证明 → **你可以向客户证明 AI 没作弊**

适用场景:现有桌游 app 想加强 AI 对手、桌游平台想提供 AI 教练、物理桌游 companion app 需要实时建议。

### 可插拔 belief 采样:从均匀到启发式到神经网络

同一个 `randomize_unseen` 接口支持三种强度递进:

- **均匀采样**(简单随机游戏如 Azul)
- **手写概率启发式**(Coup 范例:claim/challenge 历史驱动的非均匀 sampler,避免「永不质疑永不诈唬」的退化均衡)
- **神经 belief network**(框架预留接口,可用时序模型替换启发式)

这是框架少有的**专门为不完美信息游戏设计**的地方——大多数 AlphaZero 开源实现只处理完美信息。

### 训练-推理一致性

自我对弈、评估、Web 对战、录像掉分分析——**全部跑在同一份 C++ MCTS 代码上**。没有「训练时 Python、推理时重写 C++」的翻译漂移。Web 上玩家看到的「这步棋让胜率从 62% 掉到 41%」和训练里的 value head 输出是同一个数字。

### 工程纪律

- 两层测试:框架不变量在固定 3 游戏 matrix 上跑;每个游戏 `tests/<game>/` 下有自己完整的验收清单,**包括用游戏自身的守恒律(token / 卡 / 棋子总量、容量上限、可达性)做规则正确性验证**,新游戏 ready 是单一一次绿测
- `docs/KNOWN_ISSUES.md` 收录 22 个已解决的 bug 和设计取舍——这是别人接入新游戏时**能跳过的每一个坑**
- 严格的 no-fallback 纪律(详见 `CLAUDE.md`):静默降级一律拒绝,错误必须抛到表面

---

## 新游戏接入:一次对话就够

你已经不需要手写 game bundle。典型工作流:

1. 对 Claude Code 说「加上 [游戏名]」
2. AI 读 `docs/GAME_DEVELOPMENT_GUIDE.md` 和 `docs/KNOWN_ISSUES.md`,模仿 Quoridor / Splendor 等范例实现规则
3. 在 `tests/<新游戏>/` 下放一份 `test_checklist.py`(从最相近的现有游戏复制,改 `GAME = "..."`)——`pytest tests/<新游戏>/` 一次全绿就是「游戏 ready」的明确信号
4. AI 根据测试失败迭代修复,直到全绿
5. `python -m training.cli --game <id>` 启动训练
6. Web 上打开对战界面验收

**你的工作缩到:提一次需求 + review 生成的 PR + 启动训练**。

这个流程不是理论——它之所以可行,是因为框架的每个接入点都有**可机械验证**的契约(测试 + KNOWN_ISSUES 踩坑清单),AI 能闭环自修。

---

## 架构

```
┌─────────────────────────────────────────────────────────┐
│                    Python 层                            │
│  training/pipeline.py  ←→  bindings/py_engine.cpp       │
│  training/cli.py            (pybind11)                  │
│  platform/app.py       ←→  GameSession                  │
├─────────────────────────────────────────────────────────┤
│                    C++ 引擎                             │
│  ┌──────────┐  ┌──────────────┐  ┌──────────────────┐   │
│  │ runtime/ │  │ search/      │  │ infer/           │   │
│  │ selfplay │→ │ NetMCTS      │→ │ ONNX Evaluator   │   │
│  │ arena    │  │ (ISMCTS DAG) │  │ (可选 ONNX)      │   │
│  │ heuristic│  │ TailSolver   │  │                  │   │
│  └──────────┘  └──────────────┘  └──────────────────┘   │
│  ┌──────────────────────────────────────────────────┐   │
│  │ core/ — 接口定义                                   │   │
│  │ IGameState · IGameRules · IFeatureEncoder        │   │
│  │ IBeliefTracker · GameRegistry · GameBundle       │   │
│  └──────────────────────────────────────────────────┘   │
├─────────────────────────────────────────────────────────┤
│                    游戏实现                             │
│  games/tictactoe/  games/quoridor/                      │
│  games/splendor/   games/azul/                          │
│  games/loveletter/ games/coup/                          │
└─────────────────────────────────────────────────────────┘
```

完整目录树、每个文件的职责见本文末尾。

---

## 快速上手

### 环境要求

本项目核心是 C++ 引擎,Python 只是胶水。需要:

- **C++17 编译器** — Mac: `xcode-select --install`;Linux: `apt install build-essential`;Windows: [Visual Studio 2022 Build Tools](https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2022)(这是单独的命令行编译器安装包,**不是完整 Visual Studio IDE** —— 几百 MB,不会装 IDE 进来)。安装器里勾选 **"使用 C++ 的桌面开发"** 然后点 Install 即可。
- **Python ≥ 3.9** + `pybind11` + `torch`(训练用)
- **ONNX Runtime** — Web 对战 / 自我对弈 / 评估**都要加载 `.onnx` 模型让 AI 出棋**,这是刚需不是可选。
  - **Linux x64**:已随仓库自带在 `third_party/onnxruntime-linux-x64-1.17.3/`,`git clone` 就能用,无需下载。
  - **Windows x64**:已随仓库自带在 `third_party/onnxruntime-win-x64-1.17.3/`,`git clone` 就能用,无需下载。
  - **Mac**:`brew install onnxruntime`
  - **其他平台(Linux ARM 等)**:从 [ONNX Runtime releases](https://github.com/microsoft/onnxruntime/releases) 下载对应平台的包并解压,通过 `BOARD_AI_ONNXRUNTIME_ROOT` 指定路径。

### 构建(Mac / Linux)

```bash
pip install pybind11 torch

# 标准构建(Mac brew / Linux 自带 bundle 自动检测到 ONNX Runtime)
pip install -e .

# 如果 ONNX Runtime 不在标准路径,显式指定
BOARD_AI_WITH_ONNX=1 \
  BOARD_AI_ONNXRUNTIME_ROOT=/path/to/onnxruntime \
  pip install -e .

# 验证
python -c "import dinoboard_engine; print(dinoboard_engine.available_games())"
# ['azul', 'azul_2p', ..., 'quoridor', 'splendor', ..., 'tictactoe']
```

### 构建(Windows)

Build Tools 装完后,在开始菜单搜 **"x64 Native Tools Command Prompt for VS 2022"** —— 这是装 Build Tools 时自动创建的快捷方式,本质上就是个普通命令行,只不过预先把 MSVC `cl.exe` 加进了 PATH。打开它,`cd` 到 clone 下来的仓库目录,然后:

```bat
pip install pybind11 torch

REM 仓库自带的 Windows ONNX Runtime 会自动检测,无需设环境变量
pip install -e .

REM 验证
python -c "import dinoboard_engine; print(dinoboard_engine.available_games())"
```

> 为什么要用这个命令行而不是普通 `cmd` / PowerShell?MSVC 编译时需要一组环境变量(`INCLUDE`、`LIB`、`PATH` 里的工具链路径)预先设好才能找到 `cl.exe`。这个快捷方式背后会先跑 `vcvars64.bat` 帮你配好。直接用普通 `cmd` 会报 "cl 不是内部或外部命令"。

构建会自动把 `onnxruntime.dll` 复制到编译好的扩展旁边,`import` 时直接加载,无需改 `PATH`。

> `setup.py` 在没检测到 ONNX Runtime 时会打印 WARNING 并继续构建——这只是为了跑基础测试能过,**Web 对战和训练都会因为加载不了模型而失败**。

### 训练

```bash
# TicTacToe — 约 5 分钟
python -m training.cli --game tictactoe --output runs/tictactoe_001

# Quoridor — 数小时(含 warm start + heuristic guidance)
python -m training.cli --game quoridor --output runs/quoridor_001 \
    --workers 4 --eval-every 25 --eval-games 40 --eval-benchmark heuristic
```

训练配置由 `games/<game>/config/game.json` 驱动——不改代码,只改 JSON。详见 [功能概览 §配置速查](docs/GAME_FEATURES_OVERVIEW.md#配置速查)。

### Web 对战

```bash
pip install -r requirements.txt  # fastapi, uvicorn
cd platform && python -m uvicorn app:app --host 0.0.0.0 --port 8000
open http://localhost:8000
```

功能:6 款游戏、三档难度(Heuristic / Casual / Expert)、多人座位选择、悔棋、智能提示、录像回放 + 掉分分析。

---

## 核心概念速记

- **GameBundle 注册** — 每个游戏一个工厂函数,返回 state + rules + encoder + 若干可选组件。详见 [游戏开发指南](docs/GAME_DEVELOPMENT_GUIDE.md)。
- **ISMCTS** — Root sampling + DAG + UCT2,隐藏信息游戏的原生方案。详见 [docs/MCTS_ALGORITHM.md](docs/MCTS_ALGORITHM.md)。
- **AI API** — Observation-only REST 接口,第三方桌游 app 直接调用,无需嵌入引擎代码。同时作为 AI 不作弊的信息论证明。详见 [GAME_DEVELOPMENT_GUIDE §17](docs/GAME_DEVELOPMENT_GUIDE.md#17-ai-api-分离验收)。
- **训练管线** — 自我对弈 → Replay Buffer → SGD → ONNX 导出 → gating eval(≥60% 胜率更新 best)。支持 Warm Start、Heuristic Guidance、Auxiliary Score、Training Action Filter、MCTS Schedule。

---

## 文档

- **[功能概览](docs/GAME_FEATURES_OVERVIEW.md)** — 框架能力速查
- **[游戏开发指南](docs/GAME_DEVELOPMENT_GUIDE.md)** — 添加新游戏的单一权威来源
- **[MCTS 算法](docs/MCTS_ALGORITHM.md)** — ISMCTS 的 DAG 搜索推导
- **[新游戏验收测试](docs/NEW_GAME_TEST_GUIDE.md)** — 11 步验收流程 + 两层测试架构原则
- **[已知问题与踩坑](docs/KNOWN_ISSUES.md)** — BUG-001~022 postmortem + 设计取舍

---

## 合作与联系

本项目核心面向**研究型桌游 AI 工程**和**数字桌游的 AI 外包/咨询**场景。如果你需要:

- 给某款数字桌游加强力 AI 对手 / AI 教练
- 在你的游戏平台里集成 observation-only 的 AI 推理服务

欢迎通过 Issues / Discussions 联系。

---

## 目录结构

```
DinoBoard/
├── engine/                         # C++ 通用引擎
│   ├── core/                       # 接口定义
│   │   ├── game_interfaces.h       #   IGameState, IGameRules, IStateValueModel
│   │   ├── feature_encoder.h       #   IFeatureEncoder
│   │   ├── belief_tracker.h        #   IBeliefTracker(隐藏信息)
│   │   ├── game_registry.h         #   GameBundle, GameRegistrar
│   │   ├── types.h                 #   ActionId, StateHash64, UndoToken
│   │   └── action_constraint.h     #   IActionConstraint
│   ├── search/                     # 搜索算法
│   │   ├── net_mcts.h/.cpp         #   PUCT-MCTS + 神经网络评估
│   │   ├── tail_solver.h/.cpp      #   Alpha-Beta 残局求解
│   │   ├── root_noise.h            #   Dirichlet 噪声
│   │   └── temperature_schedule.h  #   温度衰减调度
│   ├── infer/                      # 推理
│   │   └── onnx_policy_value_evaluator.*  # ONNX Runtime 推理
│   └── runtime/                    # 运行时
│       ├── selfplay_runner.*       #   自我对弈循环 + FilteredRulesWrapper
│       ├── arena_runner.*          #   模型对战
│       └── heuristic_runner.*      #   启发式对局生成
│
├── games/                          # 游戏实现(每个游戏一个子目录)
│   ├── tictactoe/  quoridor/  splendor/  azul/  loveletter/  coup/
│
├── bindings/py_engine.cpp          # pybind11 Python 绑定
│
├── training/                       # Python 训练框架
│   ├── pipeline.py   model.py   cli.py
│
├── platform/                       # FastAPI Web 对战平台 + AI 推理 API
│   ├── app.py                      #   主服务器
│   ├── ai_service/                 #   Observation-only AI REST API
│   ├── game_service/               #   游戏会话、异步管线、录像分析
│   └── static/                     #   通用前端资源
│
├── tests/
│   ├── framework/                  #   框架不变量(3 游戏 matrix carrier)
│   ├── tictactoe/  quoridor/  splendor/  azul/  loveletter/  coup/
│   │                               #   每个游戏完整验收清单
├── docs/                           # 文档
└── setup.py · requirements.txt     # 构建
```

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
