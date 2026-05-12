# Web 前端开发指南

本文档讲游戏开发者要写什么（§1-§5）和框架已经替你做好了什么（§6-§14）。

视觉与交互层面的总体原则参见 [WEB_DESIGN_PRINCIPLES.md](WEB_DESIGN_PRINCIPLES.md)，新游戏前端开发**必读**。

后端接入与配置看：

- [GAME_DEVELOPMENT_GUIDE.md](GAME_DEVELOPMENT_GUIDE.md) — IGameState / IGameRules / IFeatureEncoder / GameBundle 等接入接口
- [`config/web.json` 字段说明](CONFIG_REFERENCE.md) — Web 平台配置（AI 难度、tail-solve、动作过滤等）

---

## 目录

### 游戏开发者要写什么

1. [StateSerializer — 状态序列化](#1-stateserializer--状态序列化)
2. [ActionDescriptor — 动作描述](#2-actiondescriptor--动作描述)
3. [目录结构](#3-目录结构)
4. [交互设计原则](#4-交互设计原则)
5. [createApp(config) — 通用框架 API](#5-createappconfig--通用框架-api)

### 框架替你做好了什么

6. [通用布局](#6-通用布局)
7. [高级操作](#7-高级操作)
8. [AI Pipeline 与动作分析](#8-ai-pipeline-与动作分析)
9. [录像回放](#9-录像回放)
10. [统一录像格式](#10-统一录像格式)
11. [模型评估工具](#11-模型评估工具)
12. [通用功能（common.js）](#12-通用功能commonjs)
13. [核心 API](#13-核心-api)
14. [交互流程](#14-交互流程)

---

## 1. StateSerializer — 状态序列化

**用途**：Web 前端通过 `/api/games/{session_id}` 获取游戏状态 JSON。

**签名**：`(const IGameState&) -> AnyMap`

`AnyMap` 是 `std::map<std::string, std::any>`。Python bindings 自动将 `std::any` 转为 Python 对象，支持的类型：

| C++ 类型 | Python 类型 |
|----------|-------------|
| `int` | `int` |
| `double` / `float` | `float` |
| `bool` | `bool` |
| `std::string` | `str` |
| `std::vector<int>` | `list[int]` |
| `std::vector<AnyMap>` | `list[dict]` |
| `AnyMap` | `dict` |

**示例**：
```cpp
AnyMap serialize_quoridor(const IGameState& state) {
  const auto& s = checked_cast<QuoridorState>(state);
  AnyMap m;
  m["current_player"] = std::any(s.current_player());
  m["board_size"] = std::any(static_cast<int>(kBoardSize));

  std::vector<AnyMap> pawns;
  for (int i = 0; i < kPlayers; ++i) {
    pawns.push_back({
        {"player", std::any(i)},
        {"row", std::any(static_cast<int>(s.pawn_row[i]))},
        {"col", std::any(static_cast<int>(s.pawn_col[i]))},
    });
  }
  m["pawns"] = std::any(pawns);
  return m;
}
```

## 2. ActionDescriptor — 动作描述

**用途**：Web 前端将 ActionId 翻译为人类可读的描述。

**签名**：`(ActionId) -> AnyMap`

```cpp
AnyMap describe_quoridor(ActionId action) {
  AnyMap m;
  m["action_id"] = std::any(static_cast<int>(action));
  if (is_move_action(action)) {
    m["type"] = std::any(std::string("move"));
    m["row"] = std::any(decode_move_row(action));
    m["col"] = std::any(decode_move_col(action));
  }
  // ...
  return m;
}
```

## 3. 目录结构

```
games/<game>/web/
├── index.html    # 主页面
├── styles.css    # 样式
└── <game>.js     # 游戏逻辑
```

`platform/app.py` 会自动扫描 `games/*/web/` 并挂载到 `/games/<game>/`。

## 4. 交互设计原则

> 完整的视觉与交互指引见 [WEB_DESIGN_PRINCIPLES.md](WEB_DESIGN_PRINCIPLES.md)。本节只列实现相关要点。

**自然交互——不要给每个动作一个按钮**。动作空间可能有几百个，逐一列出既不美观也不可操作。按照物理游戏的交互方式设计 UI：

| 游戏动作类型 | 推荐交互方式 | 说明 |
|-------------|-------------|------|
| 移动棋子 | 点击棋子 → 高亮可到达位置 → 点击目标 | 两步点击 |
| 放置墙/棋子 | 悬停预览 → 点击确认 | 鼠标跟随 |
| 选择资源 | 点击资源池 → 点击目标位置 | 拖放或两步点击 |
| 组合动作 | 分步骤引导，每步缩小选择范围 | 层级选择 |

游戏 JS 负责将手势翻译为 `ActionId`，通过 `ctx.submitAction(actionId)` 提交。玩家不需要看到或理解 ActionId 编码。

**空间锚定——固定区域不动，只有内容变化**。游戏中固定存在的容器（棋盘格、工厂盘、中心区、玩家面板、牌库位置）必须有固定的屏幕位置和尺寸，不随内容数量变化而移动、缩放或重排。只有容器内部的元素（棋子、牌、token）可以出现、消失、移动。

玩家靠空间记忆快速定位信息——"左下角是我的图案线，右上角是公共牌库"。如果容器位置随内容增减而漂移（比如 Azul 中心区的 token 被拿走后区域收缩，导致旁边的工厂盘位移），玩家每步都要重新扫描整个画面，严重影响可玩性。物理桌游天然满足这个约束（棋盘不会自己挪位置），前端实现时要显式保持这一点。

实践要点：
- 用固定尺寸的容器（`width`/`height` 写死或 `min-width`/`min-height`），不用 `fit-content`
- 元素减少时容器留白，不收缩；元素增加时内部滚动或缩放，容器不撑大
- 避免对容器级元素使用 `flexbox` 的 `gap` + 自动换行——内容变化会改变行数，推动后续容器位移

## 5. createApp(config) — 通用框架 API

游戏前端的入口是调用 `createApp(config)`（从 `general/app.js` 导入）。框架处理所有通用逻辑（开局、AI 对弈、悔棋、提示、录像），游戏只需提供渲染和格式化函数。

### config 对象字段

| 字段 | 类型 | 必须 | 说明 |
|------|------|------|------|
| `gameId` | string | 是 | 游戏 ID，匹配 game_registry 注册的 id |
| `numPlayers` | int | 否 | 玩家数（默认 2） |
| `renderBoard` | `(container, gameState, ctx)` | 是 | 渲染公共游戏区域（棋盘/牌桌） |
| `renderPlayerArea` | `(container, gameState, ctx)` | 是 | 渲染玩家私有区域 |
| `formatOpponentMove` | `(actionInfo, actionId) -> string` | 否 | 格式化对手上一步的文字描述 |
| `formatSuggestedMove` | `(actionInfo, actionId) -> string` | 否 | 格式化 AI 提示推荐动作的文字描述 |
| `getPlayerSymbol` | `(aiPlayer) -> string` | 否 | 返回玩家身份描述（默认"先手"/"后手"） |
| `extensions` | `(gameState) -> [{label, value}]` | 否 | 信息栏扩展内容（如"牌堆剩余"） |
| `gameIntro` | string | 否 | 开局后写入侧栏 ops-msg 的简短操作说明，AI 第一次落子时自动清空，后续让位给"已悔棋"/掉分提示等瞬态信息 |
| `disableForce` | bool | 否 | 默认 false。设 true 时禁用"替对手落子"——侧栏不渲染按钮，pipeline 也不会进入 forceMode。隐藏信息游戏必须开启，见下文 §5.1 |
| `showWinrateDefault` | bool | 否 | "显示胜率预估"复选框的默认值（可被 localStorage 覆盖）。默认 true（完全信息游戏开启）；隐藏信息游戏必须显式置为 false，见下文 §5.1 |
| `onGameStart` | `() -> void` | 否 | 开局回调（可用于清理 UI 状态） |
| `onActionSubmitted` | `() -> void` | 否 | 玩家提交动作后回调 |
| `onUndo` | `() -> void` | 否 | 悔棋后回调 |

### ctx 对象（传给 renderBoard/renderPlayerArea）

| 方法/属性 | 说明 |
|-----------|------|
| `ctx.canPlay` | 当前是否允许人类操作（综合判断：非终局、非 busy、轮到人类或替对手模式） |
| `ctx.state` | 当前 app 状态（含 `aiPlayer`、`busy`、`forceMode` 等） |
| `ctx.submitAction(actionId)` | 提交玩家动作。游戏将点击/拖拽手势翻译为 actionId 后调用此方法 |
| `ctx.rerender()` | 强制重新渲染（用于游戏内部状态变化后触发更新） |

### gameState 对象（来自后端 state_serializer）

后端 `session_response()` 返回的对象，包含：
- `current_player`：当前玩家
- `is_terminal`：是否终局
- `winner`：胜者（-1 为平局或未终局）
- `legal_actions`：合法动作 ID 列表
- `last_action_id`：上一步动作 ID
- `last_action_info`：上一步 `action_descriptor` 返回的信息
- `difficulty`：当前难度
- 游戏自定义字段（由 `state_serializer` 返回的所有 key-value）

### 典型游戏 JS 结构

```javascript
import { createApp } from '/static/general/app.js';

createApp({
  gameId: 'mygame',
  numPlayers: 2,

  renderBoard(container, gameState, ctx) {
    container.innerHTML = '';
    if (!gameState) {
      // 还没创建对局：渲染一份"空盘"占位（如空网格），让棋盘区尺寸保持稳定
      renderEmptyBoard(container);
      return;
    }
    // 根据 gameState 渲染棋盘到 container
    // 用户交互后调用 ctx.submitAction(actionId)
  },

  renderPlayerArea(container, gameState, ctx) {
    // 渲染玩家手牌/个人区域
  },

  formatOpponentMove(actionInfo, actionId) {
    return `对手${actionInfo.type === 'move' ? '移动到' : '放置墙于'} (${actionInfo.row},${actionInfo.col})`;
  },

  formatSuggestedMove(actionInfo, actionId) {
    return `建议${actionInfo.type === 'move' ? '移动到' : '放墙于'} (${actionInfo.row},${actionInfo.col})`;
  },
});
```

### 5.1 隐藏信息游戏的两个必备开关

如果游戏存在对人类玩家不可见的对手私密状态（手牌、身份牌等），必须在 `createApp({...})` 同时设置：

```js
disableForce: true,
showWinrateDefault: false,
```

两者都是为了**避免 Web 前端从 AI 决策中泄漏隐藏信息回给玩家**：

- **替对手落子（`disableForce`）**：force 模式让人类替 AI 走一步，但人类看不到对手手牌，任意动作都是猜测——更糟的是提交后引擎会按真实手牌检查合法性，等于把"哪些动作合法"反馈给玩家。等价于让玩家看牌。所以 Love Letter / Coup 这类全靠手牌的游戏直接关闭这个功能。
- **胜率预估（`showWinrateDefault`）**：信息栏胜率读的是 MCTS 根节点 `root_values[humanPlayer]`，搜索从**真实状态**（含人类已知的自己手牌 + 对手隐藏手牌）出发，胜率会随对手实际拿到的牌剧烈摆动；玩家逆向就能推出对手的牌。"掉分分析"也来自同一份 root values，所以由同一个开关同时管控（同时不显示"失误/严重失误"标记）。隐藏信息游戏默认关闭，玩家想看可在侧栏「高级功能 → 显示胜率预估」自行开启。

部分隐藏的游戏（Splendor 牌堆暗保留 vs 桌面明保留）可以**只屏蔽暗的部分**，不需要整体 `disableForce`——这种粒度由游戏前端在 `renderPlayerArea` 中按 `item.visible` 自行决定哪些动作在 force 模式下点不了即可，参考 `games/splendor/web/splendor.js`。

### 5.2 参考实现

- **简单参考**：`games/tictactoe/web/`（9 格棋盘，最简交互）
- **复杂参考**：`games/quoridor/web/`（9×9 棋盘 + 墙放置 + 棋子跳跃）

---

## 6. 通用布局

general 层提供统一的页面布局，游戏前端只需填充内容区域：

```
┌─────────────────────────────────────────────────────┐
│                    上方区域                           │
│  ┌──────────────────────┐  ┌──────────────────────┐  │
│  │                      │  │  信息栏               │  │
│  │                      │  │  回合指示 / 胜率 /    │  │
│  │   公共游戏区          │  │  AI 提示              │  │
│  │   （棋盘 / 牌桌）     │  ├──────────────────────┤  │
│  │                      │  │  录像窗口              │  │
│  │                      │  │  （对局中隐藏，        │  │
│  │                      │  │   结束后显示）         │  │
│  └──────────────────────┘  └──────────────────────┘  │
├─────────────────────────────────────────────────────┤
│                    玩家区域                           │
│  2 人:左右分列                                       │
│  ┌────────────────────┐  ┌────────────────────┐      │
│  │     玩家 0          │  │     玩家 1          │      │
│  └────────────────────┘  └────────────────────┘      │
│                                                      │
│  3-4 人:网格排列                                     │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐│
│  │  玩家 0  │ │  玩家 1  │ │  玩家 2  │ │  玩家 3  ││
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘│
└─────────────────────────────────────────────────────┘
```

- **上方左侧**：`renderBoard()` 渲染的公共区域（棋盘、牌桌、公共资源等）
- **上方右侧**：general 层自动管理的信息栏（上半）和录像窗口（下半）。信息栏包含回合指示、对手上一步描述（`formatOpponentMove`）、AI 胜率/提示。录像窗口在对局中隐藏，结束后显示
- **下方**：`renderPlayerArea()` 渲染的玩家私有区域。2 人游戏左右分列，3-4 人游戏自动切换为网格排列

## 7. 高级操作

general 层统一实现以下操作，游戏前端**不需要额外代码**：

| 操作 | API | 说明 |
|------|-----|------|
| **悔棋** | `POST /step-back` | 回到人类上一串连续行动的起点。循环调用 step-back，跳过 AI 回应和人类的连续回合（通过 `last_actor` 判断）。Splendor 拿币→退币退到拿币处；Azul 跨轮连续行动退到该串的第一步。本质上不区分子动作和连续回合，统一按 `last_actor` 处理 |
| **替对手落子** | `ctx.state.forceMode` | 回退到目标 AI 玩家的一串连续行动起点，由人类替其选动作。同样通过 `last_actor` 跳过连续回合。多人游戏侧边栏有每个 AI 的独立按钮，可指定替哪个 AI 落子。**`config.disableForce: true` 时该按钮整组不渲染**，用于隐藏信息游戏（见 §5.1） |
| **智能提示** | `POST /ai-hint` | AI 推荐最佳动作和胜率，不落子。`formatSuggestedMove()` 格式化显示 |
| **显示胜率预估** | 侧栏勾选 | 控制信息栏胜率 pill 与"失误/严重失误"标记是否显示。状态写入 `localStorage['dinoboard.showWinrate.<gameId>']`，默认值由 `config.showWinrateDefault` 决定（完全信息默认 true，隐藏信息必须 false） |
| **对局中显示录像栏** | 侧栏勾选 | 状态写入 `localStorage['dinoboard.showReplayPanelAlways']`，全游戏共享 |

**侧栏 ops-msg 的瞬态语义**：开局时写入 `config.gameIntro`（操作提示）；AI 第一次落子时自动清空；悔棋时显示"已悔棋"；点击"替对手落子"时显示该流程的简短说明；后续掉分提示等覆盖写入。新开局时配合 `infoPanel.reset()` 一并清掉上一局残留 pill。

## 8. AI Pipeline 与动作分析

Pipeline 协调对局中的 AI 决策和分析。核心设计：**每个 human-to-play 局面只跑一次 MCTS（precompute），结果同时用于"智能提示"和"人类走棋后的掉分分析"**。

### 每个 human→AI 循环的 MCTS 运行

| 运行 | 触发时机 | 用途 |
|------|---------|------|
| **Precompute** | AI 落子完成后（或游戏开局 / 悔棋后）立刻启动，用 `analysis_simulations` 次模拟在**当前 human-to-play 局面**上跑 MCTS | 一份结果两用：① 人类点"智能提示"时直接返回 ② 人类落子后读取缓存的 `action_values[chosen_action]` 算掉分 |
| **AI decision** | 人类落子后进入 AI 回合，用 `simulations` 次模拟在 AI-to-play 局面上跑 MCTS | AI 决定下一步 |

**不存在"分析用户上一步"和"下一轮提示"两次独立搜索**——同一个 precompute 结果在两个时刻被消费。

### Pipeline 阶段机

```
[AI 落子完成] → [done] → 立即启动 precompute（后台）
  │
  ▼ 人类走棋
  │
[analyzing] —— 纯缓存读取：等待当前 precompute 完成（如果还没好），从其
               action_values 中取出人类所选动作的 Q 值，计算掉分
  │
  ▼
[ai_thinking] —— AI 的 MCTS 决策
  │
  ▼
[done] —— 触发下一轮 precompute，循环继续
```

**关键**：`[analyzing]` 阶段只是等 precompute 的结果 + 读缓存，**不触发新的 MCTS**。只有当 precompute 异常（被取消、8 秒超时）才有兜底的 inline MCTS。

非专家难度跳过 analyzing 阶段和 precompute，只做 AI 思考落子。

### 智能提示（ai-hint）

人类回合点击"智能提示"按钮时：

1. 前端立即在信息栏显示"局面分析中..."
2. `POST /ai-hint` 后端：
   - 如果 precompute 结果已就绪 → 立即返回
   - 如果仍在跑 → 阻塞等待至多 8 秒后返回
3. 前端更新信息栏为提示结果

用户看起来是无缝的："分析没好时显示分析中，好了就显示结果"。后端用的仍然是同一份 precompute，不产生额外 MCTS。

前端通过轮询 `GET /pipeline` 获取当前阶段（`analyzing` / `ai_thinking` / `done` / `error`）。pipeline 在后台线程执行 MCTS，GIL 已通过 `py::gil_scoped_release` 释放，不阻塞前端请求。

### 动作分析（掉分检测）

分析基于 precompute 的 MCTS 结果。搜索树的每个根边记录 N 维 Q 值（`action_values: {action_id: [p0_q, p1_q, ...]}`），是该边所有叶子的 per-player value 均值。

**计算公式**：
- `best_wr = (root_values[human_player] + 1) / 2`：最优动作对应的人类胜率
- `actual_wr = (action_values[chosen_action][human_player] + 1) / 2`：用户实际选择的动作对应的人类胜率
- `drop = (best_wr - actual_wr) * 100`：掉分百分比

一次搜索同时得到所有合法动作的 Q 值，直接读取 `human_player` 维度即可，支持任意人数。

**掉分阈值**：≥5% 标记为失误（warn），≥10% 标记为严重失误（blunder）。

### 高级操作与 pipeline 的交互

| 操作 | 与 pipeline 的关系 |
|------|-------------------|
| **悔棋** (`POST /step-back`) | 先 `cancel_pipeline` 终止正在进行的搜索，回退动作历史，重建 GameSession，然后重新触发 precompute |
| **智能提示** (`POST /ai-hint`) | 优先复用 precompute 已有的搜索结果（等待最多 8s）；超时或无结果则临时跑一次独立 MCTS。不触发 pipeline，不落子 |
| **替对手落子** (`POST /action` with forceMode) | 和正常用户走棋相同流程，触发 pipeline |

**实现文件**：`platform/game_service/pipeline.py`（调度和分析）、`platform/game_service/routes.py`（API 端点）。

### 交互规范

所有用户操作遵循"先渲染、后计算"——用户点击后立即看到画面更新，后台任务异步执行。

**用户走棋**：
1. cancel 旧 pipeline，立即 apply 并渲染新画面
2. 若 precompute 分析已完成：显示掉分分析，然后触发 AI 思考
3. 若 precompute 分析未完成：等分析完成后显示掉分，再触发 AI 思考
4. AI 思考完成后渲染 AI 落子，触发下一轮 precompute

**悔棋**：
1. cancel 当前 pipeline
2. 循环 step-back：跳过 AI 回应和人类的连续行动（`last_actor` 匹配），直到回到人类一串行动的真正起点
3. 立即渲染正确画面
4. 立即触发 precompute 分析新局面

**替对手落子**：
1. cancel 当前 pipeline
2. 循环 step-back：跳过连续行动（`last_actor` 匹配），直到到达目标 AI 玩家一串行动的起点
3. 立即渲染，等用户替对手落子后，按正常"用户走棋"流程处理

**智能提示**：
- 专家难度且 precompute 已完成：直接显示分析结果
- 专家难度但 precompute 未完成：显示"局面分析中"，完成后自动更新
- 非专家难度：显示"局面分析中"，临时跑分析，完成后显示
- 提示过程中用户仍可正常落子（提示不阻塞交互）

## 9. 录像回放

仅**专家难度**可用。对局结束后自动进入回放模式。

**功能**：
- 每帧附带掉分分析（见 §8）
- general 层提供回放控制（前进/后退/跳到下一个失误）
- 游戏前端只需确保 `renderBoard()` 能渲染任意帧的状态（通过 `GET /replay` 获取帧列表）

**开发者无需额外代码**：只要 `state_serializer` 和 `renderBoard()` 正确实现，录像功能自动可用。

## 10. 统一录像格式

在线对局和测试脚本使用统一的 JSON 格式：

```json
{
  "game_id": "quoridor",
  "seed": 42,
  "players": {
    "player_0": {"name": "latest_step500", "type": "model"},
    "player_1": {"name": "heuristic_t0.2", "type": "heuristic"}
  },
  "result": {"winner": 0, "draw": false, "total_plies": 75},
  "config": { ... },
  "action_history": [42, 130, 5, ...],
  "frames": [...]
}
```

- **actor** 统一为 `"player_0"` / `"player_1"` / ... / `"start"`，通过 `players` 字典标注名字和类型（human/model/heuristic/ai）
- **frames** 可选：在线对局包含完整帧（带 analysis），测试录像仅含 `action_history`，前端加载时通过 `POST /api/replay/build` 自动回放生成帧
- 录像核心模块：`platform/game_service/replay.py`（`make_replay_frame`、`build_frames_from_actions`、`build_replay_dict`）

## 11. 模型评估工具

`platform/tools/eval_model.py` — 独立评估训练成果的脚本，支持并行对局和胜率统计：

```bash
# 快速评估：40 局 4 worker 并行，只看胜率（不保存录像）
python3 platform/tools/eval_model.py \
  --game quoridor \
  --model-a runs/quoridor_v14/models/model_best.onnx \
  --sims 400 --games 40 --workers 4 --no-save -o /tmp/eval

# model vs heuristic，保存录像供前端回放
python3 platform/tools/eval_model.py \
  --game quoridor \
  --model-a runs/quoridor_v14/models/model_latest.onnx --name-a latest \
  --heuristic-temp 0.2 --sims 800 --games 20 \
  -o games/quoridor/replay/latest_vs_heuristic

# model vs model
python3 platform/tools/eval_model.py \
  --game quoridor \
  --model-a models/step500.onnx --name-a step500 \
  --model-b models/model_init.onnx --name-b init \
  --temp 0.1 --games 20 --workers 4 \
  -o games/quoridor/replay/step500_vs_init

# constrained 模式（动作过滤）
python3 platform/tools/eval_model.py \
  --game quoridor --constrained \
  --model-a models/latest.onnx --name-a latest \
  --games 10 -o games/quoridor/replay/constrained_test
```

参数说明：`--workers N` 并行跑 N 局，`--no-save` 只输出统计不写文件。自动交替先后手，输出按先后手分别统计胜率。录像为轻量 JSON（仅含 action_history），前端加载时自动回放生成帧。

## 12. 通用功能（common.js）

`common.js` 提供以下自动注入的通用功能，所有游戏前端自动获得：

| 功能 | 说明 |
|------|------|
| 缩放控件 | 左上角 +/- 按钮，缩放棋盘区域，状态保存到 localStorage |
| 侧边栏收起 | 侧边栏边缘"收起/展开"按钮，状态保存到 localStorage |
| 尚未开局提示 | 棋盘区域显示"尚未开局"占位文字 |

**尚未开局提示**：layout.css 提供 `.not-started-placeholder` 样式类供游戏自由选用——但目前的参考实现（quoridor、azul、splendor、loveletter）都不依赖它，而是在 `gameState` 为空时直接 `renderEmptyBoard(container)` 渲染一份占位空盘（空网格 / 空 factory 区 / 空 tableau），让棋盘区域尺寸稳定。这个做法把"开局前的视觉占位"和"游戏专属的空盘语义"合在一起，省一层 DOM 注入。

游戏的 `renderBoard()` 函数应在 `gameState` 为空时**返回一个空盘渲染**，而不是把 container 留空——后者会让侧边栏 / info 栏跟着塌缩，违反空间锚定原则（详见 `WEB_DESIGN_PRINCIPLES.md`）。

## 13. 核心 API

| 端点 | 方法 | 说明 |
|------|------|------|
| `/api/games/available` | GET | 列出已注册游戏（ID、显示名、玩家数、是否有 web） |
| `/api/games` | POST | 创建新游戏，参数：`game_id`, `seed`, `human_player`, `difficulty` |
| `/api/games/{id}` | GET | 获取当前状态（调用 `state_serializer`） |
| `/api/games/{id}/action` | POST | 执行玩家动作，参数：`action_id` |
| `/api/games/{id}/ai-action` | POST | 触发 AI 落子（AI 先手时用） |
| `/api/games/{id}/pipeline` | GET | 轮询 AI 响应状态 |
| `/api/games/{id}/ai-hint` | POST | 获取 AI 推荐动作（不落子） |
| `/api/games/{id}/step-back` | POST | 悔棋 |
| `/api/games/{id}/replay` | GET | 获取完整录像帧（含 players 信息） |
| `/api/replay/build` | POST | 从 action_history 回放生成帧（参数：game_id, seed, action_history） |
| `/api/replay/file` | GET | 读取项目内的录像 JSON 文件（参数：path） |

## 14. 交互流程

1. POST `/api/games` 创建对局 → 得到 `session_id`
2. GET `/api/games/{id}` 获取状态 → 渲染棋盘
3. 用户点击 → 将点击映射为 `action_id`
4. POST `/api/games/{id}/action` 提交动作
5. 轮询 GET `/api/games/{id}/pipeline` 直到 `phase == "done"`
6. 重新获取状态 → 渲染新棋盘
