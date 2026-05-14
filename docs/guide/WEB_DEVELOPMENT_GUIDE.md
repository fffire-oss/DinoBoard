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
12. [框架提供的通用 UI](#12-框架提供的通用-ui)
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
| `formatOpponentMove` | `(actionInfo, actionId, actorIdx) -> string` | 否 | 格式化对手上一步的文字描述。`actorIdx` 是动作发起者的 seat 编号（多人游戏需要） |
| `formatSuggestedMove` | `(actionInfo, actionId, actorIdx) -> string` | 否 | 格式化 AI 提示推荐动作的文字描述 |
| `getPlayerSymbol` | `(aiPlayer) -> string` | 否 | 返回玩家身份描述（默认"先手"/"后手"） |
| `extensions` | `Array<{ render(el, gameState) }>` | 否 | 信息栏扩展条目数组，每个条目实现 `render(el, gameState)`，由 `info_panel.updateExtensions(gameState, extensions)` 在每次状态更新时调用——直接 DOM 写 `el.textContent` 或 `el.innerHTML` 即可。例：`{ render(el, gs) { el.textContent = `袋中：${gs.state.bag_total}`; } }` |
| `gameIntro` | string | 否 | 开局后写入侧栏 ops-msg 的简短操作说明，AI 第一次落子时自动清空，后续让位给"已悔棋"/掉分提示等瞬态信息 |
| `disableForce` | bool | 否 | 默认 false。设 true 时禁用"替对手落子"——侧栏不渲染按钮，pipeline 也不会进入 forceMode。隐藏信息游戏必须开启，见下文 §5.1 |
| `showWinrateDefault` | bool | 否 | "显示胜率预估"复选框的默认值（可被 localStorage 覆盖）。默认 true（完全信息游戏开启）；隐藏信息游戏必须显式置为 false，见下文 §5.1 |
| `describeTransition` | `(prevState, newState, actionInfo, actionId) -> AnimStep[] \| null` | 否 | 动作动画描述函数。返回动画步骤数组，框架按序播放后再切换到新状态；返回 `null`/`[]` 直接切换；抛异常自动降级直接切换。详见 §5.3 |
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

### gameState 对象（来自后端 `session_response`）

后端 `platform/game_service/sessions.py::session_response()` 返回的对象。**注意 `state_serializer` 返回的游戏自定义字段被嵌在 `state` 子对象下**，不在顶层：

顶层框架字段：
- `session_id`：本局 ID
- `current_player`：当前玩家 seat
- `is_terminal` / `winner` / `is_turn_start`
- `legal_actions`：合法动作 ID 列表
- `last_actor`：上一步动作发起者（来自 replay frames 的最后一帧 actor，可能为 `null`）
- `last_action_info` / `last_action_id` / `last_action_actor`：**这三个字段来自后端的 pipeline 结果，由 `app.js` 在 AI 落子完成后客户端写入 `state.gameState`**（`session_response` 本身不直接返回它们）。游戏 JS 在 `formatOpponentMove` / `formatSuggestedMove` 里只读不写
- `num_players` / `human_player` / `ai_player` / `ai_players` / `difficulty`

`state` 子对象（`gs.get_state_dict()` 调用 `state_serializer` 的返回值）：
- `state_serializer` 返回的所有 key-value 都在这层，例：`gameState.state.walls_remaining`、`gameState.state.bag_total`
- `extensions` 的 `render(el, gameState)` 拿到的就是顶层 `gameState`，访问游戏字段记得 `gameState.state.xxx`

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

  formatOpponentMove(actionInfo, actionId, actorIdx) {
    return `对手${actionInfo.type === 'move' ? '移动到' : '放置墙于'} (${actionInfo.row},${actionInfo.col})`;
  },

  formatSuggestedMove(actionInfo, actionId, actorIdx) {
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

### 5.3 `describeTransition` —— 动画步骤描述

> 何时**应该**做动画、`popup` vs `reveal` 怎么取舍这种**设计判据**写在 [`WEB_DESIGN_PRINCIPLES.md` §3](WEB_DESIGN_PRINCIPLES.md)。本节只列**实现细节**：可用的 step 类型、字段、`data-*` 选择器约定、推荐时长。

`describeTransition(prevState, newState, actionInfo, actionId)` 返回一个 `AnimStep[]`，框架按序 await。所有 step 都跑在 z-index 10001 的 `.anim-overlay` 层上，不会修改业务 DOM；动画结束后 app 会重渲染到 `newState`。

#### Step 类型

| type | 必备字段 | 可选字段 | 说明 |
|------|---------|---------|------|
| `fly` | `from`, `to`, `createElement` | `width`/`height`, `duration`, `onStart(srcEl)`, `hideFrom`, `onComplete` | 创建一个飞行 sprite 从 `from` 飞到 `to`。`from`/`to` 接受 CSS 选择器或 `HTMLElement` |
| `flyGroup` | `flights[]`（无 `type` 的 fly 字段对象数组） | — | 一组 fly 并行播放，整体在最慢的那条结束时返回 |
| `group` | `children[]`（任意 step 数组） | — | 任意类型 step 并行播放 |
| `popup` | `target`, `content` | `className`, `width`, `height`, `duration` | 数字/文字气泡浮现再淡出。`content` 是字符串或 `HTMLElement` |
| `run` | `fn` | — | 在 step 间执行 DOM mutation。常用来在 fly 之间改动业务 DOM 让下一步看到中间状态 |
| `fadeOut` | `target` | `duration` | 淡出一个 DOM 元素 |
| `highlight` | `target` | `className`, `duration` | 短暂加 CSS class，`duration` 后移除 |
| `pause` | `duration` | — | 等待一段时间，纯粹用于节奏 |
| `reveal` | `title`, `body` | `buttonText`（默认"知道了"）, `className`, `timeoutMs` | 阻塞式弹窗，玩家点确认后才继续。**默认不要给 `timeoutMs`**——强制看清就是它存在的意义 |

#### `fly` 的中间状态机制

一个动作常常是多步动画的串联（购买卡牌 = 宝石飞回银行 + 卡牌飞到玩家区）。每步动画结束后，前一步的 DOM 状态需要保持，否则后面的步骤会从错误的初始位置出发。三种维护方式：

- **`onStart(srcEl)`**：fly 启动时执行（在源 rect 已采集、sprite 已生成、动画即将开始的时刻）。把源 DOM 改成"取走后"的样子，sprite 从原位飞走，视觉上容器位置不动只有实体离开。**容器有空态背景时优先用这个**（详见 [`WEB_DESIGN_PRINCIPLES.md` §3.5](WEB_DESIGN_PRINCIPLES.md) 的 hideFrom 陷阱）
- **`hideFrom: true`**：fly 结束后源元素 `visibility: hidden`（保留布局占位）。简单粗暴，但会把整个容器（含空态背景）一起藏掉，仅适用于"取走后容器本来就该完全消失"的场景
- **`onComplete`**：fly 结束后跑回调，比如递减计数器文字，让后续 step 看到正确数字

所有动画播完后框架自动恢复 `visibility: hidden` 的元素，再触发业务 DOM re-render。

#### `data-*` 选择器约定

`describeTransition` 用 CSS 选择器定位 DOM。渲染函数需要在关键元素上加 `data-*` 属性。命名没有强制要求，**唯一的硬约束是 `data-player` / `data-opponent`**（§3.6 头顶气泡需要它定位行动玩家容器）：

| 属性 | 含义 |
|------|------|
| `data-player="<idx>"` | 人类玩家自己的区域（**必须**，每个 player-area 都要加，含人类自己的） |
| `data-opponent="<idx>"` | 对手玩家区域（**必须**） |
| `data-bank-gem="<color>"` | 银行宝石（按颜色索引）—— Splendor 风格示例 |
| `data-tableau="<tier>-<slot>"` | 公开牌位 |
| `data-player-gem="<player>-<color>"` | 玩家宝石 |
| `data-reserved="<player>-<slot>"` | 保留卡 |
| `data-noble="<slot>"` / `data-deck="<tier>"` | 贵族 / 牌堆 |

只要 `describeTransition` 和渲染函数用一致的选择器即可。

#### 推荐时长

| 动画类型 | 推荐时长 |
|---------|---------|
| 宝石/token 移动 | 350–450ms |
| 卡牌移动 | 450–600ms |
| popup 头顶气泡 | 1200–1800ms |
| 步骤间间隔 | 60ms（`STEP_GAP`，框架自动插入） |
| 整体超时上限 | 30000ms（`MAX_QUEUE_MS`，仅为兜底失控的动画链） |

`fly` 默认 `duration = 350ms`（`DEFAULT_DURATION`）。token 类小元素够用，卡牌偏快——把卡牌单独提到 ~550ms 是 Splendor 实测舒适的数值。`flyGroup` 里的多条 flight 应该用相同或相近 duration，否则视觉上割裂。

更详细的速度调优经验、为什么"动画不要太快"，见 [`WEB_DESIGN_PRINCIPLES.md` §3 速度建议](WEB_DESIGN_PRINCIPLES.md)。

#### 最低实现清单

每个 `describeTransition` 返回的动画链都必须满足：

1. 有可视化实体位移的动作（拿宝石、移砖、买卡）→ 至少一个 `fly` / `flyGroup`
2. 没有实体位移的动作（声明、决策、偷看）→ 至少一个 `popup`（[`WEB_DESIGN_PRINCIPLES.md` §3.6](WEB_DESIGN_PRINCIPLES.md)）
3. 揭示玩家本人才知道的私密信息（Priest/Baron 等）→ 用 `reveal`，**不要**用 popup（[`WEB_DESIGN_PRINCIPLES.md` §3.7](WEB_DESIGN_PRINCIPLES.md)）
4. 录像回放走同一份 `describeTransition`——所有 step 在复盘里也会触发，含 `reveal` 的"知道了"按钮

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

`platform/tools/eval_model.py` — 独立评估训练成果的脚本，支持并行对局和胜率统计。**搜索强度（`simulations` / `temperature` 等）由 `--profile` 指定的 MCTS profile 决定**，没有 `--sims` / `--temp` 这类直接旋钮（避免和 profile 配置漂移）。

```bash
# 快速评估：40 局 4 worker 并行，只看胜率（不保存录像）；强度走 arena profile
python3 platform/tools/eval_model.py \
  --game quoridor \
  --model runs/quoridor_v14/models/model_best.onnx \
  --games 40 --workers 4 --no-save -o /tmp/eval

# model vs heuristic，保存录像供前端回放；用 web_expert profile 跑高强度
python3 platform/tools/eval_model.py \
  --game quoridor --profile web_expert \
  --model runs/quoridor_v14/models/model_latest.onnx --name-model latest \
  --heuristic-temp 0.2 --games 20 \
  -o games/quoridor/replay/latest_vs_heuristic

# model vs model
python3 platform/tools/eval_model.py \
  --game quoridor \
  --model models/step500.onnx --name-model step500 \
  --opponent models/model_init.onnx --name-opponent init \
  --games 20 --workers 4 \
  -o games/quoridor/replay/step500_vs_init
```

主要参数：

| 参数 | 说明 |
|------|------|
| `--game` | game id（如 `quoridor` / `loveletter_4p` / `azul_3p`） |
| `--model` | 待测模型路径（轮转所有座位） |
| `--opponent` | 对手；模型路径或字符串 `heuristic`（默认 heuristic） |
| `--profile` | MCTS profile 名（默认 `arena`；常用 `web_expert`） |
| `--name-model` / `--name-opponent` | 录像里的显示名（缺省取文件 stem） |
| `--heuristic-temp` | heuristic 对手温度（仅 model vs heuristic） |
| `--games` / `--workers` | 局数 / 并行数 |
| `-o` / `--output` | 录像输出目录 |
| `--no-save` | 只打印统计，不写录像 |

自动交替先后手，输出按先后手分别统计胜率。录像为轻量 JSON（仅含 action_history），前端加载时自动回放生成帧。

## 12. 框架提供的通用 UI

`createApp` 自动注入以下通用功能，游戏前端**不需要额外代码**：

| 功能 | 实现位置 | 说明 |
|------|---------|------|
| 缩放控件 | `general/layout.js` | 左上角 +/- 按钮缩放棋盘区域，状态保存到 `localStorage['dino_board_zoom']` |
| 侧边栏收起 | `general/sidebar.js` | 侧边栏边缘"收起/展开"按钮，状态保存到 localStorage |
| 信息栏 | `general/info_panel.js` | 回合 / 对手动作 pill / 胜率 pill / AI 提示 / `extensions` 渲染入口 |
| 录像窗口 | `general/replay.js` | 对局结束后自动展示，专家难度逐帧含掉分分析 |
| 国际化 | `general/i18n.js` + `i18n_strings.js` | `t(key)` 返回当前语言文案 |

### 12.1 游戏 JS 可直接 `import` 的 framework 辅助 API

`general/` 下绝大多数模块都由 `createApp` 内部实例化、通过 `ctx` 或 config callback 暴露给游戏；**游戏 JS 应当只 `import` 下列三个模块**，其余的不要直接 import（直接 import 会绕过 `createApp` 的生命周期管理和参数解析，迁移时容易踩坑）。

| 模块 | 导出 | 何时用 |
|------|------|------|
| `general/app.js` | `createApp(config)` | **唯一**入口；游戏 JS 末尾调一次 |
| `general/i18n.js` | `t(key, params?)` / `tList(key)` / `register(dict)` / `getLang()` / `setLang(lang)` | 文案翻译。游戏首屏调 `register({zh:{...},en:{...}})` 注入自己的字典，渲染时调 `t('ns.key', {a, b})`；`tList` 返回 `string[]`（用于多行说明） |
| `general/api.js` | `apiGet(path)` / `apiPost(path, body)` / `API_BASE` | 只在游戏需要走标准 fetch 之外的特殊 API 调用时引入（目前仅 Quoridor 用到）。常规对局、AI、悔棋等都已封装在 `createApp` 内部，**不需要游戏自己 fetch** |

`sidebar` / `modal` / `info_panel` / `replay` / `animate` / `layout` / `pipeline` 是 framework-internal——它们要么由 `createApp` 自己 `new` 一份再绑回 ctx（`ctx.sidebar.setOpsMsg(...)`、`ctx.infoPanel.setSuggest(...)`，见 §5 ctx 表），要么完全无需游戏触碰。直接 import 这些模块的代码不应进 review。

### 12.2 游戏 JS 不要自己操作 sidebar / info_panel DOM

`sidebar.setOpsMsg` / `sidebar.setStartMsg` / `infoPanel.setTurn` / `infoPanel.setMessage` / `infoPanel.setWinrate` / `infoPanel.setSuggest` 等方法**全部在 framework 内部由 `app.js` 调用**——例如悔棋后写 ops-msg、AI 思考中改 turn pill、smart-hint 写 suggest pill。游戏 JS **不应**自己 import 这些模块或拿它们的实例操作 DOM。游戏想要影响信息栏内容的合法通道有两个：

1. **`config.extensions`**（[§5.1](#51-隐藏信息游戏的两个必备开关)）—— 把自定义 pill 注入信息栏第 5+ 行；framework 在每次 re-render 时调你的 `render(el, gameState)`。
2. **`config.formatOpponentMove` / `config.formatSuggestedMove`** —— 返回的字符串会被 framework 写入 opp-move pill / suggest pill。

如果出现"我的 setOpsMsg 不生效"、"我手写 infoPanel.setTurn 想替换框架文案"这类需求，那是设计偏离了——框架的 ops-msg 流是**唯一**瞬态文字通道（详见 [WEB_DESIGN_PRINCIPLES §5.3](WEB_DESIGN_PRINCIPLES.md)），改文案应该走 `i18n.register(...)` 覆盖对应 key（例如 `app.undone`、`app.force_done`），而不是绕过 framework 直接写 DOM。

**尚未开局提示**：`layout.css` 提供 `.not-started-placeholder` 样式类供游戏自由选用——但目前的参考实现（quoridor、azul、splendor、loveletter）都不依赖它，而是在 `gameState` 为空时直接 `renderEmptyBoard(container)` 渲染一份占位空盘（空网格 / 空 factory 区 / 空 tableau），让棋盘区域尺寸稳定。这个做法把"开局前的视觉占位"和"游戏专属的空盘语义"合在一起，省一层 DOM 注入。

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
