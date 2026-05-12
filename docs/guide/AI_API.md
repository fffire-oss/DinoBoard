# DinoBoard AI API

> 用本框架训练出的 AI 接入任意第三方游戏服务器的完整文档。
> 第三方只要实现"观察 → 动作 ID"的翻译层就能把 AI 当作对手或助手。

---

## 设计原则（为什么这套 API 这么小）

- **No state crosses the boundary.** API 只在 `action_id`（整数）和事件（`{kind, payload}` 字典）这两种形态上通信。第三方**从不**需要向 AI 传递游戏的完整 state——所有观察都以动作和事件的形式流入。
- **AI 自己维护所看到的一切。** 每个 session 内部持有一份 "从 AI 视角观察到的" 游戏 state + belief tracker。接入方只负责把 ground truth 的动作和公开事件翻成 API 请求。
- **Public state 完全由消息流重建**。每次 `observe` 之后，session 的所有公开字段（牌面、分数、棋盘等 schema 中 `viz[..., viewer]=1` 的 slot）都从 `public_snapshot` 反向覆写——AI 不会重放规则（observer 路径上没有 `do_action_fast`），所以即使 AI 的采样和 truth 不同，observer 看到的公开局面也跟 truth byte-equal（`test_public_snapshot_round_trip` 守护）。
- **Hidden state 是采样、不是拷贝**。AI 的 hidden 字段由 `seed` 控制的 RNG 通过 belief tracker **每次 observe 后重新采样**，跟 truth 在数值上独立。`seed` 只影响 hidden 部分；public 部分跟 seed 无关（`test_public_hash_excludes_internal_rng` 守护）——这意味着接入方的 `seed` 选择不会让 AI 看到的公开局面和 truth 漂移。

---

## 接入的最小实现

第三方需要做的工作，按工序：

1. **选一个游戏 ID**（比如 `quoridor`）。每个游戏都有一份 `docs/games/<game>_api.md` 专属说明，介绍动作编码和事件格式。
2. **调 `POST /ai/sessions`** 创建 session，拿到 `session_id`。
3. **每次 ground truth 发生一个动作**（不管是 AI 自己的还是对手的）：
   - 调 `POST /ai/sessions/{id}/observe` 把 `action_id` 喂给 AI
   - 对有隐藏信息的游戏，需要同时传 `events` 和 `public_snapshot`（见专属文档）
4. **轮到 AI 决策时**（ground truth 的 `current_player == my_seat`）：
   - 调 `POST /ai/sessions/{id}/decide` 获取 AI 选择的 `action_id`
   - 把这个动作应用到 ground truth 上
   - 然后再调一次 `observe` 喂给 AI（AI 需要观察到自己刚才做了什么，因为 decide 不改内部 state——只是推理）
5. **游戏结束后**调 `DELETE /ai/sessions/{id}` 释放资源。

---

## 端点参考

### `POST /ai/sessions` — 创建

**请求**：
```json
{
  "game_id": "quoridor",
  "seed": 12345,
  "my_seat": 0,
  "initial_observation": {}
}
```

| 字段 | 说明 |
|------|------|
| `game_id` | 注册的游戏 ID（见 `docs/games/`）。多人变体用 `{game}_3p` / `{game}_4p` |
| `seed` | 可选。AI 内部 RNG seed。**对有隐藏信息的游戏可以任意选**，不需要等于 ground truth 的 seed——AI 的内部世界本来就是采样的；省略则服务端用 `secrets.randbits(64)` 自取 |
| `my_seat` | AI 扮演的座位（0-indexed） |
| `initial_observation` | 隐藏信息游戏开局时这个 perspective 才能看到的事实（如自己的初始手牌）。隐藏信息游戏必填；完全公开的游戏忽略此字段 |

> AI 强度（`simulations` / `temperature`）由服务端从 `web.json::difficulty_overrides.expert` 解析（fallback `{simulations: 800, temperature: 0.0}`），**客户端不可指定**。

**响应**：
```json
{
  "session_id": "a1b2c3d4",
  "game_id": "quoridor",
  "num_players": 2,
  "my_seat": 0,
  "current_player": 0,
  "is_terminal": false
}
```

### `POST /ai/sessions/{id}/observe` — 同步观察

**请求**：
```json
{
  "action_id": 67,
  "events": [],
  "public_snapshot": {}
}
```

`events` 是这次 transition 里公开可观察到的事件序列（如 Splendor 的 `deck_flip`、Azul 的 `factory_refill`），按 producer 发出的顺序排列；只喂给 belief tracker，不会作用到 state 上。`public_snapshot` 是 GT 端这一步之后所有公开 slot 的值——observer 用它整体覆写自己 session 的公开部分。**完全可观察的游戏**（TicTacToe、Quoridor）`events` 和 `public_snapshot` 都传 `[]` / `{}` 即可，session 通过 `do_action_fast` 重放动作推进。

事件格式：
```json
{
  "kind": "deck_flip",
  "payload": {"tier": 1, "slot": 0, "card_id": 23}
}
```

具体每个游戏可能发的事件类型见 `docs/games/<game>_api.md`。

**响应**：
```json
{
  "actions_observed": 1,
  "current_player": 1,
  "is_terminal": false
}
```

### `POST /ai/sessions/{id}/decide` — 获取 AI 动作

**无请求体**（AI 用已经观察到的历史推理）。

**响应**：
```json
{
  "action_id": 42,
  "action_info": {"type": "move", "row": 1, "col": 4},
  "stats": {
    "simulations": 800,
    "best_value": 0.63,
    "dag_reuse_hits": 127,
    "tail_solved": false
  },
  "current_player": 0,
  "is_terminal": false
}
```

- `action_id`：AI 选的动作整数 ID
- `action_info`：人类可读的动作描述（`type` 字段 + 游戏特定字段）
- `stats`：MCTS 诊断——`best_value` 是从当前玩家视角 AI 对局面的估值（-1..+1），可用作"AI 对自己胜率的判断"

**重要**：`decide` 不修改 session 的动作历史。调完 `decide` 拿到 action 后，**把这个 action 应用到 ground truth 再调一次 `observe` 回喂**，AI 才知道自己出了这个动作。

### `GET /ai/sessions/{id}` — 状态查询

无请求体。返回 session 当前状态——`is_terminal`、`winner`、`actions_observed` 等。

### `DELETE /ai/sessions/{id}` — 关闭

释放内部 GameSession 和 ONNX 运行时资源。

---

## 一个完整的 curl 示例（Quoridor 2 人对弈）

```bash
# 1. 创建 session，AI 扮演玩家 0
SID=$(curl -sX POST http://localhost:8000/ai/sessions \
  -H 'Content-Type: application/json' \
  -d '{"game_id":"quoridor","seed":7,"my_seat":0}' \
  | jq -r .session_id)

# 2. AI 先手。要 AI 出招
MOVE=$(curl -sX POST http://localhost:8000/ai/sessions/$SID/decide | jq -r .action_id)
echo "AI plays: $MOVE"

# 3. 把这个动作应用到 ground truth（第三方系统）
# (省略第三方游戏服务器的 apply 调用)

# 4. 把 AI 的动作回喂 observe（让 AI 内部状态同步）
curl -sX POST http://localhost:8000/ai/sessions/$SID/observe \
  -H 'Content-Type: application/json' \
  -d "{\"action_id\":$MOVE,\"events\":[],\"public_snapshot\":{}}"

# 5. 对手出招（假设对手选了 action 13）
curl -sX POST http://localhost:8000/ai/sessions/$SID/observe \
  -H 'Content-Type: application/json' \
  -d '{"action_id":13,"events":[],"public_snapshot":{}}'

# 6. 轮到 AI 再决策，回到步骤 2 ...

# 7. 游戏结束，关闭
curl -sX DELETE http://localhost:8000/ai/sessions/$SID
```

---

## 公开事件（隐藏信息游戏必读）

对有隐藏信息的游戏（Splendor / Azul / Love Letter / Coup），接入方必须在 `observe` 里附上正确的 `events` 列表和 `public_snapshot`，否则 AI 的 belief tracker 会偏离 ground truth、observer 的公开 state 也会失真。

事件流和 snapshot 各管一摊：
- **`events`**：这次 transition 里 observer 能看到的公开事实序列（牌被翻开、tokens 被支付、谁宣称了什么角色），由 GT 端计算后 broadcast，**只喂给 belief tracker** 用来更新对隐藏信息的推断；不会作用到 observer 的 state 上。list 内顺序就是 producer 发出的顺序，tracker 只把它当事实序列消费，不区分动作前后。
- **`public_snapshot`**：动作之后 GT 端所有公开 slot 的值（按 schema field name 索引）。observer 收到后整体覆写自己 session 的公开部分——observer 路径上**没有 `do_action_fast`**，公开局面完全靠 snapshot 反向重建。

具体每个游戏发什么事件、payload 格式、哪些 slot 在 snapshot 里，详见每个游戏的专属 `docs/games/<game>_api.md`。

---

## 错误处理

| HTTP 状态 | 场景 |
|----------|------|
| 400 | game_id 不存在、`observe` 时 action 非法、session 已关闭 |
| 404 | `session_id` 不存在 |
| 409 | 状态冲突（如调 `decide` 时轮不到 AI，或调 `observe` 时传对手动作但 AI 的内部 state 认为是自己的回合） |

所有错误响应都是 `{"detail": "具体原因"}`。

---

## 性能提示

- 一个 session 对应一份独立的 ONNX 运行时实例。大量并发会话需要注意内存。
- 决策延迟主要由服务端配置的 `simulations` 决定（线性）。2p Quoridor 的 800 sim 大约 300ms（视 CPU），Splendor 大约 500ms。调整在 `web.json::difficulty_overrides.expert` 里改。
- 单个 session 内部的 `decide` 是线程安全的——多个 session 间的并发 request 互不干扰。
- 长时间不用的 session 应该及时 `DELETE`；没有自动 GC。

---

## 给新游戏写 API 说明的模板

新增游戏时，必须为它写一份 `docs/games/<game>_api.md`，至少包含：

1. **动作空间总览**：总动作数，高层分类
2. **逐项编码表**：每个 action_id 范围对应什么含义（用 `offset` / `count` 和可读的公式表达）
3. **事件列表**（若有隐藏信息）：每种 kind、payload 结构、谁能看到
4. **多人变体差异**：2p / 3p / 4p 的动作空间差异（如果有）
5. **序列化的 state_dict 字段**（若第三方想读 `status`）：哪些是公开、哪些是私有
6. **一个完整的示例对局**：curl 或 Python 代码演示

模板见 `docs/games/tictactoe_api.md`（最简单的例子）和 `docs/games/splendor_api.md`（随机 + 隐藏信息的复杂例子）。

---

## 当前支持的游戏

| 游戏 | API 文档 |
|------|---------|
| TicTacToe | [docs/games/tictactoe_api.md](games/tictactoe_api.md) |
| Quoridor | [docs/games/quoridor_api.md](games/quoridor_api.md) |
| Splendor | [docs/games/splendor_api.md](games/splendor_api.md) |
| Azul | [docs/games/azul_api.md](games/azul_api.md) |
| Love Letter | [docs/games/loveletter_api.md](games/loveletter_api.md) |
| Coup | [docs/games/coup_api.md](games/coup_api.md) |
