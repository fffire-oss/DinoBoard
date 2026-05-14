# Splendor — AI API 接入文档

## 概览

- **game_id**：`splendor` / `splendor_2p` / `splendor_3p` / `splendor_4p`
- **玩家数**：2–4
- **动作空间**：随玩家数变化（2p=70 / 3p=71 / 4p=72）。`kChooseNobleCount = num_players + 1`，多一个贵族就多一个 action_id。其它分段（buy / reserve / take token / return / pass）规模不变
- **隐藏信息**：有（暗牌：玩家可以盲预订一张自己看得但对手看不见的 deck 卡）
- **公开事件**：`deck_flip`（post）、`self_reserve_deck`（post，仅 perspective 自己的盲预订）、`opp_buy_reserved_reveal`（pre，对手买自己暗牌时揭示）

---

## 动作编码

按 offset / count 分段：

| 范围 | 类型 | 说明 |
|------|------|------|
| 0–11 | 买 faceup 卡 | 买 tableau 上第 `(tier × 4 + slot)` 张卡。tier 0..2，slot 0..3 |
| 12–23 | 预订 faceup 卡 | 预订 tableau 上第 `(tier × 4 + slot)` 张卡（公开） |
| 24–26 | 盲预订 deck 顶 | 从第 `tier` 层（0/1/2）deck 顶预订一张牌（对手看不到具体牌） |
| 27–29 | 买自己预订的卡 | 买自己第 `(action_id - 27)` 张已预订卡（slot 0/1/2） |
| 30–39 | 拿 3 个不同宝石 | 10 种组合（从 5 色里选 3） |
| 40–49 | 拿 2 个不同宝石 | 10 种组合（bank 不足 3 色时才合法） |
| 50–54 | 拿 1 个宝石 | 5 色各一个（bank 几乎空时用） |
| 55–59 | 拿 2 个同色宝石 | 5 色各一个（该色 bank ≥ 4 时合法） |
| 60+ | 选贵族 | 多张贵族可选时选哪一张。2p：60–62；3p：60–63；4p：60–64 |
| 紧随贵族段 | 归还宝石 | 拿宝石后超过 10 个上限时归还一个；5 色 + 金币 = 6 种。2p offset=63、3p=64、4p=65 |
| 最后一个 | Pass | 极少用（仅在无合法动作的死角）。2p=69、3p=70、4p=71 |

**TakeThree 组合的 id → colors 映射**见 `games/splendor/splendor_rules.cpp:kTakeThreeCombos`。从 action_id 反推色彩需要查这个表。实际上接入方通常不需要反推，只需要把 legal_actions 里的 id 原样传给 AI——AI 的 `action_info` 字段会返回人类可读的描述（`type`, `colors`, `card_id` 等）。

---

## 公开事件

### `deck_flip`（post）

tableau 上某格的卡变了——可能是买/预订后翻新，也可能是位置重排。

```json
{
  "kind": "deck_flip",
  "payload": {"tier": 0, "slot": 2, "card_id": 47}
}
```

- `tier`：0–2（对应游戏内 tier 1–3）
- `slot`：0–3
- `card_id`：新放上去的卡 ID（-1 表示该 slot 变空）

**何时发**：任何动作后，对比 `state_before.tableau[tier][slot]` vs `state_after.tableau[tier][slot]`，不相等就发一个 `deck_flip`。

**注意**：一次 `buy_faceup` 或 `reserve_faceup` 动作会让 tier 里所有后续 slot 发生位移（slot 0 被拿走 → slot 1/2/3 都左移一位）。所以**一个动作可能发 4 个 deck_flip**（整个 tier 重排），这是正常的。

### `self_reserve_deck`（post）

仅当 `my_seat == actor` 且动作是 `reserve_deck`（24–26）时发——AI 自己盲预订，事件告诉 AI 具体拿到的是哪张牌。

```json
{
  "kind": "self_reserve_deck",
  "payload": {"player": 0, "slot": 2, "card_id": 58}
}
```

对手盲预订时**不发此事件**（对手看不到自己拿的具体牌）。

### `opp_buy_reserved_reveal`（pre）

当对手买自己的**盲预订**卡时（action 27/28/29 且 `reserved_visible==0`），由于卡的成本被从 bank 扣除（公开），卡的身份不得不揭示。在 AI 执行 `do_action_fast` 之前，先把这张卡的真实 id 告诉 AI，否则 AI 用采样的占位牌算出来的 gem 消耗会错。

```json
{
  "kind": "opp_buy_reserved_reveal",
  "payload": {"player": 1, "slot": 0, "card_id": 22}
}
```

---

## state_dict 字段（观察者视角）

公开的：`bank`（长度 6，索引 0..4 = 色 0..4，索引 5 = 金币）、`tableau`（3 tiers × 4 slots × card_id）、`nobles`、`deck_sizes`（`[tier0, tier1, tier2]`）、`current_player`、`plies`、`scores`、`players[i].bonuses`、`players[i].gems`、`players[i].points`、`players[i].cards_count`、`players[i].reserved`（其中 `visible=true` 的 `card_id` 可见）、`stage`、`pending_returns`

玩家私有的：自己的 `players[my_seat].reserved` 里 `visible=false` 的卡（对手看不到），AI 通过 `self_reserve_deck` 事件追踪

---

## 完整示例（Python）

```python
import requests
BASE = "http://localhost:8000"

initial_observation = gt_session.extract_initial_observation(0)
sess = requests.post(f"{BASE}/ai/sessions", json={
    "game_id": "splendor_2p", "seed": 777, "my_seat": 0,
    "initial_observation": initial_observation,
}).json()
sid = sess["session_id"]

# 假设 ground truth 从 splendor_rules 自己维护。它在每一步计算 observation 事件。
# 这里用伪代码表示。
def apply_on_ground_truth(action_id):
    """返回 (events, public_snapshot) — 由 ground truth 端计算。
    具体实现见 games/splendor/splendor_register.cpp::extract_events 的翻译版。
    events: 这次 transition 里 observer 能看到的公开事实序列（按 producer 顺序）。
    public_snapshot: 动作之后 GT 端所有公开 slot 的值（按 schema field name 索引）。"""
    ...

while True:
    status = requests.get(f"{BASE}/ai/sessions/{sid}").json()
    if status["is_terminal"]: break

    if status["current_player"] == 0:
        r = requests.post(f"{BASE}/ai/sessions/{sid}/decide").json()
        action_id = r["action_id"]
    else:
        action_id = opp_pick_action()  # 你的对手逻辑

    events, snapshot = apply_on_ground_truth(action_id)  # 你的 ground truth 算事件 + snapshot
    requests.post(f"{BASE}/ai/sessions/{sid}/observe", json={
        "action_id": action_id, "events": events, "public_snapshot": snapshot,
    })

requests.delete(f"{BASE}/ai/sessions/{sid}")
```

---

## Tail solver

Splendor 注册了 tail solver。`decide` 返回 `stats.tail_solved=true` 时表示必胜/必负已证明。

---

## 常见踩坑

- **不发 `deck_flip` → AI 的 tableau 滞后**：接入方最容易忘的是"只卡换了新牌时发事件"，但**位移也算变化**——任何 slot 的 card_id 变了就要发
- **对手盲预订不发 `self_reserve_deck`**：这是隐藏信息，AI 不应该看到。AI 会在 `randomize_unseen` 里自己采样一个占位牌
- **`opp_buy_reserved_reveal` 必须出现在 `events` 里**：observer 不会重放规则，所以"对手盲预订的牌被买出"这种揭露事实必须显式发到 tracker，否则 belief 会偏（observer state 的公开部分由 `public_snapshot` 整体覆写——cost 不会算错；事件主要是给 tracker 维护对手盲预订集合用的）
