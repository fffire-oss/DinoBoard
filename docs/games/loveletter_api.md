# Love Letter — AI API 接入文档

## 概览

- **game_id**：`loveletter` / `loveletter_2p` / `loveletter_3p` / `loveletter_4p`
- **玩家数**：2–4
- **动作空间**：47
- **隐藏信息**：有（每人手牌 + 轮到的人抽牌 + 牌堆顺序）
- **公开事件**：LL **不发任何 `events`**（`out.events` 始终为空）。所有"reveal 对手手牌"语义（Priest peek、Baron 比较、King 交换、Prince 弃手）都由 GT 端 rules 调 `viz::reveal_slot_to` / `swap_slot_owned` 写进 `state.viz_`，schema-driven walker 把这些 viz=1 槽位的真值塞进 `public_snapshot` 的 `hand[*]` / `drawn_card`，observer 用 snapshot 整体覆写后那些槽位天然 viz=1 + 真值。**接入方只发 `public_snapshot`**，`events` 传 `[]`。

---

## 动作编码

Love Letter 的动作覆盖所有"打出某张牌 + 目标（若该牌需要目标）+ 猜测（Guard 专用）"的组合：

| 范围 | 类型 | 说明 |
|------|------|------|
| 0–27 | Guard 猜 | `target × 7 + (guess - 2)`。target 0..3，guess 值 2..8（7 种猜测；不能猜 Guard=1 因为规则不允许） |
| 28–31 | Priest | target 0..3（偷看该玩家手牌） |
| 32–35 | Baron | target 0..3（和该玩家比牌） |
| 36 | Handmaid | 自己保护一轮 |
| 37–40 | Prince | target 0..3（让该玩家弃手牌并重抽，可选自己） |
| 41–44 | King | target 0..3（交换手牌） |
| 45 | Countess | 自己打出 Countess（无效果，但规则强制与 King/Prince 同手时必须打） |
| 46 | Princess | 自己打出 Princess（立刻自我淘汰） |

**target** 用的是**相对座位**：target=0 表示自己下一位玩家，target=1 是下下位……映射到真实玩家：`target_player = (current_player + 1 + target) % num_players`。2p 游戏里只有 target=0 合法；3p/4p 多选。

**Guard 猜牌的 `guess` 值**：
| 值 | 牌 |
|----|----|
| 2 | Priest |
| 3 | Baron |
| 4 | Handmaid |
| 5 | Prince |
| 6 | King |
| 7 | Countess |
| 8 | Princess |

---

## 公开事件

LL 协议下 `events` 始终是 `[]`——所有 perspective-private 的 reveal 都通过 viz schema + walker 走 `public_snapshot` 通道，**不再有平行私人事件**。

`public_snapshot` 由 GT 端的 schema walker（`viz::serialize_public(state, schema, perspective)`）每个 perspective 各产出一份。它包含该视角下 viz=1 槽位的所有真值：

- **自己手牌** (`hand[my_seat]`)：始终 viz=1
- **自己当前抽到的牌** (`drawn_card`)：在自己回合 reveal 后 viz=1
- **被 Priest peek 的对手手牌**：peek 后 viz=1 给 actor，第三方仍 viz=0
- **King 交换后双方的新手牌**：swap 后双方对各自手牌 viz=1，并相互对对方 viz=1
- **Baron 比较后**：分支 1（牌不等）loser 全员 viz=1；分支 2（平手）actor 与 target 互相 viz=1
- **Prince 让 target 弃手**：target 原手牌进 discard pile，公开

接入方只需要：rules 端按规则正确调用 `reveal_slot_to(viewer)` / `swap_slot_owned(a, b)` / `reset_to_base(slot)`，walker 自动把 viz=1 槽位序列化进 snapshot——不用单独发任何 `hand_override` / `drawn_override` 事件。

---

## state_dict 字段

公开：`alive`（每玩家 bool）、`protected_flags`（Handmaid 状态）、`discard_piles`（每玩家的弃牌堆）、`hand_exposed`（Countess 被宣告等公开信息）、`deck_size`、`face_up_removed`（2p 开局移走的 3 张）、`ply`、`current_player`、`winner`

玩家私有：`hand[my_seat]`、`drawn_card`（若 `current_player == my_seat` 且已抽牌）。被 Priest peek / Baron 比较 / King 交换合法看到的对手手牌也以 viz=1 出现在 my_seat 的 session state 里——不需要 AI 端单独维护 `known_hand` 字典，直接读 session state 上对应 `hand[opp]` 槽位即可。

---

## 完整示例（Python）

```python
import requests
BASE = "http://localhost:8000"

# AI 扮演 3 人局的第 1 号玩家
sess = requests.post(f"{BASE}/ai/sessions", json={
    "game_id": "loveletter_3p", "seed": 99, "my_seat": 1,
    "simulations": 800, "temperature": 0.0,
}).json()
sid = sess["session_id"]

# Ground truth 端要实现：每个动作算出 events + public_snapshot
def compute_trace(state_before, action_id, state_after, perspective):
    """对照 games/loveletter/loveletter_register.cpp::extract_events 的语义，
    返回 {"events": [], "public_snapshot": {...}}。
    LL 不发 events（所有对手手牌 reveal 走 viz=1 + walker 进 snapshot）；
    public_snapshot 是动作之后所有公开 slot 的值（按 perspective 分别产出），
    observer 收到后整体覆写公开字段。"""
    ...

while True:
    status = requests.get(f"{BASE}/ai/sessions/{sid}").json()
    if status["is_terminal"]: break

    if status["current_player"] == 1:
        r = requests.post(f"{BASE}/ai/sessions/{sid}/decide").json()
        action_id = r["action_id"]
    else:
        action_id = opp_pick()

    trace = compute_trace(state_before, action_id, state_after, perspective=1)
    requests.post(f"{BASE}/ai/sessions/{sid}/observe", json={
        "action_id": action_id,
        "events": trace["events"],
        "public_snapshot": trace["public_snapshot"],
    })

requests.delete(f"{BASE}/ai/sessions/{sid}")
```

---

## 常见踩坑

- **observer 路径不重放规则**：动作的具体后果（弃牌、King 后双方的新手牌、谁淘汰、Priest peek 的对手 cid）全部由 `public_snapshot` 携带；接入方只需保证 rules 端正确调 `reveal_slot_to` / `swap_slot_owned` / `reset_to_base` 维护 viz，walker 自动把每个 perspective 的 viz=1 槽位序列化进 snapshot
- **Priest peek 是 per-perspective 的 viz reveal**：`reveal_slot_to(actor)` 只让 actor 这一个 perspective 的 viz 转 1，第三方 perspective 的 snapshot 里那个槽位仍是 placeholder——信息隔离结构上保证
- **淘汰时手牌自然公开**：被 Guard 猜中、Baron 比输、Prince 弃的 Princess 等情况，被淘汰的玩家手牌进入 discard pile，AI 通过 discard_piles state 自然知道
- **多人座位轮转**：`target` 是**相对座位**。3/4 人局里同一 action_id 对不同的 actor 指的 target 玩家不同
