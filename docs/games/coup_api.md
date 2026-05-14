# Coup — AI API 接入文档

## 概览

- **game_id**：`coup` / `coup_2p` / `coup_3p` / `coup_4p`
- **玩家数**：2–4
- **动作空间**：32
- **隐藏信息**：有（每人 2 张 influence 卡 + deck 剩余卡序列）
- **公开事件**：当前 Coup 的 `public_event_extractor` **不发任何 events**（`out.events` 始终为空）。所有"对手揭示一张牌"、"Ambassador 换牌完成"等语义都通过 `(actor, action_id)` + `public_snapshot` 的 `influences[*].revealed/character` 槽位 + `viz` 切片来表达——`reveal_slot_to` 把 viz=1 翻给所有人，walker 把真值塞进 snapshot
- **特点**：AI 用**启发式概率 sampler**（而非均匀采样）来采样对手暗牌——tracker 在 `observe_public_event(actor, action, events=[])` 的 phase 1 中根据 claim/challenge 动作累积信号，无需游戏端额外发事件

---

## 动作编码

固定 32 个动作 ID：

| 范围 | 类型 | 说明 |
|------|------|------|
| 0 | Income | +1 币 |
| 1 | Foreign Aid | +2 币（可被 Duke 挡） |
| 2–5 | Coup target 0..3 | 花 7 币强制 target 丢一张 influence |
| 6 | Tax | +3 币（claim Duke，可被挑战） |
| 7–10 | Assassinate target 0..3 | 花 3 币让 target 丢 influence（claim Assassin，可被挑战或 Contessa 挡） |
| 11–14 | Steal target 0..3 | 从 target 偷 2 币（claim Captain，可被挑战或 Captain/Ambassador 挡） |
| 15 | Exchange | claim Ambassador，从 deck 抽 2 张换手牌 |
| 16 | Challenge | 质疑当前 claim |
| 17 | Allow | 不挑战（普通动作） |
| 18 | Block Duke | claim Duke 挡 Foreign Aid |
| 19 | Block Contessa | claim Contessa 挡 Assassinate |
| 20 | Block Ambassador | claim Ambassador 挡 Steal |
| 21 | Block Captain | claim Captain 挡 Steal |
| 22 | Allow No Block | （不挡 Foreign Aid，默认流程） |
| 23 | Reveal Slot 0 | 被挑战后揭示 slot 0 影响 |
| 24 | Reveal Slot 1 | 被挑战后揭示 slot 1 |
| 25 | Lose Slot 0 | 被 coup / 挑战失败时选择丢 slot 0 |
| 26 | Lose Slot 1 | 同上，slot 1 |
| 27–31 | Return 角色 | Ambassador exchange 最后把某角色牌还回 deck。27=Duke, 28=Assassin, 29=Captain, 30=Ambassador, 31=Contessa |

**多阶段回合**：Coup 的一个"回合"可能产生多个 action_id（出招 → 被挑战 → 揭示 → 丢牌 等）。AI 每收到一个 action_id 就更新一步内部 state，直到 `current_player` 走到下一真正的决策者。

---

## 公开事件

Coup 协议下 `events` 始终是 `[]`。所有可见信息——揭示的角色身份、Ambassador 换牌后手牌发生重洗——都体现在 `public_snapshot`：rules 端在 `do_action_fast` 里调 `reveal_slot_to(...)` 把 `influences[player][slot]` 的 `revealed/character` 槽位的 viz 翻成对所有人 1，walker 自动把这些槽位的真值序列化进 snapshot。

接入方不需要构造任何 PublicEvent payload；tracker 的 `observe_public_event` 通过 `(actor, action_id)` 自身就能累积所有 claim/challenge/reveal 信号（见下文 "Coup 特有：启发式 belief"）。

---

## state_dict 字段

公开：`alive`（每玩家）、`coins`（每玩家）、`influences[i]`（长度 2 的数组，其中 `revealed=true` 的 `character` 可见）、`deck_size`、`stage`、`declared_action`（当前被挑战的 action）、`active_player`（出招方）、`action_target`、`blocker`、`challenger`、`ply`

私有：`influences[my_seat]` 里 `revealed=false` 的 `character`（自己的暗牌）。对手的暗牌不可见，AI 靠启发式 sampler 采样。

---

## 完整示例（Python）

```python
import requests
BASE = "http://localhost:8000"

initial_observation = gt_session.extract_initial_observation(3)
sess = requests.post(f"{BASE}/ai/sessions", json={
    "game_id": "coup_4p", "seed": 555, "my_seat": 3,
    "initial_observation": initial_observation,
}).json()
sid = sess["session_id"]

# AI 需要所有玩家的 claim / challenge / reveal 事件才能正确建模
while True:
    status = requests.get(f"{BASE}/ai/sessions/{sid}").json()
    if status["is_terminal"]: break

    if status["current_player"] == 3:
        r = requests.post(f"{BASE}/ai/sessions/{sid}/decide").json()
        action_id = r["action_id"]
    else:
        action_id = opp_pick()

    # Ground truth 端必须：
    # 1. 应用 action 到自己的 state
    # 2. 算出公开事件序列（card_revealed / exchange_complete / 等）+ 动作之后的 public_snapshot
    events, snapshot = compute_trace(action_id)
    requests.post(f"{BASE}/ai/sessions/{sid}/observe", json={
        "action_id": action_id, "events": events, "public_snapshot": snapshot,
    })

requests.delete(f"{BASE}/ai/sessions/{sid}")
```

---

## Coup 特有：启发式 belief

Coup 的 tracker 不是 uniform 采样。它在内部维护每个对手对每个角色的**信号计数**：

- 对手**claim 某角色且未被挑战** → +1 信号
- 对手**被挑战成功/失败** → 相关信号清零（牌已经暴露/洗回 deck，信号失效）
- 对手**挑战别人的某 claim** → 该对手对该角色 +1 信号（他敢挑战，说明他自己可能持有）
- 对手**揭示某角色** → 清空对那角色的信号
- 对手**完成 Ambassador exchange** → 清空所有信号

采样时权重 = `pool_remaining × (1 + 0.5 × signal_count)`，加硬约束保证全局守恒（每角色总共 ≤ 3 张）。

**接入方什么都不用做**——tracker 的 `observe_public_event` 在 `events=[]` 情况下也会从 `(actor, action_id)` 推出所有 claim 类、challenge、allow 动作，自己累积信号。

详见 `games/coup/coup_net_adapter.cpp::CoupBeliefTracker::observe_public_event`。

---

## 常见踩坑

- **多阶段回合要逐 action 喂**：一次出招 → 挑战 → reveal → 丢牌 是 4 个连续的 action。每个 action 都必须 `observe` 一次，每次都带最新的 `public_snapshot`，不能跳过
- **被揭示的牌靠 `viz` + snapshot 自动同步**：rules 在 `do_action_fast` 里调 `reveal_slot_to(everyone, "influences", {p, slot})` 后，walker 把 `influences[p][slot].character/revealed` 写进 snapshot；observer 整体覆写后 viz=1，无需任何额外事件
- **不要尝试自己构造 `card_revealed` / `exchange_complete` payload**：tracker 不消费这两个 kind（旧文档的描述已过时）。换牌完成后的"信号清零"逻辑也已迁移到 `(actor, action_id)` 驱动，不需要游戏端额外标记
