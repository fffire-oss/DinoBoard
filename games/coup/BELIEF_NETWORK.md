# Coup 信念网络（Learned Belief Posterior）

本文档介绍 Coup 上启用的网络化 belief 实现。框架级接口/调用流见
[ALGORITHM_OVERVIEW §8.4](../../ALGORITHM_OVERVIEW.md#84-可选网络化-belieflearned-posterior)；
配置字段见 [CONFIG_REFERENCE belief 块](../../docs/guide/CONFIG_REFERENCE.md#belief-网络可选)。
本文只讲 Coup 自己的特征 / 输出 / tracker 状态机 / 采样公式。

---

## 1. 为什么 Coup 需要它

Coup 的 root determinization 必须给每个对手抽一组具体的影响牌。最朴素
的 `randomize_unseen` 是从 `remaining[R]`（公开池子里 R 角色还剩几张）
按 R 的张数 weighted 抽：

```
weight[R] = remaining[R]    // uniform-by-pool
```

这条在 Splendor 这种"袋里有什么我都看得到"的游戏里没问题。**Coup 上
有结构性 bug**：

1. **诈唬游戏的纳什均衡是混合策略**——但 ISMCTS 训练的是确定性策略。
   均衡上"该 claim Duke 的频率"是一条软概率；网络只能拟合 argmax。
2. **如果 belief 也 uniform**，对手早期 claim Tax 的信号在 sim 里完全
   消失——sim 入口随手把对手抽成 Captain / Ambassador 都和"实际 claim
   过 Duke"一样。结果：网络很快学会**永远 Challenge**——因为它在 sim
   世界里看到的对手手牌跟 claim 无关，挑战的胜率被 inflate。
3. 实测：早期 selfplay 在 vs heuristic_free 上能跑到 95% 但策略空洞，
   gating vs best 来回震荡；定性看 AI 不会装、也不会基于 claim 做有
   信息的决策。

**Coup 实际想要的 weight**：

```
weight[R] = remaining[R] × P(opp 持有 R | 公开历史)
```

第二项的 `P(...)` 是后验，理想形态是个跟 claim/challenge 历史相关的
分布。手写一份 hand-craft 加权能避开 uniform 退化，但权重函数的形状
是作者手挑的（每个 signal 加多少 alpha、pending claim 加多少 boost），
跟 Coup 真实的策略均衡几乎肯定不对齐。

**网络化 belief 的解法**：训练一个独立的小网络，输入 perspective 视角
下的 claim/challenge 公开历史，输出 `(N-1) × 5` 的 opp×role 后验
`pi[opp][R]`。把它喂进 Wallenius 加权采样：

```
weight[R] = remaining[R] × pi[opp_index(slot.owner)][R]
```

对 deck 槽位（`pi_` 没有 row）保留 `prior = 1.0`——deck 是公开剩余池
本身。

---

## 2. 整体数据流

```
                  GT (truth)                          AI session
                  ────────                            ──────────
                                                      tracker
                                                      ├── observe_public_event(...)  ← claim/challenge 事件流
                                                      └── prepare_for_root(masked_root, root_player,
                                                                            extractor, evaluator)
                                                            │
                                                            ▼
                                          ┌─── extractor.extract_from_state ───┐
                                          │   (输入 6 + 28×(N-1) 维 features)    │
                                          └────────────┬───────────────────────┘
                                                       ▼
                                         ┌── OnnxBeliefEvaluator.evaluate ──┐
                                         │      (5×(N-1) logits)            │
                                         └────────────┬─────────────────────┘
                                                      ▼
                                  softmax(logits / belief.temperature) per opp
                                                      ▼
                                            tracker.pi_ 缓存
                                                      ▼
       ┌─ MCTS sim 入口 ───────────────────────────────┐
       │ sim_tracker = session.tracker.clone()         │ ← pi_ 一起 clone
       │ sim_tracker.randomize_unseen(sim_state, ...)  │
       │   ↳ weight[R] = remaining[R] × pi[oi][R]     │
       │      Wallenius 抽样填 viz=0 槽位             │
       └───────────────────────────────────────────────┘

  selfplay 训练时另一条 GT-side label 流：
  ────────────────────────────────────────
  GT runner 每 ply 调 belief_label_extractor.extract(truth, observer)
    → (hand_counts[opp][R], remaining[R], alive[opp])
  连同 observer 视角 features 一起 emit 成 BeliefSample
  下一个 PV-step 训练循环里跑 KL(softmax(logits/T) ‖ hand_counts/remaining)
```

四条物理屏障保证决策侧不读真值（详见
[ALGORITHM_OVERVIEW §8.4](../../ALGORITHM_OVERVIEW.md#84-可选网络化-belieflearned-posterior)
"关键性质"）：

1. `extract_from_state` 入参锁 `const IGameState& masked_state`，非虚
   入口先 `make_masked_state` 再 forward，viz=0 永远是 placeholder；
2. `belief_label_extractor` 物理上只在 GT runner 持有，`AI session /
   tracker / wire / 训练循环之外` 都拿不到；
3. `randomize_unseen` 在 sim_tracker（clone）上调，session tracker 永远
   不被 sim 改写；
4. `pi_observer_` 校验：sim 里要用网络化路径必须 `pi_observer_ ==
   observer`，否则降级回 hand-craft signals 路径（见 §6）。

---

## 3. Tracker 维护的状态机

Tracker 的 raw 字段（`coup_net_adapter.h:`）：

| 字段 | 形状 | 含义 |
|---|---|---|
| `signals_[p][r]` | `N × 5` | 通用 hand-craft 加权器（fallback 路径用） |
| `pre_claim_counts_[p][r]` | `N × 5` | **跨过最近一次洗牌**累计的 claim 次数 |
| `post_claim_counts_[p][r]` | `N × 5` | 自最近一次洗牌**之后**新的 claim 次数 |
| `pre_challenge_initiated_[p][r]` | `N × 5` | 同上，发起挑战的次数 |
| `post_challenge_initiated_[p][r]` | `N × 5` | 同上 |
| `last_reshuffle_kind_[p]` | `N` | `{None, Exchange, RevealTruthful}` 之一 |
| `last_revealed_role_[p]` | `N` | 上次洗牌对应被亮出的 role（Exchange 时 -1） |
| `pending_claimer_` / `pending_claim_role_` / `pending_challenged_` | scalars | 正在解决的 claim cycle 状态 |

### 3.1 claim/challenge 事件如何累加

每收到一个 public event，处理在 `observe_public_event`
（`coup_net_adapter.cpp:264-393`）：

- **Action-level（claim opening）**：`actor` 出 Tax / Steal / Assassinate
  / Exchange / Block 之一 → `post_claim_counts_[actor][claimed_role] += 1`，
  并把 `(pending_claimer_, pending_claim_role_)` 设到这个 actor。
- **Action-level（challenge declared）**：`actor` 出 Challenge → `signals_
  [actor][pending_claim_role_] += 1`（"我敢挑战 R 说明我倾向认为我手里
  有 R"），同时 `post_challenge_initiated_[actor][pending_claim_role_]
  += 1`。
- **Event `claim_unchallenged` / `block_unchallenged`**：通过 → `signals_
  [c][r] += 1`。**不动 pre/post**——没洗牌。
- **Event `claim_resolved_truthful` / `block_resolved_truthful`**：
  挑战失败、claimer 把那张牌洗回牌堆（Coup 规则）→
  `signals_[c][r] = 0` + `promote_post_to_pre(c)` +
  `last_reshuffle_kind_[c] = RevealTruthful` + `last_revealed_role_[c] = r`。
- **Event `card_revealed`**：lose-influence / 挑战成功导致那张影响牌
  公开亮出 → `signals_[p][r] = 0`。**不 promote**——那张牌物理离场而
  非回牌堆，pre/post 不动。
- **Event `exchange_complete`**：Ambassador 换完牌 → `signals_[p].fill
  (0)` + `promote_post_to_pre(p)` + `last_reshuffle_kind_[p] = Exchange`。

### 3.2 promote_post_to_pre：累加，不替换

关键的洗牌处理（`coup_net_adapter.h:114`）：

```cpp
void promote_post_to_pre(int player) {
  for (int r = 0; r < kCharacterCount; ++r) {
    pre_claim_counts_[player][r] += post_claim_counts_[player][r];   // 加上
    post_claim_counts_[player][r] = 0;                                // post 清零
    pre_challenge_initiated_[player][r] +=
        post_challenge_initiated_[player][r];
    post_challenge_initiated_[player][r] = 0;
  }
}
```

语义上 `pre + post = 这个玩家全游戏累计 claim 次数`。**洗牌前的
claim 不会被丢弃**，只是被打上"已经经过一次洗牌、信息含金量降低"的
标记——下一轮 claim 落到 fresh `post`，belief 网络可以根据这两个数
组的 split 学到"刚 claim 的更可信" vs "claim 过但已经洗过牌、可能不
持有了"的区别。

### 3.3 pending claim cycle

Coup 一个 claim 跨多个 ply（claim → 对手 challenge / 对手 allow → 揭
牌 / 跳过）。tracker 用 `pending_claimer_ / pending_claim_role_ /
pending_challenged_` 三个 scalar 跟踪这个 cycle：

- 出牌阶段（任一 claim-bearing action） → 设 pending；
- Challenge → 设 `pending_challenged_ = true`；
- claim_resolved_truthful / block_resolved_truthful / card_revealed
  对上 pending → 清 pending；
- claim_unchallenged / block_unchallenged 对上 pending → 清 pending。

Fallback 采样路径会在 `pending_claimer_ == slot.owner && pending_claim_
role_ == r && !pending_challenged_` 时给 R 加 +2 boost——belief 网络
打开后这条 boost 不再生效（走 §6 的 `pi_` 优先级）。

---

## 4. Feature 输入（28×(N-1) + 6 维）

由 `CoupBeliefFeatureExtractor::extract`（`coup_net_adapter.cpp:748` 起）
emit。布局：

### 4.1 Global block（6 维）

| 维度 | 含义 |
|---|---|
| 5 | `remaining[R]` for `R in {Duke, Assassin, Captain, Ambassador, Contessa}`，observer 视角公开剩余池 |
| 1 | `ply_count / 200`，进度 |

`remaining[R]` 的算法是：3（每角色总数）− 公开 revealed 的 R 张数 −
observer 自己未 reveal 的 R 张数。其它玩家未 reveal 的槽位是 viz=0
→ kPlaceholder，**结构性贡献为 0**——这正是 belief 网络要估算的部分。

### 4.2 Per-opp block（28 × (N-1) 维）

按 `opp_to_player(observer, i, N) = (observer + 1 + i) % N` 的 i 顺序
排列——这与 belief net 输出的 `(N-1) × 5` 行索引对齐。每个对手 28 维：

| 维度 | 含义 |
|---|---|
| 5 | `pre_claim_counts[R] / 4` clamp [0, 1] |
| 5 | `post_claim_counts[R] / 4` clamp [0, 1] |
| 5 | `pre_challenge_initiated[R] / 4` clamp [0, 1] |
| 5 | `post_challenge_initiated[R] / 4` clamp [0, 1] |
| 3 | `last_reshuffle_kind` one-hot ∈ {None, Exchange, RevealTruthful} |
| 5 | `last_revealed_role` one-hot（kind != RevealTruthful 时全 0） |

**所有维度都来自 tracker 的公开事件累积**——没有任何字段读 truth state
的 `influence[]`。这是网络无法泄漏的物理保证。

---

## 5. 网络结构与输出

| 项 | 值 |
|---|---|
| 输入维度 | 6 + 28 × (N-1) = `34/62/90`（2p/3p/4p） |
| 输出 | `(N-1) × 5` logits per opp × role |
| 网络 | MLP `[256, 256, 128]`（默认）+ ReLU + BatchNorm |
| 温度 | `belief.temperature = 2.0` |
| 后处理 | `pi_[opp][R] = softmax(logits[opp] / 2.0)`，per-row |

输出 logits 经 softmax + 缓存到 tracker `pi_` 字段。`temperature=2.0`
是诈唬游戏避开"网络欠训时过度自信"的妥协——欠训阶段 logits 噪声大，
温度高一些让 prior 更软，sim 不会过早被错的高自信先验带偏。temperature
→ ∞ 等价 hand-craft uniform，正好回到 §1 那条退化情形（保护降级行为
在数学上一致）。

---

## 6. Wallenius 加权采样

`randomize_unseen`（`coup_net_adapter.cpp:411-646`）在 sim 入口被 clone
后的 sim_tracker 调用一次，把 viz=0 槽位填上具体 role：

```
对每个 viz=0 slot：
  for R in {Duke, Assassin, Captain, Ambassador, Contessa}:
    avail = remaining[R] - 已分配[R]
    if avail == 0:
      weight[R] = 0
      continue
    if pi_valid_ and pi_observer_ == observer and slot is opp slot:
      prior = pi_[opp_index(slot.owner)][R]
    elif slot is opp slot:                   # 网络降级
      prior = 1 + alpha * (signals_[owner][R] + pending_boost)
    else:                                    # deck slot
      prior = 1.0
    weight[R] = avail × prior

  multinomial(weights)
```

注意点：

1. **池一致性硬约束**——若 `sum(remaining) != slot count` 直接 throw
   `std::logic_error`，按 CLAUDE.md "No Fallbacks"。
2. **observer 自己的槽位**结构性 viz=1，不应进 `slots`；如果走到这里
   说明 viz mis-tagged，throw。
3. **Wallenius 而非 Fisher**——每抽一张减少 `remaining[R]` 而不是按
   独立伯努利重抽。这才匹配"deck 是固定 15 张" 的物理约束。
4. **Belief 降级路径保留**——belief.onnx 没加载（早期训练）/ 没
   `prepare_for_root` / `pi_observer_ != observer` 时回到 §3.1 的
   hand-craft signals，Coup selfplay 在 belief 未训练好时仍可用。

---

## 7. 训练数据与损失

`CoupBeliefLabelExtractor`（`coup_net_adapter.cpp:850`）在 selfplay
runner 的 emit 路径调用，**只在 GT 端**：

```cpp
struct BeliefSample {
  std::vector<float> features;          // 6 + 28×(N-1)
  std::vector<std::vector<int>> hand_counts;  // (N-1) × 5
  std::vector<int> remaining;            // 5
  std::vector<int> alive_per_opp;        // (N-1)
};
```

label 来自 truth：
- `hand_counts[i][R]` = opp i 当前未 reveal 的 R 角色张数（0/1/2）
- `remaining[R]` 同 §4.1 公式但走真值
- `alive_per_opp[i]` = opp i 是否还有任意未 reveal 的影响牌

**Loss = KL divergence**（`coup/config/game.json::belief.training.loss
= "kl"`）：

```
target[opp][R] = hand_counts[opp][R] / max(1, alive[opp])     # row-prob
loss = mean over (opp, R) where alive[opp]: KL(softmax(logits[opp]/T) ‖ target[opp])
```

dead opps（`alive=0`）不贡献 loss——他们的手牌已经全部公开 reveal
出来了，没什么"belief"可学。

训练循环（`training/pipeline.py`）：每 PV step 跑 `belief.training.
steps_per_pv_step` 次 belief mini-batch（默认 1）；独立 cosine LR
schedule（默认 `lr_max=3e-4 → lr_min=3e-5`）；独立 replay buffer
`b_buf=200000`。日志里 `b_loss` 即此项 KL，目前 step 200 在 ~0.85
（完全 uniform target 下 KL ≈ 1.6，说明已经学到大量公开信息约束）。

---

## 8. Config 参考

`games/coup/config/game.json::belief` 块：

```json
"belief": {
  "architecture": [256, 256, 128],
  "activation": "relu",
  "batch_norm": true,
  "temperature": 2.0,
  "training": {
    "lr_schedule": {"type": "cosine", "lr_max": 3e-4, "lr_min": 3e-5},
    "batch_size": 1024,
    "steps_per_pv_step": 1,
    "loss": "kl"
  }
}
```

详见 [CONFIG_REFERENCE belief 块](../../docs/guide/CONFIG_REFERENCE.md#belief-网络可选)。

---

## 9. 文件索引

| 文件 | 内容 |
|---|---|
| `coup_net_adapter.h:85-170` | tracker 字段 + `promote_post_to_pre` |
| `coup_net_adapter.h:171-220` | `CoupBeliefFeatureExtractor` 接口 |
| `coup_net_adapter.cpp:264-393` | `observe_public_event` 状态机 |
| `coup_net_adapter.cpp:411-646` | `randomize_unseen` Wallenius 采样 |
| `coup_net_adapter.cpp:649-700` | `prepare_for_root`（softmax + cache） |
| `coup_net_adapter.cpp:741-840` | `extract`（28×(N-1)+6 维 features） |
| `coup_net_adapter.cpp:850-908` | `CoupBeliefLabelExtractor::extract` |
| `coup_register.cpp` | GameBundle 三字段注册 |
| `model/coup_belief_2p.onnx` 等 | ship 出去的 ONNX |

新人接手时，按 §3 → §4 → §5 → §6 → §7 顺序读这些文件即可。
