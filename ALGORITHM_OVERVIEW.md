# DinoBoard - 算法概览

> 这是 DinoBoard **算法层的第一份文档**。新读者从这里入门:先建立词汇表,再看一句话主线,然后按 §1–§9 顺读。加新游戏主要看 §12 的开发者职责清单。
>

## 名词表

读后面章节前先认识这些词。它们是本框架的核心抽象,在代码里都有对应的类
或函数名。表里按**数据 → 角色 → 推进 → 派生 → 三家消费者 → belief 栈 →
算法**排。

**数据**(state 的形状)

| 名词 | 定位 | 一句话 |
|------|------|--------|
| **schema** | 字段元数据声明 | 每个字段的 name + 数据 shape + base viz tensor;可见性**完全由 viz 承载**。**单一事实源**,hash / encoder / snapshot 都按 schema 序遍历 |
| **state** | 游戏的所有数据 | schema 实例化后的字段集合(scores / hands / decks / boards / ...);**每个字段都额外挂一个 viz** |
| **viz**(visibility mask) | 字段级可见性张量 | 与字段同 shape;`viz[i, p] = 1` 表示玩家 p 此刻能看到该字段第 i 槽的真值 |

**角色**(谁持有 state、谁跑搜索)

| 名词 | 定位 | 一句话 |
|------|------|--------|
| **GT**(ground truth) | 跑游戏的"上帝视角" | 持有完整 state(含全部隐藏字段),按 rules 推进、抽真随机、判终局,每 ply 给每个 perspective 广播一份 perspective 视图 |
| **AI session** | 一个 perspective 的客户端 | 收 GT 的 perspective 视图,只持有自己能看见的 state;驱动 MCTS、把决策返回给 GT |
| **sim** | MCTS 内一次模拟 | 从 session 克隆出一个 state,先把未知字段采样成具体值,然后 deterministic 地往下走 |

**推进**

| 名词 | 定位 | 一句话 |
|------|------|--------|
| **rules** | 推进 state 的代码 | `do_action_fast(state, action, rng)`,**同时**改业务字段和 viz——是 viz 的**唯一 writer** |

**派生流水线**(从 state + viz 算出 perspective 视图)

| 名词 | 定位 | 一句话 |
|------|------|--------|
| **walker** | schema 驱动的遍历器 | 一遍走完 state 的所有字段、所有槽位,对每个槽位决定"该露还是该藏" |
| **MaskedState** | walker 的产物 | state + perspective + viz=0 槽位被替换成 placeholder 哨兵的视图;**snapshot / hash / encoder 三家共用** |

**三家消费者**(都从同一份 MaskedState 出发)

| 名词 | 定位 | 一句话 |
|------|------|--------|
| **snapshot** | wire 协议负载 | GT 每 ply 发出的消息:`public_snapshot` 是按字段名 keyed 的 AnyMap,框架 walker 按 schema 顺序对每个 slot 检查 `viz[..., viewer]`,只把 viz=1 的真值 emit 进去;AI 端 wholesale 替换 session.state 的公开部分 |
| **hash** | MCTS 节点 key | 按 perspective 哈希 MaskedState,**外加 `step_count` 一并入 hash 以防 DAG 成环**;同一信息集在 DAG 里共享一个节点 |
| **encoder** | 网络输入张量化 | 把 MaskedState 转成 `vector<float>`,viz=0 槽位永远读到 placeholder——结构上不可能编进真值 |

**belief 栈**(隐藏信息游戏才需要)

| 名词 | 定位 | 一句话 |
|------|------|--------|
| **tracker** | belief 沙池 + (可选)网络特征源 | `IBeliefTracker` 一个类干两件事:`observe_public_event` 维护观察记忆——只从公开事件流推,不读 state 真值;`randomize_unseen(state, observer, rng)` 按这份记忆把 viz=0 槽位填上具体值。同时记忆本身也可作为公开历史的派生聚合(Splendor 公开展示牌、Coup claim 历史)喂给网络。**注册与否取决于游戏需不需要 viz=0 sim 入口采样**:有非对称隐藏信息的游戏必须注册(Love Letter / Splendor / Coup);纯公开物理随机的 Azul 不注册——它的物理随机由 sim 入口的 sim_rng 在 `do_action_fast` 里直接消费;TicTacToe / Quoridor 没有任何 hidden 也不需要 |
| **belief** | 未知字段的一个采样世界 | `tracker.randomize_unseen(state, observer, rng)` 的产物——viz=0 槽位被填成具体值的完整 state,sim 入口用。**最朴素的实现是从未见过的池子里 uniform 抽**(Splendor 就走这条);需要更精细的先验(如 Coup 的 claim-driven 加权)由 game 在 tracker 里写自定义采样 |

**算法**

| 名词 | 定位 | 一句话 |
|------|------|--------|
| **ISMCTS** | 本框架的搜索算法 | Information Set MCTS:每次 sim 持独立 RNG,root 从 belief 采一个 determinized world,descent 给定 sim 种子完全可复现(同一 sim_rng 也驱动 descent 里 `do_action_fast` 的物理随机),用 perspective hash 共享 DAG 节点;详见 §9 |

## 一句话主线

整套契约在 GT 和 AI 两端共享 schema + rules + walker,但角色不同:

> **GT 端**(持有完整 state):rules 推进 state、同时维护 `state.viz_`
> → walker 按每个 perspective 把 state mask 成 `MaskedState` →
> 打包成 snapshot 广播给对应 AI session。
>
> **AI 端**(每个 perspective 一个 session):收 snapshot 整张替换 session
> state(含 viz)→ 维护 tracker(public 历史的派生摘要)→ MCTS 每个
> sim 入口从 belief 采样一个具体世界——`tracker.randomize_unseen` 把
> viz=0 槽位填成具体值,得到 determinized state → 同一份 rules 推进、
> 同一份 walker mask → hash / encoder 消费 MaskedState、按 perspective
> hash 共享 DAG 节点 → ISMCTS。

每个箭头都是一条强约束(§11 不变量速查),违反任何一条都不是风格问题
——是结构性 bug,CI 必须立刻红。

## 怎么读这份文档

- **第一次读**:§0 五条原则 → §1 三个角色 → §5 walker / MaskedState → §9 MCTS 主算法。其他章节按需查。
- **加新游戏**:§12 开发者职责清单。先确认你的游戏属于 §12.1 / §12.2 / §12.3 哪一类。
- **改框架某一处**(viz / RNG / hash / encoder / tracker / 协议):先读 §11 的不变量速查,再去对应章节(§2 / §3 / §4 / §5 / §6 / §7 / §8 / §10)。

---

## 0. 五条原则

1. **可见性是规则的一部分**：一个字段对谁可见、什么时候翻面公开、回合
   末是否翻回私人，这些是设计游戏时就要回答的问题。**rules 是 viz 的唯
   一 writer**。框架提供 schema、helper、walker、序列化协议；不替作者决
   定语义。
2. **state 是充分统计量**：rules 推进游戏所需的全部信息都在 state（含
   `viz_`）里。任何"靠历史才能算"的字段必须物化进 state，rules 不接受
   外部 history buffer。
3. **同一根水线，三家消费**：snapshot、hash、encoder 都从
   `make_masked_state(state, schema, perspective)` 出发，不重复遍历、
   不绕过 viz。这是 §5 的核心。
4. **Tracker 同时是 belief 沙池和（可选）网络特征源**：`IBeliefTracker`
   一个类干两件事——`observe_public_event` 维护观察记忆、
   `randomize_unseen` 按记忆把 viz=0 槽位填上具体世界。注册与否跟"游戏
   有没有随机/隐藏"无关,而是看作者要不要缓存 GT 不 care 的公开衍生特征
   喂网络;有 viz=0 槽位的游戏额外用它承载 `randomize_unseen`。
5. **Sim 入口的 belief 采样是纯函数**：产出世界中所有公开槽位（任意
   perspective 都能看到的 viz=1 槽位）的内容只取决于 tracker 看过的
   event，与输入 state 的 hidden 内容、调用者 RNG 都无关。

---

## 1. 角色：GT、AI session、sim

### 1.1 GT（ground truth）

GT 的工作是：**假设一切信息都已固定（hidden 全部确定下来），维护游戏的
进行**——按规则推进 state、判合法、判终局、抽真随机、广播 ply 后的
perspective snapshot。GT 不做 belief、不做搜索、不做信息集推断；它就是
"完整可观察世界下的一台游戏机器"。

每 ply 结束 GT 给每个 perspective 发一条
`(actor, action_id, events, public_snapshot)`，载体是
`PublicEventTrace`（`engine/core/game_interfaces.h:47`）：

- `events: std::vector<PublicEvent>`——这次 transition 里对该 perspective
  公开可观察到的事实序列，**只用来喂 tracker**（不再驱动 observer state
  mutation）。list 内顺序就是 producer 发出的顺序，tracker 把它当一段事实
  日志吃掉。例：Splendor `deck_flip`（tableau 翻新）/ `self_reserve_deck`
  （perspective 自己 reserve deck top 时只发给自己）/ `opp_buy_reserved_reveal`
  （对手买暗压牌时公开那张 reserved 的真实 `card_id`）/ Azul `factory_refill`
  / Coup `card_revealed` / `exchange_complete` / `self_influence_redraw`
- `public_snapshot: AnyMap` = 按字段名 keyed 的整张 post-action 公开视图；
  framework walker 按 schema 顺序遍历 slot，对 `viz[..., viewer]=1` 的 slot
  调 `read_field_slot` 写入，viz=0 的 slot 不写。viewer 端按 schema 顺序
  `apply_public` 反向写回 state（无独立的 visibility_mask 字段、无 bit-packing）
- 每个 `PublicEvent = std::pair<std::string, AnyMap>`；`kind` 字符串由游戏
  自定（`"deck_flip"` / `"opp_buy_reserved_reveal"` / ...），`payload` 里的
  key 也由游戏决定。**没有 phase 概念**——observer 不跑 `do_action_fast`，
  事件没有"动作前 vs 动作后"可挂的时机
- **`actor` 必传**——多人协议下 current_player 不一定等于 actor（Coup
  challenge 由非当前玩家触发）；event audience（self-reveal = actor only）
  依赖 actor

事件流的语义负载就是 tracker belief 更新——`tracker.observe_public_event(
actor, action, events)` 把这次 transition 的事实折进观察记忆。observer 的
session state **不再跑 `do_action_fast`**：public 字段直接被
`public_state_applier(public_snapshot)` 整张覆盖，**viz=0 hidden 槽位
session 端不维护**(决策侧物理上读不到——hash 用 `kHiddenHashSentinel`、
encoder 用 placeholder、MCTS sims 在自己的克隆上 determinize)。事件不应
用到 state 上，只应用到 tracker 上。完整一次 ply 的时序见 §10.2。

### 1.2 AI session

AI 端面对信息不全。收到
`observe(actor, action_id, events, public_snapshot)`:

- **整张替换** session 公开字段 ← `public_snapshot`(每个 field 走
  `write_field_slot` / SnapshotApplier;不 OR-merge、不重算 viz)
- 若游戏有 tracker,`events` 喂给 `tracker.observe_public_event(actor,
  action, events)` 维护观察记忆——事件**只**走 tracker 这一条路
- viz=0 的 hidden 槽位 ← session **不维护**。决策侧没人读 viz=0:MCTS 在
  sim 入口自己 determinize(§1.3),hash 对 viz=0 混 `kHiddenHashSentinel`
  (§5),encoder 只读 `MaskedState` 里的 placeholder(§6)。session state 的
  viz=0 槽位留着上一次 observe 的旧值即可,无需 freshen
- perspective 私人事实(自己的 hand、Priest 偷看到的对手手牌、对手公开了
  card_id 的 reserved 槽 ...)都由 `public_snapshot` 自然带回——只要当前
  `viz[..., perspective]=1`(不论是 base 就如此还是 rules 临时翻明的),
  truth 端 walker 在 perspective 自己的 snapshot 里就会写入真值,observer
  端 `apply_public` 整张覆盖时直接落回对应槽位。schema 的 base viz 只决定
  "未发生任何 reveal 时谁能看",运行时翻面完全跟着 rules 跑(§4)

**snapshot-path 游戏的 session 不跑 `do_action_fast`、不跑 `randomize_unseen`**
——这一类(Splendor / Love Letter / Coup / Azul) session 只负责"接 snapshot + 喂
tracker",viz=1 整张覆盖、viz=0 留着上次的旧值,等下一个 observe 就行。"采一个
具体世界"是 sim 入口的事(§1.3),session 不重复做。

> 例外:**fully-public no-snapshot 游戏**(TicTacToe / Quoridor)没注册
> `public_state_applier`,这类游戏的 session 路径上**会**跑
> `do_action_fast(seat, view_step_rng)`——它们没有 viz=0 槽位,也没有 snapshot
> 协议,推进 state 的唯一办法就是替这一座位 replay 那个动作。但仍然不跑
> `randomize_unseen`(没有 viz=0 槽位要填),也不读 truth 任何字段:输入只有
> `(action_id, seat 自己的当前 state)`。这条例外不破坏"决策路径不读 truth"——
> 它只是"snapshot 不存在时,怎么把公开转移做出来"的实作答案。

**AI 端永远不在 observe 阶段重新算 viz**——viz 只在 GT 端算一次、协议
传过来。运行时正确性 = 数据传输完整性。

**perspective 的"私人知识"全部表达成 `viz[..., perspective]=1` 的 state
槽位**——不存在"state 之外的私人事实"。Coup exchange 看过的牌也是 state
字段，rules 在抽看时 `reveal_slot_to(viz, ..., actor)`。

### 1.3 sim（MCTS 内）

每个 sim 入口：

```cpp
state = session.state.clone_state();             // viz 作为 state 的字段一起 clone
sim_tracker = session.belief_tracker.clone();    // tracker 也 clone，sim 内可写
sim_tracker->randomize_unseen(*state, observer, per_sim_rng);  // 把 viz=0 槽位填上具体值
```

填完 state 完整（所有字段都有值）才能跑 `do_action_fast`。Descent 每步
走 GT 端**同一份** `do_action_fast`，rules 顺手维护 sim-local viz；descent
途中如有公开 event，sim_tracker 也照常 `observe_public_event` 累积——反
正 sim 结束 state + tracker 一起丢弃，下个 sim 从 session 重新克隆。
**绝不 cross-sim 污染**。

> Rules 引擎只服务于"已完全确定"状态：GT 直接喂真值跑一次；AI 在 sim 里
> sample 出一个具体世界后跑同一份代码。这条决定了下面所有"谁负责什么"。

---

## 2. Visibility schema

每款游戏在 `games/<id>/<id>_visibility.cpp` 声明 state 字段元数据：

- state 里**每一个**字段都必须 declare：name + 数据 shape + base viz
  tensor。schema 没列的字段 register-time 校验 fail
- 字段是 shape `[D1, …, Dk]` 的张量（scalar = `k=0`），其可见性张量
  shape = `[D1, …, Dk, NPlayers]`，**末尾多挂一根 viewer 轴**——每个
  数据槽位独立带一个 `bool[NPlayers]`
- **base viz** 是字段在"未发生任何翻面"时的可见性——纯静态值：全公开 =
  `all_public`、owner-only 手牌 = `owner_only_first_axis`、对所有人不可见 =
  `all_hidden`
- declare 的 shape 与 viewer-axis 大小由 `declare_field`
  (`visibility_schema.h:148`)在 register-time 校验:字段名重复、shape
  与 base viz tensor 形状不匹配等会立即抛错。**编译期反射对齐不存在**
  ——字段是运行时字符串(`name + std::vector<int> idx`),写错字段名要
  到 register / 单测才暴露

schema 只回答"这个字段长什么样、最静态可见性如何"——什么时候翻面是
rules 的事（§4）。可见性完全由 viz tensor 承载：walker / hash / encoder
对每个槽位看 `viz[..., perspective]`，0 即跳过，1 即按真值消费。

---

## 3. RNG 管理

state 里框架级字段只有 `step_count_`(DAG 防环,§9.5)和 `viz_`,
**RNG 不在 state 上、不进 hash、不进 wire**。

每个游戏的 `reset_with_seed(seed)` 内部用一次性 `std::mt19937_64(seed)`
做初始洗牌即弃,函数返回时栈变量销毁。运行时只有三类 RNG,**互不派生、
互不共享、各自由调用方独立 seed**:

| RNG | 持有者 | 生命周期 | 用途 |
|-----|-------|---------|------|
| **gt_rng** | GT runner | 整局 | GT 端所有 `do_action_fast` |
| **session_rng** | 每个 AI session 一份 | session 整寿命 | session 不调 `do_action_fast`、不调 `randomize_unseen`;public state 从 snapshot 整体重建,viz=0 槽位不动。session_rng 只在内部 step_rng_/seed 推导上使用 |
| **sim_rng** | sim 局部栈变量 | 一次 sim | sim 入口的 `sim_tracker.randomize_unseen` + descent 中所有 `do_action_fast` |

每个 rng 在自己的作用域内被反复消费(每次 `rng()` 推进内部状态),这是
`std::mt19937_64` 本来的工作方式——不是"调用一次就废"。

**关键约束**:三者之间不互相派生——`session_rng` 不从 `gt_rng` salt 派
生,`sim_rng` 不从 `session_rng` salt 派生。任何一处 reseed 不影响别处。
GT 端、AI session 端、sim 端的随机决策因此完全解耦,这条是 ISMCTS 各端
独立性的物理基础。

---

## 4. Rules：唯一的 viz writer

### 4.1 `do_action_fast_impl` 同时维护 state 和 viz

游戏作者重写的是 protected `do_action_fast_impl`（不是 public 的 `do_action_fast`）。后者是框架的 non-virtual wrapper，在调用 impl 之前自动 `state.step_count_ += 1`，作者既看不到也无法忘记。

```cpp
void do_action_fast_impl(State& s, Action a, std::mt19937_64& rng) const {
    if (a.type == kRevealInfluence) {
        // 业务字段
        s.influence[a.player][a.slot].revealed = true;
        // viz: 这张牌从 owner-only 变全公开
        viz::reveal_slot(s, "influence", {a.player, a.slot});
    }
    if (a.type == kPlayHand) {
        s.hand[a.player] = kEmpty;
        // 槽位空了，viz 重置回 base
        viz::reset_to_base(s, "hand", State::schema(), {a.player});
    }
    // step_count_ 由框架的 `IGameRules::do_action_fast` wrapper 在调用
    // `do_action_fast_impl` 之前自动 +1（§9.5 DAG 防环唯一原因），
    // 游戏作者既看不到也无法忘记。
}
```

framework 提供的 helper(仅简化常见模式,不替作者决定语义):

- `viz::reveal_slot(IGameState& s, const std::string& name,
   const std::vector<int>& idx)`——把指定槽位 viz 设为全 1
- `viz::reveal_slot_to(IGameState& s, const std::string& name,
   const std::vector<int>& idx, int viewer)`——只对 viewer 可见
  (Priest peek 风格)
- `viz::reset_to_base(IGameState& s, const std::string& name,
   const VisibilitySchema& schema, const std::vector<int>& idx)`——
   重置回 schema 声明的 base(因此需要 schema 入参)

字段 path 是**运行时字符串**(不是编译期 member-pointer),`declare_field`
register-time 校验存在性。写错字段名编译过、register-time 抛错。
**没有"编译期 viewer-axis 校验"的语法保证**——这层只跑运行时检查。

### 4.2 `do_action_fast` 不支持 undo

`do_action_fast` 是 MCTS / selfplay / arena / web 的热路径，sim 用完直接
丢弃 state，不会 undo。**`do_action_fast_impl` 里不要 push undo_stack、
不要拍 viz 快照**——sims 跑成百万次，多余的 clone 是纯浪费。

如果游戏要接 tail solver，**额外**重写一对配对的 protected impl：

- `do_action_deterministic_impl(state, action)`——和 `do_action_fast_impl`
  同样的状态推进，但**禁止从 hidden 源抽**（deck / bag / opp hand）。一般用
  game-side sentinel（如 Splendor 的 `forced_draw_override = -2`）freeze
  随机。这一路径**支持 undo**：push undo_stack、snapshot 任何被改的字段
- `undo_action_impl(state, token)`——配对的恢复入口，按 token 还原
  state / viz_。框架的 `IGameRules::undo_action` wrapper 在 impl 返回后
  自动 `state.step_count_ -= 1`，作者不需要、也不能直接碰 step_count_

`undo_action` 的两个调用点都在 `engine/search/tail_solver.cpp`，永远跟
`do_action_deterministic` 配对。**绝不要在 `do_action_fast_impl` 里 push
undo_stack** 然后指望 `undo_action` 能恢复——那条路径的 state 不会被
restore。

### 4.3 viz 不单调

viz **不是**单调累积量——槽位内容变化时 rules 主动 `reset_to_base`。
例：

- Love Letter 出牌：`hand[p]` 清空 → `reset_to_base(viz, hand, p)` →
  玩家摸新牌后 hand 重新变 owner-only
- Coup 出牌（非 reveal）：影响牌槽位换内容时 rules 决定保持
  `revealed=true` 还是重置 base

下游不依赖单调性：

- AI session 是**整张替换** not OR-merge（merge 会把已经翻回私人的槽位
  错锁成公开）
- MCTS sim-local viz 跟着 rules 走，不需要单独的 `update_viz` 步骤

---

## 5. Walker 一次产出 MaskedState，三家共用

这是整份文档的 fulcrum。

### 5.1 `make_masked_state`

```cpp
auto masked = make_masked_state(state, schema, perspective);
// MaskedState = IGameState typedef，跟 State 同 struct layout
```

内部用 walker 按 schema 遍历每槽，对每槽位检查 `viz[..., perspective]`：

- =1 → 复制真值
- =0 → 写 `kPlaceholder` sentinel
  (`INT32_MIN / INT8_MIN / false`；变长字段长度保留、内容置 placeholder)

物化由 `IGameState::mask_all_hidden_slots` 驱动——非虚的 base 实现走
walker，对每个 viz=0 的槽位调虚的 `mask_field_slot(name, idx)`，游戏
override 它写 placeholder。

### 5.2 三家消费者，同一份 MaskedState

| 消费者 | 何时调用 | 怎么读 | 出口 |
|---|---|---|---|
| **snapshot** | GT 每 ply | 按 schema 序遍历 MaskedState:对每个 `viz[..., viewer]=1` 的 slot 调 `read_field_slot` 写 `public_snapshot` dict;viz=0 的 slot 不写。base viz 是 owner-only / all_hidden 的字段,如果 rules 临时翻成 viewer 可见(LL Priest peek、Splendor `opp_buy_reserved_reveal` 之后的对手 reserved 槽),也走这条路自然带回 | wire 序列化 |
| **hash** | sim descent 每步 | 按 schema 序遍历，每槽位调 `hash_field_slot(name, idx)` 合 digest，placeholder 合固定 sentinel | `StateHash64` (DAG key) |
| **encoder** | sim 新节点 | `encode_with_masked(masked, perspective, ...)` | `vector<float>` (NN input) |

**hash 和 encoder 共用同一份 MaskedState**——sim descent 每步 mask 一
次，先 hash（决定是不是新节点），是新节点才 encode，两者吃同一个副本。
GT 端 snapshot 是另一条独立 mask 调用（perspective 是 GT 视角下当前
回合的 viewer）。

**state 里所有变量都被 schema 收编,不留后门**——hash / snapshot / encoder
读到的每个字段都必须挂在 FieldDecl 上,没有"绕开 walker 的游戏侧手写
emitter"这条逃生通道。

### 5.3 防呆是结构性的

接口入参类型锁 `const MaskedState&`：

```cpp
class IFeatureEncoder {
  virtual void encode_public(const MaskedState&, int perspective,
                             const IBeliefTracker* tracker,
                             std::vector<float>* out) const = 0;
  virtual void encode_private(const MaskedState&, int perspective,
                              const IBeliefTracker* tracker,
                              std::vector<float>* out) const = 0;
};

class IPolicyValueEvaluator {
  virtual bool evaluate(const MaskedState& masked, int perspective_player,
                        const IBeliefTracker* tracker,
                        const std::vector<ActionId>& legal_actions,
                        std::vector<float>* priors,
                        std::vector<float>* values) const = 0;
};
```

`tracker` 提供 perspective 视角下的公开衍生特征（如 Splendor 的 `seen_cards` 多重集统计、Coup 的 claim 历史聚合）；perspective 私密知识不在 tracker 上（它在 `state.viz_` 的 viz=1 槽位上），所以 encoder 读 tracker 拿到的永远是公开聚合，不会泄漏。游戏没注册 tracker 时传 `nullptr`。

读 viz=0 槽位只能读到 placeholder——MaskedState 上 placeholder 自身就是
"看不见"的信号，作者不需要查 viz 也不需要 `is_visible_to` 之类的辅助
函数。忘记按 placeholder 分流最多 emit 错误占位符特征，不会泄漏真值。

> **Rules 例外**：rules 要写、要算合法性、要读隐藏字段（如要从牌堆抽
> 牌、要看自己手牌内容），仍直接吃 `State&`，不走 walker / MaskedState。

### 5.4 完整链路图

```
GT 端
  state(GT) ──make_masked_state──▶ MaskedState ──serialize_public / SnapshotIO──▶ wire (public_snapshot)
                                                                          │
                                                                          ▼
                                                          AI 端 apply_observation
                                                          (session 公开字段 ← public_snapshot;
                                                           viz=0 hidden 槽 session 不维护——
                                                           hash 看 kHiddenHashSentinel,
                                                           encoder 看 placeholder)

每个 MCTS sim：
  session.state ──clone──▶ sim_state
                              │
                              ▼
          tracker->randomize_unseen(sim_state, observer, rng)  ← hidden 槽填具体值
                              │
                              ▼
              descent 每步：do_action_fast(sim_state, action)
                              │  ← rules 顺手维护 sim_state.viz_
                              ▼
                  make_masked_state(sim_state, ...) → MaskedState
                              │
                              ├─▶ hash → DAG key
                              │
                              └─ 新节点？─是─▶ encode_with_masked（同一份副本）
                                          ─否─▶ 复用旧节点 prior
```

---

## 6. Encoder：`MaskedState → vector<float>`

**输入面 = MaskedState + tracker**——和 hash / belief 同一条原则：网络
只能编码 perspective 视角下能看到的事实。viz=0 槽位（不论 belief 有没
有 sample 过）真值一律不可读，因为 §5.3 的接口锁保证 encoder 拿到的就是
placeholder。

**张量形状是 schema 决定的固定 shape**——encoder 必须按 schema 声明的
槽位逐一 emit 特征，不能因为某槽位 viz=0 就跳过。所以 viz=0 槽位 emit
**占位符特征**：

| 槽位状态 | MaskedState 里的值 | encoder 行为 |
|---|---|---|
| `viz[slot, perspective]=1` | 真值 | 编码具体特征 |
| `viz[slot, perspective]=0` | `kPlaceholder` | emit 占位符特征 |

座位旋转就是 encoder 自己排 tensor 时的索引顺序：
`mstate.hand[(perspective+i) % N]`——layout 排序，不是 mask 工序，框架
不为它单独提供 walker 变种。

派生量（stage one-hot、plies ratio、`current_player == perspective`）的
底料都是 schema 已有字段（stage / ply / current_player）的展开形式，
encoder 在 MaskedState 上做 one-hot / ratio / flag 即可。

**Public/private 分割**：

- `encode_public` 没有 player 参数，**MUST NOT** 读任何玩家的 private
  字段（用 `encode_public` 的纯接口签名作语法守护）
- `encode_private(perspective)` **MUST NOT** 读其他玩家的 private 字段
- 默认 `encode(...)` 把两者依次拼成 flat tensor，和 hash 拼接顺序对齐
- `encode_with_masked(masked, ...)` 是 sim descent 的快路径，跳过二次
  clone+mask；`encode(state, ...)` 内部 `make_masked_state` 一次后转发

CI 守护：`test_encoder_respects_hash_scope` 验证 opp private 变化、
public + own private 不变时 encoder bit-equal。这是 encoder/hash scope
对齐的硬守门员。

---

## 7. Tracker（可选的网络特征缓存）

### 7.1 边界

**state 由 rules 维护**——推进游戏（do_action_fast、合法性、终局）所
需的全部信息都在 state 里。任何 rules 用得到的字段必须在 state 中。

**tracker 维护"GT 不 care、但 NN 想要的公开衍生特征"**——纯网络输入辅
助，跟规则、合法性、协议、belief 都无关。例：Coup 每家对各角色的历史
claim 分布，GT 推进游戏不读这个，network 想拿来当特征。

判别准则：**这个量 GT 端的规则引擎需不需要？**

- 需要 → 进 state（含 viz）
- 不需要、但 NN 想要 → 进 tracker
- 不需要、NN 也不要 → 不存在

**perspective 的"私人知识"建模成 `viz[..., perspective]=1` 的 state 字
段**，不在 tracker 里。Coup exchange 看过又放回去的 2 张牌也是 state 字
段——rules 在 exchange action 里 `reveal_slot_to(..., actor)`。

### 7.2 接口与寿命

```cpp
class IBeliefTracker {
  void init(const AnyMap& initial_observation);
  void observe_public_event(int actor, ActionId action,
                            const std::vector<PublicEvent>& events);
  void randomize_unseen(IGameState& state, int observer,
                        std::mt19937_64& rng) const;
  std::unique_ptr<IBeliefTracker> clone() const;
  AnyMap serialize() const;            // 调试 / 测试用
};
```

实现可以从 `events` 现场算（推荐），或维护 incremental 状态（性能考虑）。
tracker 接口里**没有 `IGameState*`**——`init` 拿 `AnyMap`、
`observe_public_event` 拿事件流；只有 `randomize_unseen` 拿 state，但那是
**写**端（向 viz=0 槽填值），不是读真值的入口。`observer` 形参由
`randomize_unseen` 传入,tracker 自身 perspective-agnostic。

寿命：session 启动时 `init` 一次,之后每收到 public event 调一次
`observe_public_event`。MCTS sim 入口**clone 一份 sim_tracker**:先调
`randomize_unseen` 写 state,descent 途中遇到公开 event 同样 `observe_
public_event` 累积；sim 结束 sim_tracker 连同 state 一起丢弃,session 持
有的 tracker 不被污染。

**tracker 不进 hash、不进协议**——节点 identity 只走 schema slot 的
state hash;wire 协议只传 `(actor, action, events, public_snapshot)`,
tracker 自己的内部状态不上线。但 **tracker 参与决策路径**:`randomize_
unseen` 是 ISMCTS 每个 sim 入口必经一步(sim_tracker 在 cloned sim_state
上把 viz=0 槽位填出一个具体世界,直接决定该 sim 的合法动作和 value
backup),encoder 读 tracker 的衍生公开统计是另一条用途。`belief` 公式可
以读 tracker 的公开统计当辅助输入,但**不允许把 prior 缓存进 tracker**。

当前六款游戏的实际选择：TicTacToe / Quoridor / Azul 没注册 tracker——
前两者无随机无隐藏、没有 NN 衍生特征想缓存,Azul 的物理随机也是发牌时
即时从公开 `bag_counts` 用 rng 抽,作者也没要喂 NN 别的衍生统计。
Splendor / Love Letter / Coup 注册了 tracker(`randomize_unseen` + 可选
encoder 衍生特征)。如果哪天 Azul 作者想喂 NN 一些公开手算统计,照样可
以注册 tracker——这是设计选择,不是规则要求。

---

## 8. Belief = `tracker.randomize_unseen`

### 8.1 接口

Belief 不是独立类——它就是 `IBeliefTracker::randomize_unseen` 这一个
方法：

```cpp
void randomize_unseen(IGameState& state, int observer,
                      std::mt19937_64& rng) const;
```

- 读：tracker 自己累积的观察 + state 上 perspective 可见的槽位（viz=1）
- 写：state 上 perspective 不可见的槽位（viz=0）

调用语义:**唯一调用点是 MCTS sim 入口**(在 cloned sim_state 和
cloned sim_tracker 上),传入一个公开字段已对齐 observer 视角的 state,
tracker 按自己累积的 belief 把 viz=0 hidden 槽填上一个具体世界。
session 自己**不**调 `randomize_unseen`(DEC-003)——session 的 viz=0
槽位是 unread bytes,决策路径(hash / encoder / sim 入口)都用结构性手
段从读侧屏蔽掉,不需要 freshen。

约束：产出世界中所有公开槽位（任意 perspective 都能看到的 viz=1 槽位）
的内容，必须**只**取决于 tracker 的观察历史，与输入 state 的 hidden 内
容、调用者的 RNG 无关——换句话说，重放同一观察序列产出的 MaskedState
公开部分必须 byte-equal。

### 8.2 简单情况 vs 加权情况

- 纯信息博弈但简单（Love Letter）：rules 通过 `reveal_slot_to` /
  `swap_slot_owned` 把 Priest 偷看 / Baron / King 后的确定信息写到
  viz=1 槽位；tracker 自身 stateless，`randomize_unseen` 只对剩余 viz=0
  槽位做剩余牌池均匀采样
- 隐藏多重集（Splendor 三层 deck 顶部）：tracker 维护剩余多重集，
  `randomize_unseen` uniform 抽取
- bluff game（Coup）：tracker 维护 claim/challenge 历史，
  `randomize_unseen` 用历史驱动的加权联合采样，避免诈唬游戏的 uniform
  退化均衡

**没有"框架 default UniformBelief"——每个隐藏信息游戏都要自己写一份
`randomize_unseen`**（哪怕只是简单 uniform，也要自己枚举 unseen pool）。
Splendor / Love Letter / Coup 各自实现。Azul 没有隐藏字段，根本不需
要 belief tracker。

### 8.3 默认退化场景

**简单 uniform 在私人 state field 上退化**：把私人记忆建模成 state 字段
后（如 Coup exchange 看过的两张牌），observer 在那一格的 `viz=0`，
`randomize_unseen` 默认就是从牌堆里随机取——不反映 actor 当时的策略偏
向。框架不挡这条退化，作者要强 AI 必须在自家 `randomize_unseen` 里加
weighted prior（Coup tracker 已经做了 claim-driven weighting）。不写就
是退化为均匀，AI 弱一档，但游戏正常进行。

---

## 9. MCTS = ISMCTS over a DAG

§1–§8 把 state、viz、RNG、MaskedState、encoder、tracker、belief 都铺好了。
本节讲 MCTS 怎么把它们串成搜索算法。

### 9.1 七条互锁的实现属性

1. **Root 采样 determinization**：每个 sim 持独立 RNG;root 步
   `tracker->randomize_unseen(state, observer, sim_rng)` 采一个完整
   世界,descent 期间该 sim_rng 仍可能被 `do_action_fast` 消费处理物
   理随机(如 Azul 工厂 refill)。给定 sim 种子整个 sim 完全可复现
2. **Per-acting-player 节点 keying**：`state_hash_for_perspective
   (state.current_player())` 作 key——哪位玩家在决策，就用那位玩家的
   信息集
3. **博弈 DAG**（不是 tree）：per-search `unordered_map<StateHash64, int>`
   表(每次 `search_root` 调用本地构建,sim 间共享、search 间不复用),
   acting player 视角下 `viz=1` 的所有槽位 + `step_count` 一致,无论从
   哪条路径到达都是同一个节点
4. **UCT2 UCB**：`sqrt()` 分子底用"刚经过的入边"的 visit_count，不是
   node 的 global visit_count（§9.6）
5. **无 chance node 专门机制**：每个 sim 持独立 RNG,root 步先调
   `randomize_unseen` 把 viz=0 槽位填成具体世界,descent 期间该 RNG 仍
   可能被 `do_action_fast` 消费处理物理随机(如 Azul 工厂 refill),给
   定 sim 种子整个展开完全可复现。
   - **有 belief tracker 的游戏**(LL / Splendor):root 步从信息集中
     采一个世界,unseen 池由 tracker 维护
   - **完全公开 + 物理随机的游戏**(Azul):不注册 tracker,sim 入口直接
     跳过 `randomize_unseen`(没有 viz=0 槽位),物理随机靠 descent 里
     `do_action_fast(state, action, sim_rng)` 处理(§9.2)

   不同 sim 采不同世界 → 不同观察者可见后继 → 不同 hash → 自然分叉
   (§9.7)
6. **Step counter 防环**：`step_count_` 每次 `do_action_fast` 递增，
   纳入 public hash → DAG 结构性 acyclic（§9.5）
7. **Encoder 对齐 hash scope**：encoder 只读 acting player 视角下 `viz=1`
   的槽位，和 hash 看的是同一组事实 → DAG 节点 ⇄ network features 1:1

去掉任一个都会破坏其他。

### 9.2 三类游戏，统一算法

按游戏内禀结构(schema 长什么样 + 有没有隐藏槽位要 belief sample)分:

| 游戏类型 | 例子 | schema | sim 入口要不要 belief sample |
|---|---|---|---|
| 完全公开 + 确定 | TicTacToe、Quoridor | 全 `all_public` | 不需要 |
| 完全公开 + 物理随机 | Azul | 全 `all_public`(袋子/box-lid 都是公开多重集计数) | 不需要——物理随机在 `do_action_fast` 里靠 `sim_rng` 即时抽,state 上没有藏起来的字段 |
| 信息不对称 | Love Letter、Splendor、Coup | 有 owner-only 字段(手牌、盲压等),不同 perspective 看到的 viz 不一样 | 需要——sim 入口 `sim_tracker.randomize_unseen` 把 viz=0 槽位填具体世界 |

**Tracker 是否注册跟上面这张表无关**——tracker 的职责是"缓存 GT 不
care、但 NN 想吃的公开衍生特征"以及(对有隐藏槽位的游戏)承载
`randomize_unseen`。一个完全公开 + 确定的游戏如果作者想喂 NN 一些手算
公开统计量,照样可以注册 tracker;反过来,信息不对称游戏如果作者懒得算
weighted prior,`randomize_unseen` 内部 uniform 也能跑(AI 弱一档,见
§8.3)。

同一套 MCTS 代码——区别只在 schema 决定 walker 物化 MaskedState 时哪些
槽位被置 placeholder,以及有没有调用 `randomize_unseen`。

### 9.3 数据结构

```cpp
struct Edge {
  ActionId action;
  float prior;               // 网络首次 expand 时给的先验
  int child;                 // 指向 nodes[]（-1 = 没走过）
  int visit_count;
  float value_sum;
};

struct Node {
  int to_play;               // acting player
  bool expanded;
  int visit_count;
  float value_sum;
  std::vector<Edge> edges;   // 一条 edge = 一个合法动作
};

std::vector<Node> nodes;                              // 所有节点线性存储
std::unordered_map<StateHash64, int> node_index;      // per-search DAG 查找表
                                                      // (search_root 每次重建)
```

**Edge 里没有 chance_children**——DAG 查找统一走全局表。**每个节点一
个 to_play**——UCB 选边是从 to_play 的 info set 看。**`nodes[0]` 是根**。

### 9.4 `search_root` 完整流程

```
输入: root_state, rules, evaluator, value_model, NetMctsConfig, search_seed

【初始化】(整个 search 一次)
1. nodes.push_back(Node{ to_play = root_state.current_player(),
                          expanded = false, visit_count = 0,
                          value_sum = 0, edges = [] })
   node_index[root_state.state_hash_for_perspective(
       root_state.current_player())] = 0
2. expand_node(0, root_state)        // 见下方
3. 对 root edges 加 Dirichlet 噪声（训练探索）

【每次 sim】(派生独立 sim_rng,§3)
1. sim_state  = root_state.clone()
   sim_tracker = session.belief_tracker
                   ? session.belief_tracker.clone() : nullptr
   sim_rng    = mt19937_64(derive_subseed(search_seed, sim_index))
2. if (sim_tracker) sim_tracker.randomize_unseen(*sim_state,
                                                 root.current_player(),
                                                 sim_rng)

3. path = []                          // 每条路径上的 (node_idx, edge_idx)
   cur = 0
   incoming_edge_visits = nodes[0].visit_count   // 根节点入边记 sentinel
   leaf_values = nullptr

4. while True:
     if sim_state.is_terminal():
       leaf_values = value_model.terminal_values(sim_state)
       break

     if not nodes[cur].expanded:
       leaf_values = expand_node(cur, sim_state)
       break

     // UCT2（§9.6）
     sqrt_parent = sqrt(max(1, incoming_edge_visits))
     edge_idx = argmax over nodes[cur].edges:
                  q(edge) + c_puct * edge.prior
                            * sqrt_parent / (1 + edge.visit_count)
     edge = &nodes[cur].edges[edge_idx]

     rules.do_action_fast(sim_state, edge.action, sim_rng)
     // sim 内 sim_tracker 也消费同一公开事件:
     if (sim_tracker) sim_tracker.observe_public_event(...)

     next_hash = sim_state.state_hash_for_perspective(
                     sim_state.current_player())
     if next_hash in node_index:
       next_idx = node_index[next_hash]
     else:
       nodes.push_back(Node{ sim_state.current_player(),
                              expanded=false, 0, 0, [] })
       next_idx = nodes.size() - 1
       node_index[next_hash] = next_idx
     edge.child = next_idx              // 回填,下次访问免再 hash 查表

     path.push((cur, edge_idx))
     incoming_edge_visits = edge.visit_count
     cur = next_idx

【backup】(沿 path_nodes 从 leaf 反向走到 root)
5. for i in reversed(range(len(path_nodes))):
     node_idx = path_nodes[i]
     n = &nodes[node_idx]
     // leaf value 进 backup 前先 clip(§13 `value_clip`)——避免单次极端
     // 估值过度污染 Q
     v_clipped = clip_value(leaf_values[n.to_play], cfg.value_clip)
     n.visit_count += 1
     n.value_sum   += v_clipped
     if i > 0:
       parent_idx  = path_nodes[i - 1]
       edge_idx    = path_edges[i - 1]      // 走入当前节点的那条入边
       parent_edge = &nodes[parent_idx].edges[edge_idx]
       // 入边用**父节点** to_play 视角累加(那条边代表父节点的决策),
       // 同样过 clip
       pv_clipped = clip_value(leaf_values[nodes[parent_idx].to_play],
                                cfg.value_clip)
       parent_edge.visit_count += 1
       parent_edge.value_sum   += pv_clipped

【选 root 最优】
6. argmax over nodes[0].edges.visit_count
   // 多条 edge 共享 max visit 时,用独立 rng(salt 由 `derive_subseed`
   // 派生,跟 sim_rng 不交叉)做 uniform tie-break——避免依赖 vector
   // 顺序导致的隐性偏置

────────
expand_node(idx, state):
  // Terminal fast path:`legal_actions(state).empty()` 时不调
  // evaluator,直接用 `value_model.terminal_values` 收尾。saves
  // 网络评估开销
  if rules.legal_actions(state).empty():
    nodes[idx].expanded = true
    nodes[idx].edges.clear()
    return value_model.terminal_values(state)

  masked = make_masked_state(state, state.schema_ref(),
                              state.current_player())
  evaluator.evaluate(*masked, state.current_player(), legal)
       → priors, values
  for action in legal:
    nodes[idx].edges.push_back(Edge{ action, prior=priors[action],
                                      child=-1, visit_count=0,
                                      value_sum=0 })
  nodes[idx].expanded = true
  return values   // 长度 = num_players,already 旋回绝对座位序
```

要点:
- **Path 显式记录**——backup 才能更新沿途 edge / node 统计。
- **Edge 和 node 双计数**——UCT2 选边读 `edge.visit_count`(§9.6,这条
  入边在这次 sim 路径上被走过几次),`node.visit_count` 是 DAG 节点被
  访问总次数(根节点没有入边,UCT2 把它当 sentinel,等于截至当前的
  sim 数)。
- **`edge.child` 回填**——重复 sim 走到同一条边时直接读,不再走 hash
  表查询。
- **sim_tracker clone**——sim 内可以 `observe_public_event` 维护
  这条 sim 自己的 belief 演进,sim 结束 sim_tracker 连同 sim_state
  一起丢弃,session 持有的 tracker 不被污染。
- **sim_rng 一份**——belief 采样和 descent 内 `do_action_fast` 共用
  同一条 rng 流,跟 §3 的"sim_rng 一份"对齐。

**可选搜索变种:`cover_root_edges`**(§13 配置项,`net_mcts.cpp:471` 命
中):分析路径(pipeline `action_values` 输出)用,启用后头几个 sim 强
制每条未访问 root edge 至少访问一次——绕过 PUCT 选边,确保所有 root
action 都拿到 Q/visit 统计。selfplay / web 默认 false,只在 analysis 时
切到 true,跟主算法解耦。

**Hash 规则**：

```
hash(node) = digest(step_count, MaskedState(acting player))
```

- descent 每步 acting player 在轮转,MaskedState 跟着重新物化
- `step_count` 进 hash(保证 DAG acyclic)
- **tracker 不进 hash**(tracker 是网络特征,同一份 MaskedState 必 hash
  相等,不允许因 tracker 缓存内容不同而分裂节点)
- 实现方向:walker 按 schema 序遍历 MaskedState,每个 viz=1 槽位调
  `hash_field_slot` 合 digest,viz=0 槽位合固定 sentinel——游戏只填每槽
  typed value emission,不写整体 hash

### 9.5 DAG 节点共享 + step counter 防环

#### 信息集汇聚

不同 sim 采样不同的世界 descent,只要走到的节点在**当前 acting player
视角下**区分不开(MaskedState + step_count 一致),就合并到同一个 DAG
节点——统计天然在信息集层面聚合。

#### Transposition

不同动作序列到达同一 (MaskedState, step_count) → 同一 hash → 同一节点。

#### Step counter 为什么防环

DAG 的担心：若有 cycle（action 序列走回出发点），MCTS 陷入死循环。

`IGameState::step_count_` 由框架管理：`IGameRules::do_action_fast` /
`do_action_deterministic` 是 non-virtual wrapper，自动 `step_count_ += 1`
后再调用 `*_impl`；`undo_action` 是同样的 wrapper，先调 `undo_action_impl`
再 `step_count_ -= 1`。step_count 纳入 public hash，任何两个 state 只要
step_count 不同，hash 必不同。do_action 永远是 step++ → DAG 里 parent →
child 永远 step 增加 → 不可能回到同 hash → **结构性 acyclic**。

游戏作者唯一要求：

- `reset_with_seed` 第一行调 `reset_step_count_base()` 把 step_count_ 归零
- `step_count_` 字段 protected + IGameRules friend，作者既看不到也无法
  忘记 / 双 bump（`tests/framework/test_step_count_strict_increase.py`
  作为结构性回归守这条）

#### 多父节点与 backup 正确性

一个 DAG 节点可能有多个入边。单次 sim 的 backup 只走**这次 sim 的
path**。

- `node.visit_count` = 所有路径的访问总和（"这个 info set 被考察过多少
  次"）
- 每条入边 `edge.visit_count` 只记自己被选中的次数
- `sum(incoming.visit_count) ≈ node.visit_count`（略少于，因初次到达未
  经边）
- `node.Q = node.value_sum / node.visit_count` = info set 的期望价值
- `edge.Q = edge.value_sum / edge.visit_count` = 经过这条边后的期望价值

UCB 在父节点选边时用的是 **edge 层** 的 Q。node 的 Q 主要给 backup 用。

### 9.6 UCT2

#### 直接套 UCT 的问题

AlphaZero PUCT：

$$\text{score}(e) = Q(e) + c \cdot p(e) \cdot \frac{\sqrt{N(\text{parent})}}{1 + N(e)}$$

在 tree 里 $N(\text{parent}) \approx \sum N(\text{child edges})$，公式自洽。

DAG 里一个节点 `node.visit_count` 汇总多条入边访问。设节点 X 有两条入边
$e_a, e_b$ 各 50 次：当 sim 走 $e_a$ 到 X 选出边时，直接用
`sqrt(X.visit_count = 100)` 把 $e_b$ 的访问也算进 exploration——u 项被
高估 $\sqrt{2}$ 倍。

#### UCT2 的修正

Childs, Brodeur, Kocsis (2008) "Transpositions and Move Groups in MCTS"
提出 UCT2：

$$\text{score}(e) = Q(e) + c \cdot p(e) \cdot \frac{\sqrt{N(\text{incoming edge})}}{1 + N(e)}$$

$\sqrt{}$ 分子底换成"这次 sim 刚经过的那条入边的 visit_count"，反映"我
这条路径上到达当前节点多少次"，而不是"所有路径算上"。代码里用一个局部
变量 `incoming_edge_visits` 在 descent 循环里追踪（§9.4）。

CPU 成本 0 额外（本来就要算 sqrt），代码 ~10 行。Childs 论文实测 vs
UCT1 棋力差 5-15%，越容易 transpose 的游戏（Splendor、Quoridor、Azul）
差距越大。

Root 没有入边——用 `nodes[0].visit_count` 替代（等于已完成 sim 数），
等同于 AlphaZero 的 UCT1 root，和后续层 UCT2 无缝衔接。

### 9.7 物理随机 = sampled world

#### 观察者不可见的随机（opp 抽牌）

```
sim_1 deck = [C1, C2, C3, ...]
sim_2 deck = [C4, C5, C6, ...]
```

descent 里 opp 抽牌 → `do_action_fast` 弹 `d.deck.top()`。不同 sim 拿到
不同的牌。回到 observer 决策节点时，key 里**不含 opp hand**（observer
private 层面），两个 sim 合并到同节点——visit 在 observer 层正确汇总。

#### 观察者可见的随机（Splendor tier 翻新卡）

observer 买了 tier 1 slot 0 → deck 翻出新卡放到 slot 0（**公开**）。
sim_1 翻出 card#12、sim_2 翻出 card#37 → 公开槽位内容不同 → MaskedState
不同 → DAG hash 不同 → 自然分叉。符合"observer 确实面对两个不同公开局
面"的事实。

#### 为什么不需要显式 chance node

传统 chance-node MCTS 在每个随机事件处建 chance 节点按概率展开。但我们：

- Root 采样把"整局所有未来随机"固定成一条具体世界路径
- observer 能分辨的随机自动通过 hash 差异分叉
- observer 不能分辨的随机自动通过 hash 合并汇聚
- N 次 sim = N 条可能世界轨迹，Q 统计近似真实期望

DAG 自然承担了 chance node 的工作。

### 9.8 Root state 的来源：永远是 AI session 的 state

MCTS / encoder / legal_actions 在 selfplay / web / API 三条路径下读的都
是同一种 root state——当前 acting player 那个 AI session 维护的 state。
selfplay 也不例外:GT runner 持有的 truth state 只用来推进游戏 + 派 snapshot,
**从不传给 MCTS / encoder**。

**结构性落地**:`selfplay_runner` / `arena_runner` / `heuristic_runner`
对每个 seat 持有一份独立的 `IGameState`(`per_seat_states[p]`),每步
truth 在 `gt_rng` 上跑完 `do_action_fast` 之后,runner 把每个 seat
的 session state 通过公开事件协议推进:

```
per_perspective_extractor(truth_before, action, truth_after, p)
  → PublicEventTrace evt_p   // {events, public_snapshot}
seat[p].begin_step_for_session_observe()  // 框架包办 step_count_++
public_state_applier(seat[p], evt_p.public_snapshot)
  // ↑ 用 truth-side 抽出来的公开字段快照覆盖回 session 的 public 字段
per_perspective_trackers[p]->observe_public_event(actor, action, evt_p.events)
  // ↑ 事件只折进 tracker,不应用到 session state
// 注:session 不再调 randomize_unseen。viz=0 槽位是上一次 observe 留下
// 的旧字节,决策侧物理上读不到(hash/encoder/sim 三者都把 viz=0 屏蔽掉)
```

session state **不再调 `do_action_fast`**——信息不全跑没有意义,session
只接 snapshot + 走 tracker。两类字段经此流程后:
- **public 字段** 每步由 `public_state_applier` 从公开事件流的 snapshot
  覆盖,truth 端 walker 在每个 perspective 各自 snapshot 里只写
  `viz[..., p]=1` 的 slot,observer 端整张替换
- **viz=0 hidden 槽位** session 不维护——determinization 只在 MCTS sim
  入口对克隆出来的 sim_tracker 调一次 `randomize_unseen`,session 自身
  的 viz=0 字节是 unread bytes

调用方决定要不要走 per-seat 化:`per_seat_states` 入参为空时 runner 退
回 truth 路径(未 in-scope 的游戏走这条兼容档)。In-scope 的
TTT/Quoridor/Azul/Splendor/LoveLetter 都走 per-seat 路径。

**测试守护**:

- `test_api_mcts_policy_invariance` 守护:相同观察序列下 selfplay 路径
  和 web/API 路径的 root visit 分布一致——若 MCTS 任何一处偷读 truth 的
  hidden 字段,分布就会发散。
- `test_selfplay_no_truth_in_ai_path` 守护:selfplay 在五款 in-scope 游戏
  上(a) 同 seed 跑两次必须完全可重现 AI 决策,(b) 同一份 observation
  trace 喂给两个不同 seed 的 API session,各自 public state 必须收敛——
  session 公开视图不允许依赖 session 自身的 hidden RNG。
- `test_encoder_features_have_no_placeholder_sentinels` /
  `test_encoder_invariant_across_selfplay_plies`(均在
  `tests/framework/test_encoder_only_reads_masked_state.py`)守护:encoder
  输出永远不出现 `kPlaceholder*` sentinel——任何 game encoder 漏处理
  hidden 槽位都会让 INT32_MIN / INT8_MIN 漏到 feature 里被这两条测试抓住;
  并且同一观察序列下两个不同 session_rng 的 encoding 必须 byte-equal。
- session viz=0 槽位不再每 ply 重采:`apply_observation` / 各 runner
  `advance_per_seat_states` 末尾都不再调 `randomize_unseen`,session
  hidden 是上一次 observe 留下的 raw bytes。任何决策路径(hash / encoder
  / sim)都用结构性手段把它从读侧屏蔽掉,具体见 `kHiddenHashSentinel`
  (hash)、`MaskedState` placeholder(encoder)、`sim_tracker->
  randomize_unseen`(sim 入口);三者之外没有 consumer。`test_public_hash_excludes_internal_rng`(60-seed × 4 hidden-info 游
  戏)是这条结构性保证的回归 sweep。

---

## 10. Wire 协议

### 10.1 Snapshot 格式

GT 每 ply 给每个 perspective 发一份消息,载体是
`PublicEventTrace`(`engine/core/game_interfaces.h:47`),两个组成部分:

```json
{
  "actor": <int>,
  "action_id": <int>,
  "events":  [["<kind>", {<payload>}], ...],
  "public_snapshot": { "<field_name>": <value>, ... }
}
```

- **`events`**:这次 transition 里对该 perspective 公开可观察到的事实序列,
  **只用来喂 tracker**(observer 不再跑 `do_action_fast`,事件没有 in-rules
  的时机可以挂)。list 内顺序就是 producer 发出的顺序。例:Splendor
  `deck_flip`(tableau 翻新)/ `self_reserve_deck`(perspective 自己 reserve
  时只发给自己) / `opp_buy_reserved_reveal`(对手买暗压牌时公开那张
  reserved 的真实 card_id) / Azul `factory_refill` / Coup `card_revealed` /
  `exchange_complete` / `self_influence_redraw` / `self_exchange_draw`
- **`public_snapshot`**:整张 post-action 公开字段的 `AnyMap field_name ->
  value`(全公开字段直接进 dict,无 visibility_mask 字段、无 bit-packing)

`PublicEvent = std::pair<std::string kind, AnyMap payload>`,kind 字符串和
payload key 都由游戏自定。**没有 phase 概念**——pre/post 之分只在 observer
跑 `do_action_fast` 时有意义,observer 不跑后该区分就没意义了。一份 trace
由 `PublicEventExtractor` 在 selfplay 跑完 truth 一步后做 (state_before,
action, state_after, perspective) diff 生成(`engine/core/game_registry.h`)。

`public_snapshot` 是 `AnyMap field_name -> value`(全公开字段直接进 dict,
无 visibility_mask 字段、无 bit-packing)。observer 端 `apply_observation`
按 schema 序对每个 field 调 `write_field_slot` / SnapshotApplier,**整张
替换** session 公开字段——不 OR-merge、不重算 viz。

> **为什么 state mutation 不再走 events**:observer 不跑 `do_action_fast`
> ——信息不全跑没有意义,session 只能从自己 sample 出的 hidden 上算逻辑,
> 算出的值随后会被 snapshot 覆盖,纯属浪费。所以事件没有"动作前 vs 动作
> 后"的时序需求,合并成单一 list 就够了。public 字段每步从
> `public_snapshot` 整张覆盖,viz=0 hidden 槽位 session 不维护(DEC-003);
> session state 的最终一致性靠"public 整张覆盖 + 决策侧物理上读不到
> viz=0"这两条担保。

实现可走两条路径(wire 一致):
- **walker 化路径**:`viz::serialize_public` / `viz::apply_public`
  (`engine/core/snapshot_io.h`)——schema-walker 自动按字段序遍历每个
  `is_all_public` 字段,调 game-side `read_field_slot` /
  `write_field_slot`。Splendor 走这条
- **message-driven 路径**:`SnapshotIO + emit_snapshot / apply_snapshot`
  ——每个公开字段游戏端注册一对 `SnapshotEmitter / SnapshotApplier`
  函数。LL / Coup / Azul 走这条

两条路径产出**同一个 `AnyMap public_snapshot` shape**,wire 形态平权;选
哪条只是 game-side 的实现风格,框架不偏不倚。

`initial_observation` 是另一种 opening-fact dict(每个 perspective 在
session 启动时收到一次,内容是该 perspective 看得见的开局事实——例如
LL 自己的开局手牌、Coup 自己的两张 influence cid),由
`initial_observation_extractor` / `apply_initial_observation` 负责
读写。**它和每步 `public_snapshot` 不是同一个 type**——前者只在 create
时存在、可含 perspective-private 信息;后者每 ply 都发、纯公开。

### 10.2 协议字段限制

`CreateSessionRequest` 不接受 `seed` / `simulations` / `temperature` /
`tail_solve` 入参——AI 强度 = web expert 难度，server-controlled，不可
调（测试 / 调试场景显式传 seed 例外）。`seed` 内部
`secrets.randbits(64)` 生成。

`decide()` **不改 session 状态**——只跑 search 返 action_id。GT 收到
action_id 后在自己端 commit + 抽真随机 + 算 snapshot + 算 events，
再走 `observe` 把消息回灌给 AI session。AI 自己的 action 也走 observe
端点（对所有玩家对称）。

---

## 11. 不变量速查

§11 只列**框架结构性 / CI 能守护**的不变量。viz 语义本身是否写对(over- vs
under-permission 的代价非对称)是作者职责,框架原则上无法判定一个槽位"该"
viz=1 还是 0,见 `docs/FRAMEWORK_DESIGN_RATIONALE.md` §3.3。

| # | 不变量 | 守护机制 |
|---|---|---|
| I1 | 规则引擎是 viz 的唯一 writer | `do_action_fast(State&, Action)` 同时维护 state 和 state.viz_ |
| I2 | viz 是 (state, action, next_state) 的纯函数 | rules 不接受 history buffer 入参 |
| I3 | 内部 RNG 永不进 hash / wire | RNG 不在 state 上;只有 gt_rng / session_rng / sim_rng 三类,互不派生,各自由调用方独立 seed;`test_public_hash_excludes_internal_rng` 守护 |
| I4 | viz 不保证单调 | rules 在槽位换内容时主动 `reset_to_base`;AI 端 observe 替换 not merge |
| I5 | viz 在 GT 端算一次,AI sim 内由 rules 顺手维护 | AI observe 阶段不重算 viz |
| I6 | perspective 私人事实表达成 `viz[..., perspective]=1` 的 state 槽位 | rules 在历史 ply `reveal_slot_to`,不存在 state 之外的私人事实 |
| I7 | tracker 公开 + 可选 + 不进 hash / 协议 | tracker 不携带 perspective 私人信息(其内容是公开事件流的派生聚合);**不进 hash**——同 (MaskedState, perspective) 必 hash 相等,即使 tracker 内部 cache 不同;**不进协议**——tracker 不上线。决策路径**会读** tracker(encoder 取派生特征、`randomize_unseen` 在 sim 入口采样),这正是 tracker 存在的目的 |
| I8 | tracker 不存 belief prior | belief sampling 在 `randomize_unseen` 现场算,不缓存 prior 字段 |
| I9 | sim 入口 clone sim-local tracker | session 持有的 tracker 不被 sim 写入;sim_tracker 在 sim 结束时和 state 一起丢弃 |
| I10 | belief sample 范围 = `viz[..., perspective]=0` 全部槽位 | 单点判断 |
| I11 | encoder 永远读不到 belief 噪声 | walker 在 `make_masked_state` 时把 viz=0 槽位置 placeholder;接口锁 `const MaskedState&`,encoder 结构上读不到 viz=0 真值 |
| I12 | encoder 输入面 = MaskedState + tracker | `IFeatureEncoder` / `IPolicyValueEvaluator` 接口锁 `const MaskedState&`,viz=0 槽位读到 placeholder;`test_encoder_respects_hash_scope` 守护 |
| I13 | 节点 hash perspective = `state.current_player()` | framework 实现,游戏不写 hash |
| I14 | step_count 进 hash 保证 DAG 无环 | framework 自动加 |
| I15 | 协议传输公开字段完整(observer apply 后 hash 等价 truth) | `test_public_snapshot_round_trip` 守护——truth → make_masked_state → wire snapshot → observer apply 后 `state_hash_for_perspective(own)` 必须 byte-equal |
| I16 | actor 必传 | 协议 schema required field |
| I17 | decide 不 commit | session.state / tracker 在 decide 中只读 |
| I18 | MCTS root state = AI session state(不是 truth) | selfplay / web / API 三路对称;`test_api_mcts_policy_invariance` 守护 |
| I19 | session viz=0 槽位是 unread bytes,任何决策路径不读它 | 结构性:hash 用 `kHiddenHashSentinel`、encoder 用 `MaskedState` placeholder、sim 入口走 `sim_tracker->randomize_unseen` 在 clone 上采样;`test_public_hash_excludes_internal_rng`(60-seed × 4 game)守护 |
| I20 | belief 产出 MaskedState 公开部分 byte-equal | 只取决于 tracker 观察历史,与输入 state hidden 内容、调用者 RNG 无关;`test_api_belief_matches_selfplay` + `test_public_snapshot_round_trip` 守护 |
| I21 | tracker 接口物理上拿不到 `IGameState*` | `init` 收 `AnyMap`,`observe_public_event` 收事件流;结构上挡掉"tracker 偷看 truth" |

---

## 12. 游戏开发者职责清单

### 12.1 完全公开 + 确定（TicTacToe、Quoridor）

- `IGameState`：`current_player / is_terminal / winner / num_players /
  reset_with_seed`
- `hash_field_slot`：每槽 typed value emission（walker 在 viz=1 槽位调用）
- `mask_field_slot`：**空实现**（schema 全 `all_public`，walker 永远不会
  把任何槽位置 placeholder）
- `read_field_slot` / `write_field_slot`：schema 驱动 snapshot 序列化时
  实现
- `schema_ref()`：返回 game-static schema
- `reset_with_seed` 第一行调 `reset_step_count_base()` 把 step_count_ 归零
- `IGameRules::do_action_fast_impl`（**不支持 undo**——MCTS / selfplay /
  arena / web 都丢弃用完的 state，不要 push undo_stack / 拍快照）。
  step_count_ 由框架 wrapper 自动 +1，作者既看不到也无法忘记
- 想接 tail solver 才额外重写 `do_action_deterministic_impl` +
  `undo_action_impl`（前者 freeze 隐藏抽牌，后者按 token 还原；step_count_
  由框架 wrapper 自动配对增减）

可选(按需):`belief_tracker`——没有 viz=0 槽位不需要 `randomize_unseen`,
但作者想缓存 GT 不 care 的公开衍生特征喂 NN 也可以注册(§7)。

### 12.2 公开物理随机（Azul）

§12.1 + 以下：

- 袋子 / box-lid 都是公开多重集计数（`bag_counts[color]` /
  `box_lid_counts[color]`），schema 仍全 `all_public`——没有藏起来的
  抽牌顺序
- `do_action_fast` 接 `mt19937_64& rng`，发牌时按 `bag_counts` 即时按色
  抽取（袋子空了从 box_lid 整体回填）。**rng 不在 state 上**——sim 入
  口拿一份独立 `sim_rng`(§3)，selfplay/arena GT 端拿 `gt_rng`，调用方
  各自管理

可选(按需):`belief_tracker`——没有隐藏字段不需要 `randomize_unseen`,
NN 衍生特征想缓存就注册,不想就不注册。

### 12.3 信息不对称（Love Letter、Splendor、Coup）

§12.1 + §12.2 的 rng 接入方式 + 以下：

- schema 里 owner-only 字段（手牌、盲压等）base viz = `owner_only_first_axis`，
  walker 物化 MaskedState 时按 perspective 分流——viz=1 槽位走
  `hash_field_slot` 合 digest，viz=0 槽位 walker 调 `mask_field_slot` 写
  placeholder
- `mask_field_slot(name, idx)`：写 placeholder 到对应槽位
- 通常注册 `IBeliefTracker` 给 sim 入口填 viz=0 槽位。**tracker 永远是
  可选的**——实现质量决定 AI 强度,不决定能不能跑:
  - 不注册 / 注册个土实现(unseen 槽位 uniform 乱抽,完全不读累积观察)
    → AI 强度退化下限,搜索照跑
  - 想要强 AI 才认真维护:
    - `init(const AnyMap& initial_obs)`:根据初始观察构建 belief
      (**接口签名里没有 `IGameState*`**,tracker 物理上拿不到 truth)
    - `observe_public_event(...)`:从事件流更新 belief
    - `randomize_unseen(state, observer, rng)`:根据 belief 给 state
      填充未见字段。产出世界中所有公开槽位的内容只取决于 tracker 的观
      察历史,不依赖输入 state 的 hidden 内容
    - (可选)tracker 内部顺便缓存 GT 不 care 的公开衍生特征(claim 历
      史、多重集统计等)给 encoder 吃
- Feature encoder 读 MaskedState——viz=1 槽位读到真值,viz=0 槽位读到
  placeholder。结构上读不到 opp 真实 hidden 字段(接口锁
  `const MaskedState&`,§5.3)

**Round-trip 不变量**（`tests/framework/test_public_snapshot_round_trip.py`
守护）：truth → make_masked_state → wire snapshot → observer session →
apply snapshot 后，observer 的 `state_hash_for_perspective(own)` 必须和
truth bit-equal。新增公开字段忘记更新 schema / `read_field_slot` /
`write_field_slot` 都会立刻被这个测试抓住。

---

## 13. 配置项速查（`NetMctsConfig`）

| 字段 | 语义 | 典型值 |
|---|---|---|
| `simulations` | 每次 `search_root` 的 sim 数 | 200 / 1000+ |
| `c_puct` | UCB exploration 系数 | 1.4 |
| `max_depth` | 每 sim 最大 descent 深度 | 128 |
| `value_clip` | leaf value 裁剪 | 1.0 |
| `root_dirichlet_alpha` / `epsilon` | 根探索噪声 | 训练 0.3 / 0.25，eval 0 |
| `root_belief_tracker` | 非空 → 启用 root 采样 | 由 game bundle 决定 |
| `tail_solve_enabled` | 启用 alpha-beta 残局求解 | false / true |
| `cover_root_edges` | 分析路径用：每条 root edge 至少访问一次 | false（selfplay） / true（pipeline analysis） |

---

## 14. 参考文献

1. **Cowling, Powley, Whitehouse (2012)** — "Information Set Monte Carlo
   Tree Search"。ISMCTS 原始论文，SO-ISMCTS / MO-ISMCTS。我们的"单树
   per-acting-player keying"是两者之间的工程选择
2. **Childs, Brodeur, Kocsis (2008)** — "Transpositions and Move Groups
   in MCTS"。UCT1 / UCT2 / UCT3 对比研究。我们选 UCT2
3. **Silver et al. (2018)** — "AlphaZero"。PUCT + 网络先验
4. **Paolini et al. (2024)** — "Learning to Play 7 Wonders Duel Without
   Human Supervision" (ZeusAI)。afterstate cap；我们靠 DAG 共享代替
5. **Silver & Veness (2010)** — "Monte-Carlo Planning in Large POMDPs"
   (POMCP)。particle filter + UCT，和我们的 root 采样精神相近

