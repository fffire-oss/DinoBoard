# 框架设计动机：为什么字段级 viz

## 0. 一句话立项动机

**DinoBoard 的存在不是为了重写 MCTS，是把"可见性正确性"从开发者契约
搬到框架结构契约**。

OpenSpiel 的契约写在文档里：跑 ISMCTS + neural eval 至少要写的三件
（info-set key + network 输入 tensor + `ResampleFromInfostate` belief
采样器）**都必须只反映玩家 p 能合法看到的信息**——作者要读懂、记住、
保证语义对齐、靠 round-trip 测试抓错位。心智负担**在作者头上**。

OpenSpiel 现代的 Observer API（`open_spiel/observer.h`）已经把"前两件"
合并成一个 observer：作者实现一次 `WriteTensor` + `StringFrom`，框架
按 `IIGObservationType{public_info, private_info, perfect_recall}` 的 flag
组合派生不同观察。这条路径下"info-set string vs network tensor"两份
对齐问题被框架解决了一半。剩下的契约表面积是 observer 内部的逻辑——
"哪个字段对玩家 p 当前可见"仍然散在作者代码里，框架不持有这条事实。

DinoBoard 把同一件事改成结构契约：

```
schema 声明字段 + base viz
       ↓
rules 在 do_action_fast 里维护 viz（reveal_slot / reset_to_base）
       ↓
walker make_masked_state(state, schema, perspective) 派 MaskedState
       ↓
三家消费者（snapshot 序列化 / hash digest / encoder tensor）
签名锁 const MaskedState& —— viz=0 槽位结构性读到 placeholder
```

心智负担从"observer 内部判断字段可见性"压到"1 格 viz 写对"。作者
写错 viz 一格才能泄漏；写错"哪个字段进序列化"在编译期就过不了。差
异不是"OpenSpiel 没接口、DinoBoard 有结构"——OpenSpiel 的 Observer
API 已经是结构化接口——而是**可见性事实存放的位置**：observer 内的
作者代码 vs state 上的 viz tensor。

**这条立项动机在 LLM 辅助开发（vibe coding）下意义放大**：vibe coding
常见失败模式是"在自由度高的地方 silently drift"——多份序列化都让 LLM
写，LLM 顺手写的 round-trip 测试也来自同一个上下文，错位同向偏移、
测试抓不到。schema 把字段排列死、emission helper 锁住 typed read/write、
`const MaskedState&` 锁死接口签名——drift 的范围被压窄到 viz 那一格。
CI 三件套（`test_public_snapshot_round_trip` /
`test_encoder_respects_hash_scope` / `test_public_hash_excludes_internal_rng`）
的角色因此从"抓人类作者笔误"扩展为"抓 vibe coding 同向偏移"。

这个 trade 有代价：接受范围更窄（FOSG / 同时移动 / 长历史 / 大组合
动作不在 cover 范围内），失去 OpenSpiel 那种"作者怎么写都行"的灵活
度。ISMCTS / DAG / root determinization / UCT2 / PUCT 等算法层都是
公开论文里的成熟组合（Cowling 2012、Childs 2008、AlphaZero），不是
本框架的发明——本文档不论证算法层创新，只论证可见性正确性这一条结
构契约值不值这个设计赌注。

具体契约（不变量、数据流、谁做什么）+ MCTS 算法在
[`../ALGORITHM_OVERVIEW.md`](../ALGORITHM_OVERVIEW.md)。下文回答三个
问题：**为什么**这么选（§1-2）、**牺牲了什么**（§3）、**写错时框架
怎么兜住**（§3.3）。

---

## 1. OpenSpiel 的做法

OpenSpiel 有两条路径——legacy 四方法 API 和现代 Observer API（2020
年前后引入），实际接 ISMCTS + neural eval 时常见的是 Observer 路径，
但 legacy 方法仍然存在、不少现有游戏混用。

**Legacy 四方法 API**（`open_spiel/spiel.h`）：

```cpp
class State {
    virtual std::string InformationStateString(Player p) const;
    virtual void InformationStateTensor(Player p, absl::Span<float>) const;
    virtual std::string ObservationString(Player p) const;
    virtual void ObservationTensor(Player p, absl::Span<float>) const;
    virtual std::unique_ptr<State> ResampleFromInfostate(
        int player_id, std::function<double()> rng) const;
};
```

四个方法默认抛 `SpielFatalError`，`ResampleFromInfostate` 默认完美信
息游戏返回 clone、否则抛 error——隐藏信息游戏作者必须 override。每
份序列化各自决定"哪些字段对玩家 P 可见、按什么 layout 写出"，多份之
间靠作者保证语义对齐。

**现代 Observer API**（`open_spiel/observer.h`）：

```cpp
class Observer {
  virtual void WriteTensor(const State& state, int player,
                           Allocator* allocator) const;
  virtual std::string StringFrom(const State& state, int player) const;
};

std::shared_ptr<Observer> obs = game.MakeObserver(
    IIGObservationType{
      .public_info = true,
      .perfect_recall = false,
      .private_info = PrivateInfoType::kSinglePlayer,
    },
    /*params=*/{});
```

作者实现一次 observer，框架按 `IIGObservationType` 的 flag 组合派
生不同观察类型；`WriteTensor` 拿到的 `Allocator*` 让作者按命名子张
量写（`alloc->Get("private_hand", {kHandSize})`），框架可以按 flag 决
定要不要 emit 某个子段。同一个 observer 同时供应 string（info-set key）
和 tensor（network 输入），两者从同一份代码派生，**对齐由实现共享而
非作者契约保证**。

**两条路径共有的事实**：

- 没有"字段对谁可见"的 per-slot 结构化元数据——框架不知道每个具体
  slot 对每个 player 当前是不是可见。Observer API 把可见性折叠成
  observation-type 级别（`public_info` / `private_info` / `perfect_recall`
  三档枚举），具体到字段的"现在 viz=1 还是 viz=0"仍然是 observer
  实现内部的代码逻辑。
- ISMCTS 至少要 info-set key（observer string 或 legacy
  `InformationStateString`）+ `ResampleFromInfostate`；neural eval 再加
  observer tensor。
- belief sample 整体由作者写 `ResampleFromInfostate`——固定自己已知
  的、重抽未知的。这个方法是 `State` 的成员，作者**有能力**读真值，
  契约要求只用 info-set 信息（contractual，非 structural）。
- 灵活度高，FOSG / 同时移动 / 不完美回忆 / 长历史依赖等场景都 cover
  得到；代价是每款游戏作者要自己想清楚 observer 语义并保证 string /
  tensor / resample 三者对一致的"可见性"概念。

## 2. 我们的做法

| 维度 | OpenSpiel（Observer API） | DinoBoard |
|---|---|---|
| 可见性元数据 | observation-type 级（`public_info` / `private_info` / `perfect_recall` flag），具体字段可见性散在 observer 实现里 | per-slot per-perspective viz tensor，结构化 |
| 谁知道"槽位 X 对玩家 Y 当前可见" | observer 内部代码逻辑，框架不持有 | viz tensor，框架可查 |
| Belief sample | 作者 `ResampleFromInfostate(State*)` 手写——方法是 State 成员，作者**有能力**读真值，契约要求只用 info-set 信息 | tracker 接口只接收公开事件流（物理拿不到 `IGameState*`），`randomize_unseen(state, rng)` 只填 `viz[..., perspective]=0` 槽位 |
| 协议序列化 | 同一 observer 派生 string / tensor（命名 allocator 子张量） | framework 派生 `(visibility_mask, values dict)` —— walker 按 schema 序遍历，viz=1 槽位经 `read_field_slot` 进 values，viz=0 写 `kPlaceholder` 哨兵 |
| Hash / encoder / sampler 一致性 | observer 提供 string（info-set key）+ tensor（network 输入），两者从同一份 observer 派生；ResampleFromInfostate 是另一条独立实现，需作者保证语义对齐 | `make_masked_state(state, schema, perspective)` 单一 walker 产出 MaskedState；三家消费者（snapshot 序列化 / hash digest / encoder tensor）共享同一个对象，对齐由数据流而非约定保证 |
| 第三方 GT 接入（隐藏信息 + neural ISMCTS） | 实现 Observer + `ResampleFromInfostate`；observer 内部按需判断字段对当前 perspective 可见 | schema declare + rules 维护 viz + per-slot typed emission（`hash_field_slot` / `read_field_slot` / `write_field_slot` / `mask_field_slot`）；framework 走 walker |

### 2.1 Worked example：假如要在 OpenSpiel 实现 Love Letter

OpenSpiel 现有仓库（截至 2026-05）**没有** Love Letter。这条对比是
"假如要在 OpenSpiel 里把 LL 接得能用 ISMCTS + neural eval 的话，最优
做法是什么"，对应 DinoBoard 实际写法。

具体场景：p 这一手用 Priest 看了 q 的手牌 = X。下一回合 p 出 Guard，
AI 应当稳定地猜 X。这条信息流"用得起来"必须同时满足三件：

1. **info-set key 包含这条 peek 信息**——MCTS 把"已知 q=X"和"未知 q"
   当成不同决策节点，否则两个节点合并、Guard 选 X 的访问没法集中。
2. **belief sample 里 q 的手牌确实是 X**——determinization 出来的世界
   要锚住这张牌，否则 sim 里 q 的牌还是从池里均匀重抽，Guard 猜 X 跟
   猜别的胜率一样。
3. **网络输入 tensor 里要有"已知 q=X"这个特征**——否则 PUCT 的 prior
   和 value 估计拿不到这条线索，要靠 visit 慢慢倒推。

#### OpenSpiel 上的最优做法

不是"observer 内部记 peek_record + resample 里手动 force"那种把可见
性散在三个地方的写法。最优做法是把可见性物化成 state 字段，让
observer 和 resample 都从同一个字段派生：

```cpp
class LoveLetterState : public State {
  std::array<Card, kNumPlayers> hands_;
  // hand_known_[owner][viewer] = "viewer 当前能否看到 owner 的手牌"
  std::array<std::array<bool, kNumPlayers>, kNumPlayers> hand_known_;
};

// rules 在 Priest action 里：
void DoApplyAction(Action action) override {
  // ...
  if (action.kind == kPriest) {
    hand_known_[target][viewer] = true;
  }
  // q 抽新牌后所有 viewer 失效（包括之前看过的 viewer）
  if (action.kind == kDraw && drawer == q) {
    for (int v = 0; v < kNumPlayers; ++v) hand_known_[q][v] = false;
  }
}

// observer 同时供应 string + tensor，从 hand_known_ 派生：
void WriteTensor(const State& s, int player, Allocator* alloc) const {
  auto opp_hand = alloc->Get("opp_hand", {kNumPlayers, kCardTypes});
  for (int opp = 0; opp < kNumPlayers; ++opp) {
    if (s.hand_known_[opp][player]) {
      opp_hand[opp][s.hands_[opp]] = 1.0;
    }
  }
}

// ResampleFromInfostate 也读 hand_known_，force 已知牌：
std::unique_ptr<State> ResampleFromInfostate(int player, ...) const {
  auto sampled = /* replay history */;
  for (int opp = 0; opp < kNumPlayers; ++opp) {
    if (hand_known_[opp][player]) {
      sampled->ForceOpponentHand(opp, hands_[opp]);
    }
  }
  return sampled;
}
```

Observer API 解决了 string / tensor 对齐（同一份 `WriteTensor`），
`hand_known_` 这个字段解决了 observer / resample 对齐（两边读同一个
bit）。Priest action 里翻 1、q 抽新牌时翻 0，rules 一处改、observer
和 resample 自动跟着。

#### DinoBoard 上的写法

```cpp
// (a) rules 在 Priest action 的 do_action_fast 里：
reveal_slot_to(state, &State::hand, /*owner=*/target, /*viewer=*/viewer);

// (b) q 抽新牌：
reset_to_base(state, &State::hand, /*owner=*/q);

// (c) encoder 不知道有 peek：
void encode_public(const MaskedState& m, vector<float>& out) {
  for (int seat = 0; seat < num_players; ++seat) {
    encode_card(m.hand[seat], out);   // m.hand[seat] viz=1 时是真值、
                                      // viz=0 时是 kPlaceholder
  }
}
// peek 让 viz 翻 1 → MaskedState 那一格从 placeholder 变成真牌 → 网
// 络输入端那一格从哨兵变成 one-hot。encoder 不查 viz、不加 if、不设
// 计 "opp_hand" 字段 layout。

// (d) tracker.randomize_unseen 看到 viz=1 直接跳过，只填 viz=0 的格子。
```

#### 客观对比

OpenSpiel 上"最优做法"已经把"必须三处对齐"压成"读同一个 bit"——
和 DinoBoard 的核心机制其实是同一个想法。两边的实际差异：

1. **谁强制这条模式**。OpenSpiel 的 `hand_known_` 字段是作者自己想到
   要加的——API 不要求、framework 不提供、其它游戏也不一定这样写。
   DinoBoard 把这条模式提到 IGameState 基类：viz_ 是基类成员、
   reveal_slot / reset_to_base 是框架 helper、walker 自动遍历 schema
   把 viz 派到 hash / snapshot / encoder 三家。作者写新游戏时不是
   "想到要这样写"，是"只能这样写"。
2. **CI 集中覆盖**。DinoBoard 因为 viz 在基类、所有游戏共享一套表
   达，能写一组 round-trip 测试一次跑遍所有游戏（`test_public_
   snapshot_round_trip` / `test_encoder_respects_hash_scope` /
   `test_public_hash_excludes_internal_rng`）。OpenSpiel 即使每款游戏
   都按上面"最优做法"写，每家的 visibility 字段命名 / 形状不同，集
   中 CI 不现实——要么每款游戏单独写 round-trip 测试、要么靠各家作
   者自觉。
3. **encoder 端代码量**。OpenSpiel 上面写法的 `WriteTensor` 仍然要写
   per-game 的 if + 字段命名 + layout 设计（"opp_hand" 还是
   "known_opponent_card"？要不要加 `did_peek` mask bit 区分"没看过"
   和"看过 0号"？）。DinoBoard 写 `encode_card(m.hand[seat], ...)` 时
   和"有没有 peek"完全无关——peek 是 walker 派 MaskedState 时的事，
   encoder 拿到的是已经派好的 placeholder 或真值。这条差异在动态揭
   示频繁的游戏（Priest peek、Coup exchange）里显著；在 Bridge 这种
   发完牌就基本固定的游戏里其实不太能体现。
4. **资源型可见性的处理**。Priest peek 后 q 抽新牌让 peek 失效——
   OpenSpiel 上面写法要在 q 的 draw transition 里写一段循环把
   `hand_known_[q][*]` 全清；DinoBoard 写一行 `reset_to_base(state,
   &State::hand, q)`。差异不大但累加起来：每多一种 reveal / reset
   场景，OpenSpiel 这边作者要在 rules 多处手动维护 visibility bit，
   DinoBoard 用统一的 helper 调用。

所以"DinoBoard 在 Priest peek 这件事上比 OpenSpiel 写得简洁"——是的，
但和"如果 OpenSpiel 作者按最优做法写"相比，简洁度差异比想象中小，主
要是 framework 强制 + CI 集中两条。和"OpenSpiel 作者按 observer 内部
判断那种偷懒写法写"相比，差异才大——但拿后者作对比是对 OpenSpiel
不公平的稻草人论证。

### 2.2 viz 是 state 的成员字段（不是平行结构）

`IGameState` 基类持有 `viz_`（以及框架计步用的 `step_count_`）作为成员，
和游戏 schema 字段并列。带来三层好处：

1. **生命周期统一**：`clone(state)` 自动连 viz 一起复制，sim-local 隔离
   零额外代码；wire snapshot、deep-copy、session 替换路径全部走 state
   同一条路径
2. **rules 写法自然**：`do_action_fast(State& s, Action a, rng)` 签名不变，
   rules 改 state 时顺手用 viz helper（`reveal_slot` / `reveal_slot_to(viewer)` /
   `reset_to_base`）——和改其它字段无差别。不需要在签名里多带一个
   `VizTensor&` 参数、不需要管"两个对象同步"
3. **viz 自指与 step_count 由 framework 内部处理**：`make_masked_state`
   用 `viz_[..., p]` 当 mask 决定其它字段拷真值还是写 `kPlaceholder`。
   如果 viz_ 当 schema 字段登记，walker 会把它也当业务数据遍历——既冗
   余又会引入"用 viz 决定 viz 自己是否泄漏"的循环。我们把 viz_ 和
   step_count_ 放在 `IGameState` 基类（不进 schema），walker 不遍历它
   们；viz_ 只作为 mask 被读取，step_count_ 由 framework 单独混入 hash
   digest 保 DAG 无环。结果是作者写 schema / rules / encoder 时一般不
   会接触到这两个字段

---

## 3. 我们牺牲什么

### 3.1 viz 必须是 (state, action, next_state) 的纯函数

rules 在 `do_action_fast` 里算 viz 时只能拿到当前 state、刚执行的
action、可选的 pre-action snapshot——**不能依赖外部 history buffer**。
这条约束派生出："任何 rules / viz 用得到的历史信息必须物化进 state"。

合理的桌游基本都满足：
- "玩家 X 在 ply N 看过 Y" → ply N 当下 rules `reveal_slot_to(state,
  &State::Y, owner, X)`，之后 viz 跟着 state 一起持续到 Y 内容变
  → state 充分统计量
- 同时出牌 → 暗牌 pending field（base viz `owner_only_diag`）+ 后续揭示
  action 调 `reveal_slot(state, &State::pending)` → 框架原生支持，不需
  要"同时移动"特殊机制
- 不完美回忆 → 遗忘 action `reset_to_base(state, &State::slot)` 主动
  1→0

**唯一不能直接表达的**：私人历史 → 私人 state field 之后，**observer 的
tracker 拿不到先验**。例：Coup exchange——p 看 2 张、保 2 张、退 2 张。把
"看过的 cid"建模进 state 后：

- p 自己 `viz=1` 知道 ✓
- observer `viz=0` 时由该 perspective 的 tracker 在 sim 入口经
  `randomize_unseen` 填值——但合适的先验是"p 基于看到的 X 选了 Y"，这
  个相关性**不在 current state 里**（要建模需要 p 在 ply N 当时的 policy +
  当时 court_deck 内容）
- 若 tracker 走 uniform 兜底（`randomize_unseen` 在不可见池里均匀抽），
  belief 退化成均匀分布 → AI 弱

这不是 viz 的限制，是 **tracker 的限制**：信息不对称游戏注册的 tracker
默认在"私人 state field"上会退化为均匀，作者想要强 AI 必须在
`randomize_unseen` 里写 game-specific 加权（建模 actor 的 policy 偏向）。
Coup 走的就是这条路——`coup_net_adapter.cpp` 里 claim/challenge 历史驱
动启发式权重——也解释了为什么 Coup AI 是六款里最难训的。

### 3.2 viz 函数本身公开

"哪个槽位对谁可见"这个 mask 是全员共识的游戏规则。game rules 可以让槽
位**内容**对部分玩家隐藏，"谁能看到这个槽位"这件事在 god-view 下唯一确
定。

桌游里基本够用：手牌、暗置骰子、隐藏移动（Scotland Yard）、暗放
（Skull / Codenames）——玩家都知道"自己看得到 X、看不到 Y、对手能看到
Z"。viz 是 god-view state 的确定函数。除非这个游戏有闭眼环节，否则你没法让别人不知道你是否知道某件事。

**狼人杀阵营互见**：狼人之间互相知道身份，所以平民不知道另外两个人是不
是互相知道身份。"X 能不能看到 Y 的身份牌"取决于 X 是不是狼人，而 X 的身
份是 hidden state——viz 嵌套在 hidden 上、不再是公开 mask。但**规则引擎
层面不需要建模这件事**：

- god view 下 X 的身份确定 → X 的 viz 确定
- 平民的 ISMCTS sample 出一个世界 → 这个世界里 X 的身份确定 → X 的 viz
  也确定 → 框架照常工作
- "viz 嵌套 hidden" 的高阶层在 belief 公式（平民推理"狼人之间在共享什么
  信息"），不在 rules + viz 层

### 3.3 viz 写错的 blast radius

| 写错形式 | 后果 |
|---|---|
| 该 reveal 没 reveal（viz=0 但其实公开了） | observer 多 sample 不可能的世界，强度↓；游戏正常 |
| 高阶推理漏写（"A 知道 B 知道 A 的牌"） | 该分支多 sample，强度↓；游戏正常 |
| **viz 把 0 写成 1（base 错标公开 / 槽位换内容忘 reset_to_base / reveal_to viewer 多发）** | **`make_masked_state` 按 viz 拷真值进 wire snapshot，把不该看的真值传给 observer = 信息泄露重大错误。CI 必抓** |

**写漏 = AI 弱一点；把 0 写成 1 = 正确性塌方**。walker 按 schema 派
MaskedState 时只信 viz——viz 错一格、observer 就拿到真值。CI 强制跑
`test_public_snapshot_round_trip`：每一手 truth → `make_masked_state`
→ wire snapshot → observer session → 观察者的 `state_hash_for_perspective`
必须和 truth bit-equal；新增公开字段忘更新 schema / `read_field_slot` /
`write_field_slot` 会被抓。再加 `test_encoder_respects_hash_scope`
（对手私有变化但 viz=1 不变 → encoder bit-equal）和
`test_public_hash_excludes_internal_rng`（60 种子扫不同 `session_rng`
hash 恒等），覆盖"truth 进 hash / encoder"这条线的几个常见路径。
这些测试不能保证"全部"覆盖，只是把已知可枚举的泄漏路径前置到 CI。

OpenSpiel 那边 observer 实现里"判断字段对 perspective 是否可见"的逻
辑写错也是 silent 的，定位方式同样靠 round-trip 测试。两边都靠测试
兜底；DinoBoard 的差异是把可见性从"observer 内部代码逻辑"提到
"state 上的 viz tensor"——契约表面积更小（一格 viz 而非一段 if-else），
但不是 0。doc 早期版本用过 "structurally impossible to leak" 这种措
辞，更准确的说法是**契约表面积更小、CI 可枚举的泄漏路径前置覆盖**，
作者写错 viz 仍然能泄漏。

### 3.4 高阶推理是作者的可选投入

框架对"高阶推理"宽容。例：Love Letter King 让两人交换手牌——交换后**两
人互相知道对方的新牌**。严格建模要求 rules 在 King swap 的 `do_action_
fast` 里：

```cpp
reveal_slot_to(state, &State::hand, A, B);   // B 知道 A 现在的牌
reveal_slot_to(state, &State::hand, B, A);   // A 知道 B 现在的牌
```

下一回合 B 抽新牌后出牌：rules 一般 `reset_to_base(state, &State::hand, B)`
→ A 不再知道 B 手牌（保守建模）。如果作者想更细致——B 出的不是 A 知道
的那张，A 应该仍知道剩下那张——可以再写一段推理在 `do_action_fast` 里
判断要不要 reset。

**作者的可选投入**：在 `do_action_fast` 里多写几行高阶 viz 推理，换 AI
强度。多写一分多一分，不写也跑得通。

**最坏情况**：observer 在 belief 里多 sample 几个不可能的世界，少一点
点强度，游戏照样进行、框架照常工作。

设计意图：写新游戏先把一阶事实（自己看到的、被公开的）写对就能跑
可用 AI；高阶推理增量补不破坏正确性、不污染其它路径、不影响协议层。

**一阶事实写错** = 决策错（必修），**高阶推理漏写** = 强度低一点
（选修）。

---

## 4. 第三方接入

第三方游戏作者要在 DinoBoard 框架里写一款新游戏，**总工作量**：

1. 写 `IGameState` / `IGameRules` / `do_action_fast`（游戏规则本身——任
   何框架都要写）；undo 路径只 tail solver 用，写 `do_action_deterministic`
   + `undo_action`（可选，看是否要 tail solver）
2. schema declare 字段 name + 数据 shape + base viz tensor（`all_public` /
   `owner_only_diag` / `all_hidden`）—— 一次性元数据，写在
   `games/<id>/<id>_visibility.cpp`
3. `do_action_fast` 里维护 viz：用 framework 提供的 helper（`reveal_slot` /
   `reveal_slot_to(viewer)` / `reset_to_base`）—— 作者本来就要想可见性
4. 实现每槽 typed value emission：`hash_field_slot` / `read_field_slot` /
   `write_field_slot` / `mask_field_slot`—— framework walker 调用这些把字段
   接进 hash digest / wire values dict / placeholder mask
5. 写 encoder：签名锁 `const MaskedState&`，viz=0 槽位读到的就是
   `kPlaceholder`，作者无需查 viz、无需 `is_visible_to`，结构上不可能编进
   真值
6. （可选）写 tracker 维护公开衍生特征 + `randomize_unseen` 采样策略——
   全公开 / 物理随机游戏（TTT / Quoridor / Azul）不强制注册；信息不对称
   游戏（LL / Splendor / Coup）必须注册
7. （可选）`randomize_unseen` 走 game-specific 加权——默认 uniform 在
   "viz=0 槽位"均匀采样；私人 state field（如 Coup exchange 私人记忆 /
   claim 历史）想要强 AI 必须自己写加权（参考
   `coup_net_adapter.cpp`）

**不需要写**：hash digest 组装、wire snapshot 序列化、session-side
apply、ISMCTS sim 框架、chance node / 同时移动 / 不完美回忆的特殊机
制——全是 framework 从 schema + viz + walker 派生。OpenSpiel 同款游戏
要额外写 `InformationStateString` + 一个 Tensor 系列 + `ResampleFromInfostate`
（多份语义自洽由作者保证）+（如有公开随机）`ChanceOutcomes` 等。

---

## 5. 和其它框架的对比

为什么不直接用 OpenSpiel / PettingZoo / RLCard 等现成框架？按"工具是否
和需求匹配"分类：

| 框架 | 板游戏 AI? | 隐藏信息? | 训练 pipeline? | Web 前端? | 状态 |
|---|---|---|---|---|---|
| Stable-Baselines3 / Tianshou / RLlib | 弱（model-free） | 不擅长 | 通用 | 无 | 活跃 |
| PettingZoo / Gymnasium | env only | env only | 无 | 无 | 活跃 |
| RLCard | 仅卡牌 | NFSP/DMC | ✓（卡牌 only） | 无 | 活跃 |
| OpenSpiel | ✓ | ✓（多份序列化作者手对齐） | ✓（算法广度优先） | 无 | 活跃 |
| Polygames | ✓ | ✗ | ✓ | 无 | **死了**（2022 起无更新） |
| alpha-zero-general | ✓ | ✗ | 教学级 | 无 | 半活 |
| KataGo / Leela Chess Zero | 单游戏 | ✗ | game-specific | 有 | 活跃 |
| **DinoBoard** | ✓ | ✓ | ✓（深度优化 + 强度调教） | ✓ | 活跃 |

逐项展开：

### 5.1 通用 RL 框架（SB3 / Tianshou / RLlib / CleanRL）— 工具错了

这些是 model-free RL（PPO / DQN / SAC）的轮子。**对完美信息棋类游戏
model-free 远比 model-based search 弱**——AlphaZero 之所以 work 是因
为 MCTS lookahead + value network，不是 PPO。SB3 跑 TicTacToe 都打不过
随机加 minimax。

对隐藏信息游戏更糟：这些框架根本没 ISMCTS / determinization 概念。
Coup / Love Letter 在 PPO 里训不出强度。

### 5.2 PettingZoo / Gymnasium — 只是 env 接口

它们提供"多 agent 环境标准 API"——`reset()` / `step(action)` /
`action_space` / `observation_space`——**不提供训练 pipeline、不提供
搜索、不提供 belief tracking、不提供 web 前端**。把 DinoBoard 拆开
看，PettingZoo 只对应到 IGameState / IGameRules 那一层。

类比：PettingZoo 像 USB 接口标准，告诉你插头长什么样；DinoBoard /
OpenSpiel 是配齐的整套电脑。

### 5.3 RLCard — 只做卡牌且无 search

定位是"专为扑克 / 斗地主这种 mixed-strategy 重的卡牌做的 model-free
RL benchmark"。算法栈：DQN / NFSP / DMC / CFR-family，**没有任何
search-based 算法**（无 MCTS / 无 ISMCTS）。

不适用原因：
- Scope 限定卡牌——Splendor / Azul / Quoridor 这种 token / 棋盘游戏
  接进去要重新发明轮子
- Coup / LL 这种"隐藏信息但 strategic depth 主要靠几步 lookahead"
  的游戏，纯 NFSP / DMC 训不出强度——它的算法栈和我们的需求不匹配
- 无 web 前端、无对局 API、无复盘工具

### 5.4 OpenSpiel — 算法库更广，定位是研究 benchmark

OpenSpiel 是 game AI 圈事实标准，覆盖最广，活跃度最高。算法库覆盖完
整：`open_spiel/python/algorithms/` 下有 AlphaZero、Deep CFR、NFSP、
DQN、policy gradient、MCTS / ISMCTS、若干 CFR 变体、PSRO 等；selfplay
loop、evaluator、exploitability 工具都有。这条上 DinoBoard 不比它强。

DinoBoard 跟它的差异在三点：

**A. 可见性事实存放位置不同**——跑 ISMCTS + neural eval 在现代
Observer API 下要写一个 observer（同时供应 info-set string 和 network
tensor）+ `ResampleFromInfostate`。observer 内部按 perspective 判断哪
些字段当前可见、哪些写公开 placeholder——可见性事实在作者代码里。
DinoBoard 的 schema + viz tensor + walker 把这条事实提到 state 字段
上，作者只在 `do_action_fast` 里维护 viz 一格 0/1，walker 自动派生
hash / snapshot / encoder 三家。差异不是"OpenSpiel 没接口、DinoBoard
有结构"——OpenSpiel 的 Observer 已经是结构化接口、且 string 和
tensor 共享实现——而是契约表面积：observer 内部逻辑（一段代码） vs
state 上的 viz tensor（一格 bit）。在每款游戏 perspective-private
slot 不多时这条差异不显著；在 Priest peek / Coup exchange 这种动态
揭示频繁的游戏里 viz tensor 表达更直接。

**B. 不提供 web 前端、对局 API、复盘工具**——OpenSpiel 是**算法研究
的 benchmark 平台**，定位是"给你跑实验对比 baseline 用的"。从"接个
游戏"做到"有个能玩的 AI 在前端展示给朋友看"通常要再花 1-3 个月写
对局服务 + 前端 + 复盘 UI。DinoBoard 默认就把 selfplay → arena →
web → 复盘分析全装在同一仓库里，开箱可玩、可演示。

**C. 算法选型不同（不是创新）**——核心思想两边一样：SO-ISMCTS
（Cowling 2012）+ per-search DAG 节点共享，按 acting player 的 info
set 当 key。具体实现：

- **Key 形式**：OpenSpiel 用 `(player, infostate string)`，string 是
  作者写的 `InformationStateString`；DinoBoard 用 `state_hash_for_
  perspective(p)` 走 schema digest，不走 string。两种都能 work，正确
  性等价；digest 路径让 hash key 跟 encoder 输入消费同一份
  MaskedState 对象，对齐由数据流保证而非两份函数语义对齐。
- **防环**：OpenSpiel 没有显式机制，靠作者写的 InfoState string 通常
  含历史而单调；DinoBoard 用 `step_count_` 进 digest，DAG 结构性
  acyclic 不依赖作者怎么写。两种都能 work；step_count 路径把这条不
  变量从作者契约移进框架。
- **Transposition / DAG**：两边都是 DAG。OpenSpiel `ISMCTSBot` 用
  `absl::flat_hash_map<(player, infostate string), ISMCTSNode*>` 共享
  节点；DinoBoard 用 `state_hash_for_perspective(p)` 走 schema digest
  共享。两种都能 work，正确性等价；digest 路径让 hash key 跟 encoder
  输入消费同一份 MaskedState 对象，对齐由数据流保证而非两份函数语义
  对齐。
- **防环**：OpenSpiel 没有显式机制，靠作者写的 InfoState string 通常
  含历史而单调（Observer 模式下用 `perfect_recall=true` 一般也含历史；
  `allow_inconsistent_action_sets` flag 暗示 observation-only 模式遇到
  过这类问题）；DinoBoard 用 `step_count_` 进 digest，DAG 结构性
  acyclic 不依赖作者怎么写。两种都能 work；step_count 路径把这条不
  变量从作者契约移进框架。
- **Selection**：OpenSpiel ISMCTSBot 是 UCB1（scalar `uct_c_`，无
  policy prior），rollout/value 走 `Evaluator`。DinoBoard 用 PUCT
  （AlphaZero 风格 policy prior 进 selection）+ UCT2（Childs 2008，
  多入边场景下把 sqrt 分子换成入边 visit）。这条是真正的算法选型差
  异——neural-guided 训练用 PUCT 是 AlphaZero 之后的业界默认；UCT2
  在 transposition 频繁场景下，Childs 论文（Hex / Amazons）实测对
  UCT1 改善 5-15%——本仓库 6 款桌游没跑 UCT1 vs UCT2 的对照实验，
  这是引用论文数字而非本仓库实测。

DinoBoard 这套搜索没有算法层创新，是已发表组合的工程选型。tail
solver / heuristic guidance / 对手池调度是产品工程附加项，OpenSpiel
没有原生提供——但要在 OpenSpiel 上加一层也不难。

DinoBoard 当前接入的 roster（TTT / Quoridor / Azul / Splendor /
Love Letter / Coup）覆盖了 framework 想支撑的几类形态：完全公开+确
定、公开+物理随机、信息不对称+高阶推理（Priest peek）、信息不对称
+诈唬（claim/challenge 驱动 belief 加权）。在这个范围内 framework
跑通，但 cover 不到的场景（FOSG / 长历史 / 同时移动 / 大组合动作）
真接进来会怎样还没验证过。

什么时候**应该**选 OpenSpiel：研究 baseline 复现、需要 CFR / NFSP /
PSRO 等 mixed-strategy 算法、做学术 paper 需要审稿人认得的事实标准。
详见 §5.9。

### 5.5 Polygames — 死了

Facebook 2019-2021 年的 AlphaZero 研究框架，最像 DinoBoard 精神的
同类项。**完美信息 only，没有隐藏信息支持**。2022 年起无人维护，
GitHub issues 没人看——生产路径上等于不可用。

### 5.6 alpha-zero-general / muzero-general — 教学实现

教育用途，**不是框架**。每接一款游戏都要重写 training loop / eval /
web 展示 / 复盘。一个人花 1-2 个月能把它跑起来，但每加一款新游戏
又要重新拼，没有杠杆。

### 5.7 KataGo / Leela Chess Zero — 单游戏专用

Go / Chess 单一游戏的 SOTA 实现，代码深度优化但**完全 game-specific**，
不是框架。

### 5.8 自己直接拼

PyTorch + 自己写 MCTS + 自己写 selfplay loop 这条路对 1-2 款游戏没问
题；做多款多变体 + 持续训练 + 隐藏信息 + 网页对战时，会重复写 ISMCTS
/ ONNX 加载 / selfplay scheduler / web 前端等基础设施。是否值得自己
拼，看时间预算和有多少款游戏要做。

### 5.9 反过来：什么时候选 OpenSpiel 不选 DinoBoard

- 要做扑克 / Hanabi / 长历史依赖 / 同时移动 / 大组合动作空间游戏 →
  DinoBoard 框架非目标里写过不接
- 要研究 mixed-strategy 均衡 / CFR / NFSP / PSRO / exploitability → OpenSpiel 有，DinoBoard 没有
- 要做学术 paper 的 benchmark 复现 → OpenSpiel 是事实标准，审稿人认
- 要 Python 之外的多语言绑定（C++ / Julia / Swift） → OpenSpiel 多语言绑定成熟

### 5.10 总结

OpenSpiel 算法库更广更深，是研究 benchmark 事实标准，定位在算法研究；
DinoBoard 定位在"训出能玩的 AI 并接到 web 前端"，selfplay / arena /
web / 复盘分析在同一仓库里走通。Polygames 是 AlphaZero 全栈但完美
信息 only 且 2022 起无更新；RLCard 是隐藏信息但限于卡牌且无 search。
DinoBoard 接受的范围比 OpenSpiel 窄（FOSG / 长历史 / 大组合动作 / 同
时移动不接），换的是接 cover 范围内的隐藏信息桌游时心智负担更低。

要 baseline 复现 / CFR / NFSP / mixed-strategy 算法 / 学术 paper
事实标准：用 OpenSpiel。要把一款隐藏信息桌游训出 AI 并接到 web 前
端、不打算自己拼前端和复盘工具：DinoBoard 也许合适。两个项目目标
不重叠。
