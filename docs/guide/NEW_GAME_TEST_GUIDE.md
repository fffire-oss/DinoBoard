# 新游戏验收测试流程

> 当你实现了一个新游戏后，按照本文档的步骤逐一验证。
> 每一步都对应一个具体的 bug 模式——这些都是在 DinoBoard 开发过程中实际踩过的坑。

---

## 前置条件

```bash
pip install -e .
pip install pytest torch
```

确认游戏注册成功：

```bash
python -c "import dinoboard_engine; print(dinoboard_engine.available_games())"
# 输出应包含你的 game_id
```

---

## 第 1 步：注册 + config 一致性

**验证什么**：game.json 中的 `feature_dim` 和 `action_space` 必须与 C++ encoder 精确匹配。不一致会导致训练时 tensor shape mismatch 或 ONNX 推理崩溃。

```python
import dinoboard_engine

GAME_ID = "your_game"

def test_registered():
    assert GAME_ID in dinoboard_engine.available_games()

def test_encode_state_dim_matches_config():
    import json
    with open(f"games/{GAME_ID}/config/game.json") as f:
        cfg = json.load(f)
    info = dinoboard_engine.encode_state(GAME_ID, seed=42)
    assert len(info["features"]) == cfg["feature_dim"], (
        f"encoder says {len(info['features'])}, config says {cfg['feature_dim']}"
    )
    assert len(info["legal_mask"]) == cfg["action_space"], (
        f"encoder says {len(info['legal_mask'])}, config says {cfg['action_space']}"
    )
```

**踩坑参考**：KNOWN_ISSUES §通用踩坑 3

---

## 第 2 步：GameSession 基本交互

**验证什么**：游戏能创建、走棋、到达终局。

```python
def test_session_creation():
    gs = dinoboard_engine.GameSession(GAME_ID, seed=42)
    assert not gs.is_terminal
    assert gs.current_player >= 0
    assert gs.num_players >= 2
    assert len(gs.get_legal_actions()) > 0

def test_apply_action_changes_state():
    gs = dinoboard_engine.GameSession(GAME_ID, seed=42)
    state_before = gs.get_state_dict()
    legal = gs.get_legal_actions()
    gs.apply_action(legal[0])
    assert gs.get_state_dict() != state_before

def test_random_game_reaches_terminal():
    """随机下棋直到游戏结束或 500 步，确保不崩溃。"""
    gs = dinoboard_engine.GameSession(GAME_ID, seed=42)
    for _ in range(500):
        if gs.is_terminal:
            break
        legal = gs.get_legal_actions()
        assert len(legal) > 0, "非终局状态必须有合法动作"
        gs.apply_action(legal[0])
    # 无论是否终局，都不应 crash

def test_terminal_has_valid_winner():
    gs = dinoboard_engine.GameSession(GAME_ID, seed=42)
    for _ in range(500):
        if gs.is_terminal:
            break
        gs.apply_action(gs.get_legal_actions()[0])
    if gs.is_terminal:
        assert gs.winner >= -1  # -1 = 平局/未判定
```

---

## 第 3 步：do_action / undo_action 一致性 (MCTS 根基)

**验证什么**：MCTS 搜索中每次模拟会 do 几十步再 undo 回去。如果 undo 不完美，搜索树会被静默污染。

**最简单的验证方式**：相同 seed 应产生完全相同的 episode。

```python
def test_do_undo_via_determinism():
    """如果 do/undo 有 bug，两次运行会产生不同结果。"""
    ep1 = dinoboard_engine.run_selfplay_episode(
        game_id=GAME_ID, seed=777, model_path="", simulations=100,
        max_game_plies=50, dirichlet_alpha=0.0, dirichlet_epsilon=0.0,
    )
    ep2 = dinoboard_engine.run_selfplay_episode(
        game_id=GAME_ID, seed=777, model_path="", simulations=100,
        max_game_plies=50, dirichlet_alpha=0.0, dirichlet_epsilon=0.0,
    )
    assert ep1["total_plies"] == ep2["total_plies"]
    assert ep1["winner"] == ep2["winner"]
    for s1, s2 in zip(ep1["samples"], ep2["samples"]):
        assert s1["action_id"] == s2["action_id"]
        assert s1["features"] == s2["features"]

def test_state_stable_after_mcts():
    """MCTS 搜索后 state 应恢复到搜索前。"""
    gs = dinoboard_engine.GameSession(GAME_ID, seed=42)
    state_before = gs.get_state_dict()
    gs.get_ai_action(simulations=100, temperature=0.0)
    assert gs.get_state_dict() == state_before

def test_legal_actions_stable_after_mcts():
    gs = dinoboard_engine.GameSession(GAME_ID, seed=42)
    legal_before = sorted(gs.get_legal_actions())
    gs.get_ai_action(simulations=50, temperature=0.0)
    assert sorted(gs.get_legal_actions()) == legal_before
```

**踩坑参考**：KNOWN_ISSUES §通用踩坑 5（do_action_fast 和 undo_action 必须完美逆操作）

---

## 第 4 步：特征编码正确性 (BUG-007 重灾区)

**验证什么**：
1. 每一步的 features 必须来自当前局面（不是初始局面）
2. features 长度 == feature_dim
3. legal_mask 长度 == action_space
4. legal_mask 只在合法动作位置为 1

```python
def test_features_vary_across_plies():
    """最重要的回归测试：不同步的 features 必须不同。
    BUG-007 就是因为每步都用了初始局面的 features 训练。"""
    ep = dinoboard_engine.run_selfplay_episode(
        game_id=GAME_ID, seed=42, model_path="", simulations=10,
        max_game_plies=50,
    )
    samples = ep["samples"]
    if len(samples) >= 4:
        assert samples[0]["features"] != samples[3]["features"], (
            "ply 0 和 ply 3 的特征完全相同——检查 encoder 是否在每步正确编码当前状态"
        )

def test_features_differ_from_initial():
    """中间局面的 features 不应等于初始局面。"""
    init_info = dinoboard_engine.encode_state(GAME_ID, seed=42)
    ep = dinoboard_engine.run_selfplay_episode(
        game_id=GAME_ID, seed=42, model_path="", simulations=10,
        max_game_plies=50,
    )
    for s in ep["samples"][2:]:
        assert s["features"] != init_info["features"], (
            f"ply {s['ply']}: 特征和初始局面相同！encoder 是否编码了错误的状态？"
        )

def test_feature_and_mask_lengths():
    import json
    with open(f"games/{GAME_ID}/config/game.json") as f:
        cfg = json.load(f)
    ep = dinoboard_engine.run_selfplay_episode(
        game_id=GAME_ID, seed=42, model_path="", simulations=10,
        max_game_plies=50,
    )
    for s in ep["samples"]:
        assert len(s["features"]) == cfg["feature_dim"]
        assert len(s["legal_mask"]) == cfg["action_space"]

def test_legal_mask_matches_legal_actions():
    """legal_mask[a] == 1 当且仅当 a 是合法动作。"""
    ep = dinoboard_engine.run_selfplay_episode(
        game_id=GAME_ID, seed=42, model_path="", simulations=10,
        max_game_plies=50,
    )
    for s in ep["samples"]:
        legal_set = {i for i, m in enumerate(s["legal_mask"]) if m > 0}
        visited = {aid for aid, v in zip(
            s["policy_action_ids"], s["policy_action_visits"]) if v > 0}
        assert visited.issubset(legal_set), (
            f"ply {s['ply']}: MCTS 访问了 legal_mask 之外的动作 {visited - legal_set}"
        )
```

**踩坑参考**：KNOWN_ISSUES §BUG-007

---

## 第 5 步：Selfplay 样本完整性

**验证什么**：selfplay 运行正常，输出的训练数据结构合法。

```python
def test_selfplay_completes():
    ep = dinoboard_engine.run_selfplay_episode(
        game_id=GAME_ID, seed=42, model_path="", simulations=30,
        max_game_plies=100,
    )
    assert ep["total_plies"] > 0
    assert len(ep["samples"]) > 0

def test_policy_ids_in_range():
    import json
    with open(f"games/{GAME_ID}/config/game.json") as f:
        cfg = json.load(f)
    ep = dinoboard_engine.run_selfplay_episode(
        game_id=GAME_ID, seed=42, model_path="", simulations=10,
        max_game_plies=50,
    )
    for s in ep["samples"]:
        for aid in s["policy_action_ids"]:
            assert 0 <= aid < cfg["action_space"], (
                f"ply {s['ply']}: action_id {aid} 超出 action_space"
            )

def test_visits_equal_simulations():
    """每步的总 visit count 应恰好等于 simulations 数。"""
    SIMS = 50
    ep = dinoboard_engine.run_selfplay_episode(
        game_id=GAME_ID, seed=42, model_path="", simulations=SIMS,
        max_game_plies=50,
    )
    for s in ep["samples"][:5]:
        total = sum(s["policy_action_visits"])
        assert total == SIMS, (
            f"ply {s['ply']}: visits={total}, 应为 {SIMS}"
        )

def test_z_values_consistent_with_winner():
    ep = dinoboard_engine.run_selfplay_episode(
        game_id=GAME_ID, seed=42, model_path="", simulations=30,
        max_game_plies=100,
    )
    winner = ep["winner"]
    if ep["draw"] or winner < 0:
        return
    for s in ep["samples"]:
        z_vals = s["z_values"]
        if not z_vals:
            continue
        assert z_vals[winner] > 0, f"赢家 z 值应 > 0"
        for p in range(len(z_vals)):
            if p != winner:
                assert z_vals[p] <= 0, f"输家 z 值应 <= 0"

def test_chosen_action_was_visited():
    ep = dinoboard_engine.run_selfplay_episode(
        game_id=GAME_ID, seed=42, model_path="", simulations=30,
        max_game_plies=50,
    )
    for s in ep["samples"]:
        visit_map = dict(zip(s["policy_action_ids"], s["policy_action_visits"]))
        assert visit_map.get(s["action_id"], 0) > 0, (
            f"ply {s['ply']}: 选中动作 {s['action_id']} 的 visit count 为 0"
        )
```

---

## 第 6 步：ONNX 导出回环

**验证什么**：PyTorch 模型能成功导出为 ONNX，然后被 C++ ONNX evaluator 加载并用于 selfplay。

```python
from training.model import create_model_from_config, export_onnx

def test_onnx_roundtrip(tmp_path):
    import json
    with open(f"games/{GAME_ID}/config/game.json") as f:
        cfg = json.load(f)
    net = create_model_from_config(cfg)
    onnx_path = tmp_path / "test_model.onnx"
    export_onnx(net, onnx_path, cfg["feature_dim"])
    assert onnx_path.exists()

    # 用导出的模型跑 selfplay
    ep = dinoboard_engine.run_selfplay_episode(
        game_id=GAME_ID, seed=42, model_path=str(onnx_path),
        simulations=10, max_game_plies=30,
    )
    assert len(ep["samples"]) > 0, "ONNX 模型无法用于 selfplay"

    # 用导出的模型做 AI 决策
    gs = dinoboard_engine.GameSession(
        GAME_ID, seed=42, model_path=str(onnx_path))
    result = gs.get_ai_action(simulations=10, temperature=0.0)
    legal = gs.get_all_legal_actions()
    assert result["action"] in legal
```

---

## 第 7 步：训练 tensor 验证

**验证什么**：样本经 Python 处理后的训练 tensor 形状正确、数值有限、能跑通前向+反向传播。

```python
import torch
from training.pipeline import normalize_policy, train_step
from training.model import create_model_from_config

def test_training_tensors_and_gradient(tmp_path):
    import json
    with open(f"games/{GAME_ID}/config/game.json") as f:
        cfg = json.load(f)

    ep = dinoboard_engine.run_selfplay_episode(
        game_id=GAME_ID, seed=42, model_path="", simulations=30,
        max_game_plies=100,
    )

    features, policies, values, masks = [], [], [], []
    for s in ep["samples"]:
        feats = s["features"]
        if len(feats) != cfg["feature_dim"]:
            continue
        features.append(feats)
        policies.append(normalize_policy(
            s["policy_action_ids"], s["policy_action_visits"],
            cfg["action_space"]))
        z_vals = s["z_values"]
        player = s["player"]
        n = len(z_vals) if z_vals else cfg.get("num_players", 2)
        rotated = [z_vals[(player + i) % n] for i in range(n)] if z_vals else [0.0] * n
        values.append(rotated)
        masks.append(s["legal_mask"])

    assert len(features) > 0, "没有有效样本"

    feat_t = torch.tensor(features, dtype=torch.float32)
    pol_t = torch.tensor(policies, dtype=torch.float32)
    val_t = torch.tensor(values, dtype=torch.float32)
    mask_t = torch.tensor(masks, dtype=torch.float32)

    assert feat_t.shape[1] == cfg["feature_dim"]
    assert pol_t.shape[1] == cfg["action_space"]
    assert torch.isfinite(feat_t).all(), "features 中有 NaN/Inf"
    assert torch.isfinite(pol_t).all(), "policies 中有 NaN/Inf"

    # policy 应和为 1
    for i in range(pol_t.size(0)):
        total = pol_t[i].sum().item()
        if total > 0:
            assert abs(total - 1.0) < 1e-5, f"sample {i}: policy 总和 = {total}"

    # policy 非零位必须在 legal_mask 内
    for i in range(pol_t.size(0)):
        illegal_mass = (pol_t[i] * (1 - mask_t[i])).sum().item()
        assert illegal_mass < 1e-9, f"sample {i}: 非法动作上有 policy 权重"

    # 前向+反向传播
    net = create_model_from_config(cfg)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    metrics = train_step(net, optimizer, feat_t, pol_t, val_t,
                         legal_mask=mask_t)
    import math
    assert math.isfinite(metrics["loss"]), f"loss = {metrics['loss']}"
    assert math.isfinite(metrics["policy_loss"])
    assert math.isfinite(metrics["value_loss"])
```

---

## 第 8 步：可选组件验证

根据你注册了哪些可选组件，运行对应测试。

### 8a. heuristic_picker

```python
def test_heuristic_returns_legal():
    gs = dinoboard_engine.GameSession(GAME_ID, seed=42)
    result = gs.get_heuristic_action()
    assert "action" in result, "如果没注册 heuristic_picker 这会是空 dict"
    legal = gs.get_all_legal_actions()
    assert result["action"] in legal

def test_heuristic_episode_has_features():
    ep = dinoboard_engine.run_heuristic_episode(
        game_id=GAME_ID, seed=42, temperature=1.0, max_game_plies=50,
    )
    assert len(ep["samples"]) > 0
    import json
    with open(f"games/{GAME_ID}/config/game.json") as f:
        cfg = json.load(f)
    for s in ep["samples"]:
        assert len(s["features"]) == cfg["feature_dim"]
```

### 8b. tail_solver

```python
def test_tail_solve_api():
    r = dinoboard_engine.tail_solve(
        game_id=GAME_ID, seed=42, perspective_player=0,
        depth_limit=3, node_budget=5000,
    )
    assert "value" in r
    assert "best_action" in r
    assert "nodes_searched" in r
    assert "budget_exceeded" in r

def test_tail_solve_stats_invariant():
    ep = dinoboard_engine.run_selfplay_episode(
        game_id=GAME_ID, seed=42, model_path="", simulations=10,
        max_game_plies=200, tail_solve_enabled=True,
        tail_solve_start_ply=1, tail_solve_depth_limit=3,
        tail_solve_node_budget=500,
    )
    a = ep["tail_solve_attempts"]
    c = ep["tail_solve_completed"]
    s = ep["tail_solve_successes"]
    assert s <= c <= a, f"不变量违反: {s} <= {c} <= {a}"
```

### 8c. training_action_filter

```python
def test_filter_reduces_actions():
    gs_f = dinoboard_engine.GameSession(GAME_ID, seed=42, use_filter=True)
    gs_u = dinoboard_engine.GameSession(GAME_ID, seed=42, use_filter=False)
    assert len(gs_f.get_legal_actions()) <= len(gs_u.get_legal_actions())

def test_filter_is_subset():
    gs_f = dinoboard_engine.GameSession(GAME_ID, seed=42, use_filter=True)
    gs_u = dinoboard_engine.GameSession(GAME_ID, seed=42, use_filter=False)
    assert set(gs_f.get_legal_actions()).issubset(set(gs_u.get_legal_actions()))
```

### 8d. adjudicator

```python
def test_adjudicator_assigns_result():
    """短局应触发 adjudicator，z_values 不应全为 0。"""
    ep = dinoboard_engine.run_selfplay_episode(
        game_id=GAME_ID, seed=42, model_path="", simulations=10,
        max_game_plies=5,
    )
    if ep["total_plies"] >= 5:
        has_nonzero = any(
            any(v != 0.0 for v in s["z_values"])
            for s in ep["samples"] if s["z_values"]
        )
        assert has_nonzero or ep["draw"], (
            "adjudicator 应判定胜负或平局，而非留下未定义结果"
        )
```

### 8e. auxiliary_scorer

```python
def test_auxiliary_score_finite():
    ep = dinoboard_engine.run_selfplay_episode(
        game_id=GAME_ID, seed=42, model_path="", simulations=10,
        max_game_plies=30,
    )
    for s in ep["samples"]:
        score = s.get("auxiliary_score", 0.0)
        assert -100 < score < 100, f"ply {s['ply']}: 异常分值 {score}"
```

### 8f. 隐藏信息（IBeliefTracker + ISMCTS root 采样）

```python
def test_hidden_info_selfplay_with_high_sims():
    """高 simulations 验证 ISMCTS root 采样在重复 sim 下不崩。"""
    ep = dinoboard_engine.run_selfplay_episode(
        game_id=GAME_ID, seed=42, model_path="", simulations=100,
        max_game_plies=50,
    )
    assert ep["total_plies"] > 0

def test_hidden_info_deterministic():
    """隐藏信息游戏也必须满足种子确定性。"""
    ep1 = dinoboard_engine.run_selfplay_episode(
        game_id=GAME_ID, seed=42, model_path="", simulations=20,
        max_game_plies=30, dirichlet_alpha=0.0, dirichlet_epsilon=0.0,
    )
    ep2 = dinoboard_engine.run_selfplay_episode(
        game_id=GAME_ID, seed=42, model_path="", simulations=20,
        max_game_plies=30, dirichlet_alpha=0.0, dirichlet_epsilon=0.0,
    )
    assert ep1["total_plies"] == ep2["total_plies"]
    assert ep1["winner"] == ep2["winner"]

def test_dag_reuse_hits_present(self):
    """DAG 节点应被复用——高 sim 下信息集会被多次访问，dag_reuse_hits 应 > 0。
    调用 GameSession.apply_ai_action 获取 per-decision stats；不是每个 ep sample 都暴露。"""
    gs = dinoboard_engine.GameSession(GAME_ID, seed=42)
    result = gs.apply_ai_action(simulations=200, temperature=0.0)
    stats = result["stats"]
    assert stats["dag_reuse_hits"] > 0, (
        "dag_reuse_hits 一直为 0——hash_field_slot 可能漏了字段，或 schema 漏声明 all_public 字段"
    )
```

如果你的游戏有**有状态 belief tracker**（如 Splendor 的 `seen_cards` 追踪），还需要验证 tracker 不偷看隐藏状态：

```python
def test_randomize_unseen_does_not_peek():
    """randomize_unseen 不应读真实 deck 内容。
    如果 tracker 不偷看，随机化后的 deck 组成应与真实 deck 不同
    （因为 unseen pool 来自 全卡池-seen，包含已购买但不在 deck 中的牌）。
    """
    r = dinoboard_engine.test_belief_tracker(
        GAME_ID, seed=42, plies=20, randomize_trials=10,
    )
    assert r["plies"] > 0
    orig = sorted(r["original_deck"])
    diffs = sum(1 for td in r["trial_decks"] if sorted(td) != orig)
    assert diffs > 0, "所有 trial 都和真实 deck 一样——tracker 可能在偷看"

def test_randomize_unseen_preserves_deck_size():
    """randomize_unseen 必须保持 deck 大小不变（公开信息）。"""
    r = dinoboard_engine.test_belief_tracker(
        GAME_ID, seed=99, plies=15, randomize_trials=10,
    )
    orig_len = len(r["original_deck"])
    for i, td in enumerate(r["trial_decks"]):
        assert len(td) == orig_len, f"Trial {i}: deck 大小 {len(td)} != {orig_len}"
```

**踩坑参考**：KNOWN_ISSUES §BUG-017（belief tracker 偷看牌堆内容）

#### 8f-bis. tracker 已知信息 vs ground truth 一致性（强制）

`test_randomize_unseen_does_not_peek` 只验证 tracker **不偷看**（随机化结果不和真实 deck 一致）。它不验证 tracker **声称的"已知"信息真的对得上**。两个 tracker 用错误的事件应用逻辑可能同时把"对手手牌 = 牧师"理解错（实际是男爵），互相之间一致但都和真相相悖——`test_api_belief_matches_selfplay` 抓不到这种 bug，因为两边都错得一样。

**标准化验收**：每个有 hidden info 的游戏都要在 `tests/framework/test_tracker_consistent_with_truth.py` 加一个 checker 函数，对每个 ply 把 tracker `serialize()` 里的"声称已知"字段和 GT `get_state_dict()` 里的真实字段比对。规则是 **`claim != UNKNOWN_SENTINEL` ⇒ `claim == truth`**；声称"未知"永远 OK，声称"已知 X" 但实际是 Y 必须失败。

```python
# tests/framework/test_tracker_consistent_with_truth.py 里加：
def _check_<game>(state: dict, snap: dict, perspective: int) -> None:
    # 取 tracker 声称的已知信息
    known = snap.get("<your_known_field>")  # e.g. known_hand, known_role
    if known is None:
        return
    for p, claim in enumerate(known):
        if claim == 0:                      # UNKNOWN sentinel
            continue
        truth = state["players"][p]["<truth_field>"]
        assert claim == truth, (
            f"<game>: tracker claims player {p} has {claim} but truth is {truth}")

_CHECKERS["<game>"] = _check_<game>
```

跑 `pytest tests/framework/test_tracker_consistent_with_truth.py -v -k <game_id>`,5 个 seed 都过。

**为什么这个测试不能省**:tracker 是 AI 决策链路里**最容易悄悄出错**的地方。它不像 do/undo 一致性那样能从结果看出问题——tracker 错了只会让 AI 的局面理解偏移,胜率掉一点,看起来像是模型不够强,排查起来非常痛。这个测试把"tracker 声称的事实和真实事实不符"直接钉死成一个失败,**比任何后期复盘都便宜**。

#### 8f-ter. ISMCTS 根采样必须尊重 tracker 的"已知"声明（强制）

`8f-bis` 验证的是「tracker 声称的已知信息 == 真实信息」。这一步验证下游的另一根链子：**`belief_tracker.randomize_unseen(state, rng)` 给 MCTS 仿真填充隐藏槽位时,是否尊重 tracker 已经知道的事实**。

具体问题场景:Love Letter 里你用 Priest 看了对手手牌是 Princess——rules 端 `reveal_slot_to(actor)` 把 `state.viz["hand"][opp, :, actor]=1`，session state 上 `hand[opp]` 槽位是 viz=1 + 真值。然后 MCTS 每个 sim 都调 `randomize_unseen` 复制一份世界开始搜索。如果这一步**没正确跳过 viz=1 槽位**,搜索的根节点对手手牌可能被随机覆盖——AI 的 Guard 出牌策略就完全用不上「我知道是 Princess」这条信息,对应的策略价值估计变成噪声。

测试位置：`tests/framework/test_ismcts_samples_respect_tracker.py`。规则是 **`tracker_claim != UNKNOWN_SENTINEL` ⇒ 任何 sample 的对应字段 == claim**;声称未知时 sample 可以是任何随机抽样结果。

```python
# tests/framework/test_ismcts_samples_respect_tracker.py 里加：
def _check_<game>(snap: dict, trial_state: dict) -> None:
    known = snap.get("<your_known_field>")  # e.g. known_hand
    if known is None:
        return
    for p, claim in enumerate(known):
        if claim == 0:                              # UNKNOWN sentinel
            continue
        sampled = trial_state["players"][p]["<truth_field>"]
        assert sampled == claim, (
            f"<game>: tracker says player {p} has {claim} but "
            f"randomize_unseen sampled {sampled}")

_CHECKERS["<game>"] = _check_<game>
```

`test_belief_tracker(...)` 已经返回 `belief_snapshot` 和 `trial_states[t]`(每次 `randomize_unseen` 后 serializer 输出的完整 GT-style state dict),不需要新增 binding。

跑 `pytest tests/framework/test_ismcts_samples_respect_tracker.py -v -k <game_id>`,所有种子都过。

**为什么这个和 8f-bis 是分开的两条**:它们覆盖的是同一段代码的两个相邻接口。`8f-bis` 测的是「tracker 通过观察事件得到的认知」是否和 GT 对得上;`8f-ter` 测的是「这个认知有没有被传递给搜索」。任何一个错——观察事件漏了,或者 randomize_unseen 实现里忘了应用 known——都会让 AI 在已经看见信息的情况下表现得像没看见。两条测试都通过,信息流才完整。

### 8g. web.json 配置（如适用）

如果你的游戏有独立的 web.json 配置（AI 难度覆盖、动作过滤、残局求解等），验证配置加载正确：

```python
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "platform"))

def test_web_config_loaded():
    from game_service.sessions import load_web_configs
    configs = load_web_configs()
    assert GAME_ID in configs

def test_difficulty_overrides_applied():
    """difficulty_overrides 应正确覆盖默认 preset。"""
    from game_service.sessions import WEB_CONFIGS, DIFFICULTY_PRESETS
    web_cfg = WEB_CONFIGS.get(GAME_ID, {})
    for diff_name in ["casual", "expert"]:
        overrides = web_cfg.get("difficulty_overrides", {}).get(diff_name, {})
        preset = DIFFICULTY_PRESETS[diff_name]
        sims = overrides.get("simulations", preset["simulations"])
        assert sims > 0
```

---

## 第 9 步：多人游戏变体（如适用）

如果你的游戏支持不同人数（如 2p/3p/4p），验证所有变体：

```python
import pytest

VARIANTS = ["your_game", "your_game_3p", "your_game_4p"]

@pytest.mark.parametrize("variant", VARIANTS)
def test_variant_registered(variant):
    assert variant in dinoboard_engine.available_games()

@pytest.mark.parametrize("variant", VARIANTS)
def test_variant_selfplay(variant):
    ep = dinoboard_engine.run_selfplay_episode(
        game_id=variant, seed=42, model_path="", simulations=10,
        max_game_plies=50,
    )
    assert ep["total_plies"] > 0

@pytest.mark.parametrize("variant", VARIANTS)
def test_variant_feature_dim(variant):
    """多人变体的 feature_dim 通常和 2p 不同——必须用 encoder 报告的实际值。"""
    info = dinoboard_engine.encode_state(variant, seed=42)
    ep = dinoboard_engine.run_selfplay_episode(
        game_id=variant, seed=42, model_path="", simulations=5,
        max_game_plies=50,
    )
    for s in ep["samples"]:
        assert len(s["features"]) == info["feature_dim"]

@pytest.mark.parametrize("variant", VARIANTS)
def test_variant_z_values_length(variant):
    """z_values 长度应等于该变体的 num_players。"""
    gs = dinoboard_engine.GameSession(variant, seed=42)
    expected = gs.num_players
    ep = dinoboard_engine.run_selfplay_episode(
        game_id=variant, seed=42, model_path="", simulations=5,
        max_game_plies=50,
    )
    for s in ep["samples"]:
        z_vals = s.get("z_values", [])
        if z_vals:
            assert len(z_vals) == expected

@pytest.mark.parametrize("variant", VARIANTS)
def test_variant_all_players_get_turns(variant):
    """所有玩家都应该有轮到的回合。"""
    gs = dinoboard_engine.GameSession(variant, seed=42)
    expected = gs.num_players
    seen = set()
    for _ in range(expected * 3):
        if gs.is_terminal:
            break
        seen.add(gs.current_player)
        gs.apply_action(gs.get_legal_actions()[0])
    assert len(seen) == expected
```

**踩坑参考**：KNOWN_ISSUES §通用踩坑 9（多人变体 feature_dim 和 2p 不同，不能从 game.json 读取）

---

## 第 10 步：AI API 分离验收（强制）

这是最后一道也是最强的闸门。前 9 步验证了游戏引擎内部的一致性；第 10 步验证 AI 决策链路**真的不依赖** ground truth 的 state。通过方式是把 AI 放到独立 session 里，只靠 action 序列 + public events 驱动，看它能否和自博弈维护的 belief 逐步对齐。

### 10.1 基础 API 测试（所有游戏必做）

在 `tests/<your_game>/test_checklist.py` 里加 API 分离断言(参考 `tests/quoridor/test_checklist.py` 或 `tests/loveletter/test_checklist.py`),然后:

```bash
python -m pytest tests/<your_game>/ -v -k api
```

如果你想让框架层 carrier 也覆盖你的游戏(项目维护层面的决定),可在 `tests/framework/test_ai_api_separation.py` 的 `_PLY_BUDGET` 里加一条。但**这不是新游戏接入的硬性要求**——`tests/<your_game>/` 全绿即合格。

此步验证：
- API 契约干净（无 state 进出）
- 给定同 seed 能端到端跑完一局
- 确定性游戏即使 seed 不同也能对齐（因为公开初始 state 与 seed 无关）

### 10.2 Belief 等价测试（随机游戏必做）

如果游戏有 `belief_tracker`（随机或信息不对称），必须实现 public-event 协议并通过 belief 等价测试。详见 GAME_DEVELOPMENT_GUIDE.md §17。

```bash
# 实现 public_event_extractor / public_state_applier /
# initial_observation_extractor / initial_observation_applier 并在 GameBundle 注册
# 实现 IBeliefTracker::serialize() 输出 canonical 字典
# 在 tests/<your_game>/test_checklist.py 里加 belief / public state / legal actions 三层等价断言
# (参考 tests/loveletter/test_checklist.py)
python -m pytest tests/<your_game>/ -v -k belief
```

三个断言：
- `test_api_belief_matches_selfplay` — 每步 belief snapshot 相等
- `test_api_public_state_matches_after_trace` — 终局公开 state 相等
- `test_api_legal_actions_match_after_trace` — perspective 回合的 legal actions 相等

### 10.2-bis. Public hash 不能含内部 RNG（强制，BUG-028 回归）

新游戏接入时**必须**让 `tests/framework/test_public_hash_excludes_internal_rng.py` 在你的 `game_id` 上跑过。把游戏 id 加到 `HIDDEN_INFO_GAMES` 列表里（如果是隐藏信息游戏；纯公开信息游戏结构上不会触发）。

```bash
python -m pytest tests/framework/test_public_hash_excludes_internal_rng.py -v -k <your_game>
```

测的是：相同 observation 历史下，两个内部 seed 不同的 `GameSession` 必须产生**字节相等**的 `state_hash_for_perspective(p)`。失败几乎 100% 意味着某个 `hash_field_slot` 在 `viz=1` 路径上读到了内部 RNG / 牌堆顺序 / 未揭示 deck 等任何"应该藏起来"的 slot——这就是 BUG-028。修法是把那个 slot 的 schema base viz 改成 `all_hidden`，或检查 viz tensor 是不是被错误地翻成了 1。

为什么这条单列：BUG-028 不会让任何已有测试失败，不会崩，只会让 search 变弱、selfplay 与 API 路径分裂、训练曲线悄悄垮掉。文档级提醒（CLAUDE.md, GAME_DEVELOPMENT_GUIDE §11.1b）防不住人，CI 测试才防得住。

### 10.3 常见失败

| 症状 | 可能原因 | 修复 |
|------|---------|------|
| belief 第一步就发散 | `initial_observation_extractor` 漏传某个 perspective 可见字段 | 对照 encoder 看 perspective 能看到什么，extractor 都要返回 |
| belief 中途发散（比如某动作之后） | 该动作的 event 没塞进 `events` 列表，或 payload 字段错 | 对照 `extract_events` 看 GT 端为这个动作发什么事件，tracker 端对照 `observe_public_event` 看怎么消费 |
| public state 发散但 belief 相等 | `public_snapshot` 漏了某个公开 slot，或 GT 端 schema 没把它声明成公开 | observer 不重放规则，公开字段全靠 snapshot 整体覆写——所有 viz 设为 public 的 slot 都必须出现在 snapshot 里 |
| `no matching hidden reserved slot found` 之类运行时错 | event payload 缺 player 或 slot 等关键字段 | 事件必须自包含，不能让 tracker 猜 |

---

## 第 11 步：规则不变量（强制）

**验证什么**:你这个游戏自身的守恒律。前 10 步验证的是「框架对你的游戏的接口契约」——状态合法、特征对齐、AI 不偷看、API 收敛。第 11 步反过来验证「**你写的 rules 真的实现了这个游戏的规则**」。

每个棋牌游戏都有自己的物理守恒律(token / 卡 / 棋子的总量),以及结构约束(图形可达、容量上限、角色配额)。只要随便走几手都能让这些守恒律破掉,说明规则实现里有 bug——而前面 10 步都看不出来,因为框架不在乎你算分对不对、卡是不是凭空出现。

**例子**:

| 游戏 | 守恒律例子 |
|---|---|
| Azul | 5 色 × 20 = 100 块瓷砖,任意时刻 `bag + box + factories + center + 玩家所有可见卡槽` ≤ 100;每个图案行 `length ≤ capacity` 且不能放已经在墙上的颜色 |
| Splendor | 5 色 token 数恒等于初始供应(2p=4, 3p=5, 4p=7),金色 token = 5;玩家保留卡数 ≤ 3;每个 tier 公开卡 ≤ 4 |
| Love Letter | 16 张卡总量恒定(`deck_size + 所有可见槽 == 16`),每个卡型不超过其总数(Guard×5/Princess×1 等);活着的玩家手牌 ≥ 1,死了的玩家手牌 = 0 |
| Coup | 5 角色 × 3 = 15 张卡总量恒定;每个玩家硬币 ∈ [0, 12],影响牌槽 = 2;活着的玩家未亮牌数 ≥ 1 |
| Quoridor | 已放墙 + 剩余墙 = 20;墙坐标合法范围;**两个棋子都仍能 BFS 到自己目标行**(围栏不能完全封死) |

**实现模式**:用 `conftest.run_random_episode_states(game_id, seed, max_plies)` helper 通过随机走子驱动整局,逐步取 `state_dict` 并跑断言。helper 简单可靠——任何不需要 model 的游戏都能用。

```python
from conftest import run_random_episode_states

def _assert_<game>_invariants(state: dict) -> None:
    # 所有 token / 卡守恒
    n = state["num_players"]
    bank = state["bank"]
    for color in range(5):
        in_play = bank[color] + sum(p["gems"][color] for p in state["players"])
        assert in_play == EXPECTED[n], (
            f"color {color} broken: bank={bank[color]} "
            f"players={[p['gems'][color] for p in state['players']]} "
            f"total={in_play}")
    # 玩家局部约束
    for i, p in enumerate(state["players"]):
        assert len(p["reserved"]) <= 3
        assert sum(p["bonuses"]) == p["cards_count"]
    # 终局/非终局一致性
    if not state["is_terminal"]:
        assert 0 <= state["current_player"] < n


class TestRuleInvariants:
    @pytest.mark.parametrize("seed", list(range(10)))
    def test_invariants_hold_along_random_episode(self, seed):
        for state in run_random_episode_states(GAME, seed=seed, max_plies=200):
            _assert_<game>_invariants(state)
```

**写好这一段的关键技巧**:

1. **从守恒律开始**——总数、初始供应、容量上限。这些是最本质、最容易被规则 bug 破坏的属性。
2. **断言用 `==` 不要轻易放宽到 `<=`**,除非你已经知道有合法的"丢弃路径"(例如 Azul 的 floor 满了之后会静默丢弃,所以总瓷砖数是 `<= 100` 而非 `==`)。**如果你写 `<=`,在注释里说清楚为什么放宽**——否则下次有人引入 bug 时这个测试就抓不住了。
3. **跨多个 seed 跑**——单 seed 可能正好走不到坏路径。10–20 个 seed 是个好默认值。
4. **如果某个守恒律需要序列化器暴露当前不暴露的字段**(例如 Azul 的 `box_lid`、Love Letter 的 `set_aside_card`),改 `*_register.cpp` 的 `serialize_<game>()` 把它加进去——同时在 C++ 注释里写明这是**给测试用的可见性**,belief tracker 和 encoder 都不能读它。这样既满足测试,又不破坏 AI 分离原则。

**踩坑参考**:这一步有些坑只有写过测试才会发现——Azul 的 floor 满了会丢瓷砖、Love Letter 的 `drawn_card` 只在当前玩家行动时才设置、Coup 的影响牌槽在 mid-action 阶段会临时变成 -1 等等。这些都是规则的合法行为,断言要相应放松。

---

## 快速运行

把以上测试保存为 `tests/<your_game>/test_checklist.py`（关于目录结构见下一节），然后运行：

```bash
# 全量验证你的清单
python -m pytest tests/<your_game>/ -v

# 快速冒烟
python -m pytest tests/<your_game>/ -v -k "registered or completes or determinism"

# 与你的游戏相关的全部测试（包括框架层用其他游戏跑的参数化用例）
python -m pytest tests/ -v -k "<your_game>"

# 全套自动化测试
python -m pytest tests/ -x -q
```

---

## 测试架构原则：两层测试

> **这一节是开发者必读。** 它解释了 `tests/` 为什么是当前这种布局，以及作为新游戏开发者你应该写什么、不该改什么。

`tests/` 目录分两层，对应**两个完全不同的读者**和**两种完全不同的失败语义**。

### 第一层：`tests/framework/` —— 框架不变量

**读者**：项目维护者（写 C++ 引擎、MCTS、ISMCTS、训练管线的人）。

**触发条件**：当你修改 `engine/`、`training/`、`platform/ai_service/` 等**框架代码**时，这一层告诉你"是不是把哪个游戏的什么基础假设破坏了"。

**测什么**：跨游戏参数化的不变量——do/undo 一致、ONNX round-trip、selfplay 样本完整、tracker 不偷看真值、encoder 不泄漏对手私信、DAG 无环、API 与 selfplay 行为一致 …… 这些都是框架对**任何已注册游戏**都必须成立的性质。

**承载游戏（matrix carrier）**：`FRAMEWORK_GAMES = ["quoridor", "azul", "loveletter"]`。这三个游戏一起最小完备地覆盖了框架关心的所有结构特征：

| 特征 | quoridor | azul | loveletter |
|---|---|---|---|
| 完全公开 + 确定 | ✓ | | |
| 物理随机（对称、无玩家私信） | | ✓ | |
| 信息不对称（有玩家私信） | | | ✓ |
| 多人变体 (3p/4p) | | ✓ | ✓ |
| 玩家淘汰 | | | ✓ |
| `tail_solver` | ✓ | | |
| `belief_tracker` | | ✓ | ✓ |
| schema 含 `owner_only_first_axis` / `all_hidden` 字段 | | | ✓ |

> **作为新游戏开发者，你不需要修改 `tests/framework/`。** 框架层用 `FRAMEWORK_GAMES` 这三个固定游戏来验证框架本身——这是项目维护者的工具。如果你的新游戏带来了一个**新的结构特征**（既不是 quoridor 也不是 azul/loveletter 的子集），再考虑是否需要把它加进 `FRAMEWORK_GAMES` 或某个子集列表（`FRAMEWORK_HIDDEN_INFO_GAMES`、`FRAMEWORK_TAIL_SOLVER_GAMES` 等），这是项目维护者的决定。

### 第二层：`tests/<game>/` —— 单游戏完备清单

**读者**：你，新游戏开发者。

**触发条件**：当你实现一个新游戏，或修改自己游戏的代码时，这一层告诉你"我的游戏作为一个完整的集成对象，是不是 ready 了"。

**测什么**：你这个游戏从注册到 selfplay 到 arena 到 web 配置的**端到端验收**。每个游戏一个 `tests/<game>/` 文件夹，里面是**这个游戏自己的完整测试清单**。

**关键性质：每个游戏的测试是各自独立的——故意冗余**。即使框架层已经用 quoridor/azul/loveletter 测过 do/undo 一致性，你的 `tests/splendor/test_checklist.py` 里**仍然要测 splendor 自己的 do/undo 一致性**。原因：

1. **本地化的失败信号**：当你修改 splendor 的 rules，splendor 的清单立刻全红，不需要去看一个跨 6 个游戏参数化的失败用例慢慢推断哪个步骤挂了。
2. **明确的"游戏完成"定义**：`pytest tests/<your_game>/` 全绿 = 你的游戏 ready。这是一个清晰的、对开发者可见的契约。
3. **复制成本极低，维护成本不高**：一个清单不到 300 行，且变化频率低（写完基本不动）。
4. **架构可读性**：新游戏的开发者打开 `tests/<some_game>/` 就能看到一个完整可执行的"模板"，远比"去框架层挖出 N 个参数化测试再过滤参数"清晰。

### 你的工作流

写一个新游戏 `myGame`：

1. 复制一份现有清单作为模板。选**与你的游戏特征最接近的那个**：
   - 完全公开 + 确定 → `tests/tictactoe/test_checklist.py` 或 `tests/quoridor/test_checklist.py`（如果你有 tail_solver / heuristic / filter）
   - 公开物理随机 → `tests/azul/test_checklist.py`
   - 信息不对称 → `tests/loveletter/test_checklist.py` 或 `tests/coup/test_checklist.py`
2. 把 `GAME = "myGame"` 改成你的游戏 id，删掉你不支持的可选组件分组（例如没有 `tail_solver` → 删掉 `TestTailSolver`，并加上 `TestUnsupportedComponents::test_no_tail_solver`），按需调整 simulations / max_game_plies。
3. 运行 `pytest tests/myGame/ -v` 直到全绿。
4. **顺手**跑一遍 `pytest tests/framework/` 看你的引擎改动有没有破坏框架不变量。如果你的游戏新增了一个真正全新的结构特征，告诉项目维护者——他们会决定是否更新 `FRAMEWORK_GAMES` 或某个子集列表。

### 为什么不在框架层一次性测全部 6 个游戏？

第一版做过这件事——`tests/test_*.py` 用 `CANONICAL_GAMES = [...]` 跑全部 6 个游戏。问题是：

- 某些不变量（例如 `test_dag_reuse_hits_present`）在不同游戏上的合理参数差很多（loveletter 几十次访问就能看到 reuse；splendor 要几百次），写成参数化反而需要 if/else 调阈值，难读、易错。
- 框架层每次跑都要把 6 个游戏全跑一遍，CI 慢，调试反馈链长。
- 单游戏的失败信号被淹没在 6 倍的测试中。

新架构里，框架层用最小完备的 3 个游戏跑，CI 快、信号干净；每个游戏自己的清单用最贴合该游戏的参数测自己——两层互补，互不淹没。

---

## Checklist 总结

| 步骤 | 验证点 | 对应 bug/踩坑 |
|------|--------|-------------|
| 1 | feature_dim / action_space 与 encoder 一致 | 通用踩坑 3 |
| 2 | GameSession 能创建、走棋、到终局 | 通用踩坑 8 |
| 3 | do/undo 完美逆操作（种子确定性 + 搜索后状态不变） | 通用踩坑 5 |
| 4 | features 逐步变化、不等于初始、维度正确 | **BUG-007** |
| 5 | 样本结构合法（visits=sims、z 与 winner 一致） | 样本完整性 |
| 6 | ONNX 导出后能被 C++ 加载用于 selfplay | ONNX shape 匹配 |
| 7 | 训练 tensor 有限、policy 和为 1、梯度不 NaN | **BUG-009** |
| 8a | heuristic 返回合法动作 | - |
| 8b | tail_solve 统计不变量 | BUG-001 |
| 8c | filter 是完整动作集的子集 | BUG-003 |
| 8d | adjudicator 在超时时判定胜负 | BUG-004 |
| 8e | auxiliary_score 有限 | DESIGN-001 |
| 8f | 隐藏信息：高 sim + 确定性 + DAG 复用 + 不偷看 | ISMCTS 正确性 + **BUG-017** |
| 8f-bis | tracker 已知信息 ↔ ground truth 一致 | tracker 应用事件错位 |
| 8f-ter | ISMCTS 根采样尊重 tracker 已知信息 | randomize_unseen 漏读 known |
| 8g | web.json 配置加载 + 难度覆盖 | - |
| 9 | 多人变体：注册、feature_dim、z_values、轮转 | **通用踩坑 9** |
| 10 | **AI API 分离测试**：端到端通过 HTTP API 驱动完整对局 | 见下方第 10 步 |
| 11 | **规则不变量**：游戏自己的守恒律(token / 卡 / 棋子总量、容量上限、可达性) | 规则实现错误的最便宜捕捉点 |
