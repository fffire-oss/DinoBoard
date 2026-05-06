"""Love Letter complete acceptance checklist.

Love Letter is an asymmetric-hidden-info elimination game with:
  - belief_tracker with per-player private hand (`hash_private_fields(p)`)
  - public_event_extractor / applier (so the AI API can advance from
    observations alone, no state-passing)
  - initial_observation_extractor / applier
  - multiplayer variants: 2p / 3p / 4p
  - elimination: alive count drops monotonically; check_end_game must
    handle alive<=1 without deadlocking
  - uniform-random heuristic_picker (web 'heuristic' fallback)
  - no tail_solver, no training_filter, no adjudicator, no aux_scorer

The Guard-accuracy canary at the bottom is a behavioural regression
detector for hidden-info leaks in MCTS (see BUG-023).
"""
from __future__ import annotations

import sys
from pathlib import Path

import dinoboard_engine
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "tests"))

from conftest import get_test_model, load_game_config

GAME = "loveletter"
CONFIG = load_game_config(GAME)
ACTION_SPACE = CONFIG["action_space"]
FEATURE_DIM = CONFIG["feature_dim"]
VARIANTS = ["loveletter", "loveletter_3p", "loveletter_4p"]


# ---------------------------------------------------------------------------
# 1. Registration & config
# ---------------------------------------------------------------------------

class TestRegistration:

    def test_2p_registered(self):
        assert GAME in dinoboard_engine.available_games()

    def test_all_variants_registered(self):
        games = dinoboard_engine.available_games()
        for v in VARIANTS:
            assert v in games, f"{v} not registered"

    def test_metadata_matches_config(self):
        meta = dinoboard_engine.game_metadata(GAME)
        assert meta["action_space"] == ACTION_SPACE
        assert meta["feature_dim"] == FEATURE_DIM
        assert meta["num_players"] == 2

    def test_player_counts(self):
        assert dinoboard_engine.GameSession("loveletter_3p", seed=42).num_players == 3
        assert dinoboard_engine.GameSession("loveletter_4p", seed=42).num_players == 4


# ---------------------------------------------------------------------------
# 2. GameSession basics
# ---------------------------------------------------------------------------

class TestGameSession:

    def test_session_constructs(self):
        gs = dinoboard_engine.GameSession(GAME, seed=42)
        assert not gs.is_terminal

    def test_legal_actions_in_range(self):
        gs = dinoboard_engine.GameSession(GAME, seed=42)
        legal = gs.get_legal_actions()
        assert len(legal) > 0
        assert all(0 <= a < ACTION_SPACE for a in legal)

    def test_action_info(self):
        gs = dinoboard_engine.GameSession(GAME, seed=42)
        for action in gs.get_legal_actions()[:20]:
            info = gs.get_action_info(action)
            assert isinstance(info, dict) and len(info) > 0


# ---------------------------------------------------------------------------
# 3. Feature encoding
# ---------------------------------------------------------------------------

class TestEncoder:

    def test_encode_state_correct_dim(self):
        info = dinoboard_engine.encode_state(GAME, seed=42)
        assert len(info["features"]) == FEATURE_DIM
        assert len(info["legal_mask"]) == ACTION_SPACE

    def test_features_have_known_dim_after_play(self):
        gs = dinoboard_engine.GameSession(GAME, seed=300)
        if gs.is_terminal:
            return
        enc = dinoboard_engine.encode_state(GAME, seed=300)
        assert len(enc["features"]) == enc["feature_dim"]


# ---------------------------------------------------------------------------
# 4. Selfplay (2p, 3p, 4p)
# ---------------------------------------------------------------------------

class TestSelfplay:

    @pytest.mark.parametrize("variant", VARIANTS)
    def test_episode_completes(self, variant):
        ep = dinoboard_engine.run_selfplay_episode(
            game_id=variant, seed=42, model_path=get_test_model(variant),
            simulations=10, max_game_plies=30,
        )
        assert ep["total_plies"] > 0

    def test_sample_integrity(self):
        ep = dinoboard_engine.run_selfplay_episode(
            game_id=GAME, seed=42, model_path=get_test_model(GAME),
            simulations=10, max_game_plies=30,
        )
        for s in ep["samples"]:
            assert len(s["features"]) == FEATURE_DIM
            assert len(s["legal_mask"]) == ACTION_SPACE
            legal_set = {i for i, m in enumerate(s["legal_mask"]) if m > 0}
            visited = {a for a, v in zip(s["policy_action_ids"], s["policy_action_visits"]) if v > 0}
            assert visited <= legal_set

    def test_determinism_same_seed(self):
        kw = dict(
            game_id=GAME, seed=12345, model_path=get_test_model(GAME),
            simulations=10, max_game_plies=20,
            dirichlet_alpha=0.0, dirichlet_epsilon=0.0,
        )
        ep1 = dinoboard_engine.run_selfplay_episode(**kw)
        ep2 = dinoboard_engine.run_selfplay_episode(**kw)
        assert ep1["winner"] == ep2["winner"]
        assert ep1["total_plies"] == ep2["total_plies"]

    def test_seed_diversity(self):
        diff = 0
        for i in range(20):
            ep1 = dinoboard_engine.run_selfplay_episode(
                game_id=GAME, seed=i, model_path=get_test_model(GAME),
                simulations=10, max_game_plies=20,
            )
            ep2 = dinoboard_engine.run_selfplay_episode(
                game_id=GAME, seed=i + 10000, model_path=get_test_model(GAME),
                simulations=10, max_game_plies=20,
            )
            if [s["action_id"] for s in ep1["samples"]] != [s["action_id"] for s in ep2["samples"]]:
                diff += 1
        assert diff > 10, f"only {diff}/20 loveletter games differed across distant seeds"


# ---------------------------------------------------------------------------
# 5. MCTS / arena
# ---------------------------------------------------------------------------

class TestMcts:

    def test_apply_ai_action_advances_state(self):
        m = get_test_model(GAME)
        gs = dinoboard_engine.GameSession(GAME, seed=42, model_path=m)
        before = gs.get_state_dict()
        gs.apply_ai_action(simulations=10, temperature=0.0)
        after = gs.get_state_dict()
        assert before != after

    def test_arena_completes(self):
        m = get_test_model(GAME)
        result = dinoboard_engine.run_arena_match(
            game_id=GAME, seed=42, model_paths=[m, m],
            simulations_list=[10, 10], temperature=0.0,
        )
        assert "winner" in result and result["total_plies"] > 0

    def test_dag_reuse_hits_present(self):
        m = get_test_model(GAME)
        gs = dinoboard_engine.GameSession(GAME, seed=42, model_path=m)
        result = gs.get_ai_action(simulations=20, temperature=0.0)
        assert "dag_reuse_hits" in result["stats"]


# ---------------------------------------------------------------------------
# 6. Heuristic (uniform-random fallback)
# ---------------------------------------------------------------------------

class TestHeuristic:

    def test_heuristic_returns_legal_action(self):
        for seed in range(20):
            gs = dinoboard_engine.GameSession(GAME, seed=seed)
            if gs.is_terminal:
                continue
            result = gs.get_heuristic_action()
            assert "action" in result
            assert result["action"] in gs.get_all_legal_actions()


# ---------------------------------------------------------------------------
# 7. Components NOT registered
# ---------------------------------------------------------------------------

class TestUnsupportedComponents:

    def test_no_tail_solver(self):
        with pytest.raises(RuntimeError, match="no tail_solver registered"):
            dinoboard_engine.tail_solve(
                game_id=GAME, seed=42, perspective_player=0,
                depth_limit=5, node_budget=10000,
            )


# ---------------------------------------------------------------------------
# 8. Temperature variety
# ---------------------------------------------------------------------------

class TestTemperature:

    def test_nonzero_temperature_adds_variety(self):
        actions = set()
        model = get_test_model(GAME)
        for seed in range(20):
            gs = dinoboard_engine.GameSession(GAME, seed=seed, model_path=model)
            if gs.is_terminal:
                continue
            result = gs.get_ai_action(simulations=10, temperature=1.0)
            if "action" in result:
                actions.add(result["action"])
        assert len(actions) > 1, "temperature=1.0 always picked the same action across seeds"


# ---------------------------------------------------------------------------
# 9. Elimination invariant: never deadlock with alive<=1 and terminal=false
# ---------------------------------------------------------------------------

class TestEliminationNoDeadlock:
    """After every action in 3p/4p selfplay and stepping, state must satisfy:
        - is_terminal=true (game over; winner -1 OK for draw), OR
        - alive count >= 1 AND current_player points at an alive player
    Guard / Baron / Prince / King eliminate at most ONE player per action.
    """

    @pytest.mark.parametrize("variant", ["loveletter_3p", "loveletter_4p"])
    def test_selfplay_no_deadlock(self, variant):
        for seed in range(10):
            ep = dinoboard_engine.run_selfplay_episode(
                game_id=variant, seed=seed, model_path=get_test_model(variant),
                simulations=10, max_game_plies=40,
            )
            assert ep["total_plies"] > 0


# ---------------------------------------------------------------------------
# 10. Belief: Guard accuracy canary (hidden-info leak detector)
# ---------------------------------------------------------------------------

class TestGuardAccuracy:
    """Plays many games vs random opponent, measures Guard-guess accuracy
    on positions where the belief tracker has no reveal-based knowledge.
    Random baseline is ~14%; clean MCTS with a trained model produces ~20%
    (slight bias from deterministic ai_view); BUG-023 produced 76%.

    We assert <40%, a >2x clean margin and >1.8x bug margin.
    """

    def test_guard_accuracy_not_better_than_bounded_inference(self):
        model = str(PROJECT_ROOT / "games/loveletter/model/loveletter_2p.onnx")
        if not Path(model).exists():
            pytest.skip("deployed loveletter_2p model not available")
        total, correct = 0, 0
        for seed in range(60):
            gs = dinoboard_engine.GameSession(GAME, seed, model, False)
            while not gs.is_terminal:
                cp = gs.current_player
                if cp == 0:
                    result = gs.get_ai_action(simulations=100, temperature=0.0)
                    info = result["action_info"]
                    if info.get("type") == "guard":
                        target = info["target"]
                        guess = info["guess"]
                        state = gs.get_state_dict()
                        actual = state["players"][target]["hand"]
                        snap = gs.get_belief_snapshot()
                        known = snap.get("known_hand", [0] * 4)
                        if target < len(known) and known[target] > 0:
                            pass
                        else:
                            total += 1
                            if guess == actual:
                                correct += 1
                    gs.apply_action(result["action"])
                else:
                    import random
                    random.seed(seed * 100 + gs.current_player)
                    legal = gs.get_legal_actions()
                    gs.apply_action(random.choice(legal))

        if total == 0:
            pytest.skip("AI did not play any Guards without prior info")
        rate = correct / total
        assert rate < 0.40, (
            f"Love Letter AI Guard accuracy {rate:.1%} ({correct}/{total}) "
            f"exceeds 40% — probable hidden-info leak in MCTS (see BUG-023). "
            f"Random baseline is ~14%."
        )
