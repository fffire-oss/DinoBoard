"""Tic-Tac-Toe complete acceptance checklist.

This file is the single-game acceptance test for tictactoe written from a
game developer's perspective. It deliberately overlaps with framework
tests in tests/framework/ — that's by design. A new game is "complete"
when it passes its own checklist end-to-end without depending on
framework parametrization to surface the failure for it.

Tic-Tac-Toe is the simplest carrier:
  - fully observable, deterministic
  - 2 players only, no multiplayer variants
  - no tail_solver, no training_filter, no adjudicator, no aux_scorer,
    no belief_tracker
  - has a uniform-random heuristic_picker (for the web 'heuristic'
    difficulty fallback)
"""
import dinoboard_engine
import pytest

from conftest import GAME_CONFIGS, get_test_model

GAME = "tictactoe"
ACTION_SPACE = GAME_CONFIGS[GAME]["action_space"]
FEATURE_DIM = GAME_CONFIGS[GAME]["feature_dim"]


# ---------------------------------------------------------------------------
# 1. Registration & config
# ---------------------------------------------------------------------------

class TestRegistration:

    def test_game_registered(self):
        assert GAME in dinoboard_engine.available_games()

    def test_metadata_matches_config(self):
        meta = dinoboard_engine.game_metadata(GAME)
        assert meta["action_space"] == ACTION_SPACE
        assert meta["feature_dim"] == FEATURE_DIM
        assert meta["num_players"] == 2


# ---------------------------------------------------------------------------
# 2. GameSession basics
# ---------------------------------------------------------------------------

class TestGameSession:

    def test_session_constructs(self):
        gs = dinoboard_engine.GameSession(GAME, seed=42)
        assert gs.num_players == 2
        assert not gs.is_terminal

    def test_legal_actions_nonempty(self):
        gs = dinoboard_engine.GameSession(GAME, seed=42)
        legal = gs.get_legal_actions()
        assert len(legal) == 9  # empty board
        assert all(0 <= a < ACTION_SPACE for a in legal)

    def test_action_info_for_each_legal_action(self):
        gs = dinoboard_engine.GameSession(GAME, seed=42)
        for action in gs.get_legal_actions():
            info = gs.get_action_info(action)
            assert isinstance(info, dict) and len(info) > 0

    def test_apply_action_advances_state(self):
        gs = dinoboard_engine.GameSession(GAME, seed=42)
        before = gs.get_state_dict()
        gs.apply_action(gs.get_legal_actions()[0])
        after = gs.get_state_dict()
        assert before != after


# ---------------------------------------------------------------------------
# 3. Feature encoding
# ---------------------------------------------------------------------------

class TestEncoder:

    def test_encode_state_returns_correct_dim(self):
        info = dinoboard_engine.encode_state(GAME, seed=42)
        assert len(info["features"]) == FEATURE_DIM
        assert len(info["legal_mask"]) == ACTION_SPACE


# ---------------------------------------------------------------------------
# 4. Selfplay
# ---------------------------------------------------------------------------

class TestSelfplay:

    def test_episode_completes(self):
        ep = dinoboard_engine.run_selfplay_episode(
            game_id=GAME, seed=42, model_path=get_test_model(GAME),
            simulations=30, max_game_plies=20,
        )
        assert ep["total_plies"] > 0
        assert ep["total_plies"] <= 9, "tictactoe must terminate within 9 plies"

    def test_winner_always_valid(self):
        for seed in range(50):
            ep = dinoboard_engine.run_selfplay_episode(
                game_id=GAME, seed=seed, model_path=get_test_model(GAME),
                simulations=10, max_game_plies=9,
            )
            w = ep["winner"]
            assert w in (-1, 0, 1), f"seed {seed}: invalid winner {w}"

    def test_draw_possible(self):
        draws = sum(
            1 for seed in range(200)
            if dinoboard_engine.run_selfplay_episode(
                game_id=GAME, seed=seed, model_path=get_test_model(GAME),
                simulations=10, max_game_plies=9,
            )["draw"]
        )
        assert draws > 0, "no draws in 200 tictactoe games — encoder/rules likely broken"

    def test_sample_integrity(self):
        ep = dinoboard_engine.run_selfplay_episode(
            game_id=GAME, seed=42, model_path=get_test_model(GAME),
            simulations=10, max_game_plies=20,
        )
        for s in ep["samples"]:
            assert len(s["features"]) == FEATURE_DIM
            assert len(s["legal_mask"]) == ACTION_SPACE
            legal_set = {i for i, m in enumerate(s["legal_mask"]) if m > 0}
            visited = {a for a, v in zip(s["policy_action_ids"], s["policy_action_visits"]) if v > 0}
            assert visited <= legal_set, (
                f"ply {s['ply']}: visited illegal actions {visited - legal_set}"
            )

    def test_determinism_same_seed(self):
        kw = dict(
            game_id=GAME, seed=99, model_path=get_test_model(GAME),
            simulations=10, max_game_plies=9,
            dirichlet_alpha=0.0, dirichlet_epsilon=0.0,
        )
        ep1 = dinoboard_engine.run_selfplay_episode(**kw)
        ep2 = dinoboard_engine.run_selfplay_episode(**kw)
        assert ep1["winner"] == ep2["winner"]
        assert ep1["total_plies"] == ep2["total_plies"]


# ---------------------------------------------------------------------------
# 5. MCTS / arena
# ---------------------------------------------------------------------------

class TestMcts:

    def test_apply_ai_action_changes_state(self):
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


# ---------------------------------------------------------------------------
# 6. Heuristic (uniform-random fallback)
# ---------------------------------------------------------------------------

class TestHeuristic:

    def test_heuristic_returns_legal_action(self):
        for seed in range(20):
            gs = dinoboard_engine.GameSession(GAME, seed=seed)
            result = gs.get_heuristic_action()
            assert "action" in result
            assert result["action"] in gs.get_all_legal_actions()


# ---------------------------------------------------------------------------
# 7. Components NOT registered for tictactoe
# ---------------------------------------------------------------------------

class TestUnsupportedComponents:

    def test_no_tail_solver(self):
        with pytest.raises(RuntimeError, match="no tail_solver registered"):
            dinoboard_engine.tail_solve(
                game_id=GAME, seed=42, perspective_player=0,
                depth_limit=12, node_budget=1000000,
            )
