"""Azul complete acceptance checklist.

Azul is a public-information game with physical (symmetric) randomness:
the bag is shuffled but every player sees what's drawn into factories.

  - belief_tracker: randomize_unseen shuffles the bag; no per-player
    private fields (so hash_private_fields can stay empty)
  - multiplayer variants: 2p / 3p / 4p
  - uniform-random heuristic_picker (web 'heuristic' fallback)
  - no tail_solver, no training_filter, no adjudicator, no aux_scorer
"""
import dinoboard_engine
import pytest

from conftest import get_test_model, load_game_config

GAME = "azul"
CONFIG = load_game_config(GAME)
ACTION_SPACE = CONFIG["action_space"]
FEATURE_DIM = CONFIG["feature_dim"]
VARIANTS = ["azul", "azul_3p", "azul_4p"]


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
        assert dinoboard_engine.GameSession("azul_3p", seed=42).num_players == 3
        assert dinoboard_engine.GameSession("azul_4p", seed=42).num_players == 4


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

    def test_state_dict_valid(self):
        gs = dinoboard_engine.GameSession(GAME, seed=42)
        state = gs.get_state_dict()
        assert isinstance(state, dict)
        assert state["current_player"] >= 0


# ---------------------------------------------------------------------------
# 3. Feature encoding
# ---------------------------------------------------------------------------

class TestEncoder:

    def test_encode_state_correct_dim(self):
        info = dinoboard_engine.encode_state(GAME, seed=42)
        assert len(info["features"]) == FEATURE_DIM
        assert len(info["legal_mask"]) == ACTION_SPACE


# ---------------------------------------------------------------------------
# 4. Selfplay (2p, 3p, 4p)
# ---------------------------------------------------------------------------

class TestSelfplay:

    @pytest.mark.parametrize("variant", VARIANTS)
    def test_episode_completes(self, variant):
        ep = dinoboard_engine.run_selfplay_episode(
            game_id=variant, seed=42, model_path=get_test_model(variant),
            simulations=5, max_game_plies=100,
        )
        assert ep["total_plies"] > 0

    def test_sample_integrity(self):
        ep = dinoboard_engine.run_selfplay_episode(
            game_id=GAME, seed=42, model_path=get_test_model(GAME),
            simulations=10, max_game_plies=100,
        )
        for s in ep["samples"]:
            assert len(s["features"]) == FEATURE_DIM
            assert len(s["legal_mask"]) == ACTION_SPACE
            legal_set = {i for i, m in enumerate(s["legal_mask"]) if m > 0}
            visited = {a for a, v in zip(s["policy_action_ids"], s["policy_action_visits"]) if v > 0}
            assert visited <= legal_set

    def test_features_vary_across_plies(self):
        ep = dinoboard_engine.run_selfplay_episode(
            game_id=GAME, seed=42, model_path=get_test_model(GAME),
            simulations=10, max_game_plies=100,
        )
        if len(ep["samples"]) < 5:
            pytest.skip("too few samples")
        assert ep["samples"][0]["features"] != ep["samples"][4]["features"]

    def test_determinism_same_seed(self):
        kw = dict(
            game_id=GAME, seed=999, model_path=get_test_model(GAME),
            simulations=10, max_game_plies=50,
            dirichlet_alpha=0.0, dirichlet_epsilon=0.0,
        )
        ep1 = dinoboard_engine.run_selfplay_episode(**kw)
        ep2 = dinoboard_engine.run_selfplay_episode(**kw)
        assert ep1["winner"] == ep2["winner"]
        assert ep1["total_plies"] == ep2["total_plies"]

    def test_high_sim_count_no_crash(self):
        ep = dinoboard_engine.run_selfplay_episode(
            game_id=GAME, seed=42, model_path=get_test_model(GAME),
            simulations=100, max_game_plies=50,
        )
        assert ep["total_plies"] > 0


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
# 7. Components NOT registered for azul
# ---------------------------------------------------------------------------

class TestUnsupportedComponents:

    def test_no_tail_solver(self):
        with pytest.raises(RuntimeError, match="no tail_solver registered"):
            dinoboard_engine.tail_solve(
                game_id=GAME, seed=42, perspective_player=0,
                depth_limit=5, node_budget=10000,
            )
