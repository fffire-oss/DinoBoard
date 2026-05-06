"""Splendor complete acceptance checklist.

Splendor is a hidden-information game with:
  - belief_tracker (deck contents are hidden, but card pool is public)
  - tail_solver + tail_solve_trigger (>=12 points triggers solve)
  - multiplayer variants: 2p / 3p / 4p
  - uniform-random heuristic_picker (web 'heuristic' fallback)
  - no real heuristic, no training_filter, no adjudicator, no aux_scorer
"""
import dinoboard_engine
import pytest

from conftest import GAME_CONFIGS, get_test_model

GAME = "splendor"
ACTION_SPACE = GAME_CONFIGS[GAME]["action_space"]
FEATURE_DIM = GAME_CONFIGS[GAME]["feature_dim"]
VARIANTS = ["splendor", "splendor_3p", "splendor_4p"]


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
        assert dinoboard_engine.GameSession("splendor_3p", seed=42).num_players == 3
        assert dinoboard_engine.GameSession("splendor_4p", seed=42).num_players == 4


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

    def test_state_dict_has_basics(self):
        gs = dinoboard_engine.GameSession(GAME, seed=42)
        state = gs.get_state_dict()
        assert "current_player" in state
        assert "is_terminal" in state


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
            simulations=10, max_game_plies=50,
        )
        assert ep["total_plies"] > 0

    def test_sample_integrity(self):
        ep = dinoboard_engine.run_selfplay_episode(
            game_id=GAME, seed=42, model_path=get_test_model(GAME),
            simulations=10, max_game_plies=60,
        )
        for s in ep["samples"]:
            assert len(s["features"]) == FEATURE_DIM
            assert len(s["legal_mask"]) == ACTION_SPACE
            legal_set = {i for i, m in enumerate(s["legal_mask"]) if m > 0}
            visited = {a for a, v in zip(s["policy_action_ids"], s["policy_action_visits"]) if v > 0}
            assert visited <= legal_set, (
                f"ply {s['ply']}: visited illegal actions {visited - legal_set}"
            )

    def test_features_vary_across_plies(self):
        ep = dinoboard_engine.run_selfplay_episode(
            game_id=GAME, seed=42, model_path=get_test_model(GAME),
            simulations=10, max_game_plies=60,
        )
        if len(ep["samples"]) < 5:
            pytest.skip("too few samples")
        assert ep["samples"][0]["features"] != ep["samples"][4]["features"]

    def test_determinism_same_seed(self):
        kw = dict(
            game_id=GAME, seed=12345, model_path=get_test_model(GAME),
            simulations=10, max_game_plies=30,
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
                simulations=10, max_game_plies=30,
            )
            ep2 = dinoboard_engine.run_selfplay_episode(
                game_id=GAME, seed=i + 10000, model_path=get_test_model(GAME),
                simulations=10, max_game_plies=30,
            )
            if [s["action_id"] for s in ep1["samples"]] != [s["action_id"] for s in ep2["samples"]]:
                diff += 1
        assert diff > 10, f"only {diff}/20 splendor games differed across distant seeds"

    def test_high_sim_count_no_crash(self):
        ep = dinoboard_engine.run_selfplay_episode(
            game_id=GAME, seed=42, model_path=get_test_model(GAME),
            simulations=100, max_game_plies=30,
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
# 7. Tail solver
# ---------------------------------------------------------------------------

class TestTailSolver:

    def test_tail_solve_api(self):
        r = dinoboard_engine.tail_solve(
            game_id=GAME, seed=42, perspective_player=0,
            depth_limit=2, node_budget=1000,
        )
        assert "value" in r and "budget_exceeded" in r


# ---------------------------------------------------------------------------
# 8. Hidden info: belief tracker on deck
# ---------------------------------------------------------------------------

class TestBeliefTracker:

    def test_randomized_decks_differ_from_original(self):
        r = dinoboard_engine.test_belief_tracker(GAME, seed=42, plies=20, randomize_trials=10)
        assert r["plies"] > 0
        orig = sorted(r["original_deck"])
        diffs = sum(1 for td in r["trial_decks"] if sorted(td) != orig)
        assert diffs > 0, "All trials matched original deck — tracker may be peeking"

    def test_tableau_cards_never_in_randomized_deck(self):
        r = dinoboard_engine.test_belief_tracker(GAME, seed=123, plies=15, randomize_trials=10)
        tab = set(r["tableau_cards"])
        for i, td in enumerate(r["trial_decks"]):
            overlap = set(td) & tab
            assert not overlap, f"Trial {i}: tableau cards {overlap} in randomized deck"

    def test_deck_size_preserved(self):
        r = dinoboard_engine.test_belief_tracker(GAME, seed=99, plies=15, randomize_trials=10)
        orig_len = len(r["original_deck"])
        for i, td in enumerate(r["trial_decks"]):
            assert len(td) == orig_len, (
                f"Trial {i}: deck size {len(td)} != original {orig_len}"
            )

    def test_randomization_has_variance(self):
        r = dinoboard_engine.test_belief_tracker(GAME, seed=77, plies=10, randomize_trials=10)
        unique = {tuple(td) for td in r["trial_decks"]}
        assert len(unique) > 1, "All trials produced identical decks — no randomization"

    def test_no_duplicate_cards(self):
        r = dinoboard_engine.test_belief_tracker(GAME, seed=55, plies=15, randomize_trials=5)
        for i, td in enumerate(r["trial_decks"]):
            assert len(td) == len(set(td)), f"Trial {i}: duplicate cards {td}"
