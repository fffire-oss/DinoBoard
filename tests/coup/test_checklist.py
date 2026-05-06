"""Coup complete acceptance checklist.

Coup is asymmetric-hidden-info with bluffing. It exercises:
  - belief_tracker with per-player private hand AND a heuristic-weighted
    sampler (claim/challenge history biases opp role priors)
  - public_event_extractor / applier
  - initial_observation_extractor / applier
  - elimination + multiplayer (2p / 3p / 4p)
  - encoder must zero out opponent known_hand block
  - uniform-random heuristic_picker (web 'heuristic' fallback)
  - no tail_solver, no training_filter, no adjudicator, no aux_scorer

Build status: Coup is temporarily disabled at the build level (see
CMakeLists.txt / setup.py / docs/GAME_FEATURES_OVERVIEW.md "诈唬核心游戏"
Future Work). The whole file skips gracefully when the engine wasn't
built with Coup. Once the build is restored, all tests should run.
"""
import dinoboard_engine
import pytest

from conftest import (
    assert_api_belief_matches_selfplay,
    get_test_model,
    load_game_config,
)


def _coup_available() -> bool:
    try:
        dinoboard_engine.game_metadata("coup")
        return True
    except Exception:
        return False


# Skip the whole file if Coup is not built — applied per-class below for
# clear reasons in test output.
coup_skip = pytest.mark.skipif(
    not _coup_available(), reason="Coup temporarily disabled in build")


GAME = "coup"
VARIANTS = ["coup", "coup_3p", "coup_4p"]


# ---------------------------------------------------------------------------
# 1. Registration & config
# ---------------------------------------------------------------------------

@coup_skip
class TestRegistration:

    def test_2p_registered(self):
        assert GAME in dinoboard_engine.available_games()

    def test_all_variants_registered(self):
        games = dinoboard_engine.available_games()
        for v in VARIANTS:
            assert v in games, f"{v} not registered"

    def test_metadata_matches_config(self):
        meta = dinoboard_engine.game_metadata(GAME)
        cfg = load_game_config(GAME)
        assert meta["action_space"] == cfg["action_space"]
        assert meta["feature_dim"] == cfg["feature_dim"]
        assert meta["num_players"] == 2

    def test_player_counts(self):
        assert dinoboard_engine.GameSession("coup_3p", seed=42).num_players == 3
        assert dinoboard_engine.GameSession("coup_4p", seed=42).num_players == 4


# ---------------------------------------------------------------------------
# 2. GameSession basics
# ---------------------------------------------------------------------------

@coup_skip
class TestGameSession:

    def test_session_constructs(self):
        gs = dinoboard_engine.GameSession(GAME, seed=42)
        assert not gs.is_terminal

    def test_legal_actions_in_range(self):
        gs = dinoboard_engine.GameSession(GAME, seed=42)
        legal = gs.get_legal_actions()
        action_space = load_game_config(GAME)["action_space"]
        assert len(legal) > 0
        assert all(0 <= a < action_space for a in legal)

    def test_action_info(self):
        gs = dinoboard_engine.GameSession(GAME, seed=42)
        for action in gs.get_legal_actions()[:20]:
            info = gs.get_action_info(action)
            assert isinstance(info, dict) and len(info) > 0


# ---------------------------------------------------------------------------
# 3. Feature encoding (must zero opponent's known_hand)
# ---------------------------------------------------------------------------

@coup_skip
class TestEncoder:
    """Coup feature layout per player (18 features):
      [0] alive
      [1] coins/12
      [2-3] influence_count (2)
      [4-8] revealed_cards (5, one per character)
      [9-13] known_hand (5, one per character) ← must be all-zero for opponents
      [14-17] is_active, is_target, is_blocker, is_challenger
    """
    FEATURES_PER_PLAYER = 18
    KNOWN_HAND_OFFSET = 9
    KNOWN_HAND_SIZE = 5

    def test_encode_state_correct_dim(self):
        cfg = load_game_config(GAME)
        info = dinoboard_engine.encode_state(GAME, seed=42)
        assert len(info["features"]) == cfg["feature_dim"]
        assert len(info["legal_mask"]) == cfg["action_space"]

    def test_opponent_known_hand_is_zero(self):
        enc = dinoboard_engine.encode_state(GAME, seed=100)
        f = enc["features"]
        opp_start = self.FEATURES_PER_PLAYER
        opp_hand = f[
            opp_start + self.KNOWN_HAND_OFFSET :
            opp_start + self.KNOWN_HAND_OFFSET + self.KNOWN_HAND_SIZE
        ]
        assert all(v == 0.0 for v in opp_hand), (
            f"Opponent known_hand should be all-zero, got {opp_hand}"
        )

    def test_self_known_hand_nonzero(self):
        enc = dinoboard_engine.encode_state(GAME, seed=100)
        f = enc["features"]
        self_hand = f[
            self.KNOWN_HAND_OFFSET :
            self.KNOWN_HAND_OFFSET + self.KNOWN_HAND_SIZE
        ]
        assert any(v > 0.0 for v in self_hand), (
            f"Self known_hand should be non-zero, got {self_hand}"
        )

    def test_perspectives_hide_different_info(self):
        gs = dinoboard_engine.GameSession(GAME, seed=200)
        gs.apply_action(gs.get_legal_actions()[0])
        enc = dinoboard_engine.encode_state(GAME, seed=200)
        f = enc["features"]
        opp_start = self.FEATURES_PER_PLAYER
        p0_self = f[self.KNOWN_HAND_OFFSET : self.KNOWN_HAND_OFFSET + self.KNOWN_HAND_SIZE]
        p0_opp = f[opp_start + self.KNOWN_HAND_OFFSET :
                   opp_start + self.KNOWN_HAND_OFFSET + self.KNOWN_HAND_SIZE]
        assert all(v == 0.0 for v in p0_opp)
        assert p0_self != p0_opp


# ---------------------------------------------------------------------------
# 4. Selfplay (2p, 3p, 4p)
# ---------------------------------------------------------------------------

@coup_skip
class TestSelfplay:

    @pytest.mark.parametrize("variant", VARIANTS)
    def test_episode_completes(self, variant):
        ep = dinoboard_engine.run_selfplay_episode(
            game_id=variant, seed=42, model_path=get_test_model(variant),
            simulations=10, max_game_plies=60,
        )
        assert ep["total_plies"] > 0

    def test_sample_integrity(self):
        cfg = load_game_config(GAME)
        ep = dinoboard_engine.run_selfplay_episode(
            game_id=GAME, seed=42, model_path=get_test_model(GAME),
            simulations=10, max_game_plies=60,
        )
        for s in ep["samples"]:
            assert len(s["features"]) == cfg["feature_dim"]
            assert len(s["legal_mask"]) == cfg["action_space"]
            legal_set = {i for i, m in enumerate(s["legal_mask"]) if m > 0}
            visited = {a for a, v in zip(s["policy_action_ids"], s["policy_action_visits"]) if v > 0}
            assert visited <= legal_set

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
        assert diff > 10, f"only {diff}/20 coup games differed across distant seeds"


# ---------------------------------------------------------------------------
# 5. MCTS / arena
# ---------------------------------------------------------------------------

@coup_skip
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

@coup_skip
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
# 7. Temperature variety
# ---------------------------------------------------------------------------

@coup_skip
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
# 8. Components NOT registered
# ---------------------------------------------------------------------------

@coup_skip
class TestUnsupportedComponents:

    def test_no_tail_solver(self):
        with pytest.raises(RuntimeError, match="no tail_solver registered"):
            dinoboard_engine.tail_solve(
                game_id=GAME, seed=42, perspective_player=0,
                depth_limit=5, node_budget=10000,
            )


# ---------------------------------------------------------------------------
# 9. AI API belief / public state / legal actions equivalence
# (independent-seed API session must match self-play tracker step-by-step)
# ---------------------------------------------------------------------------

@coup_skip
class TestApiBeliefEquivalence:

    PUBLIC_KEYS = [
        "current_player", "is_terminal", "winner", "num_players",
        "ply", "stage", "active_player", "declared_action",
        "action_target", "blocker", "challenger", "deck_size",
    ]

    def test_belief_matches_selfplay_under_independent_seed(self):
        assert_api_belief_matches_selfplay(GAME, self.PUBLIC_KEYS)
