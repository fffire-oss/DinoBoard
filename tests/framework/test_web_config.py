"""Tests for the web config layer post-mcts_profiles migration.

The old `WEB_CONFIGS` / `DIFFICULTY_PRESETS` / `difficulty_overrides` constants
are gone. All web-side MCTS knobs come from named profiles in
`games/<g>/config/web.json mcts_profiles` resolved via
`training.mcts_profile.resolve_profile`. The web `create_session` reads from:

  - difficulty == "heuristic" → no profile (no MCTS)
  - difficulty == "casual"    → profile "web_casual"
  - difficulty == "expert"    → profile "web_expert"
  - analysis sims             → profile "analysis"

This file covers (1) the resolver-side values for each web profile per game,
and (2) that `create_session` flows the resolved values into the session dict.
"""
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "platform"))

from training.mcts_profile import clear_cache, resolve_profile  # noqa: E402


@pytest.fixture(autouse=True)
def _reset_profile_cache():
    clear_cache()
    yield
    clear_cache()


# ---------------------------------------------------------------------------
# Resolver-side: web profiles exist and carry sensible values
# ---------------------------------------------------------------------------


class TestWebProfilesResolve:

    @pytest.mark.skipif(
        "coup" not in __import__("dinoboard_engine").available_games(),
        reason="coup not built (manifest disabled)",
    )
    def test_coup_web_casual_simulations(self):
        p = resolve_profile("coup", "web_casual")
        assert p.simulations == 50

    @pytest.mark.skipif(
        "coup" not in __import__("dinoboard_engine").available_games(),
        reason="coup not built (manifest disabled)",
    )
    def test_coup_web_casual_temperature(self):
        p = resolve_profile("coup", "web_casual")
        assert p.temperature == 0.3

    @pytest.mark.skipif(
        "coup" not in __import__("dinoboard_engine").available_games(),
        reason="coup not built (manifest disabled)",
    )
    def test_coup_web_expert_temperature(self):
        p = resolve_profile("coup", "web_expert")
        assert p.temperature == 0.1

    @pytest.mark.skipif(
        "coup" not in __import__("dinoboard_engine").available_games(),
        reason="coup not built (manifest disabled)",
    )
    def test_coup_analysis_simulations(self):
        p = resolve_profile("coup", "analysis")
        assert p.simulations == 2000

    def test_loveletter_web_expert_temperature(self):
        p = resolve_profile("loveletter", "web_expert")
        assert p.temperature == 0.1

    def test_loveletter_web_casual_temperature(self):
        p = resolve_profile("loveletter", "web_casual")
        assert p.temperature == 0.3

    def test_quoridor_web_expert_tail_solve_enabled(self):
        p = resolve_profile("quoridor", "web_expert")
        assert p.tail_solve_enabled is True
        assert p.tail_solve_depth_limit > 0
        assert p.tail_solve_node_budget > 0

    def test_azul_web_expert_tail_solve_enabled(self):
        p = resolve_profile("azul", "web_expert")
        assert p.tail_solve_enabled is True

    def test_splendor_web_expert_tail_solve_enabled(self):
        p = resolve_profile("splendor", "web_expert")
        assert p.tail_solve_enabled is True

    def test_tictactoe_web_expert_no_tail_solve(self):
        """Tictactoe has no tail_solve_trigger — its web_expert profile must
        keep tail_solve_enabled=False (resolver would otherwise raise)."""
        p = resolve_profile("tictactoe", "web_expert")
        assert p.tail_solve_enabled is False


# ---------------------------------------------------------------------------
# Session creation: resolved profile values flow into the session dict
# ---------------------------------------------------------------------------


class TestSessionCreation:

    @staticmethod
    def _ensure_test_model(game_id):
        """Create a random model if none exists, so create_session doesn't fail."""
        import re
        base = re.sub(r"_\d+p$", "", game_id)
        variant = game_id if game_id != base else f"{base}_2p"
        model_dir = PROJECT_ROOT / "games" / base / "model"
        model_path = model_dir / f"{variant}.onnx"
        if model_path.exists():
            return
        sys.path.insert(0, str(PROJECT_ROOT))
        from conftest import get_test_model
        import shutil
        src = get_test_model(game_id)
        model_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, model_path)

    def _create_session(self, game_id, difficulty):
        from game_service.sessions import create_session
        import random
        if difficulty != "heuristic":
            self._ensure_test_model(game_id)
        seed = random.randint(0, 2**31)
        session_id, sess = create_session(
            game_id=game_id, seed=seed,
            human_player=0, num_players=2, difficulty=difficulty,
        )
        return session_id, sess

    @pytest.mark.skipif(
        "coup" not in __import__("dinoboard_engine").available_games(),
        reason="coup not built (manifest disabled)",
    )
    def test_coup_casual_session_simulations(self):
        _, sess = self._create_session("coup", "casual")
        assert sess["simulations"] == 50

    @pytest.mark.skipif(
        "coup" not in __import__("dinoboard_engine").available_games(),
        reason="coup not built (manifest disabled)",
    )
    def test_coup_casual_session_temperature(self):
        _, sess = self._create_session("coup", "casual")
        assert sess["temperature"] == 0.3

    @pytest.mark.skipif(
        "coup" not in __import__("dinoboard_engine").available_games(),
        reason="coup not built (manifest disabled)",
    )
    def test_coup_expert_session_temperature(self):
        _, sess = self._create_session("coup", "expert")
        assert sess["temperature"] == 0.1

    @pytest.mark.skipif(
        "coup" not in __import__("dinoboard_engine").available_games(),
        reason="coup not built (manifest disabled)",
    )
    def test_coup_session_analysis_sims(self):
        _, sess = self._create_session("coup", "casual")
        assert sess["analysis_simulations"] == 2000

    def test_loveletter_expert_temperature(self):
        _, sess = self._create_session("loveletter", "expert")
        assert sess["temperature"] == 0.1

    def test_unknown_difficulty_raises(self):
        from game_service.sessions import create_session
        with pytest.raises(ValueError, match="unknown difficulty"):
            create_session("tictactoe", seed=1, human_player=0,
                           num_players=2, difficulty="impossible")

    def test_heuristic_session_no_model(self):
        _, sess = self._create_session("loveletter", "heuristic")
        assert sess["use_model"] is False
