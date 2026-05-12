"""Verify training config parameters actually reach the C++ engine.

Post-`mcts_profiles` migration: this file covers two layers.

  - Engine-passthrough: the bindings honor the kwargs we send (tail_solve,
    heuristic guidance, training filter, simulations).
  - Resolver behavior: every (game, profile) combination resolves cleanly,
    illegal configs raise, and known-bad shapes are rejected.
"""
from __future__ import annotations

import json
from pathlib import Path

import dinoboard_engine
import pytest

from conftest import PROJECT_ROOT, enabled_games, get_test_model, load_game_config, run_short_selfplay
from training.mcts_profile import (
    clear_cache,
    max_game_plies,
    resolve_profile,
)


# ─── Engine-passthrough tests ────────────────────────────────────────────────


def test_tail_solve_disabled_produces_no_stats_quoridor():
    ep = dinoboard_engine.run_selfplay_episode(
        game_id="quoridor", seed=42, model_path=get_test_model("quoridor"), simulations=10,
        max_game_plies=200, tail_solve_enabled=False,
    )
    assert ep["tail_solve_attempts"] == 0


def test_tail_solve_enabled_has_valid_stats_quoridor():
    ep = dinoboard_engine.run_selfplay_episode(
        game_id="quoridor", seed=42, model_path=get_test_model("quoridor"), simulations=10,
        max_game_plies=200, tail_solve_enabled=True,
        tail_solve_depth_limit=3, tail_solve_node_budget=1000,
    )
    a = ep["tail_solve_attempts"]
    c = ep["tail_solve_completed"]
    s = ep["tail_solve_successes"]
    assert s <= c <= a, f"invariant violated: {s} <= {c} <= {a}"


def test_tail_solve_tiny_budget_has_more_exceeded():
    ep_tiny = dinoboard_engine.run_selfplay_episode(
        game_id="quoridor", seed=42, model_path=get_test_model("quoridor"), simulations=10,
        max_game_plies=50, tail_solve_enabled=True,
        tail_solve_depth_limit=3, tail_solve_node_budget=10,
    )
    ep_large = dinoboard_engine.run_selfplay_episode(
        game_id="quoridor", seed=42, model_path=get_test_model("quoridor"), simulations=10,
        max_game_plies=50, tail_solve_enabled=True,
        tail_solve_depth_limit=10, tail_solve_node_budget=200000,
    )
    tiny_ratio = ep_tiny["tail_solve_successes"] / max(1, ep_tiny["tail_solve_attempts"])
    large_ratio = ep_large["tail_solve_successes"] / max(1, ep_large["tail_solve_attempts"])
    assert large_ratio >= tiny_ratio


def test_heuristic_guidance_ratio_injects_heuristic_quoridor():
    ep_full = dinoboard_engine.run_selfplay_episode(
        game_id="quoridor", seed=100, model_path=get_test_model("quoridor"), simulations=10,
        max_game_plies=30, heuristic_guidance_ratio=1.0, heuristic_temperature=0.0,
    )
    ep_none = dinoboard_engine.run_selfplay_episode(
        game_id="quoridor", seed=100, model_path=get_test_model("quoridor"), simulations=10,
        max_game_plies=30, heuristic_guidance_ratio=0.0,
    )
    full_visits = [sum(s["policy_action_visits"]) for s in ep_full["samples"]]
    none_visits = [sum(s["policy_action_visits"]) for s in ep_none["samples"]]
    avg_full = sum(full_visits) / max(1, len(full_visits))
    avg_none = sum(none_visits) / max(1, len(none_visits))
    assert avg_full != avg_none


def test_training_filter_ratio_restricts_policy_not_mask_quoridor():
    ep_filtered = dinoboard_engine.run_selfplay_episode(
        game_id="quoridor", seed=42, model_path=get_test_model("quoridor"), simulations=10,
        max_game_plies=30, training_filter_ratio=1.0,
    )
    ep_unfiltered = dinoboard_engine.run_selfplay_episode(
        game_id="quoridor", seed=42, model_path=get_test_model("quoridor"), simulations=10,
        max_game_plies=30, training_filter_ratio=0.0,
    )
    filtered_mask_sums = [sum(s["legal_mask"]) for s in ep_filtered["samples"]]
    unfiltered_mask_sums = [sum(s["legal_mask"]) for s in ep_unfiltered["samples"]]
    avg_filtered_mask = sum(filtered_mask_sums) / max(1, len(filtered_mask_sums))
    avg_unfiltered_mask = sum(unfiltered_mask_sums) / max(1, len(unfiltered_mask_sums))
    assert avg_filtered_mask >= avg_unfiltered_mask
    filtered_policy_counts = [len(s["policy_action_ids"]) for s in ep_filtered["samples"]]
    unfiltered_policy_counts = [len(s["policy_action_ids"]) for s in ep_unfiltered["samples"]]
    avg_filtered_policy = sum(filtered_policy_counts) / max(1, len(filtered_policy_counts))
    avg_unfiltered_policy = sum(unfiltered_policy_counts) / max(1, len(unfiltered_policy_counts))
    assert avg_filtered_policy < avg_unfiltered_policy


def test_max_game_plies_limits_episode(game_id, game_config):
    max_plies = 10
    ep = run_short_selfplay(game_id, max_game_plies=max_plies)
    assert ep["total_plies"] <= max_plies


def test_simulations_affects_visit_count():
    ep_low = dinoboard_engine.run_selfplay_episode(
        game_id="tictactoe", seed=42, model_path=get_test_model("tictactoe"), simulations=5, max_game_plies=9,
    )
    ep_high = dinoboard_engine.run_selfplay_episode(
        game_id="tictactoe", seed=42, model_path=get_test_model("tictactoe"), simulations=50, max_game_plies=9,
    )
    if not ep_low["samples"] or not ep_high["samples"]:
        pytest.skip("no samples")
    visits_low = sum(ep_low["samples"][0]["policy_action_visits"])
    visits_high = sum(ep_high["samples"][0]["policy_action_visits"])
    assert visits_high > visits_low


# ─── Resolver behavior tests ─────────────────────────────────────────────────


_PROFILE_NAMES = ["selfplay", "arena", "eval", "web_expert", "web_casual", "analysis"]


@pytest.mark.parametrize("game_id", enabled_games())
@pytest.mark.parametrize("profile_name", _PROFILE_NAMES)
def test_profile_resolves(game_id, profile_name):
    """Every (game, profile) combination must resolve to a complete MctsProfile."""
    p = resolve_profile(game_id, profile_name)
    assert p.simulations > 0
    assert p.opponent_selection in {"puct", "prior"}


@pytest.mark.parametrize("game_id", enabled_games())
def test_max_game_plies_top_level(game_id):
    """game.json must have a top-level `max_game_plies`."""
    n = max_game_plies(game_id)
    assert n > 0


def test_unknown_profile_raises():
    with pytest.raises(KeyError):
        resolve_profile("quoridor", "no_such_profile")


def test_simulations_passthrough_quoridor():
    """Profile.simulations should reach the engine via run_selfplay_episode."""
    p = resolve_profile("quoridor", "selfplay")
    ep = dinoboard_engine.run_selfplay_episode(
        game_id="quoridor", seed=1, model_path=get_test_model("quoridor"),
        simulations=p.simulations, max_game_plies=20,
    )
    if ep["samples"]:
        sample_visits = sum(ep["samples"][0]["policy_action_visits"])
        assert sample_visits > 0


def test_inheritance_cycle_raises(tmp_path, monkeypatch):
    """A profile chain that loops back on itself must be rejected."""
    fake_game_root = tmp_path / "games" / "fakegame" / "config"
    fake_game_root.mkdir(parents=True)
    base = {
        "game_id": "fakegame",
        "max_game_plies": 10,
        "mcts_profiles": {
            "selfplay": {"inherits": "arena"},
            "arena": {"inherits": "selfplay"},
            "eval": {"inherits": "arena"},
        },
    }
    (fake_game_root / "game.json").write_text(json.dumps(base))
    (fake_game_root / "web.json").write_text(json.dumps({"mcts_profiles": {}}))

    from training import mcts_profile as mp
    monkeypatch.setattr(mp, "_PROJECT_ROOT", tmp_path)
    clear_cache()
    with pytest.raises(ValueError, match="cycle"):
        mp.resolve_profile("fakegame", "selfplay")


def test_unknown_field_in_profile_raises(tmp_path, monkeypatch):
    """Typo in a profile field name must surface, not silently apply BASE."""
    fake_game_root = tmp_path / "games" / "fakegame" / "config"
    fake_game_root.mkdir(parents=True)
    base = {
        "game_id": "fakegame",
        "max_game_plies": 10,
        "mcts_profiles": {
            "selfplay": {"simulations": 100, "tail_solve_start_ply": 9},
            "arena": {"inherits": "selfplay"},
            "eval": {"inherits": "arena"},
        },
    }
    (fake_game_root / "game.json").write_text(json.dumps(base))
    (fake_game_root / "web.json").write_text(json.dumps({"mcts_profiles": {}}))

    from training import mcts_profile as mp
    monkeypatch.setattr(mp, "_PROJECT_ROOT", tmp_path)
    clear_cache()
    with pytest.raises(ValueError):
        mp.resolve_profile("fakegame", "selfplay")


def test_tail_solve_enabled_without_trigger_raises(tmp_path, monkeypatch):
    """A profile with tail_solve_enabled=true for a game without a registered
    trigger must be rejected. Tictactoe has no tail_solve_trigger, so we
    write a config that flips it on for tictactoe and expect ValueError."""
    fake_root = tmp_path
    fake_game_root = fake_root / "games" / "tictactoe" / "config"
    fake_game_root.mkdir(parents=True)
    cfg = {
        "game_id": "tictactoe",
        "max_game_plies": 9,
        "mcts_profiles": {
            "selfplay": {
                "simulations": 100,
                "tail_solve_enabled": True,
            },
            "arena": {"inherits": "selfplay"},
            "eval": {"inherits": "arena"},
        },
    }
    (fake_game_root / "game.json").write_text(json.dumps(cfg))
    (fake_game_root / "web.json").write_text(json.dumps({"mcts_profiles": {}}))

    from training import mcts_profile as mp
    monkeypatch.setattr(mp, "_PROJECT_ROOT", fake_root)
    clear_cache()
    with pytest.raises(ValueError, match="tail_solve"):
        mp.resolve_profile("tictactoe", "selfplay")


def test_temperature_schedule_flat_form_rejected(tmp_path, monkeypatch):
    """The flat `temperature_initial` field must be rejected — schedule
    must be the nested `temperature_schedule.{enabled,initial,final,decay_plies}`."""
    fake_game_root = tmp_path / "games" / "fakegame" / "config"
    fake_game_root.mkdir(parents=True)
    cfg = {
        "game_id": "fakegame",
        "max_game_plies": 10,
        "mcts_profiles": {
            "selfplay": {"simulations": 100, "temperature_initial": 1.0},
            "arena": {"inherits": "selfplay"},
            "eval": {"inherits": "arena"},
        },
    }
    (fake_game_root / "game.json").write_text(json.dumps(cfg))
    (fake_game_root / "web.json").write_text(json.dumps({"mcts_profiles": {}}))

    from training import mcts_profile as mp
    monkeypatch.setattr(mp, "_PROJECT_ROOT", tmp_path)
    clear_cache()
    with pytest.raises(ValueError):
        mp.resolve_profile("fakegame", "selfplay")
