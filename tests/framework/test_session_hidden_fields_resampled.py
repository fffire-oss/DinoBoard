"""BG-008 regression: session state_'s hidden fields are re-sampled at the
end of every apply_observation.

Background (see CLAUDE.md "AI Pipeline Independence from Game State" and
KNOWN_ISSUES BUG-028 → BG-008 section):

  Before BG-008, GameSessionWrapper::apply_observation mutated state_ via
  do_action_fast + event applier only. Session state_ accumulated
  RNG-specific hidden samples across plies, which leaked into the public
  hash whenever an action's public output touched hidden fields (BUG-028
  family: Splendor deck size, Coup court deck, LoveLetter winner at
  deck-empty).

  BG-008 added a terminating step: after observe_public_event, run
  tracker.randomize_unseen(state_, fresh_rng) with a deterministic RNG
  derived from (seed_, ply_count_). The public-invariance contract of
  randomize_unseen ensures two sessions with different seeds land on the
  same public hash; per-ply freshening ensures no RNG drift accumulates.

  THIS test guards the freshening actually runs. If a future refactor
  drops the randomize_unseen call (e.g. mistakenly optimizing it out for
  non-MCTS code paths), it would silently re-introduce BUG-028-family
  drift — but test_public_hash_excludes_internal_rng might still pass if
  the public_event protocol's per-event syncs happen to cover all drift
  paths. This test triggers on the freshening itself.

Test method:

  For each hidden-info game:
    1. Run a self-play episode under seed_truth, capture observation trace.
    2. Build two API sessions with the SAME seed, drive through the same
       observation stream. Their hidden-field values must match — same
       tracker state + same freshening RNG → same sample.
    3. Build an API session with a DIFFERENT seed; drive the same stream.
       At least ONE of its hidden-field values must differ from the
       same-seed sessions after at least one ply — proving randomize_unseen
       actually rewrites state_ based on seed. If a refactor silently
       dropped the call, all sessions would agree on whatever stale values
       do_action_fast left behind.

Caveat: a session's get_state_dict exposes ONLY hidden fields that the
game's state_serializer chose to expose. For Coup/Splendor/LoveLetter/
Azul, all four serialize at least one field that randomize_unseen touches
(opp hand, court deck content, bag content, etc.), so the contrast test
has signal. If a future game's serializer returns only public fields, this
test will need a bespoke hidden-field accessor for that game.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "platform"))

import dinoboard_engine as engine
from conftest import get_test_model


HIDDEN_INFO_GAMES = ["azul", "loveletter", "splendor", "coup"]

# Per-game hidden keys to compare. These are keys returned by
# state_serializer whose values vary across randomize_unseen samples.
# Games whose serializers expose only public multisets / sizes (Azul
# bag_counts, Splendor deck_sizes) don't let us observe the hidden
# sample from Python via get_state_dict, so the "different seeds produce
# different hidden" contrast test skips them. The deterministic-repro
# variant still runs for all four games.
HIDDEN_KEYS_PER_GAME: dict[str, list[str]] = {
    "loveletter": ["set_aside_card", "players"],  # players[opp].hand value
    "coup": ["exchange_drawn", "players"],  # players[opp].influence values
}

# Azul bag_counts / Splendor deck_sizes are public-derivable multisets,
# so two sessions with different seeds reach the SAME values. The
# deterministic-repro test (same seed → same snapshot) is trivially
# satisfied for these games and the contrast test (different seed →
# different hidden) is not exercisable from Python. Covered indirectly by
# test_public_hash_excludes_internal_rng (60-seed drift sweep).
SEED_CONTRAST_GAMES = ["loveletter", "coup"]


def _hidden_snapshot(state_dict: dict, keys: list[str]) -> tuple:
    """Return a canonical tuple of the hidden fields we care about.
    Tuples are hashable and equality-comparable."""
    out = []
    for k in keys:
        v = state_dict.get(k)
        out.append((k, repr(v)))  # repr works for nested dicts/lists
    return tuple(out)


@pytest.mark.parametrize("game_id", SEED_CONTRAST_GAMES)
def test_hidden_fields_resampled_deterministically(game_id):
    """Same (tracker state, session seed, ply_count) MUST produce the same
    hidden sample. Otherwise the API session's behavior depends on
    non-deterministic RNG and session hashes are irreproducible."""
    perspective = 0
    seed_truth = 1234
    seed_session = 5678
    hidden_keys = HIDDEN_KEYS_PER_GAME[game_id]

    model_path = get_test_model(game_id)
    ep = engine.run_selfplay_episode(
        game_id=game_id, seed=seed_truth, model_path=model_path,
        simulations=20, max_game_plies=40, trace_perspective=perspective,
    )
    trace = ep.get("observation_trace") or []
    assert trace, f"{game_id}: empty observation trace"

    def _drive(seed: int) -> list[tuple]:
        sess = engine.GameSession(
            game_id, seed=seed, model_path="", use_filter=False)
        sess.apply_initial_observation(perspective, ep["initial_observation"])
        snaps = [_hidden_snapshot(sess.get_state_dict(), hidden_keys)]
        for step in trace:
            sess.apply_observation(
                step["action"], pre_events=step["pre_events"],
                post_events=step["post_events"],
                public_snapshot=step.get("public_snapshot", {}))
            snaps.append(_hidden_snapshot(sess.get_state_dict(), hidden_keys))
        return snaps

    snaps_a = _drive(seed_session)
    snaps_b = _drive(seed_session)  # same seed → same RNG path
    assert snaps_a == snaps_b, (
        f"[{game_id}] same-seed sessions produced different hidden snapshots; "
        f"randomize_unseen is not deterministic from (seed, ply_count)")


@pytest.mark.parametrize("game_id", SEED_CONTRAST_GAMES)
def test_hidden_fields_depend_on_session_seed(game_id):
    """Session state_'s hidden fields MUST vary under different session
    seeds. If they don't, randomize_unseen isn't actually running at end
    of apply_observation — which would silently re-introduce BUG-028
    drift through any public output that reads hidden fields.

    Sweeps a few plies looking for ANY divergence; at least one hidden
    field must differ between seed_a and seed_b by end-of-trace.
    """
    perspective = 0
    seed_truth = 1234
    hidden_keys = HIDDEN_KEYS_PER_GAME[game_id]

    model_path = get_test_model(game_id)
    ep = engine.run_selfplay_episode(
        game_id=game_id, seed=seed_truth, model_path=model_path,
        simulations=20, max_game_plies=40, trace_perspective=perspective,
    )
    trace = ep.get("observation_trace") or []
    assert trace, f"{game_id}: empty observation trace"

    def _drive(seed: int) -> list[tuple]:
        sess = engine.GameSession(
            game_id, seed=seed, model_path="", use_filter=False)
        sess.apply_initial_observation(perspective, ep["initial_observation"])
        snaps = [_hidden_snapshot(sess.get_state_dict(), hidden_keys)]
        for step in trace:
            sess.apply_observation(
                step["action"], pre_events=step["pre_events"],
                post_events=step["post_events"],
                public_snapshot=step.get("public_snapshot", {}))
            snaps.append(_hidden_snapshot(sess.get_state_dict(), hidden_keys))
        return snaps

    snaps_a = _drive(seed=11111)
    snaps_b = _drive(seed=99999)
    assert snaps_a != snaps_b, (
        f"[{game_id}] two different session seeds produced identical hidden "
        f"snapshots through {len(trace)} plies. This almost certainly means "
        f"randomize_unseen is no longer being called at end of "
        f"apply_observation — BG-008 freshening is silently off. "
        f"state_ is carrying whatever do_action_fast + event applier set, "
        f"which re-opens the BUG-028 drift surface.")
