"""BUG-028 regression: hash_public_fields must NOT hash internal RNG state.

Background (see CLAUDE.md "Public ≠ every state member that isn't a hand"
and KNOWN_ISSUES BUG-028):

  hash_public_fields() defines what makes two states share the same MCTS
  DAG node. If a game accidentally hashes internal RNG salt, mt19937
  snapshots, deck-shuffle order, or pre-draw bag/box_lid vector ordering
  into the public hash, then two states that ARE the same public info set
  end up at different DAG nodes — search splits, transposition reuse
  collapses, and policy/value estimates become world-specific instead of
  info-set-specific. The symptom is silent: search "runs", but is
  mysteriously weak / inconsistent / different from the API path. Never
  a crash.

Test method (catches BUG-028 by construction):

  For each hidden-info game:
    1. Run a self-play episode under seed_truth, capturing the observation
       trace from `perspective`'s point of view.
    2. Build two API sessions with DIFFERENT seeds (seed_a, seed_b). Each
       internal RNG is therefore in a different state from the start.
    3. Drive both API sessions through the same observation history.
    4. Assert state_hash_for_perspective(perspective) is bit-equal between
       them after every applied observation.

  If hash_public_fields hashes any internal RNG state — directly (rng_salt,
  mt19937 snapshot) or indirectly (vector order of cards in bag, deck,
  box_lid that no player can derive) — the two hashes diverge and the
  test fails immediately.

This is the strongest preventative we have for BUG-028: any new game that
slips a private/internal field into hash_public_fields breaks CI.
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


# Games with hidden information AND a public_event_extractor (required to
# drive the API session via apply_observation). Fully-public games
# (tictactoe, quoridor) are trivially BUG-028-free by structure.
HIDDEN_INFO_GAMES = ["azul", "loveletter", "splendor", "coup"]


@pytest.mark.parametrize("game_id", HIDDEN_INFO_GAMES)
def test_public_hash_invariant_under_internal_rng(game_id):
    """Two API sessions seeded differently but fed the same observation
    history must produce identical state_hash_for_perspective. Failure
    means hash_public_fields is reading internal RNG state — BUG-028.
    """
    perspective = 0
    seed_truth = 42
    seed_a = 9999
    seed_b = 314159  # different RNG state from seed_a

    model_path = get_test_model(game_id)
    ep = engine.run_selfplay_episode(
        game_id=game_id, seed=seed_truth, model_path=model_path,
        simulations=20, max_game_plies=40,
        trace_perspective=perspective,
    )
    trace = ep.get("observation_trace") or []
    if not trace:
        pytest.skip(f"[{game_id}] empty observation trace")

    sess_a = engine.GameSession(
        game_id, seed=seed_a, model_path="", use_filter=False)
    sess_b = engine.GameSession(
        game_id, seed=seed_b, model_path="", use_filter=False)

    sess_a.apply_initial_observation(perspective, ep["initial_observation"])
    sess_b.apply_initial_observation(perspective, ep["initial_observation"])

    h_a = sess_a.state_hash_for_perspective(perspective)
    h_b = sess_b.state_hash_for_perspective(perspective)
    assert h_a == h_b, (
        f"[{game_id}] state_hash_for_perspective diverged at initial "
        f"observation under different internal RNG seeds: {h_a:#x} vs "
        f"{h_b:#x}. This means hash_public_fields (or hash_private_fields "
        f"for perspective={perspective}) is hashing internal RNG state. "
        f"See BUG-028."
    )

    for step in trace:
        sess_a.apply_observation(
            step["action"], pre_events=step["pre_events"],
            post_events=step["post_events"])
        sess_b.apply_observation(
            step["action"], pre_events=step["pre_events"],
            post_events=step["post_events"])
        h_a = sess_a.state_hash_for_perspective(perspective)
        h_b = sess_b.state_hash_for_perspective(perspective)
        assert h_a == h_b, (
            f"[{game_id}] state_hash_for_perspective diverged at ply "
            f"{step['ply']} (action={step['action']}) under different "
            f"internal RNG seeds: {h_a:#x} vs {h_b:#x}. "
            f"Likely cause: hash_public_fields is hashing internal RNG "
            f"state (rng_salt, mt19937 snapshot, or pre-draw deck/bag "
            f"vector ordering). See BUG-028. Public hash MUST be a "
            f"function of the observation history alone."
        )


@pytest.mark.parametrize("game_id", HIDDEN_INFO_GAMES)
def test_mcts_dag_reuses_under_different_sampled_worlds(game_id):
    """Behavioral angle on BUG-028: under correct hash scope, ISMCTS root
    determinization samples many worlds per simulation (different deck
    orderings, different opp hand assignments), but all worlds in the
    same information set hash to the same DAG node and share visits.
    This produces non-trivial `dag_reuse_hits`.

    If hash_public_fields hashes internal RNG state, every sim's hash
    chain becomes unique (each randomize_unseen seed is different),
    DAG reuse collapses to ~0, and the search tree degenerates into N
    disjoint thin chains. This is the silent failure mode of BUG-028:
    search "runs" at the requested simulation count but each sim
    explores almost nothing.

    We assert reuse > 0 as a strict floor: any healthy hidden-info
    game with root determinization MUST produce at least one transposition
    hit at 500 sims.
    """
    model_path = get_test_model(game_id)
    sess = engine.GameSession(
        game_id, seed=2026, model_path=model_path, use_filter=False)
    if sess.is_terminal:
        pytest.skip(f"[{game_id}] terminal at construction")
    r = sess.get_ai_action(simulations=500, temperature=0.0)
    stats = r["stats"]
    sims = stats.get("simulations", 0)
    reuse = stats.get("dag_reuse_hits", 0)
    assert sims > 0, f"[{game_id}] no simulations ran"
    assert reuse > 0, (
        f"[{game_id}] MCTS produced {sims} simulations but "
        f"dag_reuse_hits={reuse}. With root determinization on a hidden-info "
        f"game, every sim samples a different unseen world but they should "
        f"merge by information set in the DAG. Zero reuse means every sim "
        f"hashes uniquely — almost certainly because hash_public_fields is "
        f"reading internal RNG state (rng_salt, deck/bag ordering, mt19937 "
        f"snapshot). See BUG-028. Effective search depth has collapsed to 1 "
        f"per sim — the AI is silently MUCH weaker than the simulation count "
        f"suggests."
    )


