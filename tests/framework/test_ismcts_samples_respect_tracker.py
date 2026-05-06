"""ISMCTS determinization samples must respect the tracker's certain knowledge.

`belief_tracker.randomize_unseen(state, rng)` is the engine's root-
determinization step: each MCTS simulation cloning the AI's view of the
world fills in opponent-private slots from the tracker's belief.

The contract is: anything the tracker reports as "definitely known" (e.g.
loveletter `known_hand[p]` after a Priest peek) MUST be honored exactly
in every sampled world. Slots the tracker doesn't know about are free to
be sampled randomly.

If a sample contradicts a known fact, the AI is searching trees whose
roots disagree with what the perspective player has legally observed —
its policy and value estimates become noise, and it can no longer
exploit revealed information (e.g. a Guard guess after a Priest reveal
will not exploit the reveal).

Pattern for adding a hidden-info game:
1. Identify which fields of tracker.serialize() are "definite knowledge".
2. Identify the matching ground-truth fields in trial_states[t]
   (each trial state is the result of `randomize_unseen` on a clone).
3. Add a checker comparing claim → trial-truth and assert
   "claim != UNKNOWN_SENTINEL implies sample == claim".
4. Add the game to ISMCTS_SAMPLE_GAMES.
"""
from __future__ import annotations

import dinoboard_engine
import pytest


def _check_loveletter(snap: dict, trial_state: dict) -> None:
    """Tracker claim: known_hand[p] (0 = unknown, else card 1..8).
    Trial truth: trial_state["players"][p]["hand"]."""
    known = snap.get("known_hand")
    if known is None:
        return
    players = trial_state["players"]
    for p, claim in enumerate(known):
        if claim == 0:
            continue
        if not players[p]["alive"]:
            # Eliminated player has no hand on the table; sample's hand
            # value is bookkeeping (0). Don't enforce.
            continue
        sampled = players[p]["hand"]
        assert sampled == claim, (
            f"loveletter sample contradicts tracker: tracker claims "
            f"player {p} holds card {claim}, but randomize_unseen sampled "
            f"hand={sampled}.")


_CHECKERS = {
    "loveletter": _check_loveletter,
}

ISMCTS_SAMPLE_GAMES = list(_CHECKERS.keys())


@pytest.mark.parametrize("game_id", ISMCTS_SAMPLE_GAMES)
@pytest.mark.parametrize("seed", [7, 42, 113, 555, 2024, 31337])
def test_ismcts_samples_respect_tracker_known(game_id: str, seed: int):
    """Drive the tracker forward via random play, then call randomize_unseen
    multiple times and verify every sampled world honors what the tracker
    claims to definitely know.

    `test_belief_tracker` exposes both:
      - belief_snapshot: tracker.serialize() at the test position
      - trial_states[t]: the full state_dict after randomize_unseen on a
        clone, for trial t

    so we can compare claim-by-claim across many independent
    determinizations.
    """
    r = dinoboard_engine.test_belief_tracker(
        game_id, seed=seed, plies=20, randomize_trials=20)
    snap = r["belief_snapshot"]
    trial_states = r["trial_states"]
    assert len(trial_states) == 20
    checker = _CHECKERS[game_id]
    for t, ts in enumerate(trial_states):
        if not ts:
            pytest.skip(f"no state_serializer for {game_id}")
        try:
            checker(snap, ts)
        except AssertionError as e:
            raise AssertionError(f"trial {t}: {e}") from e
