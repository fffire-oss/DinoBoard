"""OB-002 regression: Love Letter encoder must not read opponent hand
knowledge from a tracker bound to a different perspective.

Architectural rule (engine/core/belief_tracker.h::IBeliefTracker::perspective_player):
encoders that consume tracker knowledge MUST gate on tracker.perspective_player()
matching the player being encoded. Otherwise, when MCTS descends into a
node whose current_player differs from the search root, encoding for
that seat with the root's tracker would inject root's private knowledge
into a non-owner's features.

Test: drive a 3p Love Letter game where seat 0 plays Priest on seat 1
(seat 0's tracker now holds known_hand[1]). Then call encode_private for
seat 2 using a tracker that is bound to seat 0. The returned private
features for seat 2 must contain NO bits derived from tracker[0]'s
knowledge of seat 1.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT / "platform"))

import dinoboard_engine as engine


# Encoder layout for LoveLetter (NPlayers=3, kCardTypes=8):
#   For each pi in [0, NPlayers): pid = (player + pi) % NPlayers
#     dims [0..7]: hand one-hot (self=real, opp=tracker.known_hand if perspective)
#     dims [8..15]: drawn_card one-hot (self & current_player only)
# So opp slot pi has hand dims at offset pi*16 and drawn at pi*16 + 8.
_HAND_DIMS = 8
_DRAWN_DIMS = 8
_SLOT_DIMS = _HAND_DIMS + _DRAWN_DIMS


def _find_priest_scenario():
    """Find a seed where seat 0's first action is Priest on seat 1.
    Verified at write time: seed=0 works (seat 0 has Princess+Priest)."""
    seed = 0
    gs = engine.GameSession("loveletter_3p", seed=seed, model_path="", use_filter=False)
    assert gs.current_player == 0, f"expected seat 0 to act first at seed {seed}"
    legal = gs.get_legal_actions()
    # kPriestOffset=28; Priest on seat 1 is action 29.
    assert 29 in legal, f"expected Priest-on-seat-1 (action 29) legal at seed {seed}, got {legal}"
    st = gs.get_state_dict()
    seat1_hand = st["players"][1]["hand"]
    return seed, 29, seat1_hand


def test_ob002_encoder_does_not_leak_priest_knowledge_to_third_party():
    """Priest reveal: tracker bound to seat 0 learns seat 1's hand.
    Encoder asked to produce features for seat 2 (with that same tracker)
    must NOT set any opp-hand bits for seat 1, because seat 2 doesn't
    legitimately know seat 1's hand — that's seat 0's private info."""
    seed, priest_action, seat1_hand = _find_priest_scenario()

    # Sanity: seat 1's hand is a real card we can look for.
    assert 1 <= seat1_hand <= 8

    # Encode seat 2's private with tracker bound to seat 0, after seat 0
    # plays Priest on seat 1.
    out = engine.encode_state_for_perspective(
        game_id="loveletter_3p",
        seed=seed,
        encode_player=2,
        tracker_perspective=0,
        event_actions=[(0, priest_action)],
    )
    private = out["private_features"]
    assert out["tracker_perspective"] == 0

    # Slot order for player=2: pi=0 → pid=2 (self), pi=1 → pid=0, pi=2 → pid=1.
    # Seat 1 is at slot pi=2, hand bits at offset 32..39.
    seat1_slot_offset = 2 * _SLOT_DIMS
    seat1_hand_bits = private[seat1_slot_offset : seat1_slot_offset + _HAND_DIMS]
    assert all(bit == 0.0 for bit in seat1_hand_bits), (
        f"OB-002 leak: seat 2's private features include seat 1's hand bits "
        f"{list(seat1_hand_bits)}. Tracker bound to seat 0 (with known_hand[1]={seat1_hand}) "
        f"must not inject knowledge into a non-owner's encoding."
    )

    # And seat 0's slot (pi=1) should also be empty for player=2: tracker[0]
    # may know seat 0's own hand, but seat 2 doesn't.
    seat0_slot_offset = 1 * _SLOT_DIMS
    seat0_hand_bits = private[seat0_slot_offset : seat0_slot_offset + _HAND_DIMS]
    assert all(bit == 0.0 for bit in seat0_hand_bits), (
        f"OB-002 leak: seat 2's private features include seat 0's hand bits "
        f"{list(seat0_hand_bits)}."
    )


def test_ob002_perspective_player_sees_priest_knowledge():
    """Positive control: when encode_player matches tracker_perspective,
    the Priest reveal SHOULD show up in features. Otherwise the gate is
    over-eager and we've broken legitimate tracker use."""
    seed, priest_action, seat1_hand = _find_priest_scenario()

    out = engine.encode_state_for_perspective(
        game_id="loveletter_3p",
        seed=seed,
        encode_player=0,
        tracker_perspective=0,
        event_actions=[(0, priest_action)],
    )
    private = out["private_features"]
    assert out["tracker_perspective"] == 0

    # Slot order for player=0: pi=0 → pid=0 (self), pi=1 → pid=1, pi=2 → pid=2.
    # Seat 1 is at slot pi=1, hand bits at offset 16..23.
    seat1_slot_offset = 1 * _SLOT_DIMS
    seat1_hand_bits = list(private[seat1_slot_offset : seat1_slot_offset + _HAND_DIMS])
    # Card values are 1..8; one-hot bit index for card c is (c-1).
    expected_idx = seat1_hand - 1
    assert seat1_hand_bits[expected_idx] == 1.0, (
        f"Priest reveal lost: tracker bound to seat 0 should know seat 1's "
        f"hand={seat1_hand}, expected bit {expected_idx} set in features but got "
        f"{seat1_hand_bits}."
    )
    # Exactly one bit set.
    assert sum(b == 1.0 for b in seat1_hand_bits) == 1


if __name__ == "__main__":
    test_ob002_encoder_does_not_leak_priest_knowledge_to_third_party()
    test_ob002_perspective_player_sees_priest_knowledge()
    print("OB-002 tests passed")
