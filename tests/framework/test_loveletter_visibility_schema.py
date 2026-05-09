"""Phase 3 (loveletter): pin LoveLetter's visibility schema surface.

LoveLetter is the most subtle game we ship. Schema partitions:
  - all_public scalars: current_player, first_player, winner, terminal,
    ply.
  - all_public 1D: alive, protected_flags, hand_exposed (the canonical
    public "this seat's hand is revealed" flag).
  - owner_only_first_axis: hand[N]. Each player sees only their own
    hand card. hand_exposed[p] gates downstream reveals (same pattern
    as Coup's revealed[] / influence[]).
  - all_hidden: drawn_card (rules will reveal_slot_to(current_player)
    on draw, reset_to_base on play — reveal-wiring is a follow-on PR).
    set_aside_card (permanently hidden — bottom-of-deck removed at
    game start).
  - Variable-length vectors NOT in schema: deck (hidden contents,
    public size — randomize_unseen handles); discard_piles[N]
    (all-public stack); face_up_removed (2p-only public).

Higher-order belief reasoning lives in the tracker, not the schema.
The schema models direct first-order visibility only.
"""
from __future__ import annotations

import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
HEADER = PROJECT_ROOT / "games" / "loveletter" / "loveletter_state.h"
SOURCE = PROJECT_ROOT / "games" / "loveletter" / "loveletter_state.cpp"


def _read(p: Path) -> str:
    assert p.exists(), f"{p} missing"
    return p.read_text(encoding="utf-8")


def test_header_declares_schema_accessor() -> None:
    text = _read(HEADER)
    assert '#include "../../engine/core/visibility_schema.h"' in text
    assert re.search(
        r"static\s+const\s+viz::VisibilitySchema\s*&\s*schema\s*\(\s*\)\s*;",
        text,
    ), "loveletter_state.h must declare `static const viz::VisibilitySchema& schema()`"


def test_source_partitions_fields_correctly() -> None:
    text = _read(SOURCE)
    assert '#include "../../engine/core/viz_runtime.h"' in text

    public_fields = (
        "current_player", "first_player", "winner", "terminal", "ply",
        "alive", "protected_flags", "hand_exposed",
    )
    for name in public_fields:
        assert re.search(
            rf'declare_field\(\s*schema\s*,\s*"{name}"\s*,\s*viz::all_public\(',
            text,
        ), f"schema() must declare '{name}' with viz::all_public(...)"

    # hand: owner-only.
    assert re.search(
        r'declare_field\(\s*schema\s*,\s*"hand"\s*,\s*'
        r'viz::owner_only_first_axis\(',
        text,
    ), "schema() must declare 'hand' with owner_only_first_axis(...)"

    # drawn_card / set_aside_card: hidden base.
    for name in ("drawn_card", "set_aside_card"):
        assert re.search(
            rf'declare_field\(\s*schema\s*,\s*"{name}"\s*,\s*viz::all_hidden\(',
            text,
        ), f"schema() must declare '{name}' with all_hidden(...)"

    # Variable-length vectors must NOT be schema slots.
    for name in ("deck", "discard_piles", "face_up_removed"):
        assert not re.search(
            rf'declare_field\(\s*schema\s*,\s*"{name}"\s*',
            text,
        ), (
            f"{name} must NOT be a schema slot — variable-length vector; "
            "handled by hash_public_fields / randomize_unseen instead"
        )


def test_reset_with_seed_wires_init_viz() -> None:
    text = _read(SOURCE)
    assert re.search(
        r"viz::init_viz\(\s*\*this\s*,\s*schema\(\s*\)\s*\)\s*;",
        text,
    ), "reset_with_seed must call viz::init_viz(*this, schema())"
