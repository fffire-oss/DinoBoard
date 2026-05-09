"""Phase 1.4/1.5 surface tests for engine/core/masked_state.h.

Pin the locked Phase 1.4 vocabulary so Phase 1.5 / 3 / 5 can't rename
or remove the primitives the implementation plan promises. Phase 1.5
tightens the surface: make_masked_state has an inline body that
delegates to IGameState::apply_viz_mask. Per-game runtime semantics
(hidden slots become kPlaceholder, viz=1 slots survive, internal
fields always placeholder) land per-game in Phase 3 once each game's
schema + apply_viz_mask override is in place.
"""
from __future__ import annotations

import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MASKED_STATE_H = PROJECT_ROOT / "engine" / "core" / "masked_state.h"
GAME_INTERFACES_H = PROJECT_ROOT / "engine" / "core" / "game_interfaces.h"


def _read(p: Path) -> str:
    assert p.exists(), f"{p} missing — Phase 1.4 implementation incomplete"
    return p.read_text(encoding="utf-8")


def test_masked_state_h_declares_kplaceholder_constants() -> None:
    """Three sentinels: int32, int8, bool. Encoders compare against
    these to decide "this slot is hidden, emit placeholder."""
    text = _read(MASKED_STATE_H)
    for sym in ("kPlaceholderInt32", "kPlaceholderInt8", "kPlaceholderBool"):
        assert sym in text, f"masked_state.h missing {sym}"
    # The int32 sentinel must be INT32_MIN-flavored, far outside any
    # game's legitimate id range.
    assert "std::numeric_limits<std::int32_t>::min" in text


def test_masked_state_h_declares_make_masked_state() -> None:
    """Phase 1.5: inline body lands. Signature still takes
    (state, perspective, optional belief_filled)."""
    text = _read(MASKED_STATE_H)
    # Function defined inline, returns unique_ptr<IGameState>, takes
    # (state, perspective, optional belief_filled).
    assert re.search(
        r"std::unique_ptr<IGameState>\s+make_masked_state\([^)]*int\s+perspective",
        text,
    ), "make_masked_state must take an explicit perspective parameter"
    # Belief-filled mask is optional (defaults to nullptr).
    assert "belief_filled" in text
    assert "nullptr" in text, "belief_filled should default to nullptr"
    # Phase 1.5 body delegates to apply_viz_mask. Without this delegation
    # any future "smart" rewrite that walks state.viz_ from framework
    # code would re-introduce the field-reflection bridge we explicitly
    # rejected (game owns its own layout — framework just dispatches).
    assert "apply_viz_mask" in text, (
        "make_masked_state body must delegate to apply_viz_mask — "
        "the game knows its field layout, not the framework"
    )
    assert "clone_state" in text, (
        "make_masked_state body must clone before masking — caller "
        "passes `const IGameState&`, mutation of source would be UB"
    )


def test_igamestate_declares_apply_viz_mask_virtual() -> None:
    """Phase 1.5: the masking hook is a virtual on IGameState. Default
    body is no-op (correct for any state with empty viz_, which is every
    game until its Phase 3 schema lands)."""
    text = _read(GAME_INTERFACES_H)
    assert re.search(
        r"virtual\s+void\s+apply_viz_mask\([^)]*int[^)]*\)",
        text,
    ), "IGameState must declare `virtual void apply_viz_mask(int perspective)`"
    # Default body must be no-op (`{}`), not pure virtual — fully-public
    # games (TTT/Quoridor/Azul) and pre-Phase-3 hidden-info games rely on
    # the no-op default to keep working before they declare a schema.
    assert re.search(
        r"virtual\s+void\s+apply_viz_mask\([^)]*\)\s*\{\s*\}",
        text,
    ), "apply_viz_mask must have a no-op default body (not pure virtual)"


def test_masked_state_h_aliases_maskedstate_to_igamestate() -> None:
    """MaskedState is structurally IGameState — no parallel vtable.
    The mask is a post-processing pass, not a wrapper type."""
    text = _read(MASKED_STATE_H)
    assert re.search(r"using\s+MaskedState\s*=\s*IGameState\s*;", text), (
        "MaskedState must be a using-alias for IGameState (not a new "
        "wrapper class)"
    )


def test_masked_state_h_has_seat_rotation_helper() -> None:
    """for_each_seat_in_perspective_order replaces every game's
    hand-rolled `(s + perspective) % N` loop in Phase 3 encoder
    migration."""
    text = _read(MASKED_STATE_H)
    assert "for_each_seat_in_perspective_order" in text
    # Inline definition (so Phase 3 encoders can use it without linking
    # against a separate TU).
    assert re.search(
        r"inline\s+void\s+for_each_seat_in_perspective_order\(",
        text,
    ), "seat rotation helper must be inline"
