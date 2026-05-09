"""Phase 1.4 surface tests for engine/core/masked_state.h.

Pin the locked Phase 1.4 vocabulary so Phase 1.5 / 3 / 5 can't rename
or remove the primitives the implementation plan promises. Runtime
semantics tests for `make_masked_state` (hidden slots become
kPlaceholder, viz=1 slots survive, internal fields always placeholder)
land in Phase 1.5 against a synthetic state struct, and again per-game
in Phase 3.
"""
from __future__ import annotations

import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MASKED_STATE_H = PROJECT_ROOT / "engine" / "core" / "masked_state.h"


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
    """Declaration-only in Phase 1.4. Body lands in Phase 1.5."""
    text = _read(MASKED_STATE_H)
    # Function declared, returns unique_ptr<IGameState>, takes
    # (state, perspective, optional belief_filled).
    assert re.search(
        r"std::unique_ptr<IGameState>\s+make_masked_state\([^)]*int\s+perspective",
        text,
    ), "make_masked_state must take an explicit perspective parameter"
    # Belief-filled mask is optional (defaults to nullptr).
    assert "belief_filled" in text
    assert "nullptr" in text, "belief_filled should default to nullptr"


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
