"""Phase 1.5 surface tests for engine/core/viz_walker.h.

The walker body landed in this Phase. Pin its declaration shape so
Phase 3 (per-game schema bring-up) doesn't accidentally rename or
move the canonical traversal — every consumer (state hash builder,
encoder masker, snapshot extractor) must share this single visit.

Behavioral correctness (perspective filtering, empty-viz skip,
declaration-order stability, out-of-range perspective rejection) is
covered by the C++ smoke test in CI; this file only enforces the
header-level vocabulary the plan locked in.
"""
from __future__ import annotations

import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
WALKER_H = PROJECT_ROOT / "engine" / "core" / "viz_walker.h"
RUNTIME_H = PROJECT_ROOT / "engine" / "core" / "viz_runtime.h"


def _read(p: Path) -> str:
    assert p.exists(), f"{p} missing — Phase 1.5 implementation incomplete"
    return p.read_text(encoding="utf-8")


def test_viz_walker_h_defines_for_each_visible_slot() -> None:
    text = _read(WALKER_H)
    # Inline definition, takes (state, schema, perspective, fn).
    assert re.search(
        r"inline\s+void\s+for_each_visible_slot\([^)]*const\s+IGameState",
        text,
    ), "for_each_visible_slot must be inline-defined and take const IGameState&"
    assert re.search(
        r"for_each_visible_slot\([^)]*const\s+VisibilitySchema",
        text,
    ), "for_each_visible_slot must take const VisibilitySchema&"
    assert re.search(
        r"for_each_visible_slot\([^)]*int\s+perspective",
        text,
    ), "for_each_visible_slot must take an explicit int perspective"


def test_viz_walker_skips_empty_viz_fields() -> None:
    """Fields whose runtime viz is empty (no viewers, no slots) MUST
    NOT be visited — they have no semantic content for any viewer and
    must not leak into hash / encoder / snapshot. The internal/derived
    flags were removed in the "state holds only public game data"
    refactor (plan hazy-popping-wozniak.md); the walker now relies on
    `v.empty()` directly."""
    text = _read(WALKER_H)
    assert "v.empty()" in text, (
        "walker must explicitly skip empty-viz fields via v.empty()"
    )


def test_viz_walker_documents_traversal_order() -> None:
    """Single canonical visit order is the whole point of sharing the
    walker — hash / encoder / snapshot must all see the same sequence
    or they'll disagree about field semantics. Pin the order
    documentation so a future refactor doesn't quietly switch to a
    hash-map iteration order (= unstable)."""
    text = _read(WALKER_H)
    assert "row-major" in text or "row major" in text
    assert "declaration order" in text or "declaration / row-major" in text


def test_viz_runtime_h_keeps_slotvisitor_alias() -> None:
    """SlotVisitor type lives in viz_runtime.h so headers that only
    need the type (snapshot extractor, encoder) don't drag in the
    full walker definition."""
    text = _read(RUNTIME_H)
    assert "using SlotVisitor" in text, (
        "viz_runtime.h must export SlotVisitor"
    )
    assert re.search(
        r"std::function<void\([^)]*VizTensor",
        text,
    ), "SlotVisitor signature must include VizTensor reference"
