"""Phase 1.2 surface tests for engine/core/visibility_schema.h +
viz_runtime.h. These are static parse / inspection checks rather than
runtime semantics — the runtime semantics test for state.viz_ /
reveal_slot / init_viz lands in Phase 3 once a real game declares a
schema and the rules-side helpers can be exercised end-to-end. Until
then this file pins the API surface so Phase 1.5 / 3 don't accidentally
remove or rename the primitives that the implementation plan promises.

Lifecycle:
  - Phase 1.2 (this PR): file presence + symbol presence + no overlay
    artifacts (the prior overlay design was deleted).
  - Phase 1.5: walker body lands; this test is replaced/extended with a
    real C++ unit test against a synthetic state struct.
  - Phase 3: per-game schemas land; replaced with viz protocol /
    encoder / hash round-trip tests at game level.
"""
from __future__ import annotations

import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_HEADER = PROJECT_ROOT / "engine" / "core" / "visibility_schema.h"
RUNTIME_HEADER = PROJECT_ROOT / "engine" / "core" / "viz_runtime.h"
GAME_INTERFACES = PROJECT_ROOT / "engine" / "core" / "game_interfaces.h"


def _read(p: Path) -> str:
    assert p.exists(), f"{p} missing — Phase 1.2 implementation incomplete"
    return p.read_text(encoding="utf-8")


def test_schema_header_has_phase_1_2_surface() -> None:
    """visibility_schema.h declares the locked Phase 1.2 vocabulary."""
    text = _read(SCHEMA_HEADER)
    # Core types.
    assert "struct VizTensor" in text
    assert "struct FieldDecl" in text
    assert "struct VisibilitySchema" in text
    assert "struct EventDecl" in text
    # Static base-viz builders.
    for sym in ("all_public", "all_hidden", "owner_only_first_axis"):
        assert f"VizTensor {sym}(" in text or f" {sym}(" in text, f"missing {sym}"
    # Declaration helper.
    assert "declare_field" in text
    # Offset arithmetic helper used by runtime ops.
    assert "flat_offset_data_only" in text


def test_schema_header_has_no_overlay_residue() -> None:
    """The overlay closure design (kRevealWhen / OverlayKind /
    reveal_when / custom_overlay / FieldDecl::overlay) was rolled back
    in Phase 1.2 to enforce I1 (rules are sole viz writer). Prevent
    accidental reintroduction."""
    text = _read(SCHEMA_HEADER)
    banned = (
        "OverlayKind",
        "kRevealWhen",
        "kRevealToHistory",
        "kDerivedSizeOf",
        "kCustom",
        "reveal_when(",
        "custom_overlay(",
        "FieldDecl::overlay",
    )
    found = [tok for tok in banned if tok in text]
    assert not found, (
        "visibility_schema.h still references overlay-design symbols: "
        + ", ".join(found)
    )


def test_runtime_header_has_phase_1_2_helpers() -> None:
    """viz_runtime.h carries the rules-side mutation primitives + the
    SlotVisitor type alias. The `for_each_visible_slot` body itself
    lives in viz_walker.h (Phase 1.5); runtime only re-exports the
    callback type so encoder / snapshot headers can include the
    smaller header."""
    text = _read(RUNTIME_HEADER)
    for sym in ("init_viz", "reveal_slot", "reveal_slot_to",
                "reset_to_base", "SlotVisitor"):
        assert sym in text, f"viz_runtime.h missing {sym}"


def test_igamestate_carries_viz_member() -> None:
    """game_interfaces.h's IGameState exposes `viz_` keyed by string."""
    text = _read(GAME_INTERFACES)
    # Public unordered_map<string, VizTensor> member, keyed on FieldDecl::name.
    assert re.search(
        r"std::unordered_map<\s*std::string\s*,\s*viz::VizTensor\s*>\s*viz_",
        text,
    ), "IGameState must carry `std::unordered_map<std::string, viz::VizTensor> viz_`"
    # Phase 1.1 + 1.2 framework primitives still in place.
    assert "reset_with_seed_base" in text
    assert "derive_rng" in text


def test_runtime_header_includes_full_igamestate() -> None:
    """viz_runtime.h pulls in game_interfaces.h (which carries the
    concrete IGameState). Schema header itself stays free of that
    include — keeps the include graph one-directional."""
    runtime = _read(RUNTIME_HEADER)
    schema = _read(SCHEMA_HEADER)
    assert '#include "game_interfaces.h"' in runtime
    assert '#include "game_interfaces.h"' not in schema, (
        "visibility_schema.h must NOT include game_interfaces.h — "
        "would create a cycle since game_interfaces.h includes the schema"
    )
