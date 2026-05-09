"""Phase 1.6 (lightweight) surface tests for snapshot.h + protocol.h.

The Phase 1.6 type definitions + validation logic land here. The
framework-side `extract_snapshot` / `apply_observation` bodies that
actually read/write typed game fields stay deferred to Phase 3
(same constraint as Phase 1.5's apply_viz_mask: framework code
holds only `IGameState&`, has no path to typed fields without
game-side reflection).

Surface pins ensure the wire types and validators stay locked even
as Phase 3 lands per-game extract/apply overrides.
"""
from __future__ import annotations

import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SNAPSHOT_H = PROJECT_ROOT / "engine" / "core" / "snapshot.h"
PROTOCOL_H = PROJECT_ROOT / "engine" / "core" / "protocol.h"


def _read(p: Path) -> str:
    assert p.exists(), f"{p} missing — Phase 1.6 implementation incomplete"
    return p.read_text(encoding="utf-8")


def test_snapshot_h_defines_perspective_snapshot() -> None:
    text = _read(SNAPSHOT_H)
    assert "struct PerspectiveSnapshot" in text
    # Two members: visibility_mask + values.
    assert "visibility_mask" in text
    assert "values" in text
    # visibility_mask is keyed by field name → packed tensor.
    assert re.search(
        r"std::unordered_map<\s*std::string\s*,\s*PackedVizTensor\s*>\s*visibility_mask",
        text,
    ), "visibility_mask must be unordered_map<string, PackedVizTensor>"
    # values is keyed by field name → AnyMap (per-field nested payload).
    assert re.search(
        r"std::unordered_map<\s*std::string\s*,\s*AnyMap\s*>\s*values",
        text,
    ), "values must be unordered_map<string, AnyMap>"


def test_snapshot_h_provides_pack_unpack_helpers() -> None:
    text = _read(SNAPSHOT_H)
    for sym in ("PackedVizTensor", "pack_viz", "unpack_viz", "popcount_packed"):
        assert sym in text, f"snapshot.h missing {sym}"
    # bit_count is the truth source for "how many bits are real" — the
    # trailing-byte pad must NOT count toward popcount.
    assert "bit_count" in text


def test_snapshot_h_validates_self_consistency() -> None:
    """A snapshot whose `values` ships a key the mask doesn't admit is
    malformed wire payload. Validator throws so the binding rejects it
    up front, before tracker observe even runs."""
    text = _read(SNAPSHOT_H)
    assert "validate_snapshot_self_consistent" in text


def test_snapshot_h_documents_initial_and_step_share_type() -> None:
    """Golden standard §7.3 — one snapshot type for create_session's
    initial_observation AND apply_observation's snapshot. No second
    `InitialSnapshot`."""
    text = _read(SNAPSHOT_H)
    assert "create_session" in text
    assert "apply_observation" in text
    assert "no separate" in text or "single snapshot type" in text.lower()


def test_protocol_h_defines_observe_request() -> None:
    text = _read(PROTOCOL_H)
    assert "struct ObserveRequest" in text
    # Four fields: actor, action_id, snapshot, post_events.
    assert "actor" in text
    assert "action_id" in text
    assert "snapshot" in text
    assert "post_events" in text
    # actor is required (I17). The default sentinel must be invalid so
    # validate_request_actor_set rejects forgetful callers.
    assert re.search(r"int\s+actor\s*=\s*-1", text), (
        "ObserveRequest.actor must default to -1 (invalid sentinel) so "
        "missing-actor requests are caught by validation"
    )


def test_protocol_h_defines_missing_mandatory_event_error() -> None:
    text = _read(PROTOCOL_H)
    assert "class MissingMandatoryEventError" in text
    # Must subclass std::runtime_error so it propagates through normal
    # exception handling and surfaces a real message at the binding.
    assert re.search(
        r"class\s+MissingMandatoryEventError\s*:\s*public\s+std::runtime_error",
        text,
    )
    # Carries the action_id + kind for diagnostic surfacing.
    assert "action_id()" in text
    assert "kind()" in text


def test_protocol_h_validates_post_events_against_schema() -> None:
    text = _read(PROTOCOL_H)
    assert "validate_post_events" in text
    # Validator iterates the schema's action_events, not events.
    # Position-tolerant: schema is a "must include" floor.
    assert "action_events" in text


def test_protocol_h_validates_actor_required() -> None:
    text = _read(PROTOCOL_H)
    assert "validate_request_actor_set" in text
