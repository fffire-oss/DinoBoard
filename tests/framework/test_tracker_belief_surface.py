"""Phase 1.3 surface tests for engine/core/tracker.h + belief.h.

Same flavor as test_visibility_schema.py — parse-level pin of the
locked Phase 1.3 vocabulary so Phase 1.5 / 3 / 3.7 can't accidentally
rename or remove the primitives the implementation plan promises.

Lifecycle:
  - Phase 1.3 (this PR): symbol presence + interface contracts.
  - Phase 1.5: UniformBelief gets an active body; this test extends
    with a synthetic-state runtime check for "viz=0 slots get filled,
    viz=1 slots untouched."
  - Phase 3.7 (MCTS bridge): replaced by the end-to-end test
    `test_sim_local_isolation.py` once a real game routes through
    sim_context helper.
"""
from __future__ import annotations

import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRACKER_H = PROJECT_ROOT / "engine" / "core" / "tracker.h"
BELIEF_H = PROJECT_ROOT / "engine" / "core" / "belief.h"


def _read(p: Path) -> str:
    assert p.exists(), f"{p} missing — Phase 1.3 implementation incomplete"
    return p.read_text(encoding="utf-8")


def test_tracker_h_declares_itracker_and_iunseenpool() -> None:
    text = _read(TRACKER_H)
    # Pure-virtual ITracker.
    assert "class ITracker" in text
    for sig in (
        "void init(int perspective",
        "void observe(int actor",
        "void observe_in_sim(int actor",
        "std::unique_ptr<ITracker> clone()",
    ):
        assert sig in text, f"ITracker missing virtual: {sig}"
    # IUnseenPool mixin.
    assert "class IUnseenPool" in text
    assert "publicly_unseen(" in text


def test_tracker_h_does_not_pull_in_snapshot_concrete() -> None:
    """PerspectiveSnapshot is forward-declared, NOT included. The
    concrete type lands in engine/core/snapshot.h in Phase 1.6; tracker
    interface should stay orthogonal to that wire format."""
    text = _read(TRACKER_H)
    assert "struct PerspectiveSnapshot;" in text, (
        "tracker.h must forward-declare PerspectiveSnapshot"
    )
    assert '#include "snapshot.h"' not in text, (
        "tracker.h must NOT include snapshot.h — keep the forward decl"
    )


def test_belief_h_declares_ibelief_and_uniform_belief() -> None:
    text = _read(BELIEF_H)
    assert "class IBelief" in text
    assert "class UniformBelief final" in text
    # IBelief contract.
    assert "void sample(IGameState& state" in text
    assert "std::unique_ptr<IBelief> clone()" in text
    # UniformBelief skeleton: no-op fast path on empty viz_.
    assert "if (state.viz_.empty())" in text, (
        "UniformBelief.sample must early-return on empty state.viz_ "
        "(fully-public games)"
    )


def test_belief_h_calls_out_phase_1_5_body_landing() -> None:
    """The skeleton documents that the active body lands in Phase 1.5
    so future-me / future-PR-author isn't confused by the no-op."""
    text = _read(BELIEF_H)
    assert "Phase 1.5" in text, (
        "belief.h must document when UniformBelief gets its active body"
    )


def test_uniform_belief_has_pool_selector_signature() -> None:
    """Multi-pool routing (Splendor tier-1/2/3) needs the selector
    signature locked in Phase 1.3 so games can declare it during
    Phase 3 schema bring-up without waiting for 1.5."""
    text = _read(BELIEF_H)
    assert "PoolSelector" in text
    # Returns std::string, takes (IGameState, field name, idx vector).
    assert re.search(
        r"std::function<std::string\([^)]*const IGameState[^)]*\)>",
        text,
    ), "UniformBelief::PoolSelector signature must take const IGameState&"
