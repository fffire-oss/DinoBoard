"""Phase 1.5 surface tests for engine/search/sim_context.h.

Pin the four-step sim entry helper that Phase 3.7's bridge PR will
route net_mcts.cpp through. Behavioral verification (clone isolation,
belief.sample called with right perspective/tracker, belief_filled
shape == state.viz_ shape) is in the C++ smoke test in CI; this file
locks the header surface.
"""
from __future__ import annotations

import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SIM_CTX_H = PROJECT_ROOT / "engine" / "search" / "sim_context.h"


def _read(p: Path) -> str:
    assert p.exists(), f"{p} missing — Phase 1.5 implementation incomplete"
    return p.read_text(encoding="utf-8")


def test_sim_context_struct_carries_four_fields() -> None:
    text = _read(SIM_CTX_H)
    assert "struct SimContext" in text
    # All four sim-local outputs the bridge PR needs.
    for sym in (
        "std::unique_ptr<IGameState> state",
        "std::unique_ptr<ITracker> tracker",
        "belief_filled",
        "perspective",
    ):
        assert sym in text, f"SimContext missing field: {sym}"


def test_make_sim_context_inline_with_full_signature() -> None:
    """The helper takes (session_state, session_tracker, belief,
    perspective, rng) and is inline so callers can include the
    header without linking another TU."""
    text = _read(SIM_CTX_H)
    assert re.search(
        r"inline\s+SimContext\s+make_sim_context\(",
        text,
    ), "make_sim_context must be inline-defined"
    # Each parameter must appear by name, in order.
    assert re.search(
        r"const\s+IGameState&\s+session_state",
        text,
    )
    assert re.search(
        r"const\s+ITracker\*\s+session_tracker",
        text,
    ), "session_tracker must be a raw pointer (null = no tracker)"
    assert re.search(
        r"const\s+IBelief&\s+belief",
        text,
    )
    assert re.search(
        r"int\s+perspective",
        text,
    )
    assert re.search(
        r"std::mt19937&\s+rng",
        text,
    )


def test_make_sim_context_documents_four_step_order() -> None:
    """The four steps (clone state, clone tracker, allocate
    belief_filled, call belief.sample) must be performed in that
    fixed order — sample reads tracker and writes into both state
    and belief_filled, so the allocs must precede the call."""
    text = _read(SIM_CTX_H)
    # Order check: each numbered step appears, in order.
    s1 = text.find("Step 1")
    s2 = text.find("Step 2")
    s3 = text.find("Step 3")
    s4 = text.find("Step 4")
    assert -1 < s1 < s2 < s3 < s4, (
        "make_sim_context body must document and execute the four steps "
        "in order: 1) clone state, 2) clone tracker, 3) zero belief_filled, "
        "4) belief.sample"
    )


def test_sim_context_does_not_call_make_masked_state() -> None:
    """Masking is a Phase 3 / encoder-time concern — sim entry must
    return the SAMPLED state (every slot has a concrete belief value),
    not a masked one. Encoder applies the mask later, after descent
    has happened against the sampled world."""
    text = _read(SIM_CTX_H)
    assert "make_masked_state" not in text, (
        "sim_context must NOT mask the state — descent runs on the "
        "fully-sampled world, masking is the encoder's job"
    )
