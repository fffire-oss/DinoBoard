"""Phase 1.1 lint (golden standard I3, §2.3): rules code must not directly
construct std::random_device / std::mt19937{...} / std::uniform_int_distribution
etc. — all randomness must flow through IGameState::derive_rng(domain_tag).

Lifecycle:
  - Phase 1.1 (this PR): test exists but is SKIPPED for the six games
    that haven't migrated yet. The allowlist below names every game whose
    rules / state currently still constructs std::mt19937 etc.
  - Phase 2 (per-game): each game's PR removes its allowlist entry as it
    finishes migrating to derive_rng.
  - Phase 2 wrap-up: last allowlist entry is removed; this lint flips to
    error mode (any new file in games/* containing the banned patterns
    fails CI).

The lint only inspects files under games/<id>/*_rules.cpp and
*_state.cpp, since those are the files that golden standard §2.3
governs. net_adapter.cpp, register.cpp etc. are part of the
randomize_unseen / chance dispatch path that Phase 1.1 does NOT
constrain.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
GAMES_DIR = PROJECT_ROOT / "games"

# Banned patterns. `mt19937{...}` / `mt19937(...)` is the typical raw
# constructor. `random_device` is never legitimate inside rules. The
# distributions below are flagged because they're the visible smoke from
# rolling your own RNG plumbing — once derive_rng is the entry point,
# distributions get applied to the mt19937_64 it returns, but they don't
# appear in *_rules.cpp / *_state.cpp anymore (random draws are pop_back
# off a pre-shuffled deck or a `derive_rng()`-driven shuffle/index).
BANNED_PATTERNS: list[tuple[str, str]] = [
    ("std::random_device", r"std::random_device\b"),
    ("std::mt19937 constructor",
     r"std::mt19937(?:_64)?\s*[\{\(](?!\s*\)\s*;)"),
    ("std::default_random_engine", r"std::default_random_engine\b"),
    ("std::uniform_int_distribution", r"std::uniform_int_distribution\b"),
    ("std::uniform_real_distribution", r"std::uniform_real_distribution\b"),
    ("std::shuffle (raw)", r"std::shuffle\b"),
]

# Per-Phase-2 migration removal target. Phase 1.1 lands with everything
# allowlisted (none of the six games have been moved yet beyond their
# initial reset shuffle). Each entry is removed by its game's Phase 2 PR.
ALLOWLISTED_GAMES: set[str] = {
    # Phase 2 step 1: tictactoe + quoridor removed (no rng usage).
    "azul",
    "splendor",
    "loveletter",
    "coup",
}


def _files_for(game_id: str) -> list[Path]:
    base = GAMES_DIR / game_id
    out: list[Path] = []
    out.extend(sorted(base.glob("*_rules.cpp")))
    out.extend(sorted(base.glob("*_state.cpp")))
    return out


def _strip_comments(text: str) -> str:
    """Drop // line comments and /* ... */ block comments before scanning,
    so banned identifiers mentioned only in docs don't trip the lint."""
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    text = re.sub(r"//[^\n]*", "", text)
    return text


def _registered_game_dirs() -> list[str]:
    """Game dirs that contain at least one *_rules.cpp or *_state.cpp.
    Filters out reference / docs subdirs like games/6nimmit/参考."""
    out: list[str] = []
    for p in sorted(GAMES_DIR.iterdir()):
        if not p.is_dir():
            continue
        if any(p.glob("*_rules.cpp")) or any(p.glob("*_state.cpp")):
            out.append(p.name)
    return out


@pytest.mark.parametrize("game_id", _registered_game_dirs())
def test_no_raw_random_in_rules(game_id: str) -> None:
    if game_id in ALLOWLISTED_GAMES:
        pytest.skip(
            f"{game_id} not yet migrated to derive_rng; "
            f"remove from ALLOWLISTED_GAMES in Phase 2"
        )

    failures: list[str] = []
    for path in _files_for(game_id):
        body = _strip_comments(path.read_text(encoding="utf-8"))
        for label, pat in BANNED_PATTERNS:
            for m in re.finditer(pat, body):
                # Compute line number from the original (uncommented) text
                # for a useful error.
                line = body.count("\n", 0, m.start()) + 1
                failures.append(
                    f"  {path.relative_to(PROJECT_ROOT)}:{line}: "
                    f"banned `{label}` — use IGameState::derive_rng(domain_tag) instead"
                )

    assert not failures, (
        f"{game_id}: rules / state files contain raw random constructs "
        f"(golden standard §2.3 / I3):\n" + "\n".join(failures)
    )
