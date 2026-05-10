"""Phase 4 lint: snapshot keys must align with viz schema field names.

Each hidden-info game maintains TWO declarations of "what is public":
  1. The viz schema in `<game>_state.cpp` — `viz::declare_field(..., "name",
     viz::all_public(...))` for every public field.
  2. The public_event_extractor in `<game>_register.cpp` — populates
     `out.public_snapshot[ "name" ] = ...` so the AI session can rebuild its
     public state from the message stream.

These two have drifted: e.g. LoveLetter's schema declared "protected_flags"
but its snapshot used "protected"; Azul's snapshot used "line_len_flat" /
"floor_count" while its schema used "player_line_len" / "player_floor_count".

This lint pins them together. Every snapshot key must be either:
  (a) exactly a schema field name, OR
  (b) `<schema-field-name>_flat` (flattened multi-axis tensor — common
      shape for per-player 2D fields), OR
  (c) on the per-game whitelist of "snapshot-only" keys (variable-length
      vectors and partial-reveal sidecars not in the schema).

If a future PR adds a new public field, it goes through schema first;
the snapshot must follow. If it adds a snapshot key with no schema
counterpart, it must whitelist it explicitly here with a reason.
"""
from __future__ import annotations

import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
GAMES_DIR = PROJECT_ROOT / "games"


# Per-game allowlist of snapshot keys that intentionally have no schema
# counterpart. Each entry must have a comment explaining why — usually
# because it's a variable-length vector (handled by hash_public_fields /
# randomize_unseen, not the schema) or a partial-reveal sidecar (the
# schema tracks the gate flag, the snapshot ships the revealed value).
SNAPSHOT_ONLY_KEYS: dict[str, set[str]] = {
    "loveletter": {
        # Variable-length vectors: hidden contents OR all-public stacks
        # not in schema.
        "deck_size",       # public size of hidden deck
        "discard_piles",   # all-public per-player vectors
        "face_up_removed",  # 2p-only public vector
    },
    "coup": {
        "court_deck_size",      # public size of hidden court deck
        "exchange_drawn_mask",  # public occupancy of all_hidden slots
        "revealed_char_flat",   # partial-reveal: char id when revealed_flat[i]==1
    },
    "splendor": {
        "deck_sizes",                 # public per-tier deck sizes
        "reserved_faceup_ids_flat",   # partial-reveal: id when reserved_visible
    },
    "azul": set(),
}


# Schema field names that hash_public_fields legitimately omits — these
# are static-once-set fields that don't influence the equivalence class
# (e.g. azul's `game_first_player` is fixed at game start). Snapshot may
# legitimately omit them too. Listed here so the reverse direction of
# the lint (every schema field has a snapshot key) doesn't fail on them.
SCHEMA_FIELDS_NOT_IN_SNAPSHOT: dict[str, set[str]] = {
    "azul": {
        "game_first_player",  # fixed at game start; observer infers from history
    },
    "splendor": set(),
    "loveletter": set(),
    "coup": {
        "first_player",  # fixed at game start; not in hash_public_fields either
        "ply",           # mirrored as "ply" but coup snapshot uses different naming check
    },
}


def _read(p: Path) -> str:
    assert p.exists(), f"{p} missing"
    return p.read_text(encoding="utf-8")


def _extract_snapshot_keys(register_text: str) -> set[str]:
    """Find every snapshot-key write in the register file. Matches both the
    direct style `snap["name"] = ...` (used by games with hand-written
    extractors: loveletter, coup, splendor) and the SnapshotIO style where
    per-field emitter lambdas write into a generic `m` AnyMap via
    `m["name"] = ...` or via `put_int(m, "name", v)` / `put_bool(m, "name",
    v)` / `put_vec(m, "name", v)` helpers (used by azul after Phase 4 step
    B). The lint cares about *what keys end up in the snapshot*, not about
    the syntactic shape of the write."""
    keys: set[str] = set()
    keys.update(re.findall(r'snap\[\s*"([^"]+)"\s*\]\s*=', register_text))
    keys.update(re.findall(
        r'\bput_(?:int|bool|vec)\s*\(\s*m\s*,\s*"([^"]+)"',
        register_text,
    ))
    return keys


def _extract_all_public_fields(state_text: str) -> set[str]:
    """Find every `declare_field(schema, "name", viz::all_public(...))` in state.cpp."""
    pattern = re.compile(
        r'declare_field\(\s*schema\s*,\s*"([^"]+)"\s*,\s*viz::all_public\(',
        re.MULTILINE,
    )
    return set(pattern.findall(state_text))


def _matches_schema_field(snap_key: str, schema_fields: set[str]) -> bool:
    """A snap key matches a schema field if it's exact or an `_flat` form."""
    if snap_key in schema_fields:
        return True
    if snap_key.endswith("_flat"):
        base = snap_key[: -len("_flat")]
        if base in schema_fields:
            return True
    return False


GAMES_WITH_SNAPSHOT = ("loveletter", "coup", "splendor", "azul")


def test_every_snapshot_key_aligns_with_schema_or_is_whitelisted() -> None:
    """For each hidden-info game: every snap[k] write either matches a
    schema all_public field (exact name or `<field>_flat`) or is in the
    per-game SNAPSHOT_ONLY_KEYS whitelist."""
    failures: list[str] = []
    for game in GAMES_WITH_SNAPSHOT:
        state_text = _read(GAMES_DIR / game / f"{game}_state.cpp")
        register_text = _read(GAMES_DIR / game / f"{game}_register.cpp")
        schema_fields = _extract_all_public_fields(state_text)
        snap_keys = _extract_snapshot_keys(register_text)
        whitelist = SNAPSHOT_ONLY_KEYS.get(game, set())
        for k in sorted(snap_keys):
            if _matches_schema_field(k, schema_fields):
                continue
            if k in whitelist:
                continue
            failures.append(
                f"  {game}: snap[\"{k}\"] has no matching schema all_public "
                f"field and is not in SNAPSHOT_ONLY_KEYS"
            )
    assert not failures, (
        "snapshot keys drifted from viz schema:\n" + "\n".join(failures)
        + "\n\nFix by either renaming the snap key to match the schema "
          "field name (preferred) or, if it's intentionally snapshot-only "
          "(variable-length vector, partial-reveal sidecar), add it to "
          "SNAPSHOT_ONLY_KEYS in this test with a one-line reason."
    )


def test_every_schema_field_appears_in_snapshot() -> None:
    """Reverse direction: every schema all_public field must appear as a
    snap key (exact or `_flat`), unless explicitly listed in
    SCHEMA_FIELDS_NOT_IN_SNAPSHOT for that game.

    Catches the "added a public field to schema but forgot to update
    extractor" failure mode."""
    failures: list[str] = []
    for game in GAMES_WITH_SNAPSHOT:
        state_text = _read(GAMES_DIR / game / f"{game}_state.cpp")
        register_text = _read(GAMES_DIR / game / f"{game}_register.cpp")
        schema_fields = _extract_all_public_fields(state_text)
        snap_keys = _extract_snapshot_keys(register_text)
        skipped = SCHEMA_FIELDS_NOT_IN_SNAPSHOT.get(game, set())
        for f in sorted(schema_fields):
            if f in skipped:
                continue
            if f in snap_keys:
                continue
            if f"{f}_flat" in snap_keys:
                continue
            failures.append(
                f"  {game}: schema all_public field \"{f}\" has no "
                f"corresponding snap key (expected snap[\"{f}\"] or "
                f"snap[\"{f}_flat\"])"
            )
    assert not failures, (
        "schema all_public fields missing from snapshot:\n"
        + "\n".join(failures)
        + "\n\nFix by adding the field to the extractor's public_snapshot "
          "block, or — if intentional (e.g. fixed-at-game-start static) "
          "— adding it to SCHEMA_FIELDS_NOT_IN_SNAPSHOT with a reason."
    )
