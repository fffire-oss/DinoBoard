"""Model path resolution shared by game_service and ai_service.

Convention: games/<base>/model/<variant>.onnx, where <variant> is the
game_id (e.g. azul_2p.onnx, azul_3p.onnx). All variants of a game live
in the same model directory — no separate games/<name>_3p/ subdirs.

The base 2p game_id also maps to its explicit '<game>_2p.onnx' file —
callers pass either the bare base id ('azul') or the explicit variant id
('azul_2p') and get the same file.
"""
from __future__ import annotations

import re
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def base_game_id(game_id: str) -> str:
    """Strip a trailing _<N>p variant suffix.

    Examples: 'azul' → 'azul', 'azul_3p' → 'azul', 'splendor_2p' → 'splendor'.
    """
    return re.sub(r"_\d+p$", "", game_id)


def variant_model_name(game_id: str) -> str:
    """Return the variant filename stem (without .onnx extension).

    The bare base id ('azul') expands to '<base>_2p' since 2p is the default
    variant. Variant ids ('azul_3p') are returned unchanged.
    """
    base = base_game_id(game_id)
    return game_id if game_id != base else f"{base}_2p"


def expected_model_path(game_id: str) -> Path:
    """Return the canonical model path for a game (whether or not it exists)."""
    base = base_game_id(game_id)
    return _PROJECT_ROOT / "games" / base / "model" / f"{variant_model_name(game_id)}.onnx"


def find_model_path(game_id: str) -> str:
    """Return the path to the game's deployed model, or "" if none exists."""
    canonical = expected_model_path(game_id)
    if canonical.exists():
        return str(canonical)

    # Also try the bare game_id form in case the caller passed 'azul_2p'
    # but the file on disk is 'azul.onnx'. Harmless if nothing matches.
    base = base_game_id(game_id)
    alt = _PROJECT_ROOT / "games" / base / "model" / f"{game_id}.onnx"
    if alt.exists():
        return str(alt)

    return ""


def variant_belief_model_name(game_id: str) -> str:
    """Belief-net filename stem mirroring variant_model_name.

    Examples: 'coup' → 'coup_belief_2p', 'coup_3p' → 'coup_belief_3p'.
    """
    base = base_game_id(game_id)
    variant_suffix = variant_model_name(game_id)[len(base) + 1:]  # '2p' / '3p' / ...
    return f"{base}_belief_{variant_suffix}"


def expected_belief_model_path(game_id: str) -> Path:
    """Return the canonical belief-net model path for a game.

    Mirrors expected_model_path: games/<base>/model/<base>_belief_<N>p.onnx.
    """
    base = base_game_id(game_id)
    return _PROJECT_ROOT / "games" / base / "model" / f"{variant_belief_model_name(game_id)}.onnx"


def find_belief_model_path(game_id: str) -> str:
    """Return path to the game's deployed belief-net, or "" if none exists.

    Games that don't use a learned belief network (most games) simply don't
    have a file at this path; callers pass "" and the GameSession leaves
    bundle.belief_model_path empty.
    """
    canonical = expected_belief_model_path(game_id)
    if canonical.exists():
        return str(canonical)
    return ""
