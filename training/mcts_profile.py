"""MCTS profile resolver — single source of truth for MCTS callsite parameters.

Six named profiles per game:
  - selfplay / arena / eval        (in games/<g>/config/game.json mcts_profiles)
  - web_expert / web_casual / analysis (in games/<g>/config/web.json mcts_profiles)

Cross-file `inherits` is allowed; same name appearing in both files is a hard
error. The selfplay worker, arena worker, gating, web sessions, AI REST
sessions, and eval_model.py all read MCTS knobs only via
`resolve_profile(game_id, profile_name)` — every dataclass field of the
returned MctsProfile is consumed.

The `analysis` profile is a deliberate exception: only its `simulations`
field is consumed by `platform/game_service/pipeline.py`. The drop-score /
smart-hint path hard-codes `temperature=0.0`, `cover_root_edges=True`, and
`opponent_selection="puct"` regardless of profile contents — those are
architectural constraints (see CLAUDE.md "Opponent-node selection" and
docs/guide/CONFIG_REFERENCE.md "分析 pipeline 的特殊行为"), not knobs.
The other profile fields (`c_puct` / `dirichlet_*` / `tail_solve_*` /
`ai_use_action_filter`) are validated and resolved for shape consistency
but never read from the analysis profile in code. Do not configure them
expecting an effect.

Per CLAUDE.md "no silent fallbacks": every resolver branch raises on missing
keys, unknown profile names, unknown fields, inheritance cycles, illegal
opponent_selection values, or `tail_solve_enabled=true` without a registered
trigger. There is no default profile name; callsite picks the right one.
"""
from __future__ import annotations

import dataclasses
import json
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import dinoboard_engine

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_VARIANT_RE = re.compile(r"_\d+p$")

_VALID_OPPONENT_SELECTION = {"puct", "prior"}

_PROFILE_FIELDS = {
    "simulations",
    "c_puct",
    "temperature",
    "temperature_schedule",
    "dirichlet_alpha",
    "dirichlet_epsilon",
    "dirichlet_on_first_n_plies",
    "opponent_selection",
    "tail_solve_enabled",
    "tail_solve_depth_limit",
    "tail_solve_node_budget",
    "tail_solve_margin_weight",
    "ai_use_action_filter",
    "cover_root_edges",
    "inherits",
}

_TEMPERATURE_SCHEDULE_FIELDS = {"enabled", "initial", "final", "decay_plies"}


@dataclass(frozen=True)
class MctsProfile:
    """Resolved MCTS parameters for a (game, profile_name) pair."""

    name: str

    simulations: int
    c_puct: float
    temperature: float
    temperature_schedule_enabled: bool
    temperature_initial: float
    temperature_final: float
    temperature_decay_plies: int

    dirichlet_alpha: float
    dirichlet_epsilon: float
    dirichlet_on_first_n_plies: int

    opponent_selection: str

    tail_solve_enabled: bool
    tail_solve_depth_limit: int
    tail_solve_node_budget: int
    tail_solve_margin_weight: float

    ai_use_action_filter: bool
    cover_root_edges: bool


_BASE_DICT: dict[str, Any] = {
    "simulations": 200,
    "c_puct": 1.4,
    "temperature": 1.0,
    "temperature_schedule_enabled": False,
    "temperature_initial": 1.0,
    "temperature_final": 0.1,
    "temperature_decay_plies": 20,
    "dirichlet_alpha": 0.3,
    "dirichlet_epsilon": 0.25,
    "dirichlet_on_first_n_plies": 30,
    "opponent_selection": "puct",
    "tail_solve_enabled": False,
    "tail_solve_depth_limit": 10,
    "tail_solve_node_budget": 200_000,
    "tail_solve_margin_weight": 0.0,
    "ai_use_action_filter": False,
    "cover_root_edges": False,
}


_PROFILE_CACHE: dict[tuple[str, str], MctsProfile] = {}
_RAW_CACHE: dict[str, tuple[dict[str, dict], int]] = {}
_MAX_PLIES_CACHE: dict[str, int] = {}
_CACHE_LOCK = threading.Lock()


def clear_cache() -> None:
    """Drop all caches. Tests use this between fixtures."""
    with _CACHE_LOCK:
        _PROFILE_CACHE.clear()
        _RAW_CACHE.clear()
        _MAX_PLIES_CACHE.clear()


def _base_game_id(game_id: str) -> str:
    return _VARIANT_RE.sub("", game_id)


def _config_paths(game_id: str) -> tuple[Path, Path]:
    """Resolve game.json + web.json paths, falling back to base id for variants."""
    base = _base_game_id(game_id)
    candidates = [game_id, base] if game_id != base else [game_id]
    for cand in candidates:
        game_path = _PROJECT_ROOT / "games" / cand / "config" / "game.json"
        if game_path.exists():
            web_path = _PROJECT_ROOT / "games" / cand / "config" / "web.json"
            return game_path, web_path
    raise FileNotFoundError(
        f"game.json not found for {game_id!r} (tried "
        f"{', '.join(str(_PROJECT_ROOT / 'games' / c / 'config' / 'game.json') for c in candidates)})")


def _load_raw_profiles(game_id: str) -> dict[str, dict]:
    """Union mcts_profiles dicts from game.json and web.json. Collisions raise."""
    game_path, web_path = _config_paths(game_id)
    with open(game_path, "r", encoding="utf-8") as f:
        game_cfg = json.load(f)
    if "mcts_profiles" not in game_cfg:
        raise KeyError(
            f"{game_path} is missing required top-level key 'mcts_profiles'.")
    raw: dict[str, dict] = dict(game_cfg["mcts_profiles"])

    if web_path.exists():
        with open(web_path, "r", encoding="utf-8") as f:
            web_cfg = json.load(f)
        web_profiles = web_cfg.get("mcts_profiles", {})
        for name, node in web_profiles.items():
            if name in raw:
                raise ValueError(
                    f"profile {name!r} for game {game_id!r} is defined in both "
                    f"{game_path} and {web_path}; profile names must be unique "
                    f"across the two files.")
            raw[name] = node

    for name, node in raw.items():
        if not isinstance(node, dict):
            raise ValueError(
                f"profile {name!r} for game {game_id!r} must be a JSON object, "
                f"got {type(node).__name__}.")
        unknown = set(node.keys()) - _PROFILE_FIELDS
        if unknown:
            raise ValueError(
                f"profile {name!r} for game {game_id!r} has unknown field(s): "
                f"{sorted(unknown)}. Allowed: {sorted(_PROFILE_FIELDS)}.")

    return raw


def _flatten_temperature_schedule(layer: dict[str, Any]) -> dict[str, Any]:
    """Expand `temperature_schedule: {...}` into flat profile fields."""
    if "temperature_schedule" not in layer:
        return layer
    schedule = layer["temperature_schedule"]
    if not isinstance(schedule, dict):
        raise ValueError(
            f"temperature_schedule must be a JSON object with keys "
            f"{sorted(_TEMPERATURE_SCHEDULE_FIELDS)}, got {type(schedule).__name__}.")
    missing = _TEMPERATURE_SCHEDULE_FIELDS - set(schedule.keys())
    if missing:
        raise ValueError(
            f"temperature_schedule missing required keys: {sorted(missing)}.")
    extra = set(schedule.keys()) - _TEMPERATURE_SCHEDULE_FIELDS
    if extra:
        raise ValueError(
            f"temperature_schedule has unknown keys: {sorted(extra)}. "
            f"Allowed: {sorted(_TEMPERATURE_SCHEDULE_FIELDS)}.")
    out = {k: v for k, v in layer.items() if k != "temperature_schedule"}
    out["temperature_schedule_enabled"] = bool(schedule["enabled"])
    out["temperature_initial"] = float(schedule["initial"])
    out["temperature_final"] = float(schedule["final"])
    out["temperature_decay_plies"] = int(schedule["decay_plies"])
    return out


def _resolve_inheritance(name: str, raw_map: dict[str, dict]) -> dict[str, Any]:
    """Walk `inherits` chain, detect cycles, layer overrides on top of BASE."""
    visited: list[str] = []
    chain: list[dict] = []
    cur: str | None = name
    while cur is not None:
        if cur in visited:
            cycle = " -> ".join(visited + [cur])
            raise ValueError(f"profile inheritance cycle: {cycle}")
        visited.append(cur)
        if cur not in raw_map:
            raise KeyError(
                f"profile {cur!r} not found in mcts_profiles "
                f"(referenced by {' -> '.join(visited[:-1]) or '<resolve>'})")
        node = raw_map[cur]
        chain.append({k: v for k, v in node.items() if k != "inherits"})
        nxt = node.get("inherits")
        if nxt is not None and not isinstance(nxt, str):
            raise ValueError(
                f"profile {cur!r}.inherits must be a string, got {type(nxt).__name__}.")
        cur = nxt

    merged = dict(_BASE_DICT)
    for layer in reversed(chain):
        merged.update(_flatten_temperature_schedule(layer))
    return merged


def _validate_resolved(name: str, game_id: str, merged: dict[str, Any]) -> None:
    if merged["opponent_selection"] not in _VALID_OPPONENT_SELECTION:
        raise ValueError(
            f"profile {name!r} for game {game_id!r}: opponent_selection must be "
            f"one of {sorted(_VALID_OPPONENT_SELECTION)}, got "
            f"{merged['opponent_selection']!r}.")
    if merged["tail_solve_enabled"]:
        meta = dinoboard_engine.game_metadata(game_id)
        if not meta["has_tail_solve_trigger"]:
            raise ValueError(
                f"profile {name!r} for game {game_id!r}: tail_solve_enabled=true "
                f"but no tail_solve_trigger is registered in the GameBundle.")


def resolve_profile(game_id: str, profile_name: str) -> MctsProfile:
    """Return the fully-resolved MctsProfile for (game_id, profile_name)."""
    key = (game_id, profile_name)
    with _CACHE_LOCK:
        cached = _PROFILE_CACHE.get(key)
        if cached is not None:
            return cached

    raw_map = _load_raw_profiles(game_id)
    merged = _resolve_inheritance(profile_name, raw_map)
    _validate_resolved(profile_name, game_id, merged)

    profile = MctsProfile(
        name=profile_name,
        simulations=int(merged["simulations"]),
        c_puct=float(merged["c_puct"]),
        temperature=float(merged["temperature"]),
        temperature_schedule_enabled=bool(merged["temperature_schedule_enabled"]),
        temperature_initial=float(merged["temperature_initial"]),
        temperature_final=float(merged["temperature_final"]),
        temperature_decay_plies=int(merged["temperature_decay_plies"]),
        dirichlet_alpha=float(merged["dirichlet_alpha"]),
        dirichlet_epsilon=float(merged["dirichlet_epsilon"]),
        dirichlet_on_first_n_plies=int(merged["dirichlet_on_first_n_plies"]),
        opponent_selection=str(merged["opponent_selection"]),
        tail_solve_enabled=bool(merged["tail_solve_enabled"]),
        tail_solve_depth_limit=int(merged["tail_solve_depth_limit"]),
        tail_solve_node_budget=int(merged["tail_solve_node_budget"]),
        tail_solve_margin_weight=float(merged["tail_solve_margin_weight"]),
        ai_use_action_filter=bool(merged["ai_use_action_filter"]),
        cover_root_edges=bool(merged["cover_root_edges"]),
    )

    with _CACHE_LOCK:
        _PROFILE_CACHE[key] = profile
    return profile


def max_game_plies(game_id: str) -> int:
    """Top-level `max_game_plies` from game.json (game property, all profiles share)."""
    with _CACHE_LOCK:
        cached = _MAX_PLIES_CACHE.get(game_id)
        if cached is not None:
            return cached

    game_path, _ = _config_paths(game_id)
    with open(game_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    if "max_game_plies" not in cfg:
        raise KeyError(
            f"{game_path} is missing required top-level key 'max_game_plies'.")
    value = int(cfg["max_game_plies"])

    with _CACHE_LOCK:
        _MAX_PLIES_CACHE[game_id] = value
    return value


def profile_as_dict(profile: MctsProfile) -> dict[str, Any]:
    """Serialize a profile for cross-process worker handoff (ProcessPoolExecutor)."""
    return dataclasses.asdict(profile)
