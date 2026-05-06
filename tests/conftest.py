"""Shared fixtures and helpers for DinoBoard tests.

Architectural principle: this file knows ONLY about the framework matrix
carrier (quoridor + azul + loveletter). Per-game tests in tests/<game>/
are responsible for their own assertions and load their own config via
the load_game_config helper. The framework layer never embeds rules,
constants, or assertions for any specific game outside the matrix.
"""
import json
from pathlib import Path

import pytest
import torch

import dinoboard_engine

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Framework matrix — minimal carrier set used by tests/framework/. These
# three games together cover every structural feature the framework cares
# about (deterministic / symmetric-random / asymmetric-hidden, 2p / 2-4p,
# tail solver, belief tracker with and without per-player private fields,
# elimination). Per-game tests in tests/<game>/ test their own game in full
# regardless of whether it is in this list.
FRAMEWORK_GAMES = ["quoridor", "azul", "loveletter"]
FRAMEWORK_HIDDEN_INFO_GAMES = ["azul", "loveletter"]
FRAMEWORK_MULTIPLAYER_GAMES = ["azul", "loveletter"]
FRAMEWORK_TAIL_SOLVER_GAMES = ["quoridor"]


def load_game_config(game_id: str) -> dict:
    """Load a game's game.json plus C++-side metadata. Generic helper —
    works for any registered game, framework or per-game test."""
    import re
    base = re.sub(r"_\d+p$", "", game_id)
    config_path = PROJECT_ROOT / "games" / base / "config" / "game.json"
    with open(config_path, encoding="utf-8") as f:
        cfg = json.load(f)
    meta = dinoboard_engine.game_metadata(game_id)
    cfg["action_space"] = meta["action_space"]
    cfg["feature_dim"] = meta["feature_dim"]
    cfg["num_players"] = meta["num_players"]
    return cfg


_MODEL_CACHE: dict[str, str] = {}


def get_test_model(game_id: str, tmp_path_factory=None) -> str:
    """Return path to a random ONNX model for the given game. Cached per session."""
    if game_id in _MODEL_CACHE:
        return _MODEL_CACHE[game_id]
    from training.model import create_model_from_config, export_onnx
    meta = dinoboard_engine.game_metadata(game_id)
    feature_dim = meta["feature_dim"]
    action_space = meta["action_space"]
    num_players = meta["num_players"]
    cfg = {"feature_dim": feature_dim, "action_space": action_space, "num_players": num_players}
    net = create_model_from_config(cfg)
    model_dir = Path("/tmp/dinoboard_test_models")
    model_dir.mkdir(parents=True, exist_ok=True)
    path = model_dir / f"test_{game_id}.onnx"
    export_onnx(net, path, feature_dim)
    _MODEL_CACHE[game_id] = str(path)
    return str(path)


def _public_equivalent(value):
    """Drop details explicitly marked hidden before comparing public state."""
    if isinstance(value, dict):
        if value.get("visible") is False:
            return {"visible": False}
        return {k: _public_equivalent(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_public_equivalent(v) for v in value]
    return value


@pytest.fixture
def model_path(game_id):
    """Fixture providing a random ONNX model path for the current game_id."""
    return get_test_model(game_id)


@pytest.fixture
def game_config(game_id):
    """Fixture providing the loaded config for the current game_id. Thin
    convenience wrapper around load_game_config — works for any game,
    framework or otherwise."""
    return load_game_config(game_id)


# `game_id` fixture defaults to FRAMEWORK_GAMES (the matrix carrier set).
# Tests that need exhaustive coverage (e.g. tests/<game>/test_checklist.py)
# hardcode their own game id and don't use this fixture; tests that need
# a different subset can use @pytest.mark.parametrize("game_id", [...]).
@pytest.fixture(params=FRAMEWORK_GAMES)
def game_id(request):
    return request.param


def run_short_selfplay(game_id: str, seed: int = 42, **kwargs) -> dict:
    defaults = dict(
        simulations=10,
        max_game_plies=50,
    )
    defaults.update(kwargs)
    if "model_path" not in defaults:
        defaults["model_path"] = get_test_model(game_id)
    return dinoboard_engine.run_selfplay_episode(
        game_id=game_id, seed=seed, **defaults,
    )


def run_short_heuristic(game_id: str, seed: int = 42, temperature: float = 1.0) -> dict:
    return dinoboard_engine.run_heuristic_episode(
        game_id=game_id, seed=seed, temperature=temperature, max_game_plies=50,
    )


def run_random_episode_states(
    game_id: str,
    *,
    seed: int = 42,
    max_plies: int = 200,
):
    """Drive a game by uniform-random legal actions and yield the
    state_dict at every ply (including the initial state and the terminal
    state). Intended for rule-invariant tests in per-game checklists:

        for state in run_random_episode_states(GAME, seed=s):
            assert_invariants(state)

    Uses random actions instead of MCTS so the test is fast and exercises
    rare states (illegal-looking edge cases that an AI would avoid). The
    underlying engine is the C++ rules implementation, so any rule
    violation surfaces as a broken invariant.
    """
    import random as _random
    rng = _random.Random(seed)
    gs = dinoboard_engine.GameSession(
        game_id, seed=seed, model_path="", use_filter=False)
    yield gs.get_state_dict()
    for _ in range(max_plies):
        if gs.is_terminal:
            break
        legal = gs.get_legal_actions()
        if not legal:
            break
        gs.apply_action(rng.choice(legal))
        yield gs.get_state_dict()


def assert_api_belief_matches_selfplay(
    game_id: str,
    public_keys: list[str],
    *,
    perspective: int = 0,
    seed_gt: int = 42,
    seed_ai: int = 9999,
    plies: int = 80,
    simulations: int = 20,
) -> None:
    """Three-layer equivalence assertion between self-play (ground truth)
    and an independent-seed API session that only sees public events.

    Asserts:
      1. Belief tracker matches at every ply (initial + post-action)
      2. Public state fields (named in `public_keys`) match after replay
      3. Legal-action sets match on perspective-player turns

    Used by per-game checklists for any game with a public_event_extractor.
    The framework matrix carrier (azul, loveletter) calls this through its
    own framework test; other games call this directly from their checklist.
    """
    import sys as _sys
    _platform_path = str(PROJECT_ROOT / "platform")
    if _platform_path not in _sys.path:
        _sys.path.insert(0, _platform_path)

    model_path = get_test_model(game_id)
    ep = dinoboard_engine.run_selfplay_episode(
        game_id=game_id, seed=seed_gt, model_path=model_path,
        simulations=simulations, max_game_plies=plies,
        trace_perspective=perspective,
    )
    assert ep.get("observation_trace"), \
        f"[{game_id}] trace empty — extractor missing or episode terminated immediately"
    trace = ep["observation_trace"]

    api_gs = dinoboard_engine.GameSession(
        game_id, seed=seed_ai, model_path="", use_filter=False)
    api_gs.apply_initial_observation(perspective, ep["initial_observation"])
    assert api_gs.get_belief_snapshot() == ep["initial_belief_snapshot"], \
        f"[{game_id}] initial belief mismatch"

    for step in trace:
        api_gs.apply_observation(
            step["action"], pre_events=step["pre_events"],
            post_events=step["post_events"])
        assert api_gs.get_belief_snapshot() == step["belief_snapshot_after"], \
            f"[{game_id}] belief diverged at ply {step['ply']}"

    api_state = api_gs.get_state_dict()
    gt_gs = dinoboard_engine.GameSession(
        game_id, seed=seed_gt, model_path="", use_filter=False)
    for step in trace:
        gt_gs.apply_action(step["action"])
    gt_state = gt_gs.get_state_dict()
    for key in public_keys:
        assert _public_equivalent(api_state[key]) == _public_equivalent(gt_state[key]), \
            f"[{game_id}] public field '{key}' diverged after trace replay"

    api2 = dinoboard_engine.GameSession(
        game_id, seed=seed_ai + 1, model_path="", use_filter=False)
    api2.apply_initial_observation(perspective, ep["initial_observation"])
    gt2 = dinoboard_engine.GameSession(
        game_id, seed=seed_gt, model_path="", use_filter=False)
    for step in trace:
        api2.apply_observation(
            step["action"], pre_events=step["pre_events"],
            post_events=step["post_events"])
        gt2.apply_action(step["action"])
        if api2.is_terminal or gt2.is_terminal:
            continue
        if api2.current_player != perspective or gt2.current_player != perspective:
            continue
        assert sorted(api2.get_legal_actions()) == sorted(gt2.get_legal_actions()), \
            f"[{game_id}] perspective legal_actions diverged at ply {step['ply']}"
