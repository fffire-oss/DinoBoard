"""In-memory game session management."""
from __future__ import annotations

import json
import sys
import threading
import uuid
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "platform"))

import dinoboard_engine as engine
from model_paths import base_game_id as _base_game_id, find_model_path  # noqa: E402
from session_factory import SessionConfig, SessionFactory  # noqa: E402

sessions: dict = {}

DIFFICULTY_PRESETS = {
    "heuristic": {"simulations": 1, "temperature": 0.0, "use_model": False},
    "casual": {"simulations": 10, "temperature": 0.0, "use_model": True},
    "expert": {"simulations": 5000, "temperature": 0.0, "use_model": True},
}


def load_game_configs() -> dict:
    configs = {}
    games_dir = PROJECT_ROOT / "games"
    for game_dir in sorted(games_dir.iterdir()):
        config_path = game_dir / "config" / "game.json"
        if config_path.exists():
            with open(config_path, encoding="utf-8") as f:
                cfg = json.load(f)
            configs[cfg["game_id"]] = cfg
    return configs


def load_web_configs() -> dict:
    """Load per-game web.json configs. Games without web.json get an empty dict."""
    configs = {}
    games_dir = PROJECT_ROOT / "games"
    for game_dir in sorted(games_dir.iterdir()):
        web_path = game_dir / "config" / "web.json"
        if web_path.exists():
            with open(web_path, encoding="utf-8") as f:
                configs[game_dir.name] = json.load(f)
    return configs


GAME_CONFIGS = load_game_configs()
WEB_CONFIGS = load_web_configs()


def get_session(session_id: str) -> dict:
    sess = sessions.get(session_id)
    if not sess:
        raise KeyError(f"session {session_id} not found")
    return sess


def _make_pipeline_state() -> dict:
    return {
        "phase": "idle",
        "ai_action": None,
        "ai_action_info": None,
        "ai_stats": None,
        "analysis": None,
    }


def _make_precompute_state() -> dict:
    return {
        "ply_index": -1,
        "history_hash": "",
        "result": None,
    }


def create_session(
    game_id: str,
    seed: int,
    human_player: int,
    num_players: int,
    difficulty: str,
) -> tuple[str, dict]:
    if difficulty not in DIFFICULTY_PRESETS:
        raise ValueError(f"unknown difficulty {difficulty!r}, expected one of {list(DIFFICULTY_PRESETS)}")
    preset = DIFFICULTY_PRESETS[difficulty]

    if game_id not in GAME_CONFIGS:
        raise ValueError(f"unknown game_id {game_id!r}, not found in GAME_CONFIGS")

    actual_id = game_id
    if num_players != 2:
        candidate = f"{game_id}_{num_players}p"
        if candidate in engine.available_games():
            actual_id = candidate
        else:
            raise ValueError(
                f"requested {num_players}p variant for {game_id!r}, but registered game "
                f"{candidate!r} is not available"
            )

    model_path = ""
    if preset["use_model"]:
        model_path = find_model_path(actual_id)
        if not model_path:
            base = _base_game_id(actual_id)
            raise FileNotFoundError(
                f"no trained model found for {actual_id}. "
                f"Expected at games/{base}/model/{actual_id}.onnx. "
                f"Copy from runs/<run_name>/models/model_best.onnx after training, "
                f"renaming to <variant>.onnx."
            )

    web_cfg = WEB_CONFIGS.get(game_id, {})
    ai_use_filter = bool(web_cfg.get("ai_use_action_filter", False))

    diff_overrides = web_cfg.get("difficulty_overrides", {}).get(difficulty, {})
    effective_sims = diff_overrides.get("simulations", preset["simulations"])
    effective_temp = diff_overrides.get("temperature", preset["temperature"])
    analysis_sims = web_cfg.get("analysis_simulations", 5000)

    tail_cfg = web_cfg.get("tail_solve", {})
    tail_solve_enabled = bool(tail_cfg.get("enabled", False))
    tail_solve_depth = int(tail_cfg.get("depth_limit", 10))
    tail_solve_budget = int(tail_cfg.get("node_budget", 200000))

    # Live web session uses no filter (filter is for training only); analysis
    # / precompute paths construct their own isolated sessions in pipeline.py.
    gs = SessionFactory.create(SessionConfig(
        game_id=actual_id,
        seed=seed,
        model_path=model_path,
        use_action_filter=False,
        tail_solve_enabled=tail_solve_enabled,
        tail_solve_depth_limit=tail_solve_depth,
        tail_solve_node_budget=tail_solve_budget,
    ))

    actual_num_players = gs.num_players
    if actual_num_players != num_players:
        raise ValueError(
            f"requested {num_players} players for {game_id!r}, but {actual_id!r} "
            f"created a {actual_num_players}p session"
        )
    ai_players = [p for p in range(actual_num_players) if p != human_player]

    session_id = uuid.uuid4().hex[:12]
    sess = {
        "game_session": gs,
        "human_player": human_player,
        "ai_players": ai_players,
        "ai_player": ai_players[0] if ai_players else -1,
        "game_id": game_id,
        "actual_id": actual_id,
        "seed": seed,
        "difficulty": difficulty,
        "simulations": effective_sims,
        "temperature": effective_temp,
        "analysis_simulations": analysis_sims,
        "use_model": preset["use_model"],
        "model_path": model_path,
        "ai_use_filter": ai_use_filter,
        "tail_solve_enabled": tail_solve_enabled,
        "tail_solve_depth_limit": tail_solve_depth,
        "tail_solve_node_budget": tail_solve_budget,
        "action_history": [],
        "replay_frames": [],
        "pipeline": _make_pipeline_state(),
        "precompute": _make_precompute_state(),
        "pipeline_lock": threading.Lock(),
    }
    sessions[session_id] = sess
    return session_id, sess


def rebuild_game_session(sess: dict) -> None:
    """Recreate GameSession and replay action_history. The live session is for
    humans, so use_filter is always False."""
    sess["game_session"] = SessionFactory.from_session_dict(
        sess,
        use_action_filter=False,
        history=sess["action_history"],
    )


def session_response(session_id: str, sess: dict, extra: Optional[dict] = None) -> dict:
    gs = sess["game_session"]
    current = gs.current_player
    legal = gs.get_legal_actions()
    last_actor = None
    frames = sess.get("replay_frames")
    if frames and len(frames) > 0:
        last_actor = frames[-1].get("actor")

    result = {
        "session_id": session_id,
        "state": gs.get_state_dict(),
        "legal_actions": legal,
        "current_player": current,
        "is_terminal": gs.is_terminal,
        "is_turn_start": gs.is_turn_start,
        "last_actor": last_actor,
        "winner": gs.winner,
        "num_players": gs.num_players,
        "human_player": sess["human_player"],
        "ai_players": sess["ai_players"],
        "ai_player": sess["ai_player"],
        "difficulty": sess["difficulty"],
    }
    if extra:
        result.update(extra)
    return result


def pipeline_mark_done_empty(sess: dict) -> None:
    p = sess["pipeline"]
    p["phase"] = "done"
    p["ai_action"] = None
    p["ai_action_info"] = None
    p["ai_stats"] = None
    p["analysis"] = None


def pipeline_reset(sess: dict) -> None:
    p = sess["pipeline"]
    p["phase"] = "idle"
    p["ai_action"] = None
    p["ai_action_info"] = None
    p["ai_stats"] = None
    p["analysis"] = None


def precompute_clear(sess: dict) -> None:
    pc = sess["precompute"]
    pc["ply_index"] = -1
    pc["history_hash"] = ""
    pc["result"] = None


def history_hash(action_history: list) -> str:
    return str(hash(tuple(action_history)))
