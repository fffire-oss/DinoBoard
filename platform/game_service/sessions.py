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

from training.mcts_profile import resolve_profile  # noqa: E402

sessions: dict = {}

# Map difficulty string to web profile name. "heuristic" stays out of MCTS
# entirely (no model, no profile).
_DIFFICULTY_TO_PROFILE = {
    "heuristic": None,
    "casual": "web_casual",
    "expert": "web_expert",
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


GAME_CONFIGS = load_game_configs()


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
    if difficulty not in _DIFFICULTY_TO_PROFILE:
        raise ValueError(
            f"unknown difficulty {difficulty!r}, expected one of "
            f"{list(_DIFFICULTY_TO_PROFILE)}")
    profile_name = _DIFFICULTY_TO_PROFILE[difficulty]
    use_model = profile_name is not None

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
    if use_model:
        model_path = find_model_path(actual_id)
        if not model_path:
            base = _base_game_id(actual_id)
            raise FileNotFoundError(
                f"no trained model found for {actual_id}. "
                f"Expected at games/{base}/model/{actual_id}.onnx. "
                f"Copy from runs/<run_name>/models/model_best.onnx after training, "
                f"renaming to <variant>.onnx."
            )

    if use_model:
        live_profile = resolve_profile(game_id, profile_name)
        analysis_profile = resolve_profile(game_id, "analysis")
        effective_sims = live_profile.simulations
        effective_temp = live_profile.temperature
        effective_opp_sel = live_profile.opponent_selection
        ai_use_filter = live_profile.ai_use_action_filter
        tail_solve_enabled = live_profile.tail_solve_enabled
        tail_solve_depth = live_profile.tail_solve_depth_limit
        tail_solve_budget = live_profile.tail_solve_node_budget
        t_sched_enabled = live_profile.temperature_schedule_enabled
        t_initial = live_profile.temperature_initial
        t_final = live_profile.temperature_final
        t_decay = live_profile.temperature_decay_plies
        # Only `simulations` is consumed from the analysis profile. The
        # analysis pipeline (pipeline.py) hard-codes temperature=0.0,
        # cover_root_edges=True, opponent_selection="puct" — see the
        # docstring on training/mcts_profile.py.
        analysis_sims = analysis_profile.simulations
    else:
        # Heuristic difficulty: no profile, no model, no MCTS knobs in play.
        effective_sims = 1
        effective_temp = 0.0
        effective_opp_sel = "puct"
        ai_use_filter = False
        tail_solve_enabled = False
        tail_solve_depth = 10
        tail_solve_budget = 200000
        analysis_sims = 0
        t_sched_enabled = False
        t_initial = 0.0
        t_final = 0.0
        t_decay = 0

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
        "opponent_selection": effective_opp_sel,
        "analysis_simulations": analysis_sims,
        "use_model": use_model,
        "model_path": model_path,
        "ai_use_filter": ai_use_filter,
        "tail_solve_enabled": tail_solve_enabled,
        "tail_solve_depth_limit": tail_solve_depth,
        "tail_solve_node_budget": tail_solve_budget,
        "temperature_schedule_enabled": t_sched_enabled,
        "temperature_initial": t_initial,
        "temperature_final": t_final,
        "temperature_decay_plies": t_decay,
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
