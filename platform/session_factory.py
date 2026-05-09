"""Single source of truth for constructing engine.GameSession instances.

Both ai_service and game_service create GameSessions with similar shapes
(seed, model path, optional action filter, optional tail-solve, optional
action-history replay). Centralizing the construction eliminates ad-hoc
helpers like _apply_tail_solve_config and prevents call sites from
forgetting tail-solve / filter wiring (see BUG / OB-004).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import dinoboard_engine as engine


@dataclass(frozen=True)
class SessionConfig:
    """Inputs needed to build a GameSession.

    `game_id` should be the variant id used by the engine (e.g. azul_3p),
    not the base game id. `model_path` is "" when running without a model.
    """

    game_id: str
    seed: int
    model_path: str
    use_action_filter: bool = False
    tail_solve_enabled: bool = False
    tail_solve_depth_limit: int = 0
    tail_solve_node_budget: int = 0


def _from_session_dict(sess: dict, *, use_action_filter: bool) -> SessionConfig:
    """Build a SessionConfig from a game_service session dict.

    Centralizes the field-name knowledge so call sites in pipeline.py and
    sessions.py stay short. The caller picks `use_action_filter` because
    analysis/precompute paths force it False even when the live session
    has filtering enabled.
    """
    return SessionConfig(
        game_id=sess["actual_id"],
        seed=sess["seed"],
        model_path=sess["model_path"] if sess["use_model"] else "",
        use_action_filter=use_action_filter,
        tail_solve_enabled=bool(sess.get("tail_solve_enabled", False)),
        tail_solve_depth_limit=int(sess.get("tail_solve_depth_limit", 0)),
        tail_solve_node_budget=int(sess.get("tail_solve_node_budget", 0)),
    )


class SessionFactory:
    """Construct GameSession objects from a SessionConfig."""

    @staticmethod
    def create(
        cfg: SessionConfig,
        history: Optional[Sequence[int]] = None,
    ) -> engine.GameSession:
        gs = engine.GameSession(cfg.game_id, cfg.seed, cfg.model_path, cfg.use_action_filter)
        if cfg.tail_solve_enabled:
            gs.configure_tail_solve(
                True,
                cfg.tail_solve_depth_limit,
                cfg.tail_solve_node_budget,
            )
        if history:
            for action_id in history:
                gs.apply_action(action_id)
        return gs

    @staticmethod
    def from_session_dict(
        sess: dict,
        *,
        use_action_filter: bool,
        history: Optional[Sequence[int]] = None,
    ) -> engine.GameSession:
        cfg = _from_session_dict(sess, use_action_filter=use_action_filter)
        return SessionFactory.create(cfg, history=history)
