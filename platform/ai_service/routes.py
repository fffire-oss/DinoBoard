"""AI API routes. Contract: no game state crosses this boundary.

Endpoints:
  POST /ai/sessions              — create a session, returns session_id
  POST /ai/sessions/{id}/observe — record an action the AI observed
  POST /ai/sessions/{id}/decide  — ask the AI for its next action
  DELETE /ai/sessions/{id}       — close a session
  GET /ai/sessions/{id}          — status (actions observed, turn, terminal)

Hidden-info games (splendor / azul / loveletter / coup) require the caller
to pass `pre_events`, `post_events`, and `public_snapshot` on every observe
call so the AI's belief tracker stays consistent with truth — the action_id
alone is not enough information for the AI to update hidden state.
Deterministic games (tictactoe / quoridor) only need `action_id`.

AI strength is server-controlled — `simulations` and `temperature` are not
wire fields. The server resolves them per-game from `web.json`
`difficulty_overrides.expert` (with fallback `{simulations: 800,
temperature: 0.0}`). `seed` is optional; omitting it lets the server pick
a fresh `secrets.randbits(64)` value, which is the default for production
clients. Tests may pass an explicit `seed` for reproducibility.
"""
from __future__ import annotations

import secrets
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from .sessions import get_store

router = APIRouter(prefix="/ai/sessions", tags=["ai"])


class CreateSessionRequest(BaseModel):
    game_id: str = Field(..., description="e.g. 'quoridor', 'splendor'")
    seed: Optional[int] = Field(
        None,
        description="Optional RNG seed for internal belief sampling. If "
                    "omitted, the server picks a fresh secrets.randbits(64) "
                    "value. Independent from ground truth either way — the "
                    "AI's hidden state is re-sampled each ply from the "
                    "tracker's information set.",
    )
    my_seat: int = Field(..., description="Which player the AI is playing as (0-indexed)")
    initial_observation: Optional[dict] = Field(
        None,
        description="Hidden-info games only: perspective-specific facts known at "
                    "game start (e.g. own starting hand). Required for hidden-info "
                    "games; ignored for deterministic games.",
    )


class CreateSessionResponse(BaseModel):
    session_id: str
    game_id: str
    num_players: int
    my_seat: int
    current_player: int
    is_terminal: bool


class PublicEvent(BaseModel):
    kind: str = Field(..., description="Game-specific event identifier "
                                       "(e.g. 'reveal_card', 'token_supply_delta').")
    payload: dict = Field(default_factory=dict)


class ObserveRequest(BaseModel):
    action_id: int = Field(..., description="The action that was just played by the "
                                            "current player (NOT the AI's own moves — "
                                            "those come from /decide).")
    pre_events: list[PublicEvent] = Field(
        default_factory=list,
        description="Public events emitted BEFORE the action resolved. "
                    "Required for hidden-info games; ignored for deterministic games.",
    )
    post_events: list[PublicEvent] = Field(
        default_factory=list,
        description="Public events emitted AFTER the action resolved. "
                    "Required for hidden-info games; ignored for deterministic games.",
    )
    public_snapshot: dict = Field(
        default_factory=dict,
        description="Truth-side dump of the game's public fields. Required for "
                    "hidden-info games; ignored for deterministic games.",
    )


class ObserveResponse(BaseModel):
    actions_observed: int
    current_player: Optional[int]
    is_terminal: bool


class DecideResponse(BaseModel):
    action_id: int
    action_info: dict
    stats: dict
    current_player: Optional[int]
    is_terminal: bool


class StatusResponse(BaseModel):
    session_id: str
    closed: bool
    game_id: Optional[str] = None
    num_players: Optional[int] = None
    my_seat: Optional[int] = None
    is_terminal: Optional[bool]
    current_player: Optional[int]
    winner: Optional[int] = None
    actions_observed: int


@router.post("", response_model=CreateSessionResponse)
def create_session(req: CreateSessionRequest):
    seed = req.seed if req.seed is not None else secrets.randbits(64)
    try:
        sess = get_store().create(
            game_id=req.game_id,
            seed=seed,
            my_seat=req.my_seat,
            initial_observation=req.initial_observation,
        )
    except (ValueError, FileNotFoundError) as e:
        raise HTTPException(400, str(e))

    st = sess.status()
    return CreateSessionResponse(
        session_id=sess.session_id,
        game_id=sess.game_id,
        num_players=sess.num_players,
        my_seat=sess.my_seat,
        current_player=st["current_player"] if st["current_player"] is not None else -1,
        is_terminal=st["is_terminal"] if st["is_terminal"] is not None else False,
    )


@router.post("/{session_id}/observe", response_model=ObserveResponse)
def observe(session_id: str, req: ObserveRequest):
    try:
        sess = get_store().get(session_id)
    except KeyError as e:
        raise HTTPException(404, str(e))

    try:
        sess.observe(
            req.action_id,
            pre_events=[e.model_dump() for e in req.pre_events],
            post_events=[e.model_dump() for e in req.post_events],
            public_snapshot=req.public_snapshot,
        )
    except ValueError as e:
        raise HTTPException(400, str(e))
    except RuntimeError as e:
        raise HTTPException(409, str(e))

    st = sess.status()
    return ObserveResponse(
        actions_observed=st["actions_observed"],
        current_player=st["current_player"],
        is_terminal=st["is_terminal"] or False,
    )


@router.post("/{session_id}/decide", response_model=DecideResponse)
def decide(session_id: str):
    try:
        sess = get_store().get(session_id)
    except KeyError as e:
        raise HTTPException(404, str(e))

    try:
        result = sess.decide()
    except RuntimeError as e:
        raise HTTPException(409, str(e))

    st = sess.status()
    return DecideResponse(
        action_id=result["action_id"],
        action_info=result["action_info"],
        stats=result["stats"],
        current_player=st["current_player"],
        is_terminal=st["is_terminal"] or False,
    )


@router.get("/{session_id}", response_model=StatusResponse)
def get_status(session_id: str):
    try:
        sess = get_store().get(session_id)
    except KeyError as e:
        raise HTTPException(404, str(e))
    return StatusResponse(**sess.status())


@router.delete("/{session_id}")
def close_session(session_id: str):
    get_store().close(session_id)
    return {"closed": True}
