"""AI inference API service.

Session-based API: a third-party game server outsources AI decisions to us by
sending ONLY observations (action IDs and, for hidden-info games, the
public-event trace + public_snapshot) and receives ONLY action IDs back.

The API contract never accepts or returns a game state object. The separation
is structural, not procedural — see CLAUDE.md "AI Pipeline Independence from
Game State" for why the AI physically cannot read truth: the belief tracker
takes no state pointer, public state is rebuilt from the message stream, and
session hidden state is re-sampled every ply from the tracker's information
set.

Hidden-info games (Splendor, Love Letter, Coup, Azul) require the caller to
supply `events` and `public_snapshot` on every observe call so the AI's
belief tracker stays consistent with truth — action_id alone is insufficient.
Deterministic games (TicTacToe, Quoridor) only need `action_id`. The session
itself is created with an `initial_observation` for hidden-info games so the
AI knows facts visible at game start (e.g. own starting hand).
"""
