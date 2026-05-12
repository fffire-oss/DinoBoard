# DinoBoard

**中文版:** [README_CN.md](README_CN.md)

> **Drop a rulebook. Ship a superhuman AI.**
>
> Tell an LLM "add Azul" in one sentence — it reads the rulebook,
> writes the C++ rules engine, runs the full test suite, trains an
> AI, and generates the web frontend. You just sign off. That is
> how Azul's AI was actually born — and the trained model already
> **surpasses the strongest known human player**. The flow works
> because every integration point in this framework is shaped to be
> **easy for an LLM to get right and instantly caught by CI when it
> doesn't**.

An AlphaZero-style framework for **general board games**, aiming to
turn "rule engine + AI decision + web play" into an end-to-end loop
that stays trustworthy under LLM-assisted development.

---

## Core mechanisms

Three mechanisms serve one motivation: **make the rules engine and
AI decision path trustworthy to human players even under vibe coding**.

### 1. Per-field viz tensor

Each game declares its fields in `<game>_visibility.cpp` — name +
data shape + base viz tensor. `viz[..., p] = 1` means player `p` can
currently see the slot's truth. Rules are the sole writer of viz —
`do_action_fast` updates business fields and calls `reveal_slot` /
`reveal_slot_to` / `reset_to_base` alongside.

`make_masked_state(state, schema, perspective, belief_filled)` walks
the schema once and emits a MaskedState shared by three consumers
(snapshot serialization / hash / encoder tensor) — viz=0 slots
structurally read as `kPlaceholder`. "Which slot is visible to whom"
becomes a **state field**, not logic buried inside an observer
implementation.

Compared with OpenSpiel's Observer API: this isn't something
OpenSpiel can't do — its real games (Gin Rummy and others) also use
visibility bitmaps — only that OpenSpiel doesn't enforce this shape;
authors who get it wrong are caught by round-trip tests in their own
repo. DinoBoard pushes it down to a base-class facility shared by
every game, so a single set of CI tests covers everyone.

### 2. Physical separation between GT and AI session

selfplay / arena / web / API all hold one truth state to advance the
game, **plus** one session state per perspective. The AI decision
path (belief tracker / encoder / MCTS) reads only the session — the
truth pointer is never handed to it. `IBeliefTracker::randomize_unseen(state, observer, rng)`
has no `truth` parameter; the MCTS root state is
`per_seat_states[acting_player]`, never truth.

OpenSpiel can do per-seat runners too — only that
`ResampleFromInfostate` is a member of `State`, where the author
**has the ability** to read truth (contractual safety). DinoBoard's
interface signature doesn't pass truth in (structural safety). The
gap shows up under LLM / inexperienced-author conditions; for
careful authors it's near zero.

### 3. Mandatory web frontend

Every game must ship with a playable web frontend, AI-decision
visualization, and replay tools. This isn't UI polish — it's the
final yardstick for AI behavior. Training metrics (win rate / loss /
policy entropy) can look fine while the AI throws a winning card on
turn 7 — that bug is invisible on the command line, **but a human
spots it within 30 seconds of play**.

Research-track validation uses exploitability / NashConv, which is
more rigorous and reproducible than human play — that's OpenSpiel's
route. DinoBoard's target users don't publish papers and need to
"actually play it and see whether the AI feels off," so the web loop
is the last line of defense.

---

## Algorithm

ISMCTS over a DAG:

- **Root determinization**: each sim draws a complete world from the
  belief tracker; descent is fully deterministic afterward —
  physical randomness and information asymmetry are handled
  uniformly inside search
- **DAG, not tree**: keyed by `(state hash, current_player)` — the
  same info set reached through different paths shares a node;
  UCT2 (Childs 2008) corrects the multi-incoming-edge
  over-exploration bias
- **PUCT selection**: AlphaZero-style prior guidance, with the
  policy head supplying initial action weights
- **Endgame tail solver**: alpha-beta tries to solve before MCTS;
  proven wins skip MCTS

See [ALGORITHM_OVERVIEW.md](ALGORITHM_OVERVIEW.md).

---

## Observation-only AI API

```
POST /ai/sessions                  → create an AI session
POST /ai/sessions/{id}/observe     → tell the AI what happened (action_id + public events)
POST /ai/sessions/{id}/decide      → return the chosen action
DELETE /ai/sessions/{id}           → end the session
```

Callers don't need to share game-state code or embed the C++ engine
— translating their own events into action_id + public events is
enough. The GT side can be anything (an external API, a physical
tabletop). This is the public-interface instantiation of the "GT/AI
session physical separation" architecture above; selfplay / web /
API are behaviorally equivalent at the MCTS level (guarded by
`test_api_mcts_policy_invariance`).

See [docs/guide/AI_API.md](docs/guide/AI_API.md).

---

## Training

```
selfplay → collect samples → train net → gating eval → update best model → loop
```

selfplay / arena / search / solver run entirely in C++; Python only
runs the training loop and the network update. Config-driven — all
training hyperparameters live in `games/<game>/config/game.json`,
no code edits.

Optional training boosts: heuristic guidance (three-stage schedule),
auxiliary score signal, action filtering, temperature schedule,
Dirichlet noise, timeout adjudication. See
[FEATURES_OVERVIEW.md](FEATURES_OVERVIEW.md) § Training.

---

## Scope and boundaries

DinoBoard is built on the AlphaZero / ISMCTS paradigm. Don't try to
force-fit the following (see
[FEATURES_OVERVIEW.md § Framework limitations](FEATURES_OVERVIEW.md)):

1. Games requiring mixed-strategy equilibria (poker variants) — use
   OpenSpiel CFR / Deep CFR / NFSP
2. Combinatorially exploded action spaces (DouDizhu) — use
   OpenSpiel `dou_dizhu` or DouZero
3. Deck construction (Magic: The Gathering) — the framework can't
   absorb it
4. Non-zero-sum / cooperative games (Diplomacy, Hanabi) — use
   OpenSpiel `hanabi` / `bargaining`
5. Single-player games (Solitaire, 2048) — use intrinsic-motivation
   algorithms
6. Closed-eye phases (Werewolf nights, secret-write actions) — viz
   nested on top of hidden state is a higher-order uncertainty that
   collides head-on with this framework's "viz is a public rule"
   assumption

---

## Quick start

### Requirements

- **C++17 compiler** — Mac: `xcode-select --install`; Linux:
  `apt install build-essential`; Windows:
  [Visual Studio 2022 Build Tools](https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2022)
  (just tick "Desktop development with C++"; no full IDE needed)
- **Python ≥ 3.9** with `pybind11` and `torch`
- **ONNX Runtime** — required for web / selfplay / evaluation, since
  all of them load `.onnx` models. Linux x64 / Windows x64 are
  bundled in the repo. On Mac use `brew install onnxruntime`. Other
  platforms: download from
  [ONNX Runtime releases](https://github.com/microsoft/onnxruntime/releases)
  and point `BOARD_AI_ONNXRUNTIME_ROOT` at the path.

### Build (Mac / Linux)

```bash
pip install pybind11 torch
pip install -e .

python -c "import dinoboard_engine; print(dinoboard_engine.available_games())"
```

### Build (Windows)

Open "x64 Native Tools Command Prompt for VS 2022" (the Build Tools
installer creates this Start-Menu shortcut — it's just a `cmd`
session pre-loaded with MSVC environment vars), `cd` to the repo:

```bat
pip install pybind11 torch
pip install -e .
python -c "import dinoboard_engine; print(dinoboard_engine.available_games())"
```

The build copies the bundled `onnxruntime.dll` next to the compiled
extension, so `import` Just Works without touching `PATH`.

### Training

```bash
python -m training.cli --game tictactoe --output runs/tictactoe_001
python -m training.cli --game quoridor  --output runs/quoridor_001 \
    --workers 4 --eval-every 25 --eval-games 40 --eval-benchmark heuristic
```

### Web play

```bash
pip install -r requirements.txt
cd platform && python -m uvicorn app:app --host 0.0.0.0 --port 8000
open http://localhost:8000
```

Six games, three difficulty tiers, multi-player seat selection,
undo, smart hints, replay with per-move loss analysis.

---

## Adding a new game

Typical workflow:

1. Have an LLM read `docs/guide/GAME_DEVELOPMENT_GUIDE.md` and
   `docs/KNOWN_ISSUES.md`, then mirror the closest existing game's
   implementation
2. Drop a `tests/<new_game>/test_checklist.py` (copy from the
   closest existing game; flip `GAME = "..."`) — `pytest tests/<new_game>/`
   going green is the explicit "game is ready" signal
3. Iterate on test failures
4. `python -m training.cli --game <id>` kicks off training
5. Sign off in the web UI

The flow works because every integration point has a **mechanically
verifiable contract** (schema enforcement, signatures locked to
`const MaskedState&`, CI covering viz misplacement / hash scope /
belief equivalence / public-snapshot round-trip) — when the LLM
gets something wrong the test catches it immediately and the loop
self-repairs. Azul was integrated this way: drop the rulebook, kick
off the LLM, run one round of iteration, end up with a trainable
and playable implementation.

---

## Docs

- [FEATURES_OVERVIEW.md](FEATURES_OVERVIEW.md) — framework
  capabilities at a glance
- [ALGORITHM_OVERVIEW.md](ALGORITHM_OVERVIEW.md) — schema / walker
  / MaskedState / RNG / encoder / tracker / belief / ISMCTS DAG
  data-flow contract
- [docs/FRAMEWORK_DESIGN_RATIONALE.md](docs/FRAMEWORK_DESIGN_RATIONALE.md) —
  motivation, who DinoBoard isn't for, and the trade-off comparison
  with OpenSpiel
- [docs/guide/GAME_DEVELOPMENT_GUIDE.md](docs/guide/GAME_DEVELOPMENT_GUIDE.md) —
  single source of truth for adding a new game
- [docs/guide/NEW_GAME_TEST_GUIDE.md](docs/guide/NEW_GAME_TEST_GUIDE.md) —
  11-step acceptance workflow
- [docs/KNOWN_ISSUES.md](docs/KNOWN_ISSUES.md) — bug postmortems
  and design trade-offs

---

## Model status

Shipped ONNX models per variant:

| Game | 2p | 3p | 4p |
|------|----|----|----|
| TicTacToe | trained | — | — |
| Quoridor | trained | — | — |
| Splendor | trained | **untrained (random init)** | **untrained (random init)** |
| Azul | trained | **untrained (random init)** | **untrained (random init)** |
| Love Letter | trained | trained | **untrained (random init)** |
| Coup | trained | **untrained (random init)** | **untrained (random init)** |

*Untrained* variants ship a randomly-initialized network — playable
through the web UI but not competitive.
