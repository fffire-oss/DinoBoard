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

### 1. Per-field visibility tagging on state

Each game declares its data field by field in a visibility manifest:
field name, shape, and a same-shape "who can see this" tag (one bit
per slot per player). The rules code is the **sole** writer of these
tags — when it advances state it also flips slots face-up, reveals
specific slots to specific players, or resets a slot back to its
base visibility.

The framework walks this manifest once and, for each slot, decides
"hide or show, given who's looking," producing a **masked observation
view** shared by three consumers: the public snapshot sent to the
client, the MCTS node hash, and the neural-network input tensor.
Hidden slots structurally read as a placeholder sentinel — none of
the three can ever encode the true value. "Which slot is visible to
whom" becomes a **field of state**, not logic hiding inside some
observer implementation. Forcing every game through this same
base-class facility means one CI suite covers all of them at once:
authors who break round-trip / hash scope / public-snapshot
alignment are caught immediately.

### 2. Physical separation between ground truth and AI

Ground truth holds one full truth state for advancing the game,
**plus** one independent session state per player perspective. The
AI decision path (belief tracking, feature encoding, MCTS search)
reads only the session state — there is physically no interface
through which it can obtain a truth pointer. The belief tracker's
"sample a complete world from observation memory" entry point takes
only an observer identity and a randomness source — no truth state;
the MCTS root state is the acting player's own session state, never
truth. This is **structural** safety, not contractual: the author
**has no ability** to read truth, not "is asked nicely not to." The
gap shows up under LLM / inexperienced-author conditions; for
careful authors it's near zero.

### 3. Mandatory web frontend

Every game must ship with a playable web frontend, AI-decision
visualization, and replay tools. This isn't UI polish — it's the
final yardstick for AI behavior. Training metrics (win rate, loss
curves, policy entropy) can look fine while the AI obviously throws
a winning move at some point — that bug is invisible on the command
line, **but a human spots it within 30 seconds of play**.

---

## Algorithm

ISMCTS over a DAG:

- **Root determinization**: each simulation holds an independent
  randomness source; at the root, the belief tracker samples a
  complete world, and as the simulation descends, that same source
  continues to drive the rules code through any physical randomness
  (e.g. Azul factory refill). No chance nodes — physical randomness
  and information asymmetry are handled uniformly inside search
- **DAG, not a tree**: nodes are keyed by "state hash + acting
  player," so the same information set reached through different
  paths shares a node; UCT2 (Childs 2008) corrects the
  multi-incoming-edge over-exploration bias
- **PUCT selection**: AlphaZero-style prior guidance, with the
  policy head supplying initial action weights
- **Endgame solver**: alpha-beta tries to solve before MCTS;
  proven wins skip MCTS entirely

See [ALGORITHM_OVERVIEW.md](ALGORITHM_OVERVIEW.md).

---

## Training

```
selfplay → collect samples → train net → gating eval → update best model → loop
```

Selfplay, arena, search, and the solver run entirely in C++; Python
only runs the training loop and the network update. Config-driven —
training-loop hyperparameters live in `games/<game>/config/game.json`,
and MCTS strength is split across six named profiles (selfplay /
arena / eval in `game.json`, web_expert / web_casual / analysis in
`web.json`). No code edits per training run.

Optional training boosts: heuristic guidance, auxiliary score signal,
action filtering, temperature schedule, Dirichlet noise. See
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
6. Closed-eye phases (Werewolf nights, secret-write actions) —
   visibility nested on top of hidden state is a higher-order
   uncertainty that collides head-on with this framework's
   "visibility is a public rule" assumption

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

### Observation-only AI API

```
POST /ai/sessions                  → create an AI session
POST /ai/sessions/{id}/observe     → tell the AI what happened (action id + public events)
POST /ai/sessions/{id}/decide      → return the chosen action
DELETE /ai/sessions/{id}           → end the session
```

Callers don't need to share game-state code or embed the C++ engine
— translating their own events into action id + public events is
enough. The ground-truth side can be anything (an external API, a
physical tabletop). This is the public-interface instantiation of
the "ground truth and AI physically separated" architecture above.
See [docs/guide/AI_API.md](docs/guide/AI_API.md).

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
verifiable contract** (schema enforcement, signatures locked to a
masked observation view, CI covering visibility misplacement / hash
scope / belief equivalence / public-snapshot round-trip) — when the
LLM gets something wrong the test catches it immediately and the loop
self-repairs. Azul was integrated this way: drop the rulebook, kick
off the LLM, run one round of iteration, end up with a trainable
and playable implementation.

---

## Docs

- [FEATURES_OVERVIEW.md](FEATURES_OVERVIEW.md) — framework
  capabilities at a glance
- [ALGORITHM_OVERVIEW.md](ALGORITHM_OVERVIEW.md) — core algorithm
  walkthrough
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
