# DinoBoard

> **Drop a rulebook. Ship a superhuman AI.**
>
> Tell Claude Code "add Azul" in one sentence — it reads the rulebook, writes the C++, runs the full test suite, trains a superhuman AI, and generates the web frontend. You just sign off.

**中文版:** [README_CN.md](README_CN.md)

A general-purpose board-game AI engine — **one framework, one engineering investment, unlimited game reuse**. AlphaZero-style MCTS + neural self-play, supporting 2–4 player games.

---

## What this project solves

AI in digital board games is usually weak — not because the techniques don't exist, but because **rebuilding AlphaZero from scratch for every title is too costly**. MCTS, ONNX integration, training pipelines, hidden-information handling, tuning traps — every game re-walks the same road.

DinoBoard walks it once and turns the result into **a reusable engine plus a callable API**:

- **10k+ lines of C++/Python core** — MCTS, belief tracker, training, web, analysis, all generic
- **~2000 lines to add a new game** — rules + feature encoder + JSON config; the framework owns the rest
- **Two-layer test architecture** — framework invariants run on a 3-game matrix carrier (quoridor + azul + loveletter) covering every structural feature minimally; **each new game gets its own self-contained acceptance checklist** under `tests/<game>/`, so "this game is done" is a single green pytest run
- **Games covering 4 paradigms** — perfect info, symmetric randomness, asymmetric hidden info, bluffing
- **Observation-only REST API for third parties** — digital board game apps / platforms / companion apps call the AI directly without sharing any game-state code or embedding the C++ engine
- **One code path from training to web to external API** — the same C++ MCTS serves self-play, live play, replay analysis, and third-party integration, with zero "training vs. production drift"

---

## Technical introduction

### ISMCTS: DAG search for hidden-information games

A ground-up MCTS redesign for hidden-information games. **Root-sampling determinization + per-acting-player info-set keying + UCT2** — each simulation samples a full world from the belief, descent is fully deterministic afterward, and the same info set reached along different paths shares a DAG node. See [ALGORITHM_OVERVIEW.md](ALGORITHM_OVERVIEW.md).

### Observation-only AI API: trained once, callable by anyone

The framework ships a REST API that third-party systems integrate against with **zero shared code**:

```
POST /ai/sessions                         → create an AI session
POST /ai/sessions/{id}/observe            → tell the AI what happened (action id + public events)
POST /ai/sessions/{id}/decide             → get the AI's chosen action
DELETE /ai/sessions/{id}                  → end the session
```

**The caller does not need the game-state code, the C++ engine, or any knowledge of MCTS.** As long as they can translate their own game events into action ids + public events, they can use a superhuman AI as a black-box opponent or coach.

This is not a stripped-down interface — it runs **the exact same MCTS + belief tracker + ONNX inference as self-play training**. The observation-only design is a structural constraint (`IBeliefTracker::init`, `observe_public_event` and `randomize_unseen` have no read access to truth — `init`/`observe_public_event` carry only `AnyMap` / event streams, `randomize_unseen` only writes into the observer's own session state), which means:

- AI decisions depend only on observation history and can never peek at ground truth → **cheating is structurally impossible**
- The same AI serves self-play training, web play, and the third-party API → **one training investment, three deployment surfaces**
- Independent-seed belief-equivalence tests give an information-theoretic proof of the separation → **you can prove to a client that the AI does not cheat**

Fits: existing digital board game apps that want stronger AI opponents, platforms that want to offer AI coaches, physical tabletop companion apps that need live suggestions.

### Pluggable belief sampling: uniform → heuristic → neural

The same `randomize_unseen` interface supports three escalating strengths:

- **Uniform sampling** for simple stochastic games like Azul
- **Hand-crafted probabilistic heuristics** (Coup's sampler uses claim / challenge history to bias opponent-role priors, avoiding the "never-challenge-never-bluff" degenerate equilibrium)
- **Neural belief networks** (interface-ready; a sequence model can drop in to replace the heuristic)

This is one of the few places the framework was **designed specifically for imperfect-information games**. Most open-source AlphaZero projects handle perfect-info only.

### Training–inference parity

Self-play, evaluation, web play, replay analysis — **all run on the same C++ MCTS**. No "Python in training, rewritten C++ in serving" translation drift. When a player sees "this move dropped my win rate from 62% to 41%" on the web, that number is literally the value head's output during training.

### Engineering discipline

- Two-layer tests: framework invariants on a fixed 3-game matrix; each game has its own complete checklist (`tests/<game>/`) including game-specific rule conservation laws (token / card / piece totals, capacity bounds, reachability), so a new game's readiness is a single self-contained green run
- `docs/KNOWN_ISSUES.md` documents 30+ shipped bugs and design trade-offs — **every pothole the next integrator gets to skip**
- Strict no-fallback discipline (see `CLAUDE.md`): silent degradation is banned, errors must propagate to the surface

---

## Adding a new game: one conversation

You no longer need to hand-write the game bundle. The typical flow:

1. Tell Claude Code "add [game name]"
2. The AI reads `docs/guide/GAME_DEVELOPMENT_GUIDE.md` and `docs/KNOWN_ISSUES.md`, and mirrors Quoridor / Splendor / the other existing games
3. Drop a `tests/<new_game>/test_checklist.py` (copy from the closest existing game; flip `GAME = "..."`) — running `pytest tests/<new_game>/` then becomes the "is the game done?" signal
4. The AI iterates on test failures until everything is green
5. `python -m training.cli --game <id>` kicks off training
6. Open the web UI to accept

**Your job shrinks to: one request + reviewing the PR + starting the training run.**

This flow works in practice because every integration point has a **mechanically verifiable contract** (tests + `KNOWN_ISSUES` pothole list), so the AI can close the loop by itself.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Python layer                         │
│  training/pipeline.py  ←→  bindings/py_engine.cpp       │
│  training/cli.py            (pybind11)                  │
│  platform/app.py       ←→  GameSession                  │
├─────────────────────────────────────────────────────────┤
│                    C++ engine                           │
│  ┌──────────┐  ┌──────────────┐  ┌──────────────────┐   │
│  │ runtime/ │  │ search/      │  │ infer/           │   │
│  │ selfplay │→ │ NetMCTS      │→ │ ONNX Evaluator   │   │
│  │ arena    │  │ (ISMCTS DAG) │  │ (optional ONNX)  │   │
│  │ heuristic│  │ TailSolver   │  │                  │   │
│  └──────────┘  └──────────────┘  └──────────────────┘   │
│  ┌──────────────────────────────────────────────────┐   │
│  │ core/ — interface definitions                    │   │
│  │ IGameState · IGameRules · IFeatureEncoder        │   │
│  │ IBeliefTracker · GameRegistry · GameBundle       │   │
│  └──────────────────────────────────────────────────┘   │
├─────────────────────────────────────────────────────────┤
│                    Game implementations                 │
│  games/tictactoe/  games/quoridor/                      │
│  games/splendor/   games/azul/                          │
│  games/loveletter/ games/coup/                          │
└─────────────────────────────────────────────────────────┘
```

Full directory tree at the bottom of this file.

---

## Quick start

### Requirements

The core is C++; Python is glue. You need:

- **A C++17 compiler** — Mac: `xcode-select --install`; Linux: `apt install build-essential`; Windows: [Visual Studio 2022 Build Tools](https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2022) (the standalone compiler-only installer, **not the full Visual Studio IDE** — a few hundred MB, no IDE installed). In the installer, tick **"Desktop development with C++"** and click Install.
- **Python ≥ 3.9** with `pybind11` and `torch` (for training)
- **ONNX Runtime** — web play, self-play, and evaluation **all load `.onnx` models for the AI to move**; this is required, not optional.
  - **Linux x64**: already bundled at `third_party/onnxruntime-linux-x64-1.17.3/` — `git clone` is enough, no download needed.
  - **Windows x64**: already bundled at `third_party/onnxruntime-win-x64-1.17.3/` — `git clone` is enough, no download needed.
  - **Mac**: `brew install onnxruntime`
  - **Other (Linux ARM, etc.)**: download the platform package from [ONNX Runtime releases](https://github.com/microsoft/onnxruntime/releases) and unpack it; pass it via `BOARD_AI_ONNXRUNTIME_ROOT`.

### Build (Mac / Linux)

```bash
pip install pybind11 torch

# Standard build (Mac brew / Linux bundled paths detect ONNX Runtime automatically)
pip install -e .

# If ONNX Runtime is not in a standard path, point at it explicitly
BOARD_AI_WITH_ONNX=1 \
  BOARD_AI_ONNXRUNTIME_ROOT=/path/to/onnxruntime \
  pip install -e .

# Verify
python -c "import dinoboard_engine; print(dinoboard_engine.available_games())"
# ['azul', 'azul_2p', ..., 'quoridor', 'splendor', ..., 'tictactoe']
```

### Build (Windows)

After the Build Tools install finishes, open the Start Menu and search for **"x64 Native Tools Command Prompt for VS 2022"** — it's a regular command prompt that the installer pre-configures with MSVC's `cl.exe` on PATH. Open it, `cd` to the cloned repo, then:

```bat
pip install pybind11 torch

REM Bundled Windows ONNX Runtime is auto-detected — no env vars needed.
pip install -e .

REM Verify
python -c "import dinoboard_engine; print(dinoboard_engine.available_games())"
```

> Why this prompt instead of regular `cmd` / PowerShell? MSVC needs a batch of environment variables (`INCLUDE`, `LIB`, `PATH` entries) set before `cl.exe` works. The shortcut runs `vcvars64.bat` for you. Plain `cmd` will fail with "cl is not recognized".

The build copies `onnxruntime.dll` next to the compiled extension so it loads at import time without touching `PATH`.

> `setup.py` prints a warning and keeps building if ONNX Runtime is missing — that path exists only so basic tests can run. **Web play and training will both fail later because the model cannot load.**

### Training

```bash
# TicTacToe — about 5 minutes
python -m training.cli --game tictactoe --output runs/tictactoe_001

# Quoridor — several hours (with heuristic guidance schedule)
python -m training.cli --game quoridor --output runs/quoridor_001 \
    --workers 4 --eval-every 25 --eval-games 40 --eval-benchmark heuristic
```

Training is driven by `games/<game>/config/game.json` — no code changes, just JSON. See the [features overview § config quick reference](FEATURES_OVERVIEW.md).

### Web play

```bash
pip install -r requirements.txt  # fastapi, uvicorn
cd platform && python -m uvicorn app:app --host 0.0.0.0 --port 8000
open http://localhost:8000
```

Features: three difficulty tiers (Heuristic / Casual / Expert), seat selection for multi-player variants, undo, smart hints, replay with per-move loss analysis.

---

## Core concepts

- **GameBundle registration** — each game exposes a factory that returns state + rules + encoder + optional components. See the [game development guide](docs/guide/GAME_DEVELOPMENT_GUIDE.md).
- **ISMCTS** — root sampling + DAG + UCT2. A native design for hidden-info games. See [ALGORITHM_OVERVIEW.md](ALGORITHM_OVERVIEW.md).
- **AI API** — observation-only REST interface that third-party apps consume directly, without embedding engine code. Doubles as an information-theoretic proof that the AI never cheats. See [game development guide § 14](docs/guide/GAME_DEVELOPMENT_GUIDE.md#14-ai-api-分离验收--信息泄漏的唯一证明).
- **Training pipeline** — self-play → replay buffer → SGD → ONNX export → gating eval (≥60% win rate updates `best`). Includes heuristic guidance schedule (hold → linear decay → zero), auxiliary score, training action filter, MCTS schedule.

---

## Docs

- **[Features overview](FEATURES_OVERVIEW.md)** — what the framework can do
- **[Game development guide](docs/guide/GAME_DEVELOPMENT_GUIDE.md)** — single source of truth for adding a new game
- **[Framework contract & MCTS algorithm](ALGORITHM_OVERVIEW.md)** — the framework's data-flow contract (schema → walker → MaskedState → three consumers) and the ISMCTS DAG-search derivation, in one canonical doc
- **[New game test guide](docs/guide/NEW_GAME_TEST_GUIDE.md)** — 11-step acceptance workflow + the two-layer test architecture
- **[Known issues & trade-offs](docs/KNOWN_ISSUES.md)** — BUG-001 through BUG-036 postmortems plus design decisions (DEC-001 / DEC-002)

---

## Collaboration

This project targets **research-grade board-game AI engineering** and **digital board-game AI consulting**. If you need:

- A stronger AI opponent or AI coach for a digital board game
- Observation-only AI inference integrated into your gaming platform

open an Issue or Discussion.

---

## Directory layout

```
DinoBoard/
├── engine/                         # C++ general engine
│   ├── core/                       # Interface definitions
│   │   ├── game_interfaces.h       #   IGameState, IGameRules, IStateValueModel
│   │   ├── feature_encoder.h       #   IFeatureEncoder
│   │   ├── belief_tracker.h        #   IBeliefTracker (hidden info)
│   │   ├── game_registry.h         #   GameBundle, GameRegistrar
│   │   ├── types.h                 #   ActionId, StateHash64, UndoToken
│   │   └── action_constraint.h     #   IActionConstraint
│   ├── search/                     # Search algorithms
│   │   ├── net_mcts.h/.cpp         #   PUCT-MCTS + neural evaluation
│   │   ├── tail_solver.h/.cpp      #   Alpha-beta endgame solver
│   │   ├── root_noise.h            #   Dirichlet noise
│   │   └── temperature_schedule.h  #   Temperature decay schedule
│   ├── infer/                      # Inference
│   │   └── onnx_policy_value_evaluator.*  # ONNX Runtime inference
│   └── runtime/                    # Runtime
│       ├── selfplay_runner.*       #   Self-play loop + FilteredRulesWrapper
│       ├── arena_runner.*          #   Model-vs-model arena
│       └── heuristic_runner.*      #   Heuristic game generation
│
├── games/                          # Game implementations (one dir per game)
│   ├── tictactoe/  quoridor/  splendor/  azul/  loveletter/  coup/
│
├── bindings/py_engine.cpp          # pybind11 Python bindings
│
├── training/                       # Python training framework
│   ├── pipeline.py   model.py   cli.py
│
├── platform/                       # FastAPI web platform + AI inference API
│   ├── app.py                      #   Main server
│   ├── ai_service/                 #   Observation-only AI REST API
│   ├── game_service/               #   Game sessions, async pipeline, replay analysis
│   └── static/                     #   Shared frontend assets
│
├── tests/
│   ├── framework/                  #   Framework invariants (3-game matrix carrier)
│   ├── tictactoe/  quoridor/  splendor/  azul/  loveletter/  coup/
│   │                               #   Per-game complete acceptance checklists
├── docs/                           # Documentation
└── setup.py · requirements.txt     # Build
```

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

*Untrained* variants ship a randomly-initialized network — playable through the web UI but not competitive.
