# DinoBoard Design Principles

## Training Pipeline

- Training is config-driven: only `game.json` parameters, no code changes per training run.
- Self-play loss not decreasing is normal (opponent also gets stronger). Don't diagnose it as a problem.
- Latest model always moves forward, never rolls back. Best model is tracked independently.
- Eval must use the same compiled code as selfplay. If code changes between build and eval, results are meaningless.
- The entire selfplay pipeline (selfplay, eval, arena) runs in C++. Never write Python fallback or reimplementation of any game logic or search logic. If you find yourself writing Python code that "does what the C++ does but slower", stop — that's a bug. The correct fix is always to expose the needed C++ functionality through pybind11.

## No Fallbacks, No Silent Degradation

- Never write fallback logic that masks errors. If something fails, throw/raise immediately — don't return a default value that makes the caller think it succeeded.
- Every function that receives data from C++ must access fields with `dict["key"]`, not `dict.get("key", default)`. Missing fields are bugs; defaults hide them.
- ONNX model is mandatory for selfplay, arena, and eval. Empty `model_path` is always an error. There is no uniform-policy fallback.
- `evaluate()` failures must throw, never return false. A silent `return false` caused BUG-011 (weeks of wasted training).
- `catch (...) { return false; }` is banned. Re-throw with context so the caller sees the real error.
- If you're writing code that makes a broken system "still run", stop. The correct fix is to make the system not broken.

## AI Pipeline Independence from Game State

The AI decision pipeline — belief tracking, feature encoding, MCTS search — works solely from observation history. It NEVER reads ground truth's hidden fields. **This is not a convention the AI is asked to respect — it is structurally impossible for the AI to cheat**, because:

1. **The belief tracker has no `IGameState*`.** `init(perspective, initial_observation)` and `observe_public_event(actor, action, pre_events, post_events)` take messages only. There is no pointer through which the tracker could reach truth.
2. **Session public state is rebuilt from messages, not from `do_action_fast`.** After each `apply_observation`, `public_state_applier` overwrites session state_'s public fields from `PublicEventTrace.public_snapshot` — so nothing the engine's `do_action_fast` computed from sampled hidden data can leak into the observer's public view.
3. **Session hidden state is re-sampled every ply.** At the end of `apply_observation`, `randomize_unseen` rewrites all hidden fields of session state_ to a fresh tracker-consistent sample. The session's hidden values are never a copy of truth — they are a belief sample — so even if downstream code accidentally reads them, it reads belief, not ground truth.
4. **Selfplay, web and API all use the same stack.** One `per_perspective_trackers[p]` per seat, init once, every public event fed to every tracker. MCTS for the acting player reads that seat's tracker. There is no path where the runner hands the network a tracker or state that still holds truth.

Together these mean the AI sees exactly what a physical player would: the message stream, nothing more. A bug that tries to leak information must first find somewhere for truth to hide — and the three pillars above leave no such place.

**Architecture**:

- **Ground truth** maintains game progress (our C++ engine, external API, or a physical tabletop game). Ground truth advances itself and sends messages to the AI. The AI advances its own dataset + tracker from these messages.
- **AI dataset** splits into **public fields** (all players see) and **private fields per player** (only that player sees). Declared via:
  ```cpp
  virtual void hash_public_fields(Hasher&) const = 0;
  virtual void hash_private_fields(int player, Hasher&) const = 0;
  ```
  Framework derives `state_hash_for_perspective(p) = hash(public + private_of_p)`. The same partition drives encoder feature extraction.

  **Public ≠ "every state member that isn't a hand."** Public fields are the union of "facts every player can derive from the observation history" — game-state facts that can be observed (face-up cards, scores, board layout) AND publicly derivable composition (multisets that everyone can compute by subtracting placed/discarded from a known starting pile). They explicitly do NOT include:

  - **Internal RNG state** (engine `rng_salt`, `mt19937` snapshots, deck-shuffle seeds, `bag` / `box_lid` ordering vectors before they're drawn). Nobody sees these. Hashing them splits the DAG along axes invisible to every player and unobservable to the encoder, so the same information set fragments into many DAG nodes that the network cannot tell apart. This is information leakage: the search behaves as if the AI knew the deck's future order. **It is the most insidious failure mode in this framework — the symptoms are "search runs, but is mysteriously weak / inconsistent / different from API path"**, never a crash. See [BUG-028].
  - **Future hidden draws** that have not yet been resolved into a public observation.

  Rule of thumb: if you cannot describe a `hash_public_fields()` value as "the public observation history up to this step," the field doesn't belong there. Order of cards in a face-down deck/bag → out. Counts of cards by color → in (everyone can derive these). When in doubt, hash *less*; missing fields show up as DAG transposition mistakes that tests catch, while spurious fields show up as silent strength regressions that nothing catches.
- **Tracker** maintains what a given perspective player has learned through legal observation. `init(perspective, initial_observation)` and `observe_public_event(actor, action, pre_events, post_events)` take NO `IGameState*` — the tracker physically cannot peek at truth.
- **`randomize_unseen(state, rng)` contract**: produces a world whose `hash_public_fields` is byte-equal across any two trackers with the same observation history, regardless of the input state's hidden contents or the caller's RNG. Hidden multisets (deck sizes, bag sizes, court-deck size, etc.) are DERIVED from tracker's seen information, not preserved from the input state. Stale samples (opp hidden fields whose cid later became publicly seen) are re-sampled from the current unseen pool. Called in two places: (a) per-sim at MCTS root for determinization, (b) at the end of every `GameSessionWrapper::apply_observation` to freshen session state_ so hidden fields are a fresh tracker-consistent sample, never a copy of truth.
- **API / web session freshening**: `py_engine::apply_observation` runs `randomize_unseen` on `bundle_->state` immediately after `observe_public_event`, with a deterministic RNG derived from `(seed_, ply_count_)`. Session hidden fields are therefore re-sampled every ply from the tracker's information set — they are not a copy of truth, and nothing in the AI pipeline relies on them matching truth.
- **Message-driven public state**: for each hidden-info game, the `public_event_extractor` populates `PublicEventTrace.public_snapshot` with a truth-side dump of every field in `hash_public_fields`. The game also registers `GameBundle::public_state_applier` that inverts this snapshot onto a state. In `apply_observation`, after `do_action_fast + post_events`, the applier **overwrites** session state_'s public fields from the snapshot — session public state is rebuilt from the message stream, not derived from `do_action_fast`'s (possibly hidden-dependent) output. Games just implement the snapshot schema (round-trip-tested via `test_public_snapshot_round_trip`); there are no truth-sync patch events.
- **Per-perspective trackers in every path**: selfplay, arena, web and API all allocate one tracker per seat, init once at game start, and feed every public event to every tracker. MCTS for the acting player uses `per_perspective_trackers[current_player]`, which has accumulated the full observation history for that seat. There is no per-ply re-init of a shared tracker.
- **MCTS** uses root determinization: each simulation begins with `tracker->randomize_unseen(sim_state, rng)` — tracker fills certainties (e.g. known opp hand from Priest peek), rest of unseen pool sampled per tracker's policy (uniform for simple cases; heuristic weighted for bluff-heavy games like Coup where claim/challenge history biases opp role priors), remainder shuffled into deck. After sampling, descent is fully deterministic within that world.
- **Node hashing in MCTS** uses `state_hash_for_perspective(state.current_player())` — each decision node is keyed by the acting player's information set. This forms a **DAG** (transpositions reached by different paths share nodes via a global `hash → node_index` table).
- **DAG acyclicity** is guaranteed structurally by `IGameState::step_count` — a framework-provided counter that increments on every `do_action_fast`, included in public hash. No two states in the DAG share a hash unless they have the same step count.
- **UCB** uses UCT2 (DAG-aware): the `sqrt(parent.visit_count)` term uses the visit count of the specific incoming edge traversed in this simulation, not the DAG node's global visit count. Avoids over-exploration bias from multi-parent aggregation.
- **Encoder**: extracts features only from `public_fields + current_player's private_fields`. The public/private partition is the single source of truth for both hash and encoder scope — they stay aligned structurally.

**Validation** (mandatory for every game, enforced in CI):

- `tests/framework/test_ai_api_separation.py::test_full_game_via_api[<game>]` — the API carries no state fields in/out. The AI drives a full game from observations alone.
- `tests/framework/test_api_belief_matches_selfplay.py::*[<game>]` (stochastic games) — AI session seeded differently from ground truth produces identical belief + public state + legal actions after the same observation stream.
- `tests/framework/test_api_mcts_policy_invariance.py::*[<game>]` — MCTS visit distribution on the same observation history is identical across selfplay and API paths. If the AI secretly reads truth, distributions diverge.
- `tests/framework/test_encoder_respects_hash_scope.py::*[<game>]` — encoder output is bit-equal when opp private changes but (public + own private) stays the same.
- `tests/framework/test_public_hash_excludes_internal_rng.py::*[<game>]` (60-seed sweep over 4 hidden-info games) — two API sessions with different RNG seeds produce identical `state_hash_for_perspective` throughout the same observation trace. Catches any "public output secretly depends on session-RNG-specific hidden field" regression.
- `tests/framework/test_session_hidden_fields_resampled.py::*[<game>]` — session state_'s hidden fields are re-sampled at end of each `apply_observation`, not static. Confirms the freshening actually runs. Callers relying on "get_state_dict hidden values == truth" are breaking the contract — session hidden is a tracker-consistent sample, not truth.
- `tests/framework/test_public_snapshot_round_trip.py::*[<game>]` — for each hidden-info game with `public_state_applier` registered: every ply, truth→extract snapshot→observer session→apply snapshot → observer's `state_hash_for_perspective(own)` byte-equal truth's. Guards the public_state_applier stays in sync with the extractor + hash_public_fields as new game state fields are added.
- `tests/framework/test_dag_acyclic.py::*[<game>]` (current filename: `test_dag_reuse.py`) — no cycles in the MCTS DAG, asserted via step_count monotonicity.

**No chance nodes**: physical randomness (deck draws, dice) is handled entirely by root determinization — different simulations sample different worlds, and different observer-visible outcomes automatically produce different hashes → different DAG nodes. No special chance-node machinery (NoPeek / traversal limiter / `chance_outcomes` / afterstate cap / `stochastic_detector`) exists in the framework.

## Value Head / Multiplayer

- Value head outputs N-dimensional value (N = num_players), not scalar. Even 2-player games output 2 dims.
- Value output is perspective-relative, aligned with the encoder's feature rotation: `values[0]` = my value, `values[1]` = next player, etc.
- The ONNX evaluator rotates the N-dim output back to absolute player ordering for MCTS backup.
- `IStateValueModel::terminal_values()` must satisfy zero-sum: `sum(values) ≈ 0`. The network learns this from targets, not from architectural constraint.
- Training targets use `sample["z_values"]` (per-player vector) rotated to perspective order, not the scalar `sample["z"]`.
- `game_metadata(game_id)` returns `{num_players, action_space, feature_dim}` from C++ — always prefer this over hardcoded JSON values for variant-aware code.

**Legacy scalar value head compatibility (2-player only).** Older 2p models (`tictactoe`, `quoridor` 2p, `splendor` 2p, `azul` 2p, `loveletter` 2p, `coup` 2p) were trained with a `[1, 1]` scalar value head, where the single output is the perspective player's value in [-1, 1]. The full 2p AI chain — selfplay, eval, web gameplay (winrate pill), replay analysis, smart hints (drop-score) — must continue to load and run these models without retraining. Compatibility lives in exactly one place: `OnnxPolicyValueEvaluator::evaluate`'s `value_len == 1 && num_players == 2` branch (`engine/infer/onnx_policy_value_evaluator.cpp`), which expands the scalar `v` into `(v_perspective, -v_opponent)` by zero-sum and returns a length-2 `values` vector indistinguishable downstream from an N-dim model. Everything past the evaluator (MCTS leaf backup, `root_values`/`action_values` bindings, `pipeline.py` analysis) is dimension-agnostic and needs no scalar-aware code.

This compatibility does **not** extend to 3p/4p: `value_len == 1 && num_players > 2` throws — there is no zero-sum decomposition that pins individual seats, and silently broadcasting would violate "no silent degradation." When refactoring the value-output decoding path, the scalar 2p branch must be preserved or migrated explicitly; deleting it silently breaks every shipped 2p model.

## Game Architecture

- Engine is fully game-agnostic. All game-specific logic lives in the game's GameBundle registration.
- One game = state + rules + net_adapter + register + config/game.json + web frontend.

## Web Frontend Design

**Before touching any web frontend code — new game or modification — read `docs/WEB_DESIGN_PRINCIPLES.md` and follow it. It is mandatory, not a style suggestion.** When you design a new game frontend, update the doc if you discover a new principle worth codifying for the next game; don't carry the lesson only in your head.

Full visual/interaction guide: `docs/WEB_DESIGN_PRINCIPLES.md`.

- Web frontend is mandatory. It is the primary interface for players to play and for the developer to verify training results.
- Actions must NOT be mapped naively to individual buttons. Design interactions as the player would naturally play the physical game.
- Spatial anchoring: fixed game regions must have fixed screen positions and sizes. Never let a container shrink or shift when its contents change.
- Every action (human or AI) must have animated transitions via `describeTransition`. No instant state jumps.
- Visual clarity: sufficient color contrast, clear current-player indicator, hover feedback on interactive elements.
- Never hold a global lock (mutex, SQLite write lock, etc.) while running heavy computation (MCTS search, batch inference). Heavy work goes to a thread pool; locks are session-scoped and held only for state reads/writes.
- The Python GIL counts as a global lock. Any pybind11 binding that performs meaningful C++ work (MCTS, ONNX loading/inference, batch encoding) MUST wrap that work in `py::gil_scoped_release`. Without it, offloading the call to a ThreadPoolExecutor does nothing — the worker thread acquires the GIL for the entire C++ call and stalls every other Python thread, including the uvicorn request loop, causing the UI to freeze for seconds per move.

## Documentation

When implementing a new feature or fixing a bug, update documentation immediately:

- **README.md** — keep concise; only mention the feature exists, don't explain implementation details.
- **docs/GAME_FEATURES_OVERVIEW.md** — high-level "what's available" for developers. Training pipeline, search, decision-making, training enhancements, eval, web frontend, randomness handling, optional component reference, config reference, and new game development steps. Start here for a quick overview of what the framework can do.
- **docs/GAME_DEVELOPMENT_GUIDE.md** — detailed implementation guide. Covers IGameState, IGameRules, IFeatureEncoder, GameBundle registration, GameRegistrar patterns, game.json config format (all fields), CMake/setup.py build integration, all 12 optional components with signatures and examples, feature encoding best practices, and web frontend integration (createApp API, ctx/gameState objects, common.js utilities). This is the single source of truth for "how to add a new game."
- **docs/KNOWN_ISSUES.md** — bug postmortems and design trade-off records. BUG-001 through BUG-022 covers every shipped regression: tail solver TT flags, draw z-value, train-eval action space mismatch, FilteredRulesWrapper const_cast, replay buffer utilization, feature encoding pipeline bug, Splendor temperature schedule, replay buffer loss, ONNX silent degradation, model export order, z_values incomplete, legal mask filter, belief tracker peeking, adjudicator z_values + 3p+ evaluator, 2p-hardcoded multiplayer paths, pipeline stats-key mismatch, fly animation inheriting container size + sequential playback, cancel_pipeline side-effect wiping precompute cache. Plus general pitfalls and design decisions. Read this before writing new game logic or modifying the pipeline.
- **docs/NEW_GAME_TEST_GUIDE.md** — step-by-step verification checklist for new game implementations. 9 steps: registration + config consistency, GameSession interaction, do/undo consistency, feature encoding (BUG-007 regression), selfplay sample integrity, ONNX round-trip, training tensor validation, optional component verification (heuristic, tail solver, filter, adjudicator, auxiliary scorer, hidden info), and multiplayer variants. Includes instructions for joining the existing 600+ parametrized test suite.
- **docs/devlog/YYYY-MM-DD.md** — daily development log. Record what was implemented, key decisions made, config changes, and training observations. Keep entries concise and factual.
