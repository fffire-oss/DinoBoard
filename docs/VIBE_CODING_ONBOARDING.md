# DinoBoard — AI Onboarding

> Read this **after** `README.md`. **This document already absorbs the necessary content from `../FEATURES_OVERVIEW.md`** (six-file game layout, ISMCTS properties, optional-component reference, two-layer testing, framework non-goals) — do not read that file, it is a human-facing capability tour that will burn context without telling you how to actually work in the repo. Everything an AI assistant needs to start working is below; deep-dive docs are linked per task.

All paths in this document are repo-relative. Treat the repo root as `$REPO`.

---

## 1. What this codebase is, in 90 seconds

AlphaZero-style framework for turn-based tabletop games. C++ engine (MCTS, ONNX inference, selfplay, arena, tail solver); Python is glue (training loop, FastAPI web server, pybind11 bindings).

Six games shipped: `tictactoe`, `quoridor`, `splendor`, `azul`, `loveletter`, `coup`. They cover four paradigms — perfect-info / public-symmetric-randomness / asymmetric-hidden-info / bluffing — and **between them exercise every framework feature**. When adding a new game, find the closest existing one and mirror it; do not invent new patterns.

A complete game = six files:

```
games/<g>/
  <g>_state.cpp         IGameState
  <g>_rules.cpp         IGameRules (legal/do/undo, optional do_action_deterministic)
  <g>_net_adapter.cpp   IFeatureEncoder + IBeliefTracker (latter only for hidden-info)
  <g>_register.cpp      GameBundle factory + GameRegistrar
  config/game.json      training hyperparams (sims, lr, network shape, optional components)
  web/<g>.js            createApp(...) frontend
```

`engine/` is fully game-agnostic. All game-specific logic lives behind these interfaces and is wired in by `<g>_register.cpp`.

**Training pipeline shape** (config-driven via `game.json`, no per-run code changes):
`selfplay → replay buffer → SGD → ONNX export → gating eval (latest vs best, ≥60% win rate updates best) → repeat`. Selfplay, arena, and gating eval all run in C++ via the same MCTS as serving. Python is the training loop and the network's SGD step.

**Three serving surfaces, one stack**: web play, third-party `platform/ai_service` REST API (observation-only — see `docs/guide/AI_API.md`), and replay analysis all call the same C++ NetMCTS that selfplay does. The win-rate number a player sees on the web is literally the value head's output during training.

---

## 2. Hard rules (PRs that violate get rejected)

**No silent fallback.**
- `catch (...) { return false; }`, `dict.get(key, default)` masking missing fields, `evaluate()` failure swallowed into `false` — all banned. Failures throw with context.
- Read config with `cfg["key"]`, never `cfg.get(key, default)`. Missing key = bug, not default.

**Never reimplement C++ logic in Python.**
- The whole pipeline (selfplay, arena, MCTS, tail solver) lives in C++. If something Python-side needs C++ behavior, expose it through pybind11. "Same thing but slower in Python" is always a bug.

**Visibility scope is a structural constraint, not a guideline.**
- `IBeliefTracker::init` and `observe_public_event` signatures **do not take `IGameState*`** — never add one. The tracker physically cannot peek at truth.
- `IFeatureEncoder::encode_features(masked_state, perspective, tracker, out)` takes a `const IGameState& masked_state` that the framework's non-virtual `IFeatureEncoder::encode` produced via `make_masked_state` — so by convention every override sees a clone whose viz=0 slots arrive as `kPlaceholder*`. Encoder branches on the placeholder, **never** queries `viz` directly and **never** dereferences any other player's private field through a back door.
- `state_hash_for_perspective(p)` walks the visibility schema and calls `hash_field_slot(h, name, idx)` for every slot with `viz[..., p] = 1`. Anything no player can observe — internal RNG state, unrevealed deck order — must not be reachable from any schema slot's viz=1 path; if it is, the DAG splits along an invisible axis and search silently weakens. (BUG-028 is the canonical violation: hashing the deck-shuffle seed; symptom is "AI is mysteriously weak/inconsistent," never a crash.)

**Value head is N-dim perspective-relative.**
- N = `num_players`, even for 2p. `values[0]` is the acting player. Targets and predictions both rotate against the encoder's perspective.
- Legacy 2p scalar ONNX models still work via a compatibility branch in `engine/infer/onnx_policy_value_evaluator.cpp` (`value_len == 1 && num_players == 2`). **Do not delete it** — it ships in every 2p model. The branch errors out for 3p+, which is correct.

**Web/inference must release the GIL.**
- Any pybind11 binding that does meaningful C++ work (MCTS, ONNX) **must** wrap in `py::gil_scoped_release`. Without it, ThreadPoolExecutor offloading is a no-op and the UI freezes per move.

**Tests cannot be skipped past.**
- `pytest.skip` / `pytest.xfail` to make a red test green is cheating. Either fix or delete.

**Adding a new game: every `.cpp` MUST be listed in `games/manifest.json`.**
- The manifest is the single source of truth — `setup.py` and the top-level `CMakeLists.txt` both read it. Forgetting to register a game's `.cpp` is the most common silent failure: build passes, `import` passes, but `available_games()` doesn't see the new game (its `GameRegistrar` static was never linked in).

---

## 3. ISMCTS in one screen

You do not need to read `../ALGORITHM_OVERVIEW.md` unless you are modifying the framework / search itself. But you must not break these properties when touching anything adjacent.

**Seven interlocking properties, all required:**

1. **Root-sampling determinization.** Each simulation holds an independent RNG; the root step calls `tracker->randomize_unseen(state, observer, rng)` to sample a complete world from the observer's belief, and descent continues to draw from that same RNG only for `do_action_fast` physical randomness (e.g. Azul factory refill). Given the sim's seed, the whole rollout is reproducible. There are **no chance nodes**.
2. **Per-acting-player node keying.** Each decision node is keyed by `state_hash_for_perspective(state.current_player())` — a schema-driven walk over every slot whose `viz[..., acting-player] = 1`, plus `step_count`.
3. **DAG, not tree.** A global `unordered_map<hash, node_idx>` shares info-set nodes across paths. Visit/Q stats aggregate naturally.
4. **UCT2 UCB.** `sqrt()` numerator uses the **incoming edge's** visit count, not the DAG node's global visit count. Plain UCT1 over-explores in DAGs by ~√2.
5. **Step-counter acyclicity.** `IGameState::step_count_` increments on every `do_action_fast` and is folded into `state_hash_for_perspective` automatically. Two states never share a hash unless they share step count → DAG is structurally acyclic.
6. **Encoder aligned with hash scope.** Encoder reads exactly the slots that hash sees — i.e. those with `viz[..., perspective] = 1` in the masked clone the framework produced. Same partition is hash scope and encoder scope; DAG nodes ⇄ network features stay 1:1.
7. **Same MCTS for selfplay, web, API.** Selfplay, arena, web AI, and the observation-only REST API all run the identical C++ search. Session public state is rebuilt from the message stream via `public_state_applier`; hidden slots on the session are never freshened (per DEC-003) — only sims clone the tracker and call `randomize_unseen` at sim entry. The API is **structurally unable to read truth**.

**Three game types, one algorithm:**

| Type | Example | What gets registered |
|------|---------|----------------------|
| Public + deterministic | TicTacToe, Quoridor | state / rules / encoder only |
| Public + symmetric random | Azul | state / rules / encoder; no tracker (physical randomness lives on public counts, sim_rng samples on the fly inside `do_action_fast`) |
| Asymmetric hidden info | Splendor, Love Letter, Coup | + `belief_tracker` + visibility schema with owner-only fields + per-slot `hash_field_slot` / `mask_field_slot` / `read_field_slot` / `write_field_slot` dispatchers. Wire protocol unifies opening and per-ply on one walker: opening = `viz::serialize_public_snapshot` (full slot set; emits `(idx, value)` pairs for viz=1 slots and the perspective's viz slice under `__viz__`) + `tracker.pack_init_payload` (perspective-private bootstrap) → applied via `viz::apply_public_snapshot` + `tracker.init`; per-ply = `public_event_extractor` / `public_state_applier` + `tracker.observe_public_event`. Snapshot-path games wire all three extractors via one `b.install_event_protocol(events_only_diff, schema_provider)` call in `register.cpp` — the framework auto-derives `events_only_extractor` (sim path) / `public_event_extractor` (events + framework's `serialize_public_snapshot`, wire/selfplay/trace) / `public_state_applier` (framework's `apply_public_snapshot`, receiver session). Pass `viz::no_events_extractor` if the tracker is event-free (pure viz inference). **Learned-belief variant** (Coup today) additionally registers `belief_feature_extractor` + `belief_label_extractor` + `belief_model_path` so the framework runs an independent belief net at MCTS root and caches `pi[opp][R]` for sim sampling — opt-in only when uniform `randomize_unseen` degenerates (bluffing games). See §4 and `games/coup/BELIEF_NETWORK.md`. |

---

## 4. Optional components quick reference

Registered fields on `GameBundle`. None are required by the framework; opt in only if the game needs them.

| Component | Use when |
|-----------|----------|
| `belief_tracker` | Game has **asymmetric hidden info** (Love Letter / Splendor / Coup) and therefore viz=0 slots that MCTS sim entry must determinize. Public physical randomness alone (Azul) does **not** need a tracker — sim_rng samples directly inside `do_action_fast`. Tracker also implements `pack_init_payload(gt_state, perspective)` for the opening bootstrap (perspective-private content viz=1 reveals can't carry, e.g. LL starting hand). |
| `public_event_extractor` / `applier` / `public_state_applier` | Snapshot-path game (hidden-info **or** Azul) — diff truth into events for the message stream and rebuild observer-side public state from them. All three are wired via a single `b.install_event_protocol(events_only_diff, schema_provider)` call; the framework auto-derives the three from your `events_only_diff` plus its own `viz::serialize_public_snapshot` / `viz::apply_public_snapshot` halves. Pass `viz::no_events_extractor` if the tracker is event-free. |
| `belief_feature_extractor` + `belief_label_extractor` + `belief_model_path` | **Learned-belief** path (Coup today) — register all three together; framework loads `OnnxBeliefEvaluator`, runs one inference at MCTS root, caches `pi[opp][R]`, and sims do Wallenius sampling instead of `randomize_unseen` uniform/hand-craft. Use only when uniform sampling degenerates on the game (bluffing). `belief_label_extractor` is the **GT-side-only** target producer — physically unreachable from AI session / wire / web / API; that's the no-truth-leak red line. Each N-player variant ships its own `<g>_belief_<N>p.onnx`. |
| `tail_solver` / `tail_solve_trigger` | Want exact endgame solving and a smart trigger for when to fire it |
| `heuristic_picker` | Hand-written scorer to bootstrap selfplay (three-stage schedule: hold → linear decay → 0) |
| `auxiliary_scorer` | Extra learning signal beyond win/loss (e.g. score margin) |
| `training_action_filter` | Action space has obvious garbage worth pruning during training only; `legal_mask` stays full |
| `adjudicator` | Game can loop — declare a winner after `max_game_plies` |
| `episode_stats_extractor` | Track game-specific metrics during selfplay |

Full schema and per-field semantics: `docs/guide/CONFIG_REFERENCE.md` and `docs/guide/GAME_DEVELOPMENT_GUIDE.md`.

---

## 5. Tests are two-layer

- **`tests/framework/`** — framework invariants. Runs on a fixed three-game carrier `["quoridor", "azul", "loveletter"]` (covers deterministic / symmetric-random / asymmetric-hidden minimally). **You do not add games to this carrier.** Adding a game does not require touching `tests/conftest.py::FRAMEWORK_GAMES`.
- **`tests/<g>/`** — per-game complete acceptance checklist. `pytest tests/<g>/` going green is the **definition** of "this game is done." Includes `TestRuleInvariants` driven by `run_random_episode_states` to assert game-specific conservation laws (token totals, capacity bounds, etc.).

To add a game's checklist: copy `tests/<closest-existing-game>/test_checklist.py`, change `GAME = "..."`, iterate against test failures.

Hidden-info games additionally need green: `test_tracker_consistent_with_truth`, `test_ismcts_samples_respect_tracker`, `test_public_snapshot_round_trip`, `test_api_belief_matches_selfplay`, `test_api_mcts_policy_invariance`, `test_encoder_respects_hash_scope`, `test_public_hash_excludes_internal_rng`.

These framework tests are NOT all parametrized over `FRAMEWORK_GAMES = ["quoridor", "azul", "loveletter"]`. That constant is the **fixed three-game carrier** for invariants that need representatives of each category (deterministic / public-random / asymmetric-hidden) — you do not edit it when adding a game. **Capability-driven tests** (anything that requires a tracker, snapshot, or hidden info) build their own parametrize lists from `games/manifest.json` capabilities — `test_public_snapshot_round_trip` runs over all `"snapshot"`-capable games, `test_api_mcts_policy_invariance` is currently Love Letter only by `LEAK_SENSITIVE_GAMES = ["loveletter"]` (Splendor excluded for a known replay issue), `test_public_hash_excludes_internal_rng` runs over hidden-info games. Adding a game to the manifest with the right capabilities is what auto-enrolls it; touching `FRAMEWORK_GAMES` is not.

---

## 6. Baseline check before touching anything

**Prerequisite**: ONNX Runtime must be installed (`brew install onnxruntime` on macOS; tarball under `third_party/onnxruntime-linux-*` on Linux). The build refuses to proceed without it — there is no uniform-policy fallback.

```bash
python3 setup.py build_ext --inplace 2>&1 | tail -20
pytest tests/ -q 2>&1 | tail -5
```

Baseline must be green before you start. **Never edit on a red baseline** — you cannot tell which failures you introduced.

---

## 7. Task routing

| Task | Required reading | Reference |
|------|------------------|-----------|
| **Add a new game** | `docs/guide/GAME_DEVELOPMENT_GUIDE.md` (search the relevant section, do not read top-to-bottom) | `docs/guide/CONFIG_REFERENCE.md`, `docs/guide/NEW_GAME_TEST_GUIDE.md`, `docs/KNOWN_ISSUES.md` (general pitfalls + BUG-017 + BUG-023) |
| **Modify MCTS / search / belief / framework contract** | `../ALGORITHM_OVERVIEW.md` + `CLAUDE.md` "AI Pipeline Independence" section | `docs/KNOWN_ISSUES.md` framework-layer BUGs and all DEC entries |
| **Modify training / ONNX / pipeline** | `CLAUDE.md` (full) | `docs/KNOWN_ISSUES.md` framework-layer BUGs |
| **Modify web frontend** | `docs/guide/WEB_DESIGN_PRINCIPLES.md` + `docs/guide/WEB_DEVELOPMENT_GUIDE.md` | — |
| **Modify AI API / third-party integration** | `docs/guide/AI_API.md` + `docs/games/<game>_api.md` for the relevant game | `CLAUDE.md` "AI Pipeline Independence" section |
| **Fix a bug** | Search `docs/KNOWN_ISSUES.md` first — it may already be a known issue or a deliberate trade-off | — |

---

## 8. Adding a new game: 5 steps

**Step 1 — Get the rules.** Try in this order, never guess from the game name (one wrong rule wastes the entire training run):

1. Files the user dropped in `games/<X>/` (PDF / Markdown / images)
2. WebFetch a known official URL
3. WebSearch `<game name> rules` — **cross-verify multiple sources**, hobbyist sites often have errors
4. Ask the user

Then **paraphrase the core mechanics back to the user in 3–5 sentences** (player count, turn structure, win condition, presence of hidden info / randomness) and wait for confirmation.

**Step 2 — Pick template, scaffold first.** Closest existing game:

- Hidden info → Love Letter / Coup / Splendor
- Pure deterministic → Quoridor
- Smallest closed loop → TicTacToe

Write empty stubs for all six files. `legal_actions` returns `[0]`; `do_action` is a no-op. Add an entry for the game in `games/manifest.json` (id + sources list). Run `python3 setup.py build_ext --inplace` and verify `dinoboard_engine.available_games()` shows the new game. **Only fill in real logic after this passes.** Do not write 2000 lines before the first compile.

**Step 3 — Fill in implementation, write tests.** Copy `tests/<closest>/test_checklist.py`, change `GAME = "..."`, add empty `tests/<X>/__init__.py`. Iterate small: implement a slice → `pytest tests/<X>/` → next slice.

**Step 4 — Acceptance.**
- `pytest tests/<X>/ -v` all green = "the game is done"
- `pytest tests/ -q` all green = framework invariants intact

**Step 5 — Wire-up + docs.**
- Multiplayer variants (`<g>_3p` / `<g>_4p`) each need an independently-trained ONNX model. **Learned-belief games (Coup-style) ship two ONNX per variant**: `<g>_<N>p.onnx` (PV) and `<g>_belief_<N>p.onnx` (belief). Each variant inits and trains the belief net independently — the PV-net rule applies to belief too.
- `game.json` `feature_dim` / `action_space` **must** match `engine.game_metadata(game_id)` (C++ is canonical; if they disagree, fix the JSON)
- Write `docs/devlog/<today>.md` with key decisions (encoder dim choice, tracker design)
- New pothole encountered → add an entry to `docs/KNOWN_ISSUES.md`

---

## 9. After modifying encoder or feature_dim

1. Sync `games/<g>/config/game.json::feature_dim`
2. Retrain and deploy ONNX to `games/<g>/model/<variant>.onnx`
3. `tests/framework/test_deployed_models_match_encoder.py` will block — never skip it

Symptom of skipping this: web UI shows "AI thinking" forever, never plays a move (input shape mismatch).

---

## 10. Confirm with the user before doing these

Default is "just do it"; confirm only when:

- Changing interface signatures of `IBeliefTracker` / `IFeatureEncoder` / `IGameState` / `IGameRules`
- Modifying core flow of MCTS or the ONNX evaluator
- Deleting a function that "looks unused" — pybind11 may bind it from Python
- Task description has multiple plausible interpretations
- Game-rule details are uncertain

---

## 11. After every change

```bash
python3 setup.py build_ext --inplace
pytest tests/ -q
```

Add or change a feature → update the relevant doc. Fix a bug → add a `docs/KNOWN_ISSUES.md` entry (copy an existing entry's template).

---

## 12. High-frequency failure modes (what AI assistants get wrong)

- **Over-engineering.** Adding abstractions for hypothetical future use. Rule: three similar sites beat one premature abstraction.
- **Reimplementing C++ in Python**, even "just for testing." No.
- **Reading config with `.get(k, default)`** — masks missing-field bugs.
- **Skipping a failing assertion via `pytest.skip`.** That's not fixing the test.
- **Forgetting to register a new game in `games/manifest.json`.** Most insidious failure — no build error, no import error, but the game is invisible.
- **Guessing rules from the game name.** One wrong rule and the whole training run is wasted.
- **Writing hundreds of lines before the first compile.** Scaffold, build, register, then fill.
- **Deleting the legacy 2p scalar value-head branch.** Breaks every shipped 2p ONNX.
- **Bypassing `gil_scoped_release` in pybind bindings.** Multi-second UI freezes per move.
- **Letting a slot whose `hash_field_slot` reaches internal RNG state / unrevealed deck order be visible (`viz=1` to anyone).** Silently weakens search; no test catches it directly except `test_public_hash_excludes_internal_rng`. (BUG-028.)

---

## 13. Framework non-goals (don't try to make these work here)

These are deliberate scope cuts; pushing past them needs a separate framework, not patches.

**If the developer asks you to add a game that falls into one of these categories, stop and tell them before writing any code.** Do not silently try to "make it work somehow" — the framework's assumptions (deterministic policy, state-snapshot encoder, fixed action space, terminal-reward training, single-info-set search) cannot be patched around. Name the specific non-goal that applies, explain in one sentence why the framework can't deliver competitive play for it, and ask whether they want to (a) drop the game, (b) implement a reduced variant that fits the framework (e.g. a fixed-deck Hearthstone subset instead of full deckbuilding), or (c) accept a known-weak AI. Only proceed once they've chosen.

These conflict with the framework's core paradigm assumptions (AlphaZero / ISMCTS / zero-sum self-play / god-view viz) — not solvable by adding helpers or tuning. Hit one and you switch frameworks.

1. **Mixed-strategy equilibria required** (Texas Hold'em) — AlphaZero trains deterministic policies and cannot converge to a precise mixed Nash. Use CFR / DeepCFR family instead.
2. **Combinatorial action explosion** (DouDizhu, ~27k play combos) — a fixed `action_space` is both sparse and inefficient; needs hierarchical action decoding (see DouZero).
3. **Deck construction** (MTG) — two layers of difficulty: (a) the framework only solves "how to play a given deck"; deckbuilding itself is an independent meta-game; (b) the construction action space is combinatorial (choose K from N candidates, `C(N, K)` blow-up), stacking on top of #2 — fixed action slots cannot express it; needs hierarchical / autoregressive action decoding.
4. **Zero-sum only** (Diplomacy and other games requiring negotiation / coalitions / shared payoffs don't fit) — the entire training stack is built on a zero-sum assumption: value head enforces `sum(values) ≈ 0`; selfplay/arena pit candidate vs opponent one-on-one; gating only looks at win rate. In cooperative or general-sum games, "winning" has no single definition, multiple Nash equilibria can coexist, the Pareto frontier is not unique, and pure selfplay tends to converge to uncooperative "safe" strategies. Fixing this requires redesigning the training objective (joint value / coalition formation) and evaluator (N independent rewards instead of zero-sum-constrained); not a knob-tuning fix.
5. **Single-player / solitaire games unsupported** (Klondike, 2048) — selfplay bootstrap relies on "both sides being roughly equally bad still produces wins and losses": early random networks face equally random opponents, so win/loss signals arise naturally and `z_values` give a training target with variance. Single-player games have no opponent; victory is rule-defined (assemble a specific pattern / hit a score threshold), and under a random policy the probability of triggering victory is near zero — the whole episode is `z=-1`, the network gets no gradient on "which move is closer to winning," and training never gets off the ground. Fixing this requires reward shaping (score by progress to the goal) or curriculum learning (start from simplified positions); out of scope for this framework. Use intrinsic motivation / curiosity-driven algorithms instead.
6. **viz is a public god-view function — no support for "eyes-closed" phases** (Werewolf night phase, secret-write actions) — the framework assumes "which slot is visible to whom" is a uniquely-determined game rule under god view: every player knows "I see X, I don't see Y, the opponent sees Z," they just don't know the contents of Y / Z. Eyes-closed phases violate this: players **don't know whether opponents acquired information at a given moment** (werewolves opening eyes at night to recognize each other → villagers don't know whether werewolves share identities; a moderator privately telling some player a card → others don't know that disclosure happened). This kind of "viz nested on hidden" higher-order uncertainty cannot be written as a deterministic `base_viz`, nor as a deterministic reveal inside `do_action_fast`. Fixing it requires modeling "did the opponent acquire information" as a second-order probability distribution inside the belief formula, which conflicts with the current "viz is a public rule function, single source of truth" design — not patchable with a helper. See [FRAMEWORK_DESIGN_RATIONALE.md §3.2](FRAMEWORK_DESIGN_RATIONALE.md).
