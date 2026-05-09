#pragma once

// Phase 1.5: sim-local context helper.
//
// Encapsulates the four-step "MCTS sim entry" flow that Phase 3.7's
// bridge PR will route through:
//
//   1. state    = session.state.clone_state()
//   2. tracker  = session.tracker ? session.tracker->clone() : nullptr
//   3. belief_filled = zeros_like(state.viz_)
//   4. belief->sample(state, tracker.get(), perspective, belief_filled, rng)
//
// After the helper returns, `state` has every slot with a concrete
// value (viz=1 from the snapshot, viz=0 filled by sample), `tracker`
// is an independent copy that the descent can mutate via
// observe_in_sim, and `belief_filled` records which slots came from
// belief vs. snapshot — so the encoder can mask sample-derived slots
// alongside hidden ones.
//
// Phase 1.5 only ships the helper + a synthetic-session test. MCTS
// still uses the legacy `state = session.state.clone()` path. Phase
// 3.7 (after all 6 games declare schemas + tracker/belief) flips
// net_mcts.cpp to call this helper instead.
//
// Why this is a free function, not an MCTS member: keeps the search
// layer decoupled from the belief plumbing. A test can build a
// SimContext from any (state, tracker, belief, perspective) tuple
// without instantiating a full MCTS searcher.

#include <memory>
#include <random>
#include <string>
#include <unordered_map>

#include "../core/belief.h"
#include "../core/game_interfaces.h"
#include "../core/tracker.h"
#include "../core/visibility_schema.h"

namespace board_ai {

struct SimContext {
  // Sim-local clone of session.state; every viz=0 slot has been
  // overwritten by belief->sample (or left alone, if state.viz_ was
  // empty — fully-public games).
  std::unique_ptr<IGameState> state;

  // Sim-local clone of session.tracker, or nullptr if the session has
  // no tracker (fully-public games before Phase 3, and any game whose
  // GameBundle does not register a tracker).
  std::unique_ptr<ITracker> tracker;

  // Parallel to state->viz_, allocated zeros and updated by
  // belief->sample to mark slots written by belief (vs. surviving
  // viz=1 from the snapshot). Encoder consumes this alongside
  // state->viz_ to decide which slots to feed kPlaceholder. Empty
  // when state->viz_ is empty (no schema declared).
  std::unordered_map<std::string, viz::VizTensor> belief_filled;

  // The perspective this sim was determinized for. Fixed for the
  // lifetime of the sim — descent's `state->current_player()`
  // perspective rotates separately and is unrelated to belief.
  int perspective = 0;
};

// Build a sim-local context. Steps 1-4 in order; throws if `belief`
// is null and state has any viz=0 slots (a hidden-info game without
// a registered belief is a misconfiguration — not a soft fallback).
//
// `session_state` and `session_tracker` are read-only inputs; they
// are NOT mutated. If `session_tracker` is null the context's
// tracker is also null and belief->sample receives null, which is
// the contract for fully-public games whose belief is the no-op
// UniformBelief.
inline SimContext make_sim_context(
    const IGameState& session_state,
    const ITracker* session_tracker,
    const IBelief& belief,
    int perspective,
    std::mt19937& rng) {
  SimContext ctx;
  ctx.perspective = perspective;

  // Step 1: clone state (viz_ rides along inside clone_state).
  ctx.state = session_state.clone_state();

  // Step 2: clone tracker (or leave null).
  if (session_tracker != nullptr) {
    ctx.tracker = session_tracker->clone();
  }

  // Step 3: zeros_like(state.viz_). For each (name, viz) entry in the
  // session state's viz_, allocate a parallel VizTensor with the same
  // shape and all zeros. Empty viz_ → empty belief_filled.
  for (const auto& kv : ctx.state->viz_) {
    viz::VizTensor zeros;
    zeros.shape = kv.second.shape;
    zeros.data.assign(kv.second.data.size(), 0);
    ctx.belief_filled.emplace(kv.first, std::move(zeros));
  }

  // Step 4: belief->sample fills viz=0 slots and marks belief_filled.
  // For UniformBelief skeleton (Phase 1.3) on empty viz_, this is a
  // no-op — exactly what fully-public games need. The active body
  // lands in Phase 1.5+ (per-field reflection); calling it here is
  // safe regardless.
  belief.sample(*ctx.state, ctx.tracker.get(), perspective,
                ctx.belief_filled, rng);

  return ctx;
}

}  // namespace board_ai
