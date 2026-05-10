#pragma once

// Phase 3.1: schema-driven hash helper.
//
// Replaces hand-written `hash_public_fields` / `hash_private_fields`
// bodies. The framework walks `state.viz_` per the visibility schema,
// classifies each visible slot as either "public" (every viewer can
// see it under runtime viz) or "private to perspective p" (p sees it
// but at least one other viewer doesn't), and dispatches the typed
// value back to the game via `IGameState::hash_field_slot(...)`.
//
// Why a per-game hook for typed values: C++ has no field reflection,
// so the framework can't read `state.board[idx]` from `IGameState&`.
// The game's hash_field_slot is a single, mechanical (name, idx) →
// `h.add(typed_value)` switch statement — one function per game,
// replacing the per-field hand-written hash sequences.
//
// Visit order (pinned by viz::for_each_visible_slot in viz_walker.h):
//   schema declaration order × row-major data-axis indices.
// Two states with the same schema and same viz_ produce identical
// visit sequences, so hash_public is a function of (observation
// history, viz_) only — exactly what BUG-028 prevention requires.

#include <stdexcept>
#include <string>
#include <vector>

#include "game_interfaces.h"
#include "types.h"
#include "visibility_schema.h"
#include "viz_runtime.h"
#include "viz_walker.h"

namespace board_ai {
namespace framework {

namespace detail {

// "All viewers see this slot under runtime viz" — the test for
// public-hash inclusion. Runtime viz can be wider than base viz
// (a reveal_slot turns a private slot fully public), so we read
// state.viz_, not schema's base_viz.
inline bool slot_visible_to_all(const viz::VizTensor& v,
                                const std::vector<int>& idx) {
  const std::size_t base = viz::flat_offset_data_only(v.shape, idx);
  const int n_viewers = v.viewer_count();
  for (int p = 0; p < n_viewers; ++p) {
    if (!v.data[base + static_cast<std::size_t>(p)]) return false;
  }
  return true;
}

}  // namespace detail

// Public-hash: every slot whose runtime viz is 1 for every viewer.
//
// Walker is stable (declaration order × row-major idx). Calls back
// into the game's hash_field_slot for typed value emission.
//
// Why perspective=0 in the walker call: the walker filters on
// viz[idx, perspective], but for the all-viewers-see check we
// re-read viz manually inside the visitor anyway. perspective=0
// is just to satisfy the walker's bounds — any p in [0, N) works
// because we never use the per-perspective filter result.
inline void hash_public_via_schema(const IGameState& state,
                                   const viz::VisibilitySchema& schema,
                                   Hasher& h) {
  viz::for_each_visible_slot(
      state, schema, /*perspective=*/0,
      [&](const std::string& name, const std::vector<int>& idx,
          const viz::VizTensor& v) {
        if (!detail::slot_visible_to_all(v, idx)) return;
        state.hash_field_slot(h, name, idx);
      });
}

// Private-hash for perspective p: every slot p sees that is NOT
// visible to all viewers (i.e. genuinely private to p's information
// set, not part of the public projection).
//
// public + private(p) together = exactly the slots p observes,
// which matches state_hash_for_perspective's contract.
inline void hash_private_via_schema(const IGameState& state,
                                    const viz::VisibilitySchema& schema,
                                    int perspective, Hasher& h) {
  viz::for_each_visible_slot(
      state, schema, perspective,
      [&](const std::string& name, const std::vector<int>& idx,
          const viz::VizTensor& v) {
        if (detail::slot_visible_to_all(v, idx)) return;
        state.hash_field_slot(h, name, idx);
      });
}

}  // namespace framework
}  // namespace board_ai
