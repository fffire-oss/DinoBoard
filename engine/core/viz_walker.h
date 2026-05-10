#pragma once

// Phase 1.5: viz_walker body — single canonical traversal of viz=1
// data-axis slots for a given perspective.
//
// The walker is a COORDINATE generator. It yields (field_name, idx,
// viz_tensor_ref) for every non-internal field × data-axis slot whose
// `viz[idx..., perspective] == 1`. It does NOT read typed payload —
// the game-side caller (hash builder, encoder, snapshot extractor)
// reads its own `state.foo[idx]` using the yielded `idx` and decides
// what to do with the value. Framework code does not touch typed
// fields here, consistent with the apply_viz_mask design (game owns
// its layout).
//
// Three eventual consumers, one traversal source:
//   - state_hash builder (Phase 3): walks visible slots, asks the game
//     to hash each by member-pointer (or just hashes the masked state
//     in bulk — that decision lives in Phase 3's per-game migration)
//   - encoder masker (Phase 3): walks slots → emits per-slot feature
//   - extract_snapshot (Phase 1.6): walks visible slots → fills the
//     wire payload's `values` map
//
// Sharing one traversal keeps "what counts as visible" defined in
// exactly one place. If a game declares a field's base_viz wrong, all
// three consumers diverge in lockstep — a single test catches it.

#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

#include "game_interfaces.h"
#include "visibility_schema.h"
#include "viz_runtime.h"

namespace board_ai {
namespace viz {

namespace detail {

// Recurse over data-axis indices in row-major order, calling `fn` for
// each index whose viewer-axis bit is set at `perspective`.
inline void walk_data_axes_recursive(
    const std::string& name, const VizTensor& v, int perspective,
    std::vector<int>& idx, int axis,
    const SlotVisitor& fn) {
  // Data-axis count = total rank - 1 (last axis is viewer).
  const int data_rank = v.rank() - 1;
  if (axis == data_rank) {
    // At a leaf data-axis index. Look up viz[idx..., perspective].
    // flat_offset_data_only returns offset of [idx..., 0]; viewer
    // stride is 1, so add `perspective`.
    const std::size_t base = flat_offset_data_only(v.shape, idx);
    const std::size_t off = base + static_cast<std::size_t>(perspective);
    if (v.data[off]) {
      fn(name, idx, v);
    }
    return;
  }
  for (int i = 0; i < v.shape[axis]; ++i) {
    idx[axis] = i;
    walk_data_axes_recursive(name, v, perspective, idx, axis + 1, fn);
  }
}

}  // namespace detail

// for_each_visible_slot — single canonical visit order:
//   for each non-internal field f in schema.fields (declaration order):
//     for each data-axis index idx in row-major order:
//       if viz_get(state, f.name)[idx..., perspective] == 1:
//         fn(f.name, idx, viz_tensor_ref)
//
// Stable across calls: order depends only on schema declaration order
// + field shape, never on hash-map iteration. Two states with the same
// schema produce identical visit sequences for the same perspective.
//
// Bounds: throws std::out_of_range if `perspective` is not in
// [0, viewer_count) for any visited field. Callers should clamp before
// calling — we don't silently no-op.
//
// Fields with empty base_viz are silently skipped (`continue` on
// v.empty() below). The walker has nothing to visit on a 0-element viz
// tensor, so an explicit "internal" flag would be redundant.
inline void for_each_visible_slot(const IGameState& state,
                                  const VisibilitySchema& schema, int perspective,
                                  const SlotVisitor& fn) {
  for (const auto& field : schema.fields) {
    const VizTensor& v = viz_get(state, field.name);
    if (v.empty()) continue;  // declared but no viz — pass through
    if (perspective < 0 || perspective >= v.viewer_count()) {
      throw std::out_of_range(
          "viz::for_each_visible_slot: perspective " +
          std::to_string(perspective) +
          " out of range for field '" + field.name + "' (viewer_count=" +
          std::to_string(v.viewer_count()) + ")");
    }
    const int data_rank = v.rank() - 1;
    if (data_rank == 0) {
      // Scalar field — viz shape is just [n_players]; no data-axis loop.
      const std::size_t off = static_cast<std::size_t>(perspective);
      if (v.data[off]) {
        std::vector<int> empty_idx;
        fn(field.name, empty_idx, v);
      }
      continue;
    }
    std::vector<int> idx(static_cast<std::size_t>(data_rank), 0);
    detail::walk_data_axes_recursive(field.name, v, perspective, idx, 0, fn);
  }
}

}  // namespace viz
}  // namespace board_ai
