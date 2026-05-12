#pragma once

// viz_walker — single canonical traversal of viz=1 data-axis slots
// for a given perspective.
//
// The walker is a COORDINATE generator. It yields (field_name, idx,
// viz_tensor_ref) for every non-internal field × data-axis slot whose
// `viz[idx..., perspective] == 1`. It does NOT read typed payload —
// the game-side caller reads its own `state.foo[idx]` using the
// yielded `idx`. Framework code does not touch typed fields here,
// consistent with the *_field_slot per-slot dispatcher design (game
// owns its layout).
//
// One traversal feeds three consumers:
//   - hash builder (schema_hash.h): hash_field_slot per visible slot
//   - snapshot serializer (snapshot_io.h): read_field_slot / write_field_slot
//   - encoder (per-game net_adapter): reads MaskedState placeholders
//
// Sharing one traversal keeps "what counts as visible" defined in
// exactly one place.

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

namespace detail {

// Hidden dual of walk_data_axes_recursive: visits idx whose viewer bit
// is 0 OR whose belief_filled bit (if provided) is 1.
inline void walk_hidden_data_axes_recursive(
    const std::string& name, const VizTensor& v,
    const VizTensor* belief_v, int perspective,
    std::vector<int>& idx, int axis,
    const SlotVisitor& fn) {
  const int data_rank = v.rank() - 1;
  if (axis == data_rank) {
    const std::size_t base = flat_offset_data_only(v.shape, idx);
    const std::size_t off = base + static_cast<std::size_t>(perspective);
    const bool viz_off = (v.data[off] == 0);
    bool belief_marked = false;
    if (belief_v && !belief_v->empty()) {
      // Same flat layout (callers must allocate belief_filled with the
      // same shape as state.viz_ for that field).
      const std::size_t bbase = flat_offset_data_only(belief_v->shape, idx);
      const std::size_t boff = bbase + static_cast<std::size_t>(perspective);
      if (boff < belief_v->data.size()) {
        belief_marked = belief_v->data[boff] != 0;
      }
    }
    if (viz_off || belief_marked) {
      fn(name, idx, v);
    }
    return;
  }
  for (int i = 0; i < v.shape[axis]; ++i) {
    idx[axis] = i;
    walk_hidden_data_axes_recursive(name, v, belief_v, perspective, idx,
                                    axis + 1, fn);
  }
}

}  // namespace detail

// for_each_hidden_slot — dual of for_each_visible_slot.
//
// Visits every (non-internal) field × data-axis slot whose
// viz[idx..., perspective] == 0, OR whose belief_filled[idx...,
// perspective] == 1 (when belief_filled is provided). Used by
// `make_masked_state` to drive per-slot placeholder writes.
//
// Visit order: same as for_each_visible_slot — schema declaration
// order × row-major indices. The yielded VizTensor is the field's
// runtime viz_ (informational; placeholder writers don't typically
// need it).
inline void for_each_hidden_slot(
    const IGameState& state, const VisibilitySchema& schema, int perspective,
    const std::unordered_map<std::string, VizTensor>* belief_filled,
    const SlotVisitor& fn) {
  for (const auto& field : schema.fields) {
    const VizTensor& v = viz_get(state, field.name);
    if (v.empty()) continue;
    if (perspective < 0 || perspective >= v.viewer_count()) {
      throw std::out_of_range(
          "viz::for_each_hidden_slot: perspective " +
          std::to_string(perspective) +
          " out of range for field '" + field.name + "' (viewer_count=" +
          std::to_string(v.viewer_count()) + ")");
    }
    const VizTensor* belief_v = nullptr;
    if (belief_filled) {
      auto it = belief_filled->find(field.name);
      if (it != belief_filled->end()) belief_v = &it->second;
    }
    const int data_rank = v.rank() - 1;
    if (data_rank == 0) {
      const std::size_t off = static_cast<std::size_t>(perspective);
      const bool viz_off = (v.data[off] == 0);
      bool belief_marked = false;
      if (belief_v && !belief_v->empty() &&
          static_cast<std::size_t>(perspective) < belief_v->data.size()) {
        belief_marked =
            belief_v->data[static_cast<std::size_t>(perspective)] != 0;
      }
      if (viz_off || belief_marked) {
        std::vector<int> empty_idx;
        fn(field.name, empty_idx, v);
      }
      continue;
    }
    std::vector<int> idx(static_cast<std::size_t>(data_rank), 0);
    detail::walk_hidden_data_axes_recursive(field.name, v, belief_v,
                                            perspective, idx, 0, fn);
  }
}

namespace detail {

// Full-set dual: visits every (idx) regardless of viz, passing through
// the perspective's visibility bit so the caller can route visible vs
// hidden uniformly. Used by the framework hash to pin slot structure
// (field_pos × idx) and dispatch the value mix accordingly.
inline void walk_all_data_axes_recursive(
    const std::string& name, const VizTensor& v, int perspective,
    std::vector<int>& idx, int axis,
    const SlotVisitorWithVisibility& fn) {
  const int data_rank = v.rank() - 1;
  if (axis == data_rank) {
    const std::size_t base = flat_offset_data_only(v.shape, idx);
    const std::size_t off = base + static_cast<std::size_t>(perspective);
    const bool visible = v.data[off] != 0;
    fn(name, idx, v, visible);
    return;
  }
  for (int i = 0; i < v.shape[axis]; ++i) {
    idx[axis] = i;
    walk_all_data_axes_recursive(name, v, perspective, idx, axis + 1, fn);
  }
}

}  // namespace detail

// for_each_slot — schema declaration order × row-major idx, every
// slot visited (no viz filter). Callback gets the visibility bit so
// callers (currently the framework hash) can branch on visible vs
// hidden without re-querying viz.
inline void for_each_slot(const IGameState& state,
                          const VisibilitySchema& schema, int perspective,
                          const SlotVisitorWithVisibility& fn) {
  for (const auto& field : schema.fields) {
    const VizTensor& v = viz_get(state, field.name);
    if (v.empty()) continue;
    if (perspective < 0 || perspective >= v.viewer_count()) {
      throw std::out_of_range(
          "viz::for_each_slot: perspective " +
          std::to_string(perspective) +
          " out of range for field '" + field.name + "' (viewer_count=" +
          std::to_string(v.viewer_count()) + ")");
    }
    const int data_rank = v.rank() - 1;
    if (data_rank == 0) {
      const std::size_t off = static_cast<std::size_t>(perspective);
      const bool visible = v.data[off] != 0;
      std::vector<int> empty_idx;
      fn(field.name, empty_idx, v, visible);
      continue;
    }
    std::vector<int> idx(static_cast<std::size_t>(data_rank), 0);
    detail::walk_all_data_axes_recursive(field.name, v, perspective, idx, 0,
                                         fn);
  }
}

}  // namespace viz
}  // namespace board_ai
