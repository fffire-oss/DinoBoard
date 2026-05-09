#pragma once

// Runtime ops on state.viz_ (Phase 1.2 — storage + name-keyed primitives).
//
// Split from visibility_schema.h to avoid an include cycle: the schema
// header is included by game_interfaces.h (so IGameState can have a
// `std::unordered_map<std::string, VizTensor> viz_` member), and the
// runtime helpers below need the concrete IGameState type.
//
// Per golden standard I1, the only callers of reveal_slot /
// reveal_slot_to / reset_to_base are the rules' do_action_fast (and the
// undo path that restores via UndoRecord). Tests + framework code that
// READS state.viz_ (hash walker, encoder masker, snapshot extractor) do
// so directly through the public viz_ accessor without going through
// these mutators.

#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

#include "game_interfaces.h"
#include "visibility_schema.h"

namespace board_ai {
namespace viz {

// Initialize state.viz_ from a schema. Called from each game's
// reset_with_seed override (Phase 3 wires this in for each game). For
// every non-internal field in `schema.fields`, copies the field's
// `base_viz` into `state.viz_[name]`. Internal fields are also stored
// (their base_viz is empty by convention) so that lookup never returns
// "missing key" — a missing key always indicates a schema bug.
inline void init_viz(IGameState& state, const VisibilitySchema& schema) {
  state.viz_.clear();
  state.viz_.reserve(schema.fields.size());
  for (const auto& f : schema.fields) {
    state.viz_.emplace(f.name, f.base_viz);
  }
}

// Lookup `state.viz_[name]` with a clear error if it's missing — common
// case is a typo in the rules-side reveal call. Returned by reference so
// callers can mutate.
inline VizTensor& viz_get(IGameState& state, const std::string& name) {
  auto it = state.viz_.find(name);
  if (it == state.viz_.end()) {
    throw std::invalid_argument(
        "viz::viz_get: state.viz_ has no field '" + name +
        "' — schema not initialized or field name typo");
  }
  return it->second;
}
inline const VizTensor& viz_get(const IGameState& state, const std::string& name) {
  auto it = state.viz_.find(name);
  if (it == state.viz_.end()) {
    throw std::invalid_argument(
        "viz::viz_get(const): state.viz_ has no field '" + name +
        "' — schema not initialized or field name typo");
  }
  return it->second;
}

// reveal_slot(state, "field", {idx...}) — flip viz for the named slot
// to all-1 across viewers. `idx` indexes the data-axis slots only; the
// trailing viewer axis is set entirely. For a scalar field (data rank
// 0), pass an empty `idx` to flip the single per-viewer entry.
inline void reveal_slot(IGameState& state, const std::string& name,
                        const std::vector<int>& idx) {
  auto& v = viz_get(state, name);
  const std::size_t base = flat_offset_data_only(v.shape, idx);
  const int n_viewers = v.viewer_count();
  for (int p = 0; p < n_viewers; ++p) {
    v.data[base + static_cast<std::size_t>(p)] = 1;
  }
}

// reveal_slot_to(state, "field", {idx...}, viewer) — flip viz ON only
// for the named viewer; other viewers' bits left untouched. Used for
// Priest peek and similar targeted reveals.
inline void reveal_slot_to(IGameState& state, const std::string& name,
                           const std::vector<int>& idx, int viewer) {
  auto& v = viz_get(state, name);
  const int n_viewers = v.viewer_count();
  if (viewer < 0 || viewer >= n_viewers) {
    throw std::out_of_range(
        "viz::reveal_slot_to: viewer index out of range on field '" + name + "'");
  }
  const std::size_t base = flat_offset_data_only(v.shape, idx);
  v.data[base + static_cast<std::size_t>(viewer)] = 1;
}

// reset_to_base(state, "field", schema, {idx...}) — restore the named
// slot's viewer bits to the schema's declared base. Used by end-of-round
// resets where rules want to drop all in-round reveals on a slot.
inline void reset_to_base(IGameState& state, const std::string& name,
                          const VisibilitySchema& schema,
                          const std::vector<int>& idx) {
  // Find the field decl.
  const FieldDecl* fd = nullptr;
  for (const auto& f : schema.fields) {
    if (f.name == name) { fd = &f; break; }
  }
  if (!fd) {
    throw std::invalid_argument(
        "viz::reset_to_base: schema has no field '" + name + "'");
  }
  auto& v = viz_get(state, name);
  if (v.shape != fd->base_viz.shape) {
    throw std::runtime_error(
        "viz::reset_to_base: state.viz_ shape mismatch for '" + name + "'");
  }
  const std::size_t base = flat_offset_data_only(v.shape, idx);
  const int n_viewers = v.viewer_count();
  for (int p = 0; p < n_viewers; ++p) {
    v.data[base + static_cast<std::size_t>(p)] =
        fd->base_viz.data[base + static_cast<std::size_t>(p)];
  }
}

// for_each_visible_slot — declaration only. Walker body lands in Phase
// 1.5 (engine/core/viz_walker.h) once hash + encoder + snapshot extractor
// share a single traversal. The signature is locked here so Phase 1.5's
// implementation is a drop-in.
//
// Semantics: for each non-internal field in `schema`, for each data-axis
// slot whose viz[..., perspective]==1, invoke `fn(field_name, idx,
// viewer_count)`. Phase 1.5 will refine the callback signature to also
// pass field-typed slot value via the registry's member-pointer table.
using SlotVisitor = std::function<void(
    const std::string& /*name*/, const std::vector<int>& /*idx*/,
    const VizTensor& /*viz*/)>;

void for_each_visible_slot(const IGameState& state,
                           const VisibilitySchema& schema, int perspective,
                           const SlotVisitor& fn);

}  // namespace viz
}  // namespace board_ai
