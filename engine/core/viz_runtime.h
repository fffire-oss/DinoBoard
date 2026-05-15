#pragma once

// Runtime ops on state.viz_.
//
// Split from visibility_schema.h to break the include cycle: the schema
// header is included by game_interfaces.h (so IGameState can carry
// `std::unordered_map<std::string, VizTensor> viz_`), and the helpers
// below need the concrete IGameState type.
//
// Golden standard I1: rules' do_action_fast is the SOLE writer of
// state.viz_, calling reveal_slot / reveal_slot_to / reset_to_base.
// Framework readers (hash walker, snapshot serializer) read state.viz_
// directly; encoders never query viz at all (they read masked-state
// kPlaceholder* sentinels, see masked_state.h).

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "game_interfaces.h"
#include "visibility_schema.h"

namespace board_ai {
namespace viz {

// Initialize state.viz_ from a schema. Called from each game's
// reset_with_seed override. Copies every FieldDecl's base_viz into
// state.viz_[name]; missing keys always indicate a schema bug.
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

// apply_full_slice(state, "field", perspective, slice) — wholesale-replace
// `viz_[name][..., perspective]` byte-for-byte from a flat per-data-slot
// slice (length = product of data axes). Used ONLY by the framework
// snapshot applier (engine/core/snapshot_io.h). Lives here because it
// needs the concrete IGameState type; the I1 lint
// (test_rules_sole_viz_writer) only scopes `games/<id>/`, so framework
// callers are invisible to it.
//
// `slice[k]` (k = flat data offset) writes
// `viz_data[k * n_viewers + perspective]`. Throws if perspective is out
// of range or slice length doesn't match the data shape.
inline void apply_full_slice(IGameState& state, const std::string& name,
                             int perspective,
                             const std::vector<int>& slice) {
  auto& v = viz_get(state, name);
  const int n_viewers = v.viewer_count();
  if (perspective < 0 || perspective >= n_viewers) {
    throw std::out_of_range(
        "viz::apply_full_slice: perspective out of range on field '" +
        name + "'");
  }
  const std::size_t row_stride = static_cast<std::size_t>(n_viewers);
  const std::size_t total = v.data.size();
  const std::size_t n_data_slots = total / row_stride;
  if (slice.size() != n_data_slots) {
    throw std::invalid_argument(
        "viz::apply_full_slice: slice length " +
        std::to_string(slice.size()) + " != data-slot count " +
        std::to_string(n_data_slots) + " for field '" + name + "'");
  }
  for (std::size_t k = 0; k < n_data_slots; ++k) {
    v.data[k * row_stride + static_cast<std::size_t>(perspective)] =
        static_cast<std::uint8_t>(slice[k] != 0 ? 1 : 0);
  }
}

// swap_slot(state, "field", {idx_a...}, {idx_b...}) — swap the entire
// viewer-axis row between two slots. After the call, viz[a, :] holds
// what was at viz[b, :] and vice versa. Used when rules swap the slot
// CONTENTS — viz must follow content so "who has seen this cid" stays
// attached to the cid (e.g. Love Letter King swap).
inline void swap_slot(IGameState& state, const std::string& name,
                      const std::vector<int>& idx_a,
                      const std::vector<int>& idx_b) {
  auto& v = viz_get(state, name);
  const std::size_t base_a = flat_offset_data_only(v.shape, idx_a);
  const std::size_t base_b = flat_offset_data_only(v.shape, idx_b);
  if (base_a == base_b) return;
  const int n_viewers = v.viewer_count();
  for (int p = 0; p < n_viewers; ++p) {
    std::swap(v.data[base_a + static_cast<std::size_t>(p)],
              v.data[base_b + static_cast<std::size_t>(p)]);
  }
}

// swap_slot_owned(state, "field", owner_a, owner_b) — convenience for
// owner_only_first_axis fields whose first (and only) data axis is the
// owner index (e.g. LL `hand[player]`). Equivalent to
// swap_slot({owner_a}, {owner_b}) followed by reveal_slot_to(owner_a)
// and reveal_slot_to(owner_b). Captures "swap two owner-held slots and
// keep each owner seeing their new content" in one call.
inline void swap_slot_owned(IGameState& state, const std::string& name,
                            int owner_a, int owner_b) {
  auto& v = viz_get(state, name);
  // Owner-only-first-axis fields have data rank 1 (data shape = {N_owners}).
  // The full viz tensor shape is {N_owners, N_viewers}.
  if (v.shape.size() != 2) {
    throw std::invalid_argument(
        "viz::swap_slot_owned: field '" + name +
        "' is not owner_only_first_axis (data rank != 1); use swap_slot + "
        "manual reveal_slot_to for higher-rank fields");
  }
  if (owner_a == owner_b) return;
  swap_slot(state, name, {owner_a}, {owner_b});
  reveal_slot_to(state, name, {owner_a}, owner_a);
  reveal_slot_to(state, name, {owner_b}, owner_b);
}

// SlotVisitor: callback type for the walker. Body of
// for_each_visible_slot lives in viz_walker.h; the typedef is here so
// other headers can include only viz_runtime.h.
using SlotVisitor = std::function<void(
    const std::string& /*name*/, const std::vector<int>& /*idx*/,
    const VizTensor& /*viz*/)>;

// SlotVisitorWithVisibility: callback for `for_each_slot` (the full-set
// walker that visits every schema slot, regardless of viz). Adds a
// `visible` flag indicating whether `viz[idx..., perspective] == 1`.
// Used by the framework hash to dispatch visible slots to the game's
// `hash_field_slot` and hidden slots to a fixed sentinel.
using SlotVisitorWithVisibility = std::function<void(
    const std::string& /*name*/, const std::vector<int>& /*idx*/,
    const VizTensor& /*viz*/, bool /*visible*/)>;

}  // namespace viz
}  // namespace board_ai
