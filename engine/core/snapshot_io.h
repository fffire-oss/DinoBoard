#pragma once

// Snapshot wire I/O — schema-walker driven, single unified protocol.
//
// GT writes the complete observation for a perspective in two halves on
// the same `AnyMap snap`:
//
//   1) Per-field value half: `snap[field_name]` is a `vector<any>` of
//      `(idx_path, value)` pairs — one entry pair for every data slot
//      where `viz_[name][..., perspective] == 1`. `idx_path` is a
//      `vector<int>` (empty for scalar fields). `value` is whatever
//      `state.read_field_slot(name, idx)` returned.
//
//   2) Viz-slice half: `snap[kVizSliceKey]` is an `AnyMap` keyed by field
//      name; each value is a flat `vector<int>` of length
//      `prod(data_axes)` carrying `viz_[name][..., perspective]` for
//      every data slot. Receiver wholesale-replaces its own viz slice
//      from this — no derivation, no reset-to-base.
//
// Both halves are walked through the schema's full slot set. There is no
// all_public / non-all_public branching. A field declared all_public has
// every slot's viz=1, so its full row appears in the value half; a field
// declared owner_only_first_axis has only the perspective-owned rows in
// the value half. Either way, the receiver just re-applies what the wire
// says — both for values (write_field_slot) and for viz
// (viz::apply_full_slice).

#include <any>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

#include "game_interfaces.h"
#include "visibility_schema.h"
#include "viz_runtime.h"
#include "viz_walker.h"

namespace board_ai {
namespace viz {

inline constexpr const char* kVizSliceKey = "__viz__";

namespace detail {

// Number of data slots in a viz tensor (product of all data axes; viz
// shape is data_axes + [n_viewers]).
inline std::size_t data_slot_count(const VizTensor& v) {
  if (v.data.empty()) return 0;
  const int n_viewers = v.viewer_count();
  if (n_viewers <= 0) return 0;
  return v.data.size() / static_cast<std::size_t>(n_viewers);
}

// Extract a vector<int> from std::any, accepting either native vector<int>
// or vector<any>-of-int (pybind round-trips empty Python lists this way).
inline std::vector<int> any_to_int_vec(const std::any& a,
                                       const std::string& ctx) {
  if (a.type() == typeid(std::vector<int>)) {
    return std::any_cast<const std::vector<int>&>(a);
  }
  if (a.type() == typeid(std::vector<std::any>)) {
    const auto& av = std::any_cast<const std::vector<std::any>&>(a);
    std::vector<int> out;
    out.reserve(av.size());
    for (const auto& x : av) {
      if (x.type() != typeid(int)) {
        throw std::invalid_argument(
            ctx + ": vector contains non-int element");
      }
      out.push_back(std::any_cast<int>(x));
    }
    return out;
  }
  throw std::invalid_argument(ctx + ": payload is not vector<int>");
}

}  // namespace detail

// Serialize the perspective's full observation into `snap`. See file
// header for wire format. Fields named in `skip` are written to neither
// half (used for fixed-at-game-start statics like Azul's
// `game_first_player`).
inline void serialize_public_snapshot(
    const IGameState& state, const VisibilitySchema& schema,
    int perspective, AnyMap& snap,
    const std::unordered_set<std::string>& skip = {}) {
  if (perspective < 0) {
    throw std::invalid_argument(
        "viz::serialize_public_snapshot: perspective must be >= 0");
  }

  // Value half: walk visible slots, append (idx, value) pairs.
  for_each_visible_slot(
      state, schema, perspective,
      [&](const std::string& name, const std::vector<int>& idx,
          const VizTensor& /*v*/) {
        if (skip.count(name)) return;
        auto it = snap.find(name);
        if (it == snap.end()) {
          it = snap.emplace(name, std::any(std::vector<std::any>{})).first;
        }
        auto& vec = std::any_cast<std::vector<std::any>&>(it->second);
        vec.push_back(std::any(idx));
        vec.push_back(state.read_field_slot(name, idx));
      });

  // Viz-slice half: for every schema field, copy the perspective's
  // full data-slot viz row into a flat vector<int>.
  AnyMap viz_slices;
  for (const auto& f : schema.fields) {
    if (skip.count(f.name)) continue;
    if (f.base_viz.empty()) continue;
    const auto& v = viz_get(const_cast<IGameState&>(state), f.name);
    if (v.shape != f.base_viz.shape) {
      throw std::runtime_error(
          "viz::serialize_public_snapshot: state.viz_ shape mismatch for '" +
          f.name + "'");
    }
    const int n_viewers = v.viewer_count();
    if (perspective >= n_viewers) {
      throw std::out_of_range(
          "viz::serialize_public_snapshot: perspective out of range on '" +
          f.name + "'");
    }
    const std::size_t n_slots = detail::data_slot_count(v);
    std::vector<int> slice;
    slice.reserve(n_slots);
    const std::size_t row_stride = static_cast<std::size_t>(n_viewers);
    for (std::size_t k = 0; k < n_slots; ++k) {
      slice.push_back(static_cast<int>(
          v.data[k * row_stride + static_cast<std::size_t>(perspective)]));
    }
    viz_slices.emplace(f.name, std::any(std::move(slice)));
  }
  snap[kVizSliceKey] = std::any(std::move(viz_slices));
}

// Apply a perspective snapshot onto `state`. Wholesale replaces the
// perspective's viz slice and writes the (idx, value) pairs back via
// `state.write_field_slot`. Fields named in `skip` are ignored on both
// halves.
inline void apply_public_snapshot(
    IGameState& state, const VisibilitySchema& schema,
    int perspective, const AnyMap& snap,
    const std::unordered_set<std::string>& skip = {}) {
  if (perspective < 0) {
    throw std::invalid_argument(
        "viz::apply_public_snapshot: perspective must be >= 0");
  }

  // Viz-slice half: replace receiver's perspective slice byte-for-byte.
  // Done first so write_field_slot below operates on a state whose viz
  // already reflects what GT shipped (downstream readers that key on
  // viz see a consistent picture).
  auto viz_it = snap.find(kVizSliceKey);
  if (viz_it == snap.end()) {
    throw std::invalid_argument(
        std::string("viz::apply_public_snapshot: snap is missing '") +
        kVizSliceKey + "' viz-slice section");
  }
  if (viz_it->second.type() != typeid(AnyMap)) {
    throw std::invalid_argument(
        std::string("viz::apply_public_snapshot: '") + kVizSliceKey +
        "' value is not an AnyMap");
  }
  const auto& viz_slices = std::any_cast<const AnyMap&>(viz_it->second);
  for (const auto& f : schema.fields) {
    if (skip.count(f.name)) continue;
    if (f.base_viz.empty()) continue;
    auto fit = viz_slices.find(f.name);
    if (fit == viz_slices.end()) {
      throw std::invalid_argument(
          "viz::apply_public_snapshot: viz-slice missing field '" +
          f.name + "'");
    }
    const std::vector<int> slice = detail::any_to_int_vec(
        fit->second,
        std::string("viz::apply_public_snapshot: field '") + f.name + "' viz");
    apply_full_slice(state, f.name, perspective, slice);
  }

  // Value half: walk every (name, value-vec) entry and write back via
  // state.write_field_slot. Each entry is a vector<any> alternating
  // idx (vector<int>/vector<any>) and value.
  std::unordered_set<std::string> field_names;
  for (const auto& f : schema.fields) field_names.insert(f.name);

  for (const auto& [name, payload] : snap) {
    if (name == kVizSliceKey) continue;
    if (skip.count(name)) continue;
    if (!field_names.count(name)) {
      throw std::invalid_argument(
          "viz::apply_public_snapshot: snap key '" + name +
          "' is not a schema field");
    }
    if (payload.type() != typeid(std::vector<std::any>)) {
      throw std::invalid_argument(
          "viz::apply_public_snapshot: field '" + name +
          "' payload is not vector<any>");
    }
    const auto& vec = std::any_cast<const std::vector<std::any>&>(payload);
    if (vec.size() % 2 != 0) {
      throw std::invalid_argument(
          "viz::apply_public_snapshot: field '" + name +
          "' has odd entry count (idx/value pairs expected)");
    }
    for (std::size_t k = 0; k + 1 < vec.size(); k += 2) {
      const std::vector<int> idx = detail::any_to_int_vec(
          vec[k],
          std::string("viz::apply_public_snapshot: field '") + name +
              "' idx");
      state.write_field_slot(name, idx, vec[k + 1]);
    }
  }
}

}  // namespace viz
}  // namespace board_ai
