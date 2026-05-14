#pragma once

// Snapshot wire I/O — schema-walker driven.
//
// Two halves of the same wire shape:
//
//   1) `serialize_public` / `apply_public` — walks every (name, idx) of
//      every all_public field, calling `state.read_field_slot` /
//      `state.write_field_slot`. Wire shape: `AnyMap[field_name]` is a
//      `vector<any>` in row-major slot order.
//
//   2) `serialize_partial_reveals` / `apply_partial_reveals` — sidecar
//      that carries the non-all_public slots which rules have revealed
//      to a specific perspective (LL hand[owner], LL drawn_card after
//      reveal_slot_to(starter), etc.). Wire shape:
//      `snap[__partial_reveal][field_name]` = vector<any>{idx, value, ...}.

#include <any>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "game_interfaces.h"
#include "visibility_schema.h"
#include "viz_walker.h"

namespace board_ai {
namespace viz {

namespace detail {
inline bool is_all_public(const VizTensor& v) {
  if (v.data.empty()) return false;
  for (auto byte : v.data) {
    if (byte != 1) return false;
  }
  return true;
}
}  // namespace detail

// ---------------- Walker-driven serialize/apply ----------------
//
// Walks every (name, idx) of every all_public field. For each slot:
//   serialize_public: snap[name].push_back(state.read_field_slot(name, idx))
//   apply_public    : state.write_field_slot(name, idx, snap[name][k++])
// Skip set: fields the producer chooses not to ship (e.g. fixed-at-
// game-start "first_player"). Non-all_public fields ride out-of-band
// as partial-reveal sidecars (see below).

inline void serialize_public(
    const IGameState& state, const VisibilitySchema& schema,
    AnyMap& snap, const std::unordered_set<std::string>& skip = {}) {
  std::unordered_set<std::string> public_fields;
  for (const auto& f : schema.fields) {
    if (detail::is_all_public(f.base_viz)) public_fields.insert(f.name);
  }
  for_each_visible_slot(
      state, schema, /*perspective=*/0,
      [&](const std::string& name, const std::vector<int>& idx,
          const VizTensor& /*v*/) {
        if (skip.count(name)) return;
        if (!public_fields.count(name)) return;
        auto it = snap.find(name);
        if (it == snap.end()) {
          it = snap.emplace(name, std::any(std::vector<std::any>{})).first;
        }
        auto& vec = std::any_cast<std::vector<std::any>&>(it->second);
        vec.push_back(state.read_field_slot(name, idx));
      });
}

inline void apply_public(
    IGameState& state, const VisibilitySchema& schema, const AnyMap& snap,
    const std::unordered_set<std::string>& skip = {}) {
  std::unordered_map<std::string, std::size_t> cursor;
  std::unordered_set<std::string> public_fields;
  for (const auto& f : schema.fields) {
    if (detail::is_all_public(f.base_viz)) public_fields.insert(f.name);
  }
  for_each_visible_slot(
      state, schema, /*perspective=*/0,
      [&](const std::string& name, const std::vector<int>& idx,
          const VizTensor& /*v*/) {
        if (skip.count(name)) return;
        if (!public_fields.count(name)) return;
        auto it = snap.find(name);
        if (it == snap.end()) {
          throw std::invalid_argument(
              "viz::apply_public: snap is missing schema field '" + name +
              "'");
        }
        const std::size_t k = cursor[name]++;
        // py↔C++ AnyMap round-trip collapses all-int Python lists into
        // vector<int> (see py_to_any in py_engine.cpp), so the payload
        // arrives as either vector<any> (native C++) or vector<int>
        // (after pybind). Handle both.
        if (it->second.type() == typeid(std::vector<std::any>)) {
          const auto& vec = std::any_cast<const std::vector<std::any>&>(it->second);
          if (k >= vec.size()) {
            throw std::invalid_argument(
                "viz::apply_public: not enough values for field '" + name +
                "'");
          }
          state.write_field_slot(name, idx, vec[k]);
        } else if (it->second.type() == typeid(std::vector<int>)) {
          const auto& vec = std::any_cast<const std::vector<int>&>(it->second);
          if (k >= vec.size()) {
            throw std::invalid_argument(
                "viz::apply_public: not enough values for field '" + name +
                "'");
          }
          state.write_field_slot(name, idx, std::any(vec[k]));
        } else {
          throw std::invalid_argument(
              std::string("viz::apply_public: field '") + name +
              "' has unsupported any type '" + it->second.type().name() +
              "' for vector payload");
        }
      });
}

// ---------------- Walker-driven partial-reveal sidecar ----------------
//
// `serialize_public` only ships fields whose schema base_viz is all_public.
// Fields with non-all_public base (owner_only_first_axis, all_hidden) but
// whose runtime viz[idx, perspective] == 1 — Love Letter hand[owner],
// LL drawn_card after reveal_slot_to(starter), Coup influence[p,s] after
// lose-influence reveal_slot, etc. — ride out-of-band on this sidecar.
//
// Wire shape:
//   snap[kPartialRevealKey] is an AnyMap keyed by field name. For each
//   field, the value is a vector<any> of {idx, value} entries:
//     vector<any>{
//       any(vector<int> idx_path), any(value),
//       any(vector<int> idx_path), any(value),
//       ...
//     }
//   `idx_path` is the data-axis index (matches walker's idx, empty for
//   scalars). `value` is what state.read_field_slot returns for that slot.
//
// `serialize_partial_reveals(state, schema, perspective, snap)`:
//   For every field whose base_viz is NOT all_public, walk runtime viz=1
//   slots for `perspective` and append (idx, value) entries.
//
// `apply_partial_reveals(state, schema, receiver_seat, snap)`:
//   Wholesale-replacement semantics. For every non-all_public field,
//   first reset its viz to schema base (i.e. drop all in-round reveals
//   on the observer side), THEN walk the sidecar entries and:
//     - state.write_field_slot(name, idx, value)
//     - reveal_slot_to(state, name, idx, receiver_seat)
//
// The reset-to-base step is essential: rules on truth call reset_to_base
// at round transitions / King-swap / Priest-peek-end, dropping a viewer's
// runtime reveals back to schema base. The observer must mirror that
// drop, otherwise a previously-revealed slot stays viz=1 forever and the
// observer's hash diverges from truth's. Re-revealing happens via the
// sidecar entries on the same call (any slot still visible to receiver
// will be in the sidecar and will reveal_slot_to again).
//
// Receiver-seat is passed by the caller (runner / API session knows
// which seat it is). No magic key in the AnyMap.
//
// Test guard: tests/framework/test_partial_reveal_round_trip.py.

inline constexpr const char* kPartialRevealKey = "__partial_reveal";

inline void serialize_partial_reveals(
    const IGameState& state, const VisibilitySchema& schema,
    int perspective, AnyMap& snap) {
  if (perspective < 0) return;
  std::unordered_set<std::string> non_public_fields;
  for (const auto& f : schema.fields) {
    if (f.base_viz.empty()) continue;
    if (!detail::is_all_public(f.base_viz)) {
      non_public_fields.insert(f.name);
    }
  }
  if (non_public_fields.empty()) return;

  AnyMap partial;
  for_each_visible_slot(
      state, schema, perspective,
      [&](const std::string& name, const std::vector<int>& idx,
          const VizTensor& /*v*/) {
        if (!non_public_fields.count(name)) return;
        auto it = partial.find(name);
        if (it == partial.end()) {
          it = partial.emplace(name, std::any(std::vector<std::any>{})).first;
        }
        auto& vec = std::any_cast<std::vector<std::any>&>(it->second);
        vec.push_back(std::any(idx));
        vec.push_back(state.read_field_slot(name, idx));
      });

  if (!partial.empty()) {
    snap[kPartialRevealKey] = std::any(std::move(partial));
  }
}

inline void apply_partial_reveals(
    IGameState& state, const VisibilitySchema& schema,
    int receiver_seat, const AnyMap& snap) {
  if (receiver_seat < 0) {
    // Caller doesn't know its seat (Splendor / fully-public games passing
    // the default sentinel). Per the contract, sidecar should also be
    // empty in that case, so a no-op is the right behavior. But if the
    // sidecar IS present, that's a usage error.
    if (snap.find(kPartialRevealKey) != snap.end()) {
      throw std::invalid_argument(
          "viz::apply_partial_reveals: snapshot carries partial reveals "
          "but receiver_seat is -1");
    }
    return;
  }

  // Validate that every named field exists in the schema and has
  // non-all_public base — defensive against shape drift.
  std::unordered_set<std::string> non_public_fields;
  for (const auto& f : schema.fields) {
    if (f.base_viz.empty()) continue;
    if (!detail::is_all_public(f.base_viz)) {
      non_public_fields.insert(f.name);
    }
  }

  // Wholesale reset: drop receiver's viz on every non-all_public field
  // back to schema base BEFORE re-applying the sidecar's reveals. Mirrors
  // truth-side reset_to_base calls that the receiver wouldn't otherwise
  // see (round transitions, swaps, etc.).
  for (const auto& f : schema.fields) {
    if (f.base_viz.empty()) continue;
    if (detail::is_all_public(f.base_viz)) continue;
    auto& v = viz_get(state, f.name);
    if (v.shape != f.base_viz.shape) continue;
    const int n_viewers = v.viewer_count();
    if (receiver_seat >= n_viewers) continue;
    // Stride over every data-axis row, restoring receiver's viewer bit.
    const std::size_t row_stride = static_cast<std::size_t>(n_viewers);
    const std::size_t total = v.data.size();
    for (std::size_t base = 0; base + row_stride <= total; base += row_stride) {
      v.data[base + static_cast<std::size_t>(receiver_seat)] =
          f.base_viz.data[base + static_cast<std::size_t>(receiver_seat)];
    }
  }

  auto it = snap.find(kPartialRevealKey);
  if (it == snap.end()) return;

  // Sidecar is itself an AnyMap. Pybind round-trip preserves AnyMap type.
  if (it->second.type() != typeid(AnyMap)) {
    throw std::invalid_argument(
        "viz::apply_partial_reveals: sidecar value is not an AnyMap");
  }
  const auto& partial = std::any_cast<const AnyMap&>(it->second);

  for (const auto& [name, payload] : partial) {
    if (!non_public_fields.count(name)) {
      throw std::invalid_argument(
          "viz::apply_partial_reveals: field '" + name +
          "' is not a non-all_public schema field; sidecar shape drift?");
    }
    if (payload.type() != typeid(std::vector<std::any>)) {
      throw std::invalid_argument(
          "viz::apply_partial_reveals: field '" + name +
          "' payload is not vector<any>");
    }
    const auto& vec = std::any_cast<const std::vector<std::any>&>(payload);
    if (vec.size() % 2 != 0) {
      throw std::invalid_argument(
          "viz::apply_partial_reveals: field '" + name +
          "' has odd entry count (idx/value pairs expected)");
    }
    for (std::size_t k = 0; k + 1 < vec.size(); k += 2) {
      const auto& idx_any = vec[k];
      // Pybind round-trip folds all-int Python lists into vector<int>,
      // but EMPTY Python lists arrive as vector<any> (no element to peek
      // at the int-ness of). Native C++ producer always emits vector<int>.
      std::vector<int> idx;
      if (idx_any.type() == typeid(std::vector<int>)) {
        idx = std::any_cast<const std::vector<int>&>(idx_any);
      } else if (idx_any.type() == typeid(std::vector<std::any>)) {
        const auto& av = std::any_cast<const std::vector<std::any>&>(idx_any);
        idx.reserve(av.size());
        for (const auto& x : av) {
          if (x.type() != typeid(int)) {
            throw std::invalid_argument(
                "viz::apply_partial_reveals: field '" + name +
                "' idx entry contains non-int element");
          }
          idx.push_back(std::any_cast<int>(x));
        }
      } else {
        throw std::invalid_argument(
            "viz::apply_partial_reveals: field '" + name +
            "' idx entry is not vector<int>");
      }
      state.write_field_slot(name, idx, vec[k + 1]);
      reveal_slot_to(state, name, idx, receiver_seat);
    }
  }
}

}  // namespace viz
}  // namespace board_ai
