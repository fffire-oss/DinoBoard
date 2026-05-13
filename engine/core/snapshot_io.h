#pragma once

// Snapshot wire I/O — schema-walker driven.
//
// `serialize_public` walks every slot of every all_public field via
// `for_each_visible_slot` and asks the game for a typed value through
// `IGameState::read_field_slot(name, idx)`. `apply_public` does the
// reverse via `write_field_slot`. Wire shape: `AnyMap[field_name]` is a
// `vector<any>` in row-major slot order.
//
// `SnapshotIO` + `emit_snapshot` / `apply_snapshot` is the older
// per-field-emitter API still used by Coup, Love Letter, and Azul.
// Both produce the same `AnyMap public_snapshot` shape; both are
// driven by the same schema walker.

#include <any>
#include <functional>
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

// ---------------- SnapshotIO: per-field emit/apply (Coup/LL/Azul) ----------------

using SnapshotEmitter = std::function<void(const IGameState&, AnyMap&)>;
using SnapshotApplier = std::function<void(IGameState&, const AnyMap&)>;

struct SnapshotIO {
  std::unordered_map<std::string, SnapshotEmitter> emitters;
  std::unordered_map<std::string, SnapshotApplier> appliers;
};

namespace detail {
inline bool is_all_public(const VizTensor& v) {
  if (v.data.empty()) return false;
  for (auto byte : v.data) {
    if (byte != 1) return false;
  }
  return true;
}
}  // namespace detail

inline void emit_snapshot(
    const IGameState& state, const VisibilitySchema& schema,
    const SnapshotIO& io, AnyMap& snap,
    const std::unordered_set<std::string>& skip = {}) {
  for (const auto& f : schema.fields) {
    if (skip.count(f.name)) continue;
    if (!detail::is_all_public(f.base_viz)) continue;
    auto it = io.emitters.find(f.name);
    if (it == io.emitters.end()) {
      throw std::logic_error(
          "viz::emit_snapshot: no emitter registered for schema all_public "
          "field '" + f.name + "'");
    }
    it->second(state, snap);
  }
}

inline void apply_snapshot(
    IGameState& state, const VisibilitySchema& schema,
    const SnapshotIO& io, const AnyMap& snap,
    const std::unordered_set<std::string>& skip = {}) {
  for (const auto& f : schema.fields) {
    if (skip.count(f.name)) continue;
    if (!detail::is_all_public(f.base_viz)) continue;
    auto it = io.appliers.find(f.name);
    if (it == io.appliers.end()) {
      throw std::logic_error(
          "viz::apply_snapshot: no applier registered for schema all_public "
          "field '" + f.name + "'");
    }
    it->second(state, snap);
  }
}

// ---------------- Walker-driven serialize/apply (Splendor) ----------------
//
// Walks every (name, idx) of every all_public field. For each slot:
//   serialize_public: snap[name].push_back(state.read_field_slot(name, idx))
//   apply_public    : state.write_field_slot(name, idx, snap[name][k++])
// Skip set: fields the producer chooses not to ship (e.g. fixed-at-
// game-start "first_player"). Non-all_public fields ride out-of-band
// as partial-reveal sidecars.

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

// Perspective-aware variant. Walks every (name, idx) slot whose
// `viz[..., perspective] == 1` — schema declaration order × row-major,
// all fields, no all_public filter. Used for the initial observation
// handshake where the receiver knows the AnyMap shape but the producer
// must include perspective-private slots (e.g. Love Letter starting
// `hand[perspective]`, owner_only_first_axis-base) that the broadcast
// `serialize_public` filters out. Round-trips with
// `apply_public_for_perspective` below.
inline void serialize_public_for_perspective(
    const IGameState& state, const VisibilitySchema& schema, int perspective,
    AnyMap& snap, const std::unordered_set<std::string>& skip = {}) {
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
        vec.push_back(state.read_field_slot(name, idx));
      });
}

inline void apply_public_for_perspective(
    IGameState& state, const VisibilitySchema& schema, int perspective,
    const AnyMap& snap,
    const std::unordered_set<std::string>& skip = {}) {
  std::unordered_map<std::string, std::size_t> cursor;
  for_each_visible_slot(
      state, schema, perspective,
      [&](const std::string& name, const std::vector<int>& idx,
          const VizTensor& /*v*/) {
        if (skip.count(name)) return;
        auto it = snap.find(name);
        if (it == snap.end()) {
          throw std::invalid_argument(
              "viz::apply_public_for_perspective: snap is missing field '" +
              name + "'");
        }
        const std::size_t k = cursor[name]++;
        if (it->second.type() == typeid(std::vector<std::any>)) {
          const auto& vec =
              std::any_cast<const std::vector<std::any>&>(it->second);
          if (k >= vec.size()) {
            throw std::invalid_argument(
                "viz::apply_public_for_perspective: not enough values for "
                "field '" +
                name + "'");
          }
          state.write_field_slot(name, idx, vec[k]);
        } else if (it->second.type() == typeid(std::vector<int>)) {
          const auto& vec =
              std::any_cast<const std::vector<int>&>(it->second);
          if (k >= vec.size()) {
            throw std::invalid_argument(
                "viz::apply_public_for_perspective: not enough values for "
                "field '" +
                name + "'");
          }
          state.write_field_slot(name, idx, std::any(vec[k]));
        } else {
          throw std::invalid_argument(
              std::string("viz::apply_public_for_perspective: field '") +
              name + "' has unsupported any type '" + it->second.type().name() +
              "' for vector payload");
        }
      });
}

}  // namespace viz
}  // namespace board_ai
