#pragma once

// Phase 4 (step B): SnapshotIO — per-field emitter/applier tables.
//
// Each hidden-info game registers a `public_event_extractor` whose job
// is to populate `out.public_snapshot` with the truth-side dump of every
// public field, and a `public_state_applier` that writes those fields
// back onto an observer state. Historically these were two long
// hand-typed blocks of `snap["foo"] = ...` / `state.foo = snap["foo"]`,
// with no way for the framework to check that they covered the right
// set of fields. Step A renamed snapshot keys to match schema field
// names; this header makes schema the single source of truth for
// iteration.
//
//   io.emitters["scores"] = [](const IGameState& s, AnyMap& m){...};
//   io.appliers["scores"] = [](IGameState& s, const AnyMap& m){...};
//   ...
//   viz::emit_snapshot(state, schema, io, out_snap, /*skip=*/{...});
//
// `emit_snapshot` walks the schema's declaration order, and for every
// non-internal, non-derived, all-public field looks up an emitter in
// `io`. **Missing emitter for an all-public field throws** — it is not
// possible to silently forget a field. Symmetric guarantee for
// `apply_snapshot` / appliers.
//
// Variable-length vectors (azul.bag / coup.court_deck / etc.) and
// partial-reveal sidecars (coup.revealed_char_flat) are NOT schema
// fields, so they are NOT emitted by this helper. Games write/read those
// directly to/from `snap` either side of the call.
//
// Fields that the schema declares all_public but `hash_public_fields`
// legitimately omits (e.g. azul `game_first_player` — fixed at game
// start, equivalence-class-irrelevant) go in the `skip` set; the
// helper bypasses them and does NOT require an emitter entry.

#include <functional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>

#include "game_interfaces.h"
#include "visibility_schema.h"

namespace board_ai {
namespace viz {

using SnapshotEmitter = std::function<void(const IGameState&, AnyMap&)>;
using SnapshotApplier = std::function<void(IGameState&, const AnyMap&)>;

struct SnapshotIO {
  std::unordered_map<std::string, SnapshotEmitter> emitters;
  std::unordered_map<std::string, SnapshotApplier> appliers;
};

namespace detail {

// Returns true iff every byte of base_viz is 1 — i.e. every viewer sees
// every slot of this field. Equivalent to "field was declared with
// viz::all_public(...)". Empty viz returns false (treated as no viz).
inline bool is_all_public(const VizTensor& v) {
  if (v.data.empty()) return false;
  for (auto byte : v.data) {
    if (byte != 1) return false;
  }
  return true;
}

}  // namespace detail

// Walk schema fields in declaration order; for each non-internal,
// non-derived all-public field NOT in `skip`, look up an emitter in
// `io.emitters` and call it. Throws std::logic_error if any such field
// has no emitter.
//
// Caller is responsible for writing snapshot-only keys (variable-length
// vectors, partial-reveal sidecars) into `snap` separately.
inline void emit_snapshot(
    const IGameState& state,
    const VisibilitySchema& schema,
    const SnapshotIO& io,
    AnyMap& snap,
    const std::unordered_set<std::string>& skip = {}) {
  for (const auto& f : schema.fields) {
    if (f.internal || f.derived) continue;
    if (skip.count(f.name)) continue;
    if (!detail::is_all_public(f.base_viz)) continue;
    auto it = io.emitters.find(f.name);
    if (it == io.emitters.end()) {
      throw std::logic_error(
          "viz::emit_snapshot: no emitter registered for schema all_public "
          "field '" + f.name + "' (add it to SnapshotIO.emitters or include "
          "it in the skip set)");
    }
    it->second(state, snap);
  }
}

// Inverse of emit_snapshot. Walks schema in the same order; throws if
// any required field has no applier registered.
inline void apply_snapshot(
    IGameState& state,
    const VisibilitySchema& schema,
    const SnapshotIO& io,
    const AnyMap& snap,
    const std::unordered_set<std::string>& skip = {}) {
  for (const auto& f : schema.fields) {
    if (f.internal || f.derived) continue;
    if (skip.count(f.name)) continue;
    if (!detail::is_all_public(f.base_viz)) continue;
    auto it = io.appliers.find(f.name);
    if (it == io.appliers.end()) {
      throw std::logic_error(
          "viz::apply_snapshot: no applier registered for schema all_public "
          "field '" + f.name + "' (add it to SnapshotIO.appliers or include "
          "it in the skip set)");
    }
    it->second(state, snap);
  }
}

}  // namespace viz
}  // namespace board_ai
