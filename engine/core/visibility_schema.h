#pragma once

// Visibility Schema (Phase 1.2): per-game declarative field visibility.
//
// A game declares, ONCE, every field in its IGameState — its name, the
// shape of its data tensor, and a bool[shape..., NPlayers] visibility
// tensor saying which seats see which slots. The framework derives:
//   - state_hash_for_perspective(p) (hash all viz[..., p]==1 && !internal)
//   - encoder feature scope (encode all viz[..., p]==1 && !internal)
//   - extract_snapshot / apply_snapshot (filter by viz)
//   - protocol-level visibility_mask (bit-packed, sent over the wire)
//
// This header defines the *declaration surface only*: schema data
// structures + base-viz builder helpers. There is intentionally NO
// "overlay" / "reveal-when" closure machinery here. Per the framework
// golden standard (§2.2, invariant I1), rules are the SOLE writer of
// state.viz_; dynamic visibility changes (face-up reveals, end-of-round
// resets, Priest peeks, etc.) happen by `do_action_fast` calling the
// rules-side helpers `reveal_slot / reveal_slot_to / reset_to_base`,
// not by schema-declared closures. Putting "what flips when" into the
// schema would create a second viz writer and break I1.
//
// The runtime walker (`for_each_visible_slot`), state.viz_ injection on
// IGameState, init_viz, the rules-side viz-mutation helpers, and the
// `derive_size` declaration helper for observer-derived scalars are
// added in Phase 1.5 / 1.6 PRs (they need types this header introduces).
//
// The scalar-fact visibility tensor is rank N+1 where N is the data
// tensor's rank (rank=0 for a scalar). The trailing axis is the viewer
// (NPlayers wide). For a scalar field with NPlayers=4, viz is bool[4]
// (one bit per viewer); for a 1D field of length L, bool[L][4]; for a
// 2D field of shape [a, b], bool[a][b][4]; etc.

#include <cstdint>
#include <functional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace board_ai {

namespace viz {

// Static visibility tensor stored as a flat row-major bool vector with
// shape metadata. Last dim is always NPlayers. For a rank-0 (scalar)
// data field with NPlayers=N, shape={N} and data.size()=N. For a 1D
// data field of length L, shape={L, N} and data.size()=L*N.
struct VizTensor {
  std::vector<int> shape;       // includes trailing NPlayers axis
  std::vector<std::uint8_t> data;  // row-major, 0/1 bytes (not bit-packed yet)

  bool empty() const { return data.empty(); }
  int rank() const { return static_cast<int>(shape.size()); }
  int viewer_count() const { return shape.empty() ? 0 : shape.back(); }
};

// Builder helpers — small set of patterns that cover ~95% of fields.

// All viewers see all slots. Useful for fully-public fields like
// scores, board layout, factories.
inline VizTensor all_public(const std::vector<int>& data_shape, int n_players) {
  VizTensor v;
  v.shape = data_shape;
  v.shape.push_back(n_players);
  std::size_t total = 1;
  for (int d : v.shape) total *= static_cast<std::size_t>(d);
  v.data.assign(total, 1);
  return v;
}

// No viewer sees any slot. For internal RNG state, framework-managed
// step_count, etc. (Marked `internal=true` separately so consumers can
// also exclude them from hashes.)
inline VizTensor all_hidden(const std::vector<int>& data_shape, int n_players) {
  VizTensor v;
  v.shape = data_shape;
  v.shape.push_back(n_players);
  std::size_t total = 1;
  for (int d : v.shape) total *= static_cast<std::size_t>(d);
  v.data.assign(total, 0);
  return v;
}

// Owner-only along the FIRST data axis: data shape [N, ...] where the
// first axis indexes owner; viewer p sees only slot[p, ...]. Common for
// per-player private fields like hands, influence cards, reserved
// stacks indexed by owner.
//
// data_shape MUST start with N (owner axis). Resulting viz shape is
// data_shape with trailing N (viewer axis); viz[owner, ..., viewer]=1
// iff owner == viewer.
inline VizTensor owner_only_first_axis(const std::vector<int>& data_shape, int n_players) {
  VizTensor v;
  v.shape = data_shape;
  v.shape.push_back(n_players);
  std::size_t total = 1;
  for (int d : v.shape) total *= static_cast<std::size_t>(d);
  v.data.assign(total, 0);

  // Compute strides for v.shape.
  const int rank = static_cast<int>(v.shape.size());
  std::vector<std::size_t> strides(rank);
  strides[rank - 1] = 1;
  for (int i = rank - 2; i >= 0; --i) {
    strides[i] = strides[i + 1] * static_cast<std::size_t>(v.shape[i + 1]);
  }

  // For each (owner, viewer) combination set v[owner, *, viewer]=1 iff
  // owner == viewer. Iterate by setting all (owner, *, owner=viewer)
  // slots and leaving others 0.
  std::vector<int> idx(rank, 0);
  while (true) {
    if (idx[0] == idx[rank - 1]) {  // owner axis == viewer axis
      std::size_t off = 0;
      for (int i = 0; i < rank; ++i) {
        off += static_cast<std::size_t>(idx[i]) * strides[i];
      }
      v.data[off] = 1;
    }
    // Increment idx in row-major order.
    int dim = rank - 1;
    while (dim >= 0) {
      ++idx[dim];
      if (idx[dim] < v.shape[dim]) break;
      idx[dim] = 0;
      --dim;
    }
    if (dim < 0) break;
  }
  return v;
}

// One field in the game state.
//
// FieldDecl is intentionally minimal: just shape (carried by base_viz),
// the static base visibility, and two flags. Dynamic visibility is NOT
// described here — `do_action_fast` mutates state.viz_ directly via the
// rules-side helpers (Phase 1.5). That keeps rules as the sole viz
// writer (golden standard I1).
struct FieldDecl {
  std::string name;             // stable identifier; doubles as path key
                                // for snapshot serialization
  VizTensor base_viz;           // static visibility — what state.viz_
                                // gets initialized to and what
                                // reset_to_base restores to
  bool internal = false;        // RNG state, step counter, etc. — never
                                // hashed, encoded, walked, or sent
                                // over wire
  bool derived = false;         // observer-derived scalar (e.g.
                                // deck.size()); excluded from snapshot
                                // values (recomputable from another
                                // field) but still public in hash. The
                                // `derive_size` declaration helper that
                                // sets this flag is added in Phase 1.5.
};

// Describes the visibility audience of a single public-event payload.
// Used to generate the protocol-side `audience` mask alongside snapshots.
struct EventDecl {
  std::string kind;             // stable identifier
  // Phase relative to action; mirrors EventPhase in game_interfaces.h.
  // Stored as int to avoid pulling the full header in here.
  int phase = 1;                // 0=pre, 1=post (default post)
  // bool[NPlayers] — true iff that seat receives the event payload.
  std::vector<std::uint8_t> audience;
  // Optional payload schema doc — list of (key, type-string) pairs.
  // Not enforced at runtime yet; will drive auto-generated docs and
  // payload validation in a later phase.
  std::vector<std::pair<std::string, std::string>> payload_schema;
};

// Maps an action ID range or predicate to the list of events that
// action will produce. The matcher is an opaque predicate so games
// can use whatever scheme fits (range checks, switch-table, etc.).
struct ActionEventMap {
  std::function<bool(int /*action_id*/)> matches;
  std::vector<EventDecl> events;
};

// The visibility schema of one game.
struct VisibilitySchema {
  int n_players = 0;
  std::vector<FieldDecl> fields;
  std::vector<ActionEventMap> action_events;
  // post_events_required: if true, ObserveRequest must include
  // post_events for hidden-info diff (e.g. assumed-hidden derived
  // fields the observer can't reconstruct from snapshot alone).
  // Defaults false: snapshot is sufficient.
  bool post_events_required = false;
};

// ---------- ergonomic wrappers ----------
//
// Sugar for the common patterns. Games typically write:
//
//   declare_field(schema, "scores", viz::all_public({N}, N));
//   declare_field(schema, "influence", viz::owner_only_first_axis({N, 2}, N));
//
// All "what flips when" lives in rules (`do_action_fast`), not here:
//
//   // in coup_rules.cpp do_action_fast:
//   if (a.type == kRevealInfluence) {
//     viz::reveal_slot(s.viz_, &CoupState::influence, p, i);   // Phase 1.5
//     s.influence[p][i] = kEmpty;
//   }
//
// `declare_field` validates name uniqueness + viewer-count consistency
// and pushes a FieldDecl onto the schema.

inline void declare_field(VisibilitySchema& schema, const std::string& name,
                          VizTensor base_viz,
                          bool internal = false, bool derived = false) {
  // Name uniqueness — schema is the single source of truth for path keys
  // used in snapshots, hashes, and event payloads. Duplicates would let
  // one field silently shadow another.
  for (const auto& existing : schema.fields) {
    if (existing.name == name) {
      throw std::invalid_argument(
          "VisibilitySchema: duplicate field name '" + name + "'");
    }
  }
  // viewer_count consistency — every field's viz must match schema.n_players.
  // Empty viz (size==0) is allowed and means "internal/no-viz", typically
  // paired with internal=true.
  if (!base_viz.empty() && schema.n_players > 0 &&
      base_viz.viewer_count() != schema.n_players) {
    throw std::invalid_argument(
        "VisibilitySchema: field '" + name + "' viz viewer_count (" +
        std::to_string(base_viz.viewer_count()) +
        ") does not match schema.n_players (" +
        std::to_string(schema.n_players) + ")");
  }

  FieldDecl f;
  f.name = name;
  f.base_viz = std::move(base_viz);
  f.internal = internal;
  f.derived = derived;
  schema.fields.push_back(std::move(f));
}

// NOTE: overlay primitives (reveal_when / derived_size_of / custom)
// have been removed. Per golden standard I1, rules are the sole writer
// of state.viz_. Dynamic visibility flips are produced by `do_action_fast`
// calling the rules-side helpers (`reveal_slot / reveal_slot_to /
// reset_to_base`), which are added in Phase 1.5 once state.viz_ exists
// on IGameState. Observer-derived scalars use the `derive_size`
// declaration helper (also Phase 1.5), which simply sets
// FieldDecl::derived=true with an all-public base viz — no closure
// needed.

}  // namespace viz
}  // namespace board_ai
