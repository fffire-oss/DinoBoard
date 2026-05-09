#pragma once

// Phase 1.6 (lightweight): PerspectiveSnapshot wire type.
//
// Single snapshot type used in BOTH places:
//   - create_session's `initial_observation` (step=0 snapshot)
//   - apply_observation's `snapshot` parameter (step>0 snapshot)
//
// Per golden standard §7.3, there is no separate `InitialSnapshot`
// type — the wire shape is identical for "first observation after
// reset_with_seed" and "every subsequent observation". Distinguishing
// the two would force every downstream consumer to special-case
// step=0; same shape lets one apply_observation flow handle both.
//
// Layout:
//   - `visibility_mask` — schema-keyed bit-packed bool tensors. Same
//     keys as the schema's FieldDecl names; the framework writes this
//     into state.viz_ (full replacement, not merge — golden standard
//     §1.2 step 1) at apply_observation time.
//   - `values`          — sparse path→value dict. Keys are FieldDecl
//     names; values are typed payloads the game serializer/deserializer
//     understands. Only slots with `visibility_mask[..., perspective]==1`
//     appear here (golden standard §1.2: viz=0 slots are NOT shipped
//     across the wire — wire size scales with what the perspective
//     actually sees, not with full state).
//
// Body (extract / apply) lands per-game in Phase 3 once each game's
// schema is declared and its extract/apply virtuals are overridden.
// Phase 1.6 ships the type, the bit-pack helpers, and the protocol
// validation skeleton; framework-side extract_snapshot /
// apply_observation bodies live behind game-side reflection that is
// not in place yet (same constraint as Phase 1.5's apply_viz_mask).
//
// Why values is a map<string, AnyMap> rather than a flat map<string, any>:
// per-field payloads are themselves structured (e.g. an `influence`
// field carries an int8[N][2] tensor — serialized as nested keys like
// {"shape": [N,2], "data": [...]}) — so the value side is itself a
// little doc, not a scalar. AnyMap (std::map<string, std::any>) is
// the existing framework primitive for this in game_interfaces.h.

#include <cstdint>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "game_interfaces.h"  // for AnyMap
#include "visibility_schema.h"

namespace board_ai {

// Bit-packed view of one VizTensor: same shape, but `data` is packed
// 8 bools per byte (LSB = lower-index bit). Wire size = ceil(N/8)
// bytes vs. N bytes for the unpacked VizTensor.
struct PackedVizTensor {
  std::vector<int> shape;            // identical to VizTensor::shape
  std::vector<std::uint8_t> packed;  // ceil(prod(shape) / 8) bytes
  std::size_t bit_count = 0;         // == prod(shape); kept so unpack
                                     // doesn't read trailing-byte garbage
};

inline PackedVizTensor pack_viz(const viz::VizTensor& v) {
  PackedVizTensor out;
  out.shape = v.shape;
  out.bit_count = v.data.size();
  out.packed.assign((out.bit_count + 7) / 8, 0);
  for (std::size_t i = 0; i < out.bit_count; ++i) {
    if (v.data[i]) out.packed[i / 8] |= (std::uint8_t{1} << (i % 8));
  }
  return out;
}

inline viz::VizTensor unpack_viz(const PackedVizTensor& p) {
  viz::VizTensor v;
  v.shape = p.shape;
  v.data.assign(p.bit_count, 0);
  for (std::size_t i = 0; i < p.bit_count; ++i) {
    if (p.packed[i / 8] & (std::uint8_t{1} << (i % 8))) v.data[i] = 1;
  }
  return v;
}

// Count set bits in a packed mask, ignoring bits past `bit_count` so
// trailing-byte padding doesn't inflate the count.
inline std::size_t popcount_packed(const PackedVizTensor& p) {
  std::size_t c = 0;
  for (std::size_t i = 0; i < p.bit_count; ++i) {
    if (p.packed[i / 8] & (std::uint8_t{1} << (i % 8))) ++c;
  }
  return c;
}

// PerspectiveSnapshot — what one viewer learned at one observation.
//
// Concrete struct (NOT forward-declared anymore; Phase 1.3's tracker.h
// forward decl now resolves to this). Used as the parameter type for
// ITracker::init / observe and for apply_observation's snapshot input.
struct PerspectiveSnapshot {
  // visibility_mask[field_name] gives the bit-packed bool tensor. Keys
  // MUST match the live schema's FieldDecl names exactly; missing keys
  // mean "this field's viz did not change since the previous snapshot"
  // — wait, no: golden standard §1.2 step 1 says viz_ is FULLY replaced
  // by visibility_mask, not merged. So a missing key means "this
  // field is gone from the schema" (= a bug). Phase 1.6 validation
  // rejects partial masks with MissingFieldMaskError.
  std::unordered_map<std::string, PackedVizTensor> visibility_mask;

  // values[field_name] gives the per-field typed payload. Only fields
  // with at least one viz=1 bit at the perspective have an entry; a
  // field whose mask is all zeros for this perspective is omitted
  // (sparse encoding — golden standard §1.2 keeps wire size O(public)
  // not O(state)).
  //
  // The shape of each AnyMap is game-defined (Phase 3 per-game
  // serializers populate it); Phase 1.6 only carries it through.
  std::unordered_map<std::string, AnyMap> values;
};

// Self-consistency: every key in `values` must also be a key in
// `visibility_mask` AND have at least one viz=1 bit for the recipient
// perspective. Wire payload that fails this is malformed: the sender
// either shipped a value for a slot it claimed was hidden, or shipped
// a value for a field absent from the schema.
//
// Phase 1.6: callable from apply_observation before steps 1-4 to
// reject malformed snapshots up front. Returns silently on success;
// throws std::invalid_argument with a specific message on failure
// (one diagnostic per call so test assertions can be specific).
inline void validate_snapshot_self_consistent(
    const PerspectiveSnapshot& snap, int perspective) {
  for (const auto& kv : snap.values) {
    auto it = snap.visibility_mask.find(kv.first);
    if (it == snap.visibility_mask.end()) {
      throw std::invalid_argument(
          "PerspectiveSnapshot: values has key '" + kv.first +
          "' but visibility_mask has no such key");
    }
    const auto& packed = it->second;
    // Verify at least one viz=1 bit for the perspective. shape last
    // axis is viewer; perspective lives in [0, viewer_count).
    if (packed.shape.empty()) {
      throw std::invalid_argument(
          "PerspectiveSnapshot: visibility_mask for '" + kv.first +
          "' has empty shape (no viewer axis)");
    }
    const int viewer_count = packed.shape.back();
    if (perspective < 0 || perspective >= viewer_count) {
      throw std::invalid_argument(
          "PerspectiveSnapshot: perspective " + std::to_string(perspective) +
          " out of range for field '" + kv.first + "' (viewer_count=" +
          std::to_string(viewer_count) + ")");
    }
    bool any_set = false;
    for (std::size_t i = static_cast<std::size_t>(perspective);
         i < packed.bit_count; i += static_cast<std::size_t>(viewer_count)) {
      if (packed.packed[i / 8] & (std::uint8_t{1} << (i % 8))) {
        any_set = true;
        break;
      }
    }
    if (!any_set) {
      throw std::invalid_argument(
          "PerspectiveSnapshot: values has key '" + kv.first +
          "' but visibility_mask shows no viz=1 slot at perspective " +
          std::to_string(perspective) +
          " — values must be sparse-keyed (omit fully-hidden fields)");
    }
  }
}

}  // namespace board_ai
