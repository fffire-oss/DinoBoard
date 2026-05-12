#pragma once

// MaskedState — the encoder input contract (golden standard §2.5).
//
// `make_masked_state(state, schema, perspective, belief_filled)`
// clones `state` and overwrites every slot whose
// `viz[..., perspective] == 0` (or whose `belief_filled` bit is set)
// with the kPlaceholder sentinel for that slot's element type.
//
// MaskedState is the SINGLE encoder-input type — encoders read it with
// the same field-by-field syntax they always have, with no viz query
// needed. A slot whose value equals kPlaceholder is hidden (or belief-
// sampled); anything else is truth. The walker over schema × viz lives
// in viz_walker.h and feeds three consumers: hash, snapshot serializer,
// and (via this MaskedState) encoder. They share one definition of
// "visible to perspective p."

#include <cstdint>
#include <limits>
#include <memory>

#include "game_interfaces.h"

namespace board_ai {

// MaskedState is structurally identical to IGameState (typedef alias) —
// the game's own concrete subclass, with a placeholder-only pass over
// hidden slots. No new vtable.
using MaskedState = IGameState;

// kPlaceholder sentinels — encoders treat any slot whose value equals
// the corresponding sentinel as "hidden / belief-sampled." Values are
// outside legitimate ranges:
//   - kPlaceholderInt32 (INT32_MIN): outside cid / action-id ranges.
//   - kPlaceholderInt8  (INT8_MIN, -128): distinguishable from -1
//     (which is the publicly-observable "empty/revealed" marker).
//   - kPlaceholderBool  (false): bool fields needing a "hidden" state
//     should pair with an int viz-paired marker instead.
constexpr std::int32_t kPlaceholderInt32 = std::numeric_limits<std::int32_t>::min();
constexpr std::int8_t  kPlaceholderInt8  = std::numeric_limits<std::int8_t>::min();
constexpr bool         kPlaceholderBool  = false;

// Returns a fresh clone of `state` with every hidden slot overwritten
// with kPlaceholder. `belief_filled`, if non-null, masks belief-
// sampled slots in addition to viz=0 slots.
//
// Body: clone, then call the cloned state's virtual
// `mask_all_hidden_slots` — it walks the schema and dispatches each
// hidden slot back to per-game `mask_field_slot(name, idx)` for the
// placeholder write. Games whose state shares persistent immutable
// data (e.g. Splendor's shared_ptr<const SplendorData>) override
// `mask_all_hidden_slots` to detach a writable copy first.
inline std::unique_ptr<IGameState> make_masked_state(
    const IGameState& state, const viz::VisibilitySchema& schema,
    int perspective,
    const std::unordered_map<std::string, viz::VizTensor>* belief_filled = nullptr) {
  auto cloned = state.clone_state();
  cloned->mask_all_hidden_slots(schema, perspective, belief_filled);
  return cloned;
}

}  // namespace board_ai
