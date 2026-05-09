#pragma once

// Phase 1.4: masked state view + encoder input contract.
//
// `make_masked_state(state, perspective)` returns a clone of `state` with
// every slot whose `viz[..., perspective] == 0` (or whose belief_filled
// bit is set, when the caller threads one in) overwritten with the
// kPlaceholder sentinel for that field's element type. Encoders read
// the masked state with the same field-by-field syntax they always
// have; the masking is enforced at the data layer, not via a
// throw-on-access proxy.
//
// Lifecycle:
//   - Phase 1.4 (this PR): kPlaceholder constants + the make_masked_state
//     declaration + the seat-rotation helper. The body lands in Phase
//     1.5 alongside viz_walker (the body needs the same per-field
//     reflection machinery to walk slots and write the sentinel into
//     each typed payload).
//   - Phase 3 (per-game schema bring-up): each game's encoder switches
//     to taking `const MaskedState&` instead of `const State&`, and
//     emits placeholder features for slots whose value equals
//     kPlaceholder.
//   - Phase 5: the legacy "encoder reads state directly + fishes out
//     hidden fields by ad-hoc viz checks" path is removed; encoders
//     ONLY see the masked clone.
//
// Guard rationale (golden standard §6 / I13): even if an encoder
// forgets a viz check, the worst it can read is kPlaceholder — never a
// hidden truth value. This is information-theoretically equivalent to
// a throw-on-access proxy but is dramatically simpler at the call
// site (game encoders keep their existing `state.hand[p][i]` syntax).

#include <cstdint>
#include <functional>
#include <limits>
#include <memory>

#include "game_interfaces.h"

namespace board_ai {

// MaskedState is structurally `IGameState` — the same concrete subclass
// the game allocated in reset_with_seed. The "masked" part is purely a
// post-processing pass that overwrites hidden-slot values with
// kPlaceholder. Aliased here so call sites can express intent without a
// new type carrying its own vtable.
using MaskedState = IGameState;

// kPlaceholder sentinels.
//
// Encoders treat any slot whose value EQUALS the corresponding
// kPlaceholder as "hidden, emit placeholder feature." The values are
// chosen to be distinguishable from any legitimate game value:
//
//   kPlaceholderInt32 = INT32_MIN: well outside any game's id range
//                                   (cids, action ids are <= ~10^4).
//   kPlaceholderInt8  = -1       : ALSO used by some games as a real
//                                   "empty slot" marker (e.g. Coup
//                                   influence == -1 means revealed).
//                                   For those, the game's masking pass
//                                   uses kPlaceholderInt8Alt below.
//   kPlaceholderInt8Alt = INT8_MIN: -128, never a legitimate cid.
//   kPlaceholderBool  = false    : safe default; bool fields with
//                                   meaningful "hidden" state should
//                                   carry an int viz-paired marker
//                                   instead.
constexpr std::int32_t kPlaceholderInt32 = std::numeric_limits<std::int32_t>::min();
constexpr std::int8_t  kPlaceholderInt8  = std::numeric_limits<std::int8_t>::min();
constexpr bool         kPlaceholderBool  = false;

// Phase 1.5 will land the body. Until then make_masked_state is
// declaration-only; calling it from Phase 1.4 code is a compile-time
// link error, which is intentional — no caller exists yet, and any new
// caller before Phase 1.5 should fail to link rather than silently
// return an unmasked clone.
//
// Signature: returns a fresh IGameState clone with hidden slots
// rewritten. Caller takes ownership.
//
// `belief_filled` is optional. When non-null, slots whose belief_filled
// bit is set along the perspective axis are ALSO masked (the encoder
// must not treat sample-derived values as ground truth). When null,
// only viz=0 slots are masked.
std::unique_ptr<IGameState> make_masked_state(
    const IGameState& state, int perspective,
    const std::unordered_map<std::string, viz::VizTensor>* belief_filled = nullptr);

// Perspective-relative seat rotation helper.
//
// Encoders must lay out per-seat features in PERSPECTIVE-RELATIVE order:
// seat 0 of the encoded tensor is `perspective`, seat 1 is the next
// player to act, etc. This helper enforces a single canonical order so
// migrated encoders all write `for_each_seat_in_perspective_order(p, ...)`
// instead of hand-rolling `(s + perspective) % N`. Phase 3 grep'ing
// every encoder for the manual modulo and replacing it with this helper
// is part of the schema migration checklist.
inline void for_each_seat_in_perspective_order(
    int perspective, int n_players,
    const std::function<void(int /*seat*/, int /*relative_index*/)>& fn) {
  for (int rel = 0; rel < n_players; ++rel) {
    const int seat = (perspective + rel) % n_players;
    fn(seat, rel);
  }
}

}  // namespace board_ai
