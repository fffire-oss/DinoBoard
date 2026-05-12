#pragma once

// Schema-driven hash.
//
// `state_hash_for_perspective(player)` walks the visibility schema in
// declaration order and dispatches `hash_field_slot` for every slot
// whose runtime viz is 1 for `player`. The visited set is exactly
// `player`'s information set; no separate public/private partition is
// needed in the hash function — the schema + viz tensor define both.
//
// Off-schema state (variable-length lists, conditionally-public slots
// not yet wired through `viz::reveal_slot`) goes into the optional
// `hash_extra_state_fields` hook, called after the walker pass. This
// keeps a temporary escape hatch for fields that haven't migrated into
// the schema yet (Coup `revealed`, LL `discard_piles`, Splendor deck
// sizes); once those are schema-driven the hook drops to no-op.
//
// Visit order (pinned by `viz::for_each_visible_slot`): schema
// declaration order × row-major data-axis indices. Two states with the
// same schema and same viz_ produce identical visit sequences, so the
// hash is a pure function of (observation history, viz_) — required
// for DAG transposition correctness (BUG-028).

#include <stdexcept>
#include <string>
#include <vector>

#include "game_interfaces.h"
#include "masked_state.h"
#include "types.h"
#include "visibility_schema.h"
#include "viz_runtime.h"
#include "viz_walker.h"

namespace board_ai {

// Out-of-class definition of IGameState::state_hash_for_perspective.
// Lives here because the schema walker depends on viz_runtime which
// depends on game_interfaces — the body cannot live inline in the
// IGameState class.
inline StateHash64 IGameState::state_hash_for_perspective(int player) const {
  Hasher h;
  h.add(step_count_);
  viz::for_each_visible_slot(
      *this, schema_ref(), player,
      [&](const std::string& name, const std::vector<int>& idx,
          const viz::VizTensor&) {
        hash_field_slot(h, name, idx);
      });
  hash_extra_state_fields(player, h);
  return h.finalize();
}

}  // namespace board_ai
