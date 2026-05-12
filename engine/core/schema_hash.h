#pragma once

// Schema-driven hash.
//
// `state_hash_for_perspective(player)` walks the FULL schema (every
// declared slot, visible or hidden to `player`) in declaration order ×
// row-major data-axis indices. For each slot the framework first mixes
// in the slot's STRUCTURAL position (field index in schema + idx
// axes), then dispatches the value mix:
//
//   - visible to player (viz=1): call `hash_field_slot(h, name, idx)`,
//     letting the game mix only the slot's value.
//   - hidden from player (viz=0): mix `kHiddenHashSentinel` directly,
//     so "this slot is hidden" participates in the hash structure
//     without leaking truth.
//
// Framework owns the (field_pos, idx[]) mix so games CANNOT forget to
// distinguish e.g. {hand[1]=5, hand[3]=7} from {hand[0]=5, hand[1]=7}
// (BUG-037). Game-side `hash_field_slot` should mix only the value;
// any `idx`-mixing in game code is now redundant and should be
// removed.
//
// There is NO off-schema escape hatch. Every piece of state that
// participates in DAG node identity must be a schema slot — variable-
// length structures get expressed as fixed-shape count arrays
// (multisets), with any visual ordering reconstructed client-side from
// the action stream. See game_interfaces.h IGameState comment block
// and BUG-037 postmortem for the rationale.
//
// Visit order (pinned by `viz::for_each_slot`): schema declaration
// order × row-major data-axis indices. Two states with the same
// schema and same viz_ produce identical visit sequences, so the hash
// is a pure function of (observation history, viz_) — required for
// DAG transposition correctness (BUG-028, BUG-037).

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
  // Walk the FULL schema. For each slot mix the structural position
  // (field index in schema + idx axes) first, then dispatch the value
  // mix: visible → game's `hash_field_slot`, hidden → fixed sentinel.
  const auto& schema = schema_ref();
  // Build a name → field_pos map inline — schema.fields is small
  // (~tens of entries) and this runs once per hash; the lookup keeps
  // per-slot work O(1) without leaking layout to the walker.
  std::size_t cur_field_pos = 0;
  std::string cur_field_name;
  // Reset positional cursor each call. We rely on `for_each_slot`
  // visiting fields in schema.fields order, so we can advance the
  // cursor by tracking the field name as we go.
  viz::for_each_slot(
      *this, schema, player,
      [&](const std::string& name, const std::vector<int>& idx,
          const viz::VizTensor&, bool visible) {
        if (name != cur_field_name) {
          // First slot of a new field: look up its declaration index.
          // O(F) per field-boundary; F is small (~tens).
          for (std::size_t i = 0; i < schema.fields.size(); ++i) {
            if (schema.fields[i].name == name) {
              cur_field_pos = i;
              break;
            }
          }
          cur_field_name = name;
        }
        h.add(cur_field_pos);
        for (int x : idx) h.add(x);
        if (visible) {
          hash_field_slot(h, name, idx);
        } else {
          h.combine(kHiddenHashSentinel);
        }
      });
  return h.finalize();
}

}  // namespace board_ai
