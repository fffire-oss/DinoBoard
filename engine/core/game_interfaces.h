#pragma once

#include <any>
#include <cstddef>
#include <map>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "types.h"
#include "visibility_schema.h"

namespace board_ai {

// ---- Event protocol primitives (used by IBeliefTracker and game bundle) ----
// AnyMap: game-specific payload for events and observations. Keys are
// stable string names chosen by the game; values are wrapped in std::any
// so different games can carry different payload types.
using AnyMap = std::map<std::string, std::any>;

// A single public event: (kind, payload). Kind is a stable string chosen by
// the game (e.g. "deck_flip", "opp_buy_reserved_reveal"). Payload keys/values
// are game-defined.
using PublicEvent = std::pair<std::string, AnyMap>;

// Events + public snapshot for a single action.
//
// `events` — public observations the perspective player can derive from this
// transition. Consumed only by the belief tracker (observer sessions never
// run `do_action_fast`, so events have no in-rules timing to ride). The list
// order matches the producer's emission order; the tracker treats it as an
// observation log.
//
// `public_snapshot` — truth-side serialization of the post-action visible
// fields for this perspective. Games that register a `public_state_applier`
// populate this in their extractor; observer sessions then overwrite their
// state_'s public fields from it at the end of apply_observation. Empty map
// for fully-public games (tictactoe, quoridor) that do not register an
// applier.
struct PublicEventTrace {
  std::vector<PublicEvent> events{};
  AnyMap public_snapshot{};
};

class IGameState {
 public:
  virtual ~IGameState() = default;
  virtual std::unique_ptr<IGameState> clone_state() const = 0;
  virtual void copy_from(const IGameState& other) = 0;

  // Legacy full-state hash. Retained for non-ISMCTS paths (tail solver,
  // transposition debug). New code paths use state_hash_for_perspective.
  virtual StateHash64 state_hash() const = 0;

  // ========== Schema-driven hash API ==========
  //
  // `state_hash_for_perspective(player)` walks the FULL schema (every
  // slot, visible and hidden) in declaration order × row-major idx.
  // For each slot the framework first mixes the structural position
  // (field index in schema + idx[]), then dispatches the value mix:
  //   - visible to `player` (viz=1): call `hash_field_slot`, which
  //     mixes only the slot's value.
  //   - hidden from `player` (viz=0): mix `kHiddenHashSentinel`.
  //
  // Every piece of game state that participates in DAG node identity
  // MUST be a schema slot. There is no off-schema escape hatch — if
  // you find yourself wanting one, declare a fixed-shape count array
  // and treat the variable-length structure as a (multiset, ordering)
  // pair where ordering is purely a presentation concern reconstructed
  // from the action stream. (Why: any off-schema state silently breaks
  // the framework's "two states with the same observation history
  // produce the same digest" guarantee — see BUG-037 postmortem.)

  // Typed value emission for one schema slot. The framework walker
  // visits every visible (name, idx) and dispatches here; games answer
  // with `h.add(this->myfield[idx...])`. Walker has already filtered
  // on viz, so this must NOT re-consult viz_.
  virtual void hash_field_slot(Hasher& /*h*/, const std::string& /*name*/,
                               const std::vector<int>& /*idx*/) const {}

  // Typed slot read/write for walker-driven snapshot wire I/O
  // (`viz::serialize_public` / `viz::apply_public`). Game returns a
  // typed std::any for the named slot, or accepts one back. Mirrors
  // hash_field_slot / mask_field_slot dispatch shape.
  virtual std::any read_field_slot(const std::string& /*name*/,
                                   const std::vector<int>& /*idx*/) const {
    return {};
  }
  virtual void write_field_slot(const std::string& /*name*/,
                                const std::vector<int>& /*idx*/,
                                const std::any& /*value*/) {}

  // Polymorphic accessor for the game's static VisibilitySchema.
  // Framework helpers (`make_masked_state`, MCTS descent hoist) call
  // this to drive the walker without knowing the concrete state type.
  virtual const viz::VisibilitySchema& schema_ref() const = 0;

  // Framework-provided perspective hash. Walks the FULL schema, mixes
  // (field_pos, idx[]) for every slot, then dispatches the value mix:
  // visible → `hash_field_slot`, hidden → `kHiddenHashSentinel`.
  // NOT virtual — games extend only via `hash_field_slot`. Definition
  // lives in schema_hash.h to break the include cycle.
  StateHash64 state_hash_for_perspective(int player) const;

  // ========== Framework-provided step counter ==========
  //
  // Monotonically increasing across do_action_fast calls. Guarantees DAG
  // acyclicity in MCTS: any two states along a sim path have distinct
  // step_count → distinct hash (assuming hash_public_fields includes it,
  // which state_hash_for_perspective does automatically).
  //
  // Games MUST NOT touch step_count directly. The framework bumps it at the
  // start of do_action_fast via `begin_step()` and rolls it back in
  // undo_action via `end_step()`. Games call those helpers from inside
  // their overrides.
  std::uint32_t step_count() const { return step_count_; }
  void begin_step() { ++step_count_; }
  void end_step() { --step_count_; }

  virtual int current_player() const = 0;
  virtual int first_player() const { return 0; }
  virtual bool is_terminal() const = 0;
  virtual int num_players() const = 0;
  virtual int winner() const = 0;
  virtual bool is_turn_start() const { return true; }
  virtual void reset_with_seed(std::uint64_t seed) = 0;

  // Single canonical base setup for reset_with_seed overrides. Resets the
  // framework-managed step counter; games must call this as the first line
  // of their reset_with_seed override. RNG is NOT framework state — each
  // game's reset_with_seed constructs a one-shot std::mt19937_64 from the
  // seed for its initial deal/shuffle and drops it on return.
  void reset_step_count_base() {
    step_count_ = 0;
  }

 protected:
  // Counter for DAG acyclicity. Framework-managed. Reset to 0 in
  // reset_with_seed implementations; bumped by begin_step; rolled back by
  // end_step.
  std::uint32_t step_count_ = 0;

 public:
  // Per-state visibility tensor, keyed by FieldDecl::name. Initialized
  // by viz::init_viz at reset_with_seed; mutated only by rules inside
  // do_action_fast via reveal_slot / reveal_slot_to / reset_to_base
  // (golden standard I1 — rules are the sole writer). Framework readers
  // (hash walker, snapshot serializer, mask_all_hidden_slots) read it
  // directly.
  std::unordered_map<std::string, viz::VizTensor> viz_;

  // Per-slot placeholder write for hidden slots. Walker calls this for
  // each (name, idx) where viz[idx, perspective] == 0 (or belief_filled
  // is set). Game writes kPlaceholderInt32 / kPlaceholderInt8 /
  // kPlaceholderBool into the matching typed slot. Must be idempotent.
  // Default no-op covers fully-public games whose walker never visits
  // a hidden slot.
  virtual void mask_field_slot(const std::string& /*name*/,
                               const std::vector<int>& /*idx*/) {}

  // Drives the walker over hidden slots and dispatches each to
  // `mask_field_slot`. Virtual so games with COW-shared state (e.g.
  // Splendor's shared_ptr<const SplendorData>) can detach a writable
  // copy once before running the walker, instead of reseating per slot.
  virtual void mask_all_hidden_slots(
      const viz::VisibilitySchema& schema, int perspective,
      const std::unordered_map<std::string, viz::VizTensor>* belief_filled);
};

template <typename Derived>
class CloneableState : public IGameState {
 public:
  std::unique_ptr<IGameState> clone_state() const override {
    return std::make_unique<Derived>(static_cast<const Derived&>(*this));
  }
  void copy_from(const IGameState& other) override {
    static_cast<Derived&>(*this) = static_cast<const Derived&>(other);
  }
};

template <typename ConcreteState>
const ConcreteState& checked_cast(const IGameState& state) {
  const auto* p = dynamic_cast<const ConcreteState*>(&state);
  if (!p) throw std::invalid_argument("unexpected IGameState subtype");
  return *p;
}

template <typename ConcreteState>
ConcreteState& checked_cast(IGameState& state) {
  auto* p = dynamic_cast<ConcreteState*>(&state);
  if (!p) throw std::invalid_argument("unexpected IGameState subtype");
  return *p;
}

class IGameRules {
 public:
  virtual ~IGameRules() = default;
  virtual bool validate_action(const IGameState& state, ActionId action) const = 0;
  virtual std::vector<ActionId> legal_actions(const IGameState& state) const = 0;

  // Caller-owned RNG threads through hidden draws (deck pulls, bag draws).
  // The state itself holds no rng; the runner / session owns one and
  // refreshes it between sims. Deterministic-path callers should use
  // `do_action_deterministic` instead — that path is rng-free by
  // construction (games freeze hidden draws via their own mechanic, e.g.
  // Splendor's forced_draw_override = -2 sentinel).
  virtual UndoToken do_action_fast(IGameState& state, ActionId action,
                                   std::mt19937_64& rng) const = 0;
  virtual void undo_action(IGameState& state, const UndoToken& token) const = 0;

  // `do_action_deterministic` is used by the tail solver path: same as
  // `do_action_fast` but must NEVER draw from hidden sources (deck, bag,
  // opp hand). Games with physical randomness in do_action_fast should
  // override — e.g. Splendor uses `forced_draw_override = -2` sentinel
  // to mean "freeze the random source". See docs/GAME_DEVELOPMENT_GUIDE §9.2.
  virtual UndoToken do_action_deterministic(IGameState& state, ActionId action) const {
    // Default: forward to do_action_fast with a throwaway rng. Subclasses
    // that have hidden draws MUST override this to suppress them.
    std::mt19937_64 unused_rng(0);
    return do_action_fast(state, action, unused_rng);
  }
};

class IStateValueModel {
 public:
  virtual ~IStateValueModel() = default;
  virtual float terminal_value_for_player(const IGameState& state, int player) const = 0;

  virtual std::vector<float> terminal_values(const IGameState& state) const {
    const int n = state.num_players();
    std::vector<float> v(static_cast<size_t>(n));
    for (int p = 0; p < n; ++p) {
      v[static_cast<size_t>(p)] = terminal_value_for_player(state, p);
    }
    return v;
  }
};

class DefaultStateValueModel final : public IStateValueModel {
 public:
  float terminal_value_for_player(const IGameState& state, int player) const override {
    if (!state.is_terminal()) return 0.0f;
    const int w = state.winner();
    if (w < 0) return 0.0f;
    const int n = state.num_players();
    if (n <= 2) return w == player ? 1.0f : -1.0f;
    return w == player ? 1.0f : -1.0f / static_cast<float>(n - 1);
  }
};

}  // namespace board_ai
