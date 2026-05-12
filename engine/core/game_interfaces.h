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

// Phase of a public event relative to the action it describes:
//   kPreAction  — the event describes hidden info the action depends on
//                 (e.g. Baron target's hand in Love Letter). Applied BEFORE
//                 rules.apply() during event replay.
//   kPostAction — the event describes a random outcome the action produced
//                 (e.g. Splendor deck flip). Applied AFTER rules.apply().
enum class EventPhase { kPreAction = 0, kPostAction = 1 };

// A single public event: (kind, payload). Kind is a stable string chosen by
// the game (e.g. "deck_flip", "hand_override"). Payload keys/values are
// game-defined.
using PublicEvent = std::pair<std::string, AnyMap>;

// Events for a single action, split by phase.
//
// `public_snapshot` — truth-side serialization of the post-action public
// fields. Games that register a `public_state_applier` (see game_registry.h)
// populate this in their extractor; observer sessions then overwrite their
// state_'s public fields from it at the end of apply_observation, so the
// observer's public state is rebuilt from the message stream and never
// depends on `do_action_fast(observer_state_)`'s output. Empty map for
// fully-public games (tictactoe, quoridor) that do not register an applier.
struct PublicEventTrace {
  std::vector<PublicEvent> pre_events{};
  std::vector<PublicEvent> post_events{};
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
  // `state_hash_for_perspective(player)` walks the visibility schema in
  // declaration order and dispatches `hash_field_slot(h, name, idx)`
  // for every slot whose runtime viz is 1 for `player`. The visited
  // set IS `player`'s information set — schema + viz tensor define
  // both public and private partitions; the framework needs no separate
  // hash_public/private API.
  //
  // Off-schema state (variable-length lists, conditionally-public slots
  // not yet wired through `viz::reveal_slot`) folds in via the optional
  // `hash_extra_state_fields` hook, called after the walker pass.
  // LoveLetter / Coup currently use this for `discard_piles` / deck
  // size / revealed-influence patches; once those are schema-driven
  // (Phase 3 §G), the hook drops to no-op and its overrides delete.

  // Typed value emission for one schema slot. The framework walker
  // visits every visible (name, idx) and dispatches here; games answer
  // with `h.add(this->myfield[idx...])`. Walker has already filtered
  // on viz, so this must NOT re-consult viz_.
  virtual void hash_field_slot(Hasher& /*h*/, const std::string& /*name*/,
                               const std::vector<int>& /*idx*/) const {}

  // Hash off-schema state for `perspective`. Default no-op; games that
  // hold variable-length lists or conditionally-public slots not yet
  // expressed via the schema/viz pipeline override this. Called after
  // the walker pass inside `state_hash_for_perspective`.
  virtual void hash_extra_state_fields(int /*perspective*/,
                                       Hasher& /*h*/) const {}

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

  // Framework-provided perspective hash. Walks the schema and dispatches
  // `hash_field_slot` for every slot whose runtime viz is 1 for `player`,
  // then folds in `hash_extra_state_fields(player, h)` for off-schema
  // state. NOT virtual — games extend via `hash_field_slot` and the
  // optional extras hook, not by overriding this. Definition lives in
  // schema_hash.h to break the include cycle (walker depends on
  // viz_runtime which depends on this header).
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
