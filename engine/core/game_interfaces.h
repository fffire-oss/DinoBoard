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
  virtual StateHash64 state_hash(bool include_hidden_rng) const = 0;

  // ========== Public/private hash API ==========
  //
  // Game declares which fields are public and which are private per-player.
  // Framework derives state_hash_for_perspective(p) by combining:
  //   step_count, then hash_public_fields(h), then hash_private_fields(p, h).
  //
  // Contract:
  //   - hash_public_fields: hash every field that ALL players can see
  //   - hash_private_fields(p): hash every field that ONLY player p can see
  //     (plus p's own knowledge/belief-derived info that p is allowed to have)
  //   - Do NOT hash fields belonging to players other than p in
  //     hash_private_fields. Do NOT hash private fields in hash_public_fields.
  //
  // The same partition drives MCTS node keying, encoder feature extraction,
  // and the AI-pipeline-no-leak test suite. Keeping the two methods aligned
  // with encoder scope is the game author's responsibility; the framework
  // enforces the partition is well-defined via hash tests.
  virtual void hash_public_fields(Hasher& h) const = 0;
  virtual void hash_private_fields(int player, Hasher& h) const = 0;

  // Phase 3.6.a hook: typed value emission for one schema slot.
  //
  // Used by `framework::hash_public_via_schema` / `hash_private_via_schema`
  // (engine/core/schema_hash.h) when a game's hash_public_fields /
  // hash_private_fields delegate to the framework walker. Implementations
  // dispatch on `name` to `h.add(this->myfield[idx0][idx1]...)` — a single
  // mechanical switch per game replacing the per-field hand-written loops.
  //
  // Contract:
  //   - Must NOT consult viz_; visibility filtering is done by the walker
  //     before this hook is invoked.
  //   - Must emit a stable, deterministic byte sequence per (name, idx).
  //   - Default body is empty: games still using the legacy hand-written
  //     hash_public_fields / hash_private_fields don't need to override
  //     this until they migrate to the schema-driven path.
  virtual void hash_field_slot(Hasher& /*h*/, const std::string& /*name*/,
                               const std::vector<int>& /*idx*/) const {}

  // Framework-provided perspective hash. Combines step_count (for DAG
  // acyclicity) + public fields + given player's private fields. NOT
  // virtual — games override the two helpers above, not this.
  StateHash64 state_hash_for_perspective(int player) const {
    Hasher h;
    h.add(step_count_);
    hash_public_fields(h);
    hash_private_fields(player, h);
    return h.finalize();
  }

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
  // ========== Per-state visibility tensor (Phase 1.2) ==========
  //
  // viz_ stores the live per-field visibility tensor for THIS state, keyed
  // by FieldDecl::name. Initialized via viz::init_viz(*this, MyGame::schema())
  // at the end of every game's reset_with_seed override (Phase 3 wires this
  // into each game; Phase 1.2 only adds the storage + helpers). Mutated only
  // by rules inside do_action_fast via the rules-side helpers
  // (viz::reveal_slot / reveal_slot_to / reset_to_base — Phase 1.5).
  //
  // Public so framework helpers (init_viz / reveal_slot / hash walker /
  // encoder masker) can read/write without friending every utility. Games
  // MUST treat viz_ as opaque outside do_action_fast (golden standard I1:
  // rules are the sole writer).
  std::unordered_map<std::string, viz::VizTensor> viz_;

  // ========== Per-perspective masking hook (Phase 1.5) ==========
  //
  // Called by framework's `make_masked_state(state, perspective)` on a
  // freshly cloned state. Game's override walks its own viz_ and writes
  // the corresponding kPlaceholder sentinel into every C++ field slot
  // whose `viz[..., perspective] == 0`.
  //
  // Why the game writes its own mask: framework code holds only
  // `IGameState&` and has no way to reach `state.influence[p][i]` (or
  // any other game-specific typed field) without knowing the concrete
  // subclass layout. The game does know its layout, so it does the
  // write — read viz_["field_name"] for which slots to clobber, then
  // assign kPlaceholderInt32 / kPlaceholderInt8 / kPlaceholderBool
  // (from masked_state.h) into those slots.
  //
  // Default body: no-op. Correct for any state with empty viz_ (no
  // schema declared yet — fully-public games before Phase 3, and all
  // 6 games during Phase 1.5 since schemas land per-game in Phase 3).
  // Once a game declares its schema, it overrides this to perform the
  // actual mask write.
  //
  // Contract:
  //   - May NOT touch viz_ itself, only the typed payload fields.
  //   - May NOT touch step_count_.
  //   - Must be idempotent — calling twice with the same perspective
  //     produces the same state (because placeholder == placeholder).
  //   - Must handle perspective in [0, num_players()).
  virtual void apply_viz_mask(int /*perspective*/) {}
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
