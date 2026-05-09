#pragma once

#include <any>
#include <cstddef>
#include <map>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "types.h"

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

  // ========== Framework-provided deterministic RNG (Phase 1.1) ==========
  //
  // Single canonical entry point for all game-side randomness inside
  // do_action_fast / reset_with_seed. Replaces hand-rolled rng_salt /
  // draw_nonce members in each game.
  //
  // Contract:
  //   - rng_salt_ is set once per game via init_rng_state(seed) (called from
  //     each game's reset_with_seed override). It is the secret salt that
  //     varies between sessions; it MUST NEVER appear in hash_public_fields
  //     or hash_private_fields, and MUST NEVER cross the wire (snapshot,
  //     event payload, web protocol). Framework guarantees this by keeping
  //     it as a base-class member that no game's hash override touches.
  //   - draw_nonce_ is monotonically incremented every time derive_rng() is
  //     called. It is NOT secret (it's just a counter) and NOT serialized,
  //     but its monotonic increment guarantees that two consecutive
  //     derive_rng() calls within the same ply produce independent streams.
  //   - domain_tag distinguishes parallel random sources within a single
  //     ply (e.g. "draw from deck A" vs "draw from deck B" must not
  //     correlate even if the same nonce value is reused due to undo). Pass
  //     a stable per-call-site uint64 (e.g. a hash of "splendor_deck_tier1").
  //
  // Lint guard (CI): do_action_fast must not directly construct
  // std::mt19937* / random_device — it must go through derive_rng().
  std::mt19937_64 derive_rng(std::uint64_t domain_tag = 0) {
    std::uint64_t mixed = rng_salt_;
    mixed = murmur3_fmix64(mixed ^ (draw_nonce_ + kGoldenRatio64));
    mixed = murmur3_fmix64(mixed ^ (domain_tag + kGoldenRatio64));
    ++draw_nonce_;
    return std::mt19937_64(mixed);
  }

  // Helper for child reset_with_seed overrides: initializes the RNG salt
  // and resets the nonce counter. Prefer reset_with_seed_base() below; this
  // remains for randomize_unseen / belief paths that need just the RNG
  // half without touching step_count_.
  void init_rng_state(std::uint64_t seed) {
    rng_salt_ = sanitize_seed(seed);
    draw_nonce_ = 0;
  }

  // Single canonical base setup for reset_with_seed overrides (Phase 1.1).
  // Resets framework-managed members (step_count_ + rng salt + nonce) in
  // one call so every game's reset_with_seed has a single uniform first
  // line:
  //
  //   void MyState::reset_with_seed(std::uint64_t seed) {
  //     IGameState::reset_with_seed_base(seed);   // <-- mandatory first line
  //     // ... game-specific state init ...
  //   }
  //
  // CI lint test_reset_with_seed_calls_base enforces the first-line rule.
  void reset_with_seed_base(std::uint64_t seed) {
    step_count_ = 0;
    init_rng_state(seed);
  }

  // Re-randomize the RNG salt by mixing entropy from `source`. Used by
  // randomize_unseen paths in net_adapter / belief.sample to perturb the
  // stream between MCTS simulations without resetting the rest of the
  // state. Resets nonce to 0 so subsequent derive_rng calls within the
  // sim see the new salt cleanly.
  template <typename URNG>
  void reseed_rng(URNG& source) {
    std::uint64_t mix = static_cast<std::uint64_t>(source()) << 32 |
                        static_cast<std::uint64_t>(source());
    rng_salt_ = sanitize_seed(rng_salt_ ^ murmur3_fmix64(mix));
    draw_nonce_ = 0;
  }

  // Read-only accessors (for tests / framework introspection only).
  std::uint64_t rng_salt() const { return rng_salt_; }
  std::uint64_t draw_nonce() const { return draw_nonce_; }

  // Restore RNG state — for undo paths only. UndoRecord snapshots
  // (rng_salt, draw_nonce) before do_action_fast and replays via this on
  // undo to pin the deterministic stream.
  void restore_rng_state(std::uint64_t salt, std::uint64_t nonce) {
    rng_salt_ = salt;
    draw_nonce_ = nonce;
  }

 protected:
  // Counter for DAG acyclicity. Framework-managed. Reset to 0 in
  // reset_with_seed implementations; bumped by begin_step; rolled back by
  // end_step.
  std::uint32_t step_count_ = 0;

  // RNG state — see derive_rng() above for contract.
  std::uint64_t rng_salt_ = 0;
  std::uint64_t draw_nonce_ = 0;
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
  virtual UndoToken do_action_fast(IGameState& state, ActionId action) const = 0;
  virtual void undo_action(IGameState& state, const UndoToken& token) const = 0;

  // `do_action_deterministic` is used by the tail solver path: same as
  // `do_action_fast` but must NEVER draw from hidden sources (deck, bag,
  // opp hand). Games with physical randomness in do_action_fast should
  // override — e.g. Splendor uses `forced_draw_override = -2` sentinel
  // to mean "freeze the random source". See docs/GAME_DEVELOPMENT_GUIDE §9.2.
  virtual UndoToken do_action_deterministic(IGameState& state, ActionId action) const {
    return do_action_fast(state, action);
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
