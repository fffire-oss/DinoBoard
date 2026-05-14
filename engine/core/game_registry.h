#pragma once

#include <any>
#include <functional>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "belief_tracker.h"
#include "feature_encoder.h"
#include "game_interfaces.h"
#include "../search/tail_solver.h"

namespace board_ai {

// AnyMap, PublicEvent, PublicEventTrace are defined in
// game_interfaces.h so IBeliefTracker can reference them without a
// circular include.

using StateSerializer = std::function<AnyMap(const IGameState& state)>;

using ActionDescriptor = std::function<AnyMap(ActionId action)>;

struct HeuristicResult {
  std::vector<ActionId> actions;
  std::vector<double> scores;
};

using HeuristicPicker = std::function<
    HeuristicResult(IGameState& state, const IGameRules& rules, std::uint64_t rng_seed)>;

struct SelfplaySampleView {
  int ply;
  int player;
  ActionId action_id;
};

using EpisodeStatsExtractor = std::function<
    std::map<std::string, double>(
        const IGameState& final_state,
        const std::vector<SelfplaySampleView>& samples)>;

using GameAdjudicator = std::function<int(const IGameState& state)>;

using AuxiliaryScorer = std::function<float(const IGameState& state, int player)>;

using TrainingActionFilter = std::function<std::vector<ActionId>(
    const IGameState& state, const IGameRules& rules, const std::vector<ActionId>& legal)>;

using TailSolveTrigger = std::function<bool(const IGameState& state, int ply)>;

// Public-event protocol for the AI API (platform/ai_service):
//
// When the AI session is driven from observations, ground-truth's random
// outcomes (deck flips, hidden draws, revealed cards during challenges)
// must reach the AI's belief tracker as events. Each stochastic game
// registers a PublicEventExtractor — during self-play, it diffs
// state_before and state_after to produce the events an outside observer
// at perspective would have seen. Used to generate ground-truth traces
// for tests. The events go ONLY to the belief tracker (observer
// sessions do not run do_action_fast — public state is rebuilt from
// public_snapshot).
//
// Initial observation handshake (the perspective-specific facts the AI
// knows at game start, e.g. own starting hand) is NOT a per-game hook —
// it is walker-driven via viz::serialize_public_snapshot /
// viz::apply_public_snapshot, the same path the per-ply public_snapshot
// uses. Games only have to declare schema visibility correctly for
// game-start; the framework walker handles serialization in both
// directions.
//
// PublicEvent / PublicEventTrace are declared in game_interfaces.h (so
// belief_tracker.h can reference them without a circular include).

using PublicEventExtractor = std::function<PublicEventTrace(
    const IGameState& state_before,
    ActionId action,
    const IGameState& state_after,
    int perspective_player)>;

// Message-driven public state: inverse of the public-fields-only
// serialization produced by public_event_extractor. Overwrites the
// observer session's state_ public fields from the truth snapshot at
// the end of apply_observation, so the observer's public state is
// rebuilt from the message stream — it never depends on whatever
// `do_action_fast(state_)` computed from its sampled hidden fields.
//
// Contract:
//   - Applier OVERWRITES every public field that state_hash_for_perspective
//     depends on. Fields absent from the snapshot are left unchanged
//     (but see `test_public_snapshot_round_trip` — every field that
//     state_hash_for_perspective reads must round-trip through snapshot+applier).
//   - Applier MUST NOT touch hidden fields. Those are left for
//     tracker.randomize_unseen to fill.
//
// Required for all hidden-info games. Fully-public games (tictactoe,
// quoridor) do not register one — there is no hidden for do_action_fast
// to read wrong, so the public side is already tamper-proof.
// receiver_seat: the perspective whose session this state belongs to.
// Needed by `viz::apply_public_snapshot` to overwrite the receiver's
// viz slice (`viz_[name][..., receiver_seat]`) byte-for-byte from the
// wire's `__viz__` payload. Pass -1 from contexts where the seat is
// unknown / N/A (fully-public games whose appliers don't consult it).
using PublicStateApplier = std::function<void(
    IGameState& state,
    const AnyMap& snapshot,
    int receiver_seat)>;

struct GameBundle {
  std::unique_ptr<IGameState> state;
  std::unique_ptr<IGameRules> rules;
  std::unique_ptr<IStateValueModel> value_model;
  std::unique_ptr<IFeatureEncoder> encoder;
  std::unique_ptr<IBeliefTracker> belief_tracker;
  StateSerializer state_serializer;
  ActionDescriptor action_descriptor;
  HeuristicPicker heuristic_picker;
  std::unique_ptr<search::ITailSolver> tail_solver;
  // Set to true only after verifying that do_action_deterministic is
  // overridden and NEVER consumes hidden chance outcomes (drawing from
  // deck, flipping face-down, etc.). Required to combine tail_solver with
  // a belief_tracker. See create_game() for the full contract.
  bool stochastic_tail_solve_safe = false;
  TailSolveTrigger tail_solve_trigger;
  EpisodeStatsExtractor episode_stats_extractor;
  GameAdjudicator adjudicator;
  AuxiliaryScorer auxiliary_scorer;
  TrainingActionFilter training_action_filter;
  // Public-event protocol (see above). Optional — only needed for games
  // that register a belief_tracker AND want to be driveable through the
  // AI API with independent seeds.
  PublicEventExtractor public_event_extractor;
  // Required for hidden-info games. See PublicStateApplier above.
  // Fully-public games (tictactoe, quoridor) leave this unset.
  PublicStateApplier public_state_applier;
  std::string game_id;
};

using GameFactory = std::function<GameBundle(std::uint64_t seed)>;

class GameRegistry {
 public:
  static GameRegistry& instance() {
    static GameRegistry reg;
    return reg;
  }

  void register_game(const std::string& game_id, GameFactory factory) {
    factories_[game_id] = std::move(factory);
  }

  GameBundle create_game(const std::string& game_id, std::uint64_t seed) const {
    auto it = factories_.find(game_id);
    if (it == factories_.end()) {
      throw std::runtime_error("Unknown game: " + game_id);
    }
    GameBundle bundle = it->second(seed);
    // Invariant: if a game has both hidden information (belief_tracker) and a
    // tail solver, it MUST provide a deterministic action API. The tail solver
    // expands the game tree assuming each branch is deterministic; if the
    // game's do_action_fast draws from a hidden deck to advance turns, the
    // solver "peeks" at the true deck and produces a proven-win strategy that
    // is only correct against that specific realization. This silently
    // poisons training samples — it violates the AI-pipeline-independence
    // design principle.
    //
    // Games that hit this combination must override do_action_deterministic
    // AND flip stochastic_tail_solve_safe = true as an explicit
    // acknowledgment. The default is false — you have to opt in.
    if (bundle.belief_tracker && bundle.tail_solver &&
        !bundle.stochastic_tail_solve_safe) {
      throw std::runtime_error(
          "Game '" + game_id + "' registers both a belief_tracker and a "
          "tail_solver, but has not set stochastic_tail_solve_safe = true. "
          "This combination is unsafe unless the game's do_action_deterministic "
          "is overridden so tail-solver's alpha-beta search never consumes "
          "hidden chance outcomes. Override do_action_deterministic in your "
          "rules (it must advance turns without drawing hidden cards/tiles), "
          "then set stochastic_tail_solve_safe = true in the GameBundle.");
    }
    return bundle;
  }

  bool has_game(const std::string& game_id) const {
    return factories_.count(game_id) > 0;
  }

  std::vector<std::string> game_ids() const {
    std::vector<std::string> ids;
    ids.reserve(factories_.size());
    for (const auto& [id, _] : factories_) {
      ids.push_back(id);
    }
    return ids;
  }

 private:
  GameRegistry() = default;
  std::unordered_map<std::string, GameFactory> factories_;
};

struct GameRegistrar {
  GameRegistrar(const std::string& game_id, GameFactory factory) {
    GameRegistry::instance().register_game(game_id, std::move(factory));
  }
};

}  // namespace board_ai
