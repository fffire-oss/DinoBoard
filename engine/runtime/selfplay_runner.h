#pragma once

#include <any>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "../core/belief_tracker.h"
#include "../core/feature_encoder.h"
#include "../core/game_interfaces.h"
#include "../core/game_registry.h"
#include "../search/net_mcts.h"
#include "../search/tail_solver.h"
#include "../search/temperature_schedule.h"
#include "../search/root_noise.h"

namespace board_ai::runtime {

struct SelfplaySample {
  int ply = 0;
  int player = 0;
  ActionId action_id = -1;
  std::vector<ActionId> policy_action_ids{};
  std::vector<int> policy_action_visits{};
  std::vector<float> features{};
  std::vector<float> legal_mask{};
  float z = 0.0f;
  std::vector<float> z_values{};
  bool tail_solved = false;
  float auxiliary_score = 0.0f;
  bool has_auxiliary_score = false;
};

// Per-ply observation trace for API belief-equivalence tests. Only populated
// when run_selfplay_episode is called with trace_perspective >= 0 AND the
// game has registered a public_event_extractor.
//
// `public_snapshot`: populated iff the game's extractor fills it (==
// game registers public_state_applier). Empty map otherwise.
struct SelfplayObservationTrace {
  int ply = 0;
  int actor = 0;                           // player whose action this was
  ActionId action = -1;
  std::vector<std::pair<std::string, AnyMap>> events{};
  AnyMap public_snapshot{};
  std::map<std::string, std::any> belief_snapshot_after{};
};

struct SelfplayEpisodeResult {
  int winner = -1;
  bool draw = false;
  int total_plies = 0;
  std::vector<SelfplaySample> samples{};
  std::map<std::string, double> custom_stats{};
  std::unique_ptr<IGameState> final_state{};

  int tail_solve_attempts = 0;
  int tail_solve_completed = 0;
  int tail_solve_successes = 0;
  double tail_solve_total_ms = 0.0;

  bool trace_enabled = false;
  int trace_perspective = -1;
  std::map<std::string, std::any> initial_belief_snapshot{};
  AnyMap initial_observation{};
  std::vector<SelfplayObservationTrace> observation_trace{};
};

struct SelfplayConfig {
  int simulations = 200;
  float c_puct = 1.4f;
  int max_depth = 128;
  float value_clip = 1.0f;
  double temperature = 1.0;
  search::TemperatureSchedule temperature_schedule{};
  double dirichlet_alpha = 0.0;
  double dirichlet_epsilon = 0.0;
  int dirichlet_on_first_n_plies = 0;
  int max_game_plies = 500;

  bool tail_solve_enabled = false;
  search::TailSolveConfig tail_solve_config{};

  double heuristic_guidance_ratio = 0.0;
  double heuristic_temperature = 0.0;
  double training_filter_ratio = 1.0;

  // See OpponentSelection in net_mcts.h. Default kPuct is current
  // behavior. kFrozenPrior turns descent on opponent nodes into a
  // multinomial sample from the policy head's prior, mitigating
  // strategy fusion in ISMCTS for hidden-info games.
  search::OpponentSelection opponent_selection = search::OpponentSelection::kPuct;
};

using GameAdjudicator = board_ai::GameAdjudicator;
using AuxiliaryScorer = board_ai::AuxiliaryScorer;
using HeuristicPicker = board_ai::HeuristicPicker;
using TrainingActionFilter = board_ai::TrainingActionFilter;
using TailSolveTrigger = board_ai::TailSolveTrigger;

// Wraps another IGameRules, filtering legal_actions through a training
// filter. Forwards every other entry point to the inner rules. Because the
// framework wrappers around do_action_fast / do_action_deterministic /
// undo_action are non-virtual and bump state.step_count_ themselves, this
// class must NOT override them; instead it forwards through the *_impl
// hooks so the inner rules' impl runs exactly once and step_count_ stays
// in sync (we delegate to inner_'s public wrappers, which would double-bump
// if we then also bumped here — so we just call the impl side directly via
// inner_ as friend? No — IGameRules' public wrappers do the bookkeeping;
// our impls run after our own bump. Net: we forward by calling
// inner_.do_action_fast_impl etc. directly via the friend access).
class FilteredRulesWrapper final : public IGameRules {
 public:
  FilteredRulesWrapper(const IGameRules& inner, TrainingActionFilter filter)
      : inner_(inner), filter_(std::move(filter)) {}

  bool validate_action(const IGameState& state, ActionId action) const override {
    return inner_.validate_action(state, action);
  }

  std::vector<ActionId> legal_actions(const IGameState& state) const override {
    auto legal = inner_.legal_actions(state);
    if (filter_) {
      auto filtered = filter_(state, inner_, legal);
      if (!filtered.empty()) return filtered;
    }
    return legal;
  }

 protected:
  // Our wrapper bumps step_count_ once (in IGameRules::do_action_fast),
  // then we delegate to the inner rules' impl directly so it runs without
  // a second bump. Same for deterministic / undo. Sibling-instance access
  // to protected impls goes through the static invoke_* helpers on
  // IGameRules.
  void do_action_fast_impl(IGameState& state, ActionId action,
                           std::mt19937_64& rng) const override {
    invoke_do_action_fast_impl(inner_, state, action, rng);
  }
  void undo_action_impl(IGameState& state, const UndoToken& token) const override {
    invoke_undo_action_impl(inner_, state, token);
  }
  UndoToken do_action_deterministic_impl(IGameState& state,
                                         ActionId action) const override {
    return invoke_do_action_deterministic_impl(inner_, state, action);
  }

 private:
  const IGameRules& inner_;
  TrainingActionFilter filter_;
};

using PublicEventExtractor = board_ai::PublicEventExtractor;
using PublicStateApplier = board_ai::PublicStateApplier;

// Per-seat policy/value evaluator lookup. Selfplay's opponent-pool path and
// arena both need this shape — same callback wired through both runners so
// MCTS at each ply queries the network registered for the acting seat.
using PolicyEvaluatorFactory = std::function<
    const search::IPolicyValueEvaluator&(int player_index)>;

SelfplayEpisodeResult run_selfplay_episode(
    IGameState& initial_state,
    const IGameRules& rules,
    const IStateValueModel& value_model,
    PolicyEvaluatorFactory evaluator_for_player,
    const SelfplayConfig& config,
    std::uint64_t episode_seed,
    // One tracker per player seat. For hidden-info games: size == num_players;
    // the runner inits each seat once at episode start, feeds every public
    // event to every tracker, and routes MCTS root to
    // per_perspective_trackers[current_player]. For games without hidden info:
    // empty vector; the runner skips tracker wiring entirely.
    std::vector<IBeliefTracker*> per_perspective_trackers = {},
    // Required per-seat session state (size == num_players, no nullptrs).
    // MCTS/encoder/heuristic/legal_actions read from per_seat_states[player]
    // — never from truth. Each seat's session state is advanced via the
    // public-event protocol after every truth do_action_fast: hidden-info
    // games run public_state_applier(seat, snapshot) +
    // tracker.observe_public_event(events); fully-public games (no
    // public_event_extractor registered) re-run do_action_fast(seat) on
    // each seat with its own step rng. The session's viz=0 slots are
    // never freshened — decision-side reads (hash kHiddenHashSentinel,
    // encoder MaskedState placeholder, sim_tracker->randomize_unseen at
    // sim entry) make them structurally unreachable.
    std::vector<IGameState*> per_seat_states = {},
    PublicStateApplier public_state_applier = nullptr,
    const IFeatureEncoder* encoder = nullptr,
    const search::ITailSolver* tail_solver = nullptr,
    GameAdjudicator adjudicator = nullptr,
    AuxiliaryScorer auxiliary_scorer = nullptr,
    HeuristicPicker heuristic_picker = nullptr,
    TrainingActionFilter training_action_filter = nullptr,
    TailSolveTrigger tail_solve_trigger = nullptr,
    EpisodeStatsExtractor episode_stats_extractor = nullptr,
    // Tracing hooks for AI API belief-equivalence tests. trace_perspective
    // >= 0 enables recording; -1 disables (default, zero overhead).
    // trace_belief_tracker must be a SEPARATE tracker instance dedicated to
    // the trace_perspective, kept decoupled from the per_perspective_trackers
    // so tracing output stays reproducible across refactors of the MCTS
    // routing. The caller is responsible for creating this instance via the
    // game's factory.
    int trace_perspective = -1,
    IBeliefTracker* trace_belief_tracker = nullptr,
    PublicEventExtractor public_event_extractor = nullptr);

}  // namespace board_ai::runtime
