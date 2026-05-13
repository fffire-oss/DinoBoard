#pragma once

#include <cstdint>
#include <functional>
#include <memory>

#include "../core/game_interfaces.h"
#include "../search/net_mcts.h"
#include "selfplay_runner.h"

namespace board_ai::runtime {

struct ArenaPlayerConfig {
  int simulations = 200;
  float c_puct = 1.4f;
  int max_depth = 128;
  float value_clip = 1.0f;
  double temperature = 0.0;
  bool tail_solve_enabled = false;
  search::TailSolveConfig tail_solve_config{};
  const search::ITailSolver* tail_solver = nullptr;
  TailSolveTrigger tail_solve_trigger = nullptr;
  // See OpponentSelection in net_mcts.h. Per-player so candidate and
  // opponent can be independently configured (A/B testing).
  search::OpponentSelection opponent_selection = search::OpponentSelection::kPuct;
};

struct ArenaPlyStats {
  bool tail_solved = false;
  float tail_solve_value = 0.0f;
};

struct ArenaMatchResult {
  int winner = -1;
  bool draw = false;
  int total_plies = 0;
  std::vector<ActionId> action_history;
  std::vector<ArenaPlyStats> ply_stats;
};

using PolicyEvaluatorFactory = std::function<
    const search::IPolicyValueEvaluator&(int player_index)>;

ArenaMatchResult run_arena_match(
    IGameState& initial_state,
    const IGameRules& rules,
    const IStateValueModel& value_model,
    PolicyEvaluatorFactory evaluator_for_player,
    const std::vector<ArenaPlayerConfig>& player_configs,
    int max_game_plies = 500,
    std::uint64_t match_seed = 0,
    IBeliefTracker* belief_tracker = nullptr,
    GameAdjudicator adjudicator = nullptr,
    // Tracker input adapters: needed when belief_tracker is set. The
    // tracker receives events from public_event_extractor and its init
    // input from initial_observation_extractor — state reads never pass
    // through the tracker interface.
    PublicEventExtractor public_event_extractor = nullptr,
    InitialObservationExtractor initial_observation_extractor = nullptr,
    // Per-seat trackers for hidden-info games. Size == num_players for
    // games with belief_tracker registered; empty for fully-public games.
    // When non-empty, each seat's tracker is init'd once at match start
    // and accumulates the full observation history exactly like selfplay.
    // MCTS root for the acting player reads from
    // per_perspective_trackers[player].
    std::vector<IBeliefTracker*> per_perspective_trackers = {},
    // Required per-seat session state (size == num_players, no nullptrs).
    // MCTS / encoder / legal_actions read from per_seat_states[player] —
    // never from truth. Advanced after every truth do_action_fast via the
    // public-event protocol: hidden-info games run public_state_applier(
    // seat, snapshot) + tracker.observe_public_event(events); fully-public
    // games re-run do_action_fast on each seat. viz=0 slots are never
    // freshened — see selfplay_runner.h for the structural-unreachability
    // argument.
    std::vector<IGameState*> per_seat_states = {},
    PublicStateApplier public_state_applier = nullptr);

}  // namespace board_ai::runtime
