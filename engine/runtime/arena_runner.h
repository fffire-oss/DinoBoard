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
    // Per-seat trackers for in-scope hidden-info games. When non-empty
    // (size == num_players), each seat's tracker is init'd once at match
    // start and accumulates the full observation history, exactly like
    // selfplay. MCTS root for the acting player reads from
    // per_perspective_trackers[player]. Empty vector falls back to the
    // legacy `belief_tracker` path for games not yet on per-seat.
    std::vector<IBeliefTracker*> per_perspective_trackers = {},
    // Optional per-seat session state. When non-empty (size ==
    // num_players), MCTS / encoder / legal_actions read from
    // per_seat_states[player] instead of truth. Each seat's state is
    // advanced via the public-event protocol (public_state_applier
    // overwrites public from snapshot, tracker.observe_public_event
    // updates belief, tracker.randomize_unseen freshens hidden) — never
    // copied from truth, never runs do_action_fast on the session.
    // Empty vector: AI path reads truth (legacy fallback).
    std::vector<IGameState*> per_seat_states = {},
    PublicStateApplier public_state_applier = nullptr);

}  // namespace board_ai::runtime
