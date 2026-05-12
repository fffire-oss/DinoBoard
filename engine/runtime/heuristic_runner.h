#pragma once

#include <cstdint>
#include <cstddef>

#include "../core/game_registry.h"
#include "selfplay_runner.h"

namespace board_ai::runtime {

// Select an index from heuristic scores using softmax-with-temperature sampling.
// temperature <= 1e-6 means greedy argmax (first max on ties); >0 means
// exp(score/T) softmax then cumulative sample with rng_u01 in [0,1).
// Shared by selfplay (run_heuristic_episode) and eval benchmarks so both
// faces of the heuristic use the same selection rule.
std::size_t sample_heuristic_index(
    const std::vector<double>& scores,
    double temperature,
    double rng_u01);

SelfplayEpisodeResult run_heuristic_episode(
    IGameState& initial_state,
    const IGameRules& rules,
    const IStateValueModel& value_model,
    const IFeatureEncoder* encoder,
    const HeuristicPicker& heuristic,
    double temperature,
    int max_game_plies,
    std::uint64_t episode_seed,
    AuxiliaryScorer auxiliary_scorer = nullptr,
    GameAdjudicator adjudicator = nullptr,
    // Per-seat session-state mode (mirrors selfplay/arena). When non-empty
    // (size == num_players), the heuristic / encoder / legal-action
    // queries on the AI path read from per_seat_states[player] instead of
    // truth. Each seat's session state is advanced via the public-event
    // protocol after each truth do_action_fast (public_state_applier from
    // snapshot, tracker.observe_public_event(events), randomize_unseen).
    // Empty vector: AI path reads truth (legacy fallback).
    std::vector<IBeliefTracker*> per_perspective_trackers = {},
    std::vector<IGameState*> per_seat_states = {},
    PublicStateApplier public_state_applier = nullptr,
    PublicEventExtractor public_event_extractor = nullptr,
    InitialObservationExtractor initial_observation_extractor = nullptr);

}  // namespace board_ai::runtime
