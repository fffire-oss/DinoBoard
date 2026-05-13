#include "selfplay_runner.h"

#include <cassert>
#include <cmath>
#include <numeric>
#include <random>

#include "../core/rng_salt.h"

namespace board_ai::runtime {

SelfplayEpisodeResult run_selfplay_episode(
    IGameState& initial_state,
    const IGameRules& rules,
    const IStateValueModel& value_model,
    const search::IPolicyValueEvaluator& evaluator,
    const SelfplayConfig& config,
    std::uint64_t episode_seed,
    std::vector<IBeliefTracker*> per_perspective_trackers,
    std::vector<IGameState*> per_seat_states,
    PublicStateApplier public_state_applier,
    const IFeatureEncoder* encoder,
    const search::ITailSolver* tail_solver,
    GameAdjudicator adjudicator,
    AuxiliaryScorer auxiliary_scorer,
    HeuristicPicker heuristic_picker,
    TrainingActionFilter training_action_filter,
    TailSolveTrigger tail_solve_trigger,
    EpisodeStatsExtractor episode_stats_extractor,
    int trace_perspective,
    IBeliefTracker* trace_belief_tracker,
    PublicEventExtractor public_event_extractor,
    InitialObservationExtractor initial_observation_extractor) {
  SelfplayEpisodeResult result{};

  if (config.tail_solve_enabled && (!tail_solver || !tail_solve_trigger)) {
    throw std::invalid_argument(
        "run_selfplay_episode: tail_solve_enabled=true requires both a "
        "registered ITailSolver and a TailSolveTrigger; one or both are missing.");
  }

  const bool tracing =
      trace_perspective >= 0 && public_event_extractor && trace_belief_tracker;
  if (tracing) {
    result.trace_enabled = true;
    result.trace_perspective = trace_perspective;
  }

  std::unique_ptr<FilteredRulesWrapper> filtered_rules_ptr;
  if (training_action_filter) {
    filtered_rules_ptr = std::make_unique<FilteredRulesWrapper>(rules, training_action_filter);
  }

  auto state = initial_state.clone_state();
  int ply = 0;

  // Per-seat session state is mandatory: MCTS / encoder / heuristic /
  // legal-action queries on the AI path read from that seat's session
  // state, never from truth. The session state is advanced via the
  // public-event protocol every ply (mirrors py_engine::advance_ai_view_).
  const int num_players = state->num_players();
  if (static_cast<int>(per_seat_states.size()) != num_players) {
    throw std::runtime_error(
        "run_selfplay_episode: per_seat_states size != num_players");
  }
  for (int p = 0; p < num_players; ++p) {
    if (per_seat_states[p] == nullptr) {
      throw std::runtime_error(
          "run_selfplay_episode: per_seat_states[p] must not be null");
    }
  }
  auto ai_view_for = [&](int seat) -> IGameState& {
    return *per_seat_states[seat];
  };

  // Init each seat's tracker once at episode start. Every public event is
  // fed to every tracker in the main loop below; each tracker's belief
  // accumulates monotonically until game over.
  const bool use_per_perspective = !per_perspective_trackers.empty();
  if (use_per_perspective) {
    const int num_players = state->num_players();
    if (static_cast<int>(per_perspective_trackers.size()) != num_players) {
      throw std::runtime_error(
          "run_selfplay_episode: per_perspective_trackers size != num_players");
    }
    for (int p = 0; p < num_players; ++p) {
      if (per_perspective_trackers[p] && initial_observation_extractor) {
        AnyMap p_init_obs = initial_observation_extractor(*state, p);
        // Pre-§G.2 transitional: stash perspective in init obs so trackers
        // that still need own_self vs opp seat can read it. Goes away once
        // Coup migrates per-perspective knowledge to state.viz.
        p_init_obs["__perspective_player"] = p;
        per_perspective_trackers[p]->init(p_init_obs);
      }
    }
  }

  // No episode-start freshen of per-seat states. Decision-side reads
  // (encoder via MaskedState placeholder, hash via kHiddenHashSentinel,
  // MCTS sims via sim_tracker->randomize_unseen at sim entry) are
  // perspective-aware by construction; truth values that linger in
  // per_seat_states' viz=0 slots are structurally unreachable. A future
  // direct-field read of opp-private would be a bug to catch in tests,
  // not silently paper over with a session-side resample.

  // Tracing uses its own separate tracker instance so trace output stays
  // reproducible across refactors of MCTS tracker routing.
  if (tracing) {
    AnyMap trace_init_obs;
    if (initial_observation_extractor) {
      trace_init_obs = initial_observation_extractor(*state, trace_perspective);
    }
    trace_init_obs["__perspective_player"] = trace_perspective;
    trace_belief_tracker->init(trace_init_obs);
    result.initial_belief_snapshot = trace_belief_tracker->serialize();
    result.initial_observation = trace_init_obs;
  }

  std::mt19937_64 heuristic_rng(
      board_ai::rng::derive_subseed(episode_seed, "selfplay.heuristic_pick"));
  std::uniform_real_distribution<double> heuristic_dist(0.0, 1.0);
  int heuristic_moves = 0;

  // GT step rng for do_action_fast on truth state. Seeded directly from
  // episode_seed (no subseed derivation) so that any other GT runner —
  // notably the API session's `step_rng_` in py_engine.cpp — receiving
  // the same episode seed advances truth through identical draws. The
  // round-trip test `test_public_snapshot_round_trip` depends on this
  // equivalence (it replays selfplay's trace via GameSession.apply_action
  // and asserts public state matches the trace's snapshot).
  std::mt19937_64 step_rng(episode_seed);

  // Per-seat session-state advance: mirrors py_engine::advance_ai_view_.
  // Called after truth has been advanced by `chosen`, with the truth state
  // before/after both available. For each seat:
  //   - hidden-info game (public_event_extractor registered): public_snapshot
  //     overwrites the seat's public fields. Per DEC-003, viz=0 hidden
  //     slots are NOT freshened — they are unread bytes that the framework
  //     structurally hides from the hash, encoder, and sim entry.
  //     do_action_fast is NOT run on the seat — the observer has incomplete
  //     information so replaying the action there would just sample one
  //     world; the public projection arrives whole via the snapshot, and
  //     the eventual decision will be made off whatever world MCTS samples.
  //   - fully-public game (no extractor): the seat just replays do_action_fast
  //     deterministically.
  auto advance_per_seat_states =
      [&](const IGameState& truth_before, const IGameState& truth_after,
          ActionId chosen, int /*actor*/) {
    for (int p = 0; p < num_players; ++p) {
      IGameState& seat = *per_seat_states[p];
      if (!public_event_extractor) {
        // Fully-public game: just replay the action on the seat state.
        const std::uint64_t view_step_seed = board_ai::rng::derive_subseed(
            episode_seed, "selfplay.view_step",
            static_cast<std::uint64_t>(ply) * 17ULL +
                static_cast<std::uint64_t>(p));
        std::mt19937_64 view_step_rng(view_step_seed);
        rules.do_action_fast(seat, chosen, view_step_rng);
        continue;
      }
      PublicEventTrace evt_p = public_event_extractor(
          truth_before, chosen, truth_after, p);
      seat.begin_step_for_session_observe();
      if (public_state_applier && !evt_p.public_snapshot.empty()) {
        public_state_applier(seat, evt_p.public_snapshot);
      }
      // tracker.observe_public_event is called in the per_perspective loop
      // immediately below. No hidden-slot freshening — sim entry will
      // sample fresh worlds on a cloned sim_tracker (DEC-003).
    }
  };

  while (!state->is_terminal() && ply < config.max_game_plies) {
    const int player = state->current_player();

    const bool use_filter = filtered_rules_ptr &&
        config.training_filter_ratio > 0.0 &&
        heuristic_dist(heuristic_rng) < config.training_filter_ratio;
    const IGameRules& effective_rules = use_filter ? *filtered_rules_ptr : rules;

    // AI-path reads come from the acting seat's session state (truth-free)
    // when per-seat mode is on; otherwise fall back to truth (legacy path
    // for games whose tracker is still perspective-baked).
    IGameState& ai_view = ai_view_for(player);

    const auto legal = effective_rules.legal_actions(ai_view);
    if (legal.empty()) break;
    const std::vector<ActionId> full_legal_storage =
        use_filter ? rules.legal_actions(ai_view) : std::vector<ActionId>{};
    const std::vector<ActionId>& full_legal =
        use_filter ? full_legal_storage : legal;

    const bool use_heuristic = heuristic_picker &&
        config.heuristic_guidance_ratio > 0.0 &&
        heuristic_dist(heuristic_rng) < config.heuristic_guidance_ratio;

    if (use_heuristic) {
      const std::uint64_t hr_seed = board_ai::rng::derive_subseed(
          episode_seed, "selfplay.heuristic", static_cast<std::uint64_t>(ply));
      auto hr = heuristic_picker(ai_view, effective_rules, hr_seed);

      const size_t n_actions = hr.actions.size();
      std::vector<double> probs(n_actions);
      double max_score = *std::max_element(hr.scores.begin(), hr.scores.end());

      if (config.heuristic_temperature > 1e-6) {
        double inv_temp = 1.0 / config.heuristic_temperature;
        double sum_exp = 0.0;
        for (size_t i = 0; i < n_actions; ++i) {
          probs[i] = std::exp((hr.scores[i] - max_score) * inv_temp);
          sum_exp += probs[i];
        }
        for (auto& p : probs) p /= sum_exp;
      } else {
        int count = 0;
        for (size_t i = 0; i < n_actions; ++i) {
          if (hr.scores[i] >= max_score - 1e-9) ++count;
        }
        for (size_t i = 0; i < n_actions; ++i) {
          probs[i] = (hr.scores[i] >= max_score - 1e-9) ? 1.0 / count : 0.0;
        }
      }

      // Convert to fake visits for policy target
      std::vector<int> fake_visits(n_actions);
      for (size_t i = 0; i < n_actions; ++i) {
        fake_visits[i] = static_cast<int>(probs[i] * 10000.0 + 0.5);
      }

      // Sample action from distribution
      const std::uint64_t h = board_ai::rng::derive_subseed(
          hr_seed, "selfplay.heuristic_sample");
      double r = static_cast<double>(h & 0xFFFFFFFF) / 4294967296.0;
      size_t chosen_idx = n_actions - 1;
      double cumulative = 0.0;
      for (size_t i = 0; i < n_actions; ++i) {
        cumulative += probs[i];
        if (r <= cumulative) { chosen_idx = i; break; }
      }
      ActionId chosen = hr.actions[chosen_idx];

      SelfplaySample sample{};
      sample.ply = ply;
      sample.player = player;
      sample.action_id = chosen;
      sample.policy_action_ids = hr.actions;
      if (encoder) {
        const IBeliefTracker* enc_tracker =
            (use_per_perspective && player >= 0 &&
             player < static_cast<int>(per_perspective_trackers.size()))
                ? per_perspective_trackers[player]
                : nullptr;
        encoder->encode(ai_view, player, enc_tracker, full_legal, &sample.features, &sample.legal_mask);
      }
      sample.policy_action_visits = fake_visits;
      if (auxiliary_scorer) {
        sample.auxiliary_score = auxiliary_scorer(ai_view, player);
        sample.has_auxiliary_score = true;
      }
      result.samples.push_back(std::move(sample));

      std::unique_ptr<IGameState> state_before = state->clone_state();
      effective_rules.do_action_fast(*state, chosen, step_rng);
      if (use_per_perspective) {
        const int num_players = static_cast<int>(per_perspective_trackers.size());
        for (int p = 0; p < num_players; ++p) {
          if (!per_perspective_trackers[p]) continue;
          PublicEventTrace evt_p;
          if (public_event_extractor) {
            evt_p = public_event_extractor(*state_before, chosen, *state, p);
          }
          per_perspective_trackers[p]->observe_public_event(
              player, chosen, evt_p.events);
        }
      }
      // Advance per-seat session states via the public-event protocol so
      // the AI path's view of public state stays in sync with truth's
      // public projection while hidden remains a tracker-consistent sample.
      advance_per_seat_states(*state_before, *state, chosen, player);
      if (tracing) {
        SelfplayObservationTrace t{};
        t.ply = ply;
        t.actor = player;
        t.action = chosen;
        auto evt = public_event_extractor(
            *state_before, chosen, *state, trace_perspective);
        t.events = evt.events;
        t.public_snapshot = evt.public_snapshot;
        trace_belief_tracker->observe_public_event(
            player, chosen, evt.events);
        t.belief_snapshot_after = trace_belief_tracker->serialize();
        result.observation_trace.push_back(std::move(t));
      }
      ply += 1;
      heuristic_moves += 1;
      continue;
    }

    // MCTS root uses the acting player's per-perspective tracker — it has
    // accumulated the full observation history for that seat since game
    // start, giving ISMCTS an accurate belief to sample from.
    IBeliefTracker* mcts_tracker = nullptr;
    if (use_per_perspective && player >= 0 &&
        player < static_cast<int>(per_perspective_trackers.size())) {
      mcts_tracker = per_perspective_trackers[player];
    }

    const auto noise = search::resolve_root_dirichlet_noise(
        config.dirichlet_alpha, config.dirichlet_epsilon,
        config.dirichlet_on_first_n_plies, ply);

    const bool try_tail_solve = config.tail_solve_enabled && tail_solver &&
        tail_solve_trigger && tail_solve_trigger(ai_view, ply);

    search::NetMctsConfig mcts_cfg{};
    mcts_cfg.simulations = config.simulations;
    mcts_cfg.c_puct = config.c_puct;
    mcts_cfg.max_depth = config.max_depth;
    mcts_cfg.value_clip = config.value_clip;
    mcts_cfg.root_dirichlet_alpha = noise.alpha;
    mcts_cfg.root_dirichlet_epsilon = noise.epsilon;
    mcts_cfg.opponent_selection = config.opponent_selection;
    // ISMCTS: root-sampling hidden info + DAG per-acting-player keying.
    // MCTS uses the per-sim sampled world's rules.legal_actions at each node.
    if (mcts_tracker) {
      mcts_cfg.root_belief_tracker = mcts_tracker;
    }

    if (try_tail_solve) {
      mcts_cfg.tail_solve_enabled = true;
      mcts_cfg.tail_solve_config = config.tail_solve_config;
      if (auxiliary_scorer && config.tail_solve_config.margin_weight != 0.0f &&
          !config.tail_solve_config.margin_scorer) {
        mcts_cfg.tail_solve_config.margin_scorer = auxiliary_scorer;
      }
      mcts_cfg.tail_solver = tail_solver;
    }

    search::NetMcts mcts(mcts_cfg);
    search::NetMctsStats stats{};
    const std::uint64_t mcts_seed = board_ai::rng::derive_subseed(
        episode_seed, "selfplay.mcts", static_cast<std::uint64_t>(ply));
    mcts.search_root(ai_view, effective_rules, value_model, evaluator, &stats, mcts_seed);

    if (stats.tail_solve_attempted) {
      result.tail_solve_attempts += 1;
      result.tail_solve_total_ms += stats.tail_solve_elapsed_ms;
      if (stats.tail_solve_completed) {
        result.tail_solve_completed += 1;
      }
      if (stats.tail_solved) {
        result.tail_solve_successes += 1;
      }
    }

    SelfplaySample sample{};
    sample.ply = ply;
    sample.player = player;
    sample.policy_action_ids = stats.root_actions;
    sample.policy_action_visits = stats.root_action_visits;
    sample.tail_solved = stats.tail_solved;
    if (encoder) {
      const IBeliefTracker* enc_tracker =
          (use_per_perspective && player >= 0 &&
           player < static_cast<int>(per_perspective_trackers.size()))
              ? per_perspective_trackers[player]
              : nullptr;
      encoder->encode(ai_view, player, enc_tracker, full_legal, &sample.features, &sample.legal_mask);
    }
    if (auxiliary_scorer) {
      sample.auxiliary_score = auxiliary_scorer(ai_view, player);
      sample.has_auxiliary_score = true;
    }

    const double temperature = stats.tail_solved ? 0.0 :
        search::resolve_linear_temperature(config.temperature_schedule, config.temperature, ply);
    const std::uint64_t action_seed = board_ai::rng::derive_subseed(
        episode_seed, "selfplay.action_pick", static_cast<std::uint64_t>(ply));
    const ActionId chosen = search::select_action_from_visits(
        stats.root_actions, stats.root_action_visits, temperature, action_seed, legal[0]);

    sample.action_id = chosen;
    result.samples.push_back(std::move(sample));

    std::unique_ptr<IGameState> state_before = state->clone_state();
    effective_rules.do_action_fast(*state, chosen, step_rng);
    if (use_per_perspective) {
      // Every seat's tracker sees every action. Each perspective extracts
      // its own perspective-specific events.
      const int num_players = static_cast<int>(per_perspective_trackers.size());
      for (int p = 0; p < num_players; ++p) {
        if (!per_perspective_trackers[p]) continue;
        PublicEventTrace evt_p;
        if (public_event_extractor) {
          evt_p = public_event_extractor(*state_before, chosen, *state, p);
        }
        per_perspective_trackers[p]->observe_public_event(
            player, chosen, evt_p.events);
      }
    }
    advance_per_seat_states(*state_before, *state, chosen, player);
    if (tracing) {
      SelfplayObservationTrace t{};
      t.ply = ply;
      t.actor = player;
      t.action = chosen;
      auto evt = public_event_extractor(
          *state_before, chosen, *state, trace_perspective);
      t.events = evt.events;
      t.public_snapshot = evt.public_snapshot;
      trace_belief_tracker->observe_public_event(
          player, chosen, evt.events);
      t.belief_snapshot_after = trace_belief_tracker->serialize();
      result.observation_trace.push_back(std::move(t));
    }
    ply += 1;
  }

  result.total_plies = ply;
  result.final_state = state->clone_state();

  if (state->is_terminal()) {
    const int num_players = state->num_players();
    float best_val = -2.0f;
    int best_player = -1;
    bool is_draw = true;
    for (int p = 0; p < num_players; ++p) {
      const float v = value_model.terminal_value_for_player(*state, p);
      if (v > best_val) {
        best_val = v;
        best_player = p;
      }
    }
    for (int p = 0; p < num_players; ++p) {
      const float v = value_model.terminal_value_for_player(*state, p);
      if (std::abs(v - best_val) > 1e-6f) {
        is_draw = false;
        break;
      }
    }
    if (is_draw) {
      result.draw = true;
      result.winner = -1;
    } else {
      result.draw = false;
      result.winner = best_player;
    }

    const auto terminal_vals = value_model.terminal_values(*state);
    assert(std::abs(std::accumulate(terminal_vals.begin(), terminal_vals.end(), 0.0f)) < 1e-4f);
    for (auto& s : result.samples) {
      s.z_values = terminal_vals;
      if (result.draw) {
        s.z = 0.0f;
      } else {
        s.z = (s.player == result.winner) ? 1.0f : -1.0f;
      }
    }
  } else if (adjudicator) {
    const int adj_winner = adjudicator(*state);
    if (adj_winner < 0) {
      result.draw = true;
      result.winner = -1;
    } else {
      result.draw = false;
      result.winner = adj_winner;
    }
    const int num_players = state->num_players();
    std::vector<float> adj_vals(num_players, 0.0f);
    if (!result.draw) {
      const float loser_val = -1.0f / static_cast<float>(num_players - 1);
      for (int p = 0; p < num_players; ++p) {
        adj_vals[p] = (p == result.winner) ? 1.0f : loser_val;
      }
    }
    assert(std::abs(std::accumulate(adj_vals.begin(), adj_vals.end(), 0.0f)) < 1e-4f);
    for (auto& s : result.samples) {
      s.z_values = adj_vals;
      if (result.draw) {
        s.z = 0.0f;
      } else {
        s.z = (s.player == result.winner) ? 1.0f : -1.0f;
      }
    }
  } else {
    // Game didn't finish and no adjudicator — treat as draw (all zeros).
    const int np = state->num_players();
    std::vector<float> zero_vals(np, 0.0f);
    for (auto& s : result.samples) {
      s.z_values = zero_vals;
      s.z = 0.0f;
    }
    result.draw = true;
    result.winner = -1;
  }

  if (episode_stats_extractor) {
    std::vector<SelfplaySampleView> views;
    views.reserve(result.samples.size());
    for (const auto& s : result.samples) {
      views.push_back({s.ply, s.player, s.action_id});
    }
    result.custom_stats = episode_stats_extractor(*state, views);
  }

  return result;
}

}  // namespace board_ai::runtime
