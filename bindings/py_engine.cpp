#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <chrono>
#include <cstdint>
#include <memory>
#include <random>
#include <string>
#include <unordered_set>
#include <vector>

#include "../engine/core/game_interfaces.h"
#include "../engine/core/rng_salt.h"
#include "../engine/core/schema_hash.h"
#include "../engine/core/game_registry.h"
#include "../engine/core/feature_encoder.h"
#include "../engine/core/masked_state.h"
#include "../engine/core/snapshot_io.h"
#include "../engine/infer/onnx_belief_evaluator.h"
#include "../engine/infer/onnx_policy_value_evaluator.h"
#include "../engine/runtime/selfplay_runner.h"
#include "../engine/runtime/arena_runner.h"
#include "../engine/runtime/heuristic_runner.h"
#include "../engine/search/net_mcts.h"
#include "../engine/search/root_noise.h"
#include "../engine/search/temperature_schedule.h"

namespace py = pybind11;

using namespace board_ai;

namespace {

// Parse the user-facing string into the OpponentSelection enum. Unknown
// values raise — never silently fall back to puct (no_silent_degradation).
inline search::OpponentSelection parse_opponent_selection(
    const std::string& s) {
  if (s == "puct") return search::OpponentSelection::kPuct;
  if (s == "prior" || s == "frozen_prior") {
    return search::OpponentSelection::kFrozenPrior;
  }
  throw std::invalid_argument(
      "opponent_selection: unknown value '" + s +
      "' (expected 'puct' or 'prior')");
}

// Tracker adapter helpers: convert the old (state_before, action, state_after)
// trio into the tracker's new event-only input. Games with a registered
// public_event_extractor produce the event stream; games without one pass
// empty vectors (tracker becomes effectively a no-op for those games, which
// matches their pre-migration behavior since they had no hidden info).
inline void tracker_init(IBeliefTracker& bt, const GameBundle& /*bundle*/,
                         IGameState& state, int perspective,
                         const AnyMap& payload = {}) {
  // Two-step init: walker `apply_public_snapshot` (the broadcast
  // public snapshot path) is the responsibility of the caller; here
  // we just hand the
  // perspective-private bootstrap payload to the tracker. The tracker
  // writes any private slots into `state` and seeds its own memory by
  // reading public facts from `state`.
  bt.init(state, perspective, payload);
}

inline void tracker_observe(IBeliefTracker& bt, const GameBundle& bundle,
                            const IGameState& before, ActionId action,
                            const IGameState& after, int perspective) {
  PublicEventTrace trace;
  if (bundle.public_event_extractor) {
    trace = bundle.public_event_extractor(before, action, after, perspective);
  }
  const int actor = before.current_player();
  bt.observe_public_event(actor, action, trace.events);
}

std::any py_to_any(const py::handle& obj);

AnyMap py_dict_to_any_map(const py::dict& d) {
  AnyMap m;
  for (const auto& [k, v] : d) {
    const std::string key = py::cast<std::string>(k);
    m[key] = py_to_any(v);
  }
  return m;
}

std::any py_to_any(const py::handle& obj) {
  if (py::isinstance<py::bool_>(obj)) return py::cast<bool>(obj);
  if (py::isinstance<py::int_>(obj))  return py::cast<int>(obj);
  if (py::isinstance<py::float_>(obj)) return py::cast<double>(obj);
  if (py::isinstance<py::str>(obj))   return py::cast<std::string>(obj);
  if (py::isinstance<py::dict>(obj)) {
    return py_dict_to_any_map(py::reinterpret_borrow<py::dict>(obj));
  }
  if (py::isinstance<py::list>(obj) || py::isinstance<py::tuple>(obj)) {
    // Inspect first element to decide vector<int> vs vector<any>.
    py::sequence seq = py::reinterpret_borrow<py::sequence>(obj);
    bool all_int = !seq.empty() ? py::isinstance<py::int_>(seq[0]) : false;
    if (all_int) {
      for (const auto& item : seq) {
        if (!py::isinstance<py::int_>(item) || py::isinstance<py::bool_>(item)) {
          all_int = false;
          break;
        }
      }
    }
    if (all_int) {
      std::vector<int> out;
      out.reserve(seq.size());
      for (const auto& item : seq) out.push_back(py::cast<int>(item));
      return out;
    }
    std::vector<std::any> out;
    out.reserve(seq.size());
    for (const auto& item : seq) out.push_back(py_to_any(item));
    return out;
  }
  throw std::runtime_error("py_to_any: unsupported Python type");
}

py::object any_to_py(const std::any& val) {
  if (val.type() == typeid(int))
    return py::cast(std::any_cast<int>(val));
  if (val.type() == typeid(float))
    return py::cast(std::any_cast<float>(val));
  if (val.type() == typeid(double))
    return py::cast(std::any_cast<double>(val));
  if (val.type() == typeid(bool))
    return py::cast(std::any_cast<bool>(val));
  if (val.type() == typeid(std::string))
    return py::cast(std::any_cast<std::string>(val));
  if (val.type() == typeid(std::vector<int>))
    return py::cast(std::any_cast<std::vector<int>>(val));
  if (val.type() == typeid(std::vector<std::vector<int>>)) {
    py::list lst;
    for (const auto& inner :
         std::any_cast<std::vector<std::vector<int>>>(val)) {
      lst.append(py::cast(inner));
    }
    return lst;
  }
  if (val.type() == typeid(std::vector<std::any>)) {
    py::list lst;
    for (const auto& item : std::any_cast<std::vector<std::any>>(val))
      lst.append(any_to_py(item));
    return lst;
  }
  if (val.type() == typeid(std::vector<AnyMap>)) {
    py::list lst;
    for (const auto& item : std::any_cast<std::vector<AnyMap>>(val)) {
      py::dict d;
      for (const auto& [k, v] : item) d[py::cast(k)] = any_to_py(v);
      lst.append(d);
    }
    return lst;
  }
  if (val.type() == typeid(AnyMap)) {
    py::dict d;
    for (const auto& [k, v] : std::any_cast<AnyMap>(val))
      d[py::cast(k)] = any_to_py(v);
    return d;
  }
  throw std::runtime_error(
      std::string("any_to_py: unsupported type: ") + val.type().name());
}

py::dict sample_to_py(const runtime::SelfplaySample& s) {
  py::dict d;
  d["ply"] = s.ply;
  d["player"] = s.player;
  d["action_id"] = s.action_id;
  d["z"] = s.z;
  d["z_values"] = s.z_values;
  d["policy_action_ids"] = s.policy_action_ids;
  d["policy_action_visits"] = s.policy_action_visits;
  d["features"] = s.features;
  d["legal_mask"] = s.legal_mask;
  d["tail_solved"] = s.tail_solved;
  d["auxiliary_score"] = s.auxiliary_score;
  return d;
}

py::dict belief_sample_to_py(const runtime::BeliefSample& s) {
  py::dict d;
  d["ply"] = s.ply;
  d["observer"] = s.observer;
  d["features"] = s.features;
  py::list hand_counts;
  for (const auto& row : s.hand_counts) hand_counts.append(py::cast(row));
  d["hand_counts"] = hand_counts;
  d["remaining"] = s.remaining;
  d["alive_per_opp"] = s.alive_per_opp;
  return d;
}

py::dict result_to_py(const runtime::SelfplayEpisodeResult& result) {
  py::list samples;
  for (const auto& s : result.samples)
    samples.append(sample_to_py(s));

  py::dict out;
  out["winner"] = result.winner;
  out["draw"] = result.draw;
  out["total_plies"] = result.total_plies;
  out["samples"] = samples;
  py::list belief_samples;
  for (const auto& bs : result.belief_samples)
    belief_samples.append(belief_sample_to_py(bs));
  out["belief_samples"] = belief_samples;
  out["belief_label_class_count"] = result.belief_label_class_count;
  out["tail_solve_attempts"] = result.tail_solve_attempts;
  out["tail_solve_completed"] = result.tail_solve_completed;
  out["tail_solve_successes"] = result.tail_solve_successes;
  out["tail_solve_total_ms"] = result.tail_solve_total_ms;

  if (!result.custom_stats.empty()) {
    py::dict stats;
    for (const auto& [k, v] : result.custom_stats) stats[py::cast(k)] = v;
    out["custom_stats"] = stats;
  }

  if (result.trace_enabled) {
    out["trace_perspective"] = result.trace_perspective;
    // initial_observation: game-defined AnyMap
    py::dict init_obs;
    for (const auto& [k, v] : result.initial_observation) init_obs[py::cast(k)] = any_to_py(v);
    out["initial_observation"] = init_obs;
    // initial_belief_snapshot: tracker's serialize() output
    py::dict init_bs;
    for (const auto& [k, v] : result.initial_belief_snapshot) init_bs[py::cast(k)] = any_to_py(v);
    out["initial_belief_snapshot"] = init_bs;
    // observation_trace: list of per-ply records
    py::list trace_list;
    for (const auto& t : result.observation_trace) {
      py::dict entry;
      entry["ply"] = t.ply;
      entry["actor"] = t.actor;
      entry["action"] = static_cast<int>(t.action);
      py::list ev_list;
      for (const auto& [kind, payload] : t.events) {
        py::dict e;
        e["kind"] = kind;
        py::dict p;
        for (const auto& [pk, pv] : payload) p[py::cast(pk)] = any_to_py(pv);
        e["payload"] = p;
        ev_list.append(e);
      }
      entry["events"] = ev_list;
      // Truth-side public snapshot. Empty for games without
      // public_state_applier registered.
      py::dict snap;
      for (const auto& [k, v] : t.public_snapshot) snap[py::cast(k)] = any_to_py(v);
      entry["public_snapshot"] = snap;
      py::dict bs;
      for (const auto& [k, v] : t.belief_snapshot_after) bs[py::cast(k)] = any_to_py(v);
      entry["belief_snapshot_after"] = bs;
      trace_list.append(entry);
    }
    out["observation_trace"] = trace_list;
  }
  return out;
}

py::dict run_selfplay_episode_py(
    const std::string& game_id,
    std::uint64_t seed,
    const std::string& model_path,
    int simulations,
    float c_puct,
    double temperature,
    double dirichlet_alpha,
    double dirichlet_epsilon,
    int dirichlet_on_first_n_plies,
    int max_game_plies,
    bool tail_solve_enabled,
    int tail_solve_depth_limit,
    std::int64_t tail_solve_node_budget,
    float tail_solve_margin_weight,
    double temperature_initial,
    double temperature_final,
    int temperature_decay_plies,
    double heuristic_guidance_ratio,
    double heuristic_temperature,
    double training_filter_ratio,
    int trace_perspective,
    const std::string& opponent_selection,
    const std::string& belief_model_path) {
  // Validate before releasing the GIL — exception propagation is cleaner.
  const auto opp_sel = parse_opponent_selection(opponent_selection);
  py::gil_scoped_release release;

  auto bundle = GameRegistry::instance().create_game(game_id, seed);
  if (!belief_model_path.empty()) bundle.belief_model_path = belief_model_path;
  // Public-event extractor feeds two consumers:
  //   - per-perspective trackers (pp_trackers below) — observe events
  //     each ply for every seat.
  //   - trace_belief_tracker — records a specific perspective's
  //     snapshot for regression tests.
  // Must be populated whenever the game registers it, regardless of
  // trace_perspective. The tracker bootstrap snapshot is now produced
  // by the framework walker (make_masked_state) — no per-game extractor.
  runtime::PublicEventExtractor trace_extractor = bundle.public_event_extractor;
  // For tracing we need a SECOND bundle (and its belief_tracker) dedicated
  // to the traced perspective, separate from both the main tracker and the
  // pp_trackers.
  // Tracing only requires public_event_extractor — that's the source
  // of the per-ply event stream + public_snapshot. trace_belief_tracker
  // is optional: tracker-bearing games (Splendor / LL) get one for
  // belief_snapshot_after; snapshot-only games (Azul) trace without a
  // tracker (belief_snapshot_after stays empty).
  std::unique_ptr<GameBundle> trace_bundle;
  IBeliefTracker* trace_bt = nullptr;
  if (trace_perspective >= 0) {
    if (!trace_extractor) {
      py::gil_scoped_acquire acquire;
      throw std::runtime_error(
          "run_selfplay_episode: trace_perspective >= 0 but game '" + game_id +
          "' did not register a public_event_extractor");
    }
    trace_bundle = std::make_unique<GameBundle>(
        GameRegistry::instance().create_game(game_id, seed));
    trace_bt = trace_bundle->belief_tracker.get();  // may be nullptr
  }

  if (model_path.empty()) {
    throw std::invalid_argument("run_selfplay_episode: model_path must not be empty");
  }
  auto evaluator = std::make_unique<infer::OnnxPolicyValueEvaluator>(
      model_path, bundle.encoder.get());
  if (!evaluator->is_ready()) {
    throw std::runtime_error("run_selfplay_episode: failed to load model: " + evaluator->last_error());
  }
  const search::IPolicyValueEvaluator* eval_ptr = evaluator.get();
  // Single-model selfplay: factory returns the same evaluator for every seat.
  runtime::PolicyEvaluatorFactory eval_factory =
      [eval_ptr](int /*player*/) -> const search::IPolicyValueEvaluator& {
    return *eval_ptr;
  };

  // Belief plumbing: the extractor is always exposed when registered (so
  // emit can produce BeliefSamples even before a belief.onnx is trained).
  // The evaluator is only constructed when a model path is set; without
  // it, the tracker's prepare_for_root is a no-op and randomize_unseen
  // falls back to uniform sampling.
  std::unique_ptr<infer::OnnxBeliefEvaluator> belief_evaluator;
  const IBeliefFeatureExtractor* belief_extractor_ptr =
      bundle.belief_feature_extractor.get();
  const IBeliefEvaluator* belief_evaluator_ptr = nullptr;
  if (bundle.belief_feature_extractor && !bundle.belief_model_path.empty()) {
    belief_evaluator = std::make_unique<infer::OnnxBeliefEvaluator>(
        bundle.belief_model_path);
    if (!belief_evaluator->is_ready()) {
      throw std::runtime_error(
          "run_selfplay_episode: failed to load belief model '" +
          bundle.belief_model_path + "': " + belief_evaluator->last_error());
    }
    belief_evaluator_ptr = belief_evaluator.get();
  }

  runtime::SelfplayConfig cfg{};
  cfg.simulations = simulations;
  cfg.c_puct = c_puct;
  cfg.temperature = temperature;
  cfg.dirichlet_alpha = dirichlet_alpha;
  cfg.dirichlet_epsilon = dirichlet_epsilon;
  cfg.dirichlet_on_first_n_plies = dirichlet_on_first_n_plies;
  cfg.max_game_plies = max_game_plies;
  cfg.tail_solve_enabled = tail_solve_enabled;
  cfg.tail_solve_config.depth_limit = tail_solve_depth_limit;
  cfg.tail_solve_config.node_budget = tail_solve_node_budget;
  cfg.tail_solve_config.margin_weight = tail_solve_margin_weight;
  cfg.heuristic_guidance_ratio = heuristic_guidance_ratio;
  cfg.heuristic_temperature = heuristic_temperature;
  cfg.training_filter_ratio = training_filter_ratio;
  cfg.opponent_selection = opp_sel;

  if (temperature_initial >= 0.0 || temperature_final >= 0.0) {
    cfg.temperature_schedule.enabled = true;
    if (temperature_initial >= 0.0) {
      cfg.temperature_schedule.has_initial = true;
      cfg.temperature_schedule.initial = temperature_initial;
    }
    if (temperature_final >= 0.0) {
      cfg.temperature_schedule.has_final = true;
      cfg.temperature_schedule.final_ = temperature_final;
    }
    cfg.temperature_schedule.decay_plies = temperature_decay_plies;
  }

  // Allocate one fresh tracker + session state per seat. Each perspective's
  // tracker accumulates belief monotonically; each pp_bundle's `state`
  // doubles as that seat's session state. The runner advances each session
  // state via the public-event protocol every ply so the AI path never
  // reads truth. For fully-public games (no belief_tracker registered) the
  // tracker entries are nullptr but the per-seat session state still drives
  // the AI view.
  std::vector<std::unique_ptr<GameBundle>> pp_bundles;
  std::vector<IBeliefTracker*> pp_trackers;
  std::vector<IGameState*> per_seat_states;
  const int num_players = bundle.state->num_players();
  pp_bundles.reserve(static_cast<size_t>(num_players));
  pp_trackers.reserve(static_cast<size_t>(num_players));
  per_seat_states.reserve(static_cast<size_t>(num_players));
  for (int p = 0; p < num_players; ++p) {
    auto pb = std::make_unique<GameBundle>(
        GameRegistry::instance().create_game(game_id, seed));
    pp_trackers.push_back(pb->belief_tracker.get());
    per_seat_states.push_back(pb->state.get());
    pp_bundles.push_back(std::move(pb));
  }

  auto result = runtime::run_selfplay_episode(
      *bundle.state, *bundle.rules, *bundle.value_model, eval_factory, cfg, seed,
      pp_trackers,
      per_seat_states,
      bundle.public_state_applier,
      bundle.encoder.get(),
      bundle.tail_solver.get(),
      bundle.adjudicator,
      bundle.auxiliary_scorer,
      bundle.heuristic_picker,
      bundle.training_action_filter,
      bundle.tail_solve_trigger,
      bundle.episode_stats_extractor,
      trace_perspective,
      trace_bt,
      trace_extractor,
      bundle.events_only_extractor,
      belief_extractor_ptr,
      belief_evaluator_ptr,
      bundle.belief_label_extractor.get());

  py::gil_scoped_acquire acquire;
  return result_to_py(result);
}

// Opponent-pool variant of run_selfplay_episode. Takes a per-seat list of
// model paths (size == num_players) and builds one OnnxPolicyValueEvaluator
// per seat. The factory routes each seat to its evaluator. Used by
// pipeline.py's pool-mode workers (latest model on `latest_seat`, a random
// historical checkpoint on every other seat). Sample filtering by seat is
// the caller's responsibility (Python side).
py::dict run_selfplay_episode_pool_py(
    const std::string& game_id,
    std::uint64_t seed,
    const std::vector<std::string>& model_paths,
    int simulations,
    float c_puct,
    double temperature,
    double dirichlet_alpha,
    double dirichlet_epsilon,
    int dirichlet_on_first_n_plies,
    int max_game_plies,
    bool tail_solve_enabled,
    int tail_solve_depth_limit,
    std::int64_t tail_solve_node_budget,
    float tail_solve_margin_weight,
    double temperature_initial,
    double temperature_final,
    int temperature_decay_plies,
    double heuristic_guidance_ratio,
    double heuristic_temperature,
    double training_filter_ratio,
    int trace_perspective,
    const std::string& opponent_selection,
    const std::string& belief_model_path) {
  const auto opp_sel = parse_opponent_selection(opponent_selection);
  py::gil_scoped_release release;

  auto bundle = GameRegistry::instance().create_game(game_id, seed);
  if (!belief_model_path.empty()) bundle.belief_model_path = belief_model_path;
  const int num_players = bundle.state->num_players();

  if (static_cast<int>(model_paths.size()) != num_players) {
    py::gil_scoped_acquire acquire;
    throw std::invalid_argument(
        "run_selfplay_episode_pool: model_paths size " +
        std::to_string(model_paths.size()) + " != num_players " +
        std::to_string(num_players));
  }
  for (size_t i = 0; i < model_paths.size(); ++i) {
    if (model_paths[i].empty()) {
      py::gil_scoped_acquire acquire;
      throw std::invalid_argument(
          "run_selfplay_episode_pool: model_paths[" + std::to_string(i) +
          "] must not be empty");
    }
  }

  runtime::PublicEventExtractor trace_extractor = bundle.public_event_extractor;
  std::unique_ptr<GameBundle> trace_bundle;
  IBeliefTracker* trace_bt = nullptr;
  if (trace_perspective >= 0) {
    if (!trace_extractor) {
      py::gil_scoped_acquire acquire;
      throw std::runtime_error(
          "run_selfplay_episode_pool: trace_perspective >= 0 but game '" +
          game_id + "' did not register a public_event_extractor");
    }
    trace_bundle = std::make_unique<GameBundle>(
        GameRegistry::instance().create_game(game_id, seed));
    trace_bt = trace_bundle->belief_tracker.get();
  }

  std::vector<std::unique_ptr<infer::OnnxPolicyValueEvaluator>> evaluators;
  std::vector<const search::IPolicyValueEvaluator*> eval_ptrs;
  evaluators.reserve(model_paths.size());
  eval_ptrs.reserve(model_paths.size());
  for (size_t i = 0; i < model_paths.size(); ++i) {
    auto ev = std::make_unique<infer::OnnxPolicyValueEvaluator>(
        model_paths[i], bundle.encoder.get());
    if (!ev->is_ready()) {
      py::gil_scoped_acquire acquire;
      throw std::runtime_error(
          "run_selfplay_episode_pool: failed to load model_" +
          std::to_string(i) + ": " + ev->last_error());
    }
    eval_ptrs.push_back(ev.get());
    evaluators.push_back(std::move(ev));
  }
  const size_t n_eval = eval_ptrs.size();
  runtime::PolicyEvaluatorFactory eval_factory =
      [&eval_ptrs, n_eval](int player) -> const search::IPolicyValueEvaluator& {
    return *eval_ptrs[static_cast<size_t>(player) % n_eval];
  };

  // Belief plumbing: extractor exposed unconditionally for emit; evaluator
  // only when a belief model path is set (see run_selfplay_episode for the
  // full rationale).
  std::unique_ptr<infer::OnnxBeliefEvaluator> belief_evaluator;
  const IBeliefFeatureExtractor* belief_extractor_ptr =
      bundle.belief_feature_extractor.get();
  const IBeliefEvaluator* belief_evaluator_ptr = nullptr;
  if (bundle.belief_feature_extractor && !bundle.belief_model_path.empty()) {
    belief_evaluator = std::make_unique<infer::OnnxBeliefEvaluator>(
        bundle.belief_model_path);
    if (!belief_evaluator->is_ready()) {
      throw std::runtime_error(
          "run_selfplay_episode_pool: failed to load belief model '" +
          bundle.belief_model_path + "': " + belief_evaluator->last_error());
    }
    belief_evaluator_ptr = belief_evaluator.get();
  }

  runtime::SelfplayConfig cfg{};
  cfg.simulations = simulations;
  cfg.c_puct = c_puct;
  cfg.temperature = temperature;
  cfg.dirichlet_alpha = dirichlet_alpha;
  cfg.dirichlet_epsilon = dirichlet_epsilon;
  cfg.dirichlet_on_first_n_plies = dirichlet_on_first_n_plies;
  cfg.max_game_plies = max_game_plies;
  cfg.tail_solve_enabled = tail_solve_enabled;
  cfg.tail_solve_config.depth_limit = tail_solve_depth_limit;
  cfg.tail_solve_config.node_budget = tail_solve_node_budget;
  cfg.tail_solve_config.margin_weight = tail_solve_margin_weight;
  cfg.heuristic_guidance_ratio = heuristic_guidance_ratio;
  cfg.heuristic_temperature = heuristic_temperature;
  cfg.training_filter_ratio = training_filter_ratio;
  cfg.opponent_selection = opp_sel;

  if (temperature_initial >= 0.0 || temperature_final >= 0.0) {
    cfg.temperature_schedule.enabled = true;
    if (temperature_initial >= 0.0) {
      cfg.temperature_schedule.has_initial = true;
      cfg.temperature_schedule.initial = temperature_initial;
    }
    if (temperature_final >= 0.0) {
      cfg.temperature_schedule.has_final = true;
      cfg.temperature_schedule.final_ = temperature_final;
    }
    cfg.temperature_schedule.decay_plies = temperature_decay_plies;
  }

  std::vector<std::unique_ptr<GameBundle>> pp_bundles;
  std::vector<IBeliefTracker*> pp_trackers;
  std::vector<IGameState*> per_seat_states;
  pp_bundles.reserve(static_cast<size_t>(num_players));
  pp_trackers.reserve(static_cast<size_t>(num_players));
  per_seat_states.reserve(static_cast<size_t>(num_players));
  for (int p = 0; p < num_players; ++p) {
    auto pb = std::make_unique<GameBundle>(
        GameRegistry::instance().create_game(game_id, seed));
    pp_trackers.push_back(pb->belief_tracker.get());
    per_seat_states.push_back(pb->state.get());
    pp_bundles.push_back(std::move(pb));
  }

  auto result = runtime::run_selfplay_episode(
      *bundle.state, *bundle.rules, *bundle.value_model, eval_factory, cfg, seed,
      pp_trackers,
      per_seat_states,
      bundle.public_state_applier,
      bundle.encoder.get(),
      bundle.tail_solver.get(),
      bundle.adjudicator,
      bundle.auxiliary_scorer,
      bundle.heuristic_picker,
      bundle.training_action_filter,
      bundle.tail_solve_trigger,
      bundle.episode_stats_extractor,
      trace_perspective,
      trace_bt,
      trace_extractor,
      bundle.events_only_extractor,
      belief_extractor_ptr,
      belief_evaluator_ptr,
      bundle.belief_label_extractor.get());

  py::gil_scoped_acquire acquire;
  return result_to_py(result);
}

py::dict run_arena_match_py(
    const std::string& game_id,
    std::uint64_t seed,
    const std::vector<std::string>& model_paths,
    const std::vector<int>& simulations_list,
    double temperature,
    int max_game_plies,
    bool tail_solve,
    int tail_solve_depth_limit,
    std::int64_t tail_solve_node_budget,
    float tail_solve_margin_weight,
    const std::vector<std::string>& opponent_selection_list,
    const std::string& belief_model_path) {
  // Per-player opponent_selection. Empty list => all "puct". Non-empty
  // must match model_paths size.
  std::vector<search::OpponentSelection> opp_sel_per_player;
  if (!opponent_selection_list.empty()) {
    if (opponent_selection_list.size() != model_paths.size()) {
      throw std::invalid_argument(
          "run_arena_match: opponent_selection_list size " +
          std::to_string(opponent_selection_list.size()) +
          " != model_paths size " + std::to_string(model_paths.size()));
    }
    opp_sel_per_player.reserve(opponent_selection_list.size());
    for (const auto& s : opponent_selection_list) {
      opp_sel_per_player.push_back(parse_opponent_selection(s));
    }
  }
  py::gil_scoped_release release;

  auto bundle = GameRegistry::instance().create_game(game_id, seed);
  if (!belief_model_path.empty()) bundle.belief_model_path = belief_model_path;

  if (model_paths.empty()) {
    throw std::invalid_argument("run_arena_match: model_paths must not be empty");
  }
  for (size_t i = 0; i < model_paths.size(); ++i) {
    if (model_paths[i].empty()) {
      throw std::invalid_argument(
          "run_arena_match: model_paths[" + std::to_string(i) + "] must not be empty");
    }
  }

  std::vector<std::unique_ptr<infer::OnnxPolicyValueEvaluator>> evaluators;
  std::vector<const search::IPolicyValueEvaluator*> eval_ptrs;
  for (size_t i = 0; i < model_paths.size(); ++i) {
    auto ev = std::make_unique<infer::OnnxPolicyValueEvaluator>(
        model_paths[i], bundle.encoder.get());
    if (!ev->is_ready()) {
      throw std::runtime_error(
          "run_arena_match: failed to load model_" + std::to_string(i) + ": " + ev->last_error());
    }
    eval_ptrs.push_back(ev.get());
    evaluators.push_back(std::move(ev));
  }

  std::vector<runtime::ArenaPlayerConfig> player_configs;
  for (size_t i = 0; i < model_paths.size(); ++i) {
    runtime::ArenaPlayerConfig cfg{};
    cfg.simulations = (i < simulations_list.size())
        ? simulations_list[i] : 200;
    cfg.temperature = temperature;
    if (i < opp_sel_per_player.size()) {
      cfg.opponent_selection = opp_sel_per_player[i];
    }
    if (tail_solve) {
      if (!bundle.tail_solver) {
        throw std::invalid_argument(
            "run_arena_match: tail_solve=true but game " + game_id +
            " has no tail_solver registered.");
      }
      if (!bundle.tail_solve_trigger) {
        throw std::invalid_argument(
            "run_arena_match: tail_solve=true but game " + game_id +
            " has no tail_solve_trigger registered.");
      }
      cfg.tail_solve_enabled = true;
      cfg.tail_solve_config.depth_limit = tail_solve_depth_limit;
      cfg.tail_solve_config.node_budget = tail_solve_node_budget;
      cfg.tail_solve_config.margin_weight = tail_solve_margin_weight;
      cfg.tail_solver = bundle.tail_solver.get();
      cfg.tail_solve_trigger = bundle.tail_solve_trigger;
    }
    player_configs.push_back(cfg);
  }

  IBeliefTracker* arena_bt = bundle.belief_tracker.get();
  const size_t n_eval = eval_ptrs.size();

  std::unique_ptr<infer::OnnxBeliefEvaluator> belief_evaluator;
  const IBeliefFeatureExtractor* belief_extractor_ptr = nullptr;
  const IBeliefEvaluator* belief_evaluator_ptr = nullptr;
  if (bundle.belief_feature_extractor && !bundle.belief_model_path.empty()) {
    belief_evaluator = std::make_unique<infer::OnnxBeliefEvaluator>(
        bundle.belief_model_path);
    if (!belief_evaluator->is_ready()) {
      throw std::runtime_error(
          "run_arena_match: failed to load belief model '" +
          bundle.belief_model_path + "': " + belief_evaluator->last_error());
    }
    belief_extractor_ptr = bundle.belief_feature_extractor.get();
    belief_evaluator_ptr = belief_evaluator.get();
  }

  // Per-seat session state + tracker for every game.
  std::vector<std::unique_ptr<GameBundle>> pp_bundles;
  std::vector<IBeliefTracker*> pp_trackers;
  std::vector<IGameState*> per_seat_states;
  const int num_players_arena = bundle.state->num_players();
  pp_bundles.reserve(static_cast<size_t>(num_players_arena));
  pp_trackers.reserve(static_cast<size_t>(num_players_arena));
  per_seat_states.reserve(static_cast<size_t>(num_players_arena));
  for (int p = 0; p < num_players_arena; ++p) {
    auto pb = std::make_unique<GameBundle>(
        GameRegistry::instance().create_game(game_id, seed));
    pp_trackers.push_back(pb->belief_tracker.get());
    per_seat_states.push_back(pb->state.get());
    pp_bundles.push_back(std::move(pb));
  }

  auto result = runtime::run_arena_match(
      *bundle.state, *bundle.rules, *bundle.value_model,
      [&eval_ptrs, n_eval](int player) -> const search::IPolicyValueEvaluator& {
        return *eval_ptrs[static_cast<size_t>(player) % n_eval];
      },
      player_configs, max_game_plies, seed,
      arena_bt, bundle.adjudicator,
      bundle.public_event_extractor,
      pp_trackers,
      per_seat_states,
      bundle.public_state_applier,
      bundle.events_only_extractor,
      belief_extractor_ptr,
      belief_evaluator_ptr);

  py::gil_scoped_acquire acquire;
  py::dict out;
  out["winner"] = result.winner;
  out["draw"] = result.draw;
  out["total_plies"] = result.total_plies;
  py::list actions;
  for (auto a : result.action_history) actions.append(static_cast<int>(a));
  out["action_history"] = actions;
  py::list ply_stats;
  for (const auto& ps : result.ply_stats) {
    py::dict d;
    d["tail_solved"] = ps.tail_solved;
    d["tail_solve_value"] = ps.tail_solve_value;
    ply_stats.append(d);
  }
  out["ply_stats"] = ply_stats;
  return out;
}

py::dict run_constrained_eval_vs_heuristic_py(
    const std::string& game_id,
    std::uint64_t seed,
    const std::string& model_path,
    int simulations,
    int model_is_player,
    bool constrained,
    double heuristic_temperature,
    const std::string& opponent_selection,
    const std::string& belief_model_path) {
  const auto opp_sel = parse_opponent_selection(opponent_selection);
  py::gil_scoped_release release;

  auto bundle = GameRegistry::instance().create_game(game_id, seed);
  if (!belief_model_path.empty()) bundle.belief_model_path = belief_model_path;

  if (model_path.empty()) {
    throw std::invalid_argument("run_constrained_eval_vs_heuristic: model_path must not be empty");
  }
  auto model_eval = std::make_unique<infer::OnnxPolicyValueEvaluator>(
      model_path, bundle.encoder.get());
  if (!model_eval->is_ready()) {
    throw std::runtime_error("run_constrained_eval_vs_heuristic: failed to load model: " + model_eval->last_error());
  }
  const search::IPolicyValueEvaluator* eval_ptr = model_eval.get();

  std::unique_ptr<infer::OnnxBeliefEvaluator> belief_evaluator;
  const IBeliefEvaluator* belief_evaluator_ptr = nullptr;
  if (bundle.belief_feature_extractor && !bundle.belief_model_path.empty()) {
    belief_evaluator = std::make_unique<infer::OnnxBeliefEvaluator>(
        bundle.belief_model_path);
    if (!belief_evaluator->is_ready()) {
      throw std::runtime_error(
          "run_constrained_eval_vs_heuristic: failed to load belief model '" +
          bundle.belief_model_path + "': " + belief_evaluator->last_error());
    }
    belief_evaluator_ptr = belief_evaluator.get();
  }

  std::unique_ptr<runtime::FilteredRulesWrapper> filtered_rules;
  if (constrained && bundle.training_action_filter) {
    filtered_rules = std::make_unique<runtime::FilteredRulesWrapper>(
        *bundle.rules, bundle.training_action_filter);
  }

  auto state = bundle.state->clone_state();
  int ply = 0;
  std::vector<ActionId> action_history;
  std::vector<runtime::ArenaPlyStats> ply_stats_vec;

  std::mt19937_64 rng(seed ^ 0xBEEF);
  IBeliefTracker* bt = bundle.belief_tracker.get();

  while (!state->is_terminal() && ply < 500) {
    const int cp = state->current_player();

    if (cp == model_is_player) {
      const IGameRules& rules_for_model =
          filtered_rules ? *filtered_rules : *bundle.rules;
      const auto legal = rules_for_model.legal_actions(*state);
      if (legal.empty()) break;

      if (bt) tracker_init(*bt, bundle, *state, cp);

      search::NetMctsConfig mcts_cfg{};
      mcts_cfg.simulations = simulations;
      mcts_cfg.c_puct = 1.4f;
      mcts_cfg.opponent_selection = opp_sel;
      if (bt) {
        mcts_cfg.root_belief_tracker = bt;
      }
      if (bundle.belief_feature_extractor && belief_evaluator_ptr) {
        mcts_cfg.belief_extractor = bundle.belief_feature_extractor.get();
        mcts_cfg.belief_evaluator = belief_evaluator_ptr;
      }
      // Sim-tracker descent maintenance — events-only flavor.
      if (bundle.events_only_extractor) {
        mcts_cfg.events_only_extractor = bundle.events_only_extractor;
      }

      if (bundle.tail_solver) {
        bool try_ts = false;
        if (bundle.tail_solve_trigger) {
          try_ts = bundle.tail_solve_trigger(*state, ply);
        } else if (ply >= 20) {
          try_ts = true;
        }
        if (try_ts) {
          mcts_cfg.tail_solve_enabled = true;
          mcts_cfg.tail_solve_config.depth_limit = 10;
          mcts_cfg.tail_solve_config.node_budget = 200000;
          mcts_cfg.tail_solver = bundle.tail_solver.get();
        }
      }

      search::NetMcts mcts(mcts_cfg);
      search::NetMctsStats stats{};
      const std::uint64_t mcts_seed = board_ai::rng::derive_subseed(
          seed, "py_engine.replay_mcts", static_cast<std::uint64_t>(ply));
      mcts.search_root(*state, rules_for_model, *bundle.value_model,
                        *eval_ptr, &stats, mcts_seed);

      ActionId chosen = search::select_action_from_visits(
          stats.root_actions, stats.root_action_visits, 0.0,
          seed ^ static_cast<std::uint64_t>(ply), legal[0]);

      action_history.push_back(chosen);
      ply_stats_vec.push_back({stats.tail_solved, stats.tail_solve_value});
      std::unique_ptr<IGameState> state_before;
      if (bt) state_before = state->clone_state();
      bundle.rules->do_action_fast(*state, chosen, rng);
      if (bt) tracker_observe(*bt, bundle, *state_before, chosen, *state, cp);
    } else {
      if (!bundle.heuristic_picker) break;
      auto hr = bundle.heuristic_picker(*state, *bundle.rules, rng());
      if (hr.actions.empty()) break;

      double u01 = static_cast<double>(rng() & 0xFFFFFFFF) / 4294967296.0;
      std::size_t idx = runtime::sample_heuristic_index(
          hr.scores, heuristic_temperature, u01);
      ActionId chosen = hr.actions[idx];
      action_history.push_back(chosen);
      ply_stats_vec.push_back({false, 0.0f});
      std::unique_ptr<IGameState> state_before;
      if (bt) state_before = state->clone_state();
      bundle.rules->do_action_fast(*state, chosen, rng);
      if (bt) tracker_observe(*bt, bundle, *state_before, chosen, *state, cp);
    }
    ++ply;
  }

  int winner = -1;
  bool draw = true;
  if (state->is_terminal()) {
    winner = state->winner();
    draw = (winner < 0);
  } else if (bundle.adjudicator) {
    winner = bundle.adjudicator(*state);
    draw = (winner < 0);
  }

  py::gil_scoped_acquire acquire;
  py::dict out;
  out["winner"] = winner;
  out["draw"] = draw;
  out["total_plies"] = ply;
  py::list actions;
  for (auto a : action_history) actions.append(static_cast<int>(a));
  out["action_history"] = actions;
  py::list pstats;
  for (const auto& ps : ply_stats_vec) {
    py::dict d;
    d["tail_solved"] = ps.tail_solved;
    d["tail_solve_value"] = ps.tail_solve_value;
    pstats.append(d);
  }
  out["ply_stats"] = pstats;
  return out;
}

py::dict run_heuristic_episode_py(
    const std::string& game_id,
    std::uint64_t seed,
    double temperature,
    int max_game_plies) {
  py::gil_scoped_release release;

  auto bundle = GameRegistry::instance().create_game(game_id, seed);

  // Per-seat session state + tracker for every game.
  std::vector<std::unique_ptr<GameBundle>> pp_bundles;
  std::vector<IBeliefTracker*> pp_trackers;
  std::vector<IGameState*> per_seat_states;
  const int num_players_heur = bundle.state->num_players();
  pp_bundles.reserve(static_cast<size_t>(num_players_heur));
  pp_trackers.reserve(static_cast<size_t>(num_players_heur));
  per_seat_states.reserve(static_cast<size_t>(num_players_heur));
  for (int p = 0; p < num_players_heur; ++p) {
    auto pb = std::make_unique<GameBundle>(
        GameRegistry::instance().create_game(game_id, seed));
    pp_trackers.push_back(pb->belief_tracker.get());
    per_seat_states.push_back(pb->state.get());
    pp_bundles.push_back(std::move(pb));
  }

  auto result = runtime::run_heuristic_episode(
      *bundle.state, *bundle.rules, *bundle.value_model,
      bundle.encoder.get(),
      bundle.heuristic_picker,
      temperature, max_game_plies, seed,
      bundle.auxiliary_scorer, bundle.adjudicator,
      pp_trackers,
      per_seat_states,
      bundle.public_state_applier,
      bundle.public_event_extractor);

  py::gil_scoped_acquire acquire;
  return result_to_py(result);
}

// Encode the bundle's initial state from a specific perspective seat,
// optionally after init'ing the bundle's belief tracker for a possibly
// DIFFERENT perspective and applying a sequence of (actor, action) public
// events on that tracker. Used by OB-002 regression test: a tracker bound
// to seat A who learned an opponent's hand via Priest must not leak that
// knowledge into the encoder output for seat B.
py::dict encode_state_for_perspective_py(
    const std::string& game_id,
    std::uint64_t seed,
    int encode_player,
    int tracker_perspective,
    py::list event_actions) {
  py::gil_scoped_release release;

  auto bundle = GameRegistry::instance().create_game(game_id, seed);
  if (encode_player < 0 || encode_player >= bundle.state->num_players()) {
    py::gil_scoped_acquire acquire;
    throw std::runtime_error("encode_state_for_perspective: encode_player out of range");
  }

  // Drive both bundle.state (truth) and bundle.belief_tracker (perspective
  // observer) forward by replaying actions. This populates tracker's
  // known_hand from real public events (e.g. Priest reveals) without us
  // synthesizing event payloads by hand.
  bool tracker_init_done = false;
  if (bundle.belief_tracker && tracker_perspective >= 0) {
    tracker_init(*bundle.belief_tracker, bundle, *bundle.state, tracker_perspective);
    tracker_init_done = true;
  }

  py::gil_scoped_acquire acquire_for_list;
  std::vector<std::pair<int, ActionId>> pairs;
  pairs.reserve(py::len(event_actions));
  for (auto h : event_actions) {
    auto t = h.cast<py::tuple>();
    pairs.emplace_back(t[0].cast<int>(), t[1].cast<ActionId>());
  }
  py::gil_scoped_release release2;

  std::mt19937_64 step_rng(
      board_ai::rng::derive_subseed(seed, "py_engine.encode_replay_step"));
  for (const auto& [actor, action] : pairs) {
    auto before = bundle.state->clone_state();
    bundle.rules->do_action_fast(*bundle.state, action, step_rng);
    if (tracker_init_done && bundle.public_event_extractor) {
      auto ev = bundle.public_event_extractor(*before, action, *bundle.state, tracker_perspective);
      bundle.belief_tracker->observe_public_event(actor, action, ev.events);
    }
  }

  std::vector<float> features;
  auto masked = make_masked_state(*bundle.state,
                                  bundle.state->schema_ref(), encode_player);
  bundle.encoder->encode_features(*masked, encode_player,
                                  bundle.belief_tracker.get(), &features);

  py::gil_scoped_acquire acquire;
  py::dict out;
  out["features"] = features;
  out["tracker_perspective"] =
      bundle.belief_tracker ? tracker_perspective : -1;
  return out;
}

py::dict encode_state_py(
    const std::string& game_id,
    std::uint64_t seed) {
  py::gil_scoped_release release;

  auto bundle = GameRegistry::instance().create_game(game_id, seed);
  const int player = bundle.state->current_player();
  const auto legal = bundle.rules->legal_actions(*bundle.state);

  std::vector<float> features;
  std::vector<float> legal_mask;
  bundle.encoder->encode(*bundle.state, player, bundle.belief_tracker.get(), legal, &features, &legal_mask);

  const bool is_terminal = bundle.state->is_terminal();
  const int action_space = bundle.encoder->action_space();
  const int feature_dim = bundle.encoder->feature_dim();

  py::gil_scoped_acquire acquire;
  py::dict out;
  out["features"] = features;
  out["legal_mask"] = legal_mask;
  out["legal_actions"] = legal;
  out["current_player"] = player;
  out["is_terminal"] = is_terminal;
  out["action_space"] = action_space;
  out["feature_dim"] = feature_dim;
  return out;
}

py::dict tail_solve_py(
    const std::string& game_id,
    std::uint64_t seed,
    int perspective_player,
    int depth_limit,
    std::int64_t node_budget) {
  py::gil_scoped_release release;

  auto bundle = GameRegistry::instance().create_game(game_id, seed);
  if (!bundle.tail_solver) {
    py::gil_scoped_acquire acquire;
    throw std::runtime_error("tail_solve: game '" + game_id + "' has no tail_solver registered");
  }

  search::TailSolveConfig cfg{};
  cfg.depth_limit = depth_limit;
  cfg.node_budget = node_budget;
  if (bundle.auxiliary_scorer) {
    cfg.margin_weight = 0.01f;
    cfg.margin_scorer = bundle.auxiliary_scorer;
  }

  auto ts = bundle.tail_solver->solve(
      *bundle.state, *bundle.rules, *bundle.value_model,
      perspective_player, cfg);

  py::gil_scoped_acquire acquire;
  py::dict out;
  out["value"] = ts.value;
  out["best_action"] = ts.best_action;
  out["nodes_searched"] = ts.nodes_searched;
  out["elapsed_ms"] = ts.elapsed_ms;
  out["budget_exceeded"] = ts.budget_exceeded;
  return out;
}

class GameSessionWrapper {
 public:
  GameSessionWrapper(const std::string& game_id, std::uint64_t seed,
                     const std::string& model_path, bool use_filter,
                     const std::string& belief_model_path = "")
      : game_id_(game_id), seed_(seed), model_path_(model_path) {
    py::gil_scoped_release release;
    bundle_ = std::make_unique<GameBundle>(
        GameRegistry::instance().create_game(game_id, seed));
    if (!belief_model_path.empty()) {
      bundle_->belief_model_path = belief_model_path;
    }
    if (!model_path.empty()) {
      evaluator_ = std::make_unique<infer::OnnxPolicyValueEvaluator>(
          model_path, bundle_->encoder.get());
      if (!evaluator_->is_ready()) {
        throw std::runtime_error("GameSession: failed to load model: " + evaluator_->last_error());
      }
    }
    if (bundle_->belief_feature_extractor &&
        !bundle_->belief_model_path.empty()) {
      belief_evaluator_ = std::make_unique<infer::OnnxBeliefEvaluator>(
          bundle_->belief_model_path);
      if (!belief_evaluator_->is_ready()) {
        throw std::runtime_error(
            "GameSession: failed to load belief model '" +
            bundle_->belief_model_path + "': " +
            belief_evaluator_->last_error());
      }
    }
    if (use_filter && bundle_->training_action_filter) {
      filtered_rules_ = std::make_unique<runtime::FilteredRulesWrapper>(
          *bundle_->rules, bundle_->training_action_filter);
    }
    bt_ = bundle_->belief_tracker.get();
    // Per-perspective AI views: MCTS searches on these, never on the truth
    // state (bundle_->state). Skipped when external_obs_mode_ is set
    // (AI API path) — in that mode bundle_->state IS the AI view.
    init_ai_views_();
  }

  void init_ai_views_() {
    if (external_obs_mode_) return;
    const int n = bundle_->state->num_players();
    ai_views_.resize(static_cast<size_t>(n));
    ai_trackers_.resize(static_cast<size_t>(n));
    ai_encoders_.resize(static_cast<size_t>(n));
    ai_evaluators_.resize(static_cast<size_t>(n));
    for (int p = 0; p < n; ++p) {
      auto extra = GameRegistry::instance().create_game(game_id_, seed_);
      ai_trackers_[p] = std::move(extra.belief_tracker);
      ai_encoders_[p] = std::move(extra.encoder);

      // Bootstrap each AI view from the truth state via the walker:
      // viz=1 slots flow through; viz=0 slots are placeholder. This is
      // the cleanest "nature action 0" snapshot — game-agnostic.
      ai_views_[p] = make_masked_state(*bundle_->state,
                                       bundle_->state->schema_ref(), p);
      if (ai_trackers_[p]) {
        tracker_init(*ai_trackers_[p], *bundle_, *ai_views_[p], p);
      }
      if (!model_path_.empty() && ai_encoders_[p]) {
        ai_evaluators_[p] = std::make_unique<infer::OnnxPolicyValueEvaluator>(
            model_path_, ai_encoders_[p].get());
        if (!ai_evaluators_[p]->is_ready()) {
          throw std::runtime_error(
              "GameSession: failed to load model (ai_view[" + std::to_string(p) +
              "]): " + ai_evaluators_[p]->last_error());
        }
      }
    }
  }

  // Advance ai_view for a single perspective. Hidden-info games: snapshot
  // overwrites the public part and the tracker observes the events. Fully-
  // public games (no extractor registered) just run do_action_fast on the
  // seat — there is no snapshot to apply. The session's viz=0 slots are
  // intentionally NOT freshened: nothing on the decision side reads them
  // (MCTS sims sample at sim entry, hash uses kHiddenHashSentinel for viz=0,
  // encoder reads masked-state placeholders).
  void advance_ai_view_(int perspective, const IGameState& truth_before,
                        ActionId action) {
    if (perspective < 0 || perspective >= static_cast<int>(ai_views_.size())) return;
    if (!ai_views_[perspective]) return;
    const int actor = truth_before.current_player();
    if (!bundle_->public_event_extractor) {
      const std::uint64_t view_step_seed = board_ai::rng::derive_subseed(
          seed_, "session.view_step",
          static_cast<std::uint64_t>(ply_count_) * 17ULL +
              static_cast<std::uint64_t>(perspective));
      std::mt19937_64 view_step_rng(view_step_seed);
      bundle_->rules->do_action_fast(*ai_views_[perspective], action, view_step_rng);
      if (ai_trackers_[perspective]) {
        ai_trackers_[perspective]->observe_public_event(actor, action, {});
      }
      return;
    }
    PublicEventTrace trace = bundle_->public_event_extractor(
        truth_before, action, *bundle_->state, perspective);
    ai_views_[perspective]->begin_step_for_session_observe();
    if (bundle_->public_state_applier && !trace.public_snapshot.empty()) {
      bundle_->public_state_applier(
          *ai_views_[perspective], trace.public_snapshot, perspective);
    }
    if (ai_trackers_[perspective]) {
      ai_trackers_[perspective]->observe_public_event(
          actor, action, trace.events);
    }
  }

  bool is_terminal() const { return bundle_->state->is_terminal(); }
  bool is_turn_start() const { return bundle_->state->is_turn_start(); }
  int current_player() const { return bundle_->state->current_player(); }
  int winner() const { return bundle_->state->winner(); }
  int num_players() const { return bundle_->state->num_players(); }
  std::string game_id() const { return game_id_; }

  // Expose perspective hash so tests can directly assert hash-scope
  // invariants (e.g. "internal RNG state must NOT affect public hash" —
  // the BUG-028 anti-pattern). Returns the truth state's hash; for
  // information-set hashing through ai_views_, callers can use
  // state_hash_for_perspective_ai_view.
  std::uint64_t state_hash_for_perspective(int player) const {
    return bundle_->state->state_hash_for_perspective(player);
  }

  // Expose step_count so tests can assert framework wrappers around
  // do_action_fast / do_action_deterministic / undo_action keep the
  // counter strictly monotonic across rules-path transitions.
  std::uint32_t step_count() const { return bundle_->state->step_count(); }

  py::dict get_state_dict() {
    if (!bundle_->state_serializer) {
      throw std::runtime_error("get_state_dict: game '" + game_id_ + "' has no state_serializer registered");
    }
    py::gil_scoped_release release;
    auto m = bundle_->state_serializer(*bundle_->state);
    py::gil_scoped_acquire acquire;
    py::dict out;
    for (const auto& [k, v] : m) out[py::cast(k)] = any_to_py(v);
    return out;
  }

  // Test hook: directly invoke the game's public_state_applier on a given
  // snapshot. Used by test_public_snapshot_round_trip to verify the applier
  // is a correct inverse of the extractor without going through
  // apply_observation.
  void apply_public_snapshot(py::dict snapshot, int receiver_seat) {
    if (!bundle_->public_state_applier) {
      throw std::runtime_error(
          "apply_public_snapshot: game '" + game_id_ +
          "' has no public_state_applier registered");
    }
    AnyMap snap = py_dict_to_any_map(snapshot);
    py::gil_scoped_release release;
    bundle_->public_state_applier(*bundle_->state, snap, receiver_seat);
  }

  py::dict get_action_info(ActionId action) {
    if (!bundle_->action_descriptor) {
      throw std::runtime_error("get_action_info: game '" + game_id_ + "' has no action_descriptor registered");
    }
    auto m = bundle_->action_descriptor(action);
    py::dict out;
    for (const auto& [k, v] : m) out[py::cast(k)] = any_to_py(v);
    return out;
  }

  std::vector<ActionId> get_legal_actions() {
    py::gil_scoped_release release;
    const IGameRules& rules = filtered_rules_ ? *filtered_rules_ : *bundle_->rules;
    return rules.legal_actions(*bundle_->state);
  }

  std::vector<ActionId> get_all_legal_actions() {
    py::gil_scoped_release release;
    return bundle_->rules->legal_actions(*bundle_->state);
  }

  void apply_action(ActionId action) {
    py::gil_scoped_release release;
    const int actor = bundle_->state->current_player();
    std::unique_ptr<IGameState> state_before = bundle_->state->clone_state();
    bundle_->rules->do_action_fast(*bundle_->state, action, step_rng_);
    if (bt_) {
      tracker_observe(*bt_, *bundle_, *state_before, action, *bundle_->state,
                      actor);
    }

    // Advance each perspective's AI view via the public-event protocol.
    if (!external_obs_mode_) {
      const int n = static_cast<int>(ai_views_.size());
      for (int p = 0; p < n; ++p) {
        advance_ai_view_(p, *state_before, action);
      }
    }
    ++ply_count_;
  }

  // Combined action + events step for the AI API. Sequence:
  //   1. begin_step_for_session_observe() bumps step_count for DAG
  //      acyclicity (do_action_fast is intentionally skipped — the AI
  //      session never runs rules).
  //   2. public_state_applier(snapshot) overwrites session state_'s public
  //      fields from the truth snapshot.
  //   3. belief_tracker.observe_public_event(actor, action, events) — the
  //      events list is fed only to the tracker; it does not mutate state.
  //
  // After apply_observation returns:
  //   - session state_'s public fields equal the truth snapshot exactly
  //     (test_public_snapshot_round_trip);
  //   - session state_'s viz=0 slots retain whatever was last written there;
  //     nothing on the decision side reads them. MCTS sims sample at sim
  //     entry on a cloned sim_tracker; the hash mixes kHiddenHashSentinel
  //     for viz=0; the encoder reads masked-state placeholders.
  //   - `state_hash_for_perspective(own)` on the session is byte-equal to
  //     running the same observation stream on any other seed
  //     (test_public_hash_excludes_internal_rng / test_api_belief_matches_selfplay).
  //
  // `events` is a list of {"kind": str, "payload": dict} entries describing
  // public observations from this transition. `public_snapshot` is the
  // truth-side dump of all public fields; when non-empty + game has
  // public_state_applier, it overwrites session state_'s public fields.
  // Fully-public games (tictactoe, quoridor) register no applier.
  void apply_observation(ActionId action,
                         py::list events,
                         py::dict public_snapshot) {
    if (!bundle_->public_state_applier) {
      throw std::runtime_error(
          "apply_observation: game '" + game_id_ +
          "' has no public_state_applier registered");
    }
    std::vector<std::pair<std::string, AnyMap>> event_list;
    for (py::handle item : events) {
      py::dict d = py::cast<py::dict>(item);
      std::string kind = py::cast<std::string>(d["kind"]);
      AnyMap payload = py_dict_to_any_map(py::cast<py::dict>(d["payload"]));
      event_list.emplace_back(std::move(kind), std::move(payload));
    }

    AnyMap snap_map;
    bool have_snapshot = false;
    if (public_snapshot && py::len(public_snapshot) > 0) {
      snap_map = py_dict_to_any_map(public_snapshot);
      have_snapshot = true;
    }

    py::gil_scoped_release release;
    external_obs_mode_ = true;
    const int actor = bundle_->state->current_player();
    bundle_->state->begin_step_for_session_observe();
    if (have_snapshot) {
      bundle_->public_state_applier(*bundle_->state, snap_map, api_perspective_);
    }

    if (bt_) {
      std::vector<PublicEvent> events_v(event_list.begin(), event_list.end());
      bt_->observe_public_event(actor, action, events_v);
    }
    ++ply_count_;
  }

  // Test/integration helper: apply an action on the truth state and ALSO
  // return the public-event trace (events + public_snapshot) for a given
  // perspective. This is what selfplay's trace machinery records per-ply,
  // exposed in a step-driven form so tests can drive a ground-truth session
  // and forward the resulting trace to a separate API/AI session.
  //
  // The trace dict shape matches `observation_trace[i]` from
  // run_selfplay_episode: {"events": [...], "public_snapshot": {...}}.
  // For deterministic games (no extractor) returns empty list / dict.
  py::dict apply_action_with_trace(ActionId action, int perspective) {
    std::vector<std::pair<std::string, AnyMap>> event_list;
    AnyMap snap_map;
    bool have_extractor = false;
    {
      py::gil_scoped_release release;
      const int actor = bundle_->state->current_player();
      std::unique_ptr<IGameState> state_before = bundle_->state->clone_state();
      bundle_->rules->do_action_fast(*bundle_->state, action, step_rng_);
      if (bundle_->public_event_extractor) {
        have_extractor = true;
        PublicEventTrace trace = bundle_->public_event_extractor(
            *state_before, action, *bundle_->state, perspective);
        for (auto& ev : trace.events)
          event_list.emplace_back(std::move(ev.first), std::move(ev.second));
        snap_map = std::move(trace.public_snapshot);
      }
      if (bt_) {
        tracker_observe(*bt_, *bundle_, *state_before, action, *bundle_->state,
                        actor);
      }
      if (!external_obs_mode_) {
        const int n = static_cast<int>(ai_views_.size());
        for (int p = 0; p < n; ++p) {
          advance_ai_view_(p, *state_before, action);
        }
      }
      ++ply_count_;
    }
    py::dict out;
    py::list ev_list;
    for (const auto& [kind, payload] : event_list) {
      py::dict e;
      e["kind"] = kind;
      py::dict p;
      for (const auto& [pk, pv] : payload) p[py::cast(pk)] = any_to_py(pv);
      e["payload"] = p;
      ev_list.append(e);
    }
    out["events"] = ev_list;
    py::dict snap;
    if (have_extractor) {
      for (const auto& [k, v] : snap_map) snap[py::cast(k)] = any_to_py(v);
    }
    out["public_snapshot"] = snap;
    return out;
  }

  // Test/integration helper: extract the initial observation for a given
  // perspective from the truth state, the way the partner-side server would
  // before sending it to the AI. Two-section wire shape, structurally
  // identical to per-ply snapshot:
  //   {"public_snapshot": <walker viz::serialize_public_snapshot(state, perspective)>,
  //    "tracker_init":    <tracker.pack_init_payload(state, perspective)>}
  // The public_snapshot half carries every (idx, value) pair where
  // viz[idx, perspective]=1 plus the full perspective viz slice;
  // perspective-internal tracker bootstrap rides tracker_init.
  py::dict extract_initial_observation(int perspective) {
    AnyMap pub;
    AnyMap tk_init;
    {
      py::gil_scoped_release release;
      viz::serialize_public_snapshot(
          *bundle_->state, bundle_->state->schema_ref(), perspective, pub);
      if (bt_) {
        tk_init = bt_->pack_init_payload(*bundle_->state, perspective);
      }
    }
    py::dict pub_d;
    for (const auto& [k, v] : pub) pub_d[py::cast(k)] = any_to_py(v);
    py::dict tk_d;
    for (const auto& [k, v] : tk_init) tk_d[py::cast(k)] = any_to_py(v);
    py::dict out;
    out["public_snapshot"] = pub_d;
    out["tracker_init"] = tk_d;
    return out;
  }

  // Partner-provided initial observation. Two-section wire shape mirrors
  // per-ply snapshot: `public_snapshot` is wholesale-applied via
  // `viz::apply_public_snapshot` (same primitive per-ply uses);
  // `tracker_init` is handed to `tracker.init` so it can write any
  // perspective-internal tracker memory. The session's viz=0 slots are
  // never touched here — sim-entry `randomize_unseen` on a clone is the
  // only path that reads them.
  void apply_initial_observation(int perspective_player, py::dict initial_obs) {
    AnyMap obs_map = py_dict_to_any_map(initial_obs);
    auto pub_it = obs_map.find("public_snapshot");
    auto tk_it = obs_map.find("tracker_init");
    if (pub_it == obs_map.end()) {
      throw std::runtime_error(
          "apply_initial_observation: missing 'public_snapshot' field");
    }
    if (pub_it->second.type() != typeid(AnyMap)) {
      throw std::runtime_error(
          "apply_initial_observation: 'public_snapshot' must be an AnyMap");
    }
    const AnyMap& pub = std::any_cast<const AnyMap&>(pub_it->second);
    AnyMap tk_payload;
    if (tk_it != obs_map.end()) {
      if (tk_it->second.type() != typeid(AnyMap)) {
        throw std::runtime_error(
            "apply_initial_observation: 'tracker_init' must be an AnyMap");
      }
      tk_payload = std::any_cast<const AnyMap&>(tk_it->second);
    }
    py::gil_scoped_release release;
    external_obs_mode_ = true;
    api_perspective_ = perspective_player;
    viz::apply_public_snapshot(
        *bundle_->state, bundle_->state->schema_ref(),
        perspective_player, pub);
    if (bt_) {
      bt_->init(*bundle_->state, perspective_player, tk_payload);
    }
  }

  // Return the belief tracker's serialized state as a dict. Canonical form:
  // two trackers with semantically identical beliefs return equal dicts.
  // Empty dict if no tracker or tracker holds no explicit state.
  py::dict get_belief_snapshot() {
    py::dict out;
    if (!bt_) return out;
    AnyMap m = bt_->serialize();
    for (const auto& [k, v] : m) out[py::cast(k)] = any_to_py(v);
    return out;
  }

  void configure_tail_solve(bool enabled, int depth_limit, std::int64_t node_budget) {
    ts_enabled_ = enabled;
    ts_depth_limit_ = depth_limit;
    ts_node_budget_ = node_budget;
  }

  py::dict get_ai_action(int simulations, double temperature,
                         bool cover_root_edges = false,
                         std::string opponent_selection = "puct",
                         double temperature_initial = -1.0,
                         double temperature_final = -1.0,
                         int temperature_decay_plies = 0) {
    const auto opp_sel = parse_opponent_selection(opponent_selection);
    py::gil_scoped_release release;

    // Resolve effective temperature from schedule + current ply. When
    // initial/final < 0 the schedule is disabled and `temperature` is
    // used as-is (back-compat with callers that don't pass schedule).
    search::TemperatureSchedule t_sched{};
    if (temperature_initial >= 0.0 || temperature_final >= 0.0) {
      t_sched.enabled = true;
      if (temperature_initial >= 0.0) {
        t_sched.has_initial = true;
        t_sched.initial = temperature_initial;
      }
      if (temperature_final >= 0.0) {
        t_sched.has_final = true;
        t_sched.final_ = temperature_final;
      }
      t_sched.decay_plies = temperature_decay_plies;
    }
    const double effective_temperature = search::resolve_linear_temperature(
        t_sched, temperature, static_cast<int>(ply_count_));

    const IGameRules& rules = filtered_rules_ ? *filtered_rules_ : *bundle_->rules;
    const int cp = bundle_->state->current_player();

    // Select the search state / evaluator / tracker.
    // External-obs mode (AI API): bundle_->state IS the AI view.
    // GameSession mode: route through perspective-specific ai_views_[cp].
    const IGameState* search_state = nullptr;
    const search::IPolicyValueEvaluator* eval_ptr = nullptr;
    IBeliefTracker* search_bt = nullptr;
    if (external_obs_mode_) {
      search_state = bundle_->state.get();
      eval_ptr = evaluator_.get();
      search_bt = bt_;
    } else {
      if (cp < 0 || cp >= static_cast<int>(ai_views_.size()) || !ai_views_[cp]) {
        throw std::runtime_error(
            "GameSession.get_ai_action: ai_view for current player not initialized");
      }
      search_state = ai_views_[cp].get();
      eval_ptr = ai_evaluators_[cp].get();
      search_bt = ai_trackers_[cp].get();
    }

    const auto legal = rules.legal_actions(*search_state);
    if (legal.empty()) {
      py::gil_scoped_acquire acquire;
      return py::dict();
    }

    if (!eval_ptr) {
      throw std::runtime_error(
          "GameSession.get_ai_action: no model loaded — create session with model_path");
    }

    // ISMCTS: root-sampling hidden info + DAG per-acting-player keying.
    search::NetMctsConfig mcts_cfg{};
    mcts_cfg.simulations = simulations;
    mcts_cfg.c_puct = 1.4f;
    mcts_cfg.cover_root_edges = cover_root_edges;
    mcts_cfg.opponent_selection = opp_sel;
    if (search_bt) {
      mcts_cfg.root_belief_tracker = search_bt;
    }
    if (bundle_->belief_feature_extractor && belief_evaluator_) {
      mcts_cfg.belief_extractor = bundle_->belief_feature_extractor.get();
      mcts_cfg.belief_evaluator = belief_evaluator_.get();
    }
    // Sim-tracker descent maintenance — events-only flavor.
    if (bundle_->events_only_extractor) {
      mcts_cfg.events_only_extractor = bundle_->events_only_extractor;
    }

    if (ts_enabled_ && bundle_->tail_solver) {
      int ply = static_cast<int>(ply_count_);
      bool try_ts = false;
      if (bundle_->tail_solve_trigger) {
        try_ts = bundle_->tail_solve_trigger(*search_state, ply);
      } else {
        try_ts = true;
      }
      if (try_ts) {
        mcts_cfg.tail_solve_enabled = true;
        mcts_cfg.tail_solve_config.depth_limit = ts_depth_limit_;
        mcts_cfg.tail_solve_config.node_budget = ts_node_budget_;
        mcts_cfg.tail_solver = bundle_->tail_solver.get();
      }
    }

    search::NetMcts mcts(mcts_cfg);
    search::NetMctsStats stats{};
    const std::uint64_t mcts_seed = board_ai::rng::derive_subseed(
        seed_, "session.mcts", static_cast<std::uint64_t>(ply_count_));
    mcts.search_root(*search_state, rules, *bundle_->value_model,
                      *eval_ptr, &stats, mcts_seed);

    // Deterministic action_seed: any non-deterministic component (e.g.
    // wall-clock) here would make argmax tie-breaks flaky across runs
    // even at temperature=0. Derive from (seed_, ply_count_) instead so
    // the session's action sequence is fully reproducible from its seed.
    std::uint64_t action_seed = mcts_seed ^ 0xBF58476D1CE4E5B9ULL;
    ActionId chosen = search::select_action_from_visits(
        stats.root_actions, stats.root_action_visits,
        effective_temperature, action_seed, legal[0]);

    py::gil_scoped_acquire acquire;
    py::dict out;
    out["action"] = chosen;
    out["action_info"] = get_action_info(chosen);
    py::dict st;
    st["simulations"] = stats.simulations_done;
    st["best_value"] = stats.best_action_value;
    py::list rv;
    for (double v : stats.root_values) rv.append(v);
    st["root_values"] = rv;
    py::dict wm;
    for (size_t ei = 0; ei < stats.root_actions.size(); ++ei) {
      if (ei < stats.root_edge_values.size()) {
        py::list vals;
        for (double v : stats.root_edge_values[ei]) vals.append(v);
        wm[py::cast(static_cast<int>(stats.root_actions[ei]))] = vals;
      }
    }
    st["action_values"] = wm;
    // Expose root visit distribution so tests can compare the full
    // MCTS policy across paths (selfplay vs API), not just argmax.
    // This is the signal that would catch a subtle info leak biasing
    // one path's priors without flipping the top pick.
    py::list root_actions_py;
    py::list root_visits_py;
    for (size_t ei = 0; ei < stats.root_actions.size(); ++ei) {
      root_actions_py.append(static_cast<int>(stats.root_actions[ei]));
      int v = ei < stats.root_action_visits.size() ? stats.root_action_visits[ei] : 0;
      root_visits_py.append(v);
    }
    st["root_actions"] = root_actions_py;
    st["root_action_visits"] = root_visits_py;
    st["tail_solved"] = stats.tail_solved;
    st["tail_solve_value"] = stats.tail_solve_value;
    st["tail_solve_attempted"] = stats.tail_solve_attempted;
    st["tail_solve_completed"] = stats.tail_solve_completed;
    st["tail_solve_elapsed_ms"] = stats.tail_solve_elapsed_ms;
    // Outcome encoded so the web layer can render "0% 残局已求解 / 100% / 平局"
    // without re-running the solver. Mirrors TailSolveOutcome enum integer.
    st["tail_solve_outcome"] = static_cast<int>(stats.tail_solve_outcome);
    st["dag_reuse_hits"] = stats.dag_reuse_hits;
    st["expanded_nodes"] = stats.expanded_nodes;
    st["simulations"] = stats.simulations_done;
    out["stats"] = st;
    return out;
  }

  py::dict get_heuristic_action() {
    if (!bundle_->heuristic_picker) {
      throw std::runtime_error("get_heuristic_action: game '" + game_id_ + "' has no heuristic_picker registered");
    }

    ActionId chosen;
    {
      py::gil_scoped_release release;
      auto hr = bundle_->heuristic_picker(
          *bundle_->state, *bundle_->rules, seed_ ^ 0xDEAD);
      if (hr.actions.empty()) {
        py::gil_scoped_acquire acquire;
        return py::dict();
      }

      double max_score = *std::max_element(hr.scores.begin(), hr.scores.end());
      chosen = hr.actions[0];
      for (size_t i = 0; i < hr.scores.size(); ++i) {
        if (hr.scores[i] >= max_score - 1e-9) {
          chosen = hr.actions[i];
          break;
        }
      }
    }

    py::dict out;
    out["action"] = chosen;
    out["action_info"] = get_action_info(chosen);
    return out;
  }

  py::dict apply_ai_action(int simulations, double temperature) {
    auto result = get_ai_action(simulations, temperature);
    if (result.contains("action")) {
      apply_action(py::cast<ActionId>(result["action"]));
    }
    return result;
  }

  // Test-only: run MCTS with the on_sim_step callback enabled, capturing
  // the cloned sim_tracker's serialize() at every descent step of every
  // sim. Returns the root tracker snapshot taken at sim entry (before any
  // descent step) plus a list of (sim, depth, sim_tracker_snapshot) tuples.
  // Used by test_sim_tracker_descent_maintained to assert the sim_tracker
  // actually evolves during descent (not frozen at root).
  //
  // This routes through the same `get_ai_action` MCTS config plumbing —
  // no path divergence — but with `on_sim_step` set to a recorder. The
  // returned action is discarded; callers care only about the traces.
  py::dict _debug_run_mcts_with_sim_traces(int simulations) {
    const IGameRules& rules = filtered_rules_ ? *filtered_rules_ : *bundle_->rules;
    const int cp = bundle_->state->current_player();

    const IGameState* search_state = nullptr;
    const search::IPolicyValueEvaluator* eval_ptr = nullptr;
    IBeliefTracker* search_bt = nullptr;
    if (external_obs_mode_) {
      search_state = bundle_->state.get();
      eval_ptr = evaluator_.get();
      search_bt = bt_;
    } else {
      if (cp < 0 || cp >= static_cast<int>(ai_views_.size()) || !ai_views_[cp]) {
        throw std::runtime_error(
            "_debug_run_mcts_with_sim_traces: ai_view for current player not initialized");
      }
      search_state = ai_views_[cp].get();
      eval_ptr = ai_evaluators_[cp].get();
      search_bt = ai_trackers_[cp].get();
    }

    if (!eval_ptr) {
      throw std::runtime_error(
          "_debug_run_mcts_with_sim_traces: no model loaded");
    }
    if (!search_bt) {
      throw std::runtime_error(
          "_debug_run_mcts_with_sim_traces: game has no belief tracker registered");
    }

    // Capture root tracker snapshot BEFORE the search call — this is
    // what every sim's clone starts from after randomize_unseen seeds it.
    AnyMap root_snapshot = search_bt->serialize();

    // Recorder: appends (sim_index, step_index, serialize()) tuples.
    struct Record { int sim; int step; AnyMap snap; };
    std::vector<Record> records;

    search::NetMctsConfig mcts_cfg{};
    mcts_cfg.simulations = simulations;
    mcts_cfg.c_puct = 1.4f;
    mcts_cfg.root_belief_tracker = search_bt;
    if (bundle_->events_only_extractor) {
      mcts_cfg.events_only_extractor = bundle_->events_only_extractor;
    }
    mcts_cfg.on_sim_step = [&records](int sim, int step,
                                       const IGameState&,
                                       const IBeliefTracker* tr) {
      if (!tr) return;
      records.push_back(Record{sim, step, tr->serialize()});
    };

    {
      py::gil_scoped_release release;
      search::NetMcts mcts(mcts_cfg);
      search::NetMctsStats stats{};
      const std::uint64_t mcts_seed = board_ai::rng::derive_subseed(
          seed_, "session.mcts.debug", static_cast<std::uint64_t>(ply_count_));
      mcts.search_root(*search_state, rules, *bundle_->value_model,
                        *eval_ptr, &stats, mcts_seed);
    }

    py::dict out;
    py::dict root_dict;
    for (const auto& [k, v] : root_snapshot) root_dict[py::cast(k)] = any_to_py(v);
    out["root_tracker"] = root_dict;

    py::list traces;
    for (const auto& r : records) {
      py::dict entry;
      entry["sim"] = r.sim;
      entry["step"] = r.step;
      py::dict snap;
      for (const auto& [k, v] : r.snap) snap[py::cast(k)] = any_to_py(v);
      entry["tracker"] = snap;
      traces.append(entry);
    }
    out["sim_traces"] = traces;
    return out;
  }

 private:
  std::string game_id_;
  std::uint64_t seed_;
  std::string model_path_;
  std::unique_ptr<GameBundle> bundle_;
  std::unique_ptr<infer::OnnxPolicyValueEvaluator> evaluator_;
  std::unique_ptr<infer::OnnxBeliefEvaluator> belief_evaluator_;
  std::unique_ptr<runtime::FilteredRulesWrapper> filtered_rules_;
  IBeliefTracker* bt_ = nullptr;
  // Perspective seat that bt_ (the session-level tracker on bundle_->state)
  // is bound to. Set by apply_initial_observation; defaults to 0 when the
  // session was constructed without an external observation. Used as the
  // observer argument to randomize_unseen on bundle_->state in
  // external_obs_mode.
  int api_perspective_ = 0;
  std::size_t ply_count_ = 0;
  // GT step rng. Seeded directly from seed_ (no subseed derivation) so
  // selfplay's gt_step rng with the same episode_seed produces identical
  // truth draws — `test_public_snapshot_round_trip` replays selfplay
  // traces through GameSession.apply_action and depends on this match.
  std::mt19937_64 step_rng_{seed_};
  bool ts_enabled_ = false;
  int ts_depth_limit_ = 10;
  std::int64_t ts_node_budget_ = 200000;

  // Per-perspective AI views. Populated by the constructor via the
  // initial-observation protocol (unless external_obs_mode_ flips on
  // first — see comments at init_ai_views_). get_ai_action routes MCTS
  // through these so searches never read hidden fields from bundle_->state.
  std::vector<std::unique_ptr<IGameState>> ai_views_;
  std::vector<std::unique_ptr<IBeliefTracker>> ai_trackers_;
  std::vector<std::unique_ptr<IFeatureEncoder>> ai_encoders_;
  std::vector<std::unique_ptr<infer::OnnxPolicyValueEvaluator>> ai_evaluators_;
  bool external_obs_mode_ = false;
};

py::dict test_belief_tracker_py(
    const std::string& game_id,
    std::uint64_t seed,
    int plies,
    int randomize_trials) {
  py::gil_scoped_release release;

  auto bundle = GameRegistry::instance().create_game(game_id, seed);
  if (!bundle.belief_tracker) {
    py::gil_scoped_acquire acquire;
    throw std::runtime_error("test_belief_tracker: game must have belief_tracker");
  }

  auto state = bundle.state->clone_state();
  IBeliefTracker* bt = bundle.belief_tracker.get();
  tracker_init(*bt, bundle, *state, 0);

  std::mt19937_64 action_rng(seed);
  std::mt19937_64 step_rng(
      board_ai::rng::derive_subseed(seed, "test_belief_tracker.gt_step"));
  int actual_plies = 0;
  for (int i = 0; i < plies && !state->is_terminal(); ++i) {
    auto legal = bundle.rules->legal_actions(*state);
    if (legal.empty()) break;
    const size_t idx = action_rng() % legal.size();
    ActionId chosen = legal[idx];
    auto state_before = state->clone_state();
    bundle.rules->do_action_fast(*state, chosen, step_rng);
    tracker_observe(*bt, bundle, *state_before, chosen, *state, 0);
    actual_plies += 1;
  }

  auto extract_deck_ids = [&](const IGameState& gs) -> std::vector<int> {
    if (!bundle.state_serializer) return {};
    auto m = bundle.state_serializer(gs);
    std::vector<int> result;
    auto it = m.find("_test_all_deck_ids");
    if (it != m.end()) {
      result = std::any_cast<std::vector<int>>(it->second);
    }
    return result;
  };

  auto extract_tableau_ids = [&](const IGameState& gs) -> std::vector<int> {
    if (!bundle.state_serializer) return {};
    auto m = bundle.state_serializer(gs);
    std::vector<int> result;
    auto it = m.find("_test_tableau_ids");
    if (it != m.end()) {
      result = std::any_cast<std::vector<int>>(it->second);
    }
    return result;
  };

  std::vector<int> original_deck = extract_deck_ids(*state);
  std::vector<int> tableau_ids = extract_tableau_ids(*state);

  AnyMap belief_snapshot = bt->serialize();
  AnyMap original_state_map;
  if (bundle.state_serializer) {
    original_state_map = bundle.state_serializer(*state);
  }

  std::vector<std::vector<int>> trial_decks;
  std::vector<AnyMap> trial_state_maps;
  trial_decks.reserve(static_cast<size_t>(randomize_trials));
  trial_state_maps.reserve(static_cast<size_t>(randomize_trials));
  for (int t = 0; t < randomize_trials; ++t) {
    auto clone = state->clone_state();
    std::mt19937_64 trial_rng(board_ai::rng::derive_subseed(
        seed, "test_belief_tracker.trial", static_cast<std::uint64_t>(t + 1)));
    bt->randomize_unseen(*clone, /*observer=*/0, trial_rng);
    trial_decks.push_back(extract_deck_ids(*clone));
    if (bundle.state_serializer) {
      trial_state_maps.push_back(bundle.state_serializer(*clone));
    } else {
      trial_state_maps.push_back({});
    }
  }

  py::gil_scoped_acquire acquire;
  py::dict out;
  out["plies"] = actual_plies;
  out["original_deck"] = original_deck;
  out["tableau_cards"] = tableau_ids;
  py::list trials;
  for (const auto& d : trial_decks) {
    trials.append(py::cast(d));
  }
  out["trial_decks"] = trials;

  py::dict bs;
  for (const auto& [k, v] : belief_snapshot) bs[py::cast(k)] = any_to_py(v);
  out["belief_snapshot"] = bs;

  py::dict orig_state;
  for (const auto& [k, v] : original_state_map) orig_state[py::cast(k)] = any_to_py(v);
  out["original_state"] = orig_state;

  py::list trial_states;
  for (const auto& m : trial_state_maps) {
    py::dict d;
    for (const auto& [k, v] : m) d[py::cast(k)] = any_to_py(v);
    trial_states.append(d);
  }
  out["trial_states"] = trial_states;
  return out;
}

}  // namespace

PYBIND11_MODULE(dinoboard_engine, m) {
  m.doc() = "DinoBoard C++ engine bindings";

  m.def("run_selfplay_episode", &run_selfplay_episode_py,
      py::arg("game_id"),
      py::arg("seed"),
      py::arg("model_path") = "",
      py::arg("simulations") = 200,
      py::arg("c_puct") = 1.4f,
      py::arg("temperature") = 1.0,
      py::arg("dirichlet_alpha") = 0.3,
      py::arg("dirichlet_epsilon") = 0.25,
      py::arg("dirichlet_on_first_n_plies") = 30,
      py::arg("max_game_plies") = 500,
      py::arg("tail_solve_enabled") = false,
      py::arg("tail_solve_depth_limit") = 5,
      py::arg("tail_solve_node_budget") = 10000000LL,
      py::arg("tail_solve_margin_weight") = 0.0f,
      py::arg("temperature_initial") = -1.0,
      py::arg("temperature_final") = -1.0,
      py::arg("temperature_decay_plies") = 0,
      py::arg("heuristic_guidance_ratio") = 0.0,
      py::arg("heuristic_temperature") = 0.0,
      py::arg("training_filter_ratio") = 1.0,
      py::arg("trace_perspective") = -1,
      py::arg("opponent_selection") = std::string("puct"),
      py::arg("belief_model_path") = std::string(""));

  m.def("run_selfplay_episode_pool", &run_selfplay_episode_pool_py,
      py::arg("game_id"),
      py::arg("seed"),
      py::arg("model_paths"),
      py::arg("simulations") = 200,
      py::arg("c_puct") = 1.4f,
      py::arg("temperature") = 1.0,
      py::arg("dirichlet_alpha") = 0.3,
      py::arg("dirichlet_epsilon") = 0.25,
      py::arg("dirichlet_on_first_n_plies") = 30,
      py::arg("max_game_plies") = 500,
      py::arg("tail_solve_enabled") = false,
      py::arg("tail_solve_depth_limit") = 5,
      py::arg("tail_solve_node_budget") = 10000000LL,
      py::arg("tail_solve_margin_weight") = 0.0f,
      py::arg("temperature_initial") = -1.0,
      py::arg("temperature_final") = -1.0,
      py::arg("temperature_decay_plies") = 0,
      py::arg("heuristic_guidance_ratio") = 0.0,
      py::arg("heuristic_temperature") = 0.0,
      py::arg("training_filter_ratio") = 1.0,
      py::arg("trace_perspective") = -1,
      py::arg("opponent_selection") = std::string("puct"),
      py::arg("belief_model_path") = std::string(""));

  m.def("run_arena_match", &run_arena_match_py,
      py::arg("game_id"),
      py::arg("seed"),
      py::arg("model_paths"),
      py::arg("simulations_list"),
      py::arg("temperature") = 0.0,
      py::arg("max_game_plies") = 500,
      py::arg("tail_solve") = false,
      py::arg("tail_solve_depth_limit") = 10,
      py::arg("tail_solve_node_budget") = std::int64_t{200000},
      py::arg("tail_solve_margin_weight") = 0.0f,
      py::arg("opponent_selection_list") = std::vector<std::string>{},
      py::arg("belief_model_path") = std::string(""));

  m.def("run_constrained_eval_vs_heuristic", &run_constrained_eval_vs_heuristic_py,
      py::arg("game_id"),
      py::arg("seed"),
      py::arg("model_path"),
      py::arg("simulations") = 200,
      py::arg("model_is_player") = 0,
      py::arg("constrained") = true,
      py::arg("heuristic_temperature") = 0.0,
      py::arg("opponent_selection") = std::string("puct"),
      py::arg("belief_model_path") = std::string(""));

  m.def("run_heuristic_episode", &run_heuristic_episode_py,
      py::arg("game_id"),
      py::arg("seed"),
      py::arg("temperature") = 0.0,
      py::arg("max_game_plies") = 200);

  m.def("encode_state", &encode_state_py,
      py::arg("game_id"),
      py::arg("seed") = 0xC0FFEE);

  m.def("encode_state_for_perspective", &encode_state_for_perspective_py,
      py::arg("game_id"),
      py::arg("seed") = 0xC0FFEE,
      py::arg("encode_player") = 0,
      py::arg("tracker_perspective") = -1,
      py::arg("event_actions") = py::list());

  m.def("tail_solve", &tail_solve_py,
      py::arg("game_id"),
      py::arg("seed"),
      py::arg("perspective_player") = 0,
      py::arg("depth_limit") = 10,
      py::arg("node_budget") = 200000LL);

  m.def("test_belief_tracker", &test_belief_tracker_py,
      py::arg("game_id"),
      py::arg("seed"),
      py::arg("plies") = 20,
      py::arg("randomize_trials") = 10);

  m.def("game_metadata", [](const std::string& game_id) -> py::dict {
    py::gil_scoped_release release;
    auto bundle = GameRegistry::instance().create_game(game_id, 0);
    const int num_players = bundle.state->num_players();
    const int action_space = bundle.encoder->action_space();
    const int feature_dim = bundle.encoder->feature_dim();
    const bool has_public_state_applier = static_cast<bool>(bundle.public_state_applier);
    const bool has_tail_solver = static_cast<bool>(bundle.tail_solver);
    const bool has_tail_solve_trigger = static_cast<bool>(bundle.tail_solve_trigger);
    const bool has_belief_extractor =
        static_cast<bool>(bundle.belief_feature_extractor);
    const int belief_feature_dim =
        has_belief_extractor ? bundle.belief_feature_extractor->feature_dim() : 0;
    const int belief_logit_count =
        has_belief_extractor ? bundle.belief_feature_extractor->output_logit_count()
                             : 0;
    py::gil_scoped_acquire acquire;
    py::dict out;
    out["num_players"] = num_players;
    out["action_space"] = action_space;
    out["feature_dim"] = feature_dim;
    // Capability flag: snapshot-path games register a public_state_applier;
    // fully-public games (tictactoe, quoridor) don't. The REST AI API uses
    // this to dispatch between the action_id-only path and the full
    // apply_observation path. extract_initial_observation /
    // apply_initial_observation are walker-driven and available for every
    // game, so no separate has_initial_observation_applier flag is needed.
    out["has_public_state_applier"] = has_public_state_applier;
    out["has_tail_solver"] = has_tail_solver;
    out["has_tail_solve_trigger"] = has_tail_solve_trigger;
    out["has_belief_extractor"] = has_belief_extractor;
    out["belief_feature_dim"] = belief_feature_dim;
    out["belief_logit_count"] = belief_logit_count;
    return out;
  }, py::arg("game_id"));

  m.def("available_games", []() -> std::vector<std::string> {
    py::gil_scoped_release release;
    return GameRegistry::instance().game_ids();
  });

  py::class_<GameSessionWrapper>(m, "GameSession")
      .def(py::init<const std::string&, std::uint64_t, const std::string&, bool,
                    const std::string&>(),
           py::arg("game_id"),
           py::arg("seed") = 0xC0FFEE,
           py::arg("model_path") = "",
           py::arg("use_filter") = false,
           py::arg("belief_model_path") = std::string(""))
      .def_property_readonly("is_terminal", &GameSessionWrapper::is_terminal)
      .def_property_readonly("is_turn_start", &GameSessionWrapper::is_turn_start)
      .def_property_readonly("current_player", &GameSessionWrapper::current_player)
      .def_property_readonly("winner", &GameSessionWrapper::winner)
      .def_property_readonly("num_players", &GameSessionWrapper::num_players)
      .def_property_readonly("game_id", &GameSessionWrapper::game_id)
      .def("state_hash_for_perspective",
           &GameSessionWrapper::state_hash_for_perspective,
           py::arg("player"))
      .def_property_readonly("step_count", &GameSessionWrapper::step_count)
      .def("get_state_dict", &GameSessionWrapper::get_state_dict)
      .def("get_action_info", &GameSessionWrapper::get_action_info)
      .def("apply_public_snapshot", &GameSessionWrapper::apply_public_snapshot,
           py::arg("snapshot"), py::arg("receiver_seat"))
      .def("get_legal_actions", &GameSessionWrapper::get_legal_actions)
      .def("get_all_legal_actions", &GameSessionWrapper::get_all_legal_actions)
      .def("apply_action", &GameSessionWrapper::apply_action)
      .def("apply_action_with_trace", &GameSessionWrapper::apply_action_with_trace,
           py::arg("action"), py::arg("perspective"))
      .def("extract_initial_observation",
           &GameSessionWrapper::extract_initial_observation,
           py::arg("perspective"))
      .def("apply_observation", &GameSessionWrapper::apply_observation,
           py::arg("action"),
           py::arg("events") = py::list(),
           py::arg("public_snapshot") = py::dict())
      .def("apply_initial_observation", &GameSessionWrapper::apply_initial_observation,
           py::arg("perspective_player"), py::arg("initial_observation"))
      .def("get_belief_snapshot", &GameSessionWrapper::get_belief_snapshot)
      .def("get_ai_action", &GameSessionWrapper::get_ai_action,
           py::arg("simulations") = 200,
           py::arg("temperature") = 0.0,
           py::arg("cover_root_edges") = false,
           py::arg("opponent_selection") = std::string("puct"),
           py::arg("temperature_initial") = -1.0,
           py::arg("temperature_final") = -1.0,
           py::arg("temperature_decay_plies") = 0)
      .def("get_heuristic_action", &GameSessionWrapper::get_heuristic_action)
      .def("configure_tail_solve", &GameSessionWrapper::configure_tail_solve,
           py::arg("enabled"),
           py::arg("depth_limit") = 10,
           py::arg("node_budget") = 200000LL)
      .def("apply_ai_action", &GameSessionWrapper::apply_ai_action,
           py::arg("simulations") = 200,
           py::arg("temperature") = 0.0)
      .def("_debug_run_mcts_with_sim_traces",
           &GameSessionWrapper::_debug_run_mcts_with_sim_traces,
           py::arg("simulations") = 16);
}
