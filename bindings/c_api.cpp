#include "c_api.h"

#include <algorithm>
#include <any>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <iomanip>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "../engine/core/action_constraint.h"
#include "../engine/core/game_registry.h"
#include "../engine/core/masked_state.h"
#include "../engine/infer/onnx_belief_evaluator.h"
#include "../engine/infer/onnx_policy_value_evaluator.h"
#include "../engine/runtime/selfplay_runner.h"
#include "../engine/search/net_mcts.h"

namespace {

using board_ai::ActionId;
using board_ai::AnyMap;
using board_ai::GameBundle;
using board_ai::GameRegistry;
using board_ai::IBeliefTracker;

constexpr const char* kAbiVersion = "dinoboard-c-abi-0.1";

char* copy_string(const std::string& text) {
  char* out = static_cast<char*>(std::malloc(text.size() + 1));
  if (!out) return nullptr;
  std::memcpy(out, text.c_str(), text.size() + 1);
  return out;
}

void set_error(char** out_error, const std::string& message) {
  if (out_error) *out_error = copy_string(message);
}

std::string json_escape(const std::string& value) {
  std::ostringstream out;
  out << '"';
  for (unsigned char ch : value) {
    switch (ch) {
      case '"': out << "\\\""; break;
      case '\\': out << "\\\\"; break;
      case '\b': out << "\\b"; break;
      case '\f': out << "\\f"; break;
      case '\n': out << "\\n"; break;
      case '\r': out << "\\r"; break;
      case '\t': out << "\\t"; break;
      default:
        if (ch < 0x20) {
          out << "\\u" << std::hex << std::setw(4) << std::setfill('0')
              << static_cast<int>(ch) << std::dec;
        } else {
          out << static_cast<char>(ch);
        }
    }
  }
  out << '"';
  return out.str();
}

template <typename T>
std::string vector_to_json(const std::vector<T>& values) {
  std::ostringstream out;
  out << '[';
  for (std::size_t i = 0; i < values.size(); ++i) {
    if (i) out << ',';
    out << values[i];
  }
  out << ']';
  return out.str();
}

std::string any_to_json(const std::any& value);

template <typename T>
std::string any_vector_to_json(const std::any& value) {
  const auto& values = std::any_cast<const std::vector<T>&>(value);
  return vector_to_json(values);
}

std::string any_map_to_json(const AnyMap& map) {
  std::ostringstream out;
  out << '{';
  bool first = true;
  for (const auto& [key, value] : map) {
    if (!first) out << ',';
    first = false;
    out << json_escape(key) << ':' << any_to_json(value);
  }
  out << '}';
  return out.str();
}

std::string public_events_to_json(const std::vector<board_ai::PublicEvent>& events) {
  std::ostringstream out;
  out << '[';
  for (std::size_t i = 0; i < events.size(); ++i) {
    if (i) out << ',';
    out << "{\"kind\":" << json_escape(events[i].first)
        << ",\"payload\":" << any_map_to_json(events[i].second) << '}';
  }
  out << ']';
  return out.str();
}

std::string any_to_json(const std::any& value) {
  if (!value.has_value()) return "null";
  const std::type_info& t = value.type();
  if (t == typeid(std::string)) return json_escape(std::any_cast<const std::string&>(value));
  if (t == typeid(const char*)) return json_escape(std::any_cast<const char*>(value));
  if (t == typeid(bool)) return std::any_cast<bool>(value) ? "true" : "false";
  if (t == typeid(int)) return std::to_string(std::any_cast<int>(value));
  if (t == typeid(std::int8_t)) return std::to_string(static_cast<int>(std::any_cast<std::int8_t>(value)));
  if (t == typeid(std::uint8_t)) return std::to_string(static_cast<int>(std::any_cast<std::uint8_t>(value)));
  if (t == typeid(std::int16_t)) return std::to_string(std::any_cast<std::int16_t>(value));
  if (t == typeid(std::uint16_t)) return std::to_string(std::any_cast<std::uint16_t>(value));
  if (t == typeid(std::int32_t)) return std::to_string(std::any_cast<std::int32_t>(value));
  if (t == typeid(std::uint32_t)) return std::to_string(std::any_cast<std::uint32_t>(value));
  if (t == typeid(std::int64_t)) return std::to_string(std::any_cast<std::int64_t>(value));
  if (t == typeid(std::uint64_t)) return std::to_string(std::any_cast<std::uint64_t>(value));
  if (t == typeid(float)) return std::to_string(std::any_cast<float>(value));
  if (t == typeid(double)) return std::to_string(std::any_cast<double>(value));
  if (t == typeid(AnyMap)) return any_map_to_json(std::any_cast<const AnyMap&>(value));
  if (t == typeid(std::vector<int>)) return any_vector_to_json<int>(value);
  if (t == typeid(std::vector<double>)) return any_vector_to_json<double>(value);
  if (t == typeid(std::vector<float>)) return any_vector_to_json<float>(value);
  if (t == typeid(std::vector<std::string>)) {
    const auto& values = std::any_cast<const std::vector<std::string>&>(value);
    std::ostringstream out;
    out << '[';
    for (std::size_t i = 0; i < values.size(); ++i) {
      if (i) out << ',';
      out << json_escape(values[i]);
    }
    out << ']';
    return out.str();
  }
  if (t == typeid(std::vector<AnyMap>)) {
    const auto& values = std::any_cast<const std::vector<AnyMap>&>(value);
    std::ostringstream out;
    out << '[';
    for (std::size_t i = 0; i < values.size(); ++i) {
      if (i) out << ',';
      out << any_map_to_json(values[i]);
    }
    out << ']';
    return out.str();
  }
  if (t == typeid(std::vector<board_ai::PublicEvent>)) {
    return public_events_to_json(std::any_cast<const std::vector<board_ai::PublicEvent>&>(value));
  }
  return json_escape("<unsupported-any>");
}

void tracker_init(IBeliefTracker& tracker, const GameBundle&,
                  board_ai::IGameState& state, int perspective) {
  tracker.init(state, perspective, {});
}

board_ai::search::OpponentSelection parse_opponent_selection() {
  return board_ai::search::OpponentSelection::kPuct;
}

class CSession {
 public:
  CSession(const std::string& game_id, const std::string& model_path,
           std::uint64_t seed, bool use_filter)
      : game_id_(game_id), seed_(seed), model_path_(model_path) {
    bundle_ = std::make_unique<GameBundle>(
        GameRegistry::instance().create_game(game_id, seed));
    evaluator_ = std::make_unique<board_ai::infer::OnnxPolicyValueEvaluator>(
        model_path, bundle_->encoder.get());
    if (!evaluator_->is_ready()) {
      throw std::runtime_error("failed to load model: " + evaluator_->last_error());
    }
    if (use_filter && bundle_->training_action_filter) {
      filtered_rules_ = std::make_unique<board_ai::runtime::FilteredRulesWrapper>(
          *bundle_->rules, bundle_->training_action_filter);
    }
    init_ai_views();
  }

  std::string decide_json(int simulations, double temperature,
                          bool cover_root_edges) {
    const board_ai::IGameRules& rules = filtered_rules_ ? *filtered_rules_ : *bundle_->rules;
    const int cp = bundle_->state->current_player();
    if (cp < 0 || cp >= static_cast<int>(ai_views_.size()) || !ai_views_[cp]) {
      throw std::runtime_error("current-player AI view is not initialized");
    }
    const board_ai::IGameState& search_state = *ai_views_[cp];
    const auto legal = rules.legal_actions(search_state);
    if (legal.empty()) {
      return "{\"action_id\":null,\"action_info\":{},\"stats\":{\"root_actions\":[],\"root_action_visits\":[]}}";
    }

    board_ai::search::NetMctsConfig cfg{};
    cfg.simulations = simulations;
    cfg.c_puct = 1.4f;
    cfg.cover_root_edges = cover_root_edges;
    cfg.opponent_selection = parse_opponent_selection();
    if (ai_trackers_[cp]) cfg.root_belief_tracker = ai_trackers_[cp].get();
    if (bundle_->events_only_extractor) cfg.events_only_extractor = bundle_->events_only_extractor;
    if (bundle_->tail_solver) {
      cfg.tail_solve_enabled = true;
      cfg.tail_solve_config.depth_limit = 6;
      cfg.tail_solve_config.node_budget = 100000;
      cfg.tail_solver = bundle_->tail_solver.get();
    }

    board_ai::search::NetMctsStats stats{};
    board_ai::search::NetMcts mcts(cfg);
    const std::uint64_t mcts_seed = seed_ ^ 0x9E3779B97F4A7C15ULL;
    mcts.search_root(search_state, rules, *bundle_->value_model,
                     *ai_evaluators_[cp], &stats, mcts_seed);
    const std::uint64_t action_seed = mcts_seed ^ 0xBF58476D1CE4E5B9ULL;
    ActionId chosen = board_ai::search::select_action_from_visits(
        stats.root_actions, stats.root_action_visits, temperature,
        action_seed, legal[0]);

    std::ostringstream out;
    out << "{\"action_id\":" << chosen
        << ",\"action_info\":" << action_info_json(chosen)
        << ",\"stats\":{"
        << "\"simulations\":" << stats.simulations_done
        << ",\"best_value\":" << stats.best_action_value
        << ",\"root_actions\":" << vector_to_json(stats.root_actions)
        << ",\"root_action_visits\":" << vector_to_json(stats.root_action_visits)
        << ",\"root_values\":" << vector_to_json(stats.root_values)
        << ",\"action_values\":{";
    for (std::size_t i = 0; i < stats.root_actions.size(); ++i) {
      if (i) out << ',';
      out << json_escape(std::to_string(stats.root_actions[i])) << ':';
      if (i < stats.root_edge_values.size()) {
        out << vector_to_json(stats.root_edge_values[i]);
      } else {
        out << "[]";
      }
    }
    out << "},\"tail_solved\":" << (stats.tail_solved ? "true" : "false")
        << ",\"expanded_nodes\":" << stats.expanded_nodes
        << "}}";
    return out.str();
  }

 private:
  void init_ai_views() {
    const int n = bundle_->state->num_players();
    ai_views_.resize(static_cast<std::size_t>(n));
    ai_trackers_.resize(static_cast<std::size_t>(n));
    ai_encoders_.resize(static_cast<std::size_t>(n));
    ai_evaluators_.resize(static_cast<std::size_t>(n));
    for (int p = 0; p < n; ++p) {
      auto extra = GameRegistry::instance().create_game(game_id_, seed_);
      ai_trackers_[p] = std::move(extra.belief_tracker);
      ai_encoders_[p] = std::move(extra.encoder);
      ai_views_[p] = board_ai::make_masked_state(
          *bundle_->state, bundle_->state->schema_ref(), p);
      if (ai_trackers_[p]) {
        tracker_init(*ai_trackers_[p], *bundle_, *ai_views_[p], p);
      }
      ai_evaluators_[p] = std::make_unique<board_ai::infer::OnnxPolicyValueEvaluator>(
          model_path_, ai_encoders_[p].get());
      if (!ai_evaluators_[p]->is_ready()) {
        throw std::runtime_error(
            "failed to load model for ai_view[" + std::to_string(p) +
            "]: " + ai_evaluators_[p]->last_error());
      }
    }
  }

  std::string action_info_json(ActionId action) const {
    if (!bundle_->action_descriptor) return "{}";
    return any_map_to_json(bundle_->action_descriptor(action));
  }

  std::string game_id_;
  std::uint64_t seed_;
  std::string model_path_;
  std::unique_ptr<GameBundle> bundle_;
  std::unique_ptr<board_ai::infer::OnnxPolicyValueEvaluator> evaluator_;
  std::unique_ptr<board_ai::runtime::FilteredRulesWrapper> filtered_rules_;
  std::vector<std::unique_ptr<board_ai::IGameState>> ai_views_;
  std::vector<std::unique_ptr<IBeliefTracker>> ai_trackers_;
  std::vector<std::unique_ptr<board_ai::IFeatureEncoder>> ai_encoders_;
  std::vector<std::unique_ptr<board_ai::infer::OnnxPolicyValueEvaluator>> ai_evaluators_;
};

std::string metadata_json(const std::string& game_id) {
  auto bundle = GameRegistry::instance().create_game(game_id, 0);
  std::ostringstream out;
  out << "{\"game_id\":" << json_escape(game_id)
      << ",\"num_players\":" << bundle.state->num_players()
      << ",\"action_space\":" << bundle.encoder->action_space()
      << ",\"feature_dim\":" << bundle.encoder->feature_dim()
      << ",\"has_public_state_applier\":"
      << (bundle.public_state_applier ? "true" : "false")
      << ",\"has_tail_solver\":" << (bundle.tail_solver ? "true" : "false")
      << ",\"has_belief_tracker\":" << (bundle.belief_tracker ? "true" : "false")
      << "}";
  return out.str();
}

}  // namespace

extern "C" {

DINOBOARD_C_API void dinoboard_string_free(char* value) {
  std::free(value);
}

DINOBOARD_C_API const char* dinoboard_abi_version(void) {
  return kAbiVersion;
}

DINOBOARD_C_API char* dinoboard_available_games_json(void) {
  try {
    auto ids = GameRegistry::instance().game_ids();
    std::sort(ids.begin(), ids.end());
    std::ostringstream out;
    out << '[';
    for (std::size_t i = 0; i < ids.size(); ++i) {
      if (i) out << ',';
      out << json_escape(ids[i]);
    }
    out << ']';
    return copy_string(out.str());
  } catch (const std::exception& e) {
    return copy_string(std::string("{\"error\":") + json_escape(e.what()) + "}");
  }
}

DINOBOARD_C_API char* dinoboard_game_metadata_json(const char* game_id) {
  try {
    if (!game_id) return copy_string("{\"error\":\"game_id is null\"}");
    return copy_string(metadata_json(game_id));
  } catch (const std::exception& e) {
    return copy_string(std::string("{\"error\":") + json_escape(e.what()) + "}");
  }
}

DINOBOARD_C_API char* dinoboard_action_info_json(const char* game_id, int action_id) {
  try {
    if (!game_id) return copy_string("{\"error\":\"game_id is null\"}");
    auto bundle = GameRegistry::instance().create_game(game_id, 0);
    if (!bundle.action_descriptor) return copy_string("{}");
    return copy_string(any_map_to_json(bundle.action_descriptor(action_id)));
  } catch (const std::exception& e) {
    return copy_string(std::string("{\"error\":") + json_escape(e.what()) + "}");
  }
}

DINOBOARD_C_API void* dinoboard_session_create(
    const char* game_id,
    const char* model_path,
    unsigned long long seed,
    int use_action_filter,
    char** out_error) {
  try {
    if (out_error) *out_error = nullptr;
    if (!game_id || !*game_id) {
      throw std::runtime_error("game_id is required");
    }
    if (!model_path || !*model_path) {
      throw std::runtime_error("model_path is required");
    }
    return new CSession(game_id, model_path, static_cast<std::uint64_t>(seed),
                        use_action_filter != 0);
  } catch (const std::exception& e) {
    set_error(out_error, e.what());
    return nullptr;
  } catch (...) {
    set_error(out_error, "unknown error");
    return nullptr;
  }
}

DINOBOARD_C_API void dinoboard_session_destroy(void* session) {
  delete static_cast<CSession*>(session);
}

DINOBOARD_C_API char* dinoboard_session_decide_json(
    void* session,
    int simulations,
    double temperature,
    int cover_root_edges,
    char** out_error) {
  try {
    if (out_error) *out_error = nullptr;
    if (!session) throw std::runtime_error("session is null");
    auto* s = static_cast<CSession*>(session);
    return copy_string(s->decide_json(simulations, temperature,
                                      cover_root_edges != 0));
  } catch (const std::exception& e) {
    set_error(out_error, e.what());
    return nullptr;
  } catch (...) {
    set_error(out_error, "unknown error");
    return nullptr;
  }
}

}  // extern "C"
