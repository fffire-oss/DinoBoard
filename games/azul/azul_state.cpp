#include "azul_state.h"

#include <algorithm>
#include <functional>

#include "../../engine/core/schema_hash.h"
#include "../../engine/core/viz_runtime.h"

namespace board_ai::azul {

template <int NPlayers>
const viz::VisibilitySchema& AzulState<NPlayers>::schema() {
  static const viz::VisibilitySchema s = []() {
    viz::VisibilitySchema schema;
    schema.n_players = Cfg::kPlayers;

    // ---- public scalars ----
    viz::declare_field(schema, "current_player",
                       viz::all_public({}, Cfg::kPlayers));
    viz::declare_field(schema, "game_first_player",
                       viz::all_public({}, Cfg::kPlayers));
    viz::declare_field(schema, "first_player_next_round",
                       viz::all_public({}, Cfg::kPlayers));
    viz::declare_field(schema, "winner", viz::all_public({}, Cfg::kPlayers));
    viz::declare_field(schema, "round_index",
                       viz::all_public({}, Cfg::kPlayers));
    viz::declare_field(schema, "terminal", viz::all_public({}, Cfg::kPlayers));
    viz::declare_field(schema, "first_player_token_in_center",
                       viz::all_public({}, Cfg::kPlayers));
    viz::declare_field(schema, "shared_victory",
                       viz::all_public({}, Cfg::kPlayers));

    // ---- public 1D / 2D ----
    viz::declare_field(schema, "scores",
                       viz::all_public({Cfg::kPlayers}, Cfg::kPlayers));
    viz::declare_field(
        schema, "factories",
        viz::all_public({Cfg::kFactories, kColors}, Cfg::kPlayers));
    viz::declare_field(schema, "center",
                       viz::all_public({kColors}, Cfg::kPlayers));

    // ---- per-player public boards (PlayerState sub-fields, flattened) ----
    viz::declare_field(
        schema, "player_line_len",
        viz::all_public({Cfg::kPlayers, kRows}, Cfg::kPlayers));
    viz::declare_field(
        schema, "player_line_color",
        viz::all_public({Cfg::kPlayers, kRows}, Cfg::kPlayers));
    viz::declare_field(
        schema, "player_wall_mask",
        viz::all_public({Cfg::kPlayers, kRows}, Cfg::kPlayers));
    viz::declare_field(
        schema, "player_floor",
        viz::all_public({Cfg::kPlayers, 7}, Cfg::kPlayers));
    viz::declare_field(schema, "player_floor_count",
                       viz::all_public({Cfg::kPlayers}, Cfg::kPlayers));
    viz::declare_field(schema, "player_score",
                       viz::all_public({Cfg::kPlayers}, Cfg::kPlayers));

    // bag_counts / box_lid_counts: tiles within a color are
    // interchangeable; only counts are real, and counts are publicly
    // derivable (start total minus what's been placed/discarded). Both
    // are first-class all_public schema fields — there is no order to
    // hide.
    viz::declare_field(schema, "bag_counts",
                       viz::all_public({kColors}, Cfg::kPlayers));
    viz::declare_field(schema, "box_lid_counts",
                       viz::all_public({kColors}, Cfg::kPlayers));

    return schema;
  }();
  return s;
}

template <int NPlayers>
AzulState<NPlayers>::AzulState() {
  reset_with_seed(0xC0FFEEu);
}

template <int NPlayers>
void AzulState<NPlayers>::reset_with_seed(std::uint64_t seed) {
  IGameState::reset_step_count_base();
  current_player_ = 0;
  game_first_player_ = 0;
  first_player_next_round = 0;
  winner_ = -1;
  round_index = 0;
  terminal = false;
  first_player_token_in_center = true;
  shared_victory = false;
  scores = {};
  factories = {};
  center = {};
  bag_counts = {};
  box_lid_counts = {};
  players = {};
  undo_stack.clear();
  persistent_tree_cache.tree.clear();
  persistent_tree_cache.chance_buckets.clear();
  persistent_tree_cache.sig_to_node.clear();
  // Bag starts with 20 of each color — a public fact.
  for (int c = 0; c < kColors; ++c) {
    bag_counts[c] = 20;
  }
  // Initial draw: build a one-shot rng from the seed (rng is not stored
  // on state — caller of do_action_fast supplies its own rng for
  // subsequent draws). Counts are the only real public fact about the
  // bag; draw_one_tile samples a color weighted by remaining counts.
  std::mt19937_64 init_rng(seed);
  refill_factories_from_rng(init_rng);
  viz::init_viz(*this, schema());
}

template <int NPlayers>
bool AzulState<NPlayers>::all_sources_empty() const {
  for (const auto& fac : factories) {
    for (std::uint8_t c : fac) {
      if (c > 0) {
        return false;
      }
    }
  }
  for (std::uint8_t c : center) {
    if (c > 0) {
      return false;
    }
  }
  return !first_player_token_in_center;
}

template <int NPlayers>
int AzulState<NPlayers>::draw_one_tile(std::mt19937_64& rng) {
  int total = 0;
  for (int c : bag_counts) total += c;
  if (total == 0) {
    // Bag empty: refill from box lid (counts copy, then zero box).
    for (int c = 0; c < kColors; ++c) {
      bag_counts[c] = box_lid_counts[c];
      box_lid_counts[c] = 0;
      total += bag_counts[c];
    }
    if (total == 0) return -1;
  }
  // Weighted uniform sample over remaining tiles. Modulo on a 64-bit
  // mt19937_64 with total <= 100 has bias < 5e-18 — irrelevant.
  int pick = static_cast<int>(rng() % static_cast<std::uint64_t>(total));
  for (int c = 0; c < kColors; ++c) {
    if (pick < bag_counts[c]) {
      --bag_counts[c];
      return c;
    }
    pick -= bag_counts[c];
  }
  return -1;  // unreachable
}

template <int NPlayers>
void AzulState<NPlayers>::refill_factories_from_rng(std::mt19937_64& rng) {
  factories = {};
  center = {};
  for (int f = 0; f < Cfg::kFactories; ++f) {
    for (int i = 0; i < 4; ++i) {
      const int color = draw_one_tile(rng);
      if (color < 0 || color >= kColors) {
        continue;
      }
      factories[f][color] = static_cast<std::uint8_t>(factories[f][color] + 1);
    }
  }
}

template <int NPlayers>
bool AzulState<NPlayers>::is_tree_cache_consistent() const {
  if (persistent_tree_cache.tree.size() < persistent_tree_cache.chance_buckets.size()) {
    return false;
  }
  for (const auto& kv : persistent_tree_cache.sig_to_node) {
    const int idx = kv.second;
    if (idx < 0 || idx >= static_cast<int>(persistent_tree_cache.tree.size())) {
      return false;
    }
  }
  return true;
}

template <int NPlayers>
StateHash64 AzulState<NPlayers>::state_hash() const {
  std::size_t h = 0;
  hash_combine(h,static_cast<std::size_t>(current_player_));
  hash_combine(h,static_cast<std::size_t>(first_player_next_round));
  hash_combine(h,static_cast<std::size_t>(winner_ + 1));
  hash_combine(h,static_cast<std::size_t>(round_index));
  hash_combine(h,static_cast<std::size_t>(terminal ? 1 : 0));
  hash_combine(h,static_cast<std::size_t>(first_player_token_in_center ? 1 : 0));
  hash_combine(h,static_cast<std::size_t>(shared_victory ? 1 : 0));
  for (int s : scores) {
    hash_combine(h,static_cast<std::size_t>(s));
  }
  for (const auto& fac : factories) {
    for (std::uint8_t c : fac) {
      hash_combine(h,static_cast<std::size_t>(c));
    }
  }
  for (std::uint8_t c : center) {
    hash_combine(h,static_cast<std::size_t>(c));
  }
  // bag/box_lid are stored as per-color counts — the only public fact
  // about them. No order to hash.
  for (int count : bag_counts) hash_combine(h, static_cast<std::size_t>(count));
  for (int count : box_lid_counts) hash_combine(h, static_cast<std::size_t>(count));
  for (const auto& p : players) {
    for (std::uint8_t len : p.line_len) {
      hash_combine(h,static_cast<std::size_t>(len));
    }
    for (std::int8_t color : p.line_color) {
      hash_combine(h,static_cast<std::size_t>(color + 1));
    }
    for (std::uint8_t m : p.wall_mask) {
      hash_combine(h,static_cast<std::size_t>(m));
    }
    hash_combine(h,static_cast<std::size_t>(p.floor_count));
    for (std::int8_t f : p.floor) {
      hash_combine(h,static_cast<std::size_t>(f + 1));
    }
    hash_combine(h,static_cast<std::size_t>(p.score));
  }
  return static_cast<StateHash64>(h);
}

template <int NPlayers>
void AzulState<NPlayers>::hash_field_slot(
    Hasher& h, const std::string& name,
    const std::vector<int>& idx) const {
  // 0-D scalars.
  if (name == "current_player") { h.add(current_player_); return; }
  if (name == "game_first_player") { h.add(game_first_player_); return; }
  if (name == "first_player_next_round") { h.add(first_player_next_round); return; }
  if (name == "winner") { h.add(winner_ + 1); return; }
  if (name == "round_index") { h.add(round_index); return; }
  if (name == "terminal") { h.add(terminal ? 1 : 0); return; }
  if (name == "first_player_token_in_center") {
    h.add(first_player_token_in_center ? 1 : 0); return;
  }
  if (name == "shared_victory") { h.add(shared_victory ? 1 : 0); return; }
  // 1-D fields.
  if (name == "scores") {
    h.add(scores[static_cast<size_t>(idx[0])]); return;
  }
  if (name == "center") {
    h.add(static_cast<int>(center[static_cast<size_t>(idx[0])])); return;
  }
  if (name == "bag_counts") {
    h.add(bag_counts[static_cast<size_t>(idx[0])]); return;
  }
  if (name == "box_lid_counts") {
    h.add(box_lid_counts[static_cast<size_t>(idx[0])]); return;
  }
  if (name == "player_floor_count") {
    h.add(static_cast<int>(players[static_cast<size_t>(idx[0])].floor_count));
    return;
  }
  if (name == "player_score") {
    h.add(players[static_cast<size_t>(idx[0])].score); return;
  }
  // 2-D fields.
  if (name == "factories") {
    h.add(static_cast<int>(
        factories[static_cast<size_t>(idx[0])][static_cast<size_t>(idx[1])]));
    return;
  }
  if (name == "player_line_len") {
    h.add(static_cast<int>(
        players[static_cast<size_t>(idx[0])].line_len[static_cast<size_t>(idx[1])]));
    return;
  }
  if (name == "player_line_color") {
    h.add(static_cast<int>(
        players[static_cast<size_t>(idx[0])].line_color[static_cast<size_t>(idx[1])]) + 1);
    return;
  }
  if (name == "player_wall_mask") {
    h.add(static_cast<int>(
        players[static_cast<size_t>(idx[0])].wall_mask[static_cast<size_t>(idx[1])]));
    return;
  }
  if (name == "player_floor") {
    h.add(static_cast<int>(
        players[static_cast<size_t>(idx[0])].floor[static_cast<size_t>(idx[1])]) + 1);
    return;
  }
}

template <int NPlayers>
std::any AzulState<NPlayers>::read_field_slot(
    const std::string& name, const std::vector<int>& idx) const {
  // 0-D scalars.
  if (name == "current_player") return std::any(current_player_);
  if (name == "game_first_player") return std::any(game_first_player_);
  if (name == "first_player_next_round") return std::any(first_player_next_round);
  if (name == "winner") return std::any(winner_);
  if (name == "round_index") return std::any(round_index);
  if (name == "terminal") return std::any(terminal);
  if (name == "first_player_token_in_center") return std::any(first_player_token_in_center);
  if (name == "shared_victory") return std::any(shared_victory);
  // 1-D fields.
  if (name == "scores") return std::any(scores[static_cast<size_t>(idx[0])]);
  if (name == "center") return std::any(static_cast<int>(center[static_cast<size_t>(idx[0])]));
  if (name == "bag_counts") return std::any(bag_counts[static_cast<size_t>(idx[0])]);
  if (name == "box_lid_counts") return std::any(box_lid_counts[static_cast<size_t>(idx[0])]);
  if (name == "player_floor_count") {
    return std::any(static_cast<int>(players[static_cast<size_t>(idx[0])].floor_count));
  }
  if (name == "player_score") return std::any(players[static_cast<size_t>(idx[0])].score);
  // 2-D fields.
  if (name == "factories") {
    return std::any(static_cast<int>(
        factories[static_cast<size_t>(idx[0])][static_cast<size_t>(idx[1])]));
  }
  if (name == "player_line_len") {
    return std::any(static_cast<int>(
        players[static_cast<size_t>(idx[0])].line_len[static_cast<size_t>(idx[1])]));
  }
  if (name == "player_line_color") {
    return std::any(static_cast<int>(
        players[static_cast<size_t>(idx[0])].line_color[static_cast<size_t>(idx[1])]));
  }
  if (name == "player_wall_mask") {
    return std::any(static_cast<int>(
        players[static_cast<size_t>(idx[0])].wall_mask[static_cast<size_t>(idx[1])]));
  }
  if (name == "player_floor") {
    return std::any(static_cast<int>(
        players[static_cast<size_t>(idx[0])].floor[static_cast<size_t>(idx[1])]));
  }
  return {};
}

template <int NPlayers>
void AzulState<NPlayers>::write_field_slot(
    const std::string& name, const std::vector<int>& idx,
    const std::any& value) {
  auto as_int = [&]() -> int {
    if (value.type() == typeid(int)) return std::any_cast<int>(value);
    if (value.type() == typeid(bool)) return std::any_cast<bool>(value) ? 1 : 0;
    return 0;
  };
  auto as_bool = [&]() -> bool {
    if (value.type() == typeid(bool)) return std::any_cast<bool>(value);
    if (value.type() == typeid(int)) return std::any_cast<int>(value) != 0;
    return false;
  };

  if (name == "current_player") current_player_ = as_int();
  else if (name == "game_first_player") game_first_player_ = as_int();
  else if (name == "first_player_next_round") first_player_next_round = as_int();
  else if (name == "winner") winner_ = as_int();
  else if (name == "round_index") round_index = as_int();
  else if (name == "terminal") terminal = as_bool();
  else if (name == "first_player_token_in_center") first_player_token_in_center = as_bool();
  else if (name == "shared_victory") shared_victory = as_bool();
  else if (name == "scores") scores[static_cast<size_t>(idx[0])] = as_int();
  else if (name == "center") {
    center[static_cast<size_t>(idx[0])] = static_cast<std::uint8_t>(as_int());
  }
  else if (name == "bag_counts") bag_counts[static_cast<size_t>(idx[0])] = as_int();
  else if (name == "box_lid_counts") box_lid_counts[static_cast<size_t>(idx[0])] = as_int();
  else if (name == "player_floor_count") {
    players[static_cast<size_t>(idx[0])].floor_count = static_cast<std::uint8_t>(as_int());
  }
  else if (name == "player_score") {
    players[static_cast<size_t>(idx[0])].score = as_int();
  }
  else if (name == "factories") {
    factories[static_cast<size_t>(idx[0])][static_cast<size_t>(idx[1])] =
        static_cast<std::uint8_t>(as_int());
  }
  else if (name == "player_line_len") {
    players[static_cast<size_t>(idx[0])].line_len[static_cast<size_t>(idx[1])] =
        static_cast<std::uint8_t>(as_int());
  }
  else if (name == "player_line_color") {
    players[static_cast<size_t>(idx[0])].line_color[static_cast<size_t>(idx[1])] =
        static_cast<std::int8_t>(as_int());
  }
  else if (name == "player_wall_mask") {
    players[static_cast<size_t>(idx[0])].wall_mask[static_cast<size_t>(idx[1])] =
        static_cast<std::uint8_t>(as_int());
  }
  else if (name == "player_floor") {
    players[static_cast<size_t>(idx[0])].floor[static_cast<size_t>(idx[1])] =
        static_cast<std::int8_t>(as_int());
  }
}

template class AzulState<2>;
template class AzulState<3>;
template class AzulState<4>;

}  // namespace board_ai::azul
