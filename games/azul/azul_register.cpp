#include <stdexcept>
#include <string>
#include <vector>

#include "../../engine/core/game_registry.h"
#include "../../engine/search/tail_solver.h"
#include "azul_state.h"
#include "azul_rules.h"
#include "azul_net_adapter.h"

namespace {

using board_ai::AnyMap;
using board_ai::ActionId;
using board_ai::IGameState;
using board_ai::EventPhase;
using board_ai::PublicEvent;
using board_ai::PublicEventTrace;

static const char* const kColorNames[5] = {"blue", "yellow", "red", "black", "white"};
static const char* const kColorLetters[5] = {"B", "Y", "R", "K", "W"};

template <int NPlayers>
int azul_wall_col_for_color(int row, int color_idx) {
  return (color_idx + row) % board_ai::azul::kColors;
}

template <int NPlayers>
AnyMap serialize_azul(const IGameState& state) {
  using Cfg = board_ai::azul::AzulConfig<NPlayers>;
  const auto& s = board_ai::checked_cast<board_ai::azul::AzulState<NPlayers>>(state);

  AnyMap m;
  m["current_player"] = std::any(s.current_player());
  m["is_terminal"] = std::any(s.is_terminal());
  m["winner"] = std::any(s.winner());
  m["num_players"] = std::any(NPlayers);
  m["round_index"] = std::any(s.round_index);
  m["first_player_token_in_center"] = std::any(s.first_player_token_in_center);

  std::vector<int> scores(NPlayers);
  for (int i = 0; i < NPlayers; ++i) scores[i] = s.scores[i];
  m["scores"] = std::any(scores);

  std::vector<std::vector<int>> factories(Cfg::kFactories);
  for (int fi = 0; fi < Cfg::kFactories; ++fi) {
    factories[fi].resize(board_ai::azul::kColors);
    for (int c = 0; c < board_ai::azul::kColors; ++c)
      factories[fi][c] = s.factories[fi][c];
  }
  std::vector<std::any> factories_any;
  for (auto& f : factories) factories_any.push_back(std::any(std::move(f)));
  m["factories"] = std::any(factories_any);

  std::vector<int> center(board_ai::azul::kColors);
  for (int c = 0; c < board_ai::azul::kColors; ++c) center[c] = s.center[c];
  m["center"] = std::any(center);

  std::vector<int> bag_counts(board_ai::azul::kColors, 0);
  for (auto tile : s.bag) {
    if (tile >= 0 && tile < board_ai::azul::kColors) bag_counts[tile]++;
  }
  m["bag_counts"] = std::any(bag_counts);
  m["bag_total"] = std::any(static_cast<int>(s.bag.size()));

  // Box lid contents (tiles returned from completed pattern rows / floor
  // overflow). Exposed for tile-conservation invariants in test suites.
  std::vector<int> box_counts(board_ai::azul::kColors, 0);
  for (auto tile : s.box_lid) {
    if (tile >= 0 && tile < board_ai::azul::kColors) box_counts[tile]++;
  }
  m["box_counts"] = std::any(box_counts);

  std::vector<AnyMap> players;
  for (int p = 0; p < NPlayers; ++p) {
    const auto& ps = s.players[p];
    AnyMap pm;
    pm["score"] = std::any(static_cast<int>(ps.score));

    std::vector<AnyMap> lines;
    for (int r = 0; r < board_ai::azul::kRows; ++r) {
      AnyMap line;
      line["length"] = std::any(static_cast<int>(ps.line_len[r]));
      line["color"] = std::any(static_cast<int>(ps.line_color[r]));
      line["capacity"] = std::any(r + 1);
      lines.push_back(std::move(line));
    }
    pm["pattern_lines"] = std::any(lines);

    std::vector<std::vector<int>> wall(board_ai::azul::kRows);
    for (int r = 0; r < board_ai::azul::kRows; ++r) {
      wall[r].resize(board_ai::azul::kColors);
      for (int c = 0; c < board_ai::azul::kColors; ++c)
        wall[r][c] = (ps.wall_mask[r] >> c) & 1;
    }
    std::vector<std::any> wall_any;
    for (auto& row : wall) wall_any.push_back(std::any(std::move(row)));
    pm["wall"] = std::any(wall_any);

    std::vector<int> floor;
    for (int fi = 0; fi < static_cast<int>(ps.floor_count); ++fi)
      floor.push_back(static_cast<int>(ps.floor[fi]));
    pm["floor"] = std::any(floor);
    pm["floor_count"] = std::any(static_cast<int>(ps.floor_count));

    players.push_back(std::move(pm));
  }
  m["players"] = std::any(players);

  return m;
}

template <int NPlayers>
AnyMap describe_azul(ActionId action) {
  using Cfg = board_ai::azul::AzulConfig<NPlayers>;
  AnyMap m;
  m["action_id"] = std::any(static_cast<int>(action));

  int source = static_cast<int>(action / (board_ai::azul::kColors * board_ai::azul::kTargetsPerColor));
  int color = static_cast<int>((action / board_ai::azul::kTargetsPerColor) % board_ai::azul::kColors);
  int target = static_cast<int>(action % board_ai::azul::kTargetsPerColor);

  m["source"] = std::any(source);
  m["color"] = std::any(color);
  m["target_line"] = std::any(target);

  bool is_center = (source == Cfg::kCenterSource);
  m["is_center"] = std::any(is_center);
  m["color_name"] = std::any(std::string(kColorNames[color]));
  m["color_letter"] = std::any(std::string(kColorLetters[color]));

  if (is_center) {
    m["source_name"] = std::any(std::string("center"));
  } else {
    m["source_name"] = std::any(std::string("factory_") + std::to_string(source));
  }

  if (target < board_ai::azul::kRows) {
    m["target_name"] = std::any(std::string("row_") + std::to_string(target + 1));
  } else {
    m["target_name"] = std::any(std::string("floor"));
  }

  return m;
}

// --- Public-event protocol for AI API belief-equivalence --------------
//
// Azul has one source of hidden randomness: the bag. At game start and at
// every round-end, factories are refilled by drawing from the bag. An AI
// API session with its own seed would produce different draws than ground
// truth, so we need to sync both events (initial factories + per-round
// refills). Azul's belief tracker is stateless (bag contents are fully
// derivable from public state); these events operate purely on state.

namespace azul_events {

using board_ai::azul::AzulConfig;
using board_ai::azul::AzulState;
using board_ai::azul::kColors;

template <int NPlayers>
std::vector<std::any> factories_to_any(const AzulState<NPlayers>& s) {
  std::vector<std::vector<int>> factories(AzulConfig<NPlayers>::kFactories);
  for (int f = 0; f < AzulConfig<NPlayers>::kFactories; ++f) {
    factories[f].resize(kColors);
    for (int c = 0; c < kColors; ++c) {
      factories[f][c] = static_cast<int>(s.factories[f][c]);
    }
  }
  std::vector<std::any> out;
  out.reserve(factories.size());
  for (auto& f : factories) out.push_back(std::any(std::move(f)));
  return out;
}

template <int NPlayers>
void overwrite_factories_from_any(AzulState<NPlayers>& s, const std::any& val) {
  auto factories_any = std::any_cast<std::vector<std::any>>(val);
  if (factories_any.size() != static_cast<size_t>(AzulConfig<NPlayers>::kFactories)) {
    throw std::runtime_error(
        "azul event: factories count mismatch (expected " +
        std::to_string(AzulConfig<NPlayers>::kFactories) + ")");
  }
  for (int f = 0; f < AzulConfig<NPlayers>::kFactories; ++f) {
    auto counts = std::any_cast<std::vector<int>>(factories_any[static_cast<size_t>(f)]);
    if (counts.size() != static_cast<size_t>(kColors)) {
      throw std::runtime_error(
          "azul event: color count mismatch for factory " + std::to_string(f));
    }
    for (int c = 0; c < kColors; ++c) {
      s.factories[f][c] = static_cast<std::uint8_t>(counts[static_cast<size_t>(c)]);
    }
  }
}

// After overriding visible tiles, recompute bag = 20-per-color minus all
// visible tiles of that color (factories, center, walls, pattern lines,
// floor, box_lid). This ensures MCTS determinizations drawing from bag
// during simulation will see a consistent pool — even if AI's own earlier
// RNG draws diverged from ground truth.
template <int NPlayers>
void recompute_bag_from_visible(AzulState<NPlayers>& s) {
  std::array<int, kColors> visible{};
  for (int f = 0; f < AzulConfig<NPlayers>::kFactories; ++f) {
    for (int c = 0; c < kColors; ++c) visible[c] += s.factories[f][c];
  }
  for (int c = 0; c < kColors; ++c) visible[c] += s.center[c];
  for (int p = 0; p < NPlayers; ++p) {
    const auto& ps = s.players[p];
    for (int r = 0; r < board_ai::azul::kRows; ++r) {
      for (int c = 0; c < kColors; ++c) {
        if ((ps.wall_mask[r] >> c) & 1U) visible[c]++;
      }
      if (ps.line_color[r] >= 0 && ps.line_color[r] < kColors) {
        visible[ps.line_color[r]] += ps.line_len[r];
      }
    }
    for (int i = 0; i < ps.floor_count; ++i) {
      const int t = ps.floor[i];
      if (t >= 0 && t < kColors) visible[t]++;
    }
  }
  // Box_lid holds tiles already "consumed" out of bag but recyclable. Keep
  // box_lid as-is; put the remainder into bag.
  for (int c = 0; c < kColors; ++c) {
    for (std::int8_t t : s.box_lid) {
      if (t == static_cast<std::int8_t>(c)) visible[c]++;
    }
  }
  s.bag.clear();
  for (int c = 0; c < kColors; ++c) {
    const int remaining = 20 - visible[c];
    for (int i = 0; i < remaining && remaining > 0; ++i) {
      s.bag.push_back(static_cast<std::int8_t>(c));
    }
  }
}

template <int NPlayers>
AnyMap extract_initial_observation(const IGameState& state, int /*perspective*/) {
  const auto& s = board_ai::checked_cast<AzulState<NPlayers>>(state);
  AnyMap out;
  out["factories"] = std::any(factories_to_any(s));
  return out;
}

template <int NPlayers>
void apply_initial_observation(IGameState& state, int /*perspective*/, const AnyMap& obs) {
  auto& s = board_ai::checked_cast<AzulState<NPlayers>>(state);
  auto it = obs.find("factories");
  if (it == obs.end()) {
    throw std::runtime_error("azul initial_observation missing 'factories'");
  }
  overwrite_factories_from_any(s, it->second);
  // At game start, center is always empty and box_lid is empty.
  for (int c = 0; c < kColors; ++c) s.center[c] = 0;
  s.box_lid.clear();
  recompute_bag_from_visible(s);
}

template <int NPlayers>
PublicEventTrace extract_events(
    const IGameState& before,
    ActionId /*action*/,
    const IGameState& after,
    int /*perspective*/) {
  const auto& sb = board_ai::checked_cast<AzulState<NPlayers>>(before);
  const auto& sa = board_ai::checked_cast<AzulState<NPlayers>>(after);
  PublicEventTrace out;
  // Round settlement increments round_index and triggers factory refill.
  // If the round advanced AND the game continues (not terminal), emit
  // a factory_refill event describing the newly drawn tiles.
  if (sa.round_index > sb.round_index && !sa.terminal) {
    AnyMap payload;
    payload["factories"] = std::any(factories_to_any(sa));
    out.post_events.emplace_back("factory_refill", std::move(payload));
  }

  // full public snapshot mirroring
  // AzulState::hash_public_fields. Azul has no per-perspective private
  // info (bag order is symmetric hidden to ALL), so the snapshot is
  // literally the whole public state.
  {
    AnyMap snap;
    snap["current_player"] = std::any(static_cast<int>(sa.current_player_));
    snap["first_player_next_round"] = std::any(static_cast<int>(sa.first_player_next_round));
    snap["winner"] = std::any(static_cast<int>(sa.winner_));
    snap["round_index"] = std::any(static_cast<int>(sa.round_index));
    snap["terminal"] = std::any(static_cast<bool>(sa.terminal));
    snap["first_player_token_in_center"] = std::any(static_cast<bool>(sa.first_player_token_in_center));
    snap["shared_victory"] = std::any(static_cast<bool>(sa.shared_victory));

    std::vector<int> scores_v(sa.scores.begin(), sa.scores.end());
    snap["scores"] = std::any(scores_v);

    // Factories: flattened kFactories * kColors.
    std::vector<int> factories_flat;
    factories_flat.reserve(sa.factories.size() * board_ai::azul::kColors);
    for (const auto& fac : sa.factories) {
      for (std::uint8_t c : fac) factories_flat.push_back(static_cast<int>(c));
    }
    snap["factories_flat"] = std::any(factories_flat);

    std::vector<int> center_v;
    center_v.reserve(sa.center.size());
    for (std::uint8_t c : sa.center) center_v.push_back(static_cast<int>(c));
    snap["center"] = std::any(center_v);

    // Bag / box_lid are multisets for hash_public_fields purposes.
    // The snapshot carries them as full vectors (preserving order)
    // because the observer's bag DRAW ORDER is sampled by
    // randomize_unseen anyway — but we need the MULTISET (count per
    // color) pinned to truth. Encoding as full vector with actual
    // truth order also works; randomize_unseen will shuffle.
    std::vector<int> bag_v;
    bag_v.reserve(sa.bag.size());
    for (auto t : sa.bag) bag_v.push_back(static_cast<int>(t));
    snap["bag"] = std::any(bag_v);

    std::vector<int> box_v;
    box_v.reserve(sa.box_lid.size());
    for (auto t : sa.box_lid) box_v.push_back(static_cast<int>(t));
    snap["box_lid"] = std::any(box_v);

    // Per-player boards: line_len, line_color, wall_mask, floor_count,
    // floor, score. Flatten into keyed arrays.
    const int NP = NPlayers;
    std::vector<int> line_len_flat(NP * board_ai::azul::kRows);
    std::vector<int> line_color_flat(NP * board_ai::azul::kRows);
    std::vector<int> wall_mask_flat(NP * board_ai::azul::kRows);
    std::vector<int> floor_flat(NP * 7);
    std::vector<int> floor_count_v(NP);
    std::vector<int> player_score_v(NP);
    for (int p = 0; p < NP; ++p) {
      const auto& pb = sa.players[p];
      for (int r = 0; r < board_ai::azul::kRows; ++r) {
        line_len_flat[p * board_ai::azul::kRows + r] = static_cast<int>(pb.line_len[r]);
        line_color_flat[p * board_ai::azul::kRows + r] = static_cast<int>(pb.line_color[r]);
        wall_mask_flat[p * board_ai::azul::kRows + r] = static_cast<int>(pb.wall_mask[r]);
      }
      for (int f = 0; f < 7; ++f) floor_flat[p * 7 + f] = static_cast<int>(pb.floor[f]);
      floor_count_v[p] = static_cast<int>(pb.floor_count);
      player_score_v[p] = pb.score;
    }
    snap["line_len_flat"] = std::any(line_len_flat);
    snap["line_color_flat"] = std::any(line_color_flat);
    snap["wall_mask_flat"] = std::any(wall_mask_flat);
    snap["floor_flat"] = std::any(floor_flat);
    snap["floor_count"] = std::any(floor_count_v);
    snap["player_score"] = std::any(player_score_v);

    out.public_snapshot = std::move(snap);
  }

  return out;
}

// applier — writes public fields from truth snapshot.
template <int NPlayers>
void apply_public_state(IGameState& state, const AnyMap& snap) {
  auto& s = board_ai::checked_cast<AzulState<NPlayers>>(state);

  auto get_int = [&](const char* key) -> int {
    auto it = snap.find(key);
    return (it != snap.end()) ? std::any_cast<int>(it->second) : 0;
  };
  auto get_bool = [&](const char* key) -> bool {
    auto it = snap.find(key);
    return (it != snap.end()) ? std::any_cast<bool>(it->second) : false;
  };
  // Robust int-vector accessor: handles both vector<int> and
  // vector<any> (empty-list case where py_to_any can't detect
  // all-int-because-empty and defaults to vector<any>).
  auto get_iv = [&](const char* key) -> std::vector<int> {
    auto it = snap.find(key);
    if (it == snap.end()) return {};
    if (it->second.type() == typeid(std::vector<int>)) {
      return std::any_cast<std::vector<int>>(it->second);
    }
    if (it->second.type() == typeid(std::vector<std::any>)) {
      const auto& av = std::any_cast<const std::vector<std::any>&>(it->second);
      std::vector<int> out;
      out.reserve(av.size());
      for (const auto& x : av) {
        if (x.type() == typeid(int)) out.push_back(std::any_cast<int>(x));
      }
      return out;
    }
    return {};
  };

  s.current_player_ = get_int("current_player");
  s.first_player_next_round = get_int("first_player_next_round");
  s.winner_ = get_int("winner");
  s.round_index = get_int("round_index");
  s.terminal = get_bool("terminal");
  s.first_player_token_in_center = get_bool("first_player_token_in_center");
  s.shared_victory = get_bool("shared_victory");

  auto scores_v = get_iv("scores");
  for (int p = 0; p < NPlayers && p < static_cast<int>(scores_v.size()); ++p) {
    s.scores[p] = scores_v[p];
  }

  auto factories_flat = get_iv("factories_flat");
  for (size_t f = 0; f < s.factories.size(); ++f) {
    for (size_t c = 0; c < s.factories[f].size(); ++c) {
      const size_t idx = f * s.factories[f].size() + c;
      if (idx < factories_flat.size()) {
        s.factories[f][c] = static_cast<std::uint8_t>(factories_flat[idx]);
      } else {
        s.factories[f][c] = 0;
      }
    }
  }

  auto center_v = get_iv("center");
  for (size_t c = 0; c < s.center.size(); ++c) {
    s.center[c] = (c < center_v.size()) ? static_cast<std::uint8_t>(center_v[c]) : 0;
  }

  auto bag_v = get_iv("bag");
  s.bag.clear();
  s.bag.reserve(bag_v.size());
  for (int t : bag_v) s.bag.push_back(static_cast<std::int8_t>(t));

  auto box_v = get_iv("box_lid");
  s.box_lid.clear();
  s.box_lid.reserve(box_v.size());
  for (int t : box_v) s.box_lid.push_back(static_cast<std::int8_t>(t));

  auto line_len_flat = get_iv("line_len_flat");
  auto line_color_flat = get_iv("line_color_flat");
  auto wall_mask_flat = get_iv("wall_mask_flat");
  auto floor_flat = get_iv("floor_flat");
  auto floor_count_v = get_iv("floor_count");
  auto player_score_v = get_iv("player_score");
  for (int p = 0; p < NPlayers; ++p) {
    auto& pb = s.players[p];
    for (int r = 0; r < board_ai::azul::kRows; ++r) {
      const int idx = p * board_ai::azul::kRows + r;
      if (idx < static_cast<int>(line_len_flat.size())) pb.line_len[r] = static_cast<std::uint8_t>(line_len_flat[idx]);
      if (idx < static_cast<int>(line_color_flat.size())) pb.line_color[r] = static_cast<std::int8_t>(line_color_flat[idx]);
      if (idx < static_cast<int>(wall_mask_flat.size())) pb.wall_mask[r] = static_cast<std::uint8_t>(wall_mask_flat[idx]);
    }
    for (int f = 0; f < 7; ++f) {
      const int idx = p * 7 + f;
      if (idx < static_cast<int>(floor_flat.size())) pb.floor[f] = static_cast<std::int8_t>(floor_flat[idx]);
    }
    if (p < static_cast<int>(floor_count_v.size())) pb.floor_count = static_cast<std::uint8_t>(floor_count_v[p]);
    if (p < static_cast<int>(player_score_v.size())) pb.score = player_score_v[p];
  }
}

template <int NPlayers>
void apply_event(IGameState& state, EventPhase phase,
                 const std::string& kind, const AnyMap& payload) {
  if (phase != EventPhase::kPostAction) {
    throw std::runtime_error(
        "azul: unexpected pre-action event '" + kind + "' (Azul has only post-action events)");
  }
  if (kind != "factory_refill") {
    throw std::runtime_error("azul: unknown event kind '" + kind + "'");
  }
  auto& s = board_ai::checked_cast<AzulState<NPlayers>>(state);
  auto it = payload.find("factories");
  if (it == payload.end()) {
    throw std::runtime_error("azul factory_refill missing 'factories'");
  }
  overwrite_factories_from_any(s, it->second);
  // Refill always clears center and resets first_player_token — these are
  // already handled by apply_round_settlement before we got here, so just
  // ensure center is empty (defensive).
  for (int c = 0; c < kColors; ++c) s.center[c] = 0;
  recompute_bag_from_visible(s);
}

}  // namespace azul_events

// Simplified Azul heuristic inspired by (but far simpler than) mosaic-azul's
// HeuristicBot (~1150 lines in the reference project). We boil it down to
// ~5 high-signal features: pattern-line completion potential, floor
// overflow penalty, wall-placement score, wall-conflict detection, and
// first-player-token weight.
//
// This reads only public + own-side info (Azul has no non-symmetric hidden
// info anyway — the bag is symmetric random — so leak risk is minimal.
// Still, we limit reads to state.factories / state.center / state.players[*]
// which are all publicly observable.)
namespace azul_heuristic {

using board_ai::azul::AzulConfig;
using board_ai::azul::AzulRules;
using board_ai::azul::AzulState;
using board_ai::azul::kColors;
using board_ai::azul::kRows;
using board_ai::azul::kTargetsPerColor;

constexpr int kFloorPenalties[7] = {1, 1, 2, 2, 2, 3, 3};

inline int floor_penalty(int floor_before, int added) {
  int total = 0;
  for (int i = 0; i < added; ++i) {
    int idx = std::min(floor_before + i, 6);
    total += kFloorPenalties[idx];
  }
  return total;
}

// Inlined from AzulRules::{wall_col_for_color,score_wall_placement} since
// those are private class helpers. Duplicating to avoid widening that API.
inline int wall_col(int row, int color_idx) {
  return (color_idx + row) % kColors;
}

template <typename PlayerT>
int wall_placement_score(const PlayerT& p, int row, int col) {
  auto filled = [&](int r, int c) -> bool {
    if (r < 0 || r >= kRows || c < 0 || c >= kColors) return false;
    return ((p.wall_mask[r] >> c) & 1U) != 0U;
  };
  int h = 1;
  for (int c = col - 1; c >= 0 && filled(row, c); --c) ++h;
  for (int c = col + 1; c < kColors && filled(row, c); ++c) ++h;
  int v = 1;
  for (int r = row - 1; r >= 0 && filled(r, col); --r) ++v;
  for (int r = row + 1; r < kRows && filled(r, col); ++r) ++v;
  if (h == 1 && v == 1) return 1;
  if (h > 1 && v > 1) return h + v;
  return std::max(h, v);
}

template <int NPlayers>
double score_action_on_state(const AzulState<NPlayers>& s, ActionId action) {
  using Cfg = AzulConfig<NPlayers>;

  const int source = static_cast<int>(action / (kColors * kTargetsPerColor));
  const int color = static_cast<int>((action / kTargetsPerColor) % kColors);
  const int target = static_cast<int>(action % kTargetsPerColor);
  const int actor = s.current_player_;
  const auto& me = s.players[actor];

  // Count tiles available from this source (public info).
  int available = 0;
  if (source == Cfg::kCenterSource) {
    available = s.center[color];
  } else if (source >= 0 && source < Cfg::kFactories) {
    available = s.factories[source][color];
  }
  if (available == 0) return -1e6;  // should be filtered by legal_actions anyway

  double score = 0.0;

  // Taking from the center when first_player_token is still there:
  // +1 but also adds penalty to floor.
  int floor_add = 0;
  if (source == Cfg::kCenterSource && s.first_player_token_in_center) {
    floor_add += 1;  // first-player token lands on floor
    score += 0.3;    // being first next round is worth something
  }

  if (target == 5) {
    // All to floor — heavy penalty
    floor_add += available;
    score -= static_cast<double>(floor_penalty(me.floor_count, floor_add)) * 1.0;
    return score;
  }

  // Target is a pattern line (0..4)
  const int line_row = target;
  const int line_cap = line_row + 1;
  const int line_len = me.line_len[line_row];
  const int existing_color = me.line_color[line_row];

  // Check wall conflict: if color already on wall at this row, illegal-ish —
  // rules should forbid this via legal_actions, but be defensive.
  const int wall_col_idx = wall_col(line_row, color);
  if (((me.wall_mask[line_row] >> wall_col_idx) & 1U) != 0U) {
    return -1e5;
  }
  // Check line-color conflict: line already has a different color.
  if (existing_color >= 0 && existing_color != color) {
    return -1e5;
  }

  const int space_left = line_cap - line_len;
  const int placed = std::min(available, space_left);
  const int overflow = available - placed;

  // Placement on line → score if line fills up this round.
  if (placed + line_len == line_cap) {
    // Line will fill — predict wall placement score.
    double wall_score = static_cast<double>(
        wall_placement_score(me, line_row, wall_col_idx));
    score += wall_score * 1.2;
    // Completing a long line is inherently valuable.
    score += line_cap * 0.5;
  } else {
    // Partial fill — small reward for progress, scaled by proximity to completion.
    double progress = static_cast<double>(placed + line_len) / line_cap;
    score += progress * 0.6;
  }

  // Overflow to floor.
  if (overflow > 0) {
    floor_add += overflow;
  }
  score -= static_cast<double>(floor_penalty(me.floor_count, floor_add)) * 1.0;

  return score;
}

template <int NPlayers>
board_ai::HeuristicResult pick(
    board_ai::IGameState& state,
    const board_ai::IGameRules& rules,
    std::uint64_t /*rng_seed*/) {
  auto& s = board_ai::checked_cast<AzulState<NPlayers>>(state);
  auto legal = rules.legal_actions(state);
  board_ai::HeuristicResult result;
  result.actions = legal;
  result.scores.reserve(legal.size());
  for (ActionId a : legal) {
    result.scores.push_back(score_action_on_state<NPlayers>(s, a));
  }
  return result;
}

}  // namespace azul_heuristic

template <int NPlayers>
board_ai::GameBundle make_azul(const std::string& game_id, std::uint64_t seed) {
  board_ai::GameBundle b;
  b.game_id = game_id;
  auto s = std::make_unique<board_ai::azul::AzulState<NPlayers>>();
  s->reset_with_seed(seed);
  b.state = std::move(s);
  b.rules = std::make_unique<board_ai::azul::AzulRules<NPlayers>>();
  b.value_model = std::make_unique<board_ai::DefaultStateValueModel>();
  b.encoder = std::make_unique<board_ai::azul::AzulFeatureEncoder<NPlayers>>();
  b.belief_tracker = std::make_unique<board_ai::azul::AzulBeliefTracker<NPlayers>>();
  b.state_serializer = serialize_azul<NPlayers>;
  b.action_descriptor = describe_azul<NPlayers>;
  b.heuristic_picker = azul_heuristic::pick<NPlayers>;
  b.public_event_extractor = azul_events::extract_events<NPlayers>;
  b.public_event_applier = azul_events::apply_event<NPlayers>;
  b.public_state_applier = azul_events::apply_public_state<NPlayers>;
  b.initial_observation_extractor = azul_events::extract_initial_observation<NPlayers>;
  b.initial_observation_applier = azul_events::apply_initial_observation<NPlayers>;

  b.tail_solver = std::make_unique<board_ai::search::AlphaBetaTailSolver>();
  // AzulRules::do_action_deterministic forces a draw-terminal (winner=-1)
  // whenever an action would trigger a round-end factory refill (the only
  // source of hidden randomness in Azul). Tail solver therefore never
  // consumes hidden chance outcomes — safe to combine with belief tracker.
  b.stochastic_tail_solve_safe = true;

  // Trigger: at least one player has ≥4 tiles in some pattern-line row,
  // AND at least 2 factories are empty. Captures "round nearly resolved
  // and a wall placement is imminent" — small enough subgame to solve.
  b.tail_solve_trigger = [](const board_ai::IGameState& state, int /*ply*/) -> bool {
    using Cfg = board_ai::azul::AzulConfig<NPlayers>;
    const auto& s = board_ai::checked_cast<board_ai::azul::AzulState<NPlayers>>(state);
    bool any_line_near_full = false;
    for (int p = 0; p < NPlayers; ++p) {
      const auto& ps = s.players[p];
      for (int r = 0; r < board_ai::azul::kRows; ++r) {
        if (static_cast<int>(ps.line_len[r]) >= 4) {
          any_line_near_full = true;
          break;
        }
      }
      if (any_line_near_full) break;
    }
    if (!any_line_near_full) return false;
    int empty_factories = 0;
    for (int f = 0; f < Cfg::kFactories; ++f) {
      int total = 0;
      for (int c = 0; c < board_ai::azul::kColors; ++c) total += s.factories[f][c];
      if (total == 0) ++empty_factories;
    }
    return empty_factories >= 2;
  };
  return b;
}

board_ai::GameRegistrar reg_azul("azul", [](std::uint64_t seed) {
  return make_azul<2>("azul", seed);
});
board_ai::GameRegistrar reg_azul_2p("azul_2p", [](std::uint64_t seed) {
  return make_azul<2>("azul_2p", seed);
});
board_ai::GameRegistrar reg_azul_3p("azul_3p", [](std::uint64_t seed) {
  return make_azul<3>("azul_3p", seed);
});
board_ai::GameRegistrar reg_azul_4p("azul_4p", [](std::uint64_t seed) {
  return make_azul<4>("azul_4p", seed);
});

}  // namespace
