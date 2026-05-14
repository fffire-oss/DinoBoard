#include "splendor_state.h"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <stdexcept>

#include "splendor_rules.h"
#include "../../engine/core/masked_state.h"
#include "../../engine/core/schema_hash.h"
#include "../../engine/core/viz_runtime.h"
#include "../../engine/core/viz_walker.h"

namespace board_ai::splendor {

namespace {

// Caller-owned rng. Modulo on a 64-bit output is unbiased to within
// 90/2^64 ≈ 5e-18 for all decks here.
template <typename T>
T take_random_from_vector(std::vector<T>& deck, std::mt19937_64& rng) {
  if (deck.empty()) return T{};
  const size_t idx = static_cast<size_t>(rng() % deck.size());
  const T picked = deck[idx];
  if (idx + 1 < deck.size()) {
    deck[idx] = deck.back();
  }
  deck.pop_back();
  return picked;
}

std::vector<SplendorCard> build_card_pool() {
  struct CardDef {
    std::int8_t tier;
    std::int8_t bonus;
    std::int8_t points;
    std::array<std::int8_t, 5> cost;
  };
  static const std::array<CardDef, 90> defs{{
      {1, 1, 0, {0, 0, 0, 0, 3}}, {1, 1, 0, {1, 0, 0, 0, 2}}, {1, 1, 0, {0, 0, 2, 0, 2}},
      {1, 1, 0, {1, 0, 2, 2, 0}}, {1, 1, 0, {0, 1, 3, 1, 0}}, {1, 1, 0, {1, 0, 1, 1, 1}},
      {1, 1, 0, {1, 0, 1, 2, 1}}, {1, 1, 1, {0, 0, 0, 4, 0}}, {1, 3, 0, {3, 0, 0, 0, 0}},
      {1, 3, 0, {0, 2, 1, 0, 0}}, {1, 3, 0, {2, 0, 0, 2, 0}}, {1, 3, 0, {2, 0, 1, 0, 2}},
      {1, 3, 0, {1, 0, 0, 1, 3}}, {1, 3, 0, {1, 1, 1, 0, 1}}, {1, 3, 0, {2, 1, 1, 0, 1}},
      {1, 3, 1, {4, 0, 0, 0, 0}}, {1, 4, 0, {0, 0, 3, 0, 0}}, {1, 4, 0, {0, 0, 2, 1, 0}},
      {1, 4, 0, {2, 0, 2, 0, 0}}, {1, 4, 0, {2, 2, 0, 1, 0}}, {1, 4, 0, {0, 0, 1, 3, 1}},
      {1, 4, 0, {1, 1, 1, 1, 0}}, {1, 4, 0, {1, 2, 1, 1, 0}}, {1, 4, 1, {0, 4, 0, 0, 0}},
      {1, 0, 0, {0, 3, 0, 0, 0}}, {1, 0, 0, {0, 0, 0, 2, 1}}, {1, 0, 0, {0, 2, 0, 0, 2}},
      {1, 0, 0, {0, 2, 2, 0, 1}}, {1, 0, 0, {3, 1, 0, 0, 1}}, {1, 0, 0, {0, 1, 1, 1, 1}},
      {1, 0, 0, {0, 1, 2, 1, 1}}, {1, 0, 1, {0, 0, 4, 0, 0}}, {1, 2, 0, {0, 0, 0, 3, 0}},
      {1, 2, 0, {2, 1, 0, 0, 0}}, {1, 2, 0, {0, 2, 0, 2, 0}}, {1, 2, 0, {0, 1, 0, 2, 2}},
      {1, 2, 0, {1, 3, 1, 0, 0}}, {1, 2, 0, {1, 1, 0, 1, 1}}, {1, 2, 0, {1, 1, 0, 1, 2}},
      {1, 2, 1, {0, 0, 0, 0, 4}}, {2, 1, 1, {0, 2, 2, 3, 0}}, {2, 1, 1, {0, 2, 3, 0, 3}},
      {2, 1, 2, {0, 5, 0, 0, 0}}, {2, 1, 2, {5, 3, 0, 0, 0}}, {2, 1, 2, {2, 0, 0, 1, 4}},
      {2, 1, 3, {0, 6, 0, 0, 0}}, {2, 3, 1, {2, 0, 0, 2, 3}}, {2, 3, 1, {0, 3, 0, 2, 3}},
      {2, 3, 2, {0, 0, 0, 0, 5}}, {2, 3, 2, {3, 0, 0, 0, 5}}, {2, 3, 2, {1, 4, 2, 0, 0}},
      {2, 3, 3, {0, 0, 0, 6, 0}}, {2, 4, 1, {3, 2, 2, 0, 0}}, {2, 4, 1, {3, 0, 3, 0, 2}},
      {2, 4, 2, {5, 0, 0, 0, 0}}, {2, 4, 2, {0, 0, 5, 3, 0}}, {2, 4, 2, {0, 1, 4, 2, 0}},
      {2, 4, 3, {0, 0, 0, 0, 6}}, {2, 0, 1, {0, 0, 3, 2, 2}}, {2, 0, 1, {2, 3, 0, 3, 0}},
      {2, 0, 2, {0, 0, 0, 5, 0}}, {2, 0, 2, {0, 0, 0, 5, 3}}, {2, 0, 2, {0, 0, 1, 4, 2}},
      {2, 0, 3, {6, 0, 0, 0, 0}}, {2, 2, 1, {2, 3, 0, 0, 2}}, {2, 2, 1, {3, 0, 2, 3, 0}},
      {2, 2, 2, {0, 0, 5, 0, 0}}, {2, 2, 2, {0, 5, 3, 0, 0}}, {2, 2, 2, {4, 2, 0, 0, 1}},
      {2, 2, 3, {0, 0, 6, 0, 0}}, {3, 1, 3, {3, 0, 3, 3, 5}}, {3, 1, 4, {7, 0, 0, 0, 0}},
      {3, 1, 4, {6, 3, 0, 0, 3}}, {3, 1, 5, {7, 3, 0, 0, 0}}, {3, 3, 3, {3, 5, 3, 0, 3}},
      {3, 3, 4, {0, 0, 7, 0, 0}}, {3, 3, 4, {0, 3, 6, 3, 0}}, {3, 3, 5, {0, 0, 7, 3, 0}},
      {3, 4, 3, {3, 3, 5, 3, 0}}, {3, 4, 4, {0, 0, 0, 7, 0}}, {3, 4, 4, {0, 0, 3, 6, 3}},
      {3, 4, 5, {0, 0, 0, 7, 3}}, {3, 0, 3, {0, 3, 3, 5, 3}}, {3, 0, 4, {0, 0, 0, 0, 7}},
      {3, 0, 4, {3, 0, 0, 3, 6}}, {3, 0, 5, {3, 0, 0, 0, 7}}, {3, 2, 3, {5, 3, 0, 3, 3}},
      {3, 2, 4, {0, 7, 0, 0, 0}}, {3, 2, 4, {3, 6, 3, 0, 0}}, {3, 2, 5, {0, 7, 3, 0, 0}},
  }};
  std::vector<SplendorCard> cards;
  cards.reserve(defs.size());
  for (const auto& def : defs) {
    SplendorCard c;
    c.tier = def.tier;
    c.bonus = def.bonus;
    c.points = def.points;
    for (int i = 0; i < kColorCount; ++i) {
      c.cost[static_cast<size_t>(i)] = def.cost[static_cast<size_t>(i)];
    }
    cards.push_back(c);
  }
  return cards;
}

}  // namespace

const std::vector<SplendorCard>& splendor_card_pool() {
  static const std::vector<SplendorCard> cards = build_card_pool();
  return cards;
}

const std::array<std::array<std::int8_t, kColorCount>, 12>& splendor_nobles() {
  static const std::array<std::array<std::int8_t, kColorCount>, 12> nobles{{
      {{0, 0, 4, 4, 0}}, {{0, 0, 0, 4, 4}}, {{0, 4, 4, 0, 0}},
      {{4, 0, 0, 0, 4}}, {{4, 4, 0, 0, 0}}, {{3, 0, 0, 3, 3}},
      {{3, 3, 3, 0, 0}}, {{0, 0, 3, 3, 3}}, {{0, 3, 3, 3, 0}},
      {{3, 3, 0, 0, 3}}, {{4, 0, 0, 4, 0}}, {{0, 3, 3, 0, 3}},
  }};
  return nobles;
}

template <int NPlayers>
SplendorPersistentState<NPlayers>::SplendorPersistentState(
    std::shared_ptr<const SplendorPersistentNode<NPlayers>> node)
    : node_(std::move(node)) {}

template <int NPlayers>
SplendorPersistentState<NPlayers> SplendorPersistentState<NPlayers>::root_from_state(std::mt19937_64& rng) {
  using Cfg = SplendorConfig<NPlayers>;
  auto data = std::make_shared<SplendorData<NPlayers>>();
  data->current_player = 0;
  data->first_player = 0;
  data->plies = 0;
  data->final_round_remaining = -1;
  data->stage = static_cast<std::int8_t>(SplendorTurnStage::kNormal);
  data->pending_returns = 0;
  data->pending_noble_slots.fill(-1);
  data->pending_nobles_size = 0;
  data->winner = -1;
  data->terminal = false;
  data->shared_victory = false;
  data->scores = {};
  for (int c = 0; c < kColorCount; ++c) {
    data->bank[static_cast<size_t>(c)] = static_cast<std::int8_t>(Cfg::kGemCount);
  }
  data->bank[5] = static_cast<std::int8_t>(Cfg::kGoldCount);
  for (int p = 0; p < Cfg::kPlayers; ++p) {
    data->player_gems[p].fill(0);
    data->player_bonuses[p].fill(0);
    data->player_points[p] = 0;
    data->player_cards_count[p] = 0;
    data->player_nobles_count[p] = 0;
    data->reserved[p] = {{-1, -1, -1}};
    data->reserved_visible[p] = {{0, 0, 0}};
    data->reserved_size[p] = 0;
  }
  for (int t = 0; t < 3; ++t) {
    data->tableau[t] = {{-1, -1, -1, -1}};
    data->tableau_size[t] = 0;
  }

  const auto& cards = splendor_card_pool();
  for (int id = 0; id < static_cast<int>(cards.size()); ++id) {
    const int tier = static_cast<int>(cards[static_cast<size_t>(id)].tier);
    if (tier >= 1 && tier <= 3) {
      data->decks[static_cast<size_t>(tier - 1)].push_back(static_cast<std::int16_t>(id));
    }
  }
  for (int t = 0; t < 3; ++t) {
    auto& deck = data->decks[static_cast<size_t>(t)];
    for (int k = 0; k < 4 && !deck.empty(); ++k) {
      data->tableau[static_cast<size_t>(t)][static_cast<size_t>(k)] =
          take_random_from_vector(deck, rng);
      data->tableau_size[static_cast<size_t>(t)] += 1;
    }
    // First-class public per-tier size — sync to decks[t] once at root.
    // Subsequent updates flow through draw_random_from_deck on the truth
    // side; observers receive deck_sizes via apply_public_snapshot from the wire.
    data->deck_sizes[static_cast<size_t>(t)] =
        static_cast<std::int16_t>(deck.size());
  }

  std::vector<int> noble_ids(12);
  for (int i = 0; i < 12; ++i) noble_ids[static_cast<size_t>(i)] = i;
  data->nobles_size = static_cast<std::int8_t>(Cfg::kNobleCount);
  data->nobles.fill(-1);
  for (int i = 0; i < Cfg::kNobleCount; ++i) {
    data->nobles[static_cast<size_t>(i)] = static_cast<std::int16_t>(
        take_random_from_vector(noble_ids, rng));
  }

  auto node = std::make_shared<SplendorPersistentNode<NPlayers>>();
  node->action_from_parent = -1;
  node->materialized = std::move(data);
  return SplendorPersistentState<NPlayers>(std::move(node));
}

template <int NPlayers>
const SplendorData<NPlayers>& SplendorPersistentState<NPlayers>::data() const {
  if (!node_) {
    throw std::runtime_error("SplendorPersistentState is not initialized");
  }
  // Materialization is eager. Every node is created with
  // materialized!=nullptr by `advance` (which runs apply_action_copy
  // with the caller's rng) or by `root_from_state`.
  if (!node_->materialized) {
    throw std::runtime_error(
        "SplendorPersistentState node has no materialized data");
  }
  return *node_->materialized;
}

template <int NPlayers>
SplendorPersistentState<NPlayers> SplendorPersistentState<NPlayers>::advance(
    ActionId action, std::mt19937_64& rng) const {
  // Eagerly materialize the child by running apply_action_copy with
  // the caller's rng. This is the only place rule transitions consume
  // randomness (other than initial setup).
  const SplendorData<NPlayers>& parent_data = data();
  auto child = std::make_shared<SplendorData<NPlayers>>(
      SplendorRules<NPlayers>::apply_action_copy(parent_data, action, rng));
  auto node = std::make_shared<SplendorPersistentNode<NPlayers>>();
  node->parent = node_;
  node->action_from_parent = action;
  node->materialized = std::move(child);
  return SplendorPersistentState<NPlayers>(std::move(node));
}

template <int NPlayers>
StateHash64 SplendorPersistentState<NPlayers>::state_hash() const {
  using Cfg = SplendorConfig<NPlayers>;
  const SplendorData<NPlayers>& d = data();
  std::size_t h = 0;
  hash_combine(h,static_cast<std::size_t>(d.current_player + 3));
  hash_combine(h,static_cast<std::size_t>(d.first_player + 5));
  hash_combine(h,static_cast<std::size_t>(d.plies + 17));
  hash_combine(h,static_cast<std::size_t>(d.final_round_remaining + 9));
  hash_combine(h,static_cast<std::size_t>(d.stage + 21));
  hash_combine(h,static_cast<std::size_t>(d.pending_returns + 25));
  hash_combine(h,static_cast<std::size_t>(d.pending_nobles_size + 27));
  for (auto slot : d.pending_noble_slots) hash_combine(h,static_cast<std::size_t>(slot + 29));
  hash_combine(h,static_cast<std::size_t>(d.winner + 11));
  hash_combine(h,static_cast<std::size_t>(d.terminal ? 1 : 0));
  for (int v : d.scores) hash_combine(h,static_cast<std::size_t>(v + 101));
  for (auto v : d.bank) hash_combine(h,static_cast<std::size_t>(v + 7));
  const int actor = d.current_player;
  for (int p = 0; p < Cfg::kPlayers; ++p) {
    for (auto v : d.player_gems[p]) hash_combine(h,static_cast<std::size_t>(v + 13));
    for (auto v : d.player_bonuses[p]) hash_combine(h,static_cast<std::size_t>(v + 19));
    hash_combine(h,static_cast<std::size_t>(d.player_points[p] + 23));
    hash_combine(h,static_cast<std::size_t>(d.player_cards_count[p] + 29));
    hash_combine(h,static_cast<std::size_t>(d.player_nobles_count[p] + 31));
    hash_combine(h,static_cast<std::size_t>(d.reserved_size[p] + 37));
    for (int i = 0; i < 3; ++i) {
      const std::int16_t cid = d.reserved[p][static_cast<size_t>(i)];
      const bool visible_to_actor = d.reserved_visible[p][static_cast<size_t>(i)] != 0;
      if (p == actor || visible_to_actor) {
        hash_combine(h,static_cast<std::size_t>(cid + 41));
      } else {
        hash_combine(h,static_cast<std::size_t>(-1 + 41));
      }
      hash_combine(h,static_cast<std::size_t>((visible_to_actor ? 1 : 0) + 43));
    }
  }
  for (int t = 0; t < 3; ++t) {
    hash_combine(h,static_cast<std::size_t>(d.tableau_size[t] + 47));
    for (auto cid : d.tableau[t]) hash_combine(h,static_cast<std::size_t>(cid + 53));
    hash_combine(h,static_cast<std::size_t>(d.decks[t].size() + 59));
  }
  hash_combine(h,static_cast<std::size_t>(d.nobles_size + 67));
  for (int i = 0; i < Cfg::kNobleCount; ++i) {
    hash_combine(h,static_cast<std::size_t>(d.nobles[static_cast<size_t>(i)] + 71));
  }
  return static_cast<StateHash64>(h);
}

template <int NPlayers>
SplendorState<NPlayers>::SplendorState() { reset_with_seed(0xC0FFEEu); }

template <int NPlayers>
const viz::VisibilitySchema& SplendorState<NPlayers>::schema() {
  static const viz::VisibilitySchema s = []() {
    viz::VisibilitySchema schema;
    schema.n_players = Cfg::kPlayers;

    // ---- public scalars ----
    viz::declare_field(schema, "current_player",
                       viz::all_public({}, Cfg::kPlayers));
    viz::declare_field(schema, "first_player",
                       viz::all_public({}, Cfg::kPlayers));
    viz::declare_field(schema, "plies", viz::all_public({}, Cfg::kPlayers));
    viz::declare_field(schema, "final_round_remaining",
                       viz::all_public({}, Cfg::kPlayers));
    viz::declare_field(schema, "stage", viz::all_public({}, Cfg::kPlayers));
    viz::declare_field(schema, "pending_returns",
                       viz::all_public({}, Cfg::kPlayers));
    viz::declare_field(schema, "pending_nobles_size",
                       viz::all_public({}, Cfg::kPlayers));
    viz::declare_field(schema, "winner", viz::all_public({}, Cfg::kPlayers));
    viz::declare_field(schema, "terminal", viz::all_public({}, Cfg::kPlayers));
    viz::declare_field(schema, "shared_victory",
                       viz::all_public({}, Cfg::kPlayers));
    viz::declare_field(schema, "nobles_size",
                       viz::all_public({}, Cfg::kPlayers));

    // ---- public 1D ----
    viz::declare_field(
        schema, "pending_noble_slots",
        viz::all_public({Cfg::kNobleCount}, Cfg::kPlayers));
    viz::declare_field(schema, "scores",
                       viz::all_public({Cfg::kPlayers}, Cfg::kPlayers));
    viz::declare_field(schema, "bank",
                       viz::all_public({kTokenTypes}, Cfg::kPlayers));
    viz::declare_field(schema, "player_points",
                       viz::all_public({Cfg::kPlayers}, Cfg::kPlayers));
    viz::declare_field(schema, "player_cards_count",
                       viz::all_public({Cfg::kPlayers}, Cfg::kPlayers));
    viz::declare_field(schema, "player_nobles_count",
                       viz::all_public({Cfg::kPlayers}, Cfg::kPlayers));
    viz::declare_field(schema, "reserved_size",
                       viz::all_public({Cfg::kPlayers}, Cfg::kPlayers));
    viz::declare_field(schema, "tableau_size",
                       viz::all_public({3}, Cfg::kPlayers));
    // deck_sizes[3]: per-tier deck size. Public count, hidden contents.
    // Size is a derived public quantity off `decks[t].size()` —
    // read/write_field_slot project to / from `decks[t].resize(...)`;
    // contents are filled by randomize_unseen.
    viz::declare_field(schema, "deck_sizes",
                       viz::all_public({3}, Cfg::kPlayers));
    viz::declare_field(schema, "nobles",
                       viz::all_public({Cfg::kNobleCount}, Cfg::kPlayers));

    // ---- public 2D ----
    viz::declare_field(
        schema, "player_gems",
        viz::all_public({Cfg::kPlayers, kTokenTypes}, Cfg::kPlayers));
    viz::declare_field(
        schema, "player_bonuses",
        viz::all_public({Cfg::kPlayers, kColorCount}, Cfg::kPlayers));
    viz::declare_field(schema, "tableau",
                       viz::all_public({3, 4}, Cfg::kPlayers));
    // reserved_visible[p][i]: public face-up flag per reserve slot.
    viz::declare_field(schema, "reserved_visible",
                       viz::all_public({Cfg::kPlayers, 3}, Cfg::kPlayers));

    // ---- private (per-owner) ----
    // reserved[p][i]: card id of player p's reserve slot i. Base
    // owner-only; rules call viz::reveal_slot(reserved, {p, i}) when a
    // reserve becomes face-up, reset_to_base when bought/discarded.
    // The encoder reads MaskedState's `reserved` slot directly — face-
    // down opp slots arrive as kPlaceholderInt32.
    viz::declare_field(
        schema, "reserved",
        viz::owner_only_first_axis({Cfg::kPlayers, 3}, Cfg::kPlayers));

    return schema;
  }();
  return s;
}

template <int NPlayers>
void SplendorState<NPlayers>::reset_with_seed(std::uint64_t seed) {
  IGameState::reset_step_count_base();
  // One-shot rng for the initial deck/noble setup. RNG is not stored
  // on state — caller of do_action_fast supplies its own rng for any
  // subsequent randomness (tableau refills after buys).
  std::mt19937_64 init_rng(seed);
  persistent = SplendorPersistentState<NPlayers>::root_from_state(init_rng);
  undo_stack.clear();
  viz::init_viz(*this, schema());
}

template <int NPlayers>
StateHash64 SplendorState<NPlayers>::state_hash() const {
  return persistent.state_hash();
}

template <int NPlayers>
void SplendorState<NPlayers>::hash_field_slot(
    Hasher& h, const std::string& name,
    const std::vector<int>& idx) const {
  using Cfg = SplendorConfig<NPlayers>;
  const SplendorData<NPlayers>& d = persistent.data();
  // 0-D scalar fields.
  if (name == "current_player") { h.add(d.current_player + 3); return; }
  if (name == "first_player") { h.add(d.first_player + 5); return; }
  if (name == "plies") { h.add(d.plies + 17); return; }
  if (name == "final_round_remaining") { h.add(d.final_round_remaining + 9); return; }
  if (name == "stage") { h.add(d.stage + 21); return; }
  if (name == "pending_returns") { h.add(d.pending_returns + 25); return; }
  if (name == "pending_nobles_size") { h.add(d.pending_nobles_size + 27); return; }
  if (name == "winner") { h.add(d.winner + 11); return; }
  if (name == "terminal") { h.add(d.terminal ? 1 : 0); return; }
  if (name == "shared_victory") { h.add(d.shared_victory ? 1 : 0); return; }
  if (name == "nobles_size") { h.add(d.nobles_size + 67); return; }
  // 1-D fields.
  if (name == "pending_noble_slots") {
    h.add(d.pending_noble_slots[static_cast<size_t>(idx[0])] + 29); return;
  }
  if (name == "scores") {
    h.add(d.scores[static_cast<size_t>(idx[0])] + 101); return;
  }
  if (name == "bank") {
    h.add(d.bank[static_cast<size_t>(idx[0])] + 7); return;
  }
  if (name == "player_points") {
    h.add(d.player_points[static_cast<size_t>(idx[0])] + 23); return;
  }
  if (name == "player_cards_count") {
    h.add(d.player_cards_count[static_cast<size_t>(idx[0])] + 29); return;
  }
  if (name == "player_nobles_count") {
    h.add(d.player_nobles_count[static_cast<size_t>(idx[0])] + 31); return;
  }
  if (name == "reserved_size") {
    h.add(d.reserved_size[static_cast<size_t>(idx[0])] + 37); return;
  }
  if (name == "tableau_size") {
    h.add(d.tableau_size[static_cast<size_t>(idx[0])] + 47); return;
  }
  if (name == "deck_sizes") {
    // First-class field — read d.deck_sizes[t] directly. Rules
    // maintain it at draw_random_from_deck; observers receive it via
    // apply_public_snapshot. Do NOT read d.decks[t].size() here: decks are
    // hidden contents, and tying the hash to a hidden container's
    // size forces observers to re-randomize every ply just to
    // recover this value (the bug Plan A surfaced).
    h.add(static_cast<std::int64_t>(d.deck_sizes[static_cast<size_t>(idx[0])]) + 59);
    return;
  }
  if (name == "nobles") {
    h.add(d.nobles[static_cast<size_t>(idx[0])] + 71); return;
  }
  // 2-D fields.
  if (name == "player_gems") {
    h.add(d.player_gems[static_cast<size_t>(idx[0])][static_cast<size_t>(idx[1])] + 13);
    return;
  }
  if (name == "player_bonuses") {
    h.add(d.player_bonuses[static_cast<size_t>(idx[0])][static_cast<size_t>(idx[1])] + 19);
    return;
  }
  if (name == "tableau") {
    h.add(d.tableau[static_cast<size_t>(idx[0])][static_cast<size_t>(idx[1])] + 53);
    return;
  }
  if (name == "reserved_visible") {
    const std::int8_t v = d.reserved_visible[static_cast<size_t>(idx[0])][static_cast<size_t>(idx[1])];
    h.add((v != 0 ? 1 : 0) + 43);
    return;
  }
  if (name == "reserved") {
    h.add(d.reserved[static_cast<size_t>(idx[0])][static_cast<size_t>(idx[1])] + 41);
    return;
  }
  // Unknown field: deliberately silent. Adding new schema fields
  // without a corresponding hash_field_slot entry is caught by
  // tests/framework/test_public_snapshot_round_trip.py (the public
  // hash will start drifting from the snapshot path).
  (void)Cfg::kPlayers;
}

// Walker-driven snapshot wire I/O dispatchers. `read_field_slot`
// returns std::any of int / bool. `write_field_slot` is gated behind
// the COW persistent — register.cpp opens a `mutate_persistent` block
// around `viz::apply_public_snapshot(...)` so all per-slot writes land
// on the detached writable copy.

template <int NPlayers>
std::any SplendorState<NPlayers>::read_field_slot(
    const std::string& name, const std::vector<int>& idx) const {
  const SplendorData<NPlayers>& d = persistent.data();
  // 0-D scalars.
  if (name == "current_player") return std::any(static_cast<int>(d.current_player));
  if (name == "first_player") return std::any(static_cast<int>(d.first_player));
  if (name == "plies") return std::any(static_cast<int>(d.plies));
  if (name == "final_round_remaining") return std::any(static_cast<int>(d.final_round_remaining));
  if (name == "stage") return std::any(static_cast<int>(d.stage));
  if (name == "pending_returns") return std::any(static_cast<int>(d.pending_returns));
  if (name == "pending_nobles_size") return std::any(static_cast<int>(d.pending_nobles_size));
  if (name == "winner") return std::any(static_cast<int>(d.winner));
  if (name == "terminal") return std::any(static_cast<bool>(d.terminal));
  if (name == "shared_victory") return std::any(static_cast<bool>(d.shared_victory));
  if (name == "nobles_size") return std::any(static_cast<int>(d.nobles_size));
  // 1-D fields.
  if (name == "pending_noble_slots") {
    return std::any(static_cast<int>(d.pending_noble_slots[static_cast<size_t>(idx[0])]));
  }
  if (name == "scores") return std::any(static_cast<int>(d.scores[static_cast<size_t>(idx[0])]));
  if (name == "bank") return std::any(static_cast<int>(d.bank[static_cast<size_t>(idx[0])]));
  if (name == "player_points") {
    return std::any(static_cast<int>(d.player_points[static_cast<size_t>(idx[0])]));
  }
  if (name == "player_cards_count") {
    return std::any(static_cast<int>(d.player_cards_count[static_cast<size_t>(idx[0])]));
  }
  if (name == "player_nobles_count") {
    return std::any(static_cast<int>(d.player_nobles_count[static_cast<size_t>(idx[0])]));
  }
  if (name == "reserved_size") {
    return std::any(static_cast<int>(d.reserved_size[static_cast<size_t>(idx[0])]));
  }
  if (name == "tableau_size") {
    return std::any(static_cast<int>(d.tableau_size[static_cast<size_t>(idx[0])]));
  }
  if (name == "deck_sizes") {
    return std::any(static_cast<int>(d.deck_sizes[static_cast<size_t>(idx[0])]));
  }
  if (name == "nobles") return std::any(static_cast<int>(d.nobles[static_cast<size_t>(idx[0])]));
  // 2-D fields.
  if (name == "player_gems") {
    return std::any(static_cast<int>(
        d.player_gems[static_cast<size_t>(idx[0])][static_cast<size_t>(idx[1])]));
  }
  if (name == "player_bonuses") {
    return std::any(static_cast<int>(
        d.player_bonuses[static_cast<size_t>(idx[0])][static_cast<size_t>(idx[1])]));
  }
  if (name == "tableau") {
    return std::any(static_cast<int>(
        d.tableau[static_cast<size_t>(idx[0])][static_cast<size_t>(idx[1])]));
  }
  if (name == "reserved_visible") {
    return std::any(
        d.reserved_visible[static_cast<size_t>(idx[0])][static_cast<size_t>(idx[1])] != 0
            ? 1 : 0);
  }
  if (name == "reserved") {
    // Only emitted when public — walker visits the slot iff its viewer
    // bit is 1 for perspective. Rules maintain `viz_["reserved"]` in
    // sync with `reserved_visible` via `viz::reveal_slot` /
    // `viz::reset_to_base`; the receiver picks up the same bits via the
    // wire's `__viz__` slice (no per-game derivation hook).
    return std::any(static_cast<int>(
        d.reserved[static_cast<size_t>(idx[0])][static_cast<size_t>(idx[1])]));
  }
  return {};
}

template <int NPlayers>
void SplendorState<NPlayers>::write_field_slot(
    const std::string& name, const std::vector<int>& idx,
    const std::any& value) {
  // Helper: cast value to int (handles AnyMap traversal where int may
  // round-trip through pybind as int).
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

  // Detach a writable SplendorData and reseat once per write. (register.cpp
  // wraps the whole apply_public_snapshot call in a single mutate_persistent
  // so typical N-write batch is still O(1) reseats; this code path is only
  // hit when called outside a wrapping mutate_persistent — kept correct
  // either way.)
  SplendorData<NPlayers> data = persistent.data();
  auto& d = data;

  bool handled = true;
  if (name == "current_player") d.current_player = static_cast<std::int8_t>(as_int());
  else if (name == "first_player") d.first_player = static_cast<std::int8_t>(as_int());
  else if (name == "plies") d.plies = static_cast<std::int16_t>(as_int());
  else if (name == "final_round_remaining") d.final_round_remaining = static_cast<std::int8_t>(as_int());
  else if (name == "stage") d.stage = static_cast<std::int8_t>(as_int());
  else if (name == "pending_returns") d.pending_returns = static_cast<std::int8_t>(as_int());
  else if (name == "pending_nobles_size") d.pending_nobles_size = static_cast<std::int8_t>(as_int());
  else if (name == "winner") d.winner = static_cast<std::int8_t>(as_int());
  else if (name == "terminal") d.terminal = as_bool();
  else if (name == "shared_victory") d.shared_victory = as_bool();
  else if (name == "nobles_size") d.nobles_size = static_cast<std::int8_t>(as_int());
  else if (name == "pending_noble_slots") {
    d.pending_noble_slots[static_cast<size_t>(idx[0])] = static_cast<std::int8_t>(as_int());
  }
  else if (name == "scores") d.scores[static_cast<size_t>(idx[0])] = as_int();
  else if (name == "bank") d.bank[static_cast<size_t>(idx[0])] = static_cast<std::int8_t>(as_int());
  else if (name == "player_points") {
    d.player_points[static_cast<size_t>(idx[0])] = static_cast<std::int16_t>(as_int());
  }
  else if (name == "player_cards_count") {
    d.player_cards_count[static_cast<size_t>(idx[0])] = static_cast<std::int16_t>(as_int());
  }
  else if (name == "player_nobles_count") {
    d.player_nobles_count[static_cast<size_t>(idx[0])] = static_cast<std::int16_t>(as_int());
  }
  else if (name == "reserved_size") {
    d.reserved_size[static_cast<size_t>(idx[0])] = static_cast<std::int8_t>(as_int());
  }
  else if (name == "tableau_size") {
    d.tableau_size[static_cast<size_t>(idx[0])] = static_cast<std::int8_t>(as_int());
  }
  else if (name == "deck_sizes") {
    // First-class public field. Write `deck_sizes[t]` directly; the
    // hidden `decks[t]` multiset is tracker territory (sim-entry
    // randomize_unseen fills it from belief). We still resize the
    // observer's `decks[t]` to match so the legacy full-state
    // `state_hash()` (used by tail solver / transposition debug) reads
    // a consistent length — but the canonical schema-walker hash source
    // is `deck_sizes[t]`.
    const int target = as_int();
    if (target >= 0) {
      d.deck_sizes[static_cast<size_t>(idx[0])] =
          static_cast<std::int16_t>(target);
      auto& deck = d.decks[static_cast<size_t>(idx[0])];
      if (static_cast<int>(deck.size()) > target) {
        deck.resize(static_cast<size_t>(target));
      } else {
        while (static_cast<int>(deck.size()) < target) {
          deck.push_back(-1);
        }
      }
    }
  }
  else if (name == "nobles") {
    d.nobles[static_cast<size_t>(idx[0])] = static_cast<std::int16_t>(as_int());
  }
  else if (name == "player_gems") {
    d.player_gems[static_cast<size_t>(idx[0])][static_cast<size_t>(idx[1])] =
        static_cast<std::int8_t>(as_int());
  }
  else if (name == "player_bonuses") {
    d.player_bonuses[static_cast<size_t>(idx[0])][static_cast<size_t>(idx[1])] =
        static_cast<std::int8_t>(as_int());
  }
  else if (name == "tableau") {
    d.tableau[static_cast<size_t>(idx[0])][static_cast<size_t>(idx[1])] =
        static_cast<std::int16_t>(as_int());
  }
  else if (name == "reserved_visible") {
    d.reserved_visible[static_cast<size_t>(idx[0])][static_cast<size_t>(idx[1])] =
        static_cast<std::int8_t>(as_int());
  }
  else if (name == "reserved") {
    d.reserved[static_cast<size_t>(idx[0])][static_cast<size_t>(idx[1])] =
        static_cast<std::int16_t>(as_int());
  }
  else handled = false;

  if (!handled) return;
  // Reseat persistent → new shared SplendorData.
  auto node = std::make_shared<SplendorPersistentNode<NPlayers>>();
  node->action_from_parent = -1;
  node->materialized = std::make_shared<const SplendorData<NPlayers>>(std::move(data));
  persistent = SplendorPersistentState<NPlayers>(std::move(node));
  undo_stack.clear();
}

// mask_field_slot writes kPlaceholder into the only hidden field
// Splendor declares — `reserved` (face-down id, owner_only_first_axis
// base). All other fields are all_public, so the walker never visits
// them as hidden.
template <int NPlayers>
void SplendorState<NPlayers>::mask_field_slot(
    const std::string& name, const std::vector<int>& idx) {
  if (name != "reserved") return;
  SplendorData<NPlayers> data = persistent.data();
  data.reserved[static_cast<size_t>(idx[0])][static_cast<size_t>(idx[1])] =
      static_cast<std::int16_t>(kPlaceholderInt32);
  auto node = std::make_shared<SplendorPersistentNode<NPlayers>>();
  node->action_from_parent = -1;
  node->materialized = std::make_shared<const SplendorData<NPlayers>>(std::move(data));
  persistent = SplendorPersistentState<NPlayers>(std::move(node));
  undo_stack.clear();
}

// COW-aware mask_all_hidden_slots. Detach once into a single writable
// SplendorData, run the walker dispatch on it, reseat once at the end.
// Avoids one shared_ptr reseat per masked slot.
template <int NPlayers>
void SplendorState<NPlayers>::mask_all_hidden_slots(
    const viz::VisibilitySchema& schema, int perspective) {
  SplendorData<NPlayers> data = persistent.data();
  viz::for_each_hidden_slot(
      *this, schema, perspective,
      [&](const std::string& name, const std::vector<int>& idx,
          const viz::VizTensor& /*v*/) {
        if (name == "reserved") {
          data.reserved[static_cast<size_t>(idx[0])][static_cast<size_t>(idx[1])] =
              static_cast<std::int16_t>(kPlaceholderInt32);
        }
      });
  auto node = std::make_shared<SplendorPersistentNode<NPlayers>>();
  node->action_from_parent = -1;
  node->materialized = std::make_shared<const SplendorData<NPlayers>>(std::move(data));
  persistent = SplendorPersistentState<NPlayers>(std::move(node));
  undo_stack.clear();
}

template <int NPlayers>
int SplendorState<NPlayers>::current_player() const {
  return persistent.data().current_player;
}

template <int NPlayers>
int SplendorState<NPlayers>::first_player() const {
  return persistent.data().first_player;
}

template <int NPlayers>
bool SplendorState<NPlayers>::is_terminal() const {
  return persistent.data().terminal;
}

template <int NPlayers>
bool SplendorState<NPlayers>::is_turn_start() const {
  return static_cast<SplendorTurnStage>(persistent.data().stage) == SplendorTurnStage::kNormal;
}

template <int NPlayers>
int SplendorState<NPlayers>::winner() const {
  return persistent.data().winner;
}

template struct SplendorData<2>;
template struct SplendorData<3>;
template struct SplendorData<4>;
template class SplendorPersistentState<2>;
template class SplendorPersistentState<3>;
template class SplendorPersistentState<4>;
template struct SplendorState<2>;
template struct SplendorState<3>;
template struct SplendorState<4>;

}  // namespace board_ai::splendor
