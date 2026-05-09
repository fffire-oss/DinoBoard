#include "azul_state.h"

#include <algorithm>
#include <functional>

namespace board_ai::azul {

template <int NPlayers>
AzulState<NPlayers>::AzulState() {
  reset_with_seed(0xC0FFEEu);
}

template <int NPlayers>
void AzulState<NPlayers>::reset_with_seed(std::uint64_t seed) {
  IGameState::reset_with_seed_base(seed);
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
  bag.clear();
  box_lid.clear();
  players = {};
  undo_stack.clear();
  persistent_tree_cache.tree.clear();
  persistent_tree_cache.chance_buckets.clear();
  persistent_tree_cache.sig_to_node.clear();
  bag.reserve(100);
  for (int c = 0; c < kColors; ++c) {
    for (int i = 0; i < 20; ++i) {
      bag.push_back(static_cast<std::int8_t>(c));
    }
  }
  // Phase 2: NO initial shuffle. The bag stores tiles in canonical
  // color order; randomness lives in draw_one_tile, which on every
  // call derives a fresh rng and picks a random INDEX from the
  // remaining bag. Equivalent to "uniformly draw from remaining
  // multiset" but never materializes future draw order into state.
  // Hash sees the bag as a multiset (already done in
  // hash_public_fields) so canonical vs. shuffled order produce the
  // same public hash for the same draw history.
  refill_factories_from_rng();
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
int AzulState<NPlayers>::draw_one_tile() {
  if (bag.empty()) {
    if (box_lid.empty()) {
      return -1;
    }
    // Box → bag refill: NO shuffle here either. Box was filled in the
    // order tiles came out of factories during the round (rules
    // pushes onto box_lid in canonical color order during factory
    // clear / floor discard). Either order is fine since the next
    // draw picks a random INDEX from the bag — the bag's storage
    // order is irrelevant to draw uniformity.
    bag.assign(box_lid.begin(), box_lid.end());
    box_lid.clear();
  }
  // Pick a uniformly-random index from the remaining bag. Each call
  // derives a fresh rng (domain = "azul_draw"), so draw_nonce_
  // increments per draw and consecutive draws are independent.
  // Modulo on a 64-bit mt19937_64 output: bag.size() <= 100, so the
  // bias is bounded by 100/2^64 ≈ 5e-18 — well below any observable
  // statistical effect. Avoids std::uniform_int_distribution which
  // golden-standard §2.3 lint discourages in rules / state code.
  auto rng = this->derive_rng(0xa4ULL /* domain: azul_draw */);
  const std::size_t idx = static_cast<std::size_t>(rng() % bag.size());
  const int t = bag[idx];
  // Swap-and-pop: O(1) removal that doesn't preserve ordering, but
  // we don't care about ordering — the bag is treated as a multiset.
  bag[idx] = bag.back();
  bag.pop_back();
  return t;
}

template <int NPlayers>
void AzulState<NPlayers>::refill_factories_from_rng() {
  factories = {};
  center = {};
  // Each draw_one_tile call derives its own rng. No shared stream
  // here — Phase 2 design treats every draw as an independent
  // randomness event keyed off (rng_salt, draw_nonce). draw_nonce
  // bumps once per tile.
  for (int f = 0; f < Cfg::kFactories; ++f) {
    for (int i = 0; i < 4; ++i) {
      const int color = draw_one_tile();
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
StateHash64 AzulState<NPlayers>::state_hash(bool include_hidden_rng) const {
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
  // Hash bag/box_lid as multisets (counts per color). Internal vector
  // order is irrelevant — Phase 2 treats the bag as an unordered
  // multiset, and hashing the order would split the DAG along an
  // axis no player observes (BUG-028 family).
  std::array<int, kColors> bag_counts_legacy{};
  for (std::int8_t t : bag) {
    if (t >= 0 && t < kColors) {
      ++bag_counts_legacy[static_cast<std::size_t>(t)];
    }
  }
  for (int count : bag_counts_legacy) hash_combine(h, static_cast<std::size_t>(count));
  std::array<int, kColors> box_counts_legacy{};
  for (std::int8_t t : box_lid) {
    if (t >= 0 && t < kColors) {
      ++box_counts_legacy[static_cast<std::size_t>(t)];
    }
  }
  for (int count : box_counts_legacy) hash_combine(h, static_cast<std::size_t>(count));
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
  if (include_hidden_rng) {
    hash_combine(h,static_cast<std::size_t>(this->rng_salt_));
    hash_combine(h,static_cast<std::size_t>(this->draw_nonce_));
  }
  return static_cast<StateHash64>(h);
}

template <int NPlayers>
void AzulState<NPlayers>::hash_public_fields(Hasher& h) const {
  // Azul is symmetric-random but fully PUBLIC with respect to composition:
  // everyone sees factory contents, center pile, each player's board, and
  // knows the bag composition derivably (bag = all tiles − placed − discarded).
  // The only thing nobody knows is the future draw ORDER, so hash bag/box
  // as multisets rather than vector order.
  h.add(current_player_);
  h.add(first_player_next_round);
  h.add(winner_ + 1);
  h.add(round_index);
  h.add(terminal ? 1 : 0);
  h.add(first_player_token_in_center ? 1 : 0);
  h.add(shared_victory ? 1 : 0);
  for (int s : scores) h.add(s);
  for (const auto& fac : factories) {
    for (std::uint8_t c : fac) h.add(c);
  }
  for (std::uint8_t c : center) h.add(c);
  std::array<int, kColors> bag_counts{};
  for (std::int8_t t : bag) {
    if (t >= 0 && t < kColors) {
      ++bag_counts[static_cast<std::size_t>(t)];
    }
  }
  for (int count : bag_counts) h.add(count);
  std::array<int, kColors> box_counts{};
  for (std::int8_t t : box_lid) {
    if (t >= 0 && t < kColors) {
      ++box_counts[static_cast<std::size_t>(t)];
    }
  }
  for (int count : box_counts) h.add(count);
  for (const auto& p : players) {
    for (std::uint8_t len : p.line_len) h.add(len);
    for (std::int8_t color : p.line_color) h.add(color + 1);
    for (std::uint8_t m : p.wall_mask) h.add(m);
    h.add(p.floor_count);
    for (std::int8_t f : p.floor) h.add(f + 1);
    h.add(p.score);
  }
}

template <int NPlayers>
void AzulState<NPlayers>::hash_private_fields(int /*player*/, Hasher& /*h*/) const {
  // Azul has no non-symmetric private info. Bag composition is
  // deterministically derivable from public placements; only order is random.
}

template class AzulState<2>;
template class AzulState<3>;
template class AzulState<4>;

}  // namespace board_ai::azul
