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
  // Initial bag shuffle: one mt19937_64 stream from derive_rng for the
  // whole shuffle; framework guarantees subsequent derive_rng calls are
  // independent via incrementing draw_nonce_.
  {
    auto rng = this->derive_rng(0xa2ULL /* domain: azul_initial_bag */);
    std::shuffle(bag.begin(), bag.end(), rng);
  }
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
int AzulState<NPlayers>::draw_one_tile(std::mt19937_64& rng) {
  if (bag.empty()) {
    if (box_lid.empty()) {
      return -1;
    }
    bag.assign(box_lid.begin(), box_lid.end());
    box_lid.clear();
    std::shuffle(bag.begin(), bag.end(), rng);
  }
  const int t = bag.back();
  bag.pop_back();
  return t;
}

template <int NPlayers>
void AzulState<NPlayers>::refill_factories_from_rng() {
  factories = {};
  center = {};
  // One derive_rng stream covers the entire factory refill (and any
  // box→bag reshuffle that triggers mid-refill). The single mt19937_64
  // state lives only on the stack — it's not stored in game state.
  auto rng = this->derive_rng(0xa3ULL /* domain: azul_refill_factories */);
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
  hash_combine(h,static_cast<std::size_t>(bag.size()));
  for (std::int8_t t : bag) {
    hash_combine(h,static_cast<std::size_t>(t + 1));
  }
  hash_combine(h,static_cast<std::size_t>(box_lid.size()));
  for (std::int8_t t : box_lid) {
    hash_combine(h,static_cast<std::size_t>(t + 1));
  }
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
