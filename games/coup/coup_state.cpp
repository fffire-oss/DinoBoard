#include "coup_state.h"

#include <functional>
#include <random>

namespace board_ai::coup {

namespace {

CharId draw_from_deck_local(std::vector<CharId>& deck, std::mt19937_64& rng) {
  if (deck.empty()) return -1;
  std::uniform_int_distribution<size_t> dist(0, deck.size() - 1);
  const size_t idx = dist(rng);
  const CharId card = deck[idx];
  if (idx + 1 < deck.size()) {
    deck[idx] = deck.back();
  }
  deck.pop_back();
  return card;
}

}  // namespace

template <int NPlayers>
CoupState<NPlayers>::CoupState() = default;

template <int NPlayers>
void CoupState<NPlayers>::reset_with_seed(std::uint64_t seed) {
  IGameState::reset_with_seed_base(seed);
  data = CoupData<NPlayers>{};
  undo_stack.clear();

  data.court_deck.clear();
  data.court_deck.reserve(kTotalCards);
  for (CharId c = 0; c < kCharacterCount; ++c) {
    for (int i = 0; i < kCardsPerCharacter; ++i) {
      data.court_deck.push_back(c);
    }
  }

  // One derive_rng stream covers the opening 2-cards-per-player deal.
  {
    auto rng = this->derive_rng(0xc1ULL /* domain: coup_initial_deal */);
    for (int p = 0; p < NPlayers; ++p) {
      data.influence[p][0] = draw_from_deck_local(data.court_deck, rng);
      data.influence[p][1] = draw_from_deck_local(data.court_deck, rng);
    }
  }
  for (int p = 0; p < NPlayers; ++p) {
    data.revealed[p] = {false, false};
    data.coins[p] = kStartingCoins;
    data.alive[p] = true;
  }

  // exchange_drawn uses -1 as sentinel for "no card"; default-init of
  // std::array<int8_t, 2> zero-fills to {0, 0}, which randomize_unseen
  // treats as "two valid Duke cards in this player's exchange hand",
  // silently stealing 2 slots from court_deck on every sim. Match
  // advance_turn (which resets after exchange cycles end).
  data.exchange_drawn = {-1, -1};

  data.current_player = 0;
  data.first_player = 0;
  data.active_player = 0;
  data.stage = CoupStage::kDeclareAction;
}

template <int NPlayers>
StateHash64 CoupState<NPlayers>::state_hash(bool include_hidden_rng) const {
  std::size_t h = 0;
  auto combine = [&](std::size_t v) {
    h ^= v + 0x9e3779b9 + (h << 6) + (h >> 2);
  };

  combine(static_cast<std::size_t>(data.current_player));
  combine(static_cast<std::size_t>(data.stage));
  combine(static_cast<std::size_t>(data.ply));
  combine(static_cast<std::size_t>(data.active_player));
  combine(static_cast<std::size_t>(data.declared_action + 1));
  combine(static_cast<std::size_t>(data.action_target + 1));
  combine(static_cast<std::size_t>(data.challenger + 1));
  combine(static_cast<std::size_t>(data.blocker + 1));
  combine(static_cast<std::size_t>(data.challenge_check_index));
  combine(static_cast<std::size_t>(data.action_challenged));
  combine(static_cast<std::size_t>(data.action_challenge_succeeded));
  combine(static_cast<std::size_t>(data.counter_challenged));
  combine(static_cast<std::size_t>(data.counter_challenge_succeeded));

  for (int p = 0; p < NPlayers; ++p) {
    combine(static_cast<std::size_t>(data.alive[p]));
    combine(static_cast<std::size_t>(data.coins[p]));
    for (int s = 0; s < 2; ++s) {
      combine(static_cast<std::size_t>(data.revealed[p][s]));
      if (data.revealed[p][s]) {
        combine(static_cast<std::size_t>(data.influence[p][s] + 1));
      }
    }
  }

  combine(data.court_deck.size());

  if (include_hidden_rng) {
    for (int p = 0; p < NPlayers; ++p) {
      for (int s = 0; s < 2; ++s) {
        if (!data.revealed[p][s]) {
          combine(static_cast<std::size_t>(data.influence[p][s] + 1));
        }
      }
    }
    for (auto c : data.court_deck) {
      combine(static_cast<std::size_t>(c + 1));
    }
    combine(static_cast<std::size_t>(this->rng_salt_));
    combine(static_cast<std::size_t>(this->draw_nonce_));
    for (int i = 0; i < 2; ++i) {
      combine(static_cast<std::size_t>(data.exchange_drawn[i] + 1));
    }
  }

  return static_cast<StateHash64>(h);
}

template <int NPlayers>
void CoupState<NPlayers>::hash_public_fields(Hasher& h) const {
  // Coup public info: stage, plies, declared actions and challenges, coins,
  // alive status, revealed (dead) influences, public deck size.
  h.add(data.current_player);
  h.add(static_cast<int>(data.stage));
  h.add(data.ply);
  h.add(data.active_player);
  h.add(data.declared_action + 1);
  h.add(data.action_target + 1);
  h.add(data.challenger + 1);
  h.add(data.blocker + 1);
  h.add(data.challenge_check_index);
  h.add(data.action_challenged);
  h.add(data.action_challenge_succeeded);
  h.add(data.counter_challenged);
  h.add(data.counter_challenge_succeeded);
  for (int p = 0; p < NPlayers; ++p) {
    h.add(data.alive[p]);
    h.add(data.coins[p]);
    for (int s = 0; s < 2; ++s) {
      h.add(data.revealed[p][s]);
      if (data.revealed[p][s]) {
        h.add(data.influence[p][s] + 1);
      }
    }
  }
  h.add(data.court_deck.size());
}

template <int NPlayers>
void CoupState<NPlayers>::hash_private_fields(int player, Hasher& h) const {
  // Coup private: player's own face-down influence cards + any exchange-drawn
  // cards when player is currently exchanging.
  if (player < 0 || player >= NPlayers) return;
  for (int s = 0; s < 2; ++s) {
    if (!data.revealed[player][s]) {
      h.add(data.influence[player][s] + 100);
    }
  }
  const bool exchanging =
      data.stage == CoupStage::kExchangeReturn1 ||
      data.stage == CoupStage::kExchangeReturn2;
  if (exchanging && data.active_player == player) {
    for (int i = 0; i < 2; ++i) {
      h.add(data.exchange_drawn[i] + 200);
    }
  }
}

template <int NPlayers>
int CoupState<NPlayers>::current_player() const {
  return data.current_player;
}

template <int NPlayers>
int CoupState<NPlayers>::first_player() const {
  return data.first_player;
}

template <int NPlayers>
bool CoupState<NPlayers>::is_terminal() const {
  return data.terminal;
}

template <int NPlayers>
int CoupState<NPlayers>::winner() const {
  return data.winner;
}

template <int NPlayers>
bool CoupState<NPlayers>::is_turn_start() const {
  return data.stage == CoupStage::kDeclareAction;
}

template struct CoupData<2>;
template struct CoupData<3>;
template struct CoupData<4>;
template struct CoupState<2>;
template struct CoupState<3>;
template struct CoupState<4>;

}  // namespace board_ai::coup
