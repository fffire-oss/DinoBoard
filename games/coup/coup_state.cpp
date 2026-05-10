#include "coup_state.h"

#include <functional>
#include <random>

#include "../../engine/core/viz_runtime.h"

namespace board_ai::coup {

namespace {
// Cards in hand: [N players][2 slots]. Owner_only viz means viewer p
// sees only influence[p][*]. Public revealed[p][s] flags (kept in
// all_public space) are how rules signal "this slot is now public" —
// a Phase 3+ consumer that walks influence will gate on revealed[p][s]
// rather than mutating viz on each reveal/loss, since the public flag
// is already the canonical "this card is now face-up" signal.
constexpr int kInfluencePerPlayer = 2;
constexpr int kExchangeDrawSlots = 2;
}  // namespace

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
const viz::VisibilitySchema& CoupState<NPlayers>::schema() {
  // Built once on first use per template instantiation.
  static const viz::VisibilitySchema s = []() {
    viz::VisibilitySchema schema;
    schema.n_players = Cfg::kPlayers;

    // ---- public scalars ----
    viz::declare_field(schema, "current_player",
                       viz::all_public({}, Cfg::kPlayers));
    viz::declare_field(schema, "first_player",
                       viz::all_public({}, Cfg::kPlayers));
    viz::declare_field(schema, "winner", viz::all_public({}, Cfg::kPlayers));
    viz::declare_field(schema, "terminal", viz::all_public({}, Cfg::kPlayers));
    viz::declare_field(schema, "ply", viz::all_public({}, Cfg::kPlayers));
    viz::declare_field(schema, "stage", viz::all_public({}, Cfg::kPlayers));
    viz::declare_field(schema, "active_player",
                       viz::all_public({}, Cfg::kPlayers));
    viz::declare_field(schema, "declared_action",
                       viz::all_public({}, Cfg::kPlayers));
    viz::declare_field(schema, "action_target",
                       viz::all_public({}, Cfg::kPlayers));
    viz::declare_field(schema, "claimed_character",
                       viz::all_public({}, Cfg::kPlayers));
    viz::declare_field(schema, "challenger",
                       viz::all_public({}, Cfg::kPlayers));
    viz::declare_field(schema, "challenge_loser",
                       viz::all_public({}, Cfg::kPlayers));
    viz::declare_field(schema, "action_challenged",
                       viz::all_public({}, Cfg::kPlayers));
    viz::declare_field(schema, "action_challenge_succeeded",
                       viz::all_public({}, Cfg::kPlayers));
    viz::declare_field(schema, "blocker", viz::all_public({}, Cfg::kPlayers));
    viz::declare_field(schema, "block_character",
                       viz::all_public({}, Cfg::kPlayers));
    viz::declare_field(schema, "counter_challenged",
                       viz::all_public({}, Cfg::kPlayers));
    viz::declare_field(schema, "counter_challenge_succeeded",
                       viz::all_public({}, Cfg::kPlayers));
    viz::declare_field(schema, "challenge_check_index",
                       viz::all_public({}, Cfg::kPlayers));
    viz::declare_field(schema, "exchange_held_count",
                       viz::all_public({}, Cfg::kPlayers));

    // ---- public per-player 1D ----
    viz::declare_field(schema, "coins",
                       viz::all_public({Cfg::kPlayers}, Cfg::kPlayers));
    viz::declare_field(schema, "alive",
                       viz::all_public({Cfg::kPlayers}, Cfg::kPlayers));
    // revealed[p][s]: public face-up flag per influence slot. all_public
    // by definition — it's the "is this card now face-up?" public bit.
    viz::declare_field(
        schema, "revealed",
        viz::all_public({Cfg::kPlayers, kInfluencePerPlayer}, Cfg::kPlayers));

    // ---- private per-player ----
    // influence[N][2]: face-down cards. Owner-only base; once
    // revealed[p][s] flips public, downstream consumers (encoder /
    // hash) read this card publicly via the revealed gate. We do NOT
    // mutate viz on reveal — the public flag is the canonical signal.
    viz::declare_field(
        schema, "influence",
        viz::owner_only_first_axis({Cfg::kPlayers, kInfluencePerPlayer},
                                   Cfg::kPlayers));

    // ---- hidden ----
    // exchange_drawn[2]: two cards drawn from court_deck during the
    // Exchange action, visible only to active_player. Base is hidden;
    // rules reveal_slot_to(active_player) on draw and reset_to_base on
    // return-to-deck. (Reveal wiring lands separately; this PR only
    // declares the base.)
    viz::declare_field(schema, "exchange_drawn",
                       viz::all_hidden({kExchangeDrawSlots}, Cfg::kPlayers));

    return schema;
  }();
  return s;
}

template <int NPlayers>
CoupState<NPlayers>::CoupState() = default;

template <int NPlayers>
void CoupState<NPlayers>::reset_with_seed(std::uint64_t seed) {
  IGameState::reset_step_count_base();
  data = CoupData<NPlayers>{};
  undo_stack.clear();

  data.court_deck.clear();
  data.court_deck.reserve(kTotalCards);
  for (CharId c = 0; c < kCharacterCount; ++c) {
    for (int i = 0; i < kCardsPerCharacter; ++i) {
      data.court_deck.push_back(c);
    }
  }

  // One-shot rng for opening 2-cards-per-player deal. RNG is not stored
  // on state — caller of do_action_fast supplies its own rng for any
  // subsequent randomness (e.g. challenge redraws).
  {
    std::mt19937_64 rng(seed);
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

  viz::init_viz(*this, schema());
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
