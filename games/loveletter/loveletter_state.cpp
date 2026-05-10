#include "loveletter_state.h"

#include <algorithm>
#include <cstdint>
#include <stdexcept>

#include "../../engine/core/viz_runtime.h"

namespace board_ai::loveletter {

template <int NPlayers>
const viz::VisibilitySchema& LoveLetterState<NPlayers>::schema() {
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

    // ---- public per-player 1D ----
    viz::declare_field(schema, "alive",
                       viz::all_public({Cfg::kPlayers}, Cfg::kPlayers));
    viz::declare_field(schema, "protected_flags",
                       viz::all_public({Cfg::kPlayers}, Cfg::kPlayers));
    // hand_exposed[p] = the canonical public flag "seat p's hand is
    // now public knowledge" (Baron-loss, showdown, Princess-played).
    viz::declare_field(schema, "hand_exposed",
                       viz::all_public({Cfg::kPlayers}, Cfg::kPlayers));

    // ---- private (owner-only) ----
    // hand[p]: each seat sees only their own hand card. When
    // hand_exposed[p] flips, consumers gate on that public flag
    // rather than mutating hand's viz — same convention as Coup's
    // revealed[] / influence[]. Baron-compare temporarily reveals
    // both compared hands to BOTH involved players; rules will use
    // viz::reveal_slot_to(hand, {p}, viewer) for that targeted peek
    // (follow-on PR; this commit only locks the base).
    viz::declare_field(
        schema, "hand",
        viz::owner_only_first_axis({Cfg::kPlayers}, Cfg::kPlayers));

    // ---- hidden ----
    // drawn_card: scalar holding the card just drawn at top of turn.
    // Held only by current_player. Base hidden; rules will
    // reveal_slot_to(current_player) on draw and reset_to_base on
    // play. Note: drawn_card is also implicitly cleared on Prince
    // discard (drawn-card-as-target), so reset_to_base must be paired
    // with the hand-write that consumes it.
    viz::declare_field(schema, "drawn_card",
                       viz::all_hidden({}, Cfg::kPlayers));
    // set_aside_card: removed from the bottom of the deck at game
    // start, NEVER revealed to anyone. Permanently hidden.
    viz::declare_field(schema, "set_aside_card",
                       viz::all_hidden({}, Cfg::kPlayers));

    // Variable-length vectors NOT in schema:
    //   - deck: hidden contents, public size, randomize_unseen handles.
    //   - discard_piles[N]: all-public stacks, hashed slot-by-slot.
    //   - face_up_removed: 2p-only, all-public, hashed slot-by-slot.

    return schema;
  }();
  return s;
}

namespace {

std::int8_t pop_top(std::vector<std::int8_t>& deck) {
  if (deck.empty()) return 0;
  const std::int8_t card = deck.back();
  deck.pop_back();
  return card;
}

}  // namespace

template <int NPlayers>
LoveLetterState<NPlayers>::LoveLetterState() {
  reset_with_seed(0xC0FFEEu);
}

template <int NPlayers>
void LoveLetterState<NPlayers>::reset_with_seed(std::uint64_t seed) {
  IGameState::reset_step_count_base();
  auto& d = data;
  d.current_player = 0;
  d.first_player = 0;
  d.winner = -1;
  d.terminal = false;
  d.ply = 0;
  d.hand.fill(0);
  d.drawn_card = 0;
  d.alive.fill(1);
  d.protected_flags.fill(0);
  d.hand_exposed.fill(0);
  d.set_aside_card = 0;
  d.face_up_removed.clear();
  for (int p = 0; p < Cfg::kPlayers; ++p) {
    d.discard_piles[static_cast<size_t>(p)].clear();
  }
  undo_stack.clear();

  d.deck.clear();
  d.deck.reserve(kTotalCards);
  for (int card = 1; card <= kCardTypes; ++card) {
    for (int c = 0; c < kCardCounts[static_cast<size_t>(card)]; ++c) {
      d.deck.push_back(static_cast<std::int8_t>(card));
    }
  }

  // One-shot rng for the initial deck shuffle. RNG is not stored on
  // state — caller of do_action_fast supplies its own rng for any
  // subsequent randomness.
  {
    std::mt19937_64 rng(seed);
    for (size_t i = d.deck.size(); i > 1; --i) {
      const size_t j = static_cast<size_t>(rng() % i);
      std::swap(d.deck[i - 1], d.deck[j]);
    }
  }

  d.set_aside_card = pop_top(d.deck);

  if constexpr (NPlayers == 2) {
    for (int i = 0; i < 3; ++i) {
      d.face_up_removed.push_back(pop_top(d.deck));
    }
  }

  for (int p = 0; p < Cfg::kPlayers; ++p) {
    d.hand[static_cast<size_t>(p)] = pop_top(d.deck);
  }

  d.drawn_card = pop_top(d.deck);

  viz::init_viz(*this, schema());
}

template <int NPlayers>
StateHash64 LoveLetterState<NPlayers>::state_hash(bool include_hidden_rng) const {
  const auto& d = data;
  std::size_t h = 0;
  hash_combine(h, static_cast<std::size_t>(d.current_player + 3));
  hash_combine(h, static_cast<std::size_t>(d.first_player + 5));
  hash_combine(h, static_cast<std::size_t>(d.ply + 7));
  hash_combine(h, static_cast<std::size_t>(d.winner + 11));
  hash_combine(h, static_cast<std::size_t>(d.terminal ? 1 : 0));

  for (int p = 0; p < Cfg::kPlayers; ++p) {
    hash_combine(h, static_cast<std::size_t>(d.alive[p] + 13));
    hash_combine(h, static_cast<std::size_t>(d.protected_flags[p] + 17));
    if (p == d.current_player || include_hidden_rng) {
      hash_combine(h, static_cast<std::size_t>(d.hand[p] + 19));
    } else {
      hash_combine(h, static_cast<std::size_t>(0 + 19));
    }
    for (auto c : d.discard_piles[static_cast<size_t>(p)]) {
      hash_combine(h, static_cast<std::size_t>(c + 23));
    }
    hash_combine(h, static_cast<std::size_t>(d.discard_piles[static_cast<size_t>(p)].size() + 29));
  }

  if (d.drawn_card != 0) {
    hash_combine(h, static_cast<std::size_t>(d.drawn_card + 31));
  }

  hash_combine(h, static_cast<std::size_t>(d.deck.size() + 37));
  if (include_hidden_rng) {
    hash_combine(h, static_cast<std::size_t>(d.set_aside_card + 41));
    for (auto c : d.deck) {
      hash_combine(h, static_cast<std::size_t>(c + 43));
    }
  }

  for (auto c : d.face_up_removed) {
    hash_combine(h, static_cast<std::size_t>(c + 47));
  }

  return static_cast<StateHash64>(h);
}

template <int NPlayers>
void LoveLetterState<NPlayers>::hash_public_fields(Hasher& h) const {
  // Public info in Love Letter: everything except any player's hand,
  // current_player's drawn_card, set_aside_card, and deck contents.
  const auto& d = data;
  h.add(d.current_player + 3);
  h.add(d.first_player + 5);
  h.add(d.ply + 7);
  h.add(d.winner + 11);
  h.add(d.terminal ? 1 : 0);

  for (int p = 0; p < Cfg::kPlayers; ++p) {
    h.add(d.alive[p] + 13);
    h.add(d.protected_flags[p] + 17);
    h.add(d.hand_exposed[p] + 53);
    for (auto c : d.discard_piles[static_cast<size_t>(p)]) h.add(c + 23);
    h.add(d.discard_piles[static_cast<size_t>(p)].size() + 29);
  }
  h.add(d.deck.size() + 37);
  for (auto c : d.face_up_removed) h.add(c + 47);
}

template <int NPlayers>
void LoveLetterState<NPlayers>::hash_private_fields(int player, Hasher& h) const {
  // Private info for player p: their hand card, plus their drawn_card
  // if p is the current_player (only current_player has a drawn card).
  const auto& d = data;
  if (player >= 0 && player < Cfg::kPlayers) {
    h.add(d.hand[player] + 19);
    if (d.current_player == player && d.drawn_card != 0) {
      h.add(d.drawn_card + 31);
    }
  }
}

template <int NPlayers>
int LoveLetterState<NPlayers>::current_player() const {
  return data.current_player;
}

template <int NPlayers>
int LoveLetterState<NPlayers>::first_player() const {
  return data.first_player;
}

template <int NPlayers>
bool LoveLetterState<NPlayers>::is_terminal() const {
  return data.terminal;
}

template <int NPlayers>
int LoveLetterState<NPlayers>::winner() const {
  return data.winner;
}

template struct LoveLetterData<2>;
template struct LoveLetterData<3>;
template struct LoveLetterData<4>;
template struct LoveLetterState<2>;
template struct LoveLetterState<3>;
template struct LoveLetterState<4>;

}  // namespace board_ai::loveletter
