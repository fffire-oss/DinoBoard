#include "loveletter_state.h"

#include <algorithm>
#include <cstdint>
#include <stdexcept>

#include "../../engine/core/masked_state.h"
#include "../../engine/core/schema_hash.h"
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
  // Start-of-game viz reveals (drawn_card → current_player) live in
  // rules.cpp per I1 (rules are sole viz writer); they're invoked
  // by the registrar's make_loveletter wrapper after this returns.
}

template <int NPlayers>
void LoveLetterState<NPlayers>::reseed_viz() {
  viz::init_viz(*this, schema());
}

template <int NPlayers>
StateHash64 LoveLetterState<NPlayers>::state_hash() const {
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
    if (p == d.current_player) {
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

  for (auto c : d.face_up_removed) {
    hash_combine(h, static_cast<std::size_t>(c + 47));
  }

  return static_cast<StateHash64>(h);
}

template <int NPlayers>
void LoveLetterState<NPlayers>::hash_extra_state_fields(int perspective,
                                                         Hasher& h) const {
  // Off-schema state: variable-length public lists (discard_piles, deck
  // size, face_up_removed) plus the actor-private drawn_card during
  // their own turn. §G migrates these to schema variable_length / viz
  // reveal; until then this hook preserves hash semantics.
  if (perspective < 0 || perspective >= Cfg::kPlayers) return;
  const auto& d = data;
  for (int p = 0; p < Cfg::kPlayers; ++p) {
    for (auto c : d.discard_piles[static_cast<size_t>(p)]) h.add(c + 23);
    h.add(d.discard_piles[static_cast<size_t>(p)].size() + 29);
  }
  h.add(d.deck.size() + 37);
  for (auto c : d.face_up_removed) h.add(c + 47);
  if (d.current_player == perspective && d.drawn_card != 0) {
    h.add(d.drawn_card + 31);
  }
}

template <int NPlayers>
void LoveLetterState<NPlayers>::hash_field_slot(
    Hasher& h, const std::string& name,
    const std::vector<int>& idx) const {
  const auto& d = data;
  // 0-D scalar fields.
  if (name == "current_player") { h.add(d.current_player + 3); return; }
  if (name == "first_player") { h.add(d.first_player + 5); return; }
  if (name == "winner") { h.add(d.winner + 11); return; }
  if (name == "terminal") { h.add(d.terminal ? 1 : 0); return; }
  if (name == "ply") { h.add(d.ply + 7); return; }
  // 1-D per-player fields.
  if (name == "alive") {
    h.add(d.alive[static_cast<size_t>(idx[0])] + 13); return;
  }
  if (name == "protected_flags") {
    h.add(d.protected_flags[static_cast<size_t>(idx[0])] + 17); return;
  }
  if (name == "hand_exposed") {
    h.add(d.hand_exposed[static_cast<size_t>(idx[0])] + 53); return;
  }
  // Owner-only.
  if (name == "hand") {
    h.add(d.hand[static_cast<size_t>(idx[0])] + 19); return;
  }
  // all_hidden in schema → walker never visits these. drawn_card is
  // appended manually in hash_private_fields; set_aside_card is
  // permanently hidden and excluded from the hash entirely.
  if (name == "drawn_card") { return; }
  if (name == "set_aside_card") { return; }
}

// Walker-driven snapshot I/O. read_field_slot returns std::any of int /
// bool for every all_public schema slot (Step 4 §G.1: replaces the
// hand-rolled loveletter_snapshot_io). Hidden slots (hand / drawn_card
// / set_aside_card) are never visited by the all_public walker.
template <int NPlayers>
std::any LoveLetterState<NPlayers>::read_field_slot(
    const std::string& name, const std::vector<int>& idx) const {
  const auto& d = data;
  if (name == "current_player") return std::any(static_cast<int>(d.current_player));
  if (name == "first_player") return std::any(static_cast<int>(d.first_player));
  if (name == "winner") return std::any(static_cast<int>(d.winner));
  if (name == "terminal") return std::any(static_cast<bool>(d.terminal));
  if (name == "ply") return std::any(static_cast<int>(d.ply));
  if (name == "alive") {
    return std::any(static_cast<int>(d.alive[static_cast<size_t>(idx[0])]));
  }
  if (name == "protected_flags") {
    return std::any(static_cast<int>(d.protected_flags[static_cast<size_t>(idx[0])]));
  }
  if (name == "hand_exposed") {
    return std::any(static_cast<int>(d.hand_exposed[static_cast<size_t>(idx[0])]));
  }
  // hand[p] is owner_only_first_axis: when walker visits with viz=1 (the
  // owner's perspective, or after rules' reveal_slot / reveal_slot_to),
  // the underlying truth value is shipped.
  if (name == "hand") {
    return std::any(static_cast<int>(d.hand[static_cast<size_t>(idx[0])]));
  }
  if (name == "drawn_card") {
    return std::any(static_cast<int>(d.drawn_card));
  }
  if (name == "set_aside_card") {
    return std::any(static_cast<int>(d.set_aside_card));
  }
  return {};
}

template <int NPlayers>
void LoveLetterState<NPlayers>::write_field_slot(
    const std::string& name, const std::vector<int>& idx,
    const std::any& value) {
  auto& d = data;
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

  if (name == "current_player") d.current_player = as_int();
  else if (name == "first_player") d.first_player = static_cast<std::int8_t>(as_int());
  else if (name == "winner") d.winner = as_int();
  else if (name == "terminal") d.terminal = as_bool();
  else if (name == "ply") d.ply = as_int();
  else if (name == "alive") {
    d.alive[static_cast<size_t>(idx[0])] = static_cast<std::int8_t>(as_int());
  }
  else if (name == "protected_flags") {
    d.protected_flags[static_cast<size_t>(idx[0])] = static_cast<std::int8_t>(as_int());
  }
  else if (name == "hand_exposed") {
    d.hand_exposed[static_cast<size_t>(idx[0])] = static_cast<std::int8_t>(as_int());
  }
  else if (name == "hand") {
    d.hand[static_cast<size_t>(idx[0])] = static_cast<std::int8_t>(as_int());
  }
  else if (name == "drawn_card") {
    d.drawn_card = static_cast<std::int8_t>(as_int());
  }
  else if (name == "set_aside_card") {
    d.set_aside_card = static_cast<std::int8_t>(as_int());
  }
}

template <int NPlayers>
void LoveLetterState<NPlayers>::mask_field_slot(
    const std::string& name, const std::vector<int>& idx) {
  // Walker has already classified this slot as hidden. Write the
  // type-matching kPlaceholder sentinel into the typed payload.
  //
  // Only schema-declared fields whose base viz can be 0 for some
  // perspective ever reach here:
  //   - hand[p]                : owner_only_first_axis. Hidden from
  //                              every non-owner perspective.
  //   - drawn_card             : all_hidden in schema (no dynamic
  //                              reveal_slot wiring this PR), so the
  //                              walker emits it for every viewer.
  //   - set_aside_card         : permanently all_hidden.
  //
  // Public scalars / per-player public arrays never call here.
  if (name == "hand") {
    data.hand[static_cast<size_t>(idx[0])] = kPlaceholderInt8;
    return;
  }
  if (name == "drawn_card") {
    data.drawn_card = kPlaceholderInt8;
    return;
  }
  if (name == "set_aside_card") {
    data.set_aside_card = kPlaceholderInt8;
    return;
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
