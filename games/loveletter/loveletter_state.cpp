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
    // both compared hands to BOTH involved players via
    // viz::reveal_slot_to(hand, {p}, viewer).
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

    // ---- multiset counts (all_public, fixed shape) ----
    // discard_count[N, kCardTypes+1]: per-player public discard
    // multiset. Walker visits every slot; observer reconstructs
    // visual order from the action stream client-side.
    viz::declare_field(
        schema, "discard_count",
        viz::all_public({Cfg::kPlayers, kCardTypes + 1}, Cfg::kPlayers));
    // face_up_count[kCardTypes+1]: 2p-only public count of cards
    // burned face-up at game start. Other variants leave it all-zero.
    viz::declare_field(
        schema, "face_up_count",
        viz::all_public({kCardTypes + 1}, Cfg::kPlayers));
    // deck_count[kCardTypes+1]: per-type counts in the hidden deck.
    // Total (sum) is public — but per-type counts are NOT something
    // every observer can see in the actual game (an observer only
    // knows total minus what they've witnessed), so we declare the
    // total via a separate `deck_size` scalar and treat
    // `deck_count` itself as all_hidden — randomize_unseen fills
    // it from the tracker's information set on each sim.
    viz::declare_field(
        schema, "deck_count",
        viz::all_hidden({kCardTypes + 1}, Cfg::kPlayers));
    viz::declare_field(schema, "deck_size",
                       viz::all_public({}, Cfg::kPlayers));

    return schema;
  }();
  return s;
}

namespace {

// Draw one card from a count-array deck using the supplied rng,
// weighted by remaining count per type. Returns 0 if the deck is
// empty. Decrements the chosen entry. Functionally equivalent to
// "shuffle once + pop_back" — both produce a uniformly distributed
// permutation, the count-array form just integrates the lazy rng.
template <std::size_t Size>
std::int8_t draw_from_count(std::array<std::int8_t, Size>& count,
                            std::mt19937_64& rng) {
  int total = 0;
  for (std::size_t i = 1; i < Size; ++i) total += count[i];
  if (total <= 0) return 0;
  std::uint64_t r = rng();
  int pick = static_cast<int>(r % static_cast<std::uint64_t>(total));
  for (std::size_t i = 1; i < Size; ++i) {
    int n = count[i];
    if (pick < n) {
      count[i] = static_cast<std::int8_t>(n - 1);
      return static_cast<std::int8_t>(i);
    }
    pick -= n;
  }
  return 0;
}

template <int NPlayers>
int deck_total(const LoveLetterData<NPlayers>& d) {
  int n = 0;
  for (int i = 1; i <= kCardTypes; ++i) n += d.deck_count[static_cast<size_t>(i)];
  return n;
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
  d.face_up_count.fill(0);
  for (int p = 0; p < Cfg::kPlayers; ++p) {
    d.discard_count[static_cast<size_t>(p)].fill(0);
  }
  undo_stack.clear();

  d.deck_count.fill(0);
  for (int card = 1; card <= kCardTypes; ++card) {
    d.deck_count[static_cast<size_t>(card)] =
        static_cast<std::int8_t>(kCardCounts[static_cast<size_t>(card)]);
  }

  // One-shot rng for the initial draws. RNG is not stored on
  // state — caller of do_action_fast supplies its own rng for any
  // subsequent randomness.
  std::mt19937_64 rng(seed);

  d.set_aside_card = draw_from_count(d.deck_count, rng);

  if constexpr (NPlayers == 2) {
    for (int i = 0; i < 3; ++i) {
      std::int8_t c = draw_from_count(d.deck_count, rng);
      d.face_up_count[static_cast<size_t>(c)]++;
    }
  }

  for (int p = 0; p < Cfg::kPlayers; ++p) {
    d.hand[static_cast<size_t>(p)] = draw_from_count(d.deck_count, rng);
  }

  d.drawn_card = draw_from_count(d.deck_count, rng);

  // Initial deck_size: sum after all start-of-game draws (set_aside,
  // 2p face-up trio, opening hands, current_player's drawn_card).
  // Rules update d.deck_size on subsequent draws.
  int total = 0;
  for (int c = 1; c <= kCardTypes; ++c) {
    total += d.deck_count[static_cast<size_t>(c)];
  }
  d.deck_size = static_cast<std::int8_t>(total);

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
  // Hand-rolled fallback for callers that don't go through the
  // perspective-aware framework path. Mirrors what the schema walker
  // produces for current_player's view, minus the framework's
  // structural (field_pos, idx) salt — close enough for parity tests
  // that don't compare bit-for-bit with the framework hash.
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
    hash_combine(h, static_cast<std::size_t>(d.hand_exposed[p] + 53));
    if (p == d.current_player) {
      hash_combine(h, static_cast<std::size_t>(d.hand[p] + 19));
    } else {
      hash_combine(h, static_cast<std::size_t>(0 + 19));
    }
    for (int c = 1; c <= kCardTypes; ++c) {
      hash_combine(h, static_cast<std::size_t>(
          d.discard_count[static_cast<size_t>(p)][static_cast<size_t>(c)] + 23 + c));
    }
  }

  if (d.drawn_card != 0) {
    hash_combine(h, static_cast<std::size_t>(d.drawn_card + 31));
  }

  for (int c = 1; c <= kCardTypes; ++c) {
    hash_combine(h, static_cast<std::size_t>(
        d.face_up_count[static_cast<size_t>(c)] + 47 + c));
  }

  int deck_size = 0;
  for (int c = 1; c <= kCardTypes; ++c) {
    deck_size += d.deck_count[static_cast<size_t>(c)];
  }
  hash_combine(h, static_cast<std::size_t>(deck_size + 37));

  return static_cast<StateHash64>(h);
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
  // Owner-only with dynamic reveals (Priest peek / Baron showdown / King
  // swap / hand_exposed). The framework hash mixes (field_pos, idx[])
  // structurally before dispatching here, so this branch only mixes
  // the value — see schema_hash.h. BUG-037 (the {hand[1]=5,hand[3]=7}
  // vs {hand[0]=5,hand[1]=7} collision) is now defended at the
  // framework level: the structural mix differs between the two
  // worlds, so the digests no longer collide even though the value
  // multisets are identical.
  if (name == "hand") {
    h.add(d.hand[static_cast<size_t>(idx[0])] + 19);
    return;
  }
  // drawn_card: schema-base all_hidden, but rules `reveal_slot_to` it to
  // the current player on draw — visible to actor, walker visits it.
  // Mix the value so two worlds with the same hand but different draws
  // hash distinctly (governs legal actions via Countess rule + plays).
  if (name == "drawn_card") { h.add(d.drawn_card + 31); return; }
  // set_aside_card: never revealed; walker always emits sentinel.
  if (name == "set_aside_card") { return; }
  // Multiset count slots. Framework already mixed (field_pos, idx[]),
  // so we just mix the value.
  if (name == "discard_count") {
    h.add(d.discard_count[static_cast<size_t>(idx[0])]
                          [static_cast<size_t>(idx[1])] + 23);
    return;
  }
  if (name == "face_up_count") {
    h.add(d.face_up_count[static_cast<size_t>(idx[0])] + 47);
    return;
  }
  // deck_count is all_hidden — walker emits sentinel for every viewer,
  // never reaches here.
  if (name == "deck_count") { return; }
  if (name == "deck_size") {
    // First-class field — read d.deck_size directly. Rules maintain
    // it at every draw/refill site; observers receive it via
    // apply_public_snapshot from the wire. Do NOT sum d.deck_count here:
    // deck_count is hidden, summing it forces observers to
    // re-randomize the multiset every ply just to recover this hash
    // value (the bug Plan A surfaced).
    h.add(d.deck_size + 37);
    return;
  }
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
  if (name == "discard_count") {
    return std::any(static_cast<int>(
        d.discard_count[static_cast<size_t>(idx[0])][static_cast<size_t>(idx[1])]));
  }
  if (name == "face_up_count") {
    return std::any(static_cast<int>(
        d.face_up_count[static_cast<size_t>(idx[0])]));
  }
  if (name == "deck_count") {
    return std::any(static_cast<int>(
        d.deck_count[static_cast<size_t>(idx[0])]));
  }
  if (name == "deck_size") {
    return std::any(static_cast<int>(d.deck_size));
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
  else if (name == "discard_count") {
    d.discard_count[static_cast<size_t>(idx[0])][static_cast<size_t>(idx[1])] =
        static_cast<std::int8_t>(as_int());
  }
  else if (name == "face_up_count") {
    d.face_up_count[static_cast<size_t>(idx[0])] =
        static_cast<std::int8_t>(as_int());
  }
  else if (name == "deck_count") {
    d.deck_count[static_cast<size_t>(idx[0])] =
        static_cast<std::int8_t>(as_int());
  }
  else if (name == "deck_size") {
    d.deck_size = static_cast<std::int8_t>(as_int());
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
  if (name == "deck_count") {
    data.deck_count[static_cast<size_t>(idx[0])] = kPlaceholderInt8;
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
