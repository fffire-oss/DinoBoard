#include "coup_state.h"

#include <cstdint>
#include <random>
#include <stdexcept>

#include "../../engine/core/masked_state.h"
#include "../../engine/core/schema_hash.h"
#include "../../engine/core/viz_runtime.h"

namespace board_ai::coup {

namespace {

// Draw one card from a count-array deck, weighted by remaining count
// per character. Returns -1 if the deck is empty. Decrements both the
// chosen entry and `deck_size`. Mirrors LL's `draw_from_count`.
template <std::size_t Size>
CharId draw_from_count(std::array<std::int8_t, Size>& count,
                       std::int8_t& deck_size,
                       std::mt19937_64& rng) {
  int total = 0;
  for (std::size_t i = 0; i < Size; ++i) total += count[i];
  if (total <= 0) return -1;
  std::uint64_t r = rng();
  int pick = static_cast<int>(r % static_cast<std::uint64_t>(total));
  for (std::size_t i = 0; i < Size; ++i) {
    int n = count[i];
    if (pick < n) {
      count[i] = static_cast<std::int8_t>(n - 1);
      deck_size = static_cast<std::int8_t>(deck_size - 1);
      return static_cast<CharId>(i);
    }
    pick -= n;
  }
  return -1;
}

}  // namespace

template <int NPlayers>
const viz::VisibilitySchema& CoupState<NPlayers>::schema() {
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
    // Public scalar count of court deck size — analogous to LL's
    // deck_size. Per-character contents (deck_count) are all_hidden;
    // the size is what every observer can see in the physical game.
    viz::declare_field(schema, "deck_size",
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
    // influence[N][2]: face-down cards. Owner-only base; rules call
    // viz::reveal_slot(state, "influence", {p, s}) when a card flips
    // public (challenge / coup / assassinate / lose-influence). Walker
    // dispatches by viz directly — downstream consumers don't gate on
    // revealed[].
    viz::declare_field(
        schema, "influence",
        viz::owner_only_first_axis({Cfg::kPlayers, kInfluencePerPlayer},
                                   Cfg::kPlayers));

    // ---- hidden ----
    // exchange_drawn[2]: two cards drawn from court_deck during the
    // Exchange action, visible only to active_player. Base is hidden;
    // rules reveal_slot_to(active_player) on draw and reset_to_base on
    // return-to-deck.
    viz::declare_field(schema, "exchange_drawn",
                       viz::all_hidden({kExchangeDrawSlots}, Cfg::kPlayers));
    // deck_count[kCharacterCount]: per-character counts in the hidden
    // deck. The size is public (deck_size), per-character contents are
    // not — randomize_unseen fills from the tracker's information set
    // on each sim entry.
    viz::declare_field(
        schema, "deck_count",
        viz::all_hidden({kCharacterCount}, Cfg::kPlayers));

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

  for (CharId c = 0; c < kCharacterCount; ++c) {
    data.deck_count[static_cast<size_t>(c)] =
        static_cast<std::int8_t>(kCardsPerCharacter);
  }
  data.deck_size = static_cast<std::int8_t>(kTotalCards);

  // One-shot rng for opening 2-cards-per-player deal. RNG is not stored
  // on state — caller of do_action_fast supplies its own rng for any
  // subsequent randomness (challenge redraws / exchange).
  {
    std::mt19937_64 rng(seed);
    for (int p = 0; p < NPlayers; ++p) {
      data.influence[p][0] = draw_from_count(data.deck_count, data.deck_size, rng);
      data.influence[p][1] = draw_from_count(data.deck_count, data.deck_size, rng);
    }
  }
  for (int p = 0; p < NPlayers; ++p) {
    data.revealed[p] = {false, false};
    data.coins[p] = kStartingCoins;
    data.alive[p] = true;
  }

  // Sentinel -1 = "no card here" for exchange_drawn.
  data.exchange_drawn = {-1, -1};
  data.exchange_held_count = 0;

  data.current_player = 0;
  data.first_player = 0;
  data.active_player = 0;
  data.stage = CoupStage::kDeclareAction;

  viz::init_viz(*this, schema());
}

template <int NPlayers>
StateHash64 CoupState<NPlayers>::state_hash() const {
  // Single canonical path: framework's perspective-aware hash for the
  // current player's information set. Mirrors LL after BUG-037 cleanup.
  return this->state_hash_for_perspective(data.current_player);
}

template <int NPlayers>
void CoupState<NPlayers>::hash_field_slot(
    Hasher& h, const std::string& name,
    const std::vector<int>& idx) const {
  const auto& d = data;
  // 0-D scalar fields.
  if (name == "current_player") { h.add(d.current_player); return; }
  if (name == "first_player") { h.add(static_cast<int>(d.first_player)); return; }
  if (name == "winner") { h.add(d.winner + 1); return; }
  if (name == "terminal") { h.add(d.terminal ? 1 : 0); return; }
  if (name == "ply") { h.add(d.ply); return; }
  if (name == "stage") { h.add(static_cast<int>(d.stage)); return; }
  if (name == "active_player") { h.add(d.active_player); return; }
  if (name == "declared_action") { h.add(d.declared_action + 1); return; }
  if (name == "action_target") { h.add(d.action_target + 1); return; }
  if (name == "claimed_character") { h.add(d.claimed_character + 1); return; }
  if (name == "challenger") { h.add(d.challenger + 1); return; }
  if (name == "challenge_loser") { h.add(d.challenge_loser + 1); return; }
  if (name == "action_challenged") { h.add(d.action_challenged ? 1 : 0); return; }
  if (name == "action_challenge_succeeded") {
    h.add(d.action_challenge_succeeded ? 1 : 0); return;
  }
  if (name == "blocker") { h.add(d.blocker + 1); return; }
  if (name == "block_character") { h.add(d.block_character + 1); return; }
  if (name == "counter_challenged") { h.add(d.counter_challenged ? 1 : 0); return; }
  if (name == "counter_challenge_succeeded") {
    h.add(d.counter_challenge_succeeded ? 1 : 0); return;
  }
  if (name == "challenge_check_index") { h.add(d.challenge_check_index); return; }
  if (name == "exchange_held_count") { h.add(d.exchange_held_count); return; }
  if (name == "deck_size") {
    h.add(d.deck_size + 37);
    return;
  }
  // 1-D per-player fields.
  if (name == "coins") {
    h.add(d.coins[static_cast<size_t>(idx[0])]); return;
  }
  if (name == "alive") {
    h.add(d.alive[static_cast<size_t>(idx[0])] ? 1 : 0); return;
  }
  // 2-D fields.
  if (name == "revealed") {
    h.add(d.revealed[static_cast<size_t>(idx[0])]
                    [static_cast<size_t>(idx[1])] ? 1 : 0);
    return;
  }
  // Owner-only with dynamic reveals (rules call viz::reveal_slot on
  // lose-influence / challenge-loss). Walker decides visibility; this
  // branch only mixes the value when invoked.
  if (name == "influence") {
    h.add(d.influence[static_cast<size_t>(idx[0])]
                     [static_cast<size_t>(idx[1])] + 100);
    return;
  }
  // exchange_drawn: schema-base all_hidden, but rules
  // reveal_slot_to(active_player) on draw. Walker visits for the
  // active player; other viewers see kHiddenHashSentinel (framework
  // never enters this branch for them).
  if (name == "exchange_drawn") {
    h.add(d.exchange_drawn[static_cast<size_t>(idx[0])] + 200);
    return;
  }
  // deck_count is all_hidden — randomize_unseen fills before sim
  // hashing. Walker visits for any viewer once the slot has been
  // filled (sim-side); on the session side viz=0 means walker emits
  // sentinel and never enters here.
  if (name == "deck_count") {
    h.add(d.deck_count[static_cast<size_t>(idx[0])] + 50);
    return;
  }
}

template <int NPlayers>
std::any CoupState<NPlayers>::read_field_slot(
    const std::string& name, const std::vector<int>& idx) const {
  const auto& d = data;
  if (name == "current_player") return std::any(static_cast<int>(d.current_player));
  if (name == "first_player") return std::any(static_cast<int>(d.first_player));
  if (name == "winner") return std::any(static_cast<int>(d.winner));
  if (name == "terminal") return std::any(static_cast<bool>(d.terminal));
  if (name == "ply") return std::any(static_cast<int>(d.ply));
  if (name == "stage") return std::any(static_cast<int>(d.stage));
  if (name == "active_player") return std::any(static_cast<int>(d.active_player));
  if (name == "declared_action") return std::any(static_cast<int>(d.declared_action));
  if (name == "action_target") return std::any(static_cast<int>(d.action_target));
  if (name == "claimed_character") return std::any(static_cast<int>(d.claimed_character));
  if (name == "challenger") return std::any(static_cast<int>(d.challenger));
  if (name == "challenge_loser") return std::any(static_cast<int>(d.challenge_loser));
  if (name == "action_challenged") return std::any(static_cast<bool>(d.action_challenged));
  if (name == "action_challenge_succeeded")
    return std::any(static_cast<bool>(d.action_challenge_succeeded));
  if (name == "blocker") return std::any(static_cast<int>(d.blocker));
  if (name == "block_character") return std::any(static_cast<int>(d.block_character));
  if (name == "counter_challenged") return std::any(static_cast<bool>(d.counter_challenged));
  if (name == "counter_challenge_succeeded")
    return std::any(static_cast<bool>(d.counter_challenge_succeeded));
  if (name == "challenge_check_index") return std::any(static_cast<int>(d.challenge_check_index));
  if (name == "exchange_held_count") return std::any(static_cast<int>(d.exchange_held_count));
  if (name == "deck_size") return std::any(static_cast<int>(d.deck_size));
  if (name == "coins") {
    return std::any(static_cast<int>(d.coins[static_cast<size_t>(idx[0])]));
  }
  if (name == "alive") {
    return std::any(static_cast<bool>(d.alive[static_cast<size_t>(idx[0])]));
  }
  if (name == "revealed") {
    return std::any(static_cast<bool>(
        d.revealed[static_cast<size_t>(idx[0])][static_cast<size_t>(idx[1])]));
  }
  // owner_only_first_axis — walker reaches here only when viz=1 for
  // the perspective. Truth is shipped as a (idx, value) pair under
  // `snap[name]` by `viz::serialize_public_snapshot`.
  if (name == "influence") {
    return std::any(static_cast<int>(
        d.influence[static_cast<size_t>(idx[0])][static_cast<size_t>(idx[1])]));
  }
  if (name == "exchange_drawn") {
    return std::any(static_cast<int>(
        d.exchange_drawn[static_cast<size_t>(idx[0])]));
  }
  if (name == "deck_count") {
    return std::any(static_cast<int>(
        d.deck_count[static_cast<size_t>(idx[0])]));
  }
  return {};
}

template <int NPlayers>
void CoupState<NPlayers>::write_field_slot(
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
  else if (name == "stage") d.stage = static_cast<CoupStage>(as_int());
  else if (name == "active_player") d.active_player = as_int();
  else if (name == "declared_action") d.declared_action = as_int();
  else if (name == "action_target") d.action_target = as_int();
  else if (name == "claimed_character")
    d.claimed_character = static_cast<CharId>(as_int());
  else if (name == "challenger") d.challenger = as_int();
  else if (name == "challenge_loser") d.challenge_loser = as_int();
  else if (name == "action_challenged") d.action_challenged = as_bool();
  else if (name == "action_challenge_succeeded")
    d.action_challenge_succeeded = as_bool();
  else if (name == "blocker") d.blocker = as_int();
  else if (name == "block_character")
    d.block_character = static_cast<CharId>(as_int());
  else if (name == "counter_challenged") d.counter_challenged = as_bool();
  else if (name == "counter_challenge_succeeded")
    d.counter_challenge_succeeded = as_bool();
  else if (name == "challenge_check_index")
    d.challenge_check_index = as_int();
  else if (name == "exchange_held_count")
    d.exchange_held_count = as_int();
  else if (name == "deck_size")
    d.deck_size = static_cast<std::int8_t>(as_int());
  else if (name == "coins") {
    d.coins[static_cast<size_t>(idx[0])] = as_int();
  }
  else if (name == "alive") {
    d.alive[static_cast<size_t>(idx[0])] = as_bool();
  }
  else if (name == "revealed") {
    d.revealed[static_cast<size_t>(idx[0])][static_cast<size_t>(idx[1])] =
        as_bool();
  }
  else if (name == "influence") {
    d.influence[static_cast<size_t>(idx[0])][static_cast<size_t>(idx[1])] =
        static_cast<CharId>(as_int());
  }
  else if (name == "exchange_drawn") {
    d.exchange_drawn[static_cast<size_t>(idx[0])] =
        static_cast<CharId>(as_int());
  }
  else if (name == "deck_count") {
    d.deck_count[static_cast<size_t>(idx[0])] =
        static_cast<std::int8_t>(as_int());
  }
}

template <int NPlayers>
void CoupState<NPlayers>::mask_field_slot(
    const std::string& name, const std::vector<int>& idx) {
  // Walker only reaches here for slots whose viz is 0 for the
  // perspective. Coup's hidden slots:
  //   - influence[p][s]   : owner_only_first_axis, hidden from any
  //                         non-owner viewer until rules reveal_slot.
  //   - exchange_drawn[i] : all_hidden; rules reveal_slot_to(active)
  //                         while exchange is in progress.
  //   - deck_count[c]     : all_hidden; randomize_unseen fills on sim.
  if (name == "influence") {
    data.influence[static_cast<size_t>(idx[0])]
                  [static_cast<size_t>(idx[1])] = kPlaceholderInt8;
    return;
  }
  if (name == "exchange_drawn") {
    data.exchange_drawn[static_cast<size_t>(idx[0])] = kPlaceholderInt8;
    return;
  }
  if (name == "deck_count") {
    data.deck_count[static_cast<size_t>(idx[0])] = kPlaceholderInt8;
    return;
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
