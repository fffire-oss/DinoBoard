#include <stdexcept>
#include <string>
#include <vector>

#include "../../engine/core/game_registry.h"
#include "../../engine/core/snapshot_io.h"
#include "../../engine/core/viz_runtime.h"
#include "loveletter_state.h"
#include "loveletter_rules.h"
#include "loveletter_net_adapter.h"

namespace {

using board_ai::AnyMap;
using board_ai::ActionId;
using board_ai::IGameState;
using board_ai::PublicEvent;
using board_ai::PublicEventTrace;

static const char* const kCardNames[9] = {
    "", "Guard", "Priest", "Baron", "Handmaid", "Prince", "King", "Countess", "Princess"};

template <int NPlayers>
AnyMap serialize_loveletter(const IGameState& state) {
  using namespace board_ai::loveletter;
  const auto& s = board_ai::checked_cast<LoveLetterState<NPlayers>>(state);
  const auto& d = s.data;

  AnyMap m;
  m["current_player"] = std::any(state.current_player());
  m["is_terminal"] = std::any(state.is_terminal());
  m["winner"] = std::any(state.winner());
  m["num_players"] = std::any(NPlayers);
  m["ply"] = std::any(d.ply);
  int deck_size = 0;
  for (int c = 1; c <= board_ai::loveletter::kCardTypes; ++c) {
    deck_size += d.deck_count[static_cast<size_t>(c)];
  }
  m["deck_size"] = std::any(deck_size);

  std::vector<AnyMap> players;
  for (int p = 0; p < NPlayers; ++p) {
    AnyMap pm;
    pm["alive"] = std::any(static_cast<bool>(d.alive[p]));
    pm["protected"] = std::any(static_cast<bool>(d.protected_flags[p]));
    pm["hand"] = std::any(static_cast<int>(d.hand[p]));
    pm["hand_name"] = std::any(std::string(
        d.hand[p] >= 1 && d.hand[p] <= 8 ? kCardNames[d.hand[p]] : ""));

    // Discard counts (per card type 1..kCardTypes); UI reconstructs
    // visual order from the action stream. Ship the multiset count
    // vector indexed 0..kCardTypes (slot 0 is unused but kept for
    // alignment with card values).
    std::vector<int> discard_counts;
    discard_counts.reserve(board_ai::loveletter::kCardTypes + 1);
    for (int c = 0; c <= board_ai::loveletter::kCardTypes; ++c) {
      discard_counts.push_back(static_cast<int>(
          d.discard_count[static_cast<size_t>(p)][static_cast<size_t>(c)]));
    }
    pm["discard_count"] = std::any(discard_counts);

    // Display-side derived list: enumerate every count copy as a flat
    // sequence in card-type order. Tests and any "show me a pile"
    // consumers iterate this; engine-side hash / encoder do not.
    std::vector<int> discards_flat;
    for (int c = 1; c <= board_ai::loveletter::kCardTypes; ++c) {
      int n = d.discard_count[static_cast<size_t>(p)][static_cast<size_t>(c)];
      for (int i = 0; i < n; ++i) discards_flat.push_back(c);
    }
    pm["discards"] = std::any(discards_flat);

    players.push_back(std::move(pm));
  }
  m["players"] = std::any(players);

  m["drawn_card"] = std::any(static_cast<int>(d.drawn_card));
  m["drawn_card_name"] = std::any(std::string(
      d.drawn_card >= 1 && d.drawn_card <= 8 ? kCardNames[d.drawn_card] : ""));

  std::vector<int> face_up_count_v;
  face_up_count_v.reserve(board_ai::loveletter::kCardTypes + 1);
  for (int c = 0; c <= board_ai::loveletter::kCardTypes; ++c) {
    face_up_count_v.push_back(static_cast<int>(
        d.face_up_count[static_cast<size_t>(c)]));
  }
  m["face_up_count"] = std::any(face_up_count_v);

  // Display-side derived list (face-up burn pile, 2p only).
  std::vector<int> face_up_flat;
  for (int c = 1; c <= board_ai::loveletter::kCardTypes; ++c) {
    int n = d.face_up_count[static_cast<size_t>(c)];
    for (int i = 0; i < n; ++i) face_up_flat.push_back(c);
  }
  m["face_up_removed"] = std::any(face_up_flat);

  // The "set-aside" card is the topmost card removed from the deck at game
  // start (always exactly 1 — Love Letter rule). It's hidden from all
  // players in normal play but exposed here for tile-conservation
  // invariants in test suites. Belief tracker / encoder must NOT read it
  // (they don't — both go through the public API only).
  m["set_aside_card"] = std::any(static_cast<int>(d.set_aside_card));

  return m;
}

AnyMap describe_loveletter(ActionId action) {
  using namespace board_ai::loveletter;
  AnyMap m;
  m["action_id"] = std::any(static_cast<int>(action));

  if (action >= kGuardOffset && action < kGuardOffset + kGuardCount) {
    int idx = action - kGuardOffset;
    int target = idx / 7;
    int guess = idx % 7 + 2;
    m["type"] = std::any(std::string("guard"));
    m["card"] = std::any(1);
    m["card_name"] = std::any(std::string("Guard"));
    m["target"] = std::any(target);
    m["guess"] = std::any(guess);
    m["guess_name"] = std::any(std::string(
        guess >= 1 && guess <= 8 ? kCardNames[guess] : ""));
  } else if (action >= kPriestOffset && action < kPriestOffset + kPriestCount) {
    m["type"] = std::any(std::string("priest"));
    m["card"] = std::any(2);
    m["card_name"] = std::any(std::string("Priest"));
    m["target"] = std::any(action - kPriestOffset);
  } else if (action >= kBaronOffset && action < kBaronOffset + kBaronCount) {
    m["type"] = std::any(std::string("baron"));
    m["card"] = std::any(3);
    m["card_name"] = std::any(std::string("Baron"));
    m["target"] = std::any(action - kBaronOffset);
  } else if (action == kHandmaidAction) {
    m["type"] = std::any(std::string("handmaid"));
    m["card"] = std::any(4);
    m["card_name"] = std::any(std::string("Handmaid"));
  } else if (action >= kPrinceOffset && action < kPrinceOffset + kPrinceCount) {
    m["type"] = std::any(std::string("prince"));
    m["card"] = std::any(5);
    m["card_name"] = std::any(std::string("Prince"));
    m["target"] = std::any(action - kPrinceOffset);
  } else if (action >= kKingOffset && action < kKingOffset + kKingCount) {
    m["type"] = std::any(std::string("king"));
    m["card"] = std::any(6);
    m["card_name"] = std::any(std::string("King"));
    m["target"] = std::any(action - kKingOffset);
  } else if (action == kCountessAction) {
    m["type"] = std::any(std::string("countess"));
    m["card"] = std::any(7);
    m["card_name"] = std::any(std::string("Countess"));
  } else if (action == kPrincessAction) {
    m["type"] = std::any(std::string("princess"));
    m["card"] = std::any(8);
    m["card_name"] = std::any(std::string("Princess"));
  }

  return m;
}

// Love Letter heuristic (heuristic guidance + benchmark baseline).
//
// Scoring is per-action (no lookahead — Love Letter decisions are local).
// Reads only public state + perspective's OWN hand/drawn (private fields
// the actor legitimately knows — they're picking their own move).
// Never reads other players' hidden hands.
namespace loveletter_heuristic {

using board_ai::loveletter::LoveLetterState;

// Card values (from Love Letter rules).
constexpr int kGuard = 1, kPriest = 2, kBaron = 3, kHandmaid = 4;
constexpr int kPrince = 5, kKing = 6, kCountess = 7, kPrincess = 8;

template <int NPlayers>
double score_action(
    const LoveLetterState<NPlayers>& s, ActionId a, std::mt19937_64& rng) {
  const auto& d = s.data;
  const int actor = d.current_player;
  // Actor's own hand + drawn card — these are fully known to actor.
  const int my_hand = d.hand[actor];
  const int my_drawn = d.drawn_card;

  auto target_from_offset = [&](int offset_in_action) {
    return (actor + 1 + offset_in_action) % NPlayers;
  };

  // --- Guard (0..27): 0 <= id < 28, id = target_offset*7 + (guess-2) ---
  if (a >= board_ai::loveletter::kGuardOffset &&
      a < board_ai::loveletter::kGuardOffset + board_ai::loveletter::kGuardCount) {
    const int rel = a - board_ai::loveletter::kGuardOffset;
    const int target_off = rel / 7;
    const int guess = 2 + (rel % 7);
    const int target = target_from_offset(target_off);
    if (target == actor || !d.alive[target]) return -100.0;
    if (d.protected_flags[target]) return -5.0;
    // Baseline: Guard is low-value unless you have tracker knowledge.
    // Without knowing target's hand, guessing Priest (common) > Princess (rare).
    // Approximate the target card distribution from discard piles to adjust.
    int seen = 0;
    std::array<int, 9> played{};
    for (int p = 0; p < NPlayers; ++p) {
      for (int c = 1; c <= board_ai::loveletter::kCardTypes; ++c) {
        int n = d.discard_count[p][c];
        played[c] += n;
        seen += n;
      }
    }
    // Count own + drawn + face-up removed so we know those too.
    played[my_hand]++;
    if (my_drawn > 0) played[my_drawn]++;
    for (int c = 1; c <= board_ai::loveletter::kCardTypes; ++c) {
      played[c] += d.face_up_count[c];
    }
    // Card counts in Love Letter:
    const std::array<int, 9> total = {0, 5, 2, 2, 2, 2, 1, 1, 1};
    double prob_guess = std::max(0.0, static_cast<double>(total[guess] - played[guess]));
    // Prefer guess on higher-remaining cards, especially 2..5.
    return 0.6 + prob_guess * 0.25;
  }

  // --- Priest (28..31): peek at target — always good if can use next turn.
  if (a >= board_ai::loveletter::kPriestOffset &&
      a < board_ai::loveletter::kPriestOffset + board_ai::loveletter::kPriestCount) {
    const int target = target_from_offset(a - board_ai::loveletter::kPriestOffset);
    if (target == actor || !d.alive[target]) return -100.0;
    if (d.protected_flags[target]) return -5.0;
    return 2.5;  // solid info gain
  }

  // --- Baron (32..35): compare hands; win if we're higher ---
  if (a >= board_ai::loveletter::kBaronOffset &&
      a < board_ai::loveletter::kBaronOffset + board_ai::loveletter::kBaronCount) {
    const int target = target_from_offset(a - board_ai::loveletter::kBaronOffset);
    if (target == actor || !d.alive[target]) return -100.0;
    if (d.protected_flags[target]) return -5.0;
    // We play Baron alongside our OTHER card (whichever we keep). Estimate
    // we keep the higher of (hand, drawn) and play the other. Here Baron IS
    // being played, so we'd retain whichever of hand/drawn isn't Baron (=3).
    int kept = (my_hand == kBaron) ? my_drawn : my_hand;
    // Higher `kept` = more likely to win Baron compare. Simplified scale.
    return static_cast<double>(kept) * 0.5 - 1.5;
  }

  // --- Handmaid (36): self-protect until next turn. Usually safe. ---
  if (a == board_ai::loveletter::kHandmaidAction) {
    return 2.2;
  }

  // --- Prince (37..40): force target to discard; may kill if Princess ---
  if (a >= board_ai::loveletter::kPrinceOffset &&
      a < board_ai::loveletter::kPrinceOffset + board_ai::loveletter::kPrinceCount) {
    const int target = target_from_offset(a - board_ai::loveletter::kPrinceOffset);
    if (!d.alive[target]) return -100.0;
    if (target != actor && d.protected_flags[target]) return -5.0;
    // Targeting self is usually bad unless we have Princess (forced discard
    // of OTHER card); but with Prince + Princess you can't play Prince
    // on self. Heuristic: prefer prince on opponents.
    if (target == actor) {
      // Self-targeting: force discard of own drawn/hand — only good if
      // we're holding Countess or low card we want to dump.
      int kept = (my_hand == kPrince) ? my_drawn : my_hand;
      if (kept == kCountess) return 0.3;  // reshuffle for better
      return -1.0;
    }
    return 1.8;
  }

  // --- King (41..44): swap hands with target ---
  if (a >= board_ai::loveletter::kKingOffset &&
      a < board_ai::loveletter::kKingOffset + board_ai::loveletter::kKingCount) {
    const int target = target_from_offset(a - board_ai::loveletter::kKingOffset);
    if (target == actor || !d.alive[target]) return -100.0;
    if (d.protected_flags[target]) return -5.0;
    // We play King, so we keep whatever's in our other slot (not King).
    int my_other = (my_hand == kKing) ? my_drawn : my_hand;
    // King good if our other card is low: swap a low card for opp's unknown.
    // Rough: if my_other <= 3, swap is net-positive.
    if (my_other <= 3) return 2.0;
    return -0.5;
  }

  // --- Countess (45): just discard. Score ~0 (no effect) but mandatory
  // in some hand combos. Rules engine handles mandatory; here baseline low.
  if (a == board_ai::loveletter::kCountessAction) {
    // If paired with King/Prince, rules force Countess. Otherwise unnecessary
    // to play voluntarily (you save a 7-card for late-round comparisons).
    if (my_hand == kKing || my_drawn == kKing ||
        my_hand == kPrince || my_drawn == kPrince) {
      return 0.5;  // forced — take it
    }
    return -0.8;  // voluntary Countess wastes a high card
  }

  // --- Princess (46): suicide. ---
  if (a == board_ai::loveletter::kPrincessAction) {
    return -50.0;  // never voluntarily
  }

  // Shouldn't reach.
  return 0.0;
}

template <int NPlayers>
board_ai::HeuristicResult pick(
    board_ai::IGameState& state,
    const board_ai::IGameRules& rules,
    std::uint64_t rng_seed) {
  auto& s = board_ai::checked_cast<LoveLetterState<NPlayers>>(state);
  auto legal = rules.legal_actions(state);
  std::mt19937_64 rng(rng_seed);

  board_ai::HeuristicResult result;
  result.actions = legal;
  result.scores.reserve(legal.size());
  for (ActionId a : legal) {
    result.scores.push_back(score_action<NPlayers>(s, a, rng));
  }
  return result;
}

}  // namespace loveletter_heuristic

// --- Public-event protocol -------------------------------------------------
//
// GT-side rules (do_action_fast) maintain `state.viz` as they reveal hidden
// slots: Priest → reveal_slot_to(actor), Baron → cross-reveal, King →
// swap_slot_owned, Prince → reset_to_base on the discarded slot, etc. The
// observer side learns those reveals via `public_snapshot` (walker copies
// every viz=1 slot) and via the `events` list below for tracker-only signals
// (discards, eliminations, winner announcement). No private payload rides on
// `events` — every fact the observer learns about hidden cards flows through
// the schema-driven snapshot path.

namespace loveletter_events {

using board_ai::loveletter::LoveLetterConfig;
using board_ai::loveletter::LoveLetterState;
using board_ai::loveletter::LoveLetterData;
using board_ai::loveletter::kGuard;
using board_ai::loveletter::kPriest;
using board_ai::loveletter::kBaron;
using board_ai::loveletter::kHandmaid;
using board_ai::loveletter::kPrince;
using board_ai::loveletter::kKing;
using board_ai::loveletter::kCountess;
using board_ai::loveletter::kPrincess;
using board_ai::loveletter::kGuardOffset;
using board_ai::loveletter::kGuardCount;
using board_ai::loveletter::kPriestOffset;
using board_ai::loveletter::kPriestCount;
using board_ai::loveletter::kBaronOffset;
using board_ai::loveletter::kBaronCount;
using board_ai::loveletter::kHandmaidAction;
using board_ai::loveletter::kPrinceOffset;
using board_ai::loveletter::kPrinceCount;
using board_ai::loveletter::kKingOffset;
using board_ai::loveletter::kKingCount;
using board_ai::loveletter::kCountessAction;
using board_ai::loveletter::kPrincessAction;

template <int NPlayers>
AnyMap extract_initial_observation(const IGameState& state, int perspective) {
  const auto& s = board_ai::checked_cast<LoveLetterState<NPlayers>>(state);
  const auto& d = s.data;
  AnyMap out;
  // Perspective's own starting hand (visible to them).
  if (perspective >= 0 && perspective < NPlayers) {
    out["my_hand"] = std::any(static_cast<int>(d.hand[perspective]));
  }
  // If perspective is the starting current_player, they've already drawn
  // a card at game start — visible to them.
  if (d.current_player == perspective) {
    out["my_drawn_card"] = std::any(static_cast<int>(d.drawn_card));
  } else {
    out["my_drawn_card"] = std::any(0);
  }
  // In 2p, 3 cards are face-up removed; public to all. Ship as a
  // count vector indexed 0..kCardTypes.
  std::vector<int> face_up_count_v;
  face_up_count_v.reserve(board_ai::loveletter::kCardTypes + 1);
  for (int c = 0; c <= board_ai::loveletter::kCardTypes; ++c) {
    face_up_count_v.push_back(static_cast<int>(
        d.face_up_count[static_cast<size_t>(c)]));
  }
  out["face_up_count"] = std::any(face_up_count_v);
  return out;
}

// Reset AI's internal state to be consistent with the initial observation:
// perspective's hand is set from `my_hand`, other players' hands are random
// consistent with the remaining card counts. The belief tracker is init'd
// separately by the caller (once per seat at game start).
template <int NPlayers>
void apply_initial_observation(IGameState& state, int perspective, const AnyMap& obs) {
  using board_ai::loveletter::kCardCounts;
  using board_ai::loveletter::kCardTypes;
  auto& s = board_ai::checked_cast<LoveLetterState<NPlayers>>(state);
  auto& d = s.data;

  auto it_hand = obs.find("my_hand");
  if (it_hand == obs.end()) {
    throw std::runtime_error("loveletter initial_observation missing 'my_hand'");
  }
  const std::int8_t my_hand = static_cast<std::int8_t>(std::any_cast<int>(it_hand->second));

  // drawn_card (0 if perspective isn't current_player at game start).
  std::int8_t my_drawn = 0;
  auto it_draw = obs.find("my_drawn_card");
  if (it_draw != obs.end()) {
    my_drawn = static_cast<std::int8_t>(std::any_cast<int>(it_draw->second));
  }

  std::vector<int> face_up_count_v;
  auto it_fu = obs.find("face_up_count");
  if (it_fu != obs.end()) {
    face_up_count_v = std::any_cast<std::vector<int>>(it_fu->second);
  }

  // Pool of remaining cards = full deck minus face_up minus my_hand.
  std::array<int, 9> remaining{};
  for (int c = 1; c <= kCardTypes; ++c) {
    remaining[c] = kCardCounts[c];
  }
  // Apply face_up_count vector (indexed 0..kCardTypes; slot 0 unused).
  for (int c = 1; c <= kCardTypes; ++c) {
    int n = (c < static_cast<int>(face_up_count_v.size()))
                ? face_up_count_v[static_cast<size_t>(c)]
                : 0;
    remaining[c] -= n;
  }
  if (my_hand >= 1 && my_hand <= kCardTypes) remaining[my_hand]--;
  if (my_drawn >= 1 && my_drawn <= kCardTypes) remaining[my_drawn]--;

  // Rebuild face_up_count from observation.
  d.face_up_count.fill(0);
  for (int c = 1; c <= kCardTypes; ++c) {
    int n = (c < static_cast<int>(face_up_count_v.size()))
                ? face_up_count_v[static_cast<size_t>(c)]
                : 0;
    d.face_up_count[static_cast<size_t>(c)] = static_cast<std::int8_t>(n);
  }

  // Seed each other player's hand with ANY remaining card (doesn't matter
  // which — AI's belief is "unknown"; randomize_unseen will re-sample when
  // called). Just pick the first available to keep things deterministic.
  auto take_one = [&](int exclude = -1) -> std::int8_t {
    for (int c = 1; c <= kCardTypes; ++c) {
      if (c == exclude) continue;
      if (remaining[c] > 0) {
        remaining[c]--;
        return static_cast<std::int8_t>(c);
      }
    }
    return 0;
  };
  for (int p = 0; p < NPlayers; ++p) {
    if (p == perspective) {
      d.hand[p] = my_hand;
    } else {
      d.hand[p] = take_one();
    }
    d.alive[p] = 1;
    d.protected_flags[p] = 0;
    d.hand_exposed[p] = 0;
    d.discard_count[p].fill(0);
  }
  d.set_aside_card = take_one();
  // Rebuild deck_count from what's left.
  d.deck_count.fill(0);
  for (int c = 1; c <= kCardTypes; ++c) {
    d.deck_count[static_cast<size_t>(c)] =
        static_cast<std::int8_t>(remaining[c]);
  }
  // drawn_card: if perspective is the starting current_player, we know it.
  // Otherwise assign a random placeholder (consistent-by-count; the actual
  // value is hidden from AI and will be corrected by events when it matters).
  if (d.current_player == perspective) {
    d.drawn_card = my_drawn;
  } else {
    // take_one() decrements `remaining`, but deck_count was already
    // populated above. The observer's local deck thus carries one
    // extra placeholder card relative to truth — randomize_unseen
    // overwrites contents on the next ply, so the count discrepancy
    // is harmless.
    d.drawn_card = take_one();
  }

  // Re-seed viz from schema base — wipes any stale reveals carried
  // over from a prior reset. Then apply the same starting reveal truth
  // applies (`drawn_card` → starting current_player) ONLY when
  // perspective is the starting current_player; otherwise this seat
  // hasn't drawn yet, and even truth's overlay won't expose drawn_card
  // to them. hand[perspective] is owner_only_first_axis (auto-revealed
  // to its owner by the schema base).
  s.reseed_viz();
  if (d.current_player == perspective) {
    board_ai::loveletter::LoveLetterRules<NPlayers>::reveal_starting_draw_to(
        state, perspective);
  }
}

// Public-event extractor. Every per-perspective hand reveal travels via
// `state.viz_` (rules call `reveal_slot` / `reveal_slot_to`). All-public
// slots ride the schema walker (`viz::serialize_public`); per-perspective
// reveals (owner-visible `hand[p]`, current-player-visible `drawn_card`,
// Priest/Baron peeks) ride the `owner_overlay` sidecar below — same pattern
// Splendor uses for face-up reserved cards. LL emits no public events;
// `out.events` stays empty.
template <int NPlayers>
PublicEventTrace extract_events(
    const IGameState& /*before*/,
    ActionId /*action*/,
    const IGameState& after,
    int perspective) {
  const auto& sa = board_ai::checked_cast<LoveLetterState<NPlayers>>(after);
  const auto& da = sa.data;
  PublicEventTrace out;

  // Schema-driven public snapshot. Walks every all_public slot via
  // read_field_slot. hand[p] / drawn_card never enter here (they're
  // not all_public-base); they ride the owner_overlay sidecar.
  AnyMap snap;
  board_ai::viz::serialize_public(after, LoveLetterState<NPlayers>::schema(),
                                  snap);

  // Per-perspective overlay: for each schema slot whose runtime viz=1
  // to `perspective` but whose base viz wasn't all_public, ship the
  // truth value so the receiver's session can mirror it. Encoded as
  // a flat int vector in a fixed slot order:
  //   [hand[0], hand[1], ..., hand[N-1], drawn_card]
  // Each entry is the truth value when viz[..., perspective]=1, else
  // -1 (meaning "still hidden to this perspective"). The receiver
  // applies it slot-by-slot.
  // Stash receiver perspective so `apply_public_state` knows whose
  // viewer-axis bits to toggle when applying owner_overlay.
  snap["__recv_perspective"] = std::any(static_cast<int>(perspective));

  std::vector<int> owner_overlay(NPlayers + 1, -1);
  if (perspective >= 0 && perspective < NPlayers) {
    const auto& hand_v = board_ai::viz::viz_get(after, "hand");
    if (!hand_v.empty()) {
      // shape = {NPlayers, n_viewers}. viewer is last axis.
      const int n_viewers = hand_v.viewer_count();
      for (int p = 0; p < NPlayers; ++p) {
        const std::size_t off =
            static_cast<std::size_t>(p) * static_cast<std::size_t>(n_viewers) +
            static_cast<std::size_t>(perspective);
        if (off < hand_v.data.size() && hand_v.data[off] != 0) {
          owner_overlay[static_cast<size_t>(p)] = static_cast<int>(da.hand[p]);
        }
      }
    }
    const auto& drawn_v = board_ai::viz::viz_get(after, "drawn_card");
    if (!drawn_v.empty()) {
      const std::size_t off = static_cast<std::size_t>(perspective);
      if (off < drawn_v.data.size() && drawn_v.data[off] != 0) {
        owner_overlay[static_cast<size_t>(NPlayers)] =
            static_cast<int>(da.drawn_card);
      }
    }
  }
  snap["owner_overlay"] = std::any(owner_overlay);

  // No more variable-length sidecar keys: discard_count / face_up_count
  // / deck_size are all_public schema slots, walked into `snap` above
  // by `viz::serialize_public`. deck_count is all_hidden (per-type
  // contents not knowable to non-actor) and reconstructed by
  // randomize_unseen.

  out.public_snapshot = std::move(snap);
  return out;
}

// Inverse of `extract_events`'s snapshot population. Walker-driven
// `viz::apply_public` writes every all_public slot back; per-field
// dispatch lives in `LoveLetterState::write_field_slot`. The variable-
// length side-channel keys (deck_size / discard_piles / face_up_removed)
// have no schema counterpart and are applied directly here.
template <int NPlayers>
void apply_public_state(IGameState& state, const AnyMap& snap) {
  auto& s = board_ai::checked_cast<LoveLetterState<NPlayers>>(state);
  auto& d = s.data;

  board_ai::viz::apply_public(state, LoveLetterState<NPlayers>::schema(), snap);

  // Per-perspective overlay sidecar: for each entry where the producer
  // marked the slot as visible to this receiver (value != -1), write
  // the truth value into the local hand / drawn_card and toggle viz to
  // 1 so future hashes / encodes treat it as known. Entries with -1
  // mean "still hidden — leave at whatever placeholder is already
  // there"; randomize_unseen runs trailing on apply_observation and
  // refreshes those slots from the tracker's information set.
  auto it_ov = snap.find("owner_overlay");
  if (it_ov != snap.end()) {
    auto extract_iv = [&]() -> std::vector<int> {
      const std::any& a = it_ov->second;
      if (a.type() == typeid(std::vector<int>)) {
        return std::any_cast<std::vector<int>>(a);
      }
      if (a.type() == typeid(std::vector<std::any>)) {
        const auto& av = std::any_cast<const std::vector<std::any>&>(a);
        std::vector<int> out;
        out.reserve(av.size());
        for (const auto& x : av) {
          if (x.type() == typeid(int)) out.push_back(std::any_cast<int>(x));
          else out.push_back(-1);
        }
        return out;
      }
      return {};
    };
    auto overlay = extract_iv();
    if (static_cast<int>(overlay.size()) >= NPlayers + 1) {
      // Determine which perspective this session is. The overlay was
      // emitted from the producer's view of THIS receiver; viz toggles
      // must use the receiver's seat. Use the runtime viz tensor to
      // find which viewer the producer thought we are: we are
      // perspective `viewer` iff state.viz_["hand"][p, viewer]=1
      // matches the overlay's `!= -1` pattern. Cheaper: the runner
      // always calls apply_public_state on the per-seat session bundle
      // in seat order, so the seat IS the perspective. We just need
      // it. The runner's calling shape is opaque here — instead,
      // embed perspective in the overlay implicitly: the producer
      // wrote viz from `perspective` arg; we recover it by scanning
      // viz tensor for the unique viewer whose visible-slot set
      // matches the overlay's marked entries. For LL the unique
      // perspective with `hand[perspective]=1 AND
      // drawn_card[perspective]=1 (when current_player==perspective)`
      // is well-defined — but simpler to compute: the receiver's seat
      // is the seat whose `hand` slot is marked in the overlay AND
      // matches the schema's owner_only_first_axis base — i.e. the
      // overlay entry at index == seat is non-negative.
      //
      // For each non-negative slot entry, write the value AND toggle
      // viz at every viewer axis the producer thought us to be — but
      // since there's no way to know, we toggle viz at the seat whose
      // `hand[seat] != -1` (the owner) and at the receiver. Simplest:
      // mirror the producer's intent — toggle viz[..., receiver] = 1
      // for every overlay entry that's non-negative. The receiver's
      // perspective is encoded in the snap directly.
      int receiver = -1;
      auto it_recv = snap.find("__recv_perspective");
      if (it_recv != snap.end() && it_recv->second.type() == typeid(int)) {
        receiver = std::any_cast<int>(it_recv->second);
      }
      auto& hand_v = board_ai::viz::viz_get(state, "hand");
      auto& drawn_v = board_ai::viz::viz_get(state, "drawn_card");
      const int n_viewers_h = hand_v.viewer_count();
      const int n_viewers_d = drawn_v.viewer_count();
      // Wholesale-replace semantics: the producer's overlay encodes
      // exactly what the receiver should see now. >=0 → visible (set
      // value + viz=1); -1 → hidden (clear viz=0). Without the
      // viz=0 path, an earlier reveal in this session sticks
      // forever and observer's hash diverges from truth's after rules
      // run reset_to_base on the truth side.
      for (int p = 0; p < NPlayers; ++p) {
        if (receiver >= 0 && receiver < n_viewers_h) {
          const std::size_t off =
              static_cast<std::size_t>(p) *
                  static_cast<std::size_t>(n_viewers_h) +
              static_cast<std::size_t>(receiver);
          if (off < hand_v.data.size()) {
            hand_v.data[off] =
                (overlay[static_cast<size_t>(p)] >= 0) ? 1 : 0;
          }
        }
        if (overlay[static_cast<size_t>(p)] >= 0) {
          d.hand[static_cast<size_t>(p)] =
              static_cast<std::int8_t>(overlay[static_cast<size_t>(p)]);
        }
      }
      const int dval = overlay[static_cast<size_t>(NPlayers)];
      if (receiver >= 0 && receiver < n_viewers_d) {
        const std::size_t off = static_cast<std::size_t>(receiver);
        if (off < drawn_v.data.size()) {
          drawn_v.data[off] = (dval >= 0) ? 1 : 0;
        }
      }
      if (dval >= 0) {
        d.drawn_card = static_cast<std::int8_t>(dval);
      }
    }
  }

  // Variable-length sidecar keys are gone. discard_count / face_up_count
  // / deck_size flowed back via `viz::apply_public` into the
  // corresponding schema slots; deck_count is all_hidden and gets
  // refilled by the trailing randomize_unseen call (the tracker keeps
  // the observer's marginal-distribution view of the hidden deck and
  // samples per-type counts from it).
}

}  // namespace loveletter_events

template <int NPlayers>
board_ai::GameBundle make_loveletter(const std::string& game_id, std::uint64_t seed) {
  using namespace board_ai::loveletter;
  board_ai::GameBundle b;
  b.game_id = game_id;
  auto s = std::make_unique<LoveLetterState<NPlayers>>();
  s->reset_with_seed(seed);
  // Apply start-of-game viz reveals (rules are sole viz writer per I1):
  // the seat starting as current_player physically holds the top
  // card just drawn from the deck — reveal `drawn_card` to them only.
  LoveLetterRules<NPlayers>::reveal_starting_draw(*s, s->data.current_player);
  b.state = std::move(s);
  b.rules = std::make_unique<LoveLetterRules<NPlayers>>();
  b.value_model = std::make_unique<board_ai::DefaultStateValueModel>();
  b.encoder = std::make_unique<LoveLetterFeatureEncoder<NPlayers>>();
  b.belief_tracker = std::make_unique<LoveLetterBeliefTracker<NPlayers>>();
  b.state_serializer = serialize_loveletter<NPlayers>;
  b.action_descriptor = describe_loveletter;
  b.heuristic_picker = loveletter_heuristic::pick<NPlayers>;
  b.public_event_extractor = loveletter_events::extract_events<NPlayers>;
  b.public_state_applier = loveletter_events::apply_public_state<NPlayers>;
  b.initial_observation_extractor = loveletter_events::extract_initial_observation<NPlayers>;
  b.initial_observation_applier = loveletter_events::apply_initial_observation<NPlayers>;
  return b;
}

board_ai::GameRegistrar reg_loveletter("loveletter", [](std::uint64_t seed) {
  return make_loveletter<2>("loveletter", seed);
});
board_ai::GameRegistrar reg_loveletter_2p("loveletter_2p", [](std::uint64_t seed) {
  return make_loveletter<2>("loveletter_2p", seed);
});
board_ai::GameRegistrar reg_loveletter_3p("loveletter_3p", [](std::uint64_t seed) {
  return make_loveletter<3>("loveletter_3p", seed);
});
board_ai::GameRegistrar reg_loveletter_4p("loveletter_4p", [](std::uint64_t seed) {
  return make_loveletter<4>("loveletter_4p", seed);
});

}  // namespace
