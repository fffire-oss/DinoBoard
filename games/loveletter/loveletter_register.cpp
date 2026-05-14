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

// Public-event extractor / applier — single unified call into the
// schema-walker driven snapshot primitive. The wire carries (a) every
// (idx, value) pair where viz[idx, perspective]=1 and (b) the
// perspective's full viz slice; the receiver wholesale-replaces both.
// LL emits no public events; `out.events` stays empty.
template <int NPlayers>
PublicEventTrace extract_events(
    const IGameState& /*before*/,
    ActionId /*action*/,
    const IGameState& after,
    int perspective) {
  PublicEventTrace out;
  board_ai::viz::serialize_public_snapshot(
      after, LoveLetterState<NPlayers>::schema(), perspective,
      out.public_snapshot);
  return out;
}

template <int NPlayers>
void apply_public_state(IGameState& state, const AnyMap& snap,
                        int receiver_seat) {
  board_ai::viz::apply_public_snapshot(
      state, LoveLetterState<NPlayers>::schema(), receiver_seat, snap);
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
