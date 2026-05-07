#include <stdexcept>
#include <string>
#include <vector>

#include "../../engine/core/game_registry.h"
#include "coup_state.h"
#include "coup_rules.h"
#include "coup_net_adapter.h"

namespace {

using board_ai::AnyMap;
using board_ai::ActionId;
using board_ai::IGameState;
using board_ai::EventPhase;
using board_ai::PublicEvent;
using board_ai::PublicEventTrace;

static const char* const kCharNames[5] = {
    "Duke", "Assassin", "Captain", "Ambassador", "Contessa"};

template <int NPlayers>
AnyMap serialize_coup(const IGameState& state) {
  using namespace board_ai::coup;
  const auto& s = board_ai::checked_cast<CoupState<NPlayers>>(state);
  const auto& d = s.data;

  AnyMap m;
  m["current_player"] = std::any(state.current_player());
  m["is_terminal"] = std::any(state.is_terminal());
  m["winner"] = std::any(state.winner());
  m["num_players"] = std::any(NPlayers);
  m["ply"] = std::any(d.ply);
  m["stage"] = std::any(static_cast<int>(d.stage));
  m["active_player"] = std::any(d.active_player);
  m["declared_action"] = std::any(static_cast<int>(d.declared_action));
  m["action_target"] = std::any(d.action_target);
  m["blocker"] = std::any(d.blocker);
  m["challenger"] = std::any(d.challenger);
  m["deck_size"] = std::any(static_cast<int>(d.court_deck.size()));

  // Exchange-drawn cards (Ambassador). Only exposed during the two
  // exchange-return stages; outside those stages the field is
  // default-zero and should not be confused with an actual drawn Duke.
  // -1 means "slot already emptied during Return1". Frontend renders
  // these as extra cards in the active player's hand during the return
  // stages so the player sees 2 hand + 2 drawn = 4 cards and can click
  // one to return.
  std::vector<int> exchange_drawn;
  if (d.stage == CoupStage::kExchangeReturn1 ||
      d.stage == CoupStage::kExchangeReturn2) {
    exchange_drawn.reserve(2);
    for (int i = 0; i < 2; ++i) {
      exchange_drawn.push_back(static_cast<int>(d.exchange_drawn[i]));
    }
  }
  m["exchange_drawn"] = std::any(exchange_drawn);

  std::vector<AnyMap> players;
  for (int p = 0; p < NPlayers; ++p) {
    AnyMap pm;
    pm["alive"] = std::any(static_cast<bool>(d.alive[p]));
    pm["coins"] = std::any(d.coins[p]);

    std::vector<AnyMap> influences;
    for (int sl = 0; sl < 2; ++sl) {
      AnyMap inf;
      inf["character"] = std::any(static_cast<int>(d.influence[p][sl]));
      inf["character_name"] = std::any(std::string(
          d.influence[p][sl] >= 0 && d.influence[p][sl] < 5
              ? kCharNames[d.influence[p][sl]] : ""));
      inf["revealed"] = std::any(static_cast<bool>(d.revealed[p][sl]));
      influences.push_back(std::move(inf));
    }
    pm["influences"] = std::any(influences);

    players.push_back(std::move(pm));
  }
  m["players"] = std::any(players);

  return m;
}

AnyMap describe_coup(ActionId action) {
  using namespace board_ai::coup;
  AnyMap m;
  m["action_id"] = std::any(static_cast<int>(action));

  if (action == kIncomeAction) {
    m["type"] = std::any(std::string("income"));
  } else if (action == kForeignAidAction) {
    m["type"] = std::any(std::string("foreign_aid"));
  } else if (action >= kCoupOffset && action < kCoupOffset + kCoupCount) {
    m["type"] = std::any(std::string("coup"));
    m["target"] = std::any(action - kCoupOffset);
  } else if (action == kTaxAction) {
    m["type"] = std::any(std::string("tax"));
    m["claimed"] = std::any(std::string("Duke"));
  } else if (action >= kAssassinateOffset && action < kAssassinateOffset + kAssassinateCount) {
    m["type"] = std::any(std::string("assassinate"));
    m["target"] = std::any(action - kAssassinateOffset);
    m["claimed"] = std::any(std::string("Assassin"));
  } else if (action >= kStealOffset && action < kStealOffset + kStealCount) {
    m["type"] = std::any(std::string("steal"));
    m["target"] = std::any(action - kStealOffset);
    m["claimed"] = std::any(std::string("Captain"));
  } else if (action == kExchangeAction) {
    m["type"] = std::any(std::string("exchange"));
    m["claimed"] = std::any(std::string("Ambassador"));
  } else if (action == kChallengeAction) {
    m["type"] = std::any(std::string("challenge"));
  } else if (action == kAllowAction) {
    m["type"] = std::any(std::string("allow"));
  } else if (action == kBlockDukeAction) {
    m["type"] = std::any(std::string("block"));
    m["claimed"] = std::any(std::string("Duke"));
  } else if (action == kBlockContessaAction) {
    m["type"] = std::any(std::string("block"));
    m["claimed"] = std::any(std::string("Contessa"));
  } else if (action == kBlockAmbassadorAction) {
    m["type"] = std::any(std::string("block"));
    m["claimed"] = std::any(std::string("Ambassador"));
  } else if (action == kBlockCaptainAction) {
    m["type"] = std::any(std::string("block"));
    m["claimed"] = std::any(std::string("Captain"));
  } else if (action == kAllowNoBlockAction) {
    m["type"] = std::any(std::string("allow_no_block"));
  } else if (action == kRevealSlot0 || action == kRevealSlot1) {
    m["type"] = std::any(std::string("reveal"));
    m["slot"] = std::any(action == kRevealSlot0 ? 0 : 1);
  } else if (action == kLoseSlot0 || action == kLoseSlot1) {
    m["type"] = std::any(std::string("lose_influence"));
    m["slot"] = std::any(action == kLoseSlot0 ? 0 : 1);
  } else if (action >= kReturnDuke && action <= kReturnContessa) {
    m["type"] = std::any(std::string("return_card"));
    int char_id = action - kReturnDuke;
    m["character"] = std::any(char_id);
    m["character_name"] = std::any(std::string(
        char_id >= 0 && char_id < 5 ? kCharNames[char_id] : ""));
  }

  return m;
}

board_ai::HeuristicResult heuristic_random(
    board_ai::IGameState& state, const board_ai::IGameRules& rules, std::uint64_t /*rng_seed*/) {
  auto legal = rules.legal_actions(state);
  board_ai::HeuristicResult result;
  result.actions = legal;
  result.scores.assign(legal.size(), 1.0);
  return result;
}

// Public-event extractor. The observer sees revealed cards in two cases:
//   1. A player loses influence (revealed flips false -> true).
//   2. A challenged claim is verified: claimer briefly shows the card, the
//      card is shuffled back to deck, and a new one is drawn. `revealed`
//      doesn't flip, but the role is publicly known.
//
// Both cases need TWO event kinds to drive the AI session correctly:
//
//   - pre `truth_reveal` (player, slot, role): emitted BEFORE the action.
//     Applied via apply_coup_event to overwrite `influence[player][slot]` in
//     the AI session's sampled world to the truth role. Without this, the
//     AI's internal `do_action_fast` branches differently across MCTS-sampled
//     worlds — one world thinks the challenge succeeds (matching claim →
//     reshuffle) while another thinks it fails (mismatch → lose influence).
//     The downstream public `revealed[]` flag thus differs across worlds and
//     splits the information set's hash. This is BUG-028's family of issue
//     (silent state-hash drift; no crash, just weaker MCTS). Override-in-
//     place pins the action's branch to truth, so all sampled worlds collapse
//     to the same successor public state.
//   - post `card_revealed` (player, role): advisory signal for the belief
//     tracker (signals_[][] update). No state mutation.
//
// We also emit `exchange_complete` when an Ambassador exchange cycle
// finishes, so the tracker can reset all of that player's signals.
template <int NPlayers>
PublicEventTrace extract_coup_events(
    const IGameState& before,
    ActionId action,
    const IGameState& after,
    int perspective) {
  using namespace board_ai::coup;
  const auto& sb = board_ai::checked_cast<CoupState<NPlayers>>(before);
  const auto& sa = board_ai::checked_cast<CoupState<NPlayers>>(after);
  const auto& db = sb.data;
  const auto& da = sa.data;
  PublicEventTrace out;

  auto emit_truth_reveal_pre = [&](int player, int slot, int role) {
    if (player < 0 || player >= NPlayers) return;
    if (slot < 0 || slot >= 2) return;
    if (role < 0 || role >= kCharacterCount) return;
    AnyMap payload;
    payload["player"] = std::any(player);
    payload["slot"] = std::any(slot);
    payload["role"] = std::any(role);
    out.pre_events.push_back({"truth_reveal", std::move(payload)});
  };

  auto emit_reveal_post = [&](int player, int role) {
    if (player < 0 || player >= NPlayers) return;
    if (role < 0 || role >= kCharacterCount) return;
    AnyMap payload;
    payload["player"] = std::any(player);
    payload["role"] = std::any(role);
    out.post_events.push_back({"card_revealed", std::move(payload)});
  };

  // Case 1: lose-influence — kLoseSlot0/1 in any lose stage flips
  // revealed=true and exposes the role. Pre-emit truth_reveal so the AI
  // session's influence[][] matches truth before do_action_fast.
  if (action == kLoseSlot0 || action == kLoseSlot1) {
    int slot = (action == kLoseSlot0) ? 0 : 1;
    int loser = db.current_player;
    if (loser >= 0 && loser < NPlayers && !db.revealed[loser][slot]) {
      emit_truth_reveal_pre(loser, slot, static_cast<int>(db.influence[loser][slot]));
    }
  }

  // Case 2: challenge reveal — kRevealSlot0/1 in challenge resolution
  // stages exposes the role regardless of whether the challenge succeeds
  // (role==claim, reshuffled) or fails (role!=claim, marked revealed).
  // Either branch requires the AI to see the truth role to take the same
  // branch as ground truth.
  if (action == kRevealSlot0 || action == kRevealSlot1) {
    int slot = (action == kRevealSlot0) ? 0 : 1;
    int revealer = -1;
    if (db.stage == CoupStage::kResolveChallengeAction) {
      revealer = db.active_player;
    } else if (db.stage == CoupStage::kResolveChallengeCounter) {
      revealer = db.blocker;
    }
    if (revealer >= 0 && revealer < NPlayers && !db.revealed[revealer][slot]) {
      emit_truth_reveal_pre(revealer, slot, static_cast<int>(db.influence[revealer][slot]));
    }
  }

  // Tracker-facing post events (Case 1: revealed flag flipped).
  for (int p = 0; p < NPlayers; ++p) {
    for (int sl = 0; sl < 2; ++sl) {
      if (!db.revealed[p][sl] && da.revealed[p][sl]) {
        emit_reveal_post(p, static_cast<int>(db.influence[p][sl]));
      }
    }
  }
  // Tracker-facing post event (Case 2: successful challenge reveal,
  // observable role survives only in `before` — `revealed` doesn't flip).
  if (action == kRevealSlot0 || action == kRevealSlot1) {
    int slot = (action == kRevealSlot0) ? 0 : 1;
    int revealer = -1;
    if (db.stage == CoupStage::kResolveChallengeAction) {
      revealer = db.active_player;
    } else if (db.stage == CoupStage::kResolveChallengeCounter) {
      revealer = db.blocker;
    }
    if (revealer >= 0 && revealer < NPlayers &&
        !db.revealed[revealer][slot] && !da.revealed[revealer][slot]) {
      emit_reveal_post(revealer, static_cast<int>(db.influence[revealer][slot]));
    }
  }

  // Case 3: exchange cycle complete.
  if (db.stage == CoupStage::kExchangeReturn2 &&
      da.stage != CoupStage::kExchangeReturn1 &&
      da.stage != CoupStage::kExchangeReturn2) {
    AnyMap payload;
    payload["player"] = std::any(db.active_player);
    out.post_events.push_back({"exchange_complete", std::move(payload)});
  }

  // Case 4: self_influence_redraw — challenge-success branch where the
  // revealer is the perspective player. When `card == claimed_character` in
  // kResolveChallengeAction / kResolveChallengeCounter, do_action_fast pushes
  // the revealed card back to court_deck and draws a fresh one into
  // influence[revealer][slot] (coup_rules.cpp lines 430-431, 508-509). If
  // revealer == perspective, that new card is in hash_private_fields(
  // perspective) but the AI session sampled a different card from its own
  // randomized court_deck — so the perspective hash drifts across worlds even
  // though perspective sees their own new card. Pin to truth here.
  //
  // Detection: kRevealSlot0/1, before stage is one of the two challenge-
  // resolve stages, revealed[][] did NOT flip (success branch — failure
  // branch goes through lose_influence_at_slot which sets revealed=true),
  // and the revealer matches perspective.
  if ((action == kRevealSlot0 || action == kRevealSlot1) &&
      perspective >= 0 && perspective < NPlayers) {
    int slot = (action == kRevealSlot0) ? 0 : 1;
    int revealer = -1;
    if (db.stage == CoupStage::kResolveChallengeAction) {
      revealer = db.active_player;
    } else if (db.stage == CoupStage::kResolveChallengeCounter) {
      revealer = db.blocker;
    }
    if (revealer == perspective &&
        !db.revealed[revealer][slot] && !da.revealed[revealer][slot]) {
      AnyMap payload;
      payload["player"] = std::any(revealer);
      payload["slot"] = std::any(slot);
      payload["role"] = std::any(static_cast<int>(da.influence[revealer][slot]));
      out.post_events.push_back({"self_influence_redraw", std::move(payload)});
    }
  }

  // Case 5: public_return_card — every return_card action publicly adds 1 card
  // back to court_deck (truth invariant: legal_actions guarantees the returned
  // char is in exchange_drawn or active player's hand → removed=true → push).
  // In an AI-sampled opponent-exchange world, the multiset may be inconsistent
  // with the future action sequence (randomize_unseen sampled before knowing
  // the public return), so do_action_fast can take the removed=false branch
  // and skip the push — drifting court_deck.size(), which IS in
  // hash_public_fields, splitting the perspective hash across worlds.
  //
  // Also carries truth's exchange_drawn AFTER the return. For opp-actor
  // case, session's exchange_drawn may hold sampled cards (from prior
  // randomize_unseen) that don't match truth's removal; without the
  // override, session's exchange_drawn count of non-(-1) slots drifts
  // from truth's, which then poisons randomize_unseen's slot-count
  // invariant at the next ply (fall-back uniform path, non-deterministic
  // public outputs). Applied post-action overrides both do_action_fast's
  // session-sampled exchange_drawn and the stage transition.
  if (action >= kReturnDuke && action <= kReturnContessa) {
    AnyMap payload;
    payload["card"] = std::any(static_cast<int>(action - kReturnDuke));
    payload["expected_deck_size"] = std::any(static_cast<int>(da.court_deck.size()));
    std::vector<int> truth_xd;
    truth_xd.reserve(2);
    for (int i = 0; i < 2; ++i) {
      truth_xd.push_back(static_cast<int>(da.exchange_drawn[i]));
    }
    payload["exchange_drawn"] = std::any(truth_xd);
    out.post_events.push_back({"public_return_card", std::move(payload)});
  }

  // Case 6: self_exchange_draw — perspective player just drew 2 cards from
  // court_deck into exchange_drawn[]. This happens when do_action_fast
  // transitions FROM a non-exchange-return stage INTO kExchangeReturn1 AND
  // the active player is the perspective. In ground truth those 2 cards
  // are specific characters; the AI session sampled different unseen
  // cards via randomize_unseen. exchange_drawn is hashed in
  // hash_private_fields(active_player), so without this override the
  // perspective hash drifts across sampled worlds even though perspective
  // sees their own draws (BUG-028 family).
  //
  // Mirrors Splendor's self_reserve_deck event.
  if (db.stage != CoupStage::kExchangeReturn1 &&
      db.stage != CoupStage::kExchangeReturn2 &&
      da.stage == CoupStage::kExchangeReturn1 &&
      da.active_player == perspective &&
      perspective >= 0 && perspective < NPlayers) {
    AnyMap payload;
    payload["player"] = std::any(da.active_player);
    std::vector<int> drawn;
    drawn.reserve(2);
    for (int i = 0; i < 2; ++i) {
      drawn.push_back(static_cast<int>(da.exchange_drawn[i]));
    }
    payload["drawn"] = std::any(drawn);
    out.post_events.push_back({"self_exchange_draw", std::move(payload)});
  }

  return out;
}

// initial_observation for AI API: perspective sees their own starting hand.
// Deck / opp hands are hidden and will be filled by randomize_unseen.
template <int NPlayers>
AnyMap extract_coup_initial_observation(const IGameState& state, int perspective) {
  using namespace board_ai::coup;
  const auto& s = board_ai::checked_cast<CoupState<NPlayers>>(state);
  const auto& d = s.data;
  AnyMap out;
  if (perspective >= 0 && perspective < NPlayers) {
    std::vector<int> my_hand;
    for (int sl = 0; sl < 2; ++sl) {
      my_hand.push_back(static_cast<int>(d.influence[perspective][sl]));
    }
    out["my_hand"] = std::any(my_hand);
  }
  out["num_players"] = std::any(NPlayers);
  return out;
}

template <int NPlayers>
void apply_coup_initial_observation(IGameState& state, int perspective, const AnyMap& obs) {
  using namespace board_ai::coup;
  auto& s = board_ai::checked_cast<CoupState<NPlayers>>(state);
  auto& d = s.data;
  if (perspective < 0 || perspective >= NPlayers) return;
  auto it = obs.find("my_hand");
  if (it == obs.end()) return;
  auto my_hand = std::any_cast<std::vector<int>>(it->second);
  for (int sl = 0; sl < 2 && sl < static_cast<int>(my_hand.size()); ++sl) {
    d.influence[perspective][sl] = static_cast<CharId>(my_hand[sl]);
  }
}

// Apply event handler. truth_reveal (pre) overrides influence[][] in the
// AI session's sampled world to the truth role just before do_action_fast,
// so all randomize_unseen worlds branch identically through the reveal/
// lose-influence resolution. card_revealed and exchange_complete (post)
// are advisory signals for the belief tracker, not state mutations.
template <int NPlayers>
void apply_coup_event(
    IGameState& state,
    EventPhase phase,
    const std::string& kind,
    const AnyMap& payload) {
  using namespace board_ai::coup;
  if (phase == EventPhase::kPreAction && kind == "truth_reveal") {
    auto& s = board_ai::checked_cast<CoupState<NPlayers>>(state);
    auto& d = s.data;
    auto it_p = payload.find("player");
    auto it_s = payload.find("slot");
    auto it_r = payload.find("role");
    if (it_p == payload.end() || it_s == payload.end() || it_r == payload.end()) return;
    int p = std::any_cast<int>(it_p->second);
    int slot = std::any_cast<int>(it_s->second);
    int role = std::any_cast<int>(it_r->second);
    if (p < 0 || p >= NPlayers) return;
    if (slot < 0 || slot >= 2) return;
    if (role < 0 || role >= kCharacterCount) return;
    if (d.revealed[p][slot]) return;
    d.influence[p][slot] = static_cast<CharId>(role);
    return;
  }
  if (phase == EventPhase::kPostAction && kind == "self_influence_redraw") {
    // Challenge-success branch: AI session's do_action_fast just drew a fresh
    // card into influence[revealer][slot] from its own randomized court_deck.
    // Override to truth, with the same court-deck balance as self_exchange_draw:
    // push AI's new card back, then remove a truth-matching card (or pop_back)
    // to keep court_deck.size() invariant.
    auto& s = board_ai::checked_cast<CoupState<NPlayers>>(state);
    auto& d = s.data;
    auto it_p = payload.find("player");
    auto it_s = payload.find("slot");
    auto it_r = payload.find("role");
    if (it_p == payload.end() || it_s == payload.end() || it_r == payload.end()) return;
    int p = std::any_cast<int>(it_p->second);
    int slot = std::any_cast<int>(it_s->second);
    int role = std::any_cast<int>(it_r->second);
    if (p < 0 || p >= NPlayers) return;
    if (slot < 0 || slot >= 2) return;
    if (role < 0 || role >= kCharacterCount) return;
    if (d.revealed[p][slot]) return;
    if (d.influence[p][slot] >= 0) {
      d.court_deck.push_back(d.influence[p][slot]);
    }
    auto it = std::find(d.court_deck.begin(), d.court_deck.end(),
                        static_cast<CharId>(role));
    if (it != d.court_deck.end()) {
      d.court_deck.erase(it);
    } else if (!d.court_deck.empty()) {
      d.court_deck.pop_back();
    }
    d.influence[p][slot] = static_cast<CharId>(role);
    return;
  }
  if (phase == EventPhase::kPostAction && kind == "public_return_card") {
    // In truth, every return_card action removes the returned char from the
    // actor's exchange_drawn or hand and pushes it to court_deck (+1 size).
    // In an AI-sampled opp-exchange world, the sampled exchange_drawn/hand
    // multiset may not contain the returned char, so do_action_fast skips
    // the push and court_deck.size() drifts. Sync to truth.
    auto& s = board_ai::checked_cast<CoupState<NPlayers>>(state);
    auto& d = s.data;
    auto it_size = payload.find("expected_deck_size");
    auto it_card = payload.find("card");
    if (it_size == payload.end() || it_card == payload.end()) return;
    int expected = std::any_cast<int>(it_size->second);
    int card = std::any_cast<int>(it_card->second);
    int cur = static_cast<int>(d.court_deck.size());
    if (cur < expected) {
      // AI session missed the push. Add the returned card to balance.
      d.court_deck.push_back(static_cast<CharId>(card));
    } else if (cur > expected) {
      // Shouldn't happen, but be defensive.
      while (static_cast<int>(d.court_deck.size()) > expected &&
             !d.court_deck.empty()) {
        d.court_deck.pop_back();
      }
    }
    // Override exchange_drawn to truth's post-action values. Session's
    // do_action_fast may have left exchange_drawn in a different shape
    // (e.g. session's sampled exchange_drawn didn't contain the returned
    // role, so Return1 fell back to influence search). The next ply's
    // randomize_unseen slot enumeration uses exchange_drawn's non-(-1)
    // count, and truth/session mismatch there poisons the
    // total_remaining == slots.size() invariant → uniform fall-back →
    // non-deterministic public output.
    auto it_xd = payload.find("exchange_drawn");
    if (it_xd != payload.end()) {
      const auto& xd = std::any_cast<const std::vector<int>&>(it_xd->second);
      if (xd.size() == 2) {
        d.exchange_drawn[0] = static_cast<CharId>(xd[0]);
        d.exchange_drawn[1] = static_cast<CharId>(xd[1]);
      }
    }
    return;
  }
  if (phase == EventPhase::kPostAction && kind == "self_exchange_draw") {
    // Perspective player just drew 2 cards from court_deck into
    // exchange_drawn[]. The AI session's randomize_unseen sampled different
    // cards. Override exchange_drawn[] to truth. court_deck contents are
    // NOT in any public hash (only court_deck.size() is), and exchange_drawn
    // IS in hash_private_fields(active_player). So we need to:
    //   (a) Set exchange_drawn := truth (necessary for hash equivalence).
    //   (b) Preserve court_deck.size() (necessary because size IS public).
    //
    // We don't need to make court_deck content-equal across worlds — that
    // multiset is intentionally sampled per world. Just keep size right by
    // replacing the AI's previously-drawn cards back into court_deck (truth
    // is now the source of exchange_drawn, AI's sampled draws aren't real).
    auto& s = board_ai::checked_cast<CoupState<NPlayers>>(state);
    auto& d = s.data;
    auto it_d = payload.find("drawn");
    if (it_d == payload.end()) return;
    auto drawn = std::any_cast<std::vector<int>>(it_d->second);
    if (drawn.size() != 2) return;
    // Push AI's previously-drawn cards back, then set exchange_drawn to
    // truth. Net deck size unchanged; exchange_drawn pinned to truth.
    for (int i = 0; i < 2; ++i) {
      if (d.exchange_drawn[i] >= 0) {
        d.court_deck.push_back(d.exchange_drawn[i]);
      }
    }
    // Remove 2 cards from court_deck to balance the push above. Prefer
    // removing truth-matching cards so court_deck composition stays
    // belief-consistent for this world (cards that "really are" in
    // exchange_drawn now shouldn't double-count in the deck).
    for (int t : drawn) {
      auto it = std::find(d.court_deck.begin(), d.court_deck.end(),
                          static_cast<CharId>(t));
      if (it != d.court_deck.end()) {
        d.court_deck.erase(it);
      } else if (!d.court_deck.empty()) {
        d.court_deck.pop_back();  // best-effort: keep size invariant.
      }
    }
    d.exchange_drawn[0] = static_cast<CharId>(drawn[0]);
    d.exchange_drawn[1] = static_cast<CharId>(drawn[1]);
    return;
  }
  // card_revealed / exchange_complete: tracker-only, no state mutation.
}

template <int NPlayers>
board_ai::GameBundle make_coup(const std::string& game_id, std::uint64_t seed) {
  using namespace board_ai::coup;
  board_ai::GameBundle b;
  b.game_id = game_id;
  auto s = std::make_unique<CoupState<NPlayers>>();
  s->reset_with_seed(seed);
  b.state = std::move(s);
  b.rules = std::make_unique<CoupRules<NPlayers>>();
  b.value_model = std::make_unique<board_ai::DefaultStateValueModel>();
  b.encoder = std::make_unique<CoupFeatureEncoder<NPlayers>>();
  b.belief_tracker = std::make_unique<CoupBeliefTracker<NPlayers>>();
  b.state_serializer = serialize_coup<NPlayers>;
  b.action_descriptor = describe_coup;
  b.heuristic_picker = heuristic_random;
  b.public_event_extractor = extract_coup_events<NPlayers>;
  b.public_event_applier = apply_coup_event<NPlayers>;
  b.initial_observation_extractor = extract_coup_initial_observation<NPlayers>;
  b.initial_observation_applier = apply_coup_initial_observation<NPlayers>;
  return b;
}

board_ai::GameRegistrar reg_coup("coup", [](std::uint64_t seed) {
  return make_coup<2>("coup", seed);
});
board_ai::GameRegistrar reg_coup_2p("coup_2p", [](std::uint64_t seed) {
  return make_coup<2>("coup_2p", seed);
});
board_ai::GameRegistrar reg_coup_3p("coup_3p", [](std::uint64_t seed) {
  return make_coup<3>("coup_3p", seed);
});
board_ai::GameRegistrar reg_coup_4p("coup_4p", [](std::uint64_t seed) {
  return make_coup<4>("coup_4p", seed);
});

}  // namespace
