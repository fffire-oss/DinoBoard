#include <stdexcept>
#include <string>
#include <vector>

#include "../../engine/core/game_registry.h"
#include "../../engine/core/snapshot_io.h"
#include "coup_state.h"
#include "coup_rules.h"
#include "coup_net_adapter.h"

namespace {

using board_ai::AnyMap;
using board_ai::ActionId;
using board_ai::IGameState;
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
  m["deck_size"] = std::any(static_cast<int>(d.deck_size));

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

namespace coup_events {

// Events-only extractor — diffs (before, after) and emits a typed event
// stream that CoupBeliefTracker consumes to maintain its claim/role
// signals. Public-state replication on the wire is handled by the
// framework's `viz::serialize_public_snapshot` (auto-derived inside
// `GameBundle::install_event_protocol`); we never serialize a snapshot
// here. Sim path consumes only this events vector — so it's the cheap
// path that keeps sim_tracker in sync with descent.
//
// Events emitted (all payloads are int / bool only; no truth leak):
//   - card_revealed { player, slot, role }
//       Any slot whose `revealed` flag flipped false→true. Covers
//       lose-influence (Coup / failed action / failed block / counter
//       lost) and bluff-caught reveals.
//   - claim_resolved_truthful { claimer, role }
//       In kResolveChallengeAction the claimer revealed the claimed
//       role, so they're holding it for real → card reshuffles back to
//       deck. Detected by `after.action_challenge_succeeded == false`.
//   - block_resolved_truthful { blocker, role }
//       In kResolveChallengeCounter the blocker revealed the block
//       role; mirror of the above.
//   - claim_unchallenged { claimer, role }
//       Challenge phase ended without anyone challenging. Detected by
//       transition out of kChallengeAction without entering
//       kResolveChallengeAction.
//   - block_unchallenged { blocker, role }
//       Counter-challenge phase ended without anyone challenging.
//   - exchange_complete { player }
//       Transition out of kExchangeReturn2 (Ambassador draws done,
//       hand reshuffled — all prior claim signals on this player
//       are stale).
template <int NPlayers>
std::vector<board_ai::PublicEvent> extract_events_only(
    const IGameState& before,
    ActionId /*action*/,
    const IGameState& after,
    int /*perspective*/) {
  using namespace board_ai::coup;
  const auto& sb = board_ai::checked_cast<CoupState<NPlayers>>(before);
  const auto& sa = board_ai::checked_cast<CoupState<NPlayers>>(after);
  const auto& db = sb.data;
  const auto& da = sa.data;
  std::vector<board_ai::PublicEvent> events;

  // 1. card_revealed: any (p, slot) whose `revealed` flipped false→true.
  for (int p = 0; p < NPlayers; ++p) {
    for (int sl = 0; sl < 2; ++sl) {
      if (!db.revealed[p][sl] && da.revealed[p][sl]) {
        AnyMap payload;
        payload["player"] = std::any(p);
        payload["slot"] = std::any(sl);
        payload["role"] = std::any(static_cast<int>(da.influence[p][sl]));
        events.emplace_back("card_revealed", std::move(payload));
      }
    }
  }

  // 2. claim_resolved_truthful: kResolveChallengeAction → action_challenge
  //    failed (claim was real). before.claimed_character holds the role.
  if (db.stage == CoupStage::kResolveChallengeAction &&
      da.stage != CoupStage::kResolveChallengeAction) {
    if (!da.action_challenge_succeeded && db.claimed_character >= 0) {
      AnyMap payload;
      payload["claimer"] = std::any(db.active_player);
      payload["role"] = std::any(static_cast<int>(db.claimed_character));
      events.emplace_back("claim_resolved_truthful", std::move(payload));
    }
  }

  // 3. block_resolved_truthful: kResolveChallengeCounter → counter
  //    challenge failed (block was real).
  if (db.stage == CoupStage::kResolveChallengeCounter &&
      da.stage != CoupStage::kResolveChallengeCounter) {
    if (!da.counter_challenge_succeeded && db.block_character >= 0) {
      AnyMap payload;
      payload["blocker"] = std::any(db.blocker);
      payload["role"] = std::any(static_cast<int>(db.block_character));
      events.emplace_back("block_resolved_truthful", std::move(payload));
    }
  }

  // 4. claim_unchallenged: kChallengeAction → out without entering
  //    kResolveChallengeAction (everyone Allowed).
  if (db.stage == CoupStage::kChallengeAction &&
      da.stage != CoupStage::kChallengeAction &&
      da.stage != CoupStage::kResolveChallengeAction) {
    if (db.claimed_character >= 0) {
      AnyMap payload;
      payload["claimer"] = std::any(db.active_player);
      payload["role"] = std::any(static_cast<int>(db.claimed_character));
      events.emplace_back("claim_unchallenged", std::move(payload));
    }
  }

  // 5. block_unchallenged: kChallengeCounter → out without entering
  //    kResolveChallengeCounter.
  if (db.stage == CoupStage::kChallengeCounter &&
      da.stage != CoupStage::kChallengeCounter &&
      da.stage != CoupStage::kResolveChallengeCounter) {
    if (db.block_character >= 0) {
      AnyMap payload;
      payload["blocker"] = std::any(db.blocker);
      payload["role"] = std::any(static_cast<int>(db.block_character));
      events.emplace_back("block_unchallenged", std::move(payload));
    }
  }

  // 6. exchange_complete: kExchangeReturn2 → out.
  if (db.stage == CoupStage::kExchangeReturn2 &&
      da.stage != CoupStage::kExchangeReturn2) {
    AnyMap payload;
    payload["player"] = std::any(db.active_player);
    events.emplace_back("exchange_complete", std::move(payload));
  }

  return events;
}

}  // namespace coup_events

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
  b.belief_feature_extractor =
      std::make_unique<CoupBeliefFeatureExtractor<NPlayers>>();
  b.belief_label_extractor =
      std::make_unique<CoupBeliefLabelExtractor<NPlayers>>();
  b.state_serializer = serialize_coup<NPlayers>;
  b.action_descriptor = describe_coup;
  b.heuristic_picker = heuristic_random;
  b.install_event_protocol(
      coup_events::extract_events_only<NPlayers>,
      []() -> const board_ai::viz::VisibilitySchema& {
        return CoupState<NPlayers>::schema();
      });

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
