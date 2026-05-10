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
// Per-field emitter/applier table for the public_snapshot. Schema's
// declaration order in coup_state.cpp drives `viz::emit_snapshot` /
// `viz::apply_snapshot`; entries here translate one schema all_public
// field to AnyMap key. Every all_public field must have an entry in
// BOTH maps unless skipped at the call site (first_player is fixed at
// game start and excluded by hash_public_fields, so it goes in skip).
// Snapshot-only keys (court_deck_size, exchange_drawn_mask,
// revealed_char_flat partial-reveal sidecar) are NOT schema fields —
// handled directly alongside this call.
template <int NPlayers>
const board_ai::viz::SnapshotIO& coup_snapshot_io() {
  using namespace board_ai::coup;
  using CState = CoupState<NPlayers>;
  static const board_ai::viz::SnapshotIO io = []() {
    using namespace board_ai;
    viz::SnapshotIO t;

    auto put_int = [](AnyMap& m, const char* key, int v) { m[key] = std::any(v); };
    auto put_bool = [](AnyMap& m, const char* key, bool v) { m[key] = std::any(v); };
    auto put_vec = [](AnyMap& m, const char* key, std::vector<int> v) {
      m[key] = std::any(std::move(v));
    };

    // ---- emitters ----
    t.emitters["current_player"] = [put_int](const IGameState& s, AnyMap& m) {
      put_int(m, "current_player",
              static_cast<int>(checked_cast<CState>(s).data.current_player));
    };
    t.emitters["winner"] = [put_int](const IGameState& s, AnyMap& m) {
      put_int(m, "winner", static_cast<int>(checked_cast<CState>(s).data.winner));
    };
    t.emitters["terminal"] = [put_bool](const IGameState& s, AnyMap& m) {
      put_bool(m, "terminal", static_cast<bool>(checked_cast<CState>(s).data.terminal));
    };
    t.emitters["ply"] = [put_int](const IGameState& s, AnyMap& m) {
      put_int(m, "ply", static_cast<int>(checked_cast<CState>(s).data.ply));
    };
    t.emitters["stage"] = [put_int](const IGameState& s, AnyMap& m) {
      put_int(m, "stage", static_cast<int>(checked_cast<CState>(s).data.stage));
    };
    t.emitters["active_player"] = [put_int](const IGameState& s, AnyMap& m) {
      put_int(m, "active_player",
              static_cast<int>(checked_cast<CState>(s).data.active_player));
    };
    t.emitters["declared_action"] = [put_int](const IGameState& s, AnyMap& m) {
      put_int(m, "declared_action",
              static_cast<int>(checked_cast<CState>(s).data.declared_action));
    };
    t.emitters["action_target"] = [put_int](const IGameState& s, AnyMap& m) {
      put_int(m, "action_target",
              static_cast<int>(checked_cast<CState>(s).data.action_target));
    };
    t.emitters["claimed_character"] = [put_int](const IGameState& s, AnyMap& m) {
      put_int(m, "claimed_character",
              static_cast<int>(checked_cast<CState>(s).data.claimed_character));
    };
    t.emitters["challenger"] = [put_int](const IGameState& s, AnyMap& m) {
      put_int(m, "challenger",
              static_cast<int>(checked_cast<CState>(s).data.challenger));
    };
    t.emitters["challenge_loser"] = [put_int](const IGameState& s, AnyMap& m) {
      put_int(m, "challenge_loser",
              static_cast<int>(checked_cast<CState>(s).data.challenge_loser));
    };
    t.emitters["action_challenged"] = [put_bool](const IGameState& s, AnyMap& m) {
      put_bool(m, "action_challenged",
               static_cast<bool>(checked_cast<CState>(s).data.action_challenged));
    };
    t.emitters["action_challenge_succeeded"] = [put_bool](const IGameState& s, AnyMap& m) {
      put_bool(m, "action_challenge_succeeded",
               static_cast<bool>(checked_cast<CState>(s).data.action_challenge_succeeded));
    };
    t.emitters["blocker"] = [put_int](const IGameState& s, AnyMap& m) {
      put_int(m, "blocker",
              static_cast<int>(checked_cast<CState>(s).data.blocker));
    };
    t.emitters["block_character"] = [put_int](const IGameState& s, AnyMap& m) {
      put_int(m, "block_character",
              static_cast<int>(checked_cast<CState>(s).data.block_character));
    };
    t.emitters["counter_challenged"] = [put_bool](const IGameState& s, AnyMap& m) {
      put_bool(m, "counter_challenged",
               static_cast<bool>(checked_cast<CState>(s).data.counter_challenged));
    };
    t.emitters["counter_challenge_succeeded"] = [put_bool](const IGameState& s, AnyMap& m) {
      put_bool(m, "counter_challenge_succeeded",
               static_cast<bool>(checked_cast<CState>(s).data.counter_challenge_succeeded));
    };
    t.emitters["challenge_check_index"] = [put_int](const IGameState& s, AnyMap& m) {
      put_int(m, "challenge_check_index",
              static_cast<int>(checked_cast<CState>(s).data.challenge_check_index));
    };
    t.emitters["exchange_held_count"] = [put_int](const IGameState& s, AnyMap& m) {
      put_int(m, "exchange_held_count",
              static_cast<int>(checked_cast<CState>(s).data.exchange_held_count));
    };
    t.emitters["coins"] = [put_vec](const IGameState& s, AnyMap& m) {
      const auto& d = checked_cast<CState>(s).data;
      std::vector<int> v(NPlayers);
      for (int p = 0; p < NPlayers; ++p) v[p] = static_cast<int>(d.coins[p]);
      put_vec(m, "coins", std::move(v));
    };
    t.emitters["alive"] = [put_vec](const IGameState& s, AnyMap& m) {
      const auto& d = checked_cast<CState>(s).data;
      std::vector<int> v(NPlayers);
      for (int p = 0; p < NPlayers; ++p) v[p] = d.alive[p] ? 1 : 0;
      put_vec(m, "alive", std::move(v));
    };
    t.emitters["revealed"] = [put_vec](const IGameState& s, AnyMap& m) {
      const auto& d = checked_cast<CState>(s).data;
      std::vector<int> flat(NPlayers * 2);
      for (int p = 0; p < NPlayers; ++p) {
        for (int sl = 0; sl < 2; ++sl) flat[p * 2 + sl] = d.revealed[p][sl] ? 1 : 0;
      }
      put_vec(m, "revealed_flat", std::move(flat));
    };

    // ---- appliers ----
    auto get_int = [](const AnyMap& m, const char* key) -> int {
      auto it = m.find(key);
      return (it != m.end()) ? std::any_cast<int>(it->second) : 0;
    };
    auto get_bool = [](const AnyMap& m, const char* key) -> bool {
      auto it = m.find(key);
      return (it != m.end()) ? std::any_cast<bool>(it->second) : false;
    };
    auto get_iv = [](const AnyMap& m, const char* key) -> std::vector<int> {
      auto it = m.find(key);
      if (it == m.end()) return {};
      if (it->second.type() == typeid(std::vector<int>)) {
        return std::any_cast<std::vector<int>>(it->second);
      }
      if (it->second.type() == typeid(std::vector<std::any>)) {
        const auto& av = std::any_cast<const std::vector<std::any>&>(it->second);
        std::vector<int> out;
        out.reserve(av.size());
        for (const auto& x : av) {
          if (x.type() == typeid(int)) out.push_back(std::any_cast<int>(x));
        }
        return out;
      }
      return {};
    };

    t.appliers["current_player"] = [get_int](IGameState& s, const AnyMap& m) {
      checked_cast<CState>(s).data.current_player =
          static_cast<std::int8_t>(get_int(m, "current_player"));
    };
    t.appliers["winner"] = [get_int](IGameState& s, const AnyMap& m) {
      checked_cast<CState>(s).data.winner =
          static_cast<std::int8_t>(get_int(m, "winner"));
    };
    t.appliers["terminal"] = [get_bool](IGameState& s, const AnyMap& m) {
      checked_cast<CState>(s).data.terminal = get_bool(m, "terminal");
    };
    t.appliers["ply"] = [get_int](IGameState& s, const AnyMap& m) {
      checked_cast<CState>(s).data.ply =
          static_cast<std::int16_t>(get_int(m, "ply"));
    };
    t.appliers["stage"] = [get_int](IGameState& s, const AnyMap& m) {
      checked_cast<CState>(s).data.stage =
          static_cast<CoupStage>(get_int(m, "stage"));
    };
    t.appliers["active_player"] = [get_int](IGameState& s, const AnyMap& m) {
      checked_cast<CState>(s).data.active_player =
          static_cast<std::int8_t>(get_int(m, "active_player"));
    };
    t.appliers["declared_action"] = [get_int](IGameState& s, const AnyMap& m) {
      checked_cast<CState>(s).data.declared_action =
          static_cast<ActionId>(get_int(m, "declared_action"));
    };
    t.appliers["action_target"] = [get_int](IGameState& s, const AnyMap& m) {
      checked_cast<CState>(s).data.action_target =
          static_cast<std::int8_t>(get_int(m, "action_target"));
    };
    t.appliers["claimed_character"] = [get_int](IGameState& s, const AnyMap& m) {
      checked_cast<CState>(s).data.claimed_character =
          static_cast<CharId>(get_int(m, "claimed_character"));
    };
    t.appliers["challenger"] = [get_int](IGameState& s, const AnyMap& m) {
      checked_cast<CState>(s).data.challenger =
          static_cast<std::int8_t>(get_int(m, "challenger"));
    };
    t.appliers["challenge_loser"] = [get_int](IGameState& s, const AnyMap& m) {
      checked_cast<CState>(s).data.challenge_loser =
          static_cast<std::int8_t>(get_int(m, "challenge_loser"));
    };
    t.appliers["action_challenged"] = [get_bool](IGameState& s, const AnyMap& m) {
      checked_cast<CState>(s).data.action_challenged = get_bool(m, "action_challenged");
    };
    t.appliers["action_challenge_succeeded"] = [get_bool](IGameState& s, const AnyMap& m) {
      checked_cast<CState>(s).data.action_challenge_succeeded =
          get_bool(m, "action_challenge_succeeded");
    };
    t.appliers["blocker"] = [get_int](IGameState& s, const AnyMap& m) {
      checked_cast<CState>(s).data.blocker =
          static_cast<std::int8_t>(get_int(m, "blocker"));
    };
    t.appliers["block_character"] = [get_int](IGameState& s, const AnyMap& m) {
      checked_cast<CState>(s).data.block_character =
          static_cast<CharId>(get_int(m, "block_character"));
    };
    t.appliers["counter_challenged"] = [get_bool](IGameState& s, const AnyMap& m) {
      checked_cast<CState>(s).data.counter_challenged = get_bool(m, "counter_challenged");
    };
    t.appliers["counter_challenge_succeeded"] = [get_bool](IGameState& s, const AnyMap& m) {
      checked_cast<CState>(s).data.counter_challenge_succeeded =
          get_bool(m, "counter_challenge_succeeded");
    };
    t.appliers["challenge_check_index"] = [get_int](IGameState& s, const AnyMap& m) {
      checked_cast<CState>(s).data.challenge_check_index =
          static_cast<std::int8_t>(get_int(m, "challenge_check_index"));
    };
    t.appliers["exchange_held_count"] = [get_int](IGameState& s, const AnyMap& m) {
      checked_cast<CState>(s).data.exchange_held_count =
          static_cast<std::int8_t>(get_int(m, "exchange_held_count"));
    };
    t.appliers["coins"] = [get_iv](IGameState& s, const AnyMap& m) {
      auto v = get_iv(m, "coins");
      auto& d = checked_cast<CState>(s).data;
      for (int p = 0; p < NPlayers && p < static_cast<int>(v.size()); ++p) {
        d.coins[p] = static_cast<std::int8_t>(v[p]);
      }
    };
    t.appliers["alive"] = [get_iv](IGameState& s, const AnyMap& m) {
      auto v = get_iv(m, "alive");
      auto& d = checked_cast<CState>(s).data;
      for (int p = 0; p < NPlayers && p < static_cast<int>(v.size()); ++p) {
        d.alive[p] = v[p] != 0;
      }
    };
    // revealed: pulls revealed_flat for the gate flags. The character ID
    // sidecar (`revealed_char_flat`) is a separate snapshot-only key
    // applied after this dispatch (since unrevealed entries should NOT
    // overwrite session-side influence).
    t.appliers["revealed"] = [get_iv](IGameState& s, const AnyMap& m) {
      auto v = get_iv(m, "revealed_flat");
      auto& d = checked_cast<CState>(s).data;
      for (int p = 0; p < NPlayers; ++p) {
        for (int sl = 0; sl < 2; ++sl) {
          const int idx = p * 2 + sl;
          if (idx < static_cast<int>(v.size())) {
            d.revealed[p][sl] = (v[idx] != 0);
          }
        }
      }
    };

    return t;
  }();
  return io;
}

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
    // If the exchange just completed for the perspective player, the AI
    // session needs the resulting unrevealed influence pinned to truth —
    // do_action_fast is no longer run on the AI side, so the in-flight
    // hand/draws have not been merged. Carry the post-exchange unrevealed
    // influence cards in the payload; the applier writes them onto state.
    if (db.active_player == perspective &&
        perspective >= 0 && perspective < NPlayers) {
      std::vector<int> influence(2, -1);
      for (int sl = 0; sl < 2; ++sl) {
        if (!da.revealed[perspective][sl]) {
          influence[sl] = static_cast<int>(da.influence[perspective][sl]);
        }
      }
      payload["influence"] = std::any(influence);
    }
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

  // full public snapshot. Mirrors
  // CoupState::hash_public_fields + terminal/winner (not hashed but
  // observable / used by is_terminal/winner accessors).
  {
    AnyMap snap;
    // Schema-driven public fields. See coup_snapshot_io above.
    // first_player is fixed at game start and excluded by
    // hash_public_fields, so it goes in the skip set.
    board_ai::viz::emit_snapshot(after, CoupState<NPlayers>::schema(),
                                 coup_snapshot_io<NPlayers>(), snap,
                                 /*skip=*/{"first_player"});

    // Snapshot-only keys (not schema fields):
    //  - court_deck_size: public count of the hidden deck
    //  - exchange_drawn_mask: public occupancy of the all_hidden 2-slot
    //    exchange_drawn area; only the slot IDs are private
    //  - revealed_char_flat: partial-reveal sidecar for the schema
    //    "revealed" gate. Character ID is public iff revealed_flat[i]==1.
    snap["court_deck_size"] = std::any(static_cast<int>(da.court_deck.size()));

    std::vector<int> xd_mask(2);
    for (int i = 0; i < 2; ++i) xd_mask[i] = (da.exchange_drawn[i] >= 0) ? 1 : 0;
    snap["exchange_drawn_mask"] = std::any(xd_mask);

    std::vector<int> revealed_char_flat(NPlayers * 2, -1);
    for (int p = 0; p < NPlayers; ++p) {
      for (int sl = 0; sl < 2; ++sl) {
        if (da.revealed[p][sl]) {
          revealed_char_flat[p * 2 + sl] = static_cast<int>(da.influence[p][sl]);
        }
      }
    }
    snap["revealed_char_flat"] = std::any(revealed_char_flat);

    out.public_snapshot = std::move(snap);
  }

  return out;
}

// applier — writes public fields from truth snapshot. Schema-driven via
// `viz::apply_snapshot`; per-field appliers live in `coup_snapshot_io`.
// Private fields (unrevealed influence, opp exchange_drawn, deck content)
// are left untouched for tracker + self_* events + randomize_unseen.
template <int NPlayers>
void apply_coup_public_state(IGameState& state, const AnyMap& snap) {
  using namespace board_ai::coup;
  auto& s = board_ai::checked_cast<CoupState<NPlayers>>(state);
  auto& d = s.data;

  // Schema-driven public fields. first_player skipped (fixed at game
  // start; excluded from hash_public_fields).
  board_ai::viz::apply_snapshot(state, CoupState<NPlayers>::schema(),
                                coup_snapshot_io<NPlayers>(), snap,
                                /*skip=*/{"first_player"});

  auto get_int = [&](const char* key) -> int {
    auto it = snap.find(key);
    return (it != snap.end()) ? std::any_cast<int>(it->second) : 0;
  };
  // Robust int-vector accessor: handles vector<int> + empty-vector<any>
  // fallback (py_to_any defaults empty lists to vector<any>).
  auto get_iv = [&](const char* key) -> std::vector<int> {
    auto it = snap.find(key);
    if (it == snap.end()) return {};
    if (it->second.type() == typeid(std::vector<int>)) {
      return std::any_cast<std::vector<int>>(it->second);
    }
    if (it->second.type() == typeid(std::vector<std::any>)) {
      const auto& av = std::any_cast<const std::vector<std::any>&>(it->second);
      std::vector<int> out;
      out.reserve(av.size());
      for (const auto& x : av) {
        if (x.type() == typeid(int)) out.push_back(std::any_cast<int>(x));
      }
      return out;
    }
    return {};
  };

  // Snapshot-only keys (handled directly):
  //
  // revealed_char_flat: partial-reveal sidecar for the schema-driven
  // `revealed` gate. For revealed slots, overwrite the character;
  // for unrevealed, leave it untouched (tracker / truth_reveal events
  // handle perspective's own, opp stays as session-sampled value).
  auto revealed_char_flat = get_iv("revealed_char_flat");
  for (int p = 0; p < NPlayers; ++p) {
    for (int sl = 0; sl < 2; ++sl) {
      const int idx = p * 2 + sl;
      if (idx < static_cast<int>(revealed_char_flat.size()) &&
          d.revealed[p][sl] && revealed_char_flat[idx] >= 0) {
        d.influence[p][sl] = static_cast<CharId>(revealed_char_flat[idx]);
      }
    }
  }

  // Court deck size public; content will be filled by randomize_unseen.
  const int target = get_int("court_deck_size");
  if (target >= 0) {
    if (static_cast<int>(d.court_deck.size()) > target) {
      d.court_deck.resize(static_cast<size_t>(target));
    } else {
      while (static_cast<int>(d.court_deck.size()) < target) {
        d.court_deck.push_back(-1);  // placeholder
      }
    }
  }

  // exchange_drawn shape: flip -1 sentinel for empty slots per the mask.
  // Preserves existing char id in non-empty slots (tracker /
  // self_exchange_draw handles perspective's own values; opp's values
  // remain session-sampled, which is correct for a private field).
  auto xd_mask = get_iv("exchange_drawn_mask");
  for (int i = 0; i < 2 && i < static_cast<int>(xd_mask.size()); ++i) {
    if (xd_mask[i] == 0) {
      d.exchange_drawn[i] = -1;
    } else if (d.exchange_drawn[i] < 0) {
      // Mask says slot is occupied but session has -1 → placeholder 0;
      // randomize_unseen will fill with a real sampled value.
      d.exchange_drawn[i] = 0;
    }
  }
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
  // card_revealed: tracker-only, no state mutation.
  // exchange_complete: when payload carries an "influence" vector (which
  // happens iff the exchange completed for the perspective player), pin
  // perspective's unrevealed influence slots to truth. The AI session no
  // longer runs do_action_fast, so without this override the perspective's
  // own hand reflects the pre-exchange characters.
  if (phase == EventPhase::kPostAction && kind == "exchange_complete") {
    auto it_p = payload.find("player");
    auto it_i = payload.find("influence");
    if (it_p == payload.end() || it_i == payload.end()) return;
    int p = std::any_cast<int>(it_p->second);
    if (p < 0 || p >= NPlayers) return;
    auto& s = board_ai::checked_cast<CoupState<NPlayers>>(state);
    auto& d = s.data;
    std::vector<int> influence;
    if (it_i->second.type() == typeid(std::vector<int>)) {
      influence = std::any_cast<std::vector<int>>(it_i->second);
    } else if (it_i->second.type() == typeid(std::vector<std::any>)) {
      const auto& av = std::any_cast<const std::vector<std::any>&>(it_i->second);
      influence.reserve(av.size());
      for (const auto& x : av) {
        influence.push_back(x.type() == typeid(int) ? std::any_cast<int>(x) : -1);
      }
    } else {
      return;
    }
    for (int sl = 0; sl < 2 && sl < static_cast<int>(influence.size()); ++sl) {
      if (d.revealed[p][sl]) continue;
      const int role = influence[sl];
      if (role < 0 || role >= kCharacterCount) continue;
      const CharId old_role = d.influence[p][sl];
      d.influence[p][sl] = static_cast<CharId>(role);
      // Keep court_deck multiset feasible: swap out an old-role copy and
      // swap in a deck card that matches the new role, mirroring
      // self_exchange_draw's invariance trick. Best-effort.
      if (old_role >= 0) {
        d.court_deck.push_back(old_role);
        auto it = std::find(d.court_deck.begin(), d.court_deck.end(),
                            static_cast<CharId>(role));
        if (it != d.court_deck.end()) {
          d.court_deck.erase(it);
        } else if (!d.court_deck.empty()) {
          d.court_deck.pop_back();
        }
      }
    }
    return;
  }
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
  b.public_state_applier = apply_coup_public_state<NPlayers>;
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
