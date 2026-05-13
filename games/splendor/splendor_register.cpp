#include <algorithm>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

#include "../../engine/core/game_registry.h"
#include "../../engine/core/snapshot_io.h"
#include "../../engine/search/tail_solver.h"
#include "splendor_state.h"
#include "splendor_rules.h"
#include "splendor_net_adapter.h"

namespace {

using board_ai::AnyMap;
using board_ai::ActionId;
using board_ai::IGameState;
using board_ai::PublicEvent;
using board_ai::PublicEventTrace;

static const char* const kColorNames[6] = {"white", "blue", "green", "red", "black", "gold"};

template <int NPlayers>
AnyMap serialize_splendor(const IGameState& state) {
  using Cfg = board_ai::splendor::SplendorConfig<NPlayers>;
  const auto& s = board_ai::checked_cast<board_ai::splendor::SplendorState<NPlayers>>(state);
  const auto& d = s.persistent.data();

  AnyMap m;
  m["current_player"] = std::any(state.current_player());
  m["is_terminal"] = std::any(state.is_terminal());
  m["winner"] = std::any(state.winner());
  m["num_players"] = std::any(NPlayers);
  m["plies"] = std::any(d.plies);
  m["stage"] = std::any(static_cast<int>(d.stage));
  m["pending_returns"] = std::any(d.pending_returns);
  m["final_round_remaining"] = std::any(d.final_round_remaining);

  std::vector<int> scores(NPlayers);
  for (int i = 0; i < NPlayers; ++i) scores[i] = d.scores[i];
  m["scores"] = std::any(scores);

  std::vector<int> bank(board_ai::splendor::kTokenTypes);
  for (int i = 0; i < board_ai::splendor::kTokenTypes; ++i) bank[i] = d.bank[i];
  m["bank"] = std::any(bank);

  std::vector<AnyMap> players;
  for (int p = 0; p < NPlayers; ++p) {
    AnyMap pm;
    std::vector<int> gems(board_ai::splendor::kTokenTypes);
    for (int j = 0; j < board_ai::splendor::kTokenTypes; ++j) gems[j] = d.player_gems[p][j];
    pm["gems"] = std::any(gems);
    std::vector<int> bonuses(board_ai::splendor::kColorCount);
    for (int j = 0; j < board_ai::splendor::kColorCount; ++j) bonuses[j] = d.player_bonuses[p][j];
    pm["bonuses"] = std::any(bonuses);
    pm["points"] = std::any(static_cast<int>(d.player_points[p]));
    pm["cards_count"] = std::any(static_cast<int>(d.player_cards_count[p]));
    pm["nobles_count"] = std::any(static_cast<int>(d.player_nobles_count[p]));

    int rs = d.reserved_size[p];
    std::vector<AnyMap> reserved;
    const auto& pool = board_ai::splendor::splendor_card_pool();
    for (int ri = 0; ri < rs; ++ri) {
      AnyMap rc;
      int card_id = d.reserved[p][ri];
      bool visible = d.reserved_visible[p][ri] != 0;
      rc["visible"] = std::any(visible);
      // Always serialize full card info whenever card_id is valid, even for
      // deck-reserved (visible=false). The owner always knows their own
      // reserved card, so the web UI must receive its face for rendering.
      // The frontend uses `(visible || isHuman)` to decide whether to show
      // the face or the card back, so opponents' deck-reserved slots still
      // render as a hidden placeholder.
      if (card_id >= 0 && card_id < static_cast<int>(pool.size())) {
        const auto& card = pool[card_id];
        rc["tier"] = std::any(static_cast<int>(card.tier));
        rc["bonus"] = std::any(static_cast<int>(card.bonus));
        rc["points"] = std::any(static_cast<int>(card.points));
        std::vector<int> cost(board_ai::splendor::kColorCount);
        for (int j = 0; j < board_ai::splendor::kColorCount; ++j) cost[j] = card.cost[j];
        rc["cost"] = std::any(cost);
      } else {
        rc["tier"] = std::any(-1);
      }
      reserved.push_back(std::move(rc));
    }
    pm["reserved"] = std::any(reserved);
    players.push_back(std::move(pm));
  }
  m["players"] = std::any(players);

  const auto& pool = board_ai::splendor::splendor_card_pool();
  std::vector<std::vector<AnyMap>> tableau(3);
  for (int tier = 0; tier < 3; ++tier) {
    int ts = d.tableau_size[tier];
    for (int slot = 0; slot < ts; ++slot) {
      int card_id = d.tableau[tier][slot];
      AnyMap cm;
      cm["card_id"] = std::any(card_id);
      if (card_id >= 0 && card_id < static_cast<int>(pool.size())) {
        const auto& card = pool[card_id];
        cm["tier"] = std::any(static_cast<int>(card.tier));
        cm["bonus"] = std::any(static_cast<int>(card.bonus));
        cm["points"] = std::any(static_cast<int>(card.points));
        std::vector<int> cost(board_ai::splendor::kColorCount);
        for (int j = 0; j < board_ai::splendor::kColorCount; ++j) cost[j] = card.cost[j];
        cm["cost"] = std::any(cost);
      }
      tableau[tier].push_back(std::move(cm));
    }
  }
  std::vector<std::any> tableau_any;
  for (auto& t : tableau) tableau_any.push_back(std::any(std::move(t)));
  m["tableau"] = std::any(tableau_any);

  std::vector<int> deck_sizes(3);
  for (int i = 0; i < 3; ++i) deck_sizes[i] = static_cast<int>(d.decks[i].size());
  m["deck_sizes"] = std::any(deck_sizes);

  std::vector<int> all_deck_ids;
  for (int t = 0; t < 3; ++t) {
    for (auto cid : d.decks[t]) all_deck_ids.push_back(static_cast<int>(cid));
  }
  m["_test_all_deck_ids"] = std::any(all_deck_ids);

  std::vector<int> tableau_ids;
  for (int t = 0; t < 3; ++t) {
    for (int slot = 0; slot < d.tableau_size[t]; ++slot) {
      tableau_ids.push_back(static_cast<int>(d.tableau[t][slot]));
    }
  }
  m["_test_tableau_ids"] = std::any(tableau_ids);

  const auto& noble_reqs = board_ai::splendor::splendor_nobles();
  std::vector<AnyMap> nobles;
  for (int ni = 0; ni < d.nobles_size; ++ni) {
    AnyMap nm;
    int noble_id = d.nobles[ni];
    nm["noble_id"] = std::any(noble_id);
    if (noble_id >= 0 && noble_id < static_cast<int>(noble_reqs.size())) {
      std::vector<int> req(board_ai::splendor::kColorCount);
      for (int j = 0; j < board_ai::splendor::kColorCount; ++j) req[j] = noble_reqs[noble_id][j];
      nm["requirements"] = std::any(req);
    }
    nobles.push_back(std::move(nm));
  }
  m["nobles"] = std::any(nobles);

  return m;
}

template <int NPlayers>
AnyMap describe_splendor(ActionId action) {
  using Cfg = board_ai::splendor::SplendorConfig<NPlayers>;
  AnyMap m;
  m["action_id"] = std::any(static_cast<int>(action));

  if (action == Cfg::kPassAction) {
    m["type"] = std::any(std::string("pass"));
  } else if (action >= Cfg::kReturnTokenOffset && action < Cfg::kReturnTokenOffset + Cfg::kReturnTokenCount) {
    int token = action - Cfg::kReturnTokenOffset;
    m["type"] = std::any(std::string("return_token"));
    m["token"] = std::any(token);
    m["token_name"] = std::any(std::string(kColorNames[token]));
  } else if (action >= Cfg::kChooseNobleOffset && action < Cfg::kChooseNobleOffset + Cfg::kChooseNobleCount) {
    m["type"] = std::any(std::string("choose_noble"));
    m["noble_slot"] = std::any(action - Cfg::kChooseNobleOffset);
  } else if (action >= Cfg::kBuyFaceupOffset && action < Cfg::kBuyFaceupOffset + Cfg::kBuyFaceupCount) {
    int idx = action - Cfg::kBuyFaceupOffset;
    m["type"] = std::any(std::string("buy_faceup"));
    m["tier"] = std::any(idx / 4);
    m["slot"] = std::any(idx % 4);
  } else if (action >= Cfg::kReserveFaceupOffset && action < Cfg::kReserveFaceupOffset + Cfg::kReserveFaceupCount) {
    int idx = action - Cfg::kReserveFaceupOffset;
    m["type"] = std::any(std::string("reserve_faceup"));
    m["tier"] = std::any(idx / 4);
    m["slot"] = std::any(idx % 4);
  } else if (action >= Cfg::kReserveDeckOffset && action < Cfg::kReserveDeckOffset + Cfg::kReserveDeckCount) {
    m["type"] = std::any(std::string("reserve_deck"));
    m["tier"] = std::any(action - Cfg::kReserveDeckOffset);
  } else if (action >= Cfg::kBuyReservedOffset && action < Cfg::kBuyReservedOffset + Cfg::kBuyReservedCount) {
    m["type"] = std::any(std::string("buy_reserved"));
    m["slot"] = std::any(action - Cfg::kBuyReservedOffset);
  } else if (action >= Cfg::kTakeThreeOffset && action < Cfg::kTakeThreeOffset + Cfg::kTakeThreeCount) {
    m["type"] = std::any(std::string("take_three"));
    int combo = action - Cfg::kTakeThreeOffset;
    int idx = 0;
    std::vector<int> colors;
    for (int a = 0; a < 5 && static_cast<int>(colors.size()) < 3; ++a)
      for (int b = a + 1; b < 5 && static_cast<int>(colors.size()) < 3; ++b)
        for (int c = b + 1; c < 5; ++c) {
          if (idx == combo) { colors = {a, b, c}; goto done_t3; }
          ++idx;
        }
    done_t3:
    m["colors"] = std::any(colors);
  } else if (action >= Cfg::kTakeTwoDifferentOffset && action < Cfg::kTakeTwoDifferentOffset + Cfg::kTakeTwoDifferentCount) {
    m["type"] = std::any(std::string("take_two_different"));
    int combo = action - Cfg::kTakeTwoDifferentOffset;
    int idx = 0;
    std::vector<int> colors;
    for (int a = 0; a < 5; ++a)
      for (int b = a + 1; b < 5; ++b) {
        if (idx == combo) { colors = {a, b}; goto done_t2d; }
        ++idx;
      }
    done_t2d:
    m["colors"] = std::any(colors);
  } else if (action >= Cfg::kTakeOneOffset && action < Cfg::kTakeOneOffset + Cfg::kTakeOneCount) {
    m["type"] = std::any(std::string("take_one"));
    m["color"] = std::any(action - Cfg::kTakeOneOffset);
  } else if (action >= Cfg::kTakeTwoSameOffset && action < Cfg::kTakeTwoSameOffset + Cfg::kTakeTwoSameCount) {
    m["type"] = std::any(std::string("take_two_same"));
    m["color"] = std::any(action - Cfg::kTakeTwoSameOffset);
  }
  return m;
}

// --- Public-event protocol -------------------------------------------------
//
// Splendor has two hidden-state touchpoints during play:
//   (1) Buying or reserving a face-up card auto-flips a new card from the
//       corresponding tier's deck onto the same tableau slot. The new card's
//       ID is public — every player at the table sees the flip.
//   (2) Reserving from the top of a deck (a "blind reserve"): the drawer
//       sees their new face-down card; other players don't.
//
// Event schema:
//   post-action "deck_flip": {"tier", "slot", "card_id"} — emitted for each
//     tableau slot whose card changed after the action. card_id = -1 means
//     the slot is now empty (deck exhausted).
//   post-action "self_reserve_deck": {"slot", "card_id"} — emitted only when
//     the acting player is the traced perspective AND the action was a
//     reserve-from-deck. Overrides the AI's randomly-drawn card with ground
//     truth's actual card so the belief tracker's seen_cards agrees.
//   (Opponent reserve-from-deck needs no event — AI's random card in the
//    face-down slot is belief-consistent, since AI never learns what
//    opponent reserved blindly.)

namespace splendor_events {

using board_ai::splendor::SplendorConfig;
using board_ai::splendor::SplendorData;
using board_ai::splendor::SplendorState;
using board_ai::splendor::SplendorPersistentNode;
using board_ai::splendor::SplendorPersistentState;

// Safe helper: materialize the state's persistent data as mutable, mutate
// it via a functor, and reseat state.persistent to point at the new data.
template <int NPlayers>
void mutate_persistent(SplendorState<NPlayers>& s,
                       const std::function<void(SplendorData<NPlayers>&)>& fn) {
  SplendorData<NPlayers> data = s.persistent.data();
  fn(data);
  auto node = std::make_shared<SplendorPersistentNode<NPlayers>>();
  node->action_from_parent = -1;
  node->materialized = std::make_shared<const SplendorData<NPlayers>>(std::move(data));
  s.persistent = SplendorPersistentState<NPlayers>(std::move(node));
  s.undo_stack.clear();
}

template <int NPlayers>
AnyMap extract_initial_observation(const IGameState& state, int /*perspective*/) {
  using Cfg = SplendorConfig<NPlayers>;
  const auto& s = board_ai::checked_cast<SplendorState<NPlayers>>(state);
  const auto& d = s.persistent.data();
  AnyMap out;
  std::vector<std::vector<int>> tableau(3);
  for (int t = 0; t < 3; ++t) {
    tableau[t].resize(4, -1);
    for (int slot = 0; slot < d.tableau_size[t]; ++slot) {
      tableau[t][slot] = static_cast<int>(d.tableau[t][slot]);
    }
  }
  std::vector<std::any> tableau_any;
  for (auto& t : tableau) tableau_any.push_back(std::any(std::move(t)));
  out["tableau"] = std::any(tableau_any);

  std::vector<int> nobles;
  for (int i = 0; i < d.nobles_size; ++i) nobles.push_back(static_cast<int>(d.nobles[i]));
  out["nobles"] = std::any(nobles);
  return out;
}

template <int NPlayers>
void apply_initial_observation(IGameState& state, int /*perspective*/, const AnyMap& obs) {
  using Cfg = SplendorConfig<NPlayers>;
  auto& s = board_ai::checked_cast<SplendorState<NPlayers>>(state);
  auto it_t = obs.find("tableau");
  auto it_n = obs.find("nobles");
  if (it_t == obs.end() || it_n == obs.end()) {
    throw std::runtime_error("splendor initial_observation needs 'tableau' and 'nobles'");
  }
  auto tableau_any = std::any_cast<std::vector<std::any>>(it_t->second);
  if (tableau_any.size() != 3) {
    throw std::runtime_error("splendor initial_observation: tableau must have 3 tiers");
  }
  std::array<std::vector<int>, 3> gt_tableau;
  for (int t = 0; t < 3; ++t) {
    gt_tableau[t] = std::any_cast<std::vector<int>>(tableau_any[t]);
  }
  std::vector<int> gt_nobles = std::any_cast<std::vector<int>>(it_n->second);

  mutate_persistent<NPlayers>(s, [&](SplendorData<NPlayers>& d) {
    // Set tableau cards and recompute tableau_size.
    for (int t = 0; t < 3; ++t) {
      int size = 0;
      for (int slot = 0; slot < 4; ++slot) {
        const int cid = (slot < static_cast<int>(gt_tableau[t].size())) ? gt_tableau[t][slot] : -1;
        d.tableau[t][slot] = static_cast<std::int16_t>(cid);
        if (cid >= 0) size = slot + 1;
      }
      d.tableau_size[t] = static_cast<std::int8_t>(size);
    }
    // Set nobles.
    for (int i = 0; i < Cfg::kNobleCount; ++i) {
      d.nobles[i] = (i < static_cast<int>(gt_nobles.size()))
          ? static_cast<std::int16_t>(gt_nobles[i]) : static_cast<std::int16_t>(-1);
    }
    d.nobles_size = static_cast<std::int8_t>(std::min<int>(gt_nobles.size(), Cfg::kNobleCount));

    // Rebuild decks: all cards that aren't on the tableau or in nobles and
    // aren't already dealt/reserved/discarded. At game start, only the
    // tableau has been dealt from decks. The persistent tree cache is now
    // stale — mutate_persistent resets that via reseating.
    std::unordered_set<int> on_tableau;
    for (int t = 0; t < 3; ++t) {
      for (int slot = 0; slot < d.tableau_size[t]; ++slot) {
        if (d.tableau[t][slot] >= 0) on_tableau.insert(d.tableau[t][slot]);
      }
    }
    const auto& pool = board_ai::splendor::splendor_card_pool();
    for (int t = 0; t < 3; ++t) d.decks[t].clear();
    for (int cid = 0; cid < static_cast<int>(pool.size()); ++cid) {
      if (on_tableau.count(cid)) continue;
      const int tier = pool[cid].tier - 1;
      if (tier >= 0 && tier < 3) {
        d.decks[tier].push_back(static_cast<std::int16_t>(cid));
      }
    }
  });
}


template <int NPlayers>
PublicEventTrace extract_events(
    const IGameState& before,
    ActionId action,
    const IGameState& after,
    int perspective) {
  using Cfg = SplendorConfig<NPlayers>;
  const auto& sb = board_ai::checked_cast<SplendorState<NPlayers>>(before);
  const auto& sa = board_ai::checked_cast<SplendorState<NPlayers>>(after);
  const auto& db = sb.persistent.data();
  const auto& da = sa.persistent.data();
  PublicEventTrace out;

  const int actor = db.current_player;

  // opp_buy_reserved_reveal (pre): when actor != perspective buys their
  // own hidden reserved card (action 27/28/29 on a reserved_visible==0
  // slot), the card ID becomes public because its cost is paid from the
  // bank. The API side had a sampled placeholder for this card — we must
  // override it to the true card ID BEFORE do_action_fast so that
  // pay_for_card computes the correct payment.
  if (action >= Cfg::kBuyReservedOffset && action < Cfg::kBuyReservedOffset + Cfg::kBuyReservedCount &&
      actor != perspective && actor >= 0 && actor < NPlayers) {
    const int idx = action - Cfg::kBuyReservedOffset;
    if (idx < db.reserved_size[actor] &&
        db.reserved_visible[actor][static_cast<size_t>(idx)] == 0) {
      AnyMap payload;
      payload["player"] = std::any(actor);
      payload["slot"] = std::any(idx);
      payload["card_id"] = std::any(static_cast<int>(db.reserved[actor][static_cast<size_t>(idx)]));
      out.events.emplace_back("opp_buy_reserved_reveal", std::move(payload));
    }
  }

  // deck_flip: any tableau slot whose card changed is a public flip.
  for (int t = 0; t < 3; ++t) {
    const int max_slot = std::max<int>(db.tableau_size[t], da.tableau_size[t]);
    for (int slot = 0; slot < max_slot; ++slot) {
      const int before_id = (slot < db.tableau_size[t]) ? static_cast<int>(db.tableau[t][slot]) : -1;
      const int after_id = (slot < da.tableau_size[t]) ? static_cast<int>(da.tableau[t][slot]) : -1;
      if (before_id != after_id) {
        AnyMap payload;
        payload["tier"] = std::any(t);
        payload["slot"] = std::any(slot);
        payload["card_id"] = std::any(after_id);
        out.events.emplace_back("deck_flip", std::move(payload));
      }
    }
  }

  // self_reserve_deck: emitted only when perspective reserved from deck top.
  const bool is_reserve_deck = action >= Cfg::kReserveDeckOffset &&
      action < Cfg::kReserveDeckOffset + Cfg::kReserveDeckCount;
  if (is_reserve_deck && actor == perspective) {
    // Find the new reserved slot (one whose card changed from -1 to a valid ID).
    for (int slot = 0; slot < 3; ++slot) {
      const int before_id = (slot < db.reserved_size[actor]) ? static_cast<int>(db.reserved[actor][slot]) : -1;
      const int after_id = (slot < da.reserved_size[actor]) ? static_cast<int>(da.reserved[actor][slot]) : -1;
      if (before_id != after_id) {
        AnyMap payload;
        payload["player"] = std::any(actor);
        payload["slot"] = std::any(slot);
        payload["card_id"] = std::any(after_id);
        out.events.emplace_back("self_reserve_deck", std::move(payload));
      }
    }
  }

  // full post-action public snapshot for
  // message-driven public state推进. Walker-driven via
  // viz::serialize_public — reads typed values through
  // SplendorState::read_field_slot.
  {
    AnyMap snap;
    board_ai::viz::serialize_public(after, SplendorState<NPlayers>::schema(), snap);

    // Snapshot-only keys (not schema fields):
    //  - reserved_faceup_ids_flat: partial-reveal sidecar for the
    //    schema "reserved_visible" gate
    // (deck_sizes is now a schema field, walked by serialize_public.)
    std::vector<int> reserved_faceup_ids_flat(NPlayers * 3, -1);
    for (int p = 0; p < NPlayers; ++p) {
      for (int i = 0; i < 3; ++i) {
        if (da.reserved_visible[p][i] != 0) {
          reserved_faceup_ids_flat[p * 3 + i] = static_cast<int>(da.reserved[p][i]);
        }
      }
    }
    snap["reserved_faceup_ids_flat"] = std::any(reserved_faceup_ids_flat);

    out.public_snapshot = std::move(snap);
  }

  return out;
}

// applier — inverse of the snapshot extractor above. Walker-driven via
// viz::apply_public — writes typed values through
// SplendorState::write_field_slot. The remaining snapshot-only key
// (reserved_faceup_ids_flat) is handled here in a trailing
// mutate_persistent block. Hidden fields (face-down reserved ids, deck
// contents) are left for randomize_unseen to fill.
template <int NPlayers>
void apply_public_state(IGameState& state, const AnyMap& snap) {
  auto& s = board_ai::checked_cast<SplendorState<NPlayers>>(state);

  // Schema-driven public fields.
  board_ai::viz::apply_public(state, SplendorState<NPlayers>::schema(), snap);

  // Snapshot-only keys: variable-length vectors and partial-reveal sidecar.
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
  auto faceup_ids_flat = get_iv("reserved_faceup_ids_flat");

  mutate_persistent<NPlayers>(s, [&](SplendorData<NPlayers>& d) {
    // Partial-reveal sidecar: face-up reserved cards overwrite from
    // snapshot (gated on schema-applied reserved_visible). Face-down
    // reserved cards: leave as-is (tracker / self_reserve_deck handle).
    // Trailing slots beyond reserved_size are inactive — clear them to
    // -1 so observer's view matches truth's remove_reserved_at, which
    // sets the freed slot to -1 after compaction. Without this clear,
    // a buy-reserved that compacts the array leaves stale cids in the
    // observer's owner-visible trailing slots and the perspective hash
    // diverges from truth.
    for (int p = 0; p < NPlayers; ++p) {
      const int rs = static_cast<int>(d.reserved_size[p]);
      for (int i = 0; i < 3; ++i) {
        const int idx = p * 3 + i;
        if (i >= rs) {
          d.reserved[p][i] = -1;
          continue;
        }
        if (idx < static_cast<int>(faceup_ids_flat.size()) &&
            d.reserved_visible[p][i] != 0 &&
            faceup_ids_flat[idx] >= 0) {
          d.reserved[p][i] = static_cast<std::int16_t>(faceup_ids_flat[idx]);
        }
      }
    }
  });

  // Re-sync state.viz_["reserved"] from the just-applied reserved_visible.
  // The API path skips do_action_fast, so the rules-side
  // reveal_slot/reset_to_base transitions never run on observer state;
  // without this call, the schema-driven hash would walk a stale viz_
  // and diverge from truth (which did run do_action_fast). The actual viz
  // writes live in splendor_rules.cpp::sync_splendor_reserved_viz to keep
  // the I1 invariant (rules.cpp is the sole writer of viz_).
  board_ai::splendor::sync_splendor_reserved_viz<NPlayers>(state);
}

}  // namespace splendor_events

// Splendor heuristic — inspired by the reference impl in the old
// DinoBoard (参考项目/DinoBoard-main/...cpp_splendor_engine_module.cpp),
// **rewritten to strictly respect the observer-vs-truth boundary**.
//
// Leak audit (from the reference):
//   - The old impl read `d.reserved[op][slot]` without checking
//     `reserved_visible[op][slot]` — i.e. it read opponent's BLIND
//     reserved card IDs, which is hidden information. That's a leak.
//   - It also iterated over noble slots (public, fine) and
//     player_bonuses / player_gems / player_points (all public, fine).
//
// Our rewrite:
//   - Self reserved: full access (we own it).
//   - Opp reserved: ONLY card IDs where reserved_visible[op][slot]==1.
//     Blind opp reserved is never inspected; its effect is approximated
//     generically (slot-count pressure only).
//   - Never touch d.decks[] content.
//
// This makes the picker safe to run on selfplay truth state: even though
// the state object contains opp's blind card IDs, the heuristic code path
// is identical to what it would read from a belief-sampled world.
namespace splendor_heuristic {

using board_ai::splendor::SplendorCard;
using board_ai::splendor::SplendorConfig;
using board_ai::splendor::SplendorData;
using board_ai::splendor::SplendorRules;
using board_ai::splendor::SplendorState;
using board_ai::splendor::kColorCount;
using board_ai::splendor::splendor_card_pool;
using board_ai::splendor::splendor_nobles;

// How many gems of each color `player` is short of affording this card,
// given their gems + bonuses (public info). Ignores gold (wildcards) for
// simplicity — mild over-estimate of deficit.
template <int NPlayers>
int deficit_for_card(const SplendorData<NPlayers>& d, int player, const SplendorCard& card) {
  int total_short = 0;
  int gold = d.player_gems[player][5];
  for (int c = 0; c < kColorCount; ++c) {
    int raw = card.cost[c];
    int bonus = d.player_bonuses[player][c];
    int have = d.player_gems[player][c];
    int need = std::max(0, raw - bonus);
    int short_c = std::max(0, need - have);
    // Gold can cover any color — greedily apply, capped by what we have.
    int gold_use = std::min(short_c, gold);
    short_c -= gold_use;
    gold -= gold_use;
    total_short += short_c;
  }
  return total_short;
}

// Noble requirement we're closest to. Returns count of bonuses still
// missing for the nearest noble (best_missing). Public info only.
template <int NPlayers>
int nearest_noble_missing(const SplendorData<NPlayers>& d, int player) {
  const auto& nobles = splendor_nobles();
  int best = 99;
  for (int i = 0; i < d.nobles_size; ++i) {
    int nid = d.nobles[i];
    if (nid < 0 || nid >= static_cast<int>(nobles.size())) continue;
    int missing = 0;
    for (int c = 0; c < kColorCount; ++c) {
      missing += std::max(0, static_cast<int>(nobles[nid][c]) - d.player_bonuses[player][c]);
    }
    best = std::min(best, missing);
  }
  return best;
}

// Board position eval from perspective of `player`. Public + own-private only.
template <int NPlayers>
double eval_position(const SplendorData<NPlayers>& d, int player) {
  if (d.terminal) {
    if (d.winner < 0) return 0.0;
    return d.winner == player ? 1000.0 : -1000.0;
  }
  double score = 0.0;
  // Points (public, public).
  for (int p = 0; p < NPlayers; ++p) {
    double sign = (p == player) ? 1.0 : -1.0 / std::max(1, NPlayers - 1);
    score += sign * static_cast<double>(d.player_points[p]) * 8.0;
  }
  // Bonuses = engine strength (public).
  for (int p = 0; p < NPlayers; ++p) {
    double sign = (p == player) ? 1.0 : -0.5 / std::max(1, NPlayers - 1);
    int bonus_total = 0;
    for (int c = 0; c < kColorCount; ++c) bonus_total += d.player_bonuses[p][c];
    score += sign * static_cast<double>(bonus_total) * 0.8;
  }
  // Nearest-noble proximity (public).
  int my_missing = nearest_noble_missing(d, player);
  score -= static_cast<double>(std::min(my_missing, 5)) * 0.6;
  // Affordable buys on tableau (public).
  const auto& cards = splendor_card_pool();
  for (int tier = 0; tier < 3; ++tier) {
    for (int slot = 0; slot < d.tableau_size[tier]; ++slot) {
      int cid = d.tableau[tier][slot];
      if (cid < 0 || cid >= static_cast<int>(cards.size())) continue;
      int deficit = deficit_for_card(d, player, cards[cid]);
      if (deficit == 0) {
        score += 0.8 + static_cast<double>(cards[cid].points) * 0.4;
      } else if (deficit <= 2) {
        score += 0.25;
      }
    }
  }
  // Own reserved (fully legal to inspect).
  for (int slot = 0; slot < d.reserved_size[player]; ++slot) {
    int cid = d.reserved[player][slot];
    if (cid < 0 || cid >= static_cast<int>(cards.size())) continue;
    int deficit = deficit_for_card(d, player, cards[cid]);
    score += static_cast<double>(cards[cid].points) * 0.6;
    score -= static_cast<double>(deficit) * 0.35;
  }
  // Opponents' reserved — ONLY visible slots; blind slots add a small
  // slot-count pressure without reading their card IDs (leak-safe).
  for (int p = 0; p < NPlayers; ++p) {
    if (p == player) continue;
    for (int slot = 0; slot < d.reserved_size[p]; ++slot) {
      bool visible = d.reserved_visible[p][slot] != 0;
      if (visible) {
        int cid = d.reserved[p][slot];
        if (cid < 0 || cid >= static_cast<int>(cards.size())) continue;
        score -= static_cast<double>(cards[cid].points) * 0.5;
        // If they can afford it, they're a threat.
        int op_def = deficit_for_card(d, p, cards[cid]);
        if (op_def == 0) score -= 0.5;
      } else {
        // Blind slot — don't read card_id. Just treat as "they have a
        // hidden bought-ready card", lightly negative.
        score -= 0.15;
      }
    }
  }
  // Coin overflow penalty (Splendor caps at 10 gems).
  int my_gems = 0;
  for (int i = 0; i < 6; ++i) my_gems += d.player_gems[player][i];
  if (my_gems > 10) score -= static_cast<double>(my_gems - 10) * 0.8;
  return score;
}

template <int NPlayers>
board_ai::HeuristicResult pick(
    board_ai::IGameState& state,
    const board_ai::IGameRules& rules,
    std::uint64_t /*rng_seed*/) {
  auto& s = board_ai::checked_cast<SplendorState<NPlayers>>(state);
  auto legal = rules.legal_actions(state);
  const int player = s.current_player();

  board_ai::HeuristicResult result;
  result.actions = legal;
  result.scores.reserve(legal.size());

  // Score each action via lookahead: apply action → eval from player's
  // perspective → undo. do_action_deterministic / undo_action are const
  // on IGameRules (state-only mutation lives on the IGameState&), so no
  // const_cast is needed even though `rules` is `const IGameRules&`.
  for (ActionId a : legal) {
    auto tok = rules.do_action_deterministic(state, a);
    double s_val = eval_position<NPlayers>(s.persistent.data(), player);
    rules.undo_action(state, tok);
    result.scores.push_back(s_val);
  }
  return result;
}

}  // namespace splendor_heuristic

template <int NPlayers>
board_ai::GameBundle make_splendor(const std::string& game_id, std::uint64_t seed) {
  using Cfg = board_ai::splendor::SplendorConfig<NPlayers>;
  board_ai::GameBundle b;
  b.game_id = game_id;
  auto s = std::make_unique<board_ai::splendor::SplendorState<NPlayers>>();
  s->reset_with_seed(seed);
  b.state = std::move(s);
  b.rules = std::make_unique<board_ai::splendor::SplendorRules<NPlayers>>();
  b.value_model = std::make_unique<board_ai::DefaultStateValueModel>();
  b.encoder = std::make_unique<board_ai::splendor::SplendorFeatureEncoder<NPlayers>>();
  b.belief_tracker = std::make_unique<board_ai::splendor::SplendorBeliefTracker<NPlayers>>();
  b.state_serializer = serialize_splendor<NPlayers>;
  b.action_descriptor = describe_splendor<NPlayers>;
  b.heuristic_picker = splendor_heuristic::pick<NPlayers>;
  b.public_event_extractor = splendor_events::extract_events<NPlayers>;
  b.public_state_applier = splendor_events::apply_public_state<NPlayers>;
  b.initial_observation_extractor = splendor_events::extract_initial_observation<NPlayers>;
  b.initial_observation_applier = splendor_events::apply_initial_observation<NPlayers>;

  b.tail_solver = std::make_unique<board_ai::search::AlphaBetaTailSolver>();
  // SplendorRules::do_action_deterministic sets forced_draw_override = -2
  // so replenishing the visible tier skips drawing from the hidden deck.
  // Tail solver never consumes hidden chance outcomes → safe.
  b.stochastic_tail_solve_safe = true;

  b.tail_solve_trigger = [](const board_ai::IGameState& state, int /*ply*/) -> bool {
    const auto& s = board_ai::checked_cast<board_ai::splendor::SplendorState<NPlayers>>(state);
    const auto& d = s.persistent.data();
    for (int p = 0; p < NPlayers; ++p) {
      if (d.player_points[static_cast<size_t>(p)] >= 10) return true;
    }
    return false;
  };

  b.episode_stats_extractor = [](
      const board_ai::IGameState& final_state,
      const std::vector<board_ai::SelfplaySampleView>& samples)
      -> std::map<std::string, double> {
    int main_actions = 0;
    for (const auto& s : samples) {
      if (s.action_id < Cfg::kChooseNobleOffset || s.action_id == Cfg::kPassAction) {
        ++main_actions;
      }
    }
    double turns = static_cast<double>(main_actions) / NPlayers;
    return {{"turns", turns}};
  };

  return b;
}

board_ai::GameRegistrar reg_splendor("splendor", [](std::uint64_t seed) {
  return make_splendor<2>("splendor", seed);
});
board_ai::GameRegistrar reg_splendor_2p("splendor_2p", [](std::uint64_t seed) {
  return make_splendor<2>("splendor_2p", seed);
});
board_ai::GameRegistrar reg_splendor_3p("splendor_3p", [](std::uint64_t seed) {
  return make_splendor<3>("splendor_3p", seed);
});
board_ai::GameRegistrar reg_splendor_4p("splendor_4p", [](std::uint64_t seed) {
  return make_splendor<4>("splendor_4p", seed);
});

}  // namespace
