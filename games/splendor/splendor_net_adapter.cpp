#include "splendor_net_adapter.h"

#include <algorithm>
#include <unordered_set>
#include <vector>

#include "../../engine/core/masked_state.h"
#include "../../engine/core/viz_runtime.h"

namespace board_ai::splendor {

namespace {

void encode_card(const SplendorCard* card, std::vector<float>* out) {
  if (!card) {
    out->insert(out->end(), 13, 0.0f);
    return;
  }
  out->push_back(1.0f);
  out->push_back(static_cast<float>(card->tier) / 3.0f);
  out->push_back(static_cast<float>(card->points) / 5.0f);
  for (int c = 0; c < kColorCount; ++c) {
    out->push_back(card->bonus == c ? 1.0f : 0.0f);
  }
  for (int c = 0; c < kColorCount; ++c) {
    out->push_back(static_cast<float>(card->cost[static_cast<size_t>(c)]) / 7.0f);
  }
}

void encode_hidden_reserved_placeholder(std::vector<float>* out) {
  out->push_back(1.0f);
  out->insert(out->end(), 12, 0.0f);
}

}  // namespace

// Layout reminder (kFeatureDim = 6 + 15*P + kNobleCount*6 + 12*13 + 39*P + 7):
//   Public (kFeatureDim - 39):
//     - bank (6)
//     - per-player stats in perspective order (15 * P)
//     - nobles (kNobleCount * 6)
//     - tableau (12 * 13)
//     - opp reserved slots (39 * (P-1)): visible → card, hidden → placeholder
//     - metadata (7)
//   Private (39):
//     - perspective's own 3 reserved slots × 13 dims (real cards — visible
//       AND hidden both carry the real card ID since perspective knows them)
template <int NPlayers>
void SplendorFeatureEncoder<NPlayers>::encode_features(
    const IGameState& state,
    int perspective_player,
    const IBeliefTracker* /*tracker*/,
    std::vector<float>* out) const {
  const auto* s = dynamic_cast<const SplendorState<NPlayers>*>(&state);
  if (!s || !out || perspective_player < 0 || perspective_player >= Cfg::kPlayers) return;
  const SplendorData<NPlayers>& d = s->persistent.data();
  const auto& cards = splendor_card_pool();
  const auto& nobles = splendor_nobles();

  for (int i = 0; i < kTokenTypes; ++i) {
    const float denom = (i == 5) ? 5.0f : static_cast<float>(Cfg::kGemCount);
    out->push_back(static_cast<float>(d.bank[static_cast<size_t>(i)]) / denom);
  }

  for (int pi = 0; pi < Cfg::kPlayers; ++pi) {
    const int pid = (perspective_player + pi) % Cfg::kPlayers;
    for (int i = 0; i < kTokenTypes; ++i) {
      out->push_back(static_cast<float>(d.player_gems[pid][static_cast<size_t>(i)]) / 10.0f);
    }
    for (int i = 0; i < kColorCount; ++i) {
      out->push_back(static_cast<float>(d.player_bonuses[pid][static_cast<size_t>(i)]) / 7.0f);
    }
    out->push_back(static_cast<float>(d.player_points[pid]) / 20.0f);
    out->push_back(static_cast<float>(d.reserved_size[pid]) / 3.0f);
    out->push_back(static_cast<float>(d.player_cards_count[pid]) / 20.0f);
    out->push_back(static_cast<float>(d.player_nobles_count[pid]) / 3.0f);
  }

  for (int i = 0; i < Cfg::kNobleCount; ++i) {
    if (i < d.nobles_size) {
      const int nid = d.nobles[static_cast<size_t>(i)];
      if (nid >= 0 && nid < static_cast<int>(nobles.size())) {
        for (int c = 0; c < kColorCount; ++c) {
          out->push_back(static_cast<float>(nobles[static_cast<size_t>(nid)][static_cast<size_t>(c)]) / 4.0f);
        }
        out->push_back(1.0f);
      } else {
        out->insert(out->end(), 6, 0.0f);
      }
    } else {
      out->insert(out->end(), 6, 0.0f);
    }
  }

  for (int tier = 0; tier < 3; ++tier) {
    for (int slot = 0; slot < 4; ++slot) {
      const bool exists = slot < d.tableau_size[static_cast<size_t>(tier)];
      if (!exists) {
        out->insert(out->end(), 13, 0.0f);
        continue;
      }
      const int cid = d.tableau[static_cast<size_t>(tier)][static_cast<size_t>(slot)];
      if (cid < 0 || cid >= static_cast<int>(cards.size())) {
        out->insert(out->end(), 13, 0.0f);
      } else {
        encode_card(&cards[static_cast<size_t>(cid)], out);
      }
    }
  }

  // Non-perspective players' reserved slots. The encoder reads the
  // MaskedState's `reserved` field directly: face-down opp slots arrive
  // as kPlaceholderInt32 (framework wrote it via mask_field_slot).
  // Visible (face-up) opp slots carry the real cid. There is no viz
  // query here — the placeholder IS the visibility signal.
  for (int pi = 1; pi < Cfg::kPlayers; ++pi) {
    const int pid = (perspective_player + pi) % Cfg::kPlayers;
    for (int slot = 0; slot < 3; ++slot) {
      const bool exists = slot < d.reserved_size[pid];
      if (!exists) {
        out->insert(out->end(), 13, 0.0f);
        continue;
      }
      const int cid = d.reserved[pid][static_cast<size_t>(slot)];
      if (cid == static_cast<std::int16_t>(kPlaceholderInt32)) {
        encode_hidden_reserved_placeholder(out);
        continue;
      }
      if (cid < 0 || cid >= static_cast<int>(cards.size())) {
        out->insert(out->end(), 13, 0.0f);
      } else {
        encode_card(&cards[static_cast<size_t>(cid)], out);
      }
    }
  }

  out->push_back(static_cast<float>(std::min(d.plies, kMaxPlies)) / static_cast<float>(kMaxPlies));
  out->push_back(d.current_player == perspective_player ? 1.0f : 0.0f);
  out->push_back(static_cast<float>(std::max(0, d.pending_returns)) / 5.0f);
  const SplendorTurnStage stage = static_cast<SplendorTurnStage>(d.stage);
  out->push_back(stage == SplendorTurnStage::kNormal ? 1.0f : 0.0f);
  out->push_back(stage == SplendorTurnStage::kReturnTokens ? 1.0f : 0.0f);
  out->push_back(stage == SplendorTurnStage::kChooseNoble ? 1.0f : 0.0f);
  out->push_back(d.first_player == perspective_player ? 1.0f : 0.0f);

  // Perspective's own 3 reserved slots, full card detail regardless of
  // public visibility — perspective is the owner so they always know.
  for (int slot = 0; slot < 3; ++slot) {
    const bool exists = slot < d.reserved_size[perspective_player];
    if (!exists) {
      out->insert(out->end(), 13, 0.0f);
      continue;
    }
    const int cid =
        d.reserved[perspective_player][static_cast<size_t>(slot)];
    if (cid < 0 || cid >= static_cast<int>(cards.size())) {
      out->insert(out->end(), 13, 0.0f);
    } else {
      encode_card(&cards[static_cast<size_t>(cid)], out);
    }
  }
}

template <int NPlayers>
void SplendorBeliefTracker<NPlayers>::init(
    IGameState& state, int /*perspective*/, const AnyMap& /*payload*/) {
  // Perspective-agnostic: tracker holds only public card-multiset
  // aggregates. Per-perspective private knowledge (own reserved card
  // ids) is read from state.viz=1 slots in randomize_unseen, not here.
  // No payload — Splendor's opening reveals nothing perspective-private
  // (the actor's reserved slots all start empty).
  if (initialized_) return;
  seen_cards_.clear();
  initialized_ = true;

  // Tableau is schema field "tableau" with shape {3, 4} and base
  // viz=all_public — every slot is visible to every perspective. Read
  // each slot via read_field_slot. Empty slots return -1 (per
  // write_field_slot: tableau slots beyond tableau_size carry -1).
  for (int t = 0; t < 3; ++t) {
    const int size = std::any_cast<int>(
        state.read_field_slot("tableau_size", {t}));
    for (int slot = 0; slot < size; ++slot) {
      const int cid = std::any_cast<int>(
          state.read_field_slot("tableau", {t, slot}));
      if (cid >= 0) seen_cards_.insert(cid);
    }
  }
  // Nobles are public but aren't "cards" for the deck-pool belief.
  // Reserved slots at game start are all empty — nothing to add.
}

template <int NPlayers>
AnyMap SplendorBeliefTracker<NPlayers>::pack_init_payload(
    const IGameState& /*gt_state*/, int /*perspective*/) const {
  // No perspective-private bootstrap — Splendor opens with empty
  // reserves; tableau / bank / nobles all flow via the broadcast public
  // snapshot. randomize_unseen handles the hidden deck contents.
  return {};
}

template <int NPlayers>
void SplendorBeliefTracker<NPlayers>::observe_public_event(
    int /*actor*/,
    ActionId /*action*/,
    const std::vector<PublicEvent>& events) {
  // deck_flip events carry tableau card IDs revealed by drawing
  // from the deck to replace a bought/reserved card. Public to every
  // observer, so accumulate unconditionally.
  //
  // No `self_reserve_deck` branch: the actor's new reserve cid is
  // already on state.reserved with viz=1 for the owner — owners read
  // it through state, not through the tracker. Other observers learn
  // nothing from a blind reserve, which is correct.
  for (const auto& ev : events) {
    if (ev.first == "deck_flip") {
      auto cit = ev.second.find("card_id");
      if (cit != ev.second.end()) {
        const int cid = std::any_cast<int>(cit->second);
        if (cid >= 0) seen_cards_.insert(cid);
      }
    }
  }
}

// randomize_unseen produces a world whose public fields are byte-equal
// across any two trackers with the same observation history, regardless
// of the input state's hidden contents. Called per-sim at MCTS root for
// determinization, AND at the end of each apply_observation to re-sample
// session state_'s hidden fields into a fresh tracker-consistent world.
//
// Canonical unseen-pool formula:
//   unseen_pool = full_pool − seen_cards − observer's viz=1 reserved
//
// Algorithm:
//   1. unseen_by_tier[t] = pool[t] − seen_cards − observer-known reserved
//      (the observer's own reserved card ids, found via
//      state.viz_["reserved"][p, slot, observer] == 1).
//   2. For every reserved slot the observer cannot see (viz=0): consume
//      one card from the matching tier of unseen_by_tier. The slot's
//      current cid is unreliable (it may be a stale sample or the
//      observer's already-known card under viewer rotation), so we
//      derive the tier from `tier_idx` carried in state.tableau /
//      schema separately — Splendor's reserved slots don't carry a
//      tier label of their own, so we fall back to the slot's current
//      cid → card.tier lookup. This is consistent with how the deck
//      partitions cards into tiers.
//   3. Remaining unseen_by_tier[t] → data.decks[t]. Size is exactly
//      |unseen_by_tier[t]| − (observer-hidden reserves of tier t),
//      which equals truth's deck size by construction.
template <int NPlayers>
void SplendorBeliefTracker<NPlayers>::randomize_unseen(IGameState& state, int observer, std::mt19937_64& rng) const {
  auto* s = dynamic_cast<SplendorState<NPlayers>*>(&state);
  if (!s) return;

  SplendorData<NPlayers> data = s->persistent.data();
  const auto& cards = splendor_card_pool();
  const int total_cards = static_cast<int>(cards.size());

  // Observer-known reserved cids: state.viz_["reserved"][p, slot, observer]==1.
  // These are already-fixed in the produced world and must be excluded
  // from the unseen pool so we don't double-deal them.
  const auto& reserved_viz = viz::viz_get(state, "reserved");
  std::unordered_set<int> observer_known;
  observer_known.reserve(static_cast<size_t>(Cfg::kPlayers * 3));
  auto is_visible_to_observer = [&](int p, int slot) -> bool {
    const std::vector<int> idx_slot{p, slot};
    const std::size_t base = viz::flat_offset_data_only(reserved_viz.shape, idx_slot);
    return reserved_viz.data[base + static_cast<std::size_t>(observer)] != 0;
  };
  for (int p = 0; p < Cfg::kPlayers; ++p) {
    for (int slot = 0; slot < data.reserved_size[p]; ++slot) {
      if (is_visible_to_observer(p, slot)) {
        const int cid = data.reserved[p][static_cast<size_t>(slot)];
        if (cid >= 0 && cid < total_cards) observer_known.insert(cid);
      }
    }
  }

  std::array<std::vector<int>, 3> unseen_by_tier{};
  for (int cid = 0; cid < total_cards; ++cid) {
    if (seen_cards_.count(cid) != 0) continue;
    if (observer_known.count(cid) != 0) continue;
    const int tier_idx = cards[static_cast<size_t>(cid)].tier - 1;
    if (tier_idx >= 0 && tier_idx < 3) {
      unseen_by_tier[static_cast<size_t>(tier_idx)].push_back(cid);
    }
  }

  for (int t = 0; t < 3; ++t) {
    std::shuffle(unseen_by_tier[static_cast<size_t>(t)].begin(),
                 unseen_by_tier[static_cast<size_t>(t)].end(), rng);
  }

  std::array<size_t, 3> idx{0, 0, 0};
  for (int p = 0; p < Cfg::kPlayers; ++p) {
    for (int slot = 0; slot < data.reserved_size[p]; ++slot) {
      if (is_visible_to_observer(p, slot)) continue;  // observer-known, keep
      const int cid = data.reserved[p][static_cast<size_t>(slot)];
      if (cid < 0 || cid >= total_cards) continue;
      const int tier_idx = cards[static_cast<size_t>(cid)].tier - 1;
      if (tier_idx < 0 || tier_idx >= 3) continue;
      auto& pool = unseen_by_tier[static_cast<size_t>(tier_idx)];
      auto& i = idx[static_cast<size_t>(tier_idx)];
      if (i < pool.size()) {
        data.reserved[p][static_cast<size_t>(slot)] =
            static_cast<std::int16_t>(pool[i++]);
      }
    }
  }

  // Deck content = remaining unseen pool per tier. Size is determined by
  // the (pool - seen - opp_hidden) formula, not by preservation of the
  // input state's deck size.
  for (int t = 0; t < 3; ++t) {
    auto& pool = unseen_by_tier[static_cast<size_t>(t)];
    auto& i = idx[static_cast<size_t>(t)];
    auto& deck = data.decks[static_cast<size_t>(t)];
    deck.clear();
    deck.reserve(pool.size() - i);
    for (; i < pool.size(); ++i) {
      deck.push_back(static_cast<std::int16_t>(pool[i]));
    }
  }

  // Caller-owned rng now drives all subsequent draws via the
  // do_action_fast(rng) path; nothing on state to reseed.
  (void)rng;

  auto node = std::make_shared<SplendorPersistentNode<NPlayers>>();
  node->action_from_parent = -1;
  node->materialized = std::make_shared<const SplendorData<NPlayers>>(std::move(data));
  s->persistent = SplendorPersistentState<NPlayers>(node);
  s->undo_stack.clear();
}

template <int NPlayers>
AnyMap SplendorBeliefTracker<NPlayers>::serialize() const {
  AnyMap out;
  // Canonical form: sorted vector of seen card IDs (unordered_set iteration
  // order varies). Two trackers with the same seen set produce equal output.
  // Perspective-agnostic: trackers fed the same observation stream from
  // different seats produce equal output (no perspective_player field).
  std::vector<int> seen(seen_cards_.begin(), seen_cards_.end());
  std::sort(seen.begin(), seen.end());
  out["seen_cards"] = seen;
  out["initialized"] = initialized_;
  return out;
}

template class SplendorFeatureEncoder<2>;
template class SplendorFeatureEncoder<3>;
template class SplendorFeatureEncoder<4>;
template class SplendorBeliefTracker<2>;
template class SplendorBeliefTracker<3>;
template class SplendorBeliefTracker<4>;

}  // namespace board_ai::splendor
