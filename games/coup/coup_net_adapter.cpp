#include "coup_net_adapter.h"

#include <algorithm>
#include <numeric>

#include "../../engine/core/viz_runtime.h"

namespace board_ai::coup {

namespace {

// Maps a claim-bearing action to the character being claimed, or -1 if the
// action itself is not a claim.
int claim_role_for_action(ActionId action) {
  if (action == kTaxAction) return kDuke;
  if (action >= kAssassinateOffset && action < kAssassinateOffset + kAssassinateCount) return kAssassin;
  if (action >= kStealOffset && action < kStealOffset + kStealCount) return kCaptain;
  if (action == kExchangeAction) return kAmbassador;
  if (action == kBlockDukeAction) return kDuke;
  if (action == kBlockContessaAction) return kContessa;
  if (action == kBlockCaptainAction) return kCaptain;
  if (action == kBlockAmbassadorAction) return kAmbassador;
  return -1;
}

// Prior multiplier per unit of signal count. prior(R) = 1 + alpha * count[R].
// With alpha=0.5 and count=3, prior=2.5 — a 2.5× bias over uniform, not
// extreme. Tuned for soft boost that keeps sampling feasible but still shifts
// MCTS toward plausible worlds.
constexpr double kSignalAlpha = 0.5;

template <int NPlayers>
int influence_count(const CoupData<NPlayers>& d, int p) {
  int n = 0;
  if (!d.revealed[p][0]) ++n;
  if (!d.revealed[p][1]) ++n;
  return n;
}

int action_type_index(ActionId action) {
  if (action == kIncomeAction) return 0;
  if (action == kForeignAidAction) return 1;
  if (action >= kCoupOffset && action < kCoupOffset + kCoupCount) return 2;
  if (action == kTaxAction) return 3;
  if (action >= kAssassinateOffset && action < kAssassinateOffset + kAssassinateCount) return 4;
  if (action >= kStealOffset && action < kStealOffset + kStealCount) return 5;
  if (action == kExchangeAction) return 6;
  return -1;
}

}  // namespace

// MaskedState read pattern (BUG-018 / encoder contract): the encoder
// reads slots through the MaskedState. viz=1 slots carry truth, viz=0
// slots carry placeholder sentinels (kPlaceholderInt8 = INT8_MIN for
// CharId/std::int8_t fields). The placeholder never equals any
// legitimate character id (0..4), so a `== c` test on a placeholder
// is always false — char-OH naturally encodes as all-zero for hidden
// slots without any explicit placeholder branch.

template <int NPlayers>
void CoupFeatureEncoder<NPlayers>::encode_public(
    const IGameState& state,
    int perspective_player,
    const IBeliefTracker* tracker,
    std::vector<float>* out) const {
  const auto* s = dynamic_cast<const CoupState<NPlayers>*>(&state);
  if (!s || !out || perspective_player < 0 || perspective_player >= NPlayers) return;
  const auto& d = s->data;
  const auto* bt =
      dynamic_cast<const CoupBeliefTracker<NPlayers>*>(tracker);

  // Per-player public (23 each). Order rotated so perspective is index 0.
  for (int pi = 0; pi < NPlayers; ++pi) {
    const int pid = (perspective_player + pi) % NPlayers;

    // alive (1), coins / 12 (1)
    out->push_back(d.alive[pid] ? 1.0f : 0.0f);
    out->push_back(static_cast<float>(d.coins[pid]) / 12.0f);

    // 2 × { revealed?(1) + char-OH(5) } = 12. char-OH only filled
    // when revealed; unrevealed slots are placeholder for non-self
    // (encoder structurally cannot read truth) — we explicitly gate
    // on `revealed` to mirror the public projection: even on the self
    // perspective we want this block to carry public-only info.
    for (int sl = 0; sl < 2; ++sl) {
      const bool rev = d.revealed[pid][sl];
      out->push_back(rev ? 1.0f : 0.0f);
      const std::int8_t card = d.influence[pid][sl];
      for (int c = 0; c < kCharacterCount; ++c) {
        out->push_back((rev && card == c) ? 1.0f : 0.0f);
      }
    }

    // role one-hots (4): active / target / blocker / challenger
    out->push_back(d.active_player == pid ? 1.0f : 0.0f);
    out->push_back(d.action_target == pid ? 1.0f : 0.0f);
    out->push_back(d.blocker == pid ? 1.0f : 0.0f);
    out->push_back(d.challenger == pid ? 1.0f : 0.0f);

    // tracker.signals[pid][role] / 4 clamped (5). A1 starter — public
    // claim/challenge/reveal history aggregated by the tracker. No
    // tracker → zero-fill.
    for (int c = 0; c < kCharacterCount; ++c) {
      float v = 0.0f;
      if (bt) {
        v = std::min(1.0f,
                     static_cast<float>(bt->signal_count(pid, c)) / 4.0f);
      }
      out->push_back(v);
    }
  }

  // ---- Global (36 + N) ----
  // stage one-hot (11)
  const int stage_idx = static_cast<int>(d.stage);
  for (int i = 0; i < kStageCount; ++i) {
    out->push_back(i == stage_idx ? 1.0f : 0.0f);
  }

  // declared_action_type one-hot (7): income/aid/coup/tax/assassinate/
  // steal/exchange
  const int action_type = action_type_index(d.declared_action);
  constexpr int kActionTypeCount = 7;
  for (int i = 0; i < kActionTypeCount; ++i) {
    out->push_back(i == action_type ? 1.0f : 0.0f);
  }

  // claimed_character one-hot (5)
  for (int c = 0; c < kCharacterCount; ++c) {
    out->push_back(d.claimed_character == c ? 1.0f : 0.0f);
  }

  // block_character one-hot (5)
  for (int c = 0; c < kCharacterCount; ++c) {
    out->push_back(d.block_character == c ? 1.0f : 0.0f);
  }

  // pending_claimer relative-to-perspective OH (N). Tracker carries
  // the pending claim cycle across stages; encoder reads it as an
  // explicit "who claimed last" pointer rotated to perspective frame.
  // No tracker → all-zero.
  for (int pi = 0; pi < NPlayers; ++pi) {
    const int pid = (perspective_player + pi) % NPlayers;
    const bool match = bt && bt->pending_claimer() == pid;
    out->push_back(match ? 1.0f : 0.0f);
  }

  // pending_challenged (1)
  out->push_back((bt && bt->pending_challenged()) ? 1.0f : 0.0f);

  // deck_size / 15 (1)
  out->push_back(static_cast<float>(d.deck_size) / 15.0f);

  // ply / 200 (1)
  constexpr float kMaxPlies = 200.0f;
  out->push_back(static_cast<float>(d.ply) / kMaxPlies);

  // revealed multiset per role (5): full-table sum of the schema
  // `revealed` flag intersected with `influence` character.
  for (int c = 0; c < kCharacterCount; ++c) {
    int count = 0;
    for (int p = 0; p < NPlayers; ++p) {
      for (int sl = 0; sl < 2; ++sl) {
        if (d.revealed[p][sl] && d.influence[p][sl] == c) ++count;
      }
    }
    out->push_back(static_cast<float>(count));
  }
}

template <int NPlayers>
void CoupFeatureEncoder<NPlayers>::encode_private(
    const IGameState& state,
    int player,
    const IBeliefTracker* /*tracker*/,
    std::vector<float>* out) const {
  const auto* s = dynamic_cast<const CoupState<NPlayers>*>(&state);
  if (!s || !out || player < 0 || player >= NPlayers) return;
  const auto& d = s->data;

  // Per-player private (22 each):
  //   2 × own influence char-OH on UNREVEALED slots (5 + 5 = 10)
  //   2 × exchange_drawn { occupied?(1) + char-OH(5) }     = 12
  // Self block populated only on perspective; other player blocks
  // zero-filled. The MaskedState was built for `player` so non-self
  // slots are placeholder — `card == c` is structurally false there
  // and would naturally zero-out, but we keep the explicit `is_self`
  // gate for clarity.
  for (int pi = 0; pi < NPlayers; ++pi) {
    const int pid = (player + pi) % NPlayers;
    const bool is_self = (pid == player);

    for (int sl = 0; sl < 2; ++sl) {
      const bool unrevealed = is_self && !d.revealed[pid][sl];
      const std::int8_t card = d.influence[pid][sl];
      for (int c = 0; c < kCharacterCount; ++c) {
        out->push_back((unrevealed && card == c) ? 1.0f : 0.0f);
      }
    }

    // exchange_drawn { occupied?(1) + char-OH(5) } × 2.
    // exchange_drawn is all_hidden base; rules call
    // reveal_slot_to(active_player) only during exchange. For
    // perspective `player`, the slot is real iff player == active_player
    // and stage is one of the exchange-return stages — outside that
    // window the fields are placeholder/-1 and naturally encode as 0.
    const bool show_drawn =
        is_self && d.active_player == player &&
        (d.stage == CoupStage::kExchangeReturn1 ||
         d.stage == CoupStage::kExchangeReturn2);
    for (int i = 0; i < kExchangeDrawSlots; ++i) {
      const std::int8_t xd = d.exchange_drawn[i];
      const bool occupied = show_drawn && xd >= 0;
      out->push_back(occupied ? 1.0f : 0.0f);
      for (int c = 0; c < kCharacterCount; ++c) {
        out->push_back((occupied && xd == c) ? 1.0f : 0.0f);
      }
    }
  }
}

template <int NPlayers>
void CoupBeliefTracker<NPlayers>::init(
    IGameState& /*state*/, int perspective,
    const AnyMap& /*payload*/) {
  // Walker-driven init: `viz::apply_public` + `viz::apply_partial_reveals`
  // already wrote the public projection and the perspective's owner_only
  // influence slots into `state` before this call. The tracker has no
  // perspective-private bootstrap to seed (Coup's only owner-private slots
  // are `influence[perspective, *]` and they ride the partial-reveal
  // sidecar, not a tracker payload). Just remember perspective for
  // `randomize_unseen` and reset signal memory.
  perspective_player_ = perspective;
  for (auto& row : signals_) row.fill(0);
  pending_claimer_ = -1;
  pending_claim_role_ = -1;
  pending_challenged_ = false;
}

template <int NPlayers>
void CoupBeliefTracker<NPlayers>::observe_public_event(
    int actor,
    ActionId action,
    const std::vector<PublicEvent>& events) {
  // ------------------------------------------------------------------
  // Phase 1: interpret `action` itself.
  // ------------------------------------------------------------------
  const int claimed = claim_role_for_action(action);
  if (claimed >= 0) {
    // A claim action starts a new claim cycle. Any unresolved pending claim
    // (shouldn't normally happen, but be safe) is discarded.
    pending_claimer_ = actor;
    pending_claim_role_ = claimed;
    pending_challenged_ = false;
  } else if (action == kChallengeAction) {
    // Challenger implicitly signals "I may hold `pending_claim_role_`" —
    // that's why they dare to challenge.
    if (pending_claim_role_ >= 0 && actor >= 0 && actor < NPlayers) {
      signals_[actor][pending_claim_role_] += 1;
    }
    pending_challenged_ = true;
  } else if (action == kAllowAction || action == kAllowNoBlockAction) {
    // No one challenged this claim (or the target chose not to block). If
    // the current claim cycle went unchallenged, the claimer's role signal
    // strengthens.
    if (pending_claimer_ >= 0 && pending_claim_role_ >= 0 && !pending_challenged_) {
      signals_[pending_claimer_][pending_claim_role_] += 1;
    }
    // Note: if pending_challenged_ is true, allow fires in a different
    // sub-flow (e.g. allowing the aftermath of a successful challenge);
    // do nothing in that case.
    // Only clear pending when the claim cycle is fully settled. A simple
    // heuristic: Allow means "we move on from this claim" — clear.
    if (!pending_challenged_) {
      pending_claimer_ = -1;
      pending_claim_role_ = -1;
    }
  }

  // ------------------------------------------------------------------
  // Phase 2: process events (revealed cards, exchange completions).
  // ------------------------------------------------------------------
  for (const auto& evt : events) {
    if (evt.first == "card_revealed") {
      const auto& payload = evt.second;
      auto it_p = payload.find("player");
      auto it_r = payload.find("role");
      if (it_p == payload.end() || it_r == payload.end()) continue;
      int p = std::any_cast<int>(it_p->second);
      int r = std::any_cast<int>(it_r->second);
      if (p < 0 || p >= NPlayers) continue;
      if (r < 0 || r >= kCharacterCount) continue;

      // Revealed role is publicly known now; any accumulated signal for
      // (p, r) is "spent" — the earlier claims are explained by this reveal
      // rather than by additional hidden copies.
      signals_[p][r] = 0;

      // Resolve pending challenge if this reveal is the claimer's response.
      if (pending_challenged_ && p == pending_claimer_ && pending_claim_role_ >= 0) {
        // Either outcome clears the claimer's signal on the claimed role:
        //   - r == pending_claim_role_: true claim, card reshuffled back to
        //     deck; claimer no longer demonstrably holds it.
        //   - r != pending_claim_role_: bluff exposed; earlier claims of
        //     pending_claim_role_ were false signals.
        signals_[pending_claimer_][pending_claim_role_] = 0;
        pending_claimer_ = -1;
        pending_claim_role_ = -1;
        pending_challenged_ = false;
      }
    } else if (evt.first == "exchange_complete") {
      const auto& payload = evt.second;
      auto it_p = payload.find("player");
      if (it_p == payload.end()) continue;
      int p = std::any_cast<int>(it_p->second);
      if (p < 0 || p >= NPlayers) continue;
      // Ambassador reshuffled p's hand — prior claim signals are stale.
      signals_[p].fill(0);
    }
  }
}

// randomize_unseen: viz-driven determinization. Reads which slots the
// observer can legitimately see from `state.viz_` (rules are sole viz
// writer), consumes those slots from the pool of remaining role counts,
// and weighted-samples the rest into viz=0 slots.
//
// Slot inventory:
//   - influence[p][s] (owner_only_first_axis): observer sees iff
//       viz["influence"][p, s, observer] == 1. Base owner_only_first_axis
//       gives viz[p, *, p] = 1 for the owner. Rules call reveal_slot on
//       lose-influence and on successful challenge → viz[p, s, *] = 1.
//   - exchange_drawn[i] (all_hidden base): observer sees iff
//       viz["exchange_drawn"][i, observer] == 1. Rules call
//       reveal_slot_to(active_player) during exchange.
//   - deck_count[c] (all_hidden): never visible. The pool's residual
//       distribution is written here; deck_size scalar is public and
//       preserved.
template <int NPlayers>
void CoupBeliefTracker<NPlayers>::randomize_unseen(
    IGameState& state, int observer, std::mt19937_64& rng) const {
  auto* s = dynamic_cast<CoupState<NPlayers>*>(&state);
  if (!s) return;
  auto& d = s->data;
  if (observer < 0 || observer >= NPlayers) return;

  const auto& influence_viz = viz::viz_get(state, "influence");
  const auto& exchange_viz = viz::viz_get(state, "exchange_drawn");
  const int n_viewers = influence_viz.viewer_count();
  if (observer >= n_viewers) return;

  auto influence_visible = [&](int p, int sl) -> bool {
    const std::size_t base = viz::flat_offset_data_only(
        influence_viz.shape, std::vector<int>{p, sl});
    return influence_viz.data[base + static_cast<std::size_t>(observer)] != 0;
  };
  auto exchange_visible = [&](int i) -> bool {
    const std::size_t base = viz::flat_offset_data_only(
        exchange_viz.shape, std::vector<int>{i});
    return exchange_viz.data[base + static_cast<std::size_t>(observer)] != 0;
  };

  // ------------------------------------------------------------------
  // Step 1: compute `remaining[role]` — copies of each role still
  // unaccounted for from the observer's perspective. Visible influence
  // and visible exchange_drawn slots consume from the pool; everything
  // else (including the entire deck and any opp's hidden slots) must
  // come out of the residual.
  // ------------------------------------------------------------------
  std::array<int, kCharacterCount> remaining{};
  for (int c = 0; c < kCharacterCount; ++c) {
    remaining[static_cast<size_t>(c)] = kCardsPerCharacter;
  }
  auto consume = [&](CharId card) {
    if (card >= 0 && card < kCharacterCount &&
        remaining[static_cast<size_t>(card)] > 0) {
      remaining[static_cast<size_t>(card)]--;
    }
  };

  for (int p = 0; p < NPlayers; ++p) {
    for (int sl = 0; sl < 2; ++sl) {
      if (influence_visible(p, sl)) consume(d.influence[p][sl]);
    }
  }
  // Exchange in-flight count comes from the public fields `stage` +
  // `exchange_held_count` rather than from `d.exchange_drawn[i]` itself.
  // Per DEC-003, viz=0 slots hold semantically-undefined bytes (stale
  // values left over from prior ply's apply_partial_reveals). Reading
  // `d.exchange_drawn[i] >= 0` on a viz=0 slot to decide if it's
  // "occupied" violates the contract: e.g. after kExchangeReturn1
  // sets the truth slot to -1 and resets viz, the session's stored
  // value (e.g. 0 = Duke from the prior ply) makes the slot look
  // occupied to the next sim's randomize_unseen — pool inconsistent.
  const bool exchange_active =
      (d.stage == CoupStage::kExchangeReturn1 ||
       d.stage == CoupStage::kExchangeReturn2);
  int drawn_in_flight = 0;
  if (exchange_active && d.active_player >= 0 && d.active_player < NPlayers) {
    drawn_in_flight =
        static_cast<int>(d.exchange_held_count) -
        influence_count(d, d.active_player);
    if (drawn_in_flight < 0) drawn_in_flight = 0;
    if (drawn_in_flight > kExchangeDrawSlots) {
      drawn_in_flight = kExchangeDrawSlots;
    }
  }
  int visible_drawn_count = 0;
  for (int i = 0; i < kExchangeDrawSlots; ++i) {
    if (exchange_active && exchange_visible(i) && d.exchange_drawn[i] >= 0) {
      consume(d.exchange_drawn[i]);
      ++visible_drawn_count;
    }
  }

  // ------------------------------------------------------------------
  // Step 2: enumerate viz=0 slots that need filling.
  //   - influence[p][sl] where viz=0 to observer
  //   - exchange_drawn[i] where viz=0 AND the slot is occupied
  //     (-1 sentinel = empty, no need to sample)
  //   - the deck residual (pool size known from deck_size scalar)
  // ------------------------------------------------------------------
  struct Slot {
    int owner;      // -1 = deck, else player index
    int slot_idx;   // for influence: 0/1; for exchange: kXBase + i
  };
  constexpr int kXBase = 100;
  std::vector<Slot> slots;
  slots.reserve(static_cast<size_t>(NPlayers * 2 + kExchangeDrawSlots + 15));

  for (int p = 0; p < NPlayers; ++p) {
    for (int sl = 0; sl < 2; ++sl) {
      if (!influence_visible(p, sl)) {
        slots.push_back({p, sl});
      }
    }
  }
  // Hidden in-flight drawn slots: derive count from public exchange_active
  // + held_count rather than reading viz=0 d.exchange_drawn[i] (DEC-003).
  // We don't know which `i` the hidden slot is at — pick visually first
  // viz=0 slots up to the in-flight count. The deck-multiset write in
  // step 4 doesn't depend on per-i identity, and the encoder only reads
  // visible slots for active perspective via MaskedState (placeholder
  // for non-active perspectives).
  if (exchange_active && drawn_in_flight > visible_drawn_count) {
    int hidden_to_assign = drawn_in_flight - visible_drawn_count;
    for (int i = 0; i < kExchangeDrawSlots && hidden_to_assign > 0; ++i) {
      if (!exchange_visible(i)) {
        slots.push_back({d.active_player, kXBase + i});
        --hidden_to_assign;
      }
    }
  }

  const int deck_size = static_cast<int>(d.deck_size);
  // Deck is treated as a single multiset slot per copy. We don't need
  // unique slot indices since the deck content is homogeneous (only the
  // count per role matters; visual order is reconstructed client-side
  // from the action stream).
  for (int i = 0; i < deck_size; ++i) {
    slots.push_back({-1, 0});
  }

  // ------------------------------------------------------------------
  // Step 3: weighted sampling. Per-slot weight[R] = remaining[R] *
  // prior[R]. `remaining` is a hard pool constraint; prior reflects
  // tracker signals + pending-claim boost for that owner.
  // ------------------------------------------------------------------
  // Sanity check: viable joint must have remaining-sum == slot count.
  const int total_remaining =
      std::accumulate(remaining.begin(), remaining.end(), 0);
  if (total_remaining != static_cast<int>(slots.size())) {
    // No silent fallback: an inconsistent pool means upstream snapshot
    // application or rules are broken. Throw with context per CLAUDE.md.
    std::string detail;
    detail += "  deck_size=" + std::to_string(deck_size) + "\n";
    detail += "  ply=" + std::to_string(d.ply) + "\n";
    detail += "  stage=" + std::to_string(static_cast<int>(d.stage)) + "\n";
    detail += "  active=" + std::to_string(d.active_player) +
              " held=" + std::to_string(static_cast<int>(d.exchange_held_count)) + "\n";
    detail += "  drawn_in_flight=" + std::to_string(drawn_in_flight) +
              " visible_drawn=" + std::to_string(visible_drawn_count) + "\n";
    detail += "  deck_count=[";
    for (int c = 0; c < kCharacterCount; ++c) {
      if (c) detail += ",";
      detail += std::to_string(static_cast<int>(d.deck_count[c]));
    }
    detail += "]\n";
    for (int p = 0; p < NPlayers; ++p) {
      detail += "  player " + std::to_string(p) + ":";
      for (int sl = 0; sl < 2; ++sl) {
        detail += " inf[" + std::to_string(sl) + "]=";
        detail += std::to_string(static_cast<int>(d.influence[p][sl]));
        detail += "(rev=" + std::to_string(d.revealed[p][sl] ? 1 : 0);
        detail += ",viz=" + std::to_string(influence_visible(p, sl) ? 1 : 0);
        detail += ")";
      }
      detail += "\n";
    }
    for (int i = 0; i < kExchangeDrawSlots; ++i) {
      detail += "  exchange_drawn[" + std::to_string(i) + "]=";
      detail += std::to_string(static_cast<int>(d.exchange_drawn[i]));
      detail += "(viz=" + std::to_string(exchange_visible(i) ? 1 : 0);
      detail += ")\n";
    }
    throw std::logic_error(
        "CoupBeliefTracker::randomize_unseen: pool inconsistency "
        "(remaining=" + std::to_string(total_remaining) +
        ", slots=" + std::to_string(slots.size()) +
        ", observer=" + std::to_string(observer) + ").\n" + detail);
  }

  std::shuffle(slots.begin(), slots.end(), rng);

  std::array<int, kCharacterCount> deck_fill{};
  for (const Slot& slot : slots) {
    std::array<double, kCharacterCount> weights{};
    double total = 0.0;
    for (int r = 0; r < kCharacterCount; ++r) {
      const int avail = remaining[static_cast<size_t>(r)];
      if (avail == 0) { weights[r] = 0.0; continue; }
      double prior = 1.0;
      if (slot.owner >= 0 && slot.owner < NPlayers) {
        int effective_signal = signals_[slot.owner][r];
        if (!pending_challenged_ &&
            pending_claimer_ == slot.owner &&
            pending_claim_role_ == r) {
          effective_signal += 2;
        }
        prior += kSignalAlpha * static_cast<double>(effective_signal);
      }
      weights[r] = static_cast<double>(avail) * prior;
      total += weights[r];
    }

    int picked = -1;
    if (total > 0.0) {
      std::uniform_real_distribution<double> dist(0.0, total);
      const double u = dist(rng);
      double acc = 0.0;
      for (int r = 0; r < kCharacterCount; ++r) {
        acc += weights[r];
        if (u < acc) { picked = r; break; }
      }
      if (picked < 0) {
        for (int r = kCharacterCount - 1; r >= 0; --r) {
          if (weights[r] > 0.0) { picked = r; break; }
        }
      }
    } else {
      for (int r = 0; r < kCharacterCount; ++r) {
        if (remaining[static_cast<size_t>(r)] > 0) { picked = r; break; }
      }
    }
    if (picked < 0) picked = 0;

    remaining[static_cast<size_t>(picked)]--;

    if (slot.owner == -1) {
      ++deck_fill[static_cast<size_t>(picked)];
    } else if (slot.slot_idx >= kXBase) {
      const int i_draw = slot.slot_idx - kXBase;
      d.exchange_drawn[i_draw] = static_cast<CharId>(picked);
    } else {
      d.influence[slot.owner][slot.slot_idx] = static_cast<CharId>(picked);
    }
  }

  // ------------------------------------------------------------------
  // Step 4: write deck residual into the multiset count array. deck_size
  // scalar stays as-is (public).
  // ------------------------------------------------------------------
  for (int c = 0; c < kCharacterCount; ++c) {
    d.deck_count[static_cast<size_t>(c)] =
        static_cast<std::int8_t>(deck_fill[static_cast<size_t>(c)]);
  }
}

template class CoupFeatureEncoder<2>;
template class CoupFeatureEncoder<3>;
template class CoupFeatureEncoder<4>;
template class CoupBeliefTracker<2>;
template class CoupBeliefTracker<3>;
template class CoupBeliefTracker<4>;

}  // namespace board_ai::coup
