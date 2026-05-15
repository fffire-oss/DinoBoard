#include "coup_net_adapter.h"

#include <algorithm>
#include <cmath>
#include <limits>
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

// Masked-state read pattern (BUG-018 / encoder contract): the encoder
// reads slots through the perspective-masked clone. viz=1 slots carry
// truth, viz=0 slots carry placeholder sentinels (kPlaceholderInt8 =
// INT8_MIN for CharId/std::int8_t fields). The placeholder never
// equals any legitimate character id (0..4), so a `== c` test on a
// placeholder is always false — char-OH naturally encodes as all-zero
// for hidden slots without any explicit placeholder branch.

// Belief-net plan §16. Encoder rewrite — one consolidated per-player
// loop (no separate "private" pass), single global block, all
// hand-craft signals removed in favor of belief-net tracker raw counts.
// The masked clone handles privacy: viz=1 → truth, viz=0 → placeholder
// (kPlaceholderInt8 = INT8_MIN); `card == c` on placeholder is always
// false → char-OH zeros out. No `is_self` gate.
template <int NPlayers>
void CoupFeatureEncoder<NPlayers>::encode_features(
    const IGameState& masked_state,
    int perspective_player,
    const IBeliefTracker* tracker,
    std::vector<float>* out) const {
  const auto* s = dynamic_cast<const CoupState<NPlayers>*>(&masked_state);
  if (!s || !out || perspective_player < 0 ||
      perspective_player >= NPlayers) {
    return;
  }
  const auto& d = s->data;
  const auto* bt =
      dynamic_cast<const CoupBeliefTracker<NPlayers>*>(tracker);

  auto clamp_norm = [](int x) -> float {
    if (x < 0) return 0.0f;
    if (x > 4) return 1.0f;
    return static_cast<float>(x) * 0.25f;
  };

  // ---- Per-player block (46 each), rotated so perspective is index 0 ----
  for (int pi = 0; pi < NPlayers; ++pi) {
    const int pid = (perspective_player + pi) % NPlayers;

    // alive(1), coins/12(1)
    out->push_back(d.alive[pid] ? 1.0f : 0.0f);
    out->push_back(static_cast<float>(d.coins[pid]) / 12.0f);

    // 2 x { revealed?(1) + char-OH(5) } = 12.
    // The masked clone makes this work uniformly: viz=1 (own unrevealed
    // + any-perspective revealed) carries truth; viz=0 (other
    // perspective unrevealed) carries placeholder → char-OH zeros.
    // `revealed?` flag is always public.
    for (int sl = 0; sl < 2; ++sl) {
      const bool rev = d.revealed[pid][sl];
      out->push_back(rev ? 1.0f : 0.0f);
      const std::int8_t card = d.influence[pid][sl];
      for (int c = 0; c < kCharacterCount; ++c) {
        out->push_back(card == c ? 1.0f : 0.0f);
      }
    }

    // role flags (4): active / target / blocker / challenger.
    out->push_back(d.active_player == pid ? 1.0f : 0.0f);
    out->push_back(d.action_target == pid ? 1.0f : 0.0f);
    out->push_back(d.blocker == pid ? 1.0f : 0.0f);
    out->push_back(d.challenger == pid ? 1.0f : 0.0f);

    // 4 x belief-net raw counts (5 each = 20).
    for (int c = 0; c < kCharacterCount; ++c) {
      out->push_back(bt ? clamp_norm(bt->pre_claim_count(pid, c)) : 0.0f);
    }
    for (int c = 0; c < kCharacterCount; ++c) {
      out->push_back(bt ? clamp_norm(bt->post_claim_count(pid, c)) : 0.0f);
    }
    for (int c = 0; c < kCharacterCount; ++c) {
      out->push_back(
          bt ? clamp_norm(bt->pre_challenge_initiated(pid, c)) : 0.0f);
    }
    for (int c = 0; c < kCharacterCount; ++c) {
      out->push_back(
          bt ? clamp_norm(bt->post_challenge_initiated(pid, c)) : 0.0f);
    }

    // last_reshuffle_kind OH (3): {None, Exchange, RevealTruthful}.
    int kind = 0;
    if (bt) kind = static_cast<int>(bt->last_reshuffle_kind(pid));
    for (int j = 0; j < 3; ++j) {
      out->push_back(kind == j ? 1.0f : 0.0f);
    }
    // last_revealed_role OH (5). -1 unless kind == RevealTruthful.
    int rev_role = bt ? bt->last_revealed_role(pid) : -1;
    for (int c = 0; c < kCharacterCount; ++c) {
      out->push_back(rev_role == c ? 1.0f : 0.0f);
    }
  }

  // ---- Global block (47 + N) ----
  // stage OH (11)
  const int stage_idx = static_cast<int>(d.stage);
  for (int i = 0; i < kStageCount; ++i) {
    out->push_back(i == stage_idx ? 1.0f : 0.0f);
  }

  // declared_action_type OH (7).
  const int action_type = action_type_index(d.declared_action);
  constexpr int kActionTypeCount = 7;
  for (int i = 0; i < kActionTypeCount; ++i) {
    out->push_back(i == action_type ? 1.0f : 0.0f);
  }

  // claimed_character OH (5).
  for (int c = 0; c < kCharacterCount; ++c) {
    out->push_back(d.claimed_character == c ? 1.0f : 0.0f);
  }

  // block_character OH (5).
  for (int c = 0; c < kCharacterCount; ++c) {
    out->push_back(d.block_character == c ? 1.0f : 0.0f);
  }

  // pending_claimer relative-to-perspective OH (N). Tracker is
  // authoritative; `randomize_unseen` does not use this field but the
  // encoder does (see plan §16 table).
  for (int pi = 0; pi < NPlayers; ++pi) {
    const int pid = (perspective_player + pi) % NPlayers;
    const bool match = bt && bt->pending_claimer() == pid;
    out->push_back(match ? 1.0f : 0.0f);
  }

  // pending_challenged (1).
  out->push_back((bt && bt->pending_challenged()) ? 1.0f : 0.0f);

  // ply / 200 (1).
  constexpr float kMaxPlies = 200.0f;
  out->push_back(static_cast<float>(d.ply) / kMaxPlies);

  // remaining[R] (5): observer-derived unseen pool. Total 3 per role
  // minus publicly revealed minus observer's own unrevealed. Other
  // players' unrevealed slots are viz=0 → placeholder, contribute
  // nothing — exactly what the belief net is estimating.
  std::array<int, kCharacterCount> revealed_count{};
  for (int p = 0; p < NPlayers; ++p) {
    for (int sl = 0; sl < kInfluencePerPlayer; ++sl) {
      if (d.revealed[p][sl]) {
        const int role = static_cast<int>(d.influence[p][sl]);
        if (role >= 0 && role < kCharacterCount) {
          ++revealed_count[role];
        }
      }
    }
  }
  std::array<int, kCharacterCount> own_unrevealed{};
  for (int sl = 0; sl < kInfluencePerPlayer; ++sl) {
    if (!d.revealed[perspective_player][sl]) {
      const int role = static_cast<int>(d.influence[perspective_player][sl]);
      if (role >= 0 && role < kCharacterCount) {
        ++own_unrevealed[role];
      }
    }
  }
  for (int r = 0; r < kCharacterCount; ++r) {
    const int rem =
        kCardsPerCharacter - revealed_count[r] - own_unrevealed[r];
    out->push_back(static_cast<float>(std::max(0, rem)));
  }

  // 2 x exchange_drawn { occupied?(1) + char-OH(5) } = 12. The masked
  // clone's viz controlled by `reveal_slot_to(active)` — only during the two
  // ExchangeReturn stages on the active player's perspective does this
  // carry truth; otherwise placeholder → both `occupied?` and OH zero.
  // `occupied = (xd != placeholder && xd >= 0)`. The first half of the
  // condition catches viz=0 (placeholder = INT8_MIN, < 0); the second
  // catches the `-1` in-bounds sentinel rules use after Return1.
  for (int i = 0; i < kExchangeDrawSlots; ++i) {
    const std::int8_t xd = d.exchange_drawn[i];
    const bool occupied = xd >= 0;
    out->push_back(occupied ? 1.0f : 0.0f);
    for (int c = 0; c < kCharacterCount; ++c) {
      out->push_back((occupied && xd == c) ? 1.0f : 0.0f);
    }
  }
}

template <int NPlayers>
void CoupBeliefTracker<NPlayers>::init(
    IGameState& /*state*/, int perspective,
    const AnyMap& /*payload*/) {
  // Walker-driven init: `viz::apply_public_snapshot` already wrote the
  // public projection (every viz=1 slot's value, plus the perspective's
  // full viz slice) into `state` before this call — including the
  // perspective's owner_only influence slots, which ride directly on the
  // wire's per-slot value pairs. The tracker has no perspective-private
  // bootstrap to seed; just remember perspective for `randomize_unseen`
  // and reset signal memory.
  perspective_player_ = perspective;
  for (auto& row : signals_) row.fill(0);
  for (auto& row : pre_claim_counts_) row.fill(0);
  for (auto& row : post_claim_counts_) row.fill(0);
  for (auto& row : pre_challenge_initiated_) row.fill(0);
  for (auto& row : post_challenge_initiated_) row.fill(0);
  last_reshuffle_kind_.fill(ReshuffleKind::kNone);
  last_revealed_role_.fill(-1);
  pending_claimer_ = -1;
  pending_claim_role_ = -1;
  pending_challenged_ = false;
}

template <int NPlayers>
void CoupBeliefTracker<NPlayers>::observe_public_event(
    int actor,
    ActionId action,
    const std::vector<PublicEvent>& events) {
  // The tracker is fed by two inputs:
  //   1. Action-level: claim openings (set pending_claimer/role) and
  //      challenge declarations (signal accrual on the challenger).
  //   2. Event-level: stage-boundary derived events emitted by
  //      `coup_events::extract_events_only`. Multi-player Allow flows
  //      collapse into a single `claim_unchallenged` / `block_unchallenged`
  //      event so signals are not double-counted across N-1 Allows.
  //
  // Stages where state.influence is rewritten in-place (challenge
  // win → reshuffle, exchange_complete → hand reshuffle) zero out the
  // affected signals because prior claim history no longer points at
  // a fixed set of cards.

  // ----- Action-level (claim opening + challenge declaration) -----
  const int claimed = claim_role_for_action(action);
  if (claimed >= 0) {
    pending_claimer_ = actor;
    pending_claim_role_ = claimed;
    pending_challenged_ = false;
    // v0.2 raw counts: claim event lands on `post_claim_counts_` for the
    // claimer (challenge-initiated is bumped on challenge events further
    // down). The post→pre promotion happens on reshuffle events.
    if (actor >= 0 && actor < NPlayers) {
      post_claim_counts_[actor][claimed] += 1;
    }
  } else if (action == kChallengeAction) {
    // Challenger implicitly signals "I may hold `pending_claim_role_`" —
    // they dare to challenge because they can rule out the claim.
    if (pending_claim_role_ >= 0 && actor >= 0 && actor < NPlayers) {
      signals_[actor][pending_claim_role_] += 1;
      post_challenge_initiated_[actor][pending_claim_role_] += 1;
    }
    pending_challenged_ = true;
  }

  // ----- Event-level -----
  for (const auto& evt : events) {
    const auto& payload = evt.second;
    if (evt.first == "card_revealed") {
      auto it_p = payload.find("player");
      auto it_r = payload.find("role");
      if (it_p == payload.end() || it_r == payload.end()) continue;
      int p = std::any_cast<int>(it_p->second);
      int r = std::any_cast<int>(it_r->second);
      if (p < 0 || p >= NPlayers) continue;
      if (r < 0 || r >= kCharacterCount) continue;
      // Revealed copy is public; any signal we'd accumulated about
      // (p, r) is now grounded — drop it so we don't double-bias the
      // pool sampler against the truth that's already on the table.
      signals_[p][r] = 0;
    } else if (evt.first == "claim_resolved_truthful") {
      auto it_c = payload.find("claimer");
      auto it_r = payload.find("role");
      if (it_c == payload.end() || it_r == payload.end()) continue;
      int c = std::any_cast<int>(it_c->second);
      int r = std::any_cast<int>(it_r->second);
      if (c < 0 || c >= NPlayers) continue;
      if (r < 0 || r >= kCharacterCount) continue;
      // Claimer truly held r and just reshuffled it back into the deck —
      // they no longer demonstrably hold r. Reset that pair; close the
      // claim cycle.
      signals_[c][r] = 0;
      promote_post_to_pre(c);
      last_reshuffle_kind_[c] = ReshuffleKind::kRevealTruthful;
      last_revealed_role_[c] = static_cast<std::int8_t>(r);
      if (pending_claimer_ == c && pending_claim_role_ == r) {
        pending_claimer_ = -1;
        pending_claim_role_ = -1;
        pending_challenged_ = false;
      }
    } else if (evt.first == "block_resolved_truthful") {
      auto it_b = payload.find("blocker");
      auto it_r = payload.find("role");
      if (it_b == payload.end() || it_r == payload.end()) continue;
      int b = std::any_cast<int>(it_b->second);
      int r = std::any_cast<int>(it_r->second);
      if (b < 0 || b >= NPlayers) continue;
      if (r < 0 || r >= kCharacterCount) continue;
      // Same as above for the block side.
      signals_[b][r] = 0;
      promote_post_to_pre(b);
      last_reshuffle_kind_[b] = ReshuffleKind::kRevealTruthful;
      last_revealed_role_[b] = static_cast<std::int8_t>(r);
      if (pending_claimer_ == b && pending_claim_role_ == r) {
        pending_claimer_ = -1;
        pending_claim_role_ = -1;
        pending_challenged_ = false;
      }
    } else if (evt.first == "claim_unchallenged") {
      auto it_c = payload.find("claimer");
      auto it_r = payload.find("role");
      if (it_c == payload.end() || it_r == payload.end()) continue;
      int c = std::any_cast<int>(it_c->second);
      int r = std::any_cast<int>(it_r->second);
      if (c < 0 || c >= NPlayers) continue;
      if (r < 0 || r >= kCharacterCount) continue;
      // Everyone Allowed — single reinforcement regardless of N.
      signals_[c][r] += 1;
      if (pending_claimer_ == c && pending_claim_role_ == r) {
        pending_claimer_ = -1;
        pending_claim_role_ = -1;
        pending_challenged_ = false;
      }
    } else if (evt.first == "block_unchallenged") {
      auto it_b = payload.find("blocker");
      auto it_r = payload.find("role");
      if (it_b == payload.end() || it_r == payload.end()) continue;
      int b = std::any_cast<int>(it_b->second);
      int r = std::any_cast<int>(it_r->second);
      if (b < 0 || b >= NPlayers) continue;
      if (r < 0 || r >= kCharacterCount) continue;
      signals_[b][r] += 1;
      if (pending_claimer_ == b && pending_claim_role_ == r) {
        pending_claimer_ = -1;
        pending_claim_role_ = -1;
        pending_challenged_ = false;
      }
    } else if (evt.first == "exchange_complete") {
      auto it_p = payload.find("player");
      if (it_p == payload.end()) continue;
      int p = std::any_cast<int>(it_p->second);
      if (p < 0 || p >= NPlayers) continue;
      // Ambassador reshuffled p's hand — prior claim signals are stale.
      signals_[p].fill(0);
      promote_post_to_pre(p);
      last_reshuffle_kind_[p] = ReshuffleKind::kExchange;
      last_revealed_role_[p] = -1;
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
  // values left over from prior ply's apply_public_snapshot). Reading
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
  // visible slots for active perspective via the masked clone
  // (placeholder for non-active perspectives).
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
    throw std::logic_error(
        "CoupBeliefTracker::randomize_unseen: pool inconsistency "
        "(remaining=" + std::to_string(total_remaining) +
        ", slots=" + std::to_string(slots.size()) +
        ", observer=" + std::to_string(observer) + ")");
  }

  std::shuffle(slots.begin(), slots.end(), rng);

  // Sampling policy (Plan §1.1):
  //   weight[R] = remaining[R] × prior[R]
  //   - Belief net active (pi_valid_ && pi_observer_ == observer): for
  //     opponent slots use `prior = pi_[player_to_opp(observer,owner,N)][R]`;
  //     for deck slots use `prior = 1.0` (uniform — we have no per-slot
  //     posterior for the deck).
  //   - Belief net not installed / not run for this observer: fall back
  //     to the legacy `signals_` + pending-claim heuristic (kept so
  //     selfplay still works without belief.onnx in early training).
  // Plan §1.2.2: `slot.owner == observer` cannot happen — observer's own
  // slots are always viz=1 to themselves. We assert defensively rather
  // than fall back, per CLAUDE.md "No Fallbacks".
  const bool use_belief_net = pi_valid_ && pi_observer_ == observer;
  auto opp_index = [observer](int p) {
    return (p - observer - 1 + NPlayers) % NPlayers;
  };

  std::array<int, kCharacterCount> deck_fill{};
  for (const Slot& slot : slots) {
    if (slot.owner == observer) {
      throw std::logic_error(
          "CoupBeliefTracker::randomize_unseen: hidden slot owned by "
          "observer (viz mis-tagged) — observer=" +
          std::to_string(observer));
    }
    std::array<double, kCharacterCount> weights{};
    double total = 0.0;
    for (int r = 0; r < kCharacterCount; ++r) {
      const int avail = remaining[static_cast<size_t>(r)];
      if (avail == 0) { weights[r] = 0.0; continue; }
      double prior = 1.0;
      if (use_belief_net) {
        if (slot.owner >= 0 && slot.owner < NPlayers) {
          // opp slot: pi from belief net (already a per-row prob).
          const int oi = opp_index(slot.owner);
          prior = static_cast<double>(pi_[oi][r]);
        }  // deck slot: prior stays 1.0
      } else if (slot.owner >= 0 && slot.owner < NPlayers) {
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

template <int NPlayers>
void CoupBeliefTracker<NPlayers>::prepare_for_root(
    const IGameState& root_state,
    int root_player,
    const IBeliefFeatureExtractor* extractor,
    const IBeliefEvaluator* evaluator) {
  // Plan §2.1: extract per-perspective features, run the belief net,
  // softmax(/T) per opp row, cache pi_ in this session-shared tracker.
  // sim_tracker (clone) inherits pi_ by value — each sim reads it
  // during `randomize_unseen` without re-evaluating the network.
  pi_valid_ = false;
  pi_observer_ = -1;
  for (auto& row : pi_) row.fill(0.0f);

  if (!extractor || !evaluator) return;
  if (root_player < 0 || root_player >= NPlayers) return;
  if (NPlayers <= 1) return;

  std::vector<float> features;
  if (!extractor->extract_from_state(root_state, root_player, this,
                                     &features)) {
    return;
  }
  std::vector<float> logits;
  if (!evaluator->evaluate(features, &logits)) return;
  const int kLogitCount = (NPlayers - 1) * kCharacterCount;
  if (static_cast<int>(logits.size()) != kLogitCount) return;

  // Per-row softmax along role axis (Plan §1.2). Subtract row max for
  // numerical stability.
  const float inv_T = 1.0f / kBeliefTemperature;
  for (int i = 0; i < NPlayers - 1; ++i) {
    float row_max = -std::numeric_limits<float>::infinity();
    for (int r = 0; r < kCharacterCount; ++r) {
      const float z = logits[i * kCharacterCount + r] * inv_T;
      if (z > row_max) row_max = z;
    }
    float row_sum = 0.0f;
    std::array<float, kCharacterCount> ez{};
    for (int r = 0; r < kCharacterCount; ++r) {
      const float z = logits[i * kCharacterCount + r] * inv_T;
      ez[r] = std::exp(z - row_max);
      row_sum += ez[r];
    }
    if (row_sum <= 0.0f || !std::isfinite(row_sum)) return;
    for (int r = 0; r < kCharacterCount; ++r) {
      pi_[i][r] = ez[r] / row_sum;
    }
  }
  pi_valid_ = true;
  pi_observer_ = root_player;
}

template <int NPlayers>
AnyMap CoupBeliefTracker<NPlayers>::serialize() const {
  // Canonical, perspective-agnostic dump used by sim_tracker /
  // perspective-invariance tests. Tracker content is a public-event
  // derivation, so two trackers fed the same observation stream from
  // different seats must produce equal output.
  AnyMap out;
  auto flatten = [&](const auto& mat) {
    std::vector<int> flat;
    flat.reserve(static_cast<size_t>(NPlayers * kCharacterCount));
    for (int p = 0; p < NPlayers; ++p) {
      for (int c = 0; c < kCharacterCount; ++c) {
        flat.push_back(mat[p][c]);
      }
    }
    return flat;
  };
  out["signals"] = flatten(signals_);
  out["pre_claim_counts"] = flatten(pre_claim_counts_);
  out["post_claim_counts"] = flatten(post_claim_counts_);
  out["pre_challenge_initiated"] = flatten(pre_challenge_initiated_);
  out["post_challenge_initiated"] = flatten(post_challenge_initiated_);
  std::vector<int> kinds;
  kinds.reserve(NPlayers);
  for (int p = 0; p < NPlayers; ++p) {
    kinds.push_back(static_cast<int>(last_reshuffle_kind_[p]));
  }
  out["last_reshuffle_kind"] = kinds;
  std::vector<int> roles;
  roles.reserve(NPlayers);
  for (int p = 0; p < NPlayers; ++p) {
    roles.push_back(static_cast<int>(last_revealed_role_[p]));
  }
  out["last_revealed_role"] = roles;
  out["pending_claimer"] = pending_claimer_;
  out["pending_claim_role"] = pending_claim_role_;
  out["pending_challenged"] = pending_challenged_;
  return out;
}

// CoupBeliefFeatureExtractor — Plan §1.3.
//
// Reads the perspective-masked state + CoupBeliefTracker and emits a
// flat feature vector in the layout documented in coup_net_adapter.h.
// All inputs are public derivations; viz=0 slots arrive as
// kPlaceholderInt8 (never compared).
template <int NPlayers>
void CoupBeliefFeatureExtractor<NPlayers>::extract(
    const IGameState& masked_state,
    int perspective_player,
    const IBeliefTracker* tracker,
    std::vector<float>* out) const {
  const auto* s = dynamic_cast<const CoupState<NPlayers>*>(&masked_state);
  if (!s || !out || perspective_player < 0 ||
      perspective_player >= NPlayers) {
    return;
  }
  const auto& d = s->data;
  const auto* bt =
      dynamic_cast<const CoupBeliefTracker<NPlayers>*>(tracker);
  out->clear();
  out->reserve(static_cast<size_t>(kFeatureDim));

  // --- Global block (6) ---
  // remaining[R]: total(3) − publicly revealed R − observer's own
  //              unrevealed R slots. All other slots are viz=0 to
  //              observer and contribute nothing (would-be R cards in
  //              opps' hands are exactly what the belief net is
  //              estimating).
  std::array<int, kCharacterCount> revealed_count{};
  for (int p = 0; p < NPlayers; ++p) {
    for (int sl = 0; sl < kInfluencePerPlayer; ++sl) {
      if (d.revealed[p][sl]) {
        const int role = static_cast<int>(d.influence[p][sl]);
        if (role >= 0 && role < kCharacterCount) {
          ++revealed_count[role];
        }
      }
    }
  }
  std::array<int, kCharacterCount> own_unrevealed{};
  for (int sl = 0; sl < kInfluencePerPlayer; ++sl) {
    if (!d.revealed[perspective_player][sl]) {
      const int role = static_cast<int>(d.influence[perspective_player][sl]);
      if (role >= 0 && role < kCharacterCount) {
        ++own_unrevealed[role];
      }
    }
  }
  for (int r = 0; r < kCharacterCount; ++r) {
    const int remaining =
        kCardsPerCharacter - revealed_count[r] - own_unrevealed[r];
    out->push_back(static_cast<float>(std::max(0, remaining)));
  }
  out->push_back(static_cast<float>(d.ply) / 200.0f);

  // --- Per-opp block (28 × (N-1)) ---
  // Walk i ∈ [0, N-2]; opp = (perspective + 1 + i) mod N. Order matches
  // the belief net's output rows (Plan §1.2.1 opp_to_player).
  auto clamp_norm = [](int x) -> float {
    if (x < 0) return 0.0f;
    if (x > 4) return 1.0f;
    return static_cast<float>(x) * 0.25f;  // /4
  };
  for (int i = 0; i < NPlayers - 1; ++i) {
    const int opp = (perspective_player + 1 + i) % NPlayers;

    // pre_claim_counts (5)
    for (int r = 0; r < kCharacterCount; ++r) {
      out->push_back(bt ? clamp_norm(bt->pre_claim_count(opp, r)) : 0.0f);
    }
    // post_claim_counts (5)
    for (int r = 0; r < kCharacterCount; ++r) {
      out->push_back(bt ? clamp_norm(bt->post_claim_count(opp, r)) : 0.0f);
    }
    // pre_challenge_initiated (5)
    for (int r = 0; r < kCharacterCount; ++r) {
      out->push_back(
          bt ? clamp_norm(bt->pre_challenge_initiated(opp, r)) : 0.0f);
    }
    // post_challenge_initiated (5)
    for (int r = 0; r < kCharacterCount; ++r) {
      out->push_back(
          bt ? clamp_norm(bt->post_challenge_initiated(opp, r)) : 0.0f);
    }
    // last_reshuffle_kind one-hot (3): {None, Exchange, RevealTruthful}
    int kind = 0;
    if (bt) {
      const auto k = bt->last_reshuffle_kind(opp);
      kind = static_cast<int>(k);
    }
    for (int j = 0; j < 3; ++j) {
      out->push_back(kind == j ? 1.0f : 0.0f);
    }
    // last_revealed_role one-hot (5). Only valid when kind ==
    // RevealTruthful; otherwise tracker stores -1 → all zero.
    int rev_role = bt ? bt->last_revealed_role(opp) : -1;
    for (int r = 0; r < kCharacterCount; ++r) {
      out->push_back(rev_role == r ? 1.0f : 0.0f);
    }
  }
}

// CoupBeliefLabelExtractor — Plan §2.2.
//
// Reads truth CoupState and emits the per-observer label tuple. Runs
// only on the GT runner (selfplay's emit path); never reachable from
// the AI session.
template <int NPlayers>
void CoupBeliefLabelExtractor<NPlayers>::extract(
    const IGameState& truth_state,
    int observer,
    std::vector<std::vector<int>>* out_hand_counts,
    std::vector<int>* out_remaining,
    std::vector<int>* out_alive) const {
  const auto& s = board_ai::checked_cast<CoupState<NPlayers>>(truth_state);
  const auto& d = s.data;

  out_hand_counts->assign(NPlayers - 1,
                          std::vector<int>(kCharacterCount, 0));
  out_remaining->assign(kCharacterCount, 0);
  out_alive->assign(NPlayers - 1, 0);

  // Public reveals across every seat.
  std::array<int, kCharacterCount> revealed_count{};
  for (int p = 0; p < NPlayers; ++p) {
    for (int sl = 0; sl < kInfluencePerPlayer; ++sl) {
      if (d.revealed[p][sl]) {
        const int role = static_cast<int>(d.influence[p][sl]);
        if (role >= 0 && role < kCharacterCount) {
          ++revealed_count[role];
        }
      }
    }
  }
  // Observer's own unrevealed influence.
  std::array<int, kCharacterCount> own_unrevealed{};
  for (int sl = 0; sl < kInfluencePerPlayer; ++sl) {
    if (!d.revealed[observer][sl]) {
      const int role = static_cast<int>(d.influence[observer][sl]);
      if (role >= 0 && role < kCharacterCount) {
        ++own_unrevealed[role];
      }
    }
  }
  for (int r = 0; r < kCharacterCount; ++r) {
    const int rem =
        kCardsPerCharacter - revealed_count[r] - own_unrevealed[r];
    (*out_remaining)[r] = std::max(0, rem);
  }

  // Per-opp hand_counts + alive flag. opp i = (observer + 1 + i) % N.
  for (int i = 0; i < NPlayers - 1; ++i) {
    const int opp = (observer + 1 + i) % NPlayers;
    int alive_slots = 0;
    for (int sl = 0; sl < kInfluencePerPlayer; ++sl) {
      if (!d.revealed[opp][sl]) {
        ++alive_slots;
        const int role = static_cast<int>(d.influence[opp][sl]);
        if (role >= 0 && role < kCharacterCount) {
          ++(*out_hand_counts)[i][role];
        }
      }
    }
    (*out_alive)[i] = alive_slots > 0 ? 1 : 0;
  }
}

template class CoupFeatureEncoder<2>;
template class CoupFeatureEncoder<3>;
template class CoupFeatureEncoder<4>;
template class CoupBeliefTracker<2>;
template class CoupBeliefTracker<3>;
template class CoupBeliefTracker<4>;
template class CoupBeliefFeatureExtractor<2>;
template class CoupBeliefFeatureExtractor<3>;
template class CoupBeliefFeatureExtractor<4>;
template class CoupBeliefLabelExtractor<2>;
template class CoupBeliefLabelExtractor<3>;
template class CoupBeliefLabelExtractor<4>;

}  // namespace board_ai::coup
