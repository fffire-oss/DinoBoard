#include "loveletter_net_adapter.h"

#include <algorithm>
#include <any>
#include <utility>
#include <vector>

#include "../../engine/core/game_interfaces.h"
#include "../../engine/core/masked_state.h"
#include "../../engine/core/viz_runtime.h"

namespace board_ai::loveletter {

template <int NPlayers>
void LoveLetterFeatureEncoder<NPlayers>::encode_public(
    const IGameState& state,
    int perspective_player,
    const IBeliefTracker* /*tracker*/,
    std::vector<float>* out) const {
  const auto* s = dynamic_cast<const LoveLetterState<NPlayers>*>(&state);
  if (!s || !out || perspective_player < 0 || perspective_player >= NPlayers) return;
  const auto& d = s->data;

  // Per-player public: alive, protected, current_player, hand_exposed,
  // discard counts by card type, discard size. All observer-visible.
  for (int pi = 0; pi < NPlayers; ++pi) {
    const int pid = (perspective_player + pi) % NPlayers;
    out->push_back(d.alive[pid] ? 1.0f : 0.0f);
    out->push_back(d.protected_flags[pid] ? 1.0f : 0.0f);
    out->push_back(d.current_player == pid ? 1.0f : 0.0f);
    out->push_back(d.hand_exposed[pid] ? 1.0f : 0.0f);

    for (int c = 1; c <= kCardTypes; ++c) {
      int count = 0;
      for (auto card : d.discard_piles[static_cast<size_t>(pid)]) {
        if (card == c) ++count;
      }
      out->push_back(static_cast<float>(count) /
                     static_cast<float>(kCardCounts[static_cast<size_t>(c)]));
    }

    out->push_back(static_cast<float>(d.discard_piles[static_cast<size_t>(pid)].size()) / 8.0f);
  }

  // Global public.
  out->push_back(static_cast<float>(d.deck.size()) / 16.0f);
  out->push_back(static_cast<float>(d.ply) / 20.0f);
  out->push_back(d.first_player == perspective_player ? 1.0f : 0.0f);

  int alive_count = 0;
  for (int p = 0; p < NPlayers; ++p) {
    if (d.alive[p]) ++alive_count;
  }
  out->push_back(static_cast<float>(alive_count) / static_cast<float>(NPlayers));

  for (int c = 1; c <= kCardTypes; ++c) {
    int count = 0;
    for (auto card : d.face_up_removed) {
      if (card == c) ++count;
    }
    out->push_back(static_cast<float>(count) /
                   static_cast<float>(kCardCounts[static_cast<size_t>(c)]));
  }
}

template <int NPlayers>
void LoveLetterFeatureEncoder<NPlayers>::encode_private(
    const IGameState& state,
    int player,
    const IBeliefTracker* /*tracker*/,
    std::vector<float>* out) const {
  const auto* s = dynamic_cast<const LoveLetterState<NPlayers>*>(&state);
  if (!s || !out || player < 0 || player >= NPlayers) return;
  const auto& d = s->data;

  // Encoder reads MaskedState directly: viz=1 slots carry truth, viz=0
  // slots carry kPlaceholderInt8 (INT8_MIN), which never equals any
  // legitimate cid in 1..8 — the one-hot naturally encodes as all-zero
  // for hidden slots without any explicit placeholder branch.
  //
  // hand[pid]: owner_only_first_axis. From perspective `player`:
  //   - pid == player        : viz=1 (truth)
  //   - other pid, no reveal : viz=0 (placeholder)
  //   - other pid, after Priest peek / Baron / King swap : viz=1 (rules
  //     called reveal_slot_to(player) on that slot — viz follows cid)
  //
  // drawn_card: all_hidden base; rules call reveal_slot_to(current_player)
  // on draw and reset_to_base on play. Visible only when player ==
  // current_player and a draw is in flight.
  for (int pi = 0; pi < NPlayers; ++pi) {
    const int pid = (player + pi) % NPlayers;
    const std::int8_t hand_card = d.hand[static_cast<size_t>(pid)];
    for (int c = 1; c <= kCardTypes; ++c) {
      out->push_back(hand_card == c ? 1.0f : 0.0f);
    }
    for (int c = 1; c <= kCardTypes; ++c) {
      const bool show_drawn = (pid == d.current_player && pid == player &&
                                d.drawn_card == c);
      out->push_back(show_drawn ? 1.0f : 0.0f);
    }
  }
}

template <int NPlayers>
void LoveLetterBeliefTracker<NPlayers>::init(
    const AnyMap& /*initial_observation*/) {
  // Tracker is perspective-agnostic and stateless. All per-perspective
  // hand/drawn knowledge lives on state.viz_ (rules-driven reveals).
  // No private fields to seed; randomize_unseen reads everything it
  // needs from the public state + observer's viz=1 slots.
}

template <int NPlayers>
void LoveLetterBeliefTracker<NPlayers>::observe_public_event(
    int /*actor*/,
    ActionId /*action*/,
    const std::vector<PublicEvent>& /*events*/) {
  // Stateless — every observation effect that affects what observer
  // can see lands on state.viz_ via rules' reveal_slot / reset_to_base
  // and on state's public fields (discard_piles / face_up_removed)
  // via wholesale public_state_applier replacement. Nothing for the
  // tracker to record.
}
// randomize_unseen produces a determinized world consistent with what
// `observer` has observed. Per §G the tracker is stateless: every fact
// the observer knows is already on state — either as a public field
// (discard_piles, face_up_removed, hand_exposed) or as a viz=1 hand /
// drawn_card slot (rules' Priest peek / Baron compare / King swap /
// drawn-card-on-own-turn).
//
// Algorithm:
//   1. Derive the unseen-card pool: full LL deck minus public discards,
//      minus face_up_removed, minus every slot the observer can see the
//      truth of (state.viz_["hand"][p, observer]==1 → consume
//      state.hand[p]; ditto drawn_card; set_aside_card is permanently
//      hidden so always in the pool).
//   2. Shuffle the pool with caller-supplied rng.
//   3. Fill set_aside_card from the pool (always).
//   4. For each viz=0 hand slot (alive players observer can't see),
//      draw the next pool card.
//   5. drawn_card: if current_player has a draw in flight (viz=0 to
//      observer), draw from pool.
//   6. Remaining pool → state.deck.
template <int NPlayers>
void LoveLetterBeliefTracker<NPlayers>::randomize_unseen(
    IGameState& state, int observer, std::mt19937_64& rng) const {
  auto* s = dynamic_cast<LoveLetterState<NPlayers>*>(&state);
  if (!s) return;
  auto& d = s->data;
  if (observer < 0 || observer >= NPlayers) return;

  const auto& hand_viz = viz::viz_get(state, "hand");
  const auto& drawn_viz = viz::viz_get(state, "drawn_card");
  const int n_viewers = hand_viz.viewer_count();
  if (observer >= n_viewers) return;

  auto hand_visible = [&](int p) -> bool {
    const std::size_t base =
        viz::flat_offset_data_only(hand_viz.shape, std::vector<int>{p});
    return hand_viz.data[base + static_cast<std::size_t>(observer)] != 0;
  };
  auto drawn_visible = [&]() -> bool {
    const std::size_t base =
        viz::flat_offset_data_only(drawn_viz.shape, std::vector<int>{});
    return drawn_viz.data[base + static_cast<std::size_t>(observer)] != 0;
  };

  std::array<int, 9> remaining{};
  for (int c = 1; c <= kCardTypes; ++c) {
    remaining[static_cast<size_t>(c)] = kCardCounts[static_cast<size_t>(c)];
  }
  auto consume = [&](std::int8_t card) {
    if (card >= 1 && card <= kCardTypes) {
      remaining[static_cast<size_t>(card)]--;
    }
  };

  for (int p = 0; p < NPlayers; ++p) {
    for (auto card : d.discard_piles[static_cast<size_t>(p)]) consume(card);
  }
  for (auto card : d.face_up_removed) consume(card);
  for (int p = 0; p < NPlayers; ++p) {
    if (!d.alive[p]) continue;
    if (hand_visible(p)) consume(d.hand[static_cast<size_t>(p)]);
  }
  if (drawn_visible() && d.drawn_card != 0) consume(d.drawn_card);

  std::vector<std::int8_t> unseen;
  for (int c = 1; c <= kCardTypes; ++c) {
    for (int i = 0; i < remaining[static_cast<size_t>(c)]; ++i) {
      unseen.push_back(static_cast<std::int8_t>(c));
    }
  }
  std::shuffle(unseen.begin(), unseen.end(), rng);

  std::size_t idx = 0;

  // set_aside_card: permanently hidden to everyone; always sample.
  if (idx < unseen.size()) {
    d.set_aside_card = unseen[idx++];
  } else {
    d.set_aside_card = 0;
  }

  // Hands: viz=1 → keep state.hand[p]; viz=0 → sample.
  for (int p = 0; p < NPlayers; ++p) {
    if (!d.alive[p]) {
      d.hand[static_cast<size_t>(p)] = 0;
      continue;
    }
    if (hand_visible(p)) continue;  // observer-known, leave alone
    if (idx < unseen.size()) {
      d.hand[static_cast<size_t>(p)] = unseen[idx++];
    } else {
      d.hand[static_cast<size_t>(p)] = 0;
    }
  }

  // drawn_card: visible to current_player after a draw (rules call
  // reveal_slot_to(drawn_card, {}, current_player)). Truth has a
  // drawn_card in flight whenever a draw has resolved and the play has
  // not yet consumed it; observer-side we check the public deck size
  // proxy via viz: rules reset_to_base("drawn_card") on play and
  // reveal_slot_to on draw, so viz=1 to observer iff observer drew.
  if (d.terminal) {
    d.drawn_card = 0;
  } else if (drawn_visible()) {
    // Observer is the current_player on their own turn — keep truth.
  } else if (d.drawn_card != 0) {
    // A draw is in flight (current_player != observer); sample.
    if (idx < unseen.size()) {
      d.drawn_card = unseen[idx++];
    } else {
      d.drawn_card = 0;
    }
  }

  d.deck.clear();
  while (idx < unseen.size()) {
    d.deck.push_back(unseen[idx++]);
  }
}

template <int NPlayers>
AnyMap LoveLetterBeliefTracker<NPlayers>::serialize() const {
  // Stateless: any two observation-equal trackers produce equal output.
  return AnyMap{};
}

template class LoveLetterFeatureEncoder<2>;
template class LoveLetterFeatureEncoder<3>;
template class LoveLetterFeatureEncoder<4>;
template class LoveLetterBeliefTracker<2>;
template class LoveLetterBeliefTracker<3>;
template class LoveLetterBeliefTracker<4>;

}  // namespace board_ai::loveletter
