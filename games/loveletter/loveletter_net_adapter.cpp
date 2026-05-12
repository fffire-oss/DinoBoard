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
    const std::vector<PublicEvent>& /*pre_events*/,
    const std::vector<PublicEvent>& /*post_events*/) {
  // Stateless — every observation effect that affects what observer
  // can see lands on state.viz_ via rules' reveal_slot / reset_to_base
  // and on state's public fields (discard_piles / face_up_removed)
  // via wholesale public_state_applier replacement. Nothing for the
  // tracker to record.
}

#if 0
template <int NPlayers>
void LoveLetterBeliefTracker<NPlayers>::observe_public_event_legacy(
    int actor, ActionId action,
    const std::vector<PublicEvent>& pre_events,
    const std::vector<PublicEvent>& post_events) {
  const PlayedCardInfo played = decode_played(action);
  const std::int8_t card = played.card;
  const int target = played.target;

  const bool i_am_actor = (actor == perspective_player_);
  const bool i_am_target = (target == perspective_player_);

  // Determine actor's pre-swap hand value (needed for King logic).
  // If actor is perspective, we already know (own_hand_ / own_drawn_card_).
  // If actor is not perspective, the event stream's pre-action
  // hand_override(actor) carries it.
  std::int8_t actor_pre_hand = 0;
  std::int8_t actor_pre_drawn = 0;
  if (i_am_actor) {
    actor_pre_hand = own_hand_;
    actor_pre_drawn = own_drawn_card_;
  } else {
    actor_pre_hand = extract_hand_override(pre_events, actor);
    actor_pre_drawn = extract_drawn_override(pre_events);
    // drawn_override pre-events carry the actor's drawn_card (only emitted
    // by extract_events when actor != perspective).
  }

  // --- Update perspective's own hand/drawn_card based on what they played
  // or received. ---
  if (i_am_actor) {
    if (card == own_drawn_card_) {
      // Played the drawn card; hand stays.
      own_drawn_card_ = 0;
    } else if (card == own_hand_) {
      // Played the hand card; drawn_card becomes the new hand.
      own_hand_ = own_drawn_card_;
      own_drawn_card_ = 0;
    }
    // For King: after the swap, perspective's hand equals the target's
    // pre-swap "other" card.
    if (card == kKing && target >= 0 && target < NPlayers) {
      const std::int8_t target_pre =
          i_am_target ? own_hand_  // already-updated (shouldn't happen: self-King)
                      : extract_hand_override(pre_events, target);
      if (target_pre != 0) {
        own_hand_ = target_pre;
      }
    }
  } else if (i_am_target) {
    // I am the target of someone else's action.
    if (card == kKing) {
      // My hand goes to actor; I receive actor's old "other" card.
      // Actor's other card = actor_pre_hand (they played King from
      // hand-or-drawn; the non-King one is actor's other).
      if (actor_pre_hand == kKing) {
        // Actor played King from hand; other is drawn_card.
        own_hand_ = actor_pre_drawn;
      } else {
        own_hand_ = actor_pre_hand;
      }
    } else if (card == kPrince) {
      // I get a new hand. The new hand is emitted via post_event
      // hand_override(target=perspective, new_card).
      for (const auto& ev : post_events) {
        if (ev.first == "hand_override") {
          auto pit = ev.second.find("player");
          auto cit = ev.second.find("card");
          if (pit != ev.second.end() && cit != ev.second.end() &&
              std::any_cast<int>(pit->second) == perspective_player_) {
            own_hand_ = static_cast<std::int8_t>(std::any_cast<int>(cit->second));
            break;
          }
        }
      }
    }
    // Priest/Baron/Guard on perspective reveal own_hand but don't change it.
  }

  // Post-event: advance_turn draws for the new current_player. If that's
  // perspective, drawn_override carries perspective's new drawn_card.
  for (const auto& ev : post_events) {
    if (ev.first == "drawn_override") {
      auto cit = ev.second.find("card");
      if (cit != ev.second.end()) {
        own_drawn_card_ = static_cast<std::int8_t>(std::any_cast<int>(cit->second));
      }
    }
  }

  // Post-event: self-target Prince emits hand_override(perspective, Z) but
  // the i_am_actor branch above doesn't handle Prince's hand-replacement
  // (the Prince-target updater is inside the i_am_target else-if). Without
  // this unconditional sweep, own_hand_ stays at perspective's pre-Prince
  // value, causing randomize_unseen's consume() to decrement the wrong
  // card and produce an infeasible world (duplicate Princess etc.).
  //
  // Safe to apply unconditionally — hand_override for perspective is only
  // emitted when truth says perspective's hand changed, which is exactly
  // when own_hand_ must follow.
  for (const auto& ev : post_events) {
    if (ev.first == "hand_override") {
      auto pit = ev.second.find("player");
      auto cit = ev.second.find("card");
      if (pit != ev.second.end() && cit != ev.second.end() &&
          std::any_cast<int>(pit->second) == perspective_player_) {
        own_hand_ = static_cast<std::int8_t>(std::any_cast<int>(cit->second));
        break;
      }
    }
  }

  // --- Update knowledge of other players' hands ---
  // When someone plays a card that we tracked, clear the tracked value
  // (they used it up).
  if (!i_am_actor && actor >= 0 && actor < Cfg::kPlayers) {
    if (known_hand_[actor] != 0 && known_hand_[actor] == card) {
      known_hand_[actor] = 0;
    }
  }

  // When actor self-targets (forced fallback when everyone else is
  // Handmaid-protected), perspective gains no opp-hand knowledge from
  // Priest/Baron/King — those branches assume target != actor when they
  // write known_hand_[target]/[actor]. Without per-case skips, Baron's
  // `i_am_target` branch would write `actor_pre_hand` into known_hand_[actor],
  // leaking own_hand into a tracker slot reserved for opp facts. Prince
  // self-target is different: target redraws regardless, so the clear of
  // known_hand_[target] must run even when target==actor.
  const bool self_target = (target == actor);

  // Card-specific knowledge updates.
  switch (card) {
    case kPriest:
      if (self_target) break;
      // Actor sees target's hand. Perspective learns it only if perspective
      // is the actor.
      if (i_am_actor && target >= 0 && target < Cfg::kPlayers) {
        const std::int8_t target_hand = extract_hand_override(pre_events, target);
        if (target_hand != 0) {
          known_hand_[target] = target_hand;
        }
      }
      break;

    case kBaron:
      if (self_target) break;
      // Both players compare; survivor known to the loser (loser is dead
      // anyway). In effect: if perspective is actor/target, learn the
      // other's card.
      if (target >= 0 && target < Cfg::kPlayers) {
        if (i_am_actor) {
          const std::int8_t target_hand = extract_hand_override(pre_events, target);
          if (target_hand != 0) known_hand_[target] = target_hand;
        }
        if (i_am_target) {
          // actor_pre_hand/drawn: the non-Baron one is actor's other card.
          // But Baron reveal compares hands, not drawn. actor plays Baron
          // from hand or drawn; the OTHER is the comparison card.
          // Perspective-as-target learns the card that was compared.
          std::int8_t other = 0;
          if (actor_pre_hand == kBaron) other = actor_pre_drawn;
          else if (actor_pre_drawn == kBaron) other = actor_pre_hand;
          else other = actor_pre_hand;  // fallback
          if (other != 0) known_hand_[actor] = other;
        }
      }
      break;

    case kKing:
      if (self_target) break;
      // After swap: actor now holds what target had; target holds what
      // actor had. Tracker records what perspective can deduce.
      if (target >= 0 && target < Cfg::kPlayers) {
        if (i_am_actor) {
          // Perspective (actor) gave target their pre-swap "other" card.
          const std::int8_t my_other =
              (actor_pre_hand == kKing) ? actor_pre_drawn : actor_pre_hand;
          known_hand_[target] = my_other;
        } else if (i_am_target) {
          // Perspective (target) gave actor their pre-target hand; they
          // now hold actor's "other" — but that's our own_hand, already
          // handled above. We don't need to record our own hand in
          // known_hand_.
          // Record actor's new hand (which was our old hand):
          // We need to know what our hand was BEFORE this King targeted us.
          // At this point own_hand_ has been updated to actor's old other.
          // Actor's new hand = perspective's pre-target hand. But by the
          // time we reach here, own_hand_ is the new value, not the old.
          // Workaround: we computed actor_pre_hand earlier (before own_hand_
          // mutation). No — actor_pre_hand is the actor's pre-swap hand.
          // The actor's new hand (what target had) = we don't track that
          // in known_hand_ (it's perspective's knowledge of opp, and
          // perspective LOST their card).
          // Actually: actor now has OUR old hand. Our old hand = the
          // own_hand_ value just before we applied the King swap update.
          // We lost that info — let's just clear known_hand_[actor].
          known_hand_[actor] = 0;
        } else {
          // Neither actor nor target — I saw the swap happen but don't
          // know either hand unless I previously tracked them.
          // Propagate: my tracked value for actor moves to target; my
          // tracked value for target moves to actor. Only if neither is
          // King (the known King was consumed).
          const std::int8_t ka = known_hand_[actor];
          const std::int8_t kt = known_hand_[target];
          known_hand_[actor] = (kt != 0 && kt != kKing) ? kt : 0;
          known_hand_[target] = (ka != 0 && ka != kKing) ? ka : 0;
        }
      }
      break;

    case kPrince:
      // Target discards + redraws. Their hand is fresh.
      if (target >= 0 && target < Cfg::kPlayers && target != perspective_player_) {
        known_hand_[target] = 0;
      }
      break;

    default:
      break;
  }

  // Death cleanup: in Love Letter, death is deterministic from Guard hits,
  // Baron loses, Princess plays. The tracker can infer deaths from the
  // public record (actor == target of successful Guard, etc.), but easier
  // to just drop known_hand for anyone whose play-sequence suggests death.
  // For correctness we infer "dead this ply" from the chain of play:
  //   - Princess played by actor → actor dies
  //   - Guard with matching guess → target dies (guess outcome is public)
  //   - Baron → one of actor/target dies (we can deduce from hand values)
  //
  // This mirrors observe_action's `for p: if !da.alive[p] clear known_hand_`
  // but derived from tracker state.
  if (card == kPrincess) {
    alive_tracked_[actor] = false;
    known_hand_[actor] = 0;
  } else if (card == kGuard && target >= 0 && target < Cfg::kPlayers) {
    // Guess is encoded in low 3 bits of the Guard action.
    const int guess = ((action - kGuardOffset) % 7) + 2;
    const std::int8_t target_hand = extract_hand_override(pre_events, target);
    if (target_hand != 0 && guess == target_hand) {
      alive_tracked_[target] = false;
      known_hand_[target] = 0;
    }
  } else if (card == kBaron && target >= 0 && target < Cfg::kPlayers) {
    const std::int8_t target_hand = extract_hand_override(pre_events, target);
    const std::int8_t actor_other_hand =
        (actor_pre_hand == kBaron) ? actor_pre_drawn : actor_pre_hand;
    if (target_hand != 0 && actor_other_hand != 0) {
      if (actor_other_hand > target_hand) {
        alive_tracked_[target] = false;
        known_hand_[target] = 0;
      } else if (target_hand > actor_other_hand) {
        alive_tracked_[actor] = false;
        known_hand_[actor] = 0;
        if (i_am_actor) {
          own_hand_ = 0;
          own_drawn_card_ = 0;
        }
      }
    }
  } else if (card == kPrince && target >= 0 && target < Cfg::kPlayers) {
    // Prince on someone holding Princess eliminates them.
    // We can detect this if target_hand == Princess in pre_events.
    if (target == perspective_player_) {
      // Perspective's own hand — was it Princess? own_hand_ was checked
      // before the swap above (line before post_events). We can compare
      // against the pre-action own_hand value. But that's tricky; skip
      // for now (alive tracking is best-effort).
    } else {
      const std::int8_t target_hand = extract_hand_override(pre_events, target);
      if (target_hand == kPrincess) {
        alive_tracked_[target] = false;
        known_hand_[target] = 0;
      }
    }
  }
}
#endif  // legacy observe_public_event

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
