#pragma once

#include <vector>

#include "game_interfaces.h"
#include "masked_state.h"

namespace board_ai {

class IBeliefTracker;


// Feature encoder split into public / private halves, both reading a
// MaskedState (golden standard §2.5 / I13).
//
// Input surface = MaskedState + tracker (ALGORITHM_OVERVIEW §6). The
// MaskedState carries every state field perspective can legally see;
// the optional `tracker` carries public derived statistics the network
// wants but the rules engine doesn't (e.g. claim history, multiset
// summaries). Games that don't need tracker-sourced features ignore
// the parameter; the tracker pointer may be null when no tracker is
// registered for the game (TicTacToe / Quoridor / Azul today).
//
// `encode` materializes a MaskedState once via `make_masked_state(state,
// schema, perspective)` and feeds the same masked clone to both halves.
// Encoders read the masked state by field name; any slot whose runtime
// viz hides it from `perspective` shows up as `kPlaceholder*` and the
// encoder branches on the placeholder value, not on a viz query. There
// is no path by which an encoder can observe another perspective's
// private truth — the placeholder write is the structural barrier.
class IFeatureEncoder {
 public:
  virtual ~IFeatureEncoder() = default;
  virtual int action_space() const = 0;
  virtual int feature_dim() const = 0;

  // Size of the public half. `feature_dim() == public_feature_dim() +
  // private_feature_dim()`.
  virtual int public_feature_dim() const = 0;

  // Size of one player's private half (same for all players).
  virtual int private_feature_dim() const = 0;

  // Encode fields visible to all players, reading from a MaskedState.
  // May use `perspective_player` for perspective-relative ordering
  // (e.g. "is_self" flag on each player slot). Slots hidden from
  // `perspective_player` arrive as kPlaceholder* — encoder must branch
  // on placeholder, never query viz.
  //
  // `tracker` is the perspective's belief tracker (or null if the game
  // didn't register one). Encoders may read public-derived statistics
  // off it; perspective-private knowledge does NOT live on the tracker
  // (it lives on `state.viz_`), so tracker reads here are structurally
  // public.
  virtual void encode_public(
      const MaskedState& state,
      int perspective_player,
      const IBeliefTracker* tracker,
      std::vector<float>* out) const = 0;

  // Encode the perspective's own private fields. The MaskedState was
  // built for `player`, so own-private slots are real and other
  // perspectives' private slots are kPlaceholder — reading a non-self
  // private slot here is structurally a placeholder read, not a leak.
  // `tracker` may be null when no tracker is registered.
  virtual void encode_private(
      const MaskedState& state,
      int player,
      const IBeliefTracker* tracker,
      std::vector<float>* out) const = 0;

  // Composes public + perspective's private + legal mask into the flat
  // feature vector consumed by the network. Materializes MaskedState
  // once and shares it with both halves. Games override
  // encode_public / encode_private, not this.
  bool encode(
      const IGameState& state,
      int perspective_player,
      const IBeliefTracker* tracker,
      const std::vector<ActionId>& legal_actions,
      std::vector<float>* features,
      std::vector<float>* legal_mask) const {
    auto masked = make_masked_state(state, state.schema_ref(),
                                    perspective_player);
    features->clear();
    features->reserve(static_cast<size_t>(feature_dim()));
    encode_public(*masked, perspective_player, tracker, features);
    encode_private(*masked, perspective_player, tracker, features);
    fill_legal_mask_impl(legal_actions, legal_mask);
    return static_cast<int>(features->size()) == feature_dim();
  }

  // Caller already has a MaskedState (e.g. MCTS descent that masked
  // once for hashing). Skips the second clone+mask.
  bool encode_with_masked(
      const MaskedState& masked,
      int perspective_player,
      const IBeliefTracker* tracker,
      const std::vector<ActionId>& legal_actions,
      std::vector<float>* features,
      std::vector<float>* legal_mask) const {
    features->clear();
    features->reserve(static_cast<size_t>(feature_dim()));
    encode_public(masked, perspective_player, tracker, features);
    encode_private(masked, perspective_player, tracker, features);
    fill_legal_mask_impl(legal_actions, legal_mask);
    return static_cast<int>(features->size()) == feature_dim();
  }

 private:
  void fill_legal_mask_impl(
      const std::vector<ActionId>& legal_actions,
      std::vector<float>* legal_mask) const {
    const int a_space = action_space();
    legal_mask->assign(static_cast<size_t>(a_space), 0.0f);
    for (ActionId a : legal_actions) {
      if (a >= 0 && a < a_space) {
        (*legal_mask)[static_cast<size_t>(a)] = 1.0f;
      }
    }
  }
};

inline void fill_legal_mask(
    int action_space,
    const std::vector<ActionId>& legal_actions,
    std::vector<float>* legal_mask) {
  legal_mask->assign(static_cast<size_t>(action_space), 0.0f);
  for (ActionId a : legal_actions) {
    if (a >= 0 && a < action_space) {
      (*legal_mask)[static_cast<size_t>(a)] = 1.0f;
    }
  }
}

}  // namespace board_ai
