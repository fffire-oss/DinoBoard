#pragma once

#include <vector>

#include "game_interfaces.h"
#include "masked_state.h"

namespace board_ai {

class IBeliefTracker;


// Feature encoder reading a perspective-masked state (golden standard
// §2.5 / I13).
//
// Input surface = a state with hidden slots overwritten by kPlaceholder*
// + tracker (ALGORITHM_OVERVIEW §6). The masked state carries every
// state field the perspective can legally see; the optional `tracker`
// carries public derived statistics the network wants but the rules
// engine doesn't (e.g. claim history, multiset summaries). Games that
// don't need tracker-sourced features ignore the parameter; the tracker
// pointer may be null when no tracker is registered for the game.
//
// `encode` materializes the masked state once via `make_masked_state(
// state, schema, perspective)` and feeds it to the encoder. Encoders
// read the masked state by field name; any slot whose runtime viz
// hides it from `perspective_player` shows up as `kPlaceholder*` and
// the encoder branches on the placeholder value, not on a viz query.
// The signature is `const IGameState&` because there is no distinct
// MaskedState type — the placeholder write is the structural barrier,
// enforced by `make_masked_state` always being the producer of states
// fed to `encode_features` (and by `test_encoder_respects_hash_scope`).
class IFeatureEncoder {
 public:
  virtual ~IFeatureEncoder() = default;
  virtual int action_space() const = 0;
  virtual int feature_dim() const = 0;

  // Encode the full feature vector for `perspective_player`, reading
  // from a perspective-masked state. Slots hidden from the perspective
  // arrive as kPlaceholder* — encoder must branch on placeholder, never
  // query viz. `tracker` is the perspective's belief tracker (or null
  // if the game didn't register one); tracker reads are public-derived
  // statistics only, perspective-private knowledge does not live there.
  virtual void encode_features(
      const IGameState& masked_state,
      int perspective_player,
      const IBeliefTracker* tracker,
      std::vector<float>* out) const = 0;

  // Composes the perspective's features + legal mask into the flat
  // feature vector consumed by the network. Materializes the masked
  // state once. Games override `encode_features`, not this.
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
    encode_features(*masked, perspective_player, tracker, features);
    fill_legal_mask_impl(legal_actions, legal_mask);
    return static_cast<int>(features->size()) == feature_dim();
  }

  // Caller already has a masked state (e.g. MCTS descent that masked
  // once for hashing). Skips the second clone+mask.
  bool encode_with_masked(
      const IGameState& masked_state,
      int perspective_player,
      const IBeliefTracker* tracker,
      const std::vector<ActionId>& legal_actions,
      std::vector<float>* features,
      std::vector<float>* legal_mask) const {
    features->clear();
    features->reserve(static_cast<size_t>(feature_dim()));
    encode_features(masked_state, perspective_player, tracker, features);
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
