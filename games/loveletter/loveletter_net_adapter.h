#pragma once

#include <array>
#include <memory>
#include <random>
#include <vector>

#include "../../engine/core/belief_tracker.h"
#include "../../engine/core/feature_encoder.h"
#include "../../engine/core/game_interfaces.h"
#include "loveletter_state.h"

namespace board_ai::loveletter {

template <int NPlayers>
class LoveLetterFeatureEncoder final : public IFeatureEncoder {
 public:
  using Cfg = LoveLetterConfig<NPlayers>;
  LoveLetterFeatureEncoder() = default;
  int action_space() const override { return kActionSpace; }
  int feature_dim() const override { return Cfg::kFeatureDim; }
  int public_feature_dim() const override { return Cfg::kPublicFeatureDim; }
  int private_feature_dim() const override { return Cfg::kPrivateFeatureDim; }

  void encode_public(
      const IGameState& state,
      int perspective_player,
      const IBeliefTracker* tracker,
      std::vector<float>* out) const override;

  void encode_private(
      const IGameState& state,
      int player,
      const IBeliefTracker* tracker,
      std::vector<float>* out) const override;
};

// LL belief tracker is perspective-agnostic and stateless after §G.
// All per-perspective knowledge lives in state.viz_["hand"] /
// state.viz_["drawn_card"] (rules are the sole writer, see
// loveletter_rules.cpp). The tracker exists only to provide
// `randomize_unseen`, which fills viz=0 slots from the public deck-
// multiset implied by state's discard_piles + face_up_removed +
// observer's viz=1 hand/drawn cards. There are no private fields.
template <int NPlayers>
class LoveLetterBeliefTracker final : public IBeliefTracker {
 public:
  using Cfg = LoveLetterConfig<NPlayers>;

  void init(const AnyMap& initial_observation) override;
  void observe_public_event(
      int actor,
      ActionId action,
      const std::vector<PublicEvent>& pre_events,
      const std::vector<PublicEvent>& post_events) override;
  void randomize_unseen(IGameState& state, int observer,
                        std::mt19937_64& rng) const override;
  std::unique_ptr<IBeliefTracker> clone() const override {
    return std::make_unique<LoveLetterBeliefTracker<NPlayers>>(*this);
  }
  AnyMap serialize() const override;
};

extern template class LoveLetterFeatureEncoder<2>;
extern template class LoveLetterFeatureEncoder<3>;
extern template class LoveLetterFeatureEncoder<4>;
extern template class LoveLetterBeliefTracker<2>;
extern template class LoveLetterBeliefTracker<3>;
extern template class LoveLetterBeliefTracker<4>;

}  // namespace board_ai::loveletter
