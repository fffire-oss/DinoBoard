#pragma once

#include <memory>
#include <random>
#include <vector>

#include "azul_state.h"
#include "../../engine/core/belief_tracker.h"
#include "../../engine/core/feature_encoder.h"
#include "../../engine/core/game_interfaces.h"

namespace board_ai::azul {

// Azul has fully-public state — every game-facing field is all_public,
// including bag and box_lid (declared in schema as per-color counts). No
// per-player hidden information; private_feature_dim() is 0 and no
// belief tracker is registered (decision-side determinization at MCTS sim
// entry has nothing to fill).
template <int NPlayers>
class AzulFeatureEncoder final : public IFeatureEncoder {
 public:
  using Cfg = AzulConfig<NPlayers>;
  int action_space() const override { return Cfg::kActionSpace; }
  int feature_dim() const override { return Cfg::kFeatureDim; }
  int public_feature_dim() const override { return Cfg::kFeatureDim; }
  int private_feature_dim() const override { return 0; }

  void encode_public(
      const IGameState& state,
      int perspective_player,
      const IBeliefTracker* tracker,
      std::vector<float>* out) const override;

  void encode_private(
      const IGameState& /*state*/,
      int /*player*/,
      const IBeliefTracker* /*tracker*/,
      std::vector<float>* /*out*/) const override {}
};

extern template class AzulFeatureEncoder<2>;
extern template class AzulFeatureEncoder<3>;
extern template class AzulFeatureEncoder<4>;

}  // namespace board_ai::azul
