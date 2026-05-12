#pragma once

#include <memory>
#include <random>
#include <unordered_set>
#include <vector>

#include "../../engine/core/belief_tracker.h"
#include "../../engine/core/feature_encoder.h"
#include "../../engine/core/game_interfaces.h"
#include "splendor_state.h"

namespace board_ai::splendor {

// Splendor has non-symmetric hidden info (blind reserved cards). The
// public half encodes tableau, bank, nobles, all players' public state
// (gems, bonuses, scores, visible reserved). The private half encodes
// the player's own blind reserved cards — observer cannot see other
// players' blind reserved.
template <int NPlayers>
class SplendorFeatureEncoder final : public IFeatureEncoder {
 public:
  using Cfg = SplendorConfig<NPlayers>;
  int action_space() const override { return Cfg::kActionSpace; }
  int feature_dim() const override { return Cfg::kFeatureDim; }
  int public_feature_dim() const override;
  int private_feature_dim() const override;

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

template <int NPlayers>
class SplendorBeliefTracker final : public IBeliefTracker {
 public:
  using Cfg = SplendorConfig<NPlayers>;
  void init(const AnyMap& initial_observation) override;
  void observe_public_event(
      int actor,
      ActionId action,
      const std::vector<PublicEvent>& pre_events,
      const std::vector<PublicEvent>& post_events) override;
  void randomize_unseen(IGameState& state, int observer,
                        std::mt19937_64& rng) const override;
  std::unique_ptr<IBeliefTracker> clone() const override {
    return std::make_unique<SplendorBeliefTracker<NPlayers>>(*this);
  }
  AnyMap serialize() const override;

 private:
  // Perspective-agnostic. Holds only public-derivable card multisets:
  // tableau cards from init + deck-flip post-events. Private knowledge
  // (own face-down reserved card ids) lives on state.viz=1 slots — not
  // here — so the tracker stays observer-independent.
  std::unordered_set<int> seen_cards_;
  bool initialized_ = false;
};

extern template class SplendorFeatureEncoder<2>;
extern template class SplendorFeatureEncoder<3>;
extern template class SplendorFeatureEncoder<4>;
extern template class SplendorBeliefTracker<2>;
extern template class SplendorBeliefTracker<3>;
extern template class SplendorBeliefTracker<4>;

}  // namespace board_ai::splendor
