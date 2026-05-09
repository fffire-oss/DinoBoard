#pragma once

#include <vector>

#include "../../engine/core/game_interfaces.h"
#include "splendor_state.h"

namespace board_ai::splendor {

template <int NPlayers>
class SplendorRules final : public IGameRules {
 public:
  using Cfg = SplendorConfig<NPlayers>;

  SplendorRules() = default;

  bool validate_action(const IGameState& state, ActionId action) const override;
  std::vector<ActionId> legal_actions(const IGameState& state) const override;
  UndoToken do_action_fast(IGameState& state, ActionId action) const override;
  UndoToken do_action_deterministic(IGameState& state, ActionId action) const override;
  void undo_action(IGameState& state, const UndoToken& token) const override;

  static bool is_terminal_data(const SplendorData<NPlayers>& data);
  static std::vector<ActionId> legal_actions_data(const SplendorData<NPlayers>& d);
  // Phase 2: takes IGameState& so randomness can flow through
  // state.derive_rng() instead of a SplendorData-internal nonce. The
  // state argument is also the only path to the framework draw_nonce_
  // counter that disambiguates per-call rng streams.
  static SplendorData<NPlayers> apply_action_copy(
      const SplendorData<NPlayers>& src, ActionId action, IGameState& state);
};

extern template class SplendorRules<2>;
extern template class SplendorRules<3>;
extern template class SplendorRules<4>;

}  // namespace board_ai::splendor
