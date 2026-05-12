// Out-of-line default body for IGameState::mask_all_hidden_slots.
//
// Lives in its own TU because the walker (viz_walker.h) includes
// game_interfaces.h, so the body cannot live in game_interfaces.h
// without an include cycle. Splendor-style games override this method
// to detach a writable persistent copy before walking; everyone else
// uses the default implementation here.

#include "game_interfaces.h"

#include "viz_walker.h"

namespace board_ai {

void IGameState::mask_all_hidden_slots(
    const viz::VisibilitySchema& schema, int perspective,
    const std::unordered_map<std::string, viz::VizTensor>* belief_filled) {
  viz::for_each_hidden_slot(
      *this, schema, perspective, belief_filled,
      [this](const std::string& name, const std::vector<int>& idx,
             const viz::VizTensor& /*v*/) {
        this->mask_field_slot(name, idx);
      });
}

}  // namespace board_ai
