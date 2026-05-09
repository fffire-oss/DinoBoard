#pragma once

// Phase 1.3: IBelief interface + UniformBelief skeleton.
//
// IBelief.sample is called ONCE at MCTS sim entry to determinize all
// `viz[..., my_seat] == 0` slots in a sim-local clone of session.state.
// After sample returns, the sim runs deterministically: rules see a
// state where every field has a concrete value, regardless of viz.
//
// Lifecycle:
//   - Phase 1.3 (this PR): interface + UniformBelief skeleton. The
//     skeleton is a no-op when state.viz_ is empty (= no schema-declared
//     hidden fields), which keeps fully-public games (TicTacToe,
//     Quoridor, Azul) running unchanged. For hidden-info games, the
//     active body lands in Phase 1.5 alongside the framework viz walker
//     (the body needs schema-driven field reflection to write into state
//     slots — schema lookup is name-keyed only at this stage).
//   - Phase 3.7 (bridge PR): MCTS sim entry calls belief->sample once
//     per simulation, after cloning session.state. Until that PR lands,
//     no framework code calls IBelief — Phase 1.3 ships the contract
//     only.
//
// Why a no-op default for empty viz: the existing fully-public games
// (TTT / Quoridor / Azul) MUST keep working when Phase 3.7 wires
// belief into the MCTS sim loop. Once schemas are declared (Phase 3),
// `state.viz_` becomes non-empty even for fully-public games, but
// every entry will be all_public; the Phase 1.5 body of UniformBelief
// will skip slots whose viz already permits the perspective. Until the
// schemas land, the no-op fallback is the safe default.

#include <functional>
#include <memory>
#include <random>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "game_interfaces.h"
#include "tracker.h"
#include "visibility_schema.h"

namespace board_ai {

class IBelief {
 public:
  virtual ~IBelief() = default;

  // Determinize all `viz[..., perspective] == 0` slots in `state` by
  // sampling concrete values consistent with `tracker`'s information
  // set. Marks the filled slots in `belief_filled` (parallel to
  // state.viz_, sim-local).
  //
  // Inputs:
  //   state — sim-local clone of session.state. Hidden slots may carry
  //           garbage (zero-initialized after extract_snapshot at
  //           session layer); sample MUST overwrite them.
  //   tracker — session.tracker->clone() (or nullptr for fully-public
  //             games without a tracker). Trackers that drive
  //             UniformBelief expose IUnseenPool; sample dynamic_casts
  //             to it as needed.
  //   perspective — session.my_seat — fixed for the lifetime of one
  //             sim; encoder's `state.current_player()` perspective
  //             rotates separately during descent.
  //   belief_filled — parallel to state.viz_, ALL ZERO at entry. Each
  //             slot the sampler writes is set to 1 along the
  //             perspective axis.
  //   rng — caller-owned, sim-local. Independent across sims so each
  //             sim sees a different sample.
  //
  // After return: state has every slot filled with a concrete value,
  // and belief_filled marks which ones came from sample (vs which were
  // already viz=1 from snapshot).
  virtual void sample(IGameState& state, const ITracker* tracker,
                      int perspective,
                      std::unordered_map<std::string, viz::VizTensor>& belief_filled,
                      std::mt19937& rng) const = 0;

  virtual std::unique_ptr<IBelief> clone() const = 0;
};

// Framework default: uniform sample from `tracker.publicly_unseen(pool)`.
// Skeleton in Phase 1.3 (no-op when state.viz_ is empty); active body
// lands in Phase 1.5 once the framework viz walker can iterate state
// slots by FieldDecl.
//
// For hidden-info games to use UniformBelief, their tracker MUST
// implement IUnseenPool. This is enforced at register-game time
// (HiddenInfoGameMissingTrackerError) — see Phase 1.2 register-time
// schema validation, scheduled to land alongside Phase 3 game-side
// schema declarations.
class UniformBelief final : public IBelief {
 public:
  // pool_selector: maps a (field_name, indices...) to the pool key the
  // sampler should pull from. Single-pool games leave this null —
  // sample uses pool_name = "default" everywhere. Multi-pool games
  // (Splendor) install a selector that reads a public field from
  // `state` (e.g. reserved_tier[p][i]) to pick the pool.
  //
  // Phase 1.3 ships the type signature only; Phase 1.5 wires it into
  // sample's body once the field walker can pass real (field, idx)
  // tuples to the lambda.
  using PoolSelector = std::function<std::string(
      const IGameState& /*state*/, const std::string& /*field_name*/,
      const std::vector<int>& /*idx*/)>;

  UniformBelief() = default;
  explicit UniformBelief(PoolSelector selector)
      : pool_selector_(std::move(selector)) {}

  void sample(IGameState& state, const ITracker* tracker,
              int perspective,
              std::unordered_map<std::string, viz::VizTensor>& belief_filled,
              std::mt19937& rng) const override {
    // Phase 1.3 skeleton: when state.viz_ is empty (= no schema
    // declared, common until Phase 3 lands per-game schemas), there's
    // nothing to sample. This is the all-public-game contract.
    if (state.viz_.empty()) {
      (void)tracker;
      (void)perspective;
      (void)belief_filled;
      (void)rng;
      return;
    }

    // Phase 1.5 body — sketched here, intentionally not active. The
    // body needs:
    //   - viz_walker.h iterating non-internal fields slot-by-slot
    //   - typed-field reflection to write `int / int8 / vector<int>`
    //     payloads into state via member-pointer registry
    //   - per-pool tracker dispatch (dynamic_cast to IUnseenPool)
    //
    // Until those are in place, throwing here would break Phase 3.7
    // bring-up for hidden-info games. Instead the skeleton flags the
    // call site so a Phase 1.5 / 3 PR can grep-find every place that
    // expects a body.
    //
    //     for each non-internal field f in state.viz_ {
    //         walk slots of f.viz with viz[idx, perspective] == 0:
    //             pool = pool_selector_ ? pool_selector_(state, f.name, idx)
    //                                   : "default";
    //             auto* unseen = dynamic_cast<const IUnseenPool*>(tracker);
    //             if (!unseen) throw HiddenInfoNoTrackerError(f.name);
    //             auto pool_copy = unseen->publicly_unseen(pool);
    //             // remove cids the perspective already saw via viz=1
    //             // slots in the same pool, then uniform-pick into state
    //             belief_filled[f.name].data[idx_offset + perspective] = 1;
    //     }
    (void)tracker;
    (void)perspective;
    (void)belief_filled;
    (void)rng;
  }

  std::unique_ptr<IBelief> clone() const override {
    return std::make_unique<UniformBelief>(pool_selector_);
  }

 private:
  PoolSelector pool_selector_;
};

}  // namespace board_ai
