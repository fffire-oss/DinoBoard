#pragma once

// Phase 1.3: ITracker interface — the golden-standard observer memory.
//
// Replaces (in the long run) `IBeliefTracker` from belief_tracker.h. The
// new contract is:
//   - tracker is OPTIONAL — only hidden-info games register one. Fully
//     public games (TicTacToe, Quoridor, Azul) do NOT need a tracker.
//   - tracker is PUBLIC — it caches NN-feature-shaped derivations of the
//     observation history (e.g. multiset of cards still in deck). It does
//     NOT participate in node-key hashing, NOT in protocol payloads, NOT
//     in MCTS prior. Hash and protocol use state.viz_ + extract_snapshot.
//   - tracker.observe is auto-called by the framework inside
//     `apply_observation` AFTER state has been updated from the snapshot
//     (golden standard §3.3). Game authors do NOT wire event streams; they
//     just implement observe.
//   - For sim-internal updates during MCTS descent, observe_in_sim takes
//     the live sim state and reconstructs the equivalent
//     PerspectiveSnapshot via extract_snapshot — no second wire format.
//
// The legacy IBeliefTracker remains in belief_tracker.h until Phase 3
// migrates each game off it. New games SHOULD implement ITracker.
//
// Phase 1.3 lands the interface only — there is no framework-side caller
// yet. Phase 1.6 wires it into apply_observation; Phase 3.7 wires
// observe_in_sim into the MCTS sim loop.

#include <memory>
#include <vector>

#include "game_interfaces.h"
#include "types.h"

namespace board_ai {

// Forward decl — full definition in engine/core/snapshot.h (Phase 1.6).
// Carrying just the type name here keeps tracker.h orthogonal to the
// snapshot wire format; downstream games include snapshot.h to get the
// concrete type.
struct PerspectiveSnapshot;

class ITracker {
 public:
  virtual ~ITracker() = default;

  // Called once at session start. `initial` is the perspective player's
  // step=0 PerspectiveSnapshot (= snapshot of state immediately after
  // reset_with_seed, taken from `perspective`'s view).
  virtual void init(int perspective, const PerspectiveSnapshot& initial) = 0;

  // Session-layer hook: called by the framework inside apply_observation
  // AFTER state has been updated from `snap`. `actor` and `action` carry
  // the action that produced the snapshot; `post_events` is the public
  // event stream attached to that action.
  //
  // Tracker MUST update its memory ONLY from (snap, post_events). It
  // MUST NOT peek at any other state or session field.
  virtual void observe(int actor, ActionId action,
                       const PerspectiveSnapshot& snap,
                       const std::vector<PublicEvent>& post_events) = 0;

  // Sim-layer hook: called during MCTS descent after each
  // do_action_fast(sim_state, action). Implementations typically
  // reconstruct a PerspectiveSnapshot from sim_state via
  // extract_snapshot(sim_state, perspective) and reuse the body of
  // observe.
  virtual void observe_in_sim(int actor, ActionId action,
                              const IGameState& sim_state) = 0;

  // Deep copy. MCTS sim entry clones session.tracker so each sim has
  // an independent tracker that can drift via observe_in_sim without
  // contaminating the session-layer tracker.
  virtual std::unique_ptr<ITracker> clone() const = 0;
};

// Hidden-info games whose Belief implementation samples from a
// publicly-unseen multiset (the default UniformBelief case) implement
// this mixin on their tracker. Multi-pool games (Splendor: tier-1/2/3)
// route by `pool_name`; single-pool games (Love Letter, Coup) ignore
// `pool_name` and return the same multiset for every key.
//
// Trackers without hidden cid pools (TicTacToe, Quoridor, Azul) do NOT
// inherit this — fully public games skip belief.sample altogether.
class IUnseenPool {
 public:
  virtual ~IUnseenPool() = default;

  // Multiset of card-ids (or whatever opaque integer slots the schema
  // uses) that the perspective player has NOT yet seen in `pool_name`.
  // Returned as a plain int vector with multiplicities — order is
  // implementation-defined, callers MUST treat it as a multiset.
  virtual std::vector<int> publicly_unseen(
      const std::string& pool_name) const = 0;
};

}  // namespace board_ai
