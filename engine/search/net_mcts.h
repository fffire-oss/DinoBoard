#pragma once

#include <cstdint>
#include <vector>

#include <functional>

#include "../core/belief_evaluator.h"
#include "../core/belief_feature_extractor.h"
#include "../core/belief_tracker.h"
#include "../core/game_interfaces.h"
#include "../core/masked_state.h"
#include "tail_solver.h"

namespace board_ai::search {

// Opponent-node selection mode (Smooth-UCT-style toggle).
//
// kPuct (default): every node — root and opponents — uses UCT2 + PUCT to
// pick the next edge. Backwards-compatible.
//
// kFrozenPrior: descent on opponent nodes (cur_idx != 0 AND
// to_play != root acting player) bypasses bandit and samples directly
// from the policy head's prior, which was set at the node's expansion
// and never updated. Mitigates ISMCTS strategy fusion / opponent
// omniscience: with bandit on opponent nodes, the descent's repeated
// visits accumulate Q toward "best response to this sim's sampled
// truth" — leaking the root-player certainty fixed by determinization.
// Frozen-prior makes the opponent's behavior a fixed mixed strategy
// across descent visits, with no per-sim feedback to exploit.
//
// Root is ALWAYS PUCT regardless of this setting. Backup, DAG sharing,
// hash, encoder, tracker, dirichlet, cover_root_edges all unaffected.
enum class OpponentSelection {
  kPuct = 0,
  kFrozenPrior = 1,
};

// ISMCTS MCTS config. The hidden-info machinery is driven entirely by
// root_belief_tracker:
//   - If non-null, each simulation clones the root state and calls
//     `tracker->randomize_unseen(*sim_state, rng)` once at sim start, to
//     sample a belief-consistent hidden world. After sampling, descent is
//     fully deterministic within that world.
//   - Tree nodes are keyed by `state.state_hash_for_perspective(current_player)`,
//     which combines public fields + acting player's private fields +
//     step_count (DAG acyclicity). Different paths that reach the same
//     (public, acting-player-private, step_count) triple share a node via
//     a global hash→node_index table — the tree is a DAG, not a pure tree.
//   - No chance node machinery: physical randomness resolves at root-sampling time.
struct NetMctsConfig {
  int simulations = 200;
  float c_puct = 1.4f;
  int max_depth = 128;
  float value_clip = 1.0f;
  float root_dirichlet_alpha = 0.0f;
  float root_dirichlet_epsilon = 0.0f;

  // Enables ISMCTS root-sampling. When non-null, each simulation clones
  // root and calls tracker->randomize_unseen(sim_state, rng) before descent.
  // When null, search runs on root directly (fully-public games like
  // TicTacToe / Quoridor / Azul, where there are no viz=0 slots to fill).
  //
  // NOTE: caller-supplied root tracker is `const` for sim cloning, but
  // search_root needs to call `prepare_for_root` on it once (which mutates
  // the cached pi posterior) when belief net is installed. The
  // implementation const_casts internally for that single call;
  // randomization remains read-only via clone(). See net_mcts.cpp.
  const IBeliefTracker* root_belief_tracker = nullptr;

  // Belief network plumbing (Coup belief net plan). When BOTH non-null,
  // `search_root` calls `root_belief_tracker->prepare_for_root(root,
  // root.current_player(), belief_extractor, belief_evaluator)` once per
  // decision, before any simulation. The tracker caches the resulting
  // pi posterior; subsequent clones into sim_trackers inherit the
  // cached pi and use it inside `randomize_unseen` (Wallenius
  // noncentral hypergeometric). When either is null, the tracker
  // falls back to its default uniform / multiset-derived sampling
  // (default `prepare_for_root` is a no-op).
  const IBeliefFeatureExtractor* belief_extractor = nullptr;
  const IBeliefEvaluator* belief_evaluator = nullptr;

  // Sim-only events extractor — returns just the events vector, skipping
  // the wholesale `viz::serialize_public_snapshot` walk that the ply
  // path's full PublicEventExtractor performs. The sim doesn't need the
  // public_snapshot half (sim_state is itself a GT-equivalent advanced
  // by do_action_fast; node-player masking is done via make_masked_state
  // directly off sim_state.viz). Skipping the snapshot pays back ~50-80%
  // of per-step extractor cost on hidden-info games.
  //
  // When non-null AND root_belief_tracker is non-null, sim descent feeds
  // the cloned sim_tracker via observe_public_event after each
  // do_action_fast — keeping the sim-local tracker in sync with descent
  // so deep-node encoder reads see the up-to-date public-derived state.
  // When null, the sim_tracker stays frozen at the root-cloned state for
  // the duration of the sim. Fully-public games leave this null (they do
  // not register a tracker either).
  // Perspective passed to the extractor is the root acting player — by
  // contract (`test_tracker_perspective_invariance`) tracker content is
  // perspective-agnostic, so any perspective produces the same events.
  using EventsOnlyExtractorFn = std::function<std::vector<PublicEvent>(
      const IGameState& state_before,
      ActionId action,
      const IGameState& state_after,
      int perspective_player)>;
  EventsOnlyExtractorFn events_only_extractor;

  // Sim-step callback (debug/test only; production paths leave null).
  // Invoked once per descent step inside a simulation, AFTER do_action_fast
  // and AFTER the sim_tracker has been updated by observe_public_event
  // (when applicable). Used by `test_sim_tracker_descent_maintained` to
  // observe sim-local tracker state without exposing the sim's stack
  // frame. Production: nullptr → zero overhead.
  using OnSimStepFn = std::function<void(
      int sim_index,
      int step_index,
      const IGameState& sim_state_after,
      const IBeliefTracker* sim_tracker)>;
  OnSimStepFn on_sim_step;

  bool tail_solve_enabled = false;
  TailSolveConfig tail_solve_config{};
  const ITailSolver* tail_solver = nullptr;

  // Analysis-only: ensure every root legal edge gets at least one visit
  // before PUCT takes over. With this off, low-prior edges can end the
  // search at visit_count == 0, which makes their action_values (q=0)
  // collapse to a 50% win-rate readout — fine for argmax decisions but
  // poison for drop-score analysis. The web AI path leaves this off; the
  // analysis-pipeline path turns it on so action_values is dense over
  // the full legal set.
  bool cover_root_edges = false;

  // See OpponentSelection comment above. Default kPuct.
  OpponentSelection opponent_selection = OpponentSelection::kPuct;
};

struct NetMctsStats {
  int simulations_done = 0;
  std::int64_t expanded_nodes = 0;
  double nodes_per_sec = 0.0;
  double best_action_value = 0.0;
  std::vector<double> root_values{};
  std::vector<std::vector<double>> root_edge_values{};
  std::vector<ActionId> root_actions{};
  std::vector<int> root_action_visits{};
  bool tail_solve_attempted = false;
  bool tail_solve_completed = false;
  bool tail_solved = false;
  TailSolveOutcome tail_solve_outcome = TailSolveOutcome::kUnknown;
  float tail_solve_value = 0.0f;
  double tail_solve_elapsed_ms = 0.0;
  // DAG-specific stats: number of times a descend hit an existing hash→node
  // and reused it instead of creating. Useful for observing how much DAG
  // sharing saves relative to the total descent steps.
  std::int64_t dag_reuse_hits = 0;
  // Number of sims that ran to max_depth without finding a terminal or
  // unexpanded leaf — the descent looped through expanded nodes only.
  // Treated as a zero-value leaf (draw equivalent) so the sim still backs
  // up. Non-zero values usually indicate policy-collapse / mutual-block
  // deadlock (e.g. Coup steal-block stalemate); see KNOWN_ISSUES.
  std::int64_t depth_out_hits = 0;
};

ActionId select_action_from_visits(
    const std::vector<ActionId>& actions,
    const std::vector<int>& visits,
    double temperature,
    std::uint64_t rng_seed,
    ActionId fallback_action);

class IPolicyValueEvaluator {
 public:
  virtual ~IPolicyValueEvaluator() = default;
  // Mask-then-evaluate. Caller (NetMcts descent, externally-driven
  // pipelines) materializes a perspective-masked state once via
  // make_masked_state and hands it to both the hash and the evaluator —
  // single walker pass per descent step (golden standard §2.5).
  //
  // `tracker` carries public-derived statistics the encoder may consume
  // alongside the masked state (ALGORITHM_OVERVIEW §6 input surface).
  // Pass null when the game has no tracker, or when the caller is on a
  // path that doesn't track belief (e.g. analysis tooling that drives
  // the evaluator directly off a state). The evaluator forwards the
  // pointer to its underlying encoder unchanged.
  virtual bool evaluate(
      const IGameState& masked_state,
      int perspective_player,
      const IBeliefTracker* tracker,
      const std::vector<ActionId>& legal_actions,
      std::vector<float>* priors,
      std::vector<float>* values) const = 0;
};

class NetMcts {
 public:
  explicit NetMcts(NetMctsConfig cfg = {});

  ActionId search_root(
      const IGameState& root,
      const IGameRules& rules,
      const IStateValueModel& value_model,
      const IPolicyValueEvaluator& evaluator,
      NetMctsStats* stats = nullptr,
      std::uint64_t seed = 0) const;

 private:
  NetMctsConfig cfg_{};
};

}  // namespace board_ai::search
