#pragma once

#include <cstdint>
#include <vector>

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
  const IBeliefTracker* root_belief_tracker = nullptr;

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
  // pipelines) materializes a MaskedState once via make_masked_state and
  // hands it to both the hash and the evaluator — single walker pass per
  // descent step (golden standard §2.5).
  //
  // `tracker` carries public-derived statistics the encoder may consume
  // alongside the MaskedState (ALGORITHM_OVERVIEW §6 input surface).
  // Pass null when the game has no tracker, or when the caller is on a
  // path that doesn't track belief (e.g. analysis tooling that drives
  // the evaluator directly off a state). The evaluator forwards the
  // pointer to its underlying encoder unchanged.
  virtual bool evaluate(
      const MaskedState& masked,
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
