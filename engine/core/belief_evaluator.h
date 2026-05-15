#pragma once

#include <vector>

namespace board_ai {

// Belief network evaluator. Mirrors IPolicyValueEvaluator's role for
// the belief net: take a flat feature vector (produced by
// IBeliefFeatureExtractor::extract) and return raw logits over the
// (N-1) × K opp-role flat output.
//
// Called once per decision at MCTS root via
// IBeliefTracker::prepare_for_root — the resulting posterior is cached
// inside the session-shared tracker, then cloned into each sim's
// sim_tracker so that randomize_unseen biases the determinization
// sample (Wallenius noncentral hypergeometric).
//
// Logits are returned uncalibrated; the tracker applies softmax with
// the per-game temperature (game.json `belief.temperature`) before
// sampling.
class IBeliefEvaluator {
 public:
  virtual ~IBeliefEvaluator() = default;

  // Returns true on success, fills `logits` with `output_logit_count()`
  // floats. On failure throws (no silent return-false: see CLAUDE.md
  // "No Fallbacks, No Silent Degradation").
  virtual bool evaluate(
      const std::vector<float>& features,
      std::vector<float>* logits) const = 0;
};

}  // namespace board_ai
