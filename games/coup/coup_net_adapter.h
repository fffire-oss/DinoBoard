#pragma once

#include <array>
#include <memory>
#include <random>
#include <vector>

#include "../../engine/core/belief_evaluator.h"
#include "../../engine/core/belief_feature_extractor.h"
#include "../../engine/core/belief_label_extractor.h"
#include "../../engine/core/belief_tracker.h"
#include "../../engine/core/feature_encoder.h"
#include "../../engine/core/game_interfaces.h"
#include "coup_state.h"

namespace board_ai::coup {

template <int NPlayers>
class CoupFeatureEncoder final : public IFeatureEncoder {
 public:
  using Cfg = CoupConfig<NPlayers>;
  int action_space() const override { return kActionSpace; }
  int feature_dim() const override { return Cfg::kFeatureDim; }

  void encode_features(
      const IGameState& masked_state,
      int perspective_player,
      const IBeliefTracker* tracker,
      std::vector<float>* out) const override;
};

// Heuristic belief tracker for Coup. Maintains a per-opponent role-signal
// count derived from public claim/challenge/reveal/exchange history and uses
// it to bias `randomize_unseen` away from uniform sampling. Motivation:
// under uniform sampling, MCTS converges to a degenerate "never challenge,
// never bluff" equilibrium on bluff-heavy games like Coup.
template <int NPlayers>
class CoupBeliefTracker final : public IBeliefTracker {
 public:
  using Cfg = CoupConfig<NPlayers>;
  CoupBeliefTracker() { last_revealed_role_.fill(-1); }
  void init(IGameState& state, int perspective,
            const AnyMap& payload) override;
  void observe_public_event(
      int actor,
      ActionId action,
      const std::vector<PublicEvent>& events) override;
  void randomize_unseen(IGameState& state, int observer,
                        std::mt19937_64& rng) const override;
  void prepare_for_root(
      const IGameState& root_state,
      int root_player,
      const IBeliefFeatureExtractor* extractor,
      const IBeliefEvaluator* evaluator) override;
  std::unique_ptr<IBeliefTracker> clone() const override {
    return std::make_unique<CoupBeliefTracker<NPlayers>>(*this);
  }
  AnyMap serialize() const override;

  // Belief-net softmax temperature. Default matches game.json
  // `belief.temperature: 2.0` (Plan §1.4/§1.5); per-game.json injection
  // is a Phase-3 follow-up. Per-row softmax along the role axis (Plan §1.2).
  static constexpr float kBeliefTemperature = 2.0f;

  // Cached per-opp posterior pi[i][R] from prepare_for_root.
  // i ∈ [0, N-2] — opp_to_player(observer, i, N) (Plan §1.2.1).
  // pi_valid_ is false when no belief net has been installed for this
  // decision; randomize_unseen falls back to uniform sampling.
  bool pi_valid() const { return pi_valid_; }
  float pi(int opp_idx, int role) const {
    return pi_[opp_idx][role];
  }
  int pi_observer() const { return pi_observer_; }

  // For tests / encoder: expose signal counts and pending-claim state.
  int signal_count(int player, int role) const {
    if (player < 0 || player >= NPlayers) return 0;
    if (role < 0 || role >= kCharacterCount) return 0;
    return signals_[player][role];
  }
  int pending_claimer() const { return pending_claimer_; }
  int pending_claim_role() const { return pending_claim_role_; }
  bool pending_challenged() const { return pending_challenged_; }

  // v0.2 raw-event accumulators. Read by CoupBeliefFeatureExtractor (which
  // is the only consumer that needs the pre/post layering).
  enum class ReshuffleKind : std::int8_t {
    kNone = 0,
    kExchange = 1,
    kRevealTruthful = 2,
  };
  int pre_claim_count(int player, int role) const {
    return pre_claim_counts_[player][role];
  }
  int post_claim_count(int player, int role) const {
    return post_claim_counts_[player][role];
  }
  int pre_challenge_initiated(int player, int role) const {
    return pre_challenge_initiated_[player][role];
  }
  int post_challenge_initiated(int player, int role) const {
    return post_challenge_initiated_[player][role];
  }
  ReshuffleKind last_reshuffle_kind(int player) const {
    return last_reshuffle_kind_[player];
  }
  int last_revealed_role(int player) const {
    return last_revealed_role_[player];
  }

 private:
  // Reshuffle promotion: pre += post; post = 0. Called from
  // observe_public_event on RevealTruthful / Exchange events.
  void promote_post_to_pre(int player) {
    if (player < 0 || player >= NPlayers) return;
    for (int r = 0; r < kCharacterCount; ++r) {
      pre_claim_counts_[player][r] += post_claim_counts_[player][r];
      post_claim_counts_[player][r] = 0;
      pre_challenge_initiated_[player][r] +=
          post_challenge_initiated_[player][r];
      post_challenge_initiated_[player][r] = 0;
    }
  }

  int perspective_player_ = -1;

  // Non-negative count of "evidence that opp p may hold role R". Bumped on
  // claim/challenge signals; reset to 0 when the evidence is invalidated
  // (reveal, successful challenge reshuffling the card back to deck, etc.).
  std::array<std::array<int, kCharacterCount>, NPlayers> signals_{};

  // v0.2 raw-event accumulators (per opponent × role). On any reshuffle
  // event (Exchange complete / claim_resolved_truthful), we promote
  // post → pre and zero post for that opp; the pair therefore encodes
  // "evidence carried over a reshuffle" vs "evidence accrued since".
  std::array<std::array<int, kCharacterCount>, NPlayers> pre_claim_counts_{};
  std::array<std::array<int, kCharacterCount>, NPlayers> post_claim_counts_{};
  std::array<std::array<int, kCharacterCount>, NPlayers>
      pre_challenge_initiated_{};
  std::array<std::array<int, kCharacterCount>, NPlayers>
      post_challenge_initiated_{};
  std::array<ReshuffleKind, NPlayers> last_reshuffle_kind_{};
  // -1 when last_reshuffle_kind != kRevealTruthful, else 0..kCharacterCount-1.
  // Default-constructed to all -1 in the ctor body so a freshly-bundled
  // (un-init()'d) tracker already reports "no reveal yet" rather than
  // role=Duke. (Cannot use lambda NSDMI here — `kCharacterCount` is not
  // visible inside the inline initializer of a template member.)
  std::array<std::int8_t, NPlayers> last_revealed_role_{};

  // Cached belief-net posterior. Filled by prepare_for_root, read by
  // randomize_unseen. clone() copies these by value, so each sim's
  // sim_tracker carries the cached posterior — extractor + evaluator
  // run only once per decision at the root (Plan §2.1).
  // pi_[i][R] is a per-row softmax: Σ_R pi_[i][R] == 1 for each i.
  std::array<std::array<float, kCharacterCount>, NPlayers - 1> pi_{};
  bool pi_valid_ = false;
  int pi_observer_ = -1;

  // Multi-step claim resolution requires tracking pending state across calls:
  //   stage t  : claimer plays Tax         -> pending_claimer=P, claim_role=Duke
  //   stage t+1: opponent plays Challenge  -> pending_challenger=Q
  //   stage t+2: claimer plays RevealSlotN -> extractor emits card_revealed,
  //              compare revealed role vs pending_claim_role, resolve signals.
  //   stage t+1': opponent plays Allow     -> no challenge, claim_unchallenged
  int pending_claimer_ = -1;
  int pending_claim_role_ = -1;
  bool pending_challenged_ = false;
};

// Belief-net feature extractor (Plan §1.3).
//
// Output layout per call (length kFeatureDim):
//
//   global block (6):
//     remaining[R] for R in {Duke,Assassin,Captain,Ambassador,Contessa}  (5)
//     ply_count / 200                                                    (1)
//
//   per-opp block (28 × (N-1)), in opp_to_player(observer, i, N) order:
//     pre_claim_counts[R]            / 4  clamp [0,1]                    (5)
//     post_claim_counts[R]           / 4  clamp [0,1]                    (5)
//     pre_challenge_initiated[R]     / 4  clamp [0,1]                    (5)
//     post_challenge_initiated[R]    / 4  clamp [0,1]                    (5)
//     last_reshuffle_kind one-hot ∈ {None, Exchange, RevealTruthful}     (3)
//     last_revealed_role one-hot (zero unless kind==RevealTruthful)      (5)
//
// `remaining[R]` is observer-derived: total 3 cards per role, minus any
// `revealed=true` slot of that role (publicly revealed across all
// perspectives), minus the observer's own unrevealed influence slots of
// that role (viz=1 to observer). Other players' unrevealed slots are
// viz=0 → kPlaceholder; do not contribute.
template <int NPlayers>
class CoupBeliefFeatureExtractor final : public IBeliefFeatureExtractor {
 public:
  using Cfg = CoupConfig<NPlayers>;
  static constexpr int kPerOppDim = 28;
  static constexpr int kGlobalDim = 6;
  static constexpr int kFeatureDim = kGlobalDim + kPerOppDim * (NPlayers - 1);
  static constexpr int kLogitCount = (NPlayers - 1) * kCharacterCount;

  int feature_dim() const override { return kFeatureDim; }
  int output_logit_count() const override { return kLogitCount; }

  void extract(
      const IGameState& masked_state,
      int perspective_player,
      const IBeliefTracker* tracker,
      std::vector<float>* out) const override;
};

extern template class CoupFeatureEncoder<2>;
extern template class CoupFeatureEncoder<3>;
extern template class CoupFeatureEncoder<4>;
extern template class CoupBeliefTracker<2>;
extern template class CoupBeliefTracker<3>;
extern template class CoupBeliefTracker<4>;
extern template class CoupBeliefFeatureExtractor<2>;
extern template class CoupBeliefFeatureExtractor<3>;
extern template class CoupBeliefFeatureExtractor<4>;

// Belief-net label extractor (Plan §2.2). GT-side counterpart to
// CoupBeliefFeatureExtractor — reads truth CoupState and emits the
// per-observer (hand_counts, remaining, alive_per_opp) tuple consumed
// by the Python training loop. Never reachable from the AI session.
//
// Shape per call:
//   hand_counts[i][R] = count of unrevealed R-cards in opp i's hand,
//                       opp i = (observer + 1 + i) mod NPlayers.
//   remaining[R]      = kCardsPerCharacter(3) − own unrevealed R
//                       − publicly revealed R across all seats.
//   alive_per_opp[i]  = 1 iff opp i has ≥ 1 unrevealed slot, else 0.
template <int NPlayers>
class CoupBeliefLabelExtractor final : public IBeliefLabelExtractor {
 public:
  int label_class_count() const override { return kCharacterCount; }

  void extract(
      const IGameState& truth_state,
      int observer,
      std::vector<std::vector<int>>* out_hand_counts,
      std::vector<int>* out_remaining,
      std::vector<int>* out_alive) const override;
};

extern template class CoupBeliefLabelExtractor<2>;
extern template class CoupBeliefLabelExtractor<3>;
extern template class CoupBeliefLabelExtractor<4>;

}  // namespace board_ai::coup
