#pragma once

#include <any>
#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "../../engine/core/game_interfaces.h"
#include "../../engine/core/visibility_schema.h"

namespace board_ai::loveletter {

constexpr std::int8_t kGuard = 1;
constexpr std::int8_t kPriest = 2;
constexpr std::int8_t kBaron = 3;
constexpr std::int8_t kHandmaid = 4;
constexpr std::int8_t kPrince = 5;
constexpr std::int8_t kKing = 6;
constexpr std::int8_t kCountess = 7;
constexpr std::int8_t kPrincess = 8;

constexpr int kCardTypes = 8;
constexpr int kTotalCards = 16;
constexpr std::array<int, 9> kCardCounts = {0, 5, 2, 2, 2, 2, 1, 1, 1};

constexpr int kGuardOffset = 0;
constexpr int kGuardCount = 28;       // target(0..3) * 7 + (guess-2)
constexpr int kPriestOffset = 28;
constexpr int kPriestCount = 4;
constexpr int kBaronOffset = 32;
constexpr int kBaronCount = 4;
constexpr int kHandmaidAction = 36;
constexpr int kPrinceOffset = 37;
constexpr int kPrinceCount = 4;       // includes self-target
constexpr int kKingOffset = 41;
constexpr int kKingCount = 4;
constexpr int kCountessAction = 45;
constexpr int kPrincessAction = 46;
constexpr int kActionSpace = 47;

template <int NPlayers>
struct LoveLetterConfig {
  static_assert(NPlayers >= 2 && NPlayers <= 4);
  static constexpr int kPlayers = NPlayers;
  // Public per-player (visible to all): alive, protected, current_player,
  // hand_exposed (4), discard counts (8), discard size (1) = 13.
  static constexpr int kPerPlayerPublicFeatures = 13;
  // Private per-player (from p's perspective, about p or p's knowledge of
  // others): hand one-hot (8) + drawn_card one-hot (8) = 16.
  static constexpr int kPerPlayerPrivateFeatures = 16;
  static constexpr int kPerPlayerFeatures =
      kPerPlayerPublicFeatures + kPerPlayerPrivateFeatures;  // 29
  static constexpr int kGlobalFeatures = 12;
  static constexpr int kFeatureDim = kPerPlayerFeatures * NPlayers + kGlobalFeatures;
  static constexpr int kPublicFeatureDim =
      kPerPlayerPublicFeatures * NPlayers + kGlobalFeatures;
  static constexpr int kPrivateFeatureDim =
      kPerPlayerPrivateFeatures * NPlayers;
  static constexpr int kFaceUpRemoved = NPlayers == 2 ? 3 : 0;
};

template <int NPlayers>
struct LoveLetterData {
  using Cfg = LoveLetterConfig<NPlayers>;

  int current_player = 0;
  std::int8_t first_player = 0;
  int winner = -1;
  bool terminal = false;
  int ply = 0;

  std::array<std::int8_t, Cfg::kPlayers> hand{};
  std::int8_t drawn_card = 0;

  std::array<std::int8_t, Cfg::kPlayers> alive{};
  std::array<std::int8_t, Cfg::kPlayers> protected_flags{};
  std::array<std::int8_t, Cfg::kPlayers> hand_exposed{};

  std::vector<std::int8_t> deck;
  std::int8_t set_aside_card = 0;
  std::array<std::vector<std::int8_t>, Cfg::kPlayers> discard_piles;
  std::vector<std::int8_t> face_up_removed;
};

template <int NPlayers>
struct LoveLetterState final : public CloneableState<LoveLetterState<NPlayers>> {
  using Cfg = LoveLetterConfig<NPlayers>;
  LoveLetterData<NPlayers> data;
  std::vector<LoveLetterData<NPlayers>> undo_stack;

  LoveLetterState();

  // Phase 3 — visibility schema. LoveLetter is the most subtle game we
  // ship; the schema captures the full visibility layout but does NOT
  // by itself wire dynamic reveals — those will land in a follow-on PR
  // that mutates state.viz_ from inside do_action_fast (Priest peek,
  // Baron compare, end-of-round flips). Partition:
  //
  //   - all_public scalars: current_player, first_player, winner,
  //     terminal, ply.
  //   - all_public 1D: alive[N], protected_flags[N], hand_exposed[N]
  //     (the public "this seat's hand is now revealed to everyone"
  //     flag — Baron-loss / showdown / Princess-played flips this).
  //   - owner_only_first_axis: hand[N]. Each player sees only their
  //     own hand card. When hand_exposed[p] flips public, downstream
  //     consumers gate on the public flag rather than mutating viz —
  //     same pattern as Coup's revealed[] gate on influence[].
  //   - all_hidden: drawn_card (rules will reveal_slot_to(current_player)
  //     on draw, reset_to_base on play; the field is briefly held only
  //     by the active player). set_aside_card is permanently hidden
  //     (never revealed to anyone — it's the card removed from the
  //     bottom of the deck at game start).
  //
  // Variable-length vectors NOT declared as schema slots:
  //   - deck: hidden contents, public size — randomize_unseen handles.
  //   - discard_piles[N]: all-public stack, variable length — already
  //     hashed in hash_public_fields slot-by-slot.
  //   - face_up_removed: 2p-only, all-public, variable length —
  //     already hashed in hash_public_fields.
  //
  // Higher-order belief reasoning (e.g. tracking "what does opp infer
  // from my last Guard guess?") is an MCTS-tracker concern, not a
  // schema concern: the schema only models direct first-order
  // visibility ("who sees what right now"). The tracker can still
  // condition on the public action history independent of viz.
  static const viz::VisibilitySchema& schema();

  void reset_with_seed(std::uint64_t seed) override;
  // Re-seed `viz_` from the visibility schema base without touching
  // the data payload. Called by the registrar on the API/session
  // path after `apply_initial_observation` mutates `data` directly,
  // so any stale reveals carried over from `reset_with_seed` are
  // cleared before the per-perspective starting reveals are re-applied
  // through rules.
  void reseed_viz();

  StateHash64 state_hash() const override;
  void hash_field_slot(Hasher& h, const std::string& name,
                       const std::vector<int>& idx) const override;
  void hash_extra_state_fields(int perspective, Hasher& h) const override;
  void mask_field_slot(const std::string& name,
                       const std::vector<int>& idx) override;
  std::any read_field_slot(const std::string& name,
                           const std::vector<int>& idx) const override;
  void write_field_slot(const std::string& name,
                        const std::vector<int>& idx,
                        const std::any& value) override;
  const viz::VisibilitySchema& schema_ref() const override { return schema(); }
  int current_player() const override;
  int first_player() const override;
  bool is_terminal() const override;
  int num_players() const override { return Cfg::kPlayers; }
  int winner() const override;
};

extern template struct LoveLetterData<2>;
extern template struct LoveLetterData<3>;
extern template struct LoveLetterData<4>;
extern template struct LoveLetterState<2>;
extern template struct LoveLetterState<3>;
extern template struct LoveLetterState<4>;

}  // namespace board_ai::loveletter
