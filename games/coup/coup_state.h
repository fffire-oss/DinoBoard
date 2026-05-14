#pragma once

#include <any>
#include <array>
#include <cstdint>
#include <vector>

#include "../../engine/core/game_interfaces.h"
#include "../../engine/core/visibility_schema.h"

namespace board_ai::coup {

using CharId = std::int8_t;
constexpr CharId kDuke = 0;
constexpr CharId kAssassin = 1;
constexpr CharId kCaptain = 2;
constexpr CharId kAmbassador = 3;
constexpr CharId kContessa = 4;
constexpr int kCharacterCount = 5;
constexpr int kCardsPerCharacter = 3;
constexpr int kTotalCards = 15;
constexpr int kStartingCoins = 2;
constexpr int kCoupCost = 7;
constexpr int kAssassinateCost = 3;
constexpr int kForceCoupThreshold = 10;
constexpr int kInfluencePerPlayer = 2;
constexpr int kExchangeDrawSlots = 2;

constexpr int kIncomeAction = 0;
constexpr int kForeignAidAction = 1;
constexpr int kCoupOffset = 2;
constexpr int kCoupCount = 4;
constexpr int kTaxAction = 6;
constexpr int kAssassinateOffset = 7;
constexpr int kAssassinateCount = 4;
constexpr int kStealOffset = 11;
constexpr int kStealCount = 4;
constexpr int kExchangeAction = 15;

constexpr int kChallengeAction = 16;
constexpr int kAllowAction = 17;

constexpr int kBlockDukeAction = 18;
constexpr int kBlockContessaAction = 19;
constexpr int kBlockAmbassadorAction = 20;
constexpr int kBlockCaptainAction = 21;
constexpr int kAllowNoBlockAction = 22;

constexpr int kRevealSlot0 = 23;
constexpr int kRevealSlot1 = 24;

constexpr int kLoseSlot0 = 25;
constexpr int kLoseSlot1 = 26;

constexpr int kReturnDuke = 27;
constexpr int kReturnAssassin = 28;
constexpr int kReturnCaptain = 29;
constexpr int kReturnAmbassador = 30;
constexpr int kReturnContessa = 31;

constexpr int kActionSpace = 32;

enum class CoupStage : std::int8_t {
  kDeclareAction = 0,
  kChallengeAction = 1,
  kResolveChallengeAction = 2,
  kChooseLoseInfluence = 3,
  kCounterAction = 4,
  kChallengeCounter = 5,
  kResolveChallengeCounter = 6,
  kChooseLoseInfluenceCounter = 7,
  kExchangeReturn1 = 8,
  kExchangeReturn2 = 9,
  kLoseInfluenceFromAction = 10,
};
constexpr int kStageCount = 11;

template <int NPlayers>
struct CoupConfig {
  static_assert(NPlayers >= 2 && NPlayers <= 4);
  static constexpr int kPlayers = NPlayers;
  // See coup_revival plan §"Encoder 设计". Per-player public 23: alive,
  // coins, 2× influence_slot{revealed + char-OH(5)}=12, role one-hot
  // (active/target/blocker/challenger)=4, signals(5).
  static constexpr int kPerPlayerPublicFeatures = 23;
  // Per-player private 22: 2× own influence char-OH (5+5=10) +
  // 2× exchange_drawn{occupied + char-OH(5)}=12. Self-block only on
  // perspective; other players' private blocks zero-filled by encoder.
  static constexpr int kPerPlayerPrivateFeatures = 22;
  static constexpr int kPerPlayerFeatures =
      kPerPlayerPublicFeatures + kPerPlayerPrivateFeatures;  // 45
  // Global: stage(11) + declared_action_type(7) + claimed_char(5) +
  // block_char(5) + pending_claimer relative-OH(N) + pending_challenged
  // (1) + deck_size/15(1) + ply/200(1) + revealed multiset per role(5)
  // = 36 + N.
  static constexpr int kGlobalFeatures = 36 + NPlayers;
  static constexpr int kFeatureDim = kPerPlayerFeatures * NPlayers + kGlobalFeatures;
  static constexpr int kPublicFeatureDim =
      kPerPlayerPublicFeatures * NPlayers + kGlobalFeatures;
  static constexpr int kPrivateFeatureDim =
      kPerPlayerPrivateFeatures * NPlayers;
};

template <int NPlayers>
struct CoupData {
  using Cfg = CoupConfig<NPlayers>;

  int current_player = 0;
  std::int8_t first_player = 0;
  int winner = -1;
  bool terminal = false;
  int ply = 0;
  CoupStage stage = CoupStage::kDeclareAction;

  std::array<std::array<CharId, kInfluencePerPlayer>, Cfg::kPlayers> influence{};
  std::array<std::array<bool, kInfluencePerPlayer>, Cfg::kPlayers> revealed{};
  std::array<int, Cfg::kPlayers> coins{};
  std::array<bool, Cfg::kPlayers> alive{};

  // Court deck as fixed-shape multiset count + size scalar. Multiset
  // shape lets the schema walker hash/serialize/randomize without
  // variable-length escape hatches; size is the public count exposed
  // to all viewers.
  std::array<std::int8_t, kCharacterCount> deck_count{};
  std::int8_t deck_size = 0;

  int active_player = 0;
  ActionId declared_action = -1;
  int action_target = -1;
  CharId claimed_character = -1;

  int challenger = -1;
  int challenge_loser = -1;
  bool action_challenged = false;
  bool action_challenge_succeeded = false;

  int blocker = -1;
  CharId block_character = -1;
  bool counter_challenged = false;
  bool counter_challenge_succeeded = false;

  int challenge_check_index = 0;

  // Exchange-drawn buffer. Sentinel -1 means "no card here". Rules
  // call viz::reveal_slot_to(active_player) when drawing and
  // viz::reset_to_base on return; the schema base is all_hidden.
  std::array<CharId, kExchangeDrawSlots> exchange_drawn{-1, -1};
  int exchange_held_count = 0;
};

template <int NPlayers>
struct CoupState final : public CloneableState<CoupState<NPlayers>> {
  using Cfg = CoupConfig<NPlayers>;
  CoupData<NPlayers> data;

  CoupState();

  // Visibility schema. Coup's hidden info partitions into:
  //   - all_public scalars/arrays: stage/coins/alive/revealed/etc.,
  //     and the multiset COUNT view (deck_size). Public to every
  //     perspective at all times.
  //   - owner_only_first_axis: influence[N][2]. base viz[p, *, p]=1.
  //     When a card is revealed by a challenge or lose-influence,
  //     rules call viz::reveal_slot(state, "influence", {p, s}) so
  //     every perspective sees the truth (not gated on a side flag).
  //   - all_hidden: exchange_drawn[2], deck_count[5]. Active player
  //     gets reveal_slot_to during exchange; deck_count is filled by
  //     the tracker's randomize_unseen for sim entries.
  static const viz::VisibilitySchema& schema();

  void reset_with_seed(std::uint64_t seed) override;
  StateHash64 state_hash() const override;
  void hash_field_slot(Hasher& h, const std::string& name,
                       const std::vector<int>& idx) const override;
  std::any read_field_slot(const std::string& name,
                           const std::vector<int>& idx) const override;
  void write_field_slot(const std::string& name,
                        const std::vector<int>& idx,
                        const std::any& value) override;
  void mask_field_slot(const std::string& name,
                       const std::vector<int>& idx) override;
  const viz::VisibilitySchema& schema_ref() const override { return schema(); }
  int current_player() const override;
  int first_player() const override;
  bool is_terminal() const override;
  int num_players() const override { return Cfg::kPlayers; }
  int winner() const override;
  bool is_turn_start() const override;
};

extern template struct CoupData<2>;
extern template struct CoupData<3>;
extern template struct CoupData<4>;
extern template struct CoupState<2>;
extern template struct CoupState<3>;
extern template struct CoupState<4>;

}  // namespace board_ai::coup
