#include "quoridor_state.h"

#include "../../engine/core/viz_runtime.h"

namespace board_ai::quoridor {

const viz::VisibilitySchema& QuoridorState::schema() {
  // Built once on first use. Quoridor is fully public — pawn positions,
  // wall placements and remaining-wall counts are all visible to every
  // viewer. Same all_public-only template as tictactoe.
  static const viz::VisibilitySchema s = []() {
    viz::VisibilitySchema schema;
    schema.n_players = kPlayers;
    viz::declare_field(schema, "current_player",
                       viz::all_public({}, kPlayers));
    viz::declare_field(schema, "winner", viz::all_public({}, kPlayers));
    viz::declare_field(schema, "terminal", viz::all_public({}, kPlayers));
    viz::declare_field(schema, "move_count", viz::all_public({}, kPlayers));
    viz::declare_field(schema, "scores",
                       viz::all_public({kPlayers}, kPlayers));
    viz::declare_field(schema, "pawn_row",
                       viz::all_public({kPlayers}, kPlayers));
    viz::declare_field(schema, "pawn_col",
                       viz::all_public({kPlayers}, kPlayers));
    viz::declare_field(schema, "walls_remaining",
                       viz::all_public({kPlayers}, kPlayers));
    viz::declare_field(schema, "h_walls",
                       viz::all_public({kWallSlots}, kPlayers));
    viz::declare_field(schema, "v_walls",
                       viz::all_public({kWallSlots}, kPlayers));
    return schema;
  }();
  return s;
}

QuoridorState::QuoridorState() {
  reset_with_seed(0xC0FFEE1234ULL);
}

void QuoridorState::reset_with_seed(std::uint64_t seed) {
  IGameState::reset_with_seed_base(seed);
  current_player_ = 0;
  winner_ = -1;
  terminal = false;
  move_count = 0;
  scores = {0, 0};
  pawn_row = {0, static_cast<std::int8_t>(kBoardSize - 1)};
  pawn_col = {4, 4};
  walls_remaining = {kMaxWallsPerPlayer, kMaxWallsPerPlayer};
  h_walls.fill(0);
  v_walls.fill(0);
  undo_stack.clear();
  viz::init_viz(*this, schema());
}

StateHash64 QuoridorState::state_hash(bool include_hidden_rng) const {
  std::size_t h = 0;

  hash_combine(h,static_cast<std::size_t>(current_player_));
  hash_combine(h,static_cast<std::size_t>(winner_ + 1));
  hash_combine(h,static_cast<std::size_t>(terminal ? 1 : 0));
  hash_combine(h,static_cast<std::size_t>(move_count));
  hash_combine(h,static_cast<std::size_t>(scores[0] + 2));
  hash_combine(h,static_cast<std::size_t>(scores[1] + 2));
  for (int p = 0; p < kPlayers; ++p) {
    hash_combine(h,static_cast<std::size_t>(pawn_row[static_cast<size_t>(p)] + 1));
    hash_combine(h,static_cast<std::size_t>(pawn_col[static_cast<size_t>(p)] + 1));
    hash_combine(h,static_cast<std::size_t>(walls_remaining[static_cast<size_t>(p)] + 1));
  }
  for (std::uint8_t w : h_walls) hash_combine(h,static_cast<std::size_t>(w));
  for (std::uint8_t w : v_walls) hash_combine(h,static_cast<std::size_t>(w));
  if (include_hidden_rng) {
    hash_combine(h,static_cast<std::size_t>(rng_salt_));
  }
  return static_cast<StateHash64>(h);
}

void QuoridorState::hash_public_fields(Hasher& h) const {
  // Quoridor is fully public: board, walls, pawns, scores all visible.
  h.add(current_player_);
  h.add(winner_ + 1);
  h.add(terminal ? 1 : 0);
  h.add(move_count);
  h.add(scores[0] + 2);
  h.add(scores[1] + 2);
  for (int p = 0; p < kPlayers; ++p) {
    h.add(pawn_row[static_cast<size_t>(p)] + 1);
    h.add(pawn_col[static_cast<size_t>(p)] + 1);
    h.add(walls_remaining[static_cast<size_t>(p)] + 1);
  }
  for (std::uint8_t w : h_walls) h.add(w);
  for (std::uint8_t w : v_walls) h.add(w);
}

void QuoridorState::hash_private_fields(int /*player*/, Hasher& /*h*/) const {
  // Quoridor has no private info.
}

}  // namespace board_ai::quoridor
