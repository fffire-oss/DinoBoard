#pragma once

#ifdef _WIN32
  #ifdef DINOBOARD_C_API_BUILD
    #define DINOBOARD_C_API __declspec(dllexport)
  #else
    #define DINOBOARD_C_API __declspec(dllimport)
  #endif
#else
  #define DINOBOARD_C_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

// Every function that returns char* transfers ownership to the caller. Free it
// with dinoboard_string_free().
DINOBOARD_C_API void dinoboard_string_free(char* value);

DINOBOARD_C_API const char* dinoboard_abi_version(void);
DINOBOARD_C_API char* dinoboard_available_games_json(void);
DINOBOARD_C_API char* dinoboard_game_metadata_json(const char* game_id);
DINOBOARD_C_API char* dinoboard_action_info_json(const char* game_id, int action_id);

// Creates a native C++ session. The returned opaque pointer owns DinoBoard
// rules, state, feature encoder, MCTS evaluator, and the loaded ONNX model.
// On failure returns null and writes a heap string to out_error when provided.
DINOBOARD_C_API void* dinoboard_session_create(
    const char* game_id,
    const char* model_path,
    unsigned long long seed,
    int use_action_filter,
    char** out_error);

DINOBOARD_C_API void dinoboard_session_destroy(void* session);

// Returns a JSON object with action_id, action_info, stats.root_actions,
// stats.root_action_visits, and stats.action_values. This call is
// non-mutating: callers must apply/observe actions through a future session
// API before asking for another live decision.
DINOBOARD_C_API char* dinoboard_session_decide_json(
    void* session,
    int simulations,
    double temperature,
    int cover_root_edges,
    char** out_error);

#ifdef __cplusplus
}
#endif
