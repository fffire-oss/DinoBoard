cmake_minimum_required(VERSION 3.19)
project(DinoBoard CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_subdirectory(engine)

# Games are listed in games/manifest.json — single source of truth shared
# with setup.py. To add a game, append an entry there; nothing else changes.
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/games/manifest.json" _games_manifest)
string(JSON _games_array GET "${_games_manifest}" games)
string(JSON _games_count LENGTH "${_games_array}")
math(EXPR _games_last "${_games_count} - 1")
set(_c_api_game_sources "")
foreach(idx RANGE 0 ${_games_last})
  string(JSON _game GET "${_games_array}" ${idx})
  string(JSON _game_id GET "${_game}" id)
  # Optional `enabled` field; default true if absent. enabled=false skips
  # the game from the build entirely (no compile, no register).
  set(_game_enabled "true")
  string(JSON _game_enabled ERROR_VARIABLE _err GET "${_game}" enabled)
  if(_err)
    set(_game_enabled "true")
  endif()
  if(_game_enabled STREQUAL "true" OR _game_enabled STREQUAL "1")
    add_subdirectory(games/${_game_id})
    string(JSON _game_sources GET "${_game}" sources)
    string(JSON _game_sources_count LENGTH "${_game_sources}")
    math(EXPR _game_sources_last "${_game_sources_count} - 1")
    foreach(src_idx RANGE 0 ${_game_sources_last})
      string(JSON _game_src GET "${_game_sources}" ${src_idx})
      list(APPEND _c_api_game_sources
           "${CMAKE_CURRENT_SOURCE_DIR}/games/${_game_id}/${_game_src}")
    endforeach()
  endif()
endforeach()

set(_c_api_engine_sources
  "${CMAKE_CURRENT_SOURCE_DIR}/engine/core/action_constraint.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/engine/core/mask_state_impl.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/engine/search/net_mcts.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/engine/search/tail_solver.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/engine/infer/onnx_policy_value_evaluator.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/engine/infer/onnx_belief_evaluator.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/engine/runtime/selfplay_runner.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/engine/runtime/arena_runner.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/engine/runtime/heuristic_runner.cpp"
)

add_library(dinoboard_c_api SHARED
  "${CMAKE_CURRENT_SOURCE_DIR}/bindings/c_api.cpp"
  ${_c_api_engine_sources}
  ${_c_api_game_sources}
)
target_include_directories(dinoboard_c_api PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
target_compile_definitions(dinoboard_c_api PRIVATE DINOBOARD_C_API_BUILD=1)
if(MSVC)
  target_compile_options(dinoboard_c_api PRIVATE /bigobj /EHsc /utf-8)
  target_compile_definitions(dinoboard_c_api PRIVATE
    NOMINMAX=1 WIN32_LEAN_AND_MEAN=1 _CRT_SECURE_NO_WARNINGS=1)
endif()

if(BOARD_AI_WITH_ONNX)
  target_compile_definitions(dinoboard_c_api PRIVATE BOARD_AI_WITH_ONNX=1)
  if(NOT BOARD_AI_ONNXRUNTIME_ROOT)
    if(WIN32)
      file(GLOB _ort_candidates
           "${CMAKE_CURRENT_SOURCE_DIR}/third_party/onnxruntime-win-*")
    elseif(UNIX AND NOT APPLE)
      file(GLOB _ort_candidates
           "${CMAKE_CURRENT_SOURCE_DIR}/third_party/onnxruntime-linux-*")
    endif()
    list(LENGTH _ort_candidates _ort_candidate_count)
    if(_ort_candidate_count GREATER 0)
      list(GET _ort_candidates 0 BOARD_AI_ONNXRUNTIME_ROOT)
    endif()
  endif()
  if(NOT BOARD_AI_ONNXRUNTIME_ROOT)
    message(FATAL_ERROR
      "BOARD_AI_WITH_ONNX=ON requires BOARD_AI_ONNXRUNTIME_ROOT or a bundled third_party/onnxruntime-* directory")
  endif()
  target_include_directories(dinoboard_c_api PRIVATE
    "${BOARD_AI_ONNXRUNTIME_ROOT}/include"
    "${BOARD_AI_ONNXRUNTIME_ROOT}/include/onnxruntime")
  target_link_directories(dinoboard_c_api PRIVATE
    "${BOARD_AI_ONNXRUNTIME_ROOT}/lib")
  target_link_libraries(dinoboard_c_api PRIVATE onnxruntime)
  if(WIN32)
    add_custom_command(TARGET dinoboard_c_api POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${BOARD_AI_ONNXRUNTIME_ROOT}/lib/onnxruntime.dll"
        "$<TARGET_FILE_DIR:dinoboard_c_api>/onnxruntime.dll")
  endif()
endif()
