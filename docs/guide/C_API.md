# DinoBoard C ABI

DinoBoard's Python API is backed by pybind11. The C ABI is a native interface
for callers that do not want to embed Python, such as a Rust local advisor.

The C ABI is implemented in:

```text
bindings/c_api.h
bindings/c_api.cpp
```

It exports C-style functions from a C++ implementation. The engine still uses
DinoBoard's native C++ rules, feature encoders, MCTS, tail solver, and ONNX
Runtime evaluator.

## Current Scope

The first ABI surface is intentionally small:

- list available games
- query game metadata
- describe an action id
- create/destroy a native session
- run a non-mutating MCTS decision and return JSON stats

This is enough for native smoke tests and GemHUD's first Rust integration. The
observation/snapshot API should be added next before relying on it for exact
live external-game state tracking.

## Windows Build

From a PowerShell session:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\build_c_api_windows.ps1 `
  -BuildDir build-capi `
  -OnnxRuntimeRoot third_party\onnxruntime-win-x64-1.17.3
```

Output:

```text
build-capi\dinoboard_c_api.dll
build-capi\dinoboard_c_api.lib
build-capi\onnxruntime.dll
```

The script tries to import the Visual Studio x64 build environment when the
current shell has `cl.exe` but lacks standard library include paths.

## Rust Smoke Test

```powershell
cd tests\rust_c_api_smoke
$env:CARGO_TARGET_DIR = "D:\codex\Haro-DinoBoard\tests\rust_c_api_smoke\target"
cargo run -- `
  D:\codex\Haro-DinoBoard\build-capi\dinoboard_c_api.dll `
  D:\codex\Haro-DinoBoard\games\splendor\model\splendor_2p.onnx
```

The smoke test loads the DLL with `LoadLibraryA`, resolves exported functions
with `GetProcAddress`, creates a `splendor_2p` session, and verifies that MCTS
returns `action_id`, `root_actions`, `root_action_visits`, and `action_values`.

## Memory Ownership

Every exported function returning `char*` transfers ownership to the caller.
Callers must release it with:

```c
dinoboard_string_free(ptr);
```

Session handles returned by `dinoboard_session_create` must be released with:

```c
dinoboard_session_destroy(session);
```

## Next ABI Additions

For exact external live-state integration, add:

- `dinoboard_session_apply_initial_observation_json`
- `dinoboard_session_apply_observation_json`
- `dinoboard_session_apply_public_snapshot_json`
- schema-driven JSON-to-`AnyMap` parsing

Those functions should mirror the existing Python AI API observation contract.
