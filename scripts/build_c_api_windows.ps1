param(
    [string]$BuildDir = "build-capi",
    [string]$OnnxRuntimeRoot = "",
    [string]$Configuration = "Release"
)

$ErrorActionPreference = "Stop"

$Root = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $Root

function Import-VcVars {
    param([string]$VcVarsPath)
    if (-not (Test-Path $VcVarsPath)) {
        return $false
    }
    $cmd = "`"$VcVarsPath`" >nul && set"
    $lines = & cmd.exe /s /c $cmd
    if ($LASTEXITCODE -ne 0) {
        return $false
    }
    foreach ($line in $lines) {
        $idx = $line.IndexOf("=")
        if ($idx -le 0) {
            continue
        }
        $name = $line.Substring(0, $idx)
        $value = $line.Substring($idx + 1)
        Set-Item -Path "Env:$name" -Value $value
    }
    return $true
}

if (-not $env:INCLUDE -or $env:INCLUDE -notmatch "MSVC") {
    $vcCandidates = @(
        "D:\visual studio\VC\Auxiliary\Build\vcvars64.bat",
        "C:\Program Files\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat",
        "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat",
        "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
    )
    foreach ($candidate in $vcCandidates) {
        if (Import-VcVars $candidate) {
            break
        }
    }
}

if (-not (Get-Command cl.exe -ErrorAction SilentlyContinue)) {
    throw "cl.exe was not found. Run this script from an x64 Native Tools prompt for Visual Studio."
}

if (-not $OnnxRuntimeRoot) {
    $candidate = Get-ChildItem -Path (Join-Path $Root "third_party") -Directory -Filter "onnxruntime-win-*" |
        Sort-Object Name |
        Select-Object -First 1
    if (-not $candidate) {
        throw "ONNX Runtime was not found under third_party/onnxruntime-win-*; pass -OnnxRuntimeRoot."
    }
    $OnnxRuntimeRoot = $candidate.FullName
}

$OnnxRuntimeRoot = Resolve-Path $OnnxRuntimeRoot
$OutDir = Join-Path $Root $BuildDir
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

$manifest = Get-Content (Join-Path $Root "games/manifest.json") -Raw | ConvertFrom-Json
$gameSources = @()
foreach ($game in $manifest.games) {
    if ($game.PSObject.Properties.Name -contains "enabled" -and -not $game.enabled) {
        continue
    }
    foreach ($src in $game.sources) {
        $gameSources += (Join-Path $Root ("games/{0}/{1}" -f $game.id, $src))
    }
}

$sources = @(
    (Join-Path $Root "bindings/c_api.cpp"),
    (Join-Path $Root "engine/core/action_constraint.cpp"),
    (Join-Path $Root "engine/core/mask_state_impl.cpp"),
    (Join-Path $Root "engine/search/net_mcts.cpp"),
    (Join-Path $Root "engine/search/tail_solver.cpp"),
    (Join-Path $Root "engine/infer/onnx_policy_value_evaluator.cpp"),
    (Join-Path $Root "engine/infer/onnx_belief_evaluator.cpp"),
    (Join-Path $Root "engine/runtime/selfplay_runner.cpp"),
    (Join-Path $Root "engine/runtime/arena_runner.cpp"),
    (Join-Path $Root "engine/runtime/heuristic_runner.cpp")
) + $gameSources

$compileArgs = @(
    "/nologo",
    "/LD",
    "/std:c++17",
    "/EHsc",
    "/bigobj",
    "/utf-8",
    "/D", "DINOBOARD_C_API_BUILD=1",
    "/D", "BOARD_AI_WITH_ONNX=1",
    "/D", "NOMINMAX=1",
    "/D", "WIN32_LEAN_AND_MEAN=1",
    "/D", "_CRT_SECURE_NO_WARNINGS=1",
    "/I", $Root,
    "/I", (Join-Path $OnnxRuntimeRoot "include")
)

if ($Configuration -eq "Release") {
    $compileArgs += @("/O2")
} else {
    $compileArgs += @("/Od", "/Zi")
}

$linkArgs = @(
    "/link",
    "/LIBPATH:$(Join-Path $OnnxRuntimeRoot "lib")",
    "onnxruntime.lib",
    "/OUT:$(Join-Path $OutDir "dinoboard_c_api.dll")",
    "/IMPLIB:$(Join-Path $OutDir "dinoboard_c_api.lib")"
)

& cl.exe @compileArgs @sources @linkArgs
if ($LASTEXITCODE -ne 0) {
    throw "cl.exe failed with exit code $LASTEXITCODE"
}
Copy-Item -Force (Join-Path $OnnxRuntimeRoot "lib/onnxruntime.dll") $OutDir
Write-Host "Built $OutDir\dinoboard_c_api.dll"
