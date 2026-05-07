"""Legacy scalar value head compatibility (2-player only).

Older 2p models export a `[1, 1]` value head where the single output is the
perspective player's value in [-1, 1]. The full 2p AI chain — selfplay, eval,
web gameplay (winrate pill), replay analysis, smart hints — must keep loading
these models without retraining. Compatibility lives in the
`value_len == 1 && num_players == 2` branch of
`engine/infer/onnx_policy_value_evaluator.cpp`, which expands the scalar `v`
into `(v_perspective, -v_opponent)` so downstream code (MCTS leaf backup,
`root_values`/`action_values` bindings, `pipeline.py`) is dimension-agnostic.

This compatibility does NOT extend to 3p/4p — scalar + N>2 must throw, never
silently broadcast (per "no silent degradation"). These tests pin both
behaviors as a regression boundary; deleting the scalar branch silently
breaks every shipped 2p `model_best.onnx`.
"""
from __future__ import annotations

import sys
from pathlib import Path

import dinoboard_engine
import pytest
import torch
import torch.nn as nn

from conftest import load_game_config

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_PLATFORM_PATH = str(_PROJECT_ROOT / "platform")
if _PLATFORM_PATH not in sys.path:
    sys.path.insert(0, _PLATFORM_PATH)
from game_service.pipeline import _human_wr_from_stats, _human_wr_for_action  # noqa: E402


class _ScalarPVNet(nn.Module):
    """Mimics the legacy pre-N-dim model: value head outputs a single scalar
    in [-1, 1] = perspective player's expected value."""

    def __init__(self, input_dim: int, policy_dim: int):
        super().__init__()
        self.backbone = nn.Sequential(nn.Linear(input_dim, 32), nn.ReLU())
        self.policy_head = nn.Linear(32, policy_dim)
        self.value_head = nn.Sequential(nn.Linear(32, 1), nn.Tanh())

    def forward(self, x: torch.Tensor):
        h = self.backbone(x)
        return self.policy_head(h), self.value_head(h)


def _export_scalar_onnx(net: _ScalarPVNet, path: Path, input_dim: int) -> str:
    net.eval()
    path.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.zeros((1, input_dim), dtype=torch.float32)
    torch.onnx.export(
        net, dummy, str(path),
        input_names=["features"],
        output_names=["policy", "value"],
        dynamic_axes={
            "features": {0: "batch"},
            "policy": {0: "batch"},
            "value": {0: "batch"},
        },
        opset_version=13,
        dynamo=False,
    )
    return str(path)


def _make_scalar_model(game_id: str, tmp_path: Path) -> str:
    cfg = load_game_config(game_id)
    net = _ScalarPVNet(cfg["feature_dim"], cfg["action_space"])
    path = tmp_path / f"{game_id}_scalar.onnx"
    return _export_scalar_onnx(net, path, cfg["feature_dim"])


# Two 2p games chosen to cover deterministic + stochastic paths through the
# evaluator. Adding more is fine but this is enough to pin the branch.
SCALAR_2P_GAMES = ["tictactoe", "quoridor"]


@pytest.mark.parametrize("game_id", SCALAR_2P_GAMES)
def test_scalar_value_head_loads_and_runs(game_id, tmp_path):
    """Scalar [1,1] value head: GameSession.get_ai_action must succeed and
    return a length-2 root_values vector that satisfies zero-sum."""
    model_path = _make_scalar_model(game_id, tmp_path)
    gs = dinoboard_engine.GameSession(game_id, seed=42, model_path=model_path)
    result = gs.get_ai_action(simulations=20, temperature=0.0)
    legal = gs.get_all_legal_actions()
    assert result["action"] in legal

    stats = result["stats"]
    rv = stats["root_values"]
    assert len(rv) == 2, f"root_values must be length 2 for 2p, got {len(rv)}"
    # Zero-sum: scalar branch sets v[p]=v, v[1-p]=-v. After search expansion
    # the values move with backup but must remain numerically zero-sum.
    assert abs(rv[0] + rv[1]) < 1e-5, f"zero-sum violated: {rv}"


@pytest.mark.parametrize("game_id", SCALAR_2P_GAMES)
def test_scalar_value_head_pipeline_analysis(game_id, tmp_path):
    """Drop-score / winrate-pill consumers in platform/game_service/pipeline.py
    must work unchanged on a scalar-value-head model."""
    model_path = _make_scalar_model(game_id, tmp_path)
    gs = dinoboard_engine.GameSession(game_id, seed=7, model_path=model_path)
    stats = gs.get_ai_action(simulations=30, temperature=0.0)["stats"]

    wr0 = _human_wr_from_stats(stats, 0)
    wr1 = _human_wr_from_stats(stats, 1)
    assert 0.0 <= wr0 <= 1.0
    assert 0.0 <= wr1 <= 1.0
    # Zero-sum scalar produces complementary winrates.
    assert abs((wr0 + wr1) - 1.0) < 1e-4, f"winrates must sum to 1: {wr0}+{wr1}"

    # action_values must be populated for at least one root action; the wr
    # accessor must return a finite probability in [0, 1].
    explored = [a for a in gs.get_all_legal_actions() if a in stats["action_values"]]
    assert explored, "expected at least one explored action in action_values"
    a = explored[0]
    wr_a0 = _human_wr_for_action(stats, a, 0)
    assert wr_a0 is not None and 0.0 <= wr_a0 <= 1.0


def test_scalar_value_head_selfplay(tmp_path):
    """Scalar value head must be usable as the evaluator in a full selfplay
    episode (the same code path eval/arena uses)."""
    model_path = _make_scalar_model("tictactoe", tmp_path)
    ep = dinoboard_engine.run_selfplay_episode(
        game_id="tictactoe", seed=42, model_path=model_path,
        simulations=10, max_game_plies=9,
    )
    assert ep["total_plies"] > 0
    assert len(ep["samples"]) > 0


def test_scalar_value_head_arena(tmp_path):
    """Arena (eval pipeline) must accept a scalar-value-head model on both
    sides without raising."""
    model_path = _make_scalar_model("tictactoe", tmp_path)
    result = dinoboard_engine.run_arena_match(
        game_id="tictactoe",
        seed=42,
        model_paths=[model_path, model_path],
        simulations_list=[10, 10],
        max_game_plies=9,
    )
    # Result schema is game-agnostic; just confirm the call returned without
    # the evaluator throwing on the scalar shape.
    assert result["total_plies"] > 0


def test_scalar_value_head_rejected_for_3p(tmp_path):
    """3p+ scalar must throw — there is no zero-sum decomposition that pins
    individual seats. Preserves "no silent degradation" doctrine."""
    model_path = _make_scalar_model("loveletter_3p", tmp_path)
    gs = dinoboard_engine.GameSession("loveletter_3p", seed=42, model_path=model_path)
    with pytest.raises(Exception, match="value output length"):
        gs.get_ai_action(simulations=5, temperature=0.0)
