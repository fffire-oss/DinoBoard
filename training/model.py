"""Neural network model for policy-value prediction."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch
import torch.nn as nn


class PVNet(nn.Module):
    def __init__(self, input_dim: int, policy_dim: int, hidden_layers: list[int],
                 auxiliary_score: bool = False, num_players: int = 2):
        super().__init__()
        self.num_players = num_players
        layers: list[nn.Module] = []
        prev = input_dim
        for h in hidden_layers:
            layers.extend([nn.Linear(prev, h), nn.ReLU()])
            prev = h
        self.backbone = nn.Sequential(*layers)
        self.policy_head = nn.Linear(prev, policy_dim)
        self.value_head = nn.Sequential(nn.Linear(prev, num_players), nn.Tanh())
        self.has_score_head = auxiliary_score
        if auxiliary_score:
            self.score_head = nn.Sequential(nn.Linear(prev, 1), nn.Tanh())

    def forward(self, x: torch.Tensor):
        h = self.backbone(x)
        if self.has_score_head:
            return self.policy_head(h), self.value_head(h), self.score_head(h)
        return self.policy_head(h), self.value_head(h)


def create_model_from_config(game_config: dict) -> PVNet:
    input_dim = game_config["feature_dim"]
    policy_dim = game_config["action_space"]
    hidden_layers = game_config.get("network", {}).get("hidden_layers", [256, 256])
    auxiliary_score = game_config.get("training", {}).get("auxiliary_score", False)
    if "num_players" not in game_config:
        raise KeyError(
            "game_config missing 'num_players'. Inject from engine.game_metadata(game_id) "
            "before calling create_model_from_config — JSON 'players.max' is not authoritative."
        )
    num_players = game_config["num_players"]
    return PVNet(input_dim, policy_dim, hidden_layers,
                 auxiliary_score=auxiliary_score, num_players=num_players)


class BeliefNet(nn.Module):
    """Belief-network: MLP with optional BatchNorm.

    Input: flat feature vector (size = belief_feature_dim).
    Output: flat logits (size = (N-1) * K), reshaped client-side to (N-1, K).
    """

    def __init__(self, input_dim: int, output_dim: int,
                 hidden_layers: Sequence[int],
                 batch_norm: bool = True):
        super().__init__()
        layers: list[nn.Module] = []
        prev = input_dim
        for h in hidden_layers:
            layers.append(nn.Linear(prev, h))
            if batch_norm:
                layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def create_belief_model_from_config(belief_config: dict, *,
                                    input_dim: int,
                                    output_dim: int) -> BeliefNet:
    hidden = belief_config.get("architecture", [256, 256, 128])
    batch_norm = belief_config.get("batch_norm", True)
    return BeliefNet(input_dim, output_dim, hidden, batch_norm=batch_norm)


def load_belief_onnx_into(net: BeliefNet, path: Path) -> None:
    """Load weights from an exported belief ONNX into a fresh BeliefNet.

    `torch.onnx.export` keeps PyTorch parameter names verbatim in the
    initializer list (e.g. `net.0.weight`), so we just numpy → tensor copy
    by name. BN running_mean / running_var are tracked buffers and arrive
    under the same `net.<idx>.running_*` names, so the full state_dict round
    trips. Shapes must match the freshly created BeliefNet — a mismatch is
    a bug and we raise.
    """
    import onnx as _onnx
    model = _onnx.load(str(path))
    onnx_state = {
        ini.name: torch.from_numpy(_onnx.numpy_helper.to_array(ini).copy())
        for ini in model.graph.initializer
    }
    net_state = net.state_dict()
    # BatchNorm's `num_batches_tracked` is not exported by torch.onnx — it's
    # a pure step counter, irrelevant to inference. Carry over net's value
    # (zero in a fresh module) so load_state_dict is happy.
    for k in net_state:
        if k.endswith("num_batches_tracked") and k not in onnx_state:
            onnx_state[k] = net_state[k]
    missing = sorted(set(net_state.keys()) - set(onnx_state.keys()))
    extra = sorted(set(onnx_state.keys()) - set(net_state.keys()))
    if missing or extra:
        raise ValueError(
            f"belief warmstart ONNX {path} parameter set differs from "
            f"freshly built BeliefNet: missing={missing}, extra={extra}")
    for k, v in net_state.items():
        if v.shape != onnx_state[k].shape:
            raise ValueError(
                f"belief warmstart shape mismatch on {k!r}: net={tuple(v.shape)} "
                f"onnx={tuple(onnx_state[k].shape)}")
    net.load_state_dict(onnx_state)


def export_belief_onnx(net: BeliefNet, path: Path, input_dim: int) -> str:
    net.eval()
    path.parent.mkdir(parents=True, exist_ok=True)
    # BN with batch=1 in eval mode reads running stats — safe. The exported
    # graph keeps a batch axis for runtime convenience even though the C++
    # evaluator always feeds batch=1.
    dummy = torch.zeros((1, input_dim), dtype=torch.float32)
    torch.onnx.export(
        net, dummy, str(path),
        input_names=["features"],
        output_names=["logits"],
        dynamic_axes={"features": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=13,
        dynamo=False,
    )
    return str(path)


def export_onnx(net: PVNet, path: Path, input_dim: int) -> str:
    net.eval()
    path.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.zeros((1, input_dim), dtype=torch.float32)

    output_names = ["policy", "value"]
    dynamic_axes = {
        "features": {0: "batch"},
        "policy": {0: "batch"},
        "value": {0: "batch"},
    }
    if net.has_score_head:
        output_names.append("score")
        dynamic_axes["score"] = {0: "batch"}

    torch.onnx.export(
        net, dummy, str(path),
        input_names=["features"],
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=13,
        dynamo=False,
    )
    return str(path)
