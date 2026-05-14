"""Convert a legacy Splendor `.pt` checkpoint (294-dim input, scalar value head,
`games.splendor.train.plugin` key naming) into a checkpoint that
`training/cli.py --init-from` can load into the current PVNet
(`training/model.py`).

Three transformations are applied to `model_state` and the result is wrapped
in the new framework's checkpoint shape:

1. **Input permutation**: `backbone.0.weight` is `[hidden, 294]` in the
   reference layout the legacy network was trained against. The current
   encoder emits 295 dims with a different field order. Apply the same
   index permutation as `scripts/fix_splendor_2p_feature_order.py` so that
   each output unit consumes the right input feature; column 255
   (`first_player` bit) is zero-initialised because the legacy network
   never saw it.

2. **Value-head scalar → N=2 mirror**: legacy `value.0.weight` is
   `[1, prev]` and is interpreted as the perspective player's value. The
   new framework expects `[num_players, prev]`. For a 2-player zero-sum
   game `tanh` is odd, so reflecting `(W, b) -> ([W; -W], [b; -b])` makes
   `forward(x) = [tanh(z), -tanh(z)] = [tanh(z), tanh(-z)]`, which is
   exactly the canonical (v, -v) zero-sum decomposition the engine's
   ONNX evaluator already applies to legacy scalar models. The new
   network is therefore numerically equivalent at the value head and can
   continue training without the perspective seam re-learning.

3. **Key rename**: `policy.*` -> `policy_head.*`, `value.0.*` ->
   `value_head.0.*`. Backbone keys are already aligned (legacy plugin
   used the same `nn.Sequential(Linear, ReLU, ...)` layout).

The legacy `optimizer_state` is **dropped**, not migrated:
- Adam moments are keyed by parameter id (`state[0]`, `state[1]`, ...),
  not parameter name, so the moments would land on the wrong parameters
  after we changed the input width and value-head shape.
- Re-initialising the optimiser is the standard practice for any
  weight-init-from checkpoint that changes architecture.

Usage:
    python3 scripts/convert_splendor_legacy_pt.py \
        --in  games/splendor/model/latest_step_13500.pt \
        --out games/splendor/model/latest_step_13500.converted.pt

The script never overwrites the input.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch


DEFAULT_IN = Path("games/splendor/model/latest_step_13500.pt")
DEFAULT_OUT = Path("games/splendor/model/latest_step_13500.converted.pt")


def feature_permutation() -> np.ndarray:
    """Same permutation as scripts/fix_splendor_2p_feature_order.py.

    perm[j] = the legacy column that should become column j in the new
    encoder's layout. -1 marks columns that have no legacy counterpart and
    must be zero-initialised (only column 255 — the first_player bit).
    """
    perm = np.empty(295, dtype=np.int64)
    perm[0:210] = np.arange(0, 210)        # bank/stats/nobles/tableau identity
    perm[210:249] = np.arange(249, 288)    # our opp_reserved -> ref opp_reserved
    perm[249:255] = np.arange(288, 294)    # our metadata -> ref metadata
    perm[255] = -1                          # first_player bit -> zero column
    perm[256:295] = np.arange(210, 249)    # our own_reserved -> ref my_reserved
    return perm


def _expand_input_layer(W: torch.Tensor) -> torch.Tensor:
    """Expand the first Linear's weight from [hidden, 294] to [hidden, 295]
    by permuting columns. The new column 255 (first_player bit) is sampled
    from a small Gaussian whose std matches the per-column std of the
    legacy weights — picking 0 would make every hidden unit's first-step
    gradient on that bit a pure function of the unit's downstream Jacobian,
    which is fine in theory but trains slower in practice; matching the
    existing column scale starts learning from the same statistical regime
    as the rest of the network."""
    if W.dim() != 2 or W.shape[1] != 294:
        raise SystemExit(
            f"backbone.0.weight has shape {tuple(W.shape)}, expected (*, 294). "
            "Has this checkpoint already been converted?")
    perm = feature_permutation()
    out = torch.zeros(W.shape[0], 295, dtype=W.dtype)
    for j, src in enumerate(perm):
        if src >= 0:
            out[:, j] = W[:, int(src)]
    col_std = float(W.std().item())
    g = torch.Generator().manual_seed(20260514)
    out[:, 255] = torch.randn(W.shape[0], generator=g, dtype=W.dtype) * col_std
    return out


def _mirror_value_head(W: torch.Tensor, b: torch.Tensor):
    """Scalar (1-row) head -> 2-row mirror head for a 2-player zero-sum game.

    Row 0 carries the original value; row 1 carries its negation. Combined
    with tanh (odd), this produces (v, -v) at every input — equivalent to
    the legacy scalar interpretation under perspective-relative semantics.
    """
    if W.shape[0] != 1 or b.shape[0] != 1:
        raise SystemExit(
            f"value head shapes {tuple(W.shape)}, {tuple(b.shape)}; expected "
            "(1, *) and (1,). Already converted or non-scalar legacy?")
    W_new = torch.cat([W, -W], dim=0)
    b_new = torch.cat([b, -b], dim=0)
    return W_new, b_new


def convert(in_path: Path, out_path: Path) -> None:
    if not in_path.exists():
        raise SystemExit(f"input checkpoint missing: {in_path}")
    if in_path.resolve() == out_path.resolve():
        raise SystemExit(
            f"refusing to overwrite input: --in and --out resolve to the same "
            f"path ({in_path})")

    ckpt = torch.load(str(in_path), map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict) or "model_state" not in ckpt:
        raise SystemExit(
            f"unexpected checkpoint shape (top-level keys: "
            f"{list(ckpt.keys()) if isinstance(ckpt, dict) else type(ckpt)}). "
            "Expected legacy plugin format with 'model_state' key.")

    legacy = ckpt["model_state"]

    expected = {
        "backbone.0.weight", "backbone.0.bias",
        "backbone.2.weight", "backbone.2.bias",
        "backbone.4.weight", "backbone.4.bias",
        "backbone.6.weight", "backbone.6.bias",
        "policy.weight", "policy.bias",
        "value.0.weight", "value.0.bias",
    }
    missing = expected - set(legacy.keys())
    extra = set(legacy.keys()) - expected
    if missing:
        raise SystemExit(f"legacy state_dict missing expected keys: {sorted(missing)}")
    if extra:
        # Not fatal — log so the user notices anything new.
        print(f"  note: dropping unexpected legacy keys: {sorted(extra)}", file=sys.stderr)

    new_state = {}

    new_state["backbone.0.weight"] = _expand_input_layer(legacy["backbone.0.weight"])
    new_state["backbone.0.bias"] = legacy["backbone.0.bias"].clone()

    for layer_idx in (2, 4, 6):
        new_state[f"backbone.{layer_idx}.weight"] = legacy[f"backbone.{layer_idx}.weight"].clone()
        new_state[f"backbone.{layer_idx}.bias"] = legacy[f"backbone.{layer_idx}.bias"].clone()

    new_state["policy_head.weight"] = legacy["policy.weight"].clone()
    new_state["policy_head.bias"] = legacy["policy.bias"].clone()

    vw, vb = _mirror_value_head(legacy["value.0.weight"], legacy["value.0.bias"])
    new_state["value_head.0.weight"] = vw
    new_state["value_head.0.bias"] = vb

    out_ckpt = {
        "model_state_dict": new_state,
        "optimizer_state_dict": None,
        "step": 0,
        "_converted_from": str(in_path),
        "_legacy_meta": {
            "input_dim": ckpt.get("input_dim"),
            "policy_dim": ckpt.get("policy_dim"),
            "hidden": ckpt.get("hidden"),
            "mlp_layers": ckpt.get("mlp_layers"),
        },
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out_ckpt, str(out_path))
    print(f"wrote converted checkpoint to {out_path}")
    print(f"  backbone.0.weight: [hidden, 294] -> [hidden, 295] (column 255 small-Gaussian, std=col_std)")
    print(f"  policy.* -> policy_head.*")
    print(f"  value.0.* (scalar) -> value_head.0.* (mirror, num_players=2)")
    print(f"  optimizer_state dropped (Adam moments do not survive shape/key changes)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--in", dest="in_path", type=Path, default=DEFAULT_IN,
                    help=f"legacy plugin checkpoint (default: {DEFAULT_IN})")
    ap.add_argument("--out", dest="out_path", type=Path, default=DEFAULT_OUT,
                    help=f"output checkpoint (default: {DEFAULT_OUT})")
    args = ap.parse_args()
    convert(args.in_path, args.out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
