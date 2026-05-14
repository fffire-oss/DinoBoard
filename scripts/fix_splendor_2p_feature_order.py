"""Adapt a Splendor-2p ONNX trained against the reference-project 294-dim
feature layout into one that ingests our current 295-dim layout.

Background. Our encoder's feature order is a permutation of the reference
order plus one extra `first_player` bit at index 255 — not "ref order
padded at the end." A naive load (or a 295→294 Slice that drops the last
dim) feeds garbage into the network because columns past index 209 are
shuffled:

    our 0..209   = ref 0..209        (bank + stats + nobles + tableau)  OK
    our 210..248 = ref opp_reserved → trained weights expect my_reserved
    our 249..254 = ref metadata     → weights expect opp_reserved
    our 255      = first_player bit → weights expect opp_reserved
    our 256..293 = own_reserved     → weights expect metadata

This script supports two input shapes:

  (A) Bare reference-trained graph: features=[*, 294], no Slice. The
      first Gemm consumes 'features' directly with weights [512, 294].
      We expand the weight to [512, 295] via the permutation and widen
      the input shape to 295.
  (B) Adapter graph: features=[*, 295] with a Slice node that drops the
      last dim before the first Gemm (weights still [512, 294]). We
      remove the Slice and permute the weight as in (A).

CLI:
    python3 scripts/fix_splendor_2p_feature_order.py \
        --in  games/splendor/model/best_model.onnx \
        --out games/splendor/model/splendor_2p_from_best.onnx

Defaults match this most common case: best_model.onnx → splendor_2p_from_best.onnx.
The script never overwrites the input model.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper


DEFAULT_IN = Path("games/splendor/model/best_model.onnx")
DEFAULT_OUT = Path("games/splendor/model/splendor_2p_from_best.onnx")


def permutation() -> np.ndarray:
    """Our index j -> reference index. -1 means "zero column"."""
    perm = np.empty(295, dtype=np.int64)
    perm[0:210] = np.arange(0, 210)          # bank + stats + nobles + tableau (identity)
    perm[210:249] = np.arange(249, 288)      # our opp_reserved -> ref opp_reserved
    perm[249:255] = np.arange(288, 294)      # our 6-bit metadata -> ref metadata
    perm[255] = -1                            # new first_player bit -> zero column
    perm[256:295] = np.arange(210, 249)      # our own_reserved -> ref my_reserved
    return perm


def _set_input_dim(graph, name: str, dim_index: int, value: int) -> None:
    """Set graph.input[*name*].shape.dim[dim_index] to a fixed `value`."""
    vi = next((v for v in graph.input if v.name == name), None)
    if vi is None:
        raise RuntimeError(f"input {name!r} not found in graph")
    dims = vi.type.tensor_type.shape.dim
    if len(dims) <= dim_index:
        raise RuntimeError(
            f"input {name!r} has {len(dims)} dims, cannot set dim[{dim_index}]")
    dims[dim_index].dim_value = value
    dims[dim_index].ClearField("dim_param")


def fix(in_path: Path, out_path: Path) -> None:
    if not in_path.exists():
        raise SystemExit(f"input model missing: {in_path}")
    if out_path.resolve() == in_path.resolve():
        raise SystemExit(
            f"refusing to overwrite input model: --in and --out resolve to the "
            f"same path ({in_path})"
        )

    model = onnx.load(str(in_path))
    g = model.graph

    # Locate the first Gemm and figure out what it consumes.
    slice_node = next(
        (n for n in g.node if n.op_type == "Slice" and "features" in n.input),
        None,
    )
    if slice_node is not None:
        # Form (B): adapter graph with a Slice. Gemm consumes the slice output.
        sliced_output = slice_node.output[0]
        gemm_node = next(
            (n for n in g.node if n.op_type == "Gemm" and sliced_output in n.input),
            None,
        )
        if gemm_node is None:
            raise SystemExit(f"no Gemm consuming {sliced_output} found")
    else:
        # Form (A): bare graph. The first Gemm consumes 'features' directly.
        gemm_node = next(
            (n for n in g.node if n.op_type == "Gemm" and "features" in n.input),
            None,
        )
        if gemm_node is None:
            raise SystemExit(
                "no Slice and no Gemm consuming 'features' found — "
                "graph layout is not recognised"
            )

    weight_name = gemm_node.input[1]
    w_ini = next((t for t in g.initializer if t.name == weight_name), None)
    if w_ini is None:
        raise SystemExit(f"initializer {weight_name!r} not found")

    W = numpy_helper.to_array(w_ini)
    if W.shape != (512, 294):
        raise SystemExit(
            f"unexpected weight shape {W.shape}, expected (512, 294). "
            "Has this model already been adapted?"
        )

    # Build permuted weight [512, 295].
    perm = permutation()
    W_new = np.zeros((512, 295), dtype=W.dtype)
    for j, src in enumerate(perm):
        if src >= 0:
            W_new[:, j] = W[:, src]

    new_w_ini = numpy_helper.from_array(W_new, name=weight_name)
    idx = list(g.initializer).index(w_ini)
    g.initializer.remove(w_ini)
    g.initializer.insert(idx, new_w_ini)

    # Rewire Gemm to read 'features' directly; remove the Slice if present.
    new_gemm_inputs = list(gemm_node.input)
    new_gemm_inputs[0] = "features"
    gemm_node.ClearField("input")
    gemm_node.input.extend(new_gemm_inputs)

    if slice_node is not None:
        g.node.remove(slice_node)
        for unused in ("_slice_starts", "_slice_ends", "_slice_axes", "_slice_steps"):
            ini = next((t for t in g.initializer if t.name == unused), None)
            if ini is not None:
                g.initializer.remove(ini)

    # Always set the 'features' input to width 295 (form A starts at 294 and
    # needs widening; form B is already 295 and the assignment is a no-op).
    _set_input_dim(g, "features", dim_index=1, value=295)

    onnx.checker.check_model(model)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(out_path))
    print(f"wrote adapted model to {out_path}")
    print("  backbone.0.weight permuted to [512, 295]; features input width = 295")
    if slice_node is not None:
        print("  removed adapter Slice node")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--in", dest="in_path", type=Path, default=DEFAULT_IN,
                    help=f"input ONNX (default: {DEFAULT_IN})")
    ap.add_argument("--out", dest="out_path", type=Path, default=DEFAULT_OUT,
                    help=f"output ONNX (default: {DEFAULT_OUT})")
    args = ap.parse_args()
    fix(args.in_path, args.out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
