"""Sweep historical Coup snapshots, measure root MCTS visit ratio at the
OPENING (empty-history) p1 response to p0 Tax.

Reproduces what a human sees on web: each fresh game, p0 plays Tax, p1's
MCTS root chooses challenge vs allow. Averages root_action_visits across
N seeds per snapshot. No dirichlet noise, opp_sel=prior, web temperature
schedule (1.0 → 0.5, decay=20). Visit-rate is temperature-independent.

Belief net auto-loaded by SessionFactory based on game metadata.
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import shutil
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "platform"))

TAX_ACTION = 6
CHALLENGE_ACTION = 16
ALLOW_ACTION = 17

WEB_SIMS = 2000


def find_snapshots(models_dir: Path, stride: int) -> list[tuple[int, Path, Path]]:
    out: list[tuple[int, Path, Path]] = []
    pat = re.compile(r"model_step_(\d{5})\.onnx$")
    for p in sorted(models_dir.glob("model_step_*.onnx")):
        m = pat.search(p.name)
        if not m:
            continue
        step = int(m.group(1))
        belief = models_dir / f"belief_step_{m.group(1)}.onnx"
        if not belief.exists():
            continue
        out.append((step, p, belief))
    return out[::stride]


def opening_visits(args: tuple) -> tuple[int, int]:
    """Returns (challenge_visits, allow_visits) at p1's response to opening Tax."""
    seed, model_path, belief_path, sims = args
    # Inject belief path through model_paths.find_belief_model_path —
    # SessionFactory looks at games/<base>/model/coup_belief_2p.onnx. Easiest
    # is to swap the deployed file temporarily — but workers run in parallel
    # so we instead pass GameSession the belief_path directly.
    import dinoboard_engine as engine
    gs = engine.GameSession(
        "coup_2p", seed, str(model_path), False, str(belief_path)
    )
    gs.apply_action(TAX_ACTION)
    res = gs.get_ai_action(
        simulations=sims,
        temperature=1.0,
        opponent_selection="prior",
        temperature_initial=1.0,
        temperature_final=0.5,
        temperature_decay_plies=20,
    )
    stats = res["stats"]
    actions = stats["root_actions"]
    visits = stats["root_action_visits"]
    cv = av = 0
    for a, v in zip(actions, visits):
        if a == CHALLENGE_ACTION:
            cv = v
        elif a == ALLOW_ACTION:
            av = v
    return cv, av


def sweep_snapshot(step: int, model_path: Path, belief_path: Path,
                   episodes: int, sims: int, base_seed: int) -> dict:
    total_chal = 0
    total_allow = 0
    for i in range(episodes):
        seed = base_seed * 1_000_003 + step * 1009 + i
        cv, av = opening_visits((seed, model_path, belief_path, sims))
        total_chal += cv
        total_allow += av
    denom = total_chal + total_allow
    rate = total_chal / denom if denom > 0 else float("nan")
    return {
        "step": step,
        "episodes": episodes,
        "sum_chal_visits": total_chal,
        "sum_allow_visits": total_allow,
        "challenge_visit_rate": rate,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--stride", type=int, default=10)
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--sims", type=int, default=WEB_SIMS)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--base-seed", type=int, default=20260516)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    run_dir = Path(args.run)
    models_dir = run_dir / "models"
    if not models_dir.exists():
        print(f"models dir not found: {models_dir}", file=sys.stderr)
        return 1

    snaps = find_snapshots(models_dir, args.stride)
    print(f"Found {len(snaps)} snapshots (stride={args.stride}), "
          f"sims={args.sims}, episodes={args.episodes}, workers={args.workers}")
    if not snaps:
        return 1

    out_path = Path(args.out) if args.out else (run_dir / "opening_tax_visits.csv")

    rows: list[dict] = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = {
            ex.submit(sweep_snapshot, step, mp, bp,
                      args.episodes, args.sims, args.base_seed): step
            for (step, mp, bp) in snaps
        }
        for fut in as_completed(futures):
            row = fut.result()
            rows.append(row)
            print(f"step {row['step']:5d}  "
                  f"visit_rate={row['challenge_visit_rate']:.3f}  "
                  f"chal_v={row['sum_chal_visits']:6d}  "
                  f"allow_v={row['sum_allow_visits']:6d}", flush=True)

    rows.sort(key=lambda r: r["step"])
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
