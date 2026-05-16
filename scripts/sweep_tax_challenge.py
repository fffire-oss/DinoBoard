"""Sweep historical Coup snapshots, measure MCTS root visit distribution at
"p1 challenges p0's Tax" decision nodes.

Reads policy_action_visits directly from selfplay samples — this is the raw
MCTS visit count BEFORE temperature sampling, so temperature is irrelevant.
For each pair (p0 Tax) → (p1 challenge-or-allow), compute
   visit[challenge] / (visit[challenge] + visit[allow])
and aggregate across episodes.

Uses max 2 workers (training is still running on 4 workers; don't OOM).
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import dinoboard_engine

# From games/coup/coup_state.h
TAX_ACTION = 6
CHALLENGE_ACTION = 16
ALLOW_ACTION = 17


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


def episode_visit_stats(args: tuple) -> tuple[int, int, int, int, int]:
    """Returns (n_pairs, sum_chal_visits, sum_allow_visits,
                n_pure_challenge_pairs, total_plies).

    n_pure_challenge_pairs counts pairs where allow_visits == 0 (collapsed).
    """
    game_id, seed, model_path, belief_path, sims, max_plies = args
    result = dinoboard_engine.run_selfplay_episode(
        game_id=game_id,
        seed=seed,
        model_path=model_path,
        simulations=sims,
        c_puct=1.4,
        temperature=1.0,
        dirichlet_alpha=0.0,
        dirichlet_epsilon=0.0,
        dirichlet_on_first_n_plies=0,
        max_game_plies=max_plies,
        opponent_selection="prior",
        belief_model_path=belief_path,
    )
    samples = result["samples"]
    n_pairs = 0
    sum_chal = 0
    sum_allow = 0
    pure_chal = 0
    for i in range(len(samples) - 1):
        a = samples[i]
        b = samples[i + 1]
        if (a["player"] == 0 and a["action_id"] == TAX_ACTION
                and b["player"] == 1
                and b["action_id"] in (CHALLENGE_ACTION, ALLOW_ACTION)):
            ids = b["policy_action_ids"]
            vis = b["policy_action_visits"]
            cv = 0
            av = 0
            for aid, v in zip(ids, vis):
                if aid == CHALLENGE_ACTION:
                    cv = v
                elif aid == ALLOW_ACTION:
                    av = v
            if cv + av == 0:
                continue
            n_pairs += 1
            sum_chal += cv
            sum_allow += av
            if av == 0:
                pure_chal += 1
    return n_pairs, sum_chal, sum_allow, pure_chal, result["total_plies"]


def sweep_snapshot(step: int, model_path: Path, belief_path: Path,
                   episodes: int, sims: int, max_plies: int,
                   base_seed: int) -> dict:
    total_pairs = 0
    total_chal = 0
    total_allow = 0
    total_pure = 0
    total_plies = 0
    for i in range(episodes):
        seed = base_seed * 1_000_003 + step * 1009 + i
        np_, cv, av, pc, plies = episode_visit_stats(
            ("coup", seed, str(model_path), str(belief_path), sims, max_plies))
        total_pairs += np_
        total_chal += cv
        total_allow += av
        total_pure += pc
        total_plies += plies
    visit_rate = (total_chal / (total_chal + total_allow)
                  if (total_chal + total_allow) > 0 else float("nan"))
    pure_rate = total_pure / total_pairs if total_pairs > 0 else float("nan")
    avg_plies = total_plies / max(1, episodes)
    return {
        "step": step,
        "n_pairs": total_pairs,
        "sum_chal_visits": total_chal,
        "sum_allow_visits": total_allow,
        "challenge_visit_rate": visit_rate,
        "pure_challenge_rate": pure_rate,
        "avg_plies": avg_plies,
        "episodes": episodes,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--stride", type=int, default=10)
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--sims", type=int, default=500)
    ap.add_argument("--max-plies", type=int, default=100)
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
    print(f"Found {len(snaps)} snapshots (stride={args.stride}), sims={args.sims}, episodes={args.episodes}")
    if not snaps:
        return 1

    out_path = Path(args.out) if args.out else (run_dir / "tax_challenge_visits.csv")

    rows: list[dict] = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = {
            ex.submit(sweep_snapshot, step, mp, bp,
                      args.episodes, args.sims, args.max_plies, args.base_seed): step
            for (step, mp, bp) in snaps
        }
        for fut in as_completed(futures):
            row = fut.result()
            rows.append(row)
            print(f"step {row['step']:5d}  visit_rate={row['challenge_visit_rate']:.3f}  "
                  f"pure_chal={row['pure_challenge_rate']:.2f}  pairs={row['n_pairs']:4d}  "
                  f"chal_v={row['sum_chal_visits']:6d}  allow_v={row['sum_allow_visits']:6d}  "
                  f"plies={row['avg_plies']:.1f}", flush=True)

    rows.sort(key=lambda r: r["step"])
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
