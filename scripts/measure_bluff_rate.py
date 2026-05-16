"""Measure the deployed Coup AI's opening bluff rate under web_expert config.

For each seed: fresh game, p0's first declare-stage move. Read p0's hand
from the truth-side state dict, run get_ai_action with web_expert settings
(sims=2000, opp_sel=prior, temperature schedule 1.0→0.5 decay=20, no
dirichlet — see games/coup/config/web.json), classify the chosen action:

  Tax        → claims Duke
  Steal-*    → claims Captain
  Assassinate→ claims Assassin
  Exchange   → claims Ambassador

A "bluff" is a claim-bearing action whose claimed character is NOT in p0's
hand. Income / Foreign Aid / Coup are claim-free, counted separately.

Uses the deployed games/coup/model/coup_2p.onnx + coup_belief_2p.onnx by
default (the same files the web frontend serves).
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "platform"))

# Coup action-id constants (mirrors games/coup/coup_state.h).
INCOME = 0
FOREIGN_AID = 1
COUP_OFFSET = 2
COUP_COUNT = 4
TAX = 6
ASSASSINATE_OFFSET = 7
ASSASSINATE_COUNT = 4
STEAL_OFFSET = 11
STEAL_COUNT = 4
EXCHANGE = 15

# Character ids.
DUKE, ASSASSIN, CAPTAIN, AMBASSADOR, CONTESSA = 0, 1, 2, 3, 4
CHAR_NAMES = ["Duke", "Assassin", "Captain", "Ambassador", "Contessa"]

# web_expert profile (games/coup/config/web.json).
WEB_SIMS = 2000
WEB_TEMPERATURE = 1.0
WEB_TEMP_INITIAL = 1.0
WEB_TEMP_FINAL = 0.5
WEB_TEMP_DECAY = 20

DEFAULT_MODEL = REPO_ROOT / "games" / "coup" / "model" / "coup_2p.onnx"
DEFAULT_BELIEF = REPO_ROOT / "games" / "coup" / "model" / "coup_belief_2p.onnx"


def classify_action(action: int) -> tuple[str, int | None]:
    """Returns (action_label, claimed_char_id_or_None)."""
    if action == INCOME:
        return "income", None
    if action == FOREIGN_AID:
        return "foreign_aid", None
    if COUP_OFFSET <= action < COUP_OFFSET + COUP_COUNT:
        return "coup", None
    if action == TAX:
        return "tax", DUKE
    if ASSASSINATE_OFFSET <= action < ASSASSINATE_OFFSET + ASSASSINATE_COUNT:
        return "assassinate", ASSASSIN
    if STEAL_OFFSET <= action < STEAL_OFFSET + STEAL_COUNT:
        return "steal", CAPTAIN
    if action == EXCHANGE:
        return "exchange", AMBASSADOR
    return f"action_{action}", None


def p0_hand(state_dict: dict) -> tuple[int, int]:
    """Returns p0's two influence character ids (truth-side)."""
    p0 = state_dict["players"][0]
    inf = p0["influences"]
    return int(inf[0]["character"]), int(inf[1]["character"])


def run_one(args: tuple) -> dict:
    seed, model_path, belief_path, sims = args
    import dinoboard_engine as engine
    gs = engine.GameSession(
        "coup_2p", seed, str(model_path), False, str(belief_path)
    )
    sd = gs.get_state_dict()
    assert sd["current_player"] == 0, "fresh coup_2p starts on p0"
    h0, h1 = p0_hand(sd)
    res = gs.get_ai_action(
        simulations=sims,
        temperature=WEB_TEMPERATURE,
        opponent_selection="prior",
        temperature_initial=WEB_TEMP_INITIAL,
        temperature_final=WEB_TEMP_FINAL,
        temperature_decay_plies=WEB_TEMP_DECAY,
    )
    action = int(res["action"])
    label, claimed = classify_action(action)
    holds = (claimed is not None) and (h0 == claimed or h1 == claimed)
    return {
        "seed": seed,
        "hand0": h0,
        "hand1": h1,
        "action": action,
        "label": label,
        "claimed_char": claimed if claimed is not None else -1,
        "is_claim": claimed is not None,
        "holds_claim": holds,
        "is_bluff": (claimed is not None) and (not holds),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=200,
                    help="number of independent seeds to evaluate")
    ap.add_argument("--sims", type=int, default=WEB_SIMS,
                    help="MCTS simulations per move (web_expert default 2000)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--base-seed", type=int, default=20260516)
    ap.add_argument("--model", default=str(DEFAULT_MODEL))
    ap.add_argument("--belief", default=str(DEFAULT_BELIEF))
    ap.add_argument("--out", default=None,
                    help="optional CSV path for per-episode rows")
    args = ap.parse_args()

    model_path = Path(args.model)
    belief_path = Path(args.belief)
    if not model_path.exists():
        print(f"model not found: {model_path}", file=sys.stderr)
        return 1
    if not belief_path.exists():
        print(f"belief not found: {belief_path}", file=sys.stderr)
        return 1

    print(f"model: {model_path}")
    print(f"belief: {belief_path}")
    print(f"profile: web_expert (sims={args.sims}, opp_sel=prior, "
          f"temp 1.0→0.5 decay=20)")
    print(f"episodes: {args.episodes}, workers: {args.workers}\n")

    seeds = [args.base_seed * 1_000_003 + i for i in range(args.episodes)]
    rows: list[dict] = []
    label_counter: Counter[str] = Counter()
    claim_total = 0
    bluff_total = 0
    role_claim: Counter[int] = Counter()
    role_bluff: Counter[int] = Counter()

    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = {
            ex.submit(run_one, (s, model_path, belief_path, args.sims)): s
            for s in seeds
        }
        for i, fut in enumerate(as_completed(futures)):
            row = fut.result()
            rows.append(row)
            label_counter[row["label"]] += 1
            if row["is_claim"]:
                claim_total += 1
                role_claim[row["claimed_char"]] += 1
                if row["is_bluff"]:
                    bluff_total += 1
                    role_bluff[row["claimed_char"]] += 1
            if (i + 1) % 20 == 0 or (i + 1) == len(seeds):
                br = bluff_total / claim_total if claim_total else float("nan")
                print(f"  [{i+1:4d}/{len(seeds)}] claim={claim_total} "
                      f"bluff={bluff_total} bluff_rate={br:.3f}", flush=True)

    print("\n=== Action distribution (p0 opening) ===")
    for label, n in sorted(label_counter.items(), key=lambda kv: -kv[1]):
        print(f"  {label:14s}  {n:5d}  ({n/len(rows):.3f})")

    print("\n=== Per-role claim / bluff ===")
    print(f"  {'role':12s}  {'claims':>7s}  {'bluffs':>7s}  {'bluff_rate':>11s}")
    for r in range(5):
        c = role_claim[r]
        b = role_bluff[r]
        rate = b / c if c else float("nan")
        print(f"  {CHAR_NAMES[r]:12s}  {c:7d}  {b:7d}  {rate:11.3f}")

    overall = bluff_total / claim_total if claim_total else float("nan")
    print(f"\n=== Overall ===")
    print(f"  episodes:       {len(rows)}")
    print(f"  claim-bearing:  {claim_total}  ({claim_total/len(rows):.3f})")
    print(f"  bluffs:         {bluff_total}")
    print(f"  bluff_rate:     {overall:.3f}   "
          f"(P(claim_role not in hand | claim))")

    if args.out:
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\nWrote per-episode rows to {args.out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
