"""Measure the deployed Coup AI's full-game bluff rate under web_expert.

Two copies of the deployed network play full coup_2p games against each
other. At every claim-bearing decision (Tax / Steal / Assassinate /
Exchange / Block-Duke / Block-Captain / Block-Ambassador / Block-Contessa)
we read the actor's truth hand and check whether the claimed role is in
hand. Stats are aggregated over the whole run.

Web-expert config: sims=2000, opp_sel=prior, temperature 1.0→0.5 decay=20,
no dirichlet (matches games/coup/config/web.json).
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

# Action ids (mirrors games/coup/coup_state.h).
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
CHALLENGE = 16
ALLOW = 17
BLOCK_DUKE = 18         # Block Foreign Aid (claims Duke)
BLOCK_CONTESSA = 19     # Block Assassinate (claims Contessa)
BLOCK_AMBASSADOR = 20   # Block Steal (claims Ambassador)
BLOCK_CAPTAIN = 21      # Block Steal (claims Captain)

DUKE, ASSASSIN, CAPTAIN, AMBASSADOR, CONTESSA = 0, 1, 2, 3, 4
CHAR_NAMES = ["Duke", "Assassin", "Captain", "Ambassador", "Contessa"]

# web_expert (games/coup/config/web.json).
WEB_SIMS = 2000
WEB_TEMPERATURE = 1.0
WEB_TEMP_INITIAL = 1.0
WEB_TEMP_FINAL = 0.5
WEB_TEMP_DECAY = 20

DEFAULT_MODEL = REPO_ROOT / "games" / "coup" / "model" / "coup_2p.onnx"
DEFAULT_BELIEF = REPO_ROOT / "games" / "coup" / "model" / "coup_belief_2p.onnx"


def claim_role(action: int) -> int | None:
    """Returns claimed character id, or None if action is claim-free."""
    if action == TAX:
        return DUKE
    if ASSASSINATE_OFFSET <= action < ASSASSINATE_OFFSET + ASSASSINATE_COUNT:
        return ASSASSIN
    if STEAL_OFFSET <= action < STEAL_OFFSET + STEAL_COUNT:
        return CAPTAIN
    if action == EXCHANGE:
        return AMBASSADOR
    if action == BLOCK_DUKE:
        return DUKE
    if action == BLOCK_CONTESSA:
        return CONTESSA
    if action == BLOCK_AMBASSADOR:
        return AMBASSADOR
    if action == BLOCK_CAPTAIN:
        return CAPTAIN
    return None


def action_kind(action: int) -> str:
    """Coarse label for action distribution reporting."""
    if action == INCOME:
        return "income"
    if action == FOREIGN_AID:
        return "foreign_aid"
    if COUP_OFFSET <= action < COUP_OFFSET + COUP_COUNT:
        return "coup"
    if action == TAX:
        return "tax"
    if ASSASSINATE_OFFSET <= action < ASSASSINATE_OFFSET + ASSASSINATE_COUNT:
        return "assassinate"
    if STEAL_OFFSET <= action < STEAL_OFFSET + STEAL_COUNT:
        return "steal"
    if action == EXCHANGE:
        return "exchange"
    if action == CHALLENGE:
        return "challenge"
    if action == ALLOW:
        return "allow"
    if action == BLOCK_DUKE:
        return "block_duke"
    if action == BLOCK_CONTESSA:
        return "block_contessa"
    if action == BLOCK_AMBASSADOR:
        return "block_ambassador"
    if action == BLOCK_CAPTAIN:
        return "block_captain"
    if action == 22:
        return "allow_no_block"
    if action == 23 or action == 24:
        return "reveal"
    if action == 25 or action == 26:
        return "lose"
    if 27 <= action <= 31:
        return "return_card"
    return f"action_{action}"


def actor_hand(state_dict: dict, actor: int) -> tuple[int, int, int, int]:
    """Returns (slot0_char, slot0_revealed, slot1_char, slot1_revealed)."""
    p = state_dict["players"][actor]
    inf = p["influences"]
    return (
        int(inf[0]["character"]), int(inf[0]["revealed"]),
        int(inf[1]["character"]), int(inf[1]["revealed"]),
    )


def alive_chars(state_dict: dict, actor: int) -> list[int]:
    h0c, h0r, h1c, h1r = actor_hand(state_dict, actor)
    out: list[int] = []
    if not h0r:
        out.append(h0c)
    if not h1r:
        out.append(h1c)
    return out


def play_one_game(args: tuple) -> dict:
    """Plays one full coup_2p game; both seats use the same model.

    Returns aggregated counters + the per-decision claim records as a
    list-of-dicts (small — full games rarely exceed ~30 plies).
    """
    seed, model_path, belief_path, sims, max_plies = args
    import dinoboard_engine as engine
    gs = engine.GameSession(
        "coup_2p", seed, str(model_path), False, str(belief_path)
    )

    decisions: list[dict] = []
    label_counter: Counter[str] = Counter()
    plies_played = 0
    while not gs.is_terminal and plies_played < max_plies:
        actor = gs.current_player
        sd = gs.get_state_dict()
        res = gs.get_ai_action(
            simulations=sims,
            temperature=WEB_TEMPERATURE,
            opponent_selection="prior",
            temperature_initial=WEB_TEMP_INITIAL,
            temperature_final=WEB_TEMP_FINAL,
            temperature_decay_plies=WEB_TEMP_DECAY,
        )
        action = int(res["action"])
        label_counter[action_kind(action)] += 1

        claimed = claim_role(action)
        if claimed is not None:
            alive = alive_chars(sd, actor)
            holds = claimed in alive
            decisions.append({
                "seed": seed,
                "ply": plies_played,
                "actor": actor,
                "stage": int(sd["stage"]),
                "action": action,
                "kind": action_kind(action),
                "claimed_char": claimed,
                "alive_chars": alive,
                "holds_claim": holds,
                "is_bluff": not holds,
            })

        gs.apply_action(action)
        plies_played += 1

    return {
        "seed": seed,
        "plies": plies_played,
        "winner": int(gs.winner) if gs.is_terminal else -1,
        "terminal": bool(gs.is_terminal),
        "decisions": decisions,
        "label_counter": dict(label_counter),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=20,
                    help="number of full games to play")
    ap.add_argument("--sims", type=int, default=WEB_SIMS)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--base-seed", type=int, default=20260516)
    ap.add_argument("--max-plies", type=int, default=120,
                    help="safety cap; coup_2p rarely exceeds 60")
    ap.add_argument("--model", default=str(DEFAULT_MODEL))
    ap.add_argument("--belief", default=str(DEFAULT_BELIEF))
    ap.add_argument("--out", default=None,
                    help="optional CSV path for per-decision rows")
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
    print(f"games: {args.games}, workers: {args.workers}\n")

    seeds = [args.base_seed * 1_000_003 + i for i in range(args.games)]
    all_decisions: list[dict] = []
    label_total: Counter[str] = Counter()
    role_claim: Counter[int] = Counter()
    role_bluff: Counter[int] = Counter()
    # Split kinds for finer reporting.
    kind_claim: Counter[str] = Counter()
    kind_bluff: Counter[str] = Counter()
    games_done = 0
    plies_total = 0

    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = {
            ex.submit(play_one_game,
                      (s, model_path, belief_path, args.sims, args.max_plies)): s
            for s in seeds
        }
        for fut in as_completed(futures):
            res = fut.result()
            games_done += 1
            plies_total += res["plies"]
            for k, v in res["label_counter"].items():
                label_total[k] += v
            for d in res["decisions"]:
                all_decisions.append(d)
                role_claim[d["claimed_char"]] += 1
                kind_claim[d["kind"]] += 1
                if d["is_bluff"]:
                    role_bluff[d["claimed_char"]] += 1
                    kind_bluff[d["kind"]] += 1
            claims = sum(role_claim.values())
            bluffs = sum(role_bluff.values())
            br = bluffs / claims if claims else float("nan")
            print(f"  game {games_done:3d}/{args.games}  "
                  f"plies={res['plies']:3d}  winner={res['winner']:+d}  "
                  f"claims={claims:4d}  bluffs={bluffs:3d}  "
                  f"bluff_rate={br:.3f}", flush=True)

    print("\n=== Action distribution (all decisions, both seats) ===")
    for label, n in sorted(label_total.items(), key=lambda kv: -kv[1]):
        print(f"  {label:18s}  {n:5d}")

    print("\n=== Per-claim-kind bluff ===")
    print(f"  {'kind':18s}  {'claims':>7s}  {'bluffs':>7s}  {'bluff_rate':>11s}")
    # Always print all 8 claim kinds, even when 0, so missing rows are visible.
    all_claim_kinds = [
        "tax", "steal", "assassinate", "exchange",
        "block_duke", "block_captain", "block_ambassador", "block_contessa",
    ]
    for k in all_claim_kinds:
        c = kind_claim[k]
        b = kind_bluff[k]
        rate = b / c if c else float("nan")
        print(f"  {k:18s}  {c:7d}  {b:7d}  {rate:11.3f}")

    print("\n=== Per-role bluff ===")
    print(f"  {'role':12s}  {'claims':>7s}  {'bluffs':>7s}  {'bluff_rate':>11s}")
    for r in range(5):
        c = role_claim[r]
        b = role_bluff[r]
        rate = b / c if c else float("nan")
        print(f"  {CHAR_NAMES[r]:12s}  {c:7d}  {b:7d}  {rate:11.3f}")

    total_claims = sum(role_claim.values())
    total_bluffs = sum(role_bluff.values())
    overall = total_bluffs / total_claims if total_claims else float("nan")
    print(f"\n=== Overall ===")
    print(f"  games:          {games_done}")
    print(f"  total plies:    {plies_total}")
    print(f"  claim-bearing:  {total_claims}")
    print(f"  bluffs:         {total_bluffs}")
    print(f"  bluff_rate:     {overall:.3f}   "
          f"(P(claimed_role not in hand | claim))")

    if args.out and all_decisions:
        # Convert alive_chars list → comma-joined string for CSV.
        for d in all_decisions:
            d["alive_chars"] = ",".join(str(c) for c in d["alive_chars"])
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(all_decisions[0].keys()))
            w.writeheader()
            w.writerows(all_decisions)
        print(f"\nWrote {len(all_decisions)} decisions to {args.out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
