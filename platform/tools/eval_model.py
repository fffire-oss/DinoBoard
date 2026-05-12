"""Evaluate trained models: run arena matches, report win rates, save replays.

All MCTS knobs (sims / temperature / opponent_selection / tail_solve) come from
the named profile (default `arena`). Pass --profile to use a different one
(e.g. `web_expert` for analysis-strength games).

Usage:
  # Quick 40-game eval against arena profile, 4 workers, stats only
  python platform/tools/eval_model.py --game quoridor \\
    --model-a runs/quoridor_v14/models/model_best.onnx \\
    --games 40 --workers 4 --no-save -o /tmp/eval

  # Eval at web_expert strength
  python platform/tools/eval_model.py --game quoridor \\
    --model-a runs/quoridor_v14/models/model_best.onnx \\
    --profile web_expert --games 10 -o replays/v14_eval
"""
import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from game_service.replay import build_replay_dict
from training.mcts_profile import max_game_plies, resolve_profile


def _run_one_game(task: dict) -> dict:
    import dinoboard_engine

    if task["is_heuristic"]:
        result = dinoboard_engine.run_constrained_eval_vs_heuristic(
            game_id=task["game"], seed=task["seed"],
            model_path=task["model_a"],
            simulations=task["sims"],
            model_is_player=task["model_player"],
            constrained=False,
            heuristic_temperature=task["heuristic_temp"],
            opponent_selection=task["opp_sel"])
    else:
        ma = task["model_a"] if task["model_is_first"] else task["model_b"]
        mb = task["model_b"] if task["model_is_first"] else task["model_a"]
        result = dinoboard_engine.run_arena_match(
            game_id=task["game"], seed=task["seed"],
            model_paths=[ma, mb],
            simulations_list=[task["sims"], task["sims"]],
            temperature=task["temp"], max_game_plies=task["max_plies"],
            tail_solve=task["tail_solve_enabled"],
            tail_solve_depth_limit=task["tail_solve_depth_limit"],
            tail_solve_node_budget=task["tail_solve_node_budget"],
            tail_solve_margin_weight=task["tail_solve_margin_weight"],
            opponent_selection_list=[task["opp_sel"], task["opp_sel"]])

    return {
        "game_idx": task["game_idx"],
        "seed": task["seed"],
        "model_is_first": task["model_is_first"],
        "model_player": task["model_player"],
        "winner": result["winner"],
        "draw": result["draw"],
        "total_plies": result["total_plies"],
        "action_history": list(result["action_history"]),
    }


def main():
    p = argparse.ArgumentParser(description="Run matches and save replay JSON")
    p.add_argument("--game", default="quoridor")
    p.add_argument("--model-a", required=True, help="Model path (the model under test)")
    p.add_argument("--model-b", default=None, help="Opponent model path (omit for heuristic)")
    p.add_argument("--name-a", default=None, help="Display name for model A")
    p.add_argument("--name-b", default=None, help="Display name for opponent")
    p.add_argument("--profile", default="arena",
                   help="MCTS profile name (default 'arena'; e.g. 'web_expert')")
    p.add_argument("--heuristic-temp", type=float, default=0.0,
                   help="Heuristic opponent temperature (heuristic-vs-model only)")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--games", type=int, default=1, help="Number of games (alternates sides)")
    p.add_argument("--workers", type=int, default=1, help="Parallel workers")
    p.add_argument("--output", "-o", required=True, help="Output dir or file path")
    p.add_argument("--no-save", action="store_true", help="Only print stats, skip saving replay JSON")
    args = p.parse_args()

    profile = resolve_profile(args.game, args.profile)
    game_max_plies = max_game_plies(args.game)

    name_a = args.name_a or Path(args.model_a).stem
    is_heuristic = args.model_b is None
    name_b = args.name_b or (f"heuristic_t{args.heuristic_temp}" if is_heuristic else Path(args.model_b).stem)

    out_path = Path(args.output)

    tasks = []
    for game_idx in range(args.games):
        seed = args.seed + game_idx
        model_is_first = (game_idx % 2 == 0)
        tasks.append({
            "game_idx": game_idx,
            "game": args.game,
            "seed": seed,
            "model_is_first": model_is_first,
            "model_player": 0 if model_is_first else 1,
            "model_a": args.model_a,
            "model_b": args.model_b,
            "is_heuristic": is_heuristic,
            "sims": profile.simulations,
            "temp": profile.temperature,
            "max_plies": game_max_plies,
            "opp_sel": profile.opponent_selection,
            "tail_solve_enabled": profile.tail_solve_enabled,
            "tail_solve_depth_limit": profile.tail_solve_depth_limit,
            "tail_solve_node_budget": profile.tail_solve_node_budget,
            "tail_solve_margin_weight": profile.tail_solve_margin_weight,
            "heuristic_temp": args.heuristic_temp,
        })

    results = []
    if args.workers <= 1:
        for t in tasks:
            results.append(_run_one_game(t))
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            results = list(pool.map(_run_one_game, tasks))

    results.sort(key=lambda r: r["game_idx"])

    wins_as_first = losses_as_first = draws_as_first = 0
    wins_as_second = losses_as_second = draws_as_second = 0

    if args.games > 1 and not args.no_save:
        out_path.mkdir(parents=True, exist_ok=True)

    for r in results:
        game_idx = r["game_idx"]
        model_is_first = r["model_is_first"]
        model_player = r["model_player"]
        model_won = not r["draw"] and r["winner"] == model_player
        side_tag = "先手" if model_is_first else "后手"

        if r["draw"]:
            winner_name = "draw"
        else:
            winner_name = name_a if model_won else name_b

        if model_is_first:
            if r["draw"]: draws_as_first += 1
            elif model_won: wins_as_first += 1
            else: losses_as_first += 1
        else:
            if r["draw"]: draws_as_second += 1
            elif model_won: wins_as_second += 1
            else: losses_as_second += 1

        print(f"[{game_idx+1}/{args.games}] {name_a}({side_tag}) vs {name_b} "
              f"seed={r['seed']}: {winner_name} wins, {r['total_plies']} plies")

        if args.no_save:
            continue

        p0_name = name_a if model_is_first else name_b
        p1_name = name_b if model_is_first else name_a
        p0_type = "model" if model_is_first else ("heuristic" if is_heuristic else "model")
        p1_type = ("heuristic" if is_heuristic else "model") if model_is_first else "model"

        replay = build_replay_dict(
            game_id=args.game,
            seed=r["seed"],
            action_history=r["action_history"],
            players={
                "player_0": {"name": p0_name, "type": p0_type},
                "player_1": {"name": p1_name, "type": p1_type},
            },
            result={
                "winner": r["winner"],
                "draw": r["draw"],
                "total_plies": r["total_plies"],
            },
            config={
                "profile": args.profile,
                "simulations": profile.simulations,
                "temperature": profile.temperature,
                "opponent_selection": profile.opponent_selection,
                "tail_solve_enabled": profile.tail_solve_enabled,
                "max_game_plies": game_max_plies,
                "heuristic_temperature": args.heuristic_temp if is_heuristic else None,
                "model_player": model_player,
            },
        )

        if args.games == 1:
            if out_path.suffix == ".json":
                dest = out_path
            else:
                out_path.mkdir(parents=True, exist_ok=True)
                dest = out_path / f"game_seed{r['seed']}.json"
        else:
            side = "1st" if model_is_first else "2nd"
            dest = out_path / f"game_{game_idx:03d}_seed{r['seed']}_{side}.json"

        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(replay, ensure_ascii=False))
        print(f"  Saved to {dest}")

    if args.games > 1:
        total_w = wins_as_first + wins_as_second
        total_l = losses_as_first + losses_as_second
        total_d = draws_as_first + draws_as_second
        games_first = wins_as_first + losses_as_first + draws_as_first
        games_second = wins_as_second + losses_as_second + draws_as_second
        print(f"\n=== {name_a} vs {name_b} ===")
        print(f"Overall: {total_w}W-{total_l}L-{total_d}D / {args.games} games "
              f"({total_w/args.games*100:.0f}%)")
        if games_first > 0:
            print(f"  As 1st: {wins_as_first}W-{losses_as_first}L-{draws_as_first}D / {games_first} "
                  f"({wins_as_first/games_first*100:.0f}%)")
        if games_second > 0:
            print(f"  As 2nd: {wins_as_second}W-{losses_as_second}L-{draws_as_second}D / {games_second} "
                  f"({wins_as_second/games_second*100:.0f}%)")


if __name__ == "__main__":
    main()
