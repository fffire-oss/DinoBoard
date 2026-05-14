"""Evaluate trained models: run arena matches, report win rates, save replays.

All MCTS knobs (sims / temperature / opponent_selection / tail_solve) come from
the named profile (default `arena`). Pass --profile to use a different one
(e.g. `web_expert` for analysis-strength games).

Number of seats is resolved from the game_id (`game_metadata(game).num_players`).
The model under test rotates across all seats; the same opponent fills the rest.

Usage:
  # 2p model-vs-heuristic
  python platform/tools/eval_model.py --game quoridor \\
    --model runs/quoridor_v14/models/model_best.onnx \\
    --opponent heuristic --games 40 --workers 4 --no-save -o /tmp/eval

  # 2p model-vs-model
  python platform/tools/eval_model.py --game quoridor \\
    --model new.onnx --opponent old.onnx --games 40 -o replays/v14

  # 4p (LL) model-vs-heuristic
  python platform/tools/eval_model.py --game loveletter_4p \\
    --model games/loveletter/model/loveletter_4p.onnx \\
    --opponent heuristic --games 40 --workers 4 --no-save -o /tmp/eval
"""
import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from game_service.replay import build_replay_dict
from training.mcts_profile import max_game_plies, resolve_profile

_HEURISTIC = "heuristic"


def _run_one_game(task: dict) -> dict:
    import dinoboard_engine

    if task["is_heuristic"]:
        result = dinoboard_engine.run_constrained_eval_vs_heuristic(
            game_id=task["game"], seed=task["seed"],
            model_path=task["model_path"],
            simulations=task["sims"],
            model_is_player=task["model_seat"],
            constrained=False,
            heuristic_temperature=task["heuristic_temp"],
            opponent_selection=task["opp_sel"])
    else:
        n = task["num_players"]
        model_paths = [task["opponent_path"]] * n
        model_paths[task["model_seat"]] = task["model_path"]
        result = dinoboard_engine.run_arena_match(
            game_id=task["game"], seed=task["seed"],
            model_paths=model_paths,
            simulations_list=[task["sims"]] * n,
            temperature=task["temp"], max_game_plies=task["max_plies"],
            tail_solve=task["tail_solve_enabled"],
            tail_solve_depth_limit=task["tail_solve_depth_limit"],
            tail_solve_node_budget=task["tail_solve_node_budget"],
            tail_solve_margin_weight=task["tail_solve_margin_weight"],
            opponent_selection_list=[task["opp_sel"]] * n)

    return {
        "game_idx": task["game_idx"],
        "seed": task["seed"],
        "model_seat": task["model_seat"],
        "winner": result["winner"],
        "draw": result["draw"],
        "total_plies": result["total_plies"],
        "action_history": list(result["action_history"]),
    }


def main():
    p = argparse.ArgumentParser(description="Run matches and save replay JSON")
    p.add_argument("--game", default="quoridor",
                   help="Game id (e.g. 'quoridor', 'loveletter_4p', 'azul_3p').")
    p.add_argument("--model", required=True,
                   help="Path to the model under test (rotates across all seats).")
    p.add_argument("--opponent", default=_HEURISTIC,
                   help="Opponent for non-test seats: model path, or 'heuristic' (default).")
    p.add_argument("--name-model", default=None, help="Display name for the test model")
    p.add_argument("--name-opponent", default=None, help="Display name for opponent")
    p.add_argument("--profile", default="arena",
                   help="MCTS profile name (default 'arena'; e.g. 'web_expert')")
    p.add_argument("--heuristic-temp", type=float, default=0.0,
                   help="Heuristic opponent temperature (heuristic-vs-model only)")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--games", type=int, default=1, help="Number of games (rotates seat).")
    p.add_argument("--workers", type=int, default=1, help="Parallel workers")
    p.add_argument("--output", "-o", required=True, help="Output dir or file path")
    p.add_argument("--no-save", action="store_true", help="Only print stats, skip saving replay JSON")
    args = p.parse_args()

    import dinoboard_engine
    meta = dinoboard_engine.game_metadata(args.game)
    num_players = meta["num_players"]

    profile = resolve_profile(args.game, args.profile)
    game_max_plies = max_game_plies(args.game)

    is_heuristic = (args.opponent == _HEURISTIC)
    if not is_heuristic and not Path(args.opponent).exists():
        raise SystemExit(f"--opponent path does not exist: {args.opponent}")

    name_model = args.name_model or Path(args.model).stem
    name_opp = args.name_opponent or (
        f"heuristic_t{args.heuristic_temp}" if is_heuristic else Path(args.opponent).stem)

    out_path = Path(args.output)

    tasks = []
    for game_idx in range(args.games):
        seed = args.seed + game_idx
        model_seat = game_idx % num_players
        tasks.append({
            "game_idx": game_idx,
            "game": args.game,
            "seed": seed,
            "num_players": num_players,
            "model_seat": model_seat,
            "model_path": args.model,
            "opponent_path": None if is_heuristic else args.opponent,
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

    per_seat = [{"w": 0, "l": 0, "d": 0} for _ in range(num_players)]
    running_w = running_l = running_d = 0

    if args.games > 1 and not args.no_save:
        out_path.mkdir(parents=True, exist_ok=True)

    def _stream_print(r: dict, n_done: int) -> None:
        """Tally + print one finished game as soon as it returns."""
        nonlocal running_w, running_l, running_d
        model_seat = r["model_seat"]
        if r["draw"]:
            winner_name = "draw"
            per_seat[model_seat]["d"] += 1
            running_d += 1
        elif r["winner"] == model_seat:
            winner_name = name_model
            per_seat[model_seat]["w"] += 1
            running_w += 1
        else:
            winner_name = name_opp
            per_seat[model_seat]["l"] += 1
            running_l += 1
        running_total = running_w + running_l + running_d
        wr = (running_w / running_total * 100) if running_total else 0.0
        # Stream-printed as games finish: completion order is NOT game_idx
        # order, but every line carries seed + game_idx so it can be re-sorted
        # if needed. The trailing "n_done/total" is wall-clock progress; the
        # leading "[game_idx+1/total]" preserves the previous log shape.
        print(f"[{r['game_idx']+1}/{args.games}] {name_model}(seat{model_seat}) vs {name_opp} "
              f"seed={r['seed']}: {winner_name} wins, {r['total_plies']} plies  "
              f"({n_done}/{args.games}, {name_model} {running_w}-{running_l}-{running_d}={wr:.0f}%)",
              flush=True)

    results: list[dict] = []
    n_done = 0
    if args.workers <= 1:
        for t in tasks:
            r = _run_one_game(t)
            results.append(r)
            n_done += 1
            _stream_print(r, n_done)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(_run_one_game, t): t for t in tasks}
            for fut in as_completed(futures):
                r = fut.result()
                results.append(r)
                n_done += 1
                _stream_print(r, n_done)

    results.sort(key=lambda r: r["game_idx"])

    for r in results:
        game_idx = r["game_idx"]
        model_seat = r["model_seat"]

        if args.no_save:
            continue

        players = {}
        for k in range(num_players):
            if k == model_seat:
                players[f"player_{k}"] = {"name": name_model, "type": "model"}
            else:
                players[f"player_{k}"] = {
                    "name": name_opp,
                    "type": "heuristic" if is_heuristic else "model",
                }

        replay = build_replay_dict(
            game_id=args.game,
            seed=r["seed"],
            action_history=r["action_history"],
            players=players,
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
                "model_seat": model_seat,
                "num_players": num_players,
            },
        )

        if args.games == 1:
            if out_path.suffix == ".json":
                dest = out_path
            else:
                out_path.mkdir(parents=True, exist_ok=True)
                dest = out_path / f"game_seed{r['seed']}.json"
        else:
            dest = out_path / f"game_{game_idx:03d}_seed{r['seed']}_seat{model_seat}.json"

        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(replay, ensure_ascii=False))
        print(f"  Saved to {dest}")

    if args.games > 1:
        total_w = sum(s["w"] for s in per_seat)
        total_l = sum(s["l"] for s in per_seat)
        total_d = sum(s["d"] for s in per_seat)
        print(f"\n=== {name_model} vs {name_opp} ({num_players}p) ===")
        print(f"Overall: {total_w}W-{total_l}L-{total_d}D / {args.games} games "
              f"({total_w/args.games*100:.0f}%)")
        for k in range(num_players):
            n_k = per_seat[k]["w"] + per_seat[k]["l"] + per_seat[k]["d"]
            if n_k == 0:
                continue
            print(f"  As seat{k}: {per_seat[k]['w']}W-{per_seat[k]['l']}L-{per_seat[k]['d']}D / {n_k} "
                  f"({per_seat[k]['w']/n_k*100:.0f}%)")


if __name__ == "__main__":
    main()
