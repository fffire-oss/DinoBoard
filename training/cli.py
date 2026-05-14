"""Unified training CLI: python -m training.cli --game splendor --output runs/splendor_001"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from .pipeline import run_training_loop


def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge `override` into `base`, returning a new dict.

    Nested dicts are merged key-by-key; non-dict values in `override` replace
    the corresponding key in `base` wholesale (lists are not concatenated).
    """
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def find_game_config(game_id: str) -> dict:
    import re
    root = Path(__file__).resolve().parents[1]
    config_path = root / "games" / game_id / "config" / "game.json"
    if not config_path.exists():
        base_id = re.sub(r"_\d+p$", "", game_id)
        config_path = root / "games" / base_id / "config" / "game.json"
        if not config_path.exists():
            raise FileNotFoundError(
                f"Game config not found for '{game_id}' (tried '{game_id}' and '{base_id}')")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    import dinoboard_engine
    meta = dinoboard_engine.game_metadata(game_id)
    cfg["action_space"] = meta["action_space"]
    cfg["feature_dim"] = meta["feature_dim"]
    cfg["num_players"] = meta["num_players"]
    return cfg


def main() -> int:
    parser = argparse.ArgumentParser(description="DinoBoard training CLI")
    parser.add_argument("--game", type=str, required=True, help="Game ID (e.g., tictactoe, splendor)")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    parser.add_argument("--steps", type=int, default=0, help="Override training steps (0 = use config)")
    parser.add_argument("--episodes", type=int, default=0, help="Override episodes per step (0 = use config)")
    parser.add_argument("--workers", type=int, default=None, help="Number of parallel workers (override config)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed (override config)")
    parser.add_argument("--eval-every", type=int, default=None, help="Evaluate every N steps (0 = disable; override config)")
    parser.add_argument("--save-every", type=int, default=None, help="Save model checkpoint every N steps (0 = same as eval-every; override config)")
    parser.add_argument("--eval-games", type=int, default=None, help="Number of evaluation games (override config)")
    parser.add_argument("--batch-size", type=int, default=0, help="Training batch size (0 = use config)")
    parser.add_argument("--lr", type=float, default=0.0, help="Learning rate (0 = use config)")
    parser.add_argument("--eval-benchmark", type=str, nargs="+", default=None,
                        help='Benchmarks: heuristic_constrained, heuristic_free, or ONNX path')
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level")
    parser.add_argument("--init-from", type=str, default=None,
                        help="Path to a .pt checkpoint to initialize weights from. "
                             "Optimizer state is reset (Adam moments do not survive "
                             "architecture changes). The checkpoint must be a dict with "
                             "'model_state_dict' whose keys / shapes match the configured "
                             "PVNet (use scripts/convert_*.py to migrate legacy formats).")
    parser.add_argument("--config-override", type=str, default=None,
                        help="Path to a JSON file whose contents are deep-merged onto the "
                             "game's game.json (overrides any field, including nested ones "
                             "like training.lr_schedule or mcts_profiles.selfplay.simulations).")
    args = parser.parse_args()

    import faulthandler
    faulthandler.enable()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_fmt = "%(asctime)s [%(levelname)s] %(message)s"
    log_datefmt = "%Y-%m-%d %H:%M:%S"
    log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    logging.basicConfig(level=log_level, format=log_fmt, datefmt=log_datefmt)
    fh = logging.FileHandler(output_dir / "train.log", encoding="utf-8")
    fh.setLevel(log_level)
    fh.setFormatter(logging.Formatter(log_fmt, datefmt=log_datefmt))
    logging.getLogger().addHandler(fh)

    game_config = find_game_config(args.game)
    if args.config_override:
        override_path = Path(args.config_override)
        if not override_path.exists():
            raise SystemExit(f"--config-override path does not exist: {override_path}")
        with open(override_path, "r", encoding="utf-8") as f:
            override = json.load(f)
        if not isinstance(override, dict):
            raise SystemExit(f"--config-override JSON must be an object at top level: {override_path}")
        game_config = deep_merge(game_config, override)
        logging.getLogger().info(f"Applied config override from {override_path}")
    train_cfg = game_config["training"]

    steps = args.steps if args.steps > 0 else train_cfg.get("steps", 1000)
    episodes = args.episodes if args.episodes > 0 else train_cfg.get("episodes_per_step", 200)
    batch_size = args.batch_size if args.batch_size > 0 else train_cfg.get("batch_size", 512)
    lr = args.lr if args.lr > 0 else train_cfg.get("learning_rate", 0.001)
    eval_benchmarks = args.eval_benchmark if args.eval_benchmark else train_cfg.get("eval_benchmarks")

    runtime_cfg = train_cfg.get("runtime", {})
    workers = args.workers if args.workers is not None else runtime_cfg.get("workers", 4)
    seed = args.seed if args.seed is not None else runtime_cfg.get("seed", 20260323)
    eval_every = args.eval_every if args.eval_every is not None else runtime_cfg.get("eval_every", 50)
    save_every = args.save_every if args.save_every is not None else runtime_cfg.get("save_every", 0)
    eval_games = args.eval_games if args.eval_games is not None else runtime_cfg.get("eval_games", 40)

    run_training_loop(
        game_id=args.game,
        game_config=game_config,
        output_dir=Path(args.output),
        steps=steps,
        episodes_per_step=episodes,
        eval_every=eval_every,
        eval_games=eval_games,
        eval_benchmarks=eval_benchmarks,
        max_workers=workers,
        batch_size=batch_size,
        learning_rate=lr,
        seed=seed,
        save_every=save_every,
        init_from=args.init_from,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
