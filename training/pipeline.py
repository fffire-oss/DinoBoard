"""Self-play training pipeline: selfplay -> collect samples -> train -> export -> repeat."""
from __future__ import annotations

import json
import logging
import math
import random
import shutil
import time
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import torch

from .mcts_profile import MctsProfile, max_game_plies, profile_as_dict, resolve_profile
from .model import (
    PVNet,
    create_belief_model_from_config,
    create_model_from_config,
    export_belief_onnx,
    export_onnx,
)

logger = logging.getLogger(__name__)


def _default_gating_threshold(num_players: int, eval_games: int) -> float:
    """Default gating win-rate threshold, scaled to the N-player null baseline.

    For an N-player zero-sum game the null hypothesis is 1/N (no skill edge),
    not 0.5. A fixed 0.55 threshold is tight for 2p but trivially loose for
    3p/4p (null already at 0.333 / 0.25). We use
        threshold = baseline + z * sqrt(baseline * (1 - baseline) / n_games)
    with z calibrated so (num_players=2, eval_games=40) recovers the historical
    0.55 default. Gives roughly 2p: 0.55, 3p: 0.38, 4p: 0.29 at 40 eval games —
    the same one-sided confidence lead over the null across player counts.
    """
    baseline = 1.0 / max(num_players, 1)
    z = 0.632  # (0.55 - 0.5) / sqrt(0.25 / 40)
    se = math.sqrt(baseline * (1.0 - baseline) / max(eval_games, 1))
    return baseline + z * se


def _worker_selfplay(args: tuple) -> dict[str, Any]:
    """Run a single selfplay episode in a worker process.

    `cfg` is a profile_as_dict() blob plus per-step overrides:
        simulations (ramped), max_game_plies (game-level),
        heuristic_guidance_ratio, heuristic_temperature,
        training_filter_ratio (already gated by profile.ai_use_action_filter).
    """
    import dinoboard_engine
    game_id, seed, model_path, cfg = args
    # Bindings use -1.0 as "schedule off" sentinel for the flat
    # temperature_initial / temperature_final args.
    if cfg["temperature_schedule_enabled"]:
        t_initial = cfg["temperature_initial"]
        t_final = cfg["temperature_final"]
        t_decay = cfg["temperature_decay_plies"]
    else:
        t_initial = -1.0
        t_final = -1.0
        t_decay = 0
    return dinoboard_engine.run_selfplay_episode(
        game_id=game_id,
        seed=seed,
        model_path=model_path,
        simulations=cfg["simulations"],
        c_puct=cfg["c_puct"],
        temperature=cfg["temperature"],
        dirichlet_alpha=cfg["dirichlet_alpha"],
        dirichlet_epsilon=cfg["dirichlet_epsilon"],
        dirichlet_on_first_n_plies=cfg["dirichlet_on_first_n_plies"],
        max_game_plies=cfg["max_game_plies"],
        tail_solve_enabled=cfg["tail_solve_enabled"],
        tail_solve_depth_limit=cfg["tail_solve_depth_limit"],
        tail_solve_node_budget=cfg["tail_solve_node_budget"],
        tail_solve_margin_weight=cfg["tail_solve_margin_weight"],
        temperature_initial=t_initial,
        temperature_final=t_final,
        temperature_decay_plies=t_decay,
        heuristic_guidance_ratio=cfg["heuristic_guidance_ratio"],
        heuristic_temperature=cfg["heuristic_temperature"],
        training_filter_ratio=cfg["training_filter_ratio"],
        opponent_selection=cfg["opponent_selection"],
        belief_model_path=cfg.get("belief_model_path", ""),
    )


def _worker_selfplay_pool(args: tuple) -> dict[str, Any]:
    """Run a single opponent-pool selfplay episode.

    args:
        (game_id, seed, model_paths_per_seat, latest_seat, cfg)

    The C++ binding builds one ONNX evaluator per seat and routes MCTS at
    each ply to evaluators[acting_seat]. Sample filtering by latest_seat
    happens caller-side; the worker just returns the full episode dict
    with `latest_seat` tagged on so the caller knows which seat to keep.
    """
    import dinoboard_engine
    game_id, seed, model_paths_per_seat, latest_seat, cfg = args
    if cfg["temperature_schedule_enabled"]:
        t_initial = cfg["temperature_initial"]
        t_final = cfg["temperature_final"]
        t_decay = cfg["temperature_decay_plies"]
    else:
        t_initial = -1.0
        t_final = -1.0
        t_decay = 0
    result = dinoboard_engine.run_selfplay_episode_pool(
        game_id=game_id,
        seed=seed,
        model_paths=model_paths_per_seat,
        simulations=cfg["simulations"],
        c_puct=cfg["c_puct"],
        temperature=cfg["temperature"],
        dirichlet_alpha=cfg["dirichlet_alpha"],
        dirichlet_epsilon=cfg["dirichlet_epsilon"],
        dirichlet_on_first_n_plies=cfg["dirichlet_on_first_n_plies"],
        max_game_plies=cfg["max_game_plies"],
        tail_solve_enabled=cfg["tail_solve_enabled"],
        tail_solve_depth_limit=cfg["tail_solve_depth_limit"],
        tail_solve_node_budget=cfg["tail_solve_node_budget"],
        tail_solve_margin_weight=cfg["tail_solve_margin_weight"],
        temperature_initial=t_initial,
        temperature_final=t_final,
        temperature_decay_plies=t_decay,
        heuristic_guidance_ratio=cfg["heuristic_guidance_ratio"],
        heuristic_temperature=cfg["heuristic_temperature"],
        training_filter_ratio=cfg["training_filter_ratio"],
        opponent_selection=cfg["opponent_selection"],
        belief_model_path=cfg.get("belief_model_path", ""),
    )
    result["latest_seat"] = latest_seat
    return result


def _dispatch_worker(task: tuple) -> dict[str, Any]:
    """Top-level (picklable) dispatcher for the mixed self/pool task list."""
    kind, args = task
    if kind == "self":
        out = _worker_selfplay(args)
        out["latest_seat"] = None  # tag so caller filter can branch
        return out
    if kind == "pool":
        return _worker_selfplay_pool(args)
    raise ValueError(f"_dispatch_worker: unknown task kind {kind!r}")


def _collect_pool_paths(models_dir: Path) -> list[str]:
    """All `model_step_*.onnx` checkpoints in models_dir, sorted.

    Excludes `model_latest.onnx`, `model_init.onnx`, `model_best.onnx`
    by glob pattern. Returns [] before the first checkpoint is saved.
    """
    paths = sorted(models_dir.glob("model_step_*.onnx"))
    return [str(p) for p in paths]


def _worker_arena(args: tuple) -> dict[str, Any]:
    """Run a single arena match in a worker process."""
    import dinoboard_engine
    game_id, seed, model_paths, sims_list, max_plies, temperature, opp_sel, \
        ts_enabled, ts_depth, ts_budget, ts_margin = args
    return dinoboard_engine.run_arena_match(
        game_id=game_id, seed=seed,
        model_paths=model_paths, simulations_list=sims_list,
        temperature=temperature,
        max_game_plies=max_plies,
        tail_solve=ts_enabled,
        tail_solve_depth_limit=ts_depth,
        tail_solve_node_budget=ts_budget,
        tail_solve_margin_weight=ts_margin,
        opponent_selection_list=[opp_sel] * len(model_paths),
    )


def _worker_eval_vs_heuristic(args: tuple) -> dict[str, Any]:
    """Run a single eval game vs heuristic in a worker process."""
    import dinoboard_engine
    (game_id, seed, model_path, simulations, model_is_player,
     constrained, h_temp, opp_sel) = args
    return dinoboard_engine.run_constrained_eval_vs_heuristic(
        game_id=game_id,
        seed=seed,
        model_path=model_path,
        simulations=simulations,
        model_is_player=model_is_player,
        constrained=constrained,
        heuristic_temperature=h_temp,
        opponent_selection=opp_sel,
    )


def normalize_policy(action_ids: list[int], visits: list[int], action_space: int) -> list[float]:
    total = sum(max(0, v) for v in visits)
    policy = [0.0] * action_space
    if total <= 0:
        return policy
    for aid, v in zip(action_ids, visits):
        if 0 <= aid < action_space:
            policy[aid] = max(0, v) / total
    return policy


def train_step(
    net: PVNet,
    optimizer: torch.optim.Optimizer,
    features: torch.Tensor,
    policy_targets: torch.Tensor,
    value_targets: torch.Tensor,
    legal_mask: torch.Tensor | None = None,
    auxiliary_targets: torch.Tensor | None = None,
    auxiliary_weight: float = 0.5,
    grad_clip_norm: float = 1.0,
) -> dict[str, float]:
    net.train()
    outputs = net(features)
    if net.has_score_head:
        policy_logits, value_pred, score_pred = outputs
    else:
        policy_logits, value_pred = outputs

    if legal_mask is not None:
        policy_logits = policy_logits.masked_fill(legal_mask == 0, -1e9)

    value_loss = torch.nn.functional.mse_loss(value_pred, value_targets)
    policy_loss = -(policy_targets * torch.nn.functional.log_softmax(policy_logits, dim=-1)).sum(dim=-1).mean()
    loss = policy_loss + value_loss

    if net.has_score_head and auxiliary_targets is not None:
        score_loss = torch.nn.functional.mse_loss(score_pred.squeeze(-1), auxiliary_targets)
        loss = loss + auxiliary_weight * score_loss
    else:
        score_loss = torch.tensor(0.0)

    optimizer.zero_grad()
    loss.backward()
    if grad_clip_norm > 0:
        torch.nn.utils.clip_grad_norm_(net.parameters(), grad_clip_norm)
    optimizer.step()
    return {
        "loss": loss.item(),
        "policy_loss": policy_loss.item(),
        "value_loss": value_loss.item(),
        "score_loss": score_loss.item(),
    }


def _train_belief_step(
    net: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    buffer: deque,
    *,
    num_players: int,
    K: int,
    batch_size: int,
    batches_per_step: int,
    grad_clip: float,
) -> float:
    """One step of belief-net KL training (Plan §2.3).

    Buffer entries: (features, hand_counts[N-1][K], remaining[K],
    alive_per_opp[N-1]). Per opp row i and role R:
      label[i][R] = (hand_counts[i][R] / remaining[R]) normalized along R
      mask_R[i][R] = (remaining[R] > 0)
      mask_row[i] = (alive_per_opp[i] > 0)
    Loss = mean over alive rows of KL(label || softmax(logits / 1.0)).
    Returns the average loss across `batches_per_step`.
    """
    if not buffer or net is None:
        return float("nan")
    n_opp = num_players - 1
    if n_opp <= 0 or K <= 0:
        return float("nan")
    buf = list(buffer)
    n = len(buf)
    feats_all = torch.tensor([s[0] for s in buf], dtype=torch.float32)
    hand_all = torch.tensor([s[1] for s in buf], dtype=torch.float32)  # [B,N-1,K]
    rem_all = torch.tensor([s[2] for s in buf], dtype=torch.float32)   # [B,K]
    alive_all = torch.tensor([s[3] for s in buf], dtype=torch.float32) # [B,N-1]

    net.train()
    total = 0.0
    eps = 1e-12
    for _ in range(batches_per_step):
        idx = torch.randint(n, (min(batch_size, n),))
        f = feats_all[idx]
        hand = hand_all[idx]
        rem = rem_all[idx]
        alive = alive_all[idx]
        B = f.size(0)
        logits = net(f).view(B, n_opp, K)

        rem_exp = rem.unsqueeze(1).expand(-1, n_opp, -1)             # [B,N-1,K]
        mask_R = (rem_exp > 0).float()
        unnorm = torch.where(mask_R.bool(), hand / (rem_exp + eps),
                             torch.zeros_like(hand))
        row_sum = unnorm.sum(dim=-1, keepdim=True)                   # [B,N-1,1]
        valid_row = (row_sum.squeeze(-1) > 0) & (alive > 0)          # [B,N-1]
        label = torch.where(row_sum > 0, unnorm / (row_sum + eps),
                            torch.zeros_like(unnorm))

        # log-softmax with -inf where mask_R = 0 (kill those R from softmax).
        neg_inf = torch.full_like(logits, float("-inf"))
        masked_logits = torch.where(mask_R.bool(), logits, neg_inf)
        log_pi = torch.nn.functional.log_softmax(masked_logits, dim=-1)
        # KL = sum_R label * (log label - log_pi); 0 * log0 := 0.
        log_label = torch.where(label > 0, torch.log(label + eps),
                                torch.zeros_like(label))
        kl_per_role = label * (log_label - log_pi)
        # Zero out masked R / dead rows safely.
        kl_per_role = torch.where(mask_R.bool(), kl_per_role,
                                   torch.zeros_like(kl_per_role))
        kl_row = kl_per_role.sum(dim=-1)                              # [B,N-1]
        kl_row = torch.where(valid_row, kl_row, torch.zeros_like(kl_row))

        n_valid = valid_row.float().sum().clamp_min(1.0)
        loss = kl_row.sum() / n_valid

        optimizer.zero_grad()
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(net.parameters(), grad_clip)
        optimizer.step()
        total += float(loss.item())
    net.eval()
    return total / max(1, batches_per_step)


def run_selfplay_batch(
    game_id: str,
    model_path: str,
    num_episodes: int,
    base_seed: int,
    train_cfg: dict,
    max_workers: int,
    pool_paths: list[str] | None = None,
    num_players: int = 1,
    pool_enabled: bool = True,
    self_ratio: float = 0.5,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Run a step's worth of selfplay, split between mirror and pool modes.

    Per the opponent-pool design (frozen-pool fictitious self-play):
    - `n_self = ceil(N * self_ratio)` workers run normal mirror selfplay
      (latest vs latest) and contribute every sample.
    - `n_pool = N - n_self` workers run latest vs a random historical
      checkpoint; the caller filters samples to the latest seat only.

    `pool_enabled=False` (training cfg `opponent_pool_enabled: false`)
    forces full mirror regardless of the pool — the feature is fully
    opt-out via game.json.

    `self_ratio` ∈ [0, 1] is read from training cfg
    `opponent_pool_self_ratio`. 0.5 = the original ceil/floor split.
    1.0 ⇒ all mirror; 0.0 ⇒ all pool (only meaningful if pool is non-empty).

    If `pool_paths` is empty (early steps before any checkpoint is saved
    OR pool disabled), n_pool is forced to 0 and every episode runs in
    mirror mode.

    Returns (episodes, stats) where stats = {"n_self": ..., "n_pool": ...,
    "pool_size": ..., "pool_enabled": ...}. Each episode dict carries
    `latest_seat: int | None` (None for mirror episodes).
    """
    if pool_paths is None:
        pool_paths = []
    if not pool_enabled:
        pool_paths = []
    pool_size = len(pool_paths)

    if not (0.0 <= self_ratio <= 1.0):
        raise ValueError(
            f"run_selfplay_batch: opponent_pool_self_ratio must be in [0, 1], "
            f"got {self_ratio}"
        )

    if pool_size == 0:
        n_self = num_episodes
        n_pool = 0
    else:
        n_self = math.ceil(num_episodes * self_ratio)
        n_self = max(0, min(num_episodes, n_self))
        n_pool = num_episodes - n_self

    tasks: list[tuple[str, tuple]] = []
    for i in range(n_self):
        args = (game_id, base_seed + i, model_path, train_cfg)
        tasks.append(("self", args))
    if n_pool > 0:
        rng = random.Random(base_seed)
        for i in range(n_pool):
            ep_idx = n_self + i
            ep_seed = base_seed + ep_idx
            latest_seat = rng.randrange(num_players)
            pool_path = rng.choice(pool_paths)
            model_paths_per_seat = [
                model_path if s == latest_seat else pool_path
                for s in range(num_players)
            ]
            args = (
                game_id, ep_seed, model_paths_per_seat,
                latest_seat, train_cfg,
            )
            tasks.append(("pool", args))

    results: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        for r in pool.map(_dispatch_worker, tasks):
            results.append(r)

    stats = {
        "n_self": n_self,
        "n_pool": n_pool,
        "pool_size": pool_size,
        "pool_enabled": pool_enabled,
    }
    return results, stats


def run_eval_vs_heuristic(
    game_id: str,
    model_path: str,
    num_games: int,
    base_seed: int,
    simulations: int,
    constrained: bool,
    heuristic_temperature: float,
    max_workers: int,
    opponent_selection: str,
) -> dict[str, Any]:
    import dinoboard_engine
    num_players = dinoboard_engine.game_metadata(game_id)["num_players"]
    tasks = []
    for i in range(num_games):
        # Rotate the model's seat across all N players so each seat carries
        # the same number of games. `i % 2` would only test seats 0 and 1,
        # leaving 3p/4p seats 2/3 unmeasured.
        model_side = i % num_players
        tasks.append((
            game_id, base_seed + i, model_path, simulations,
            model_side, constrained, heuristic_temperature,
            opponent_selection,
        ))

    wins = losses = draws = 0
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        for idx, r in enumerate(pool.map(_worker_eval_vs_heuristic, tasks)):
            model_side = tasks[idx][4]
            w = r["winner"]
            if r["draw"] or w < 0:
                draws += 1
            elif w == model_side:
                wins += 1
            else:
                losses += 1

    total = wins + losses + draws
    win_rate = wins / max(1, total)
    return {"wins": wins, "losses": losses, "draws": draws, "win_rate": win_rate}


def run_eval_batch(
    game_id: str,
    candidate_path: str,
    opponent_path: str,
    num_games: int,
    base_seed: int,
    sims_candidate: int,
    sims_opponent: int,
    max_workers: int,
    max_plies: int,
    temperature: float,
    opponent_selection: str,
    tail_solve_enabled: bool,
    tail_solve_depth_limit: int,
    tail_solve_node_budget: int,
    tail_solve_margin_weight: float,
) -> dict[str, Any]:
    import dinoboard_engine
    meta = dinoboard_engine.game_metadata(game_id)
    num_players = meta["num_players"]

    tasks = []
    candidate_seats = []
    for i in range(num_games):
        seat = i % num_players
        paths = [opponent_path] * num_players
        sims = [sims_opponent] * num_players
        paths[seat] = candidate_path
        sims[seat] = sims_candidate
        tasks.append((
            game_id, base_seed + i, paths, sims, max_plies,
            temperature, opponent_selection,
            tail_solve_enabled, tail_solve_depth_limit,
            tail_solve_node_budget, tail_solve_margin_weight,
        ))
        candidate_seats.append(seat)

    wins = losses = draws = 0
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        for idx, r in enumerate(pool.map(_worker_arena, tasks)):
            w = r["winner"]
            if r["draw"] or w < 0:
                draws += 1
            elif w == candidate_seats[idx]:
                wins += 1
            else:
                losses += 1

    total = wins + losses + draws
    win_rate = wins / max(1, total)
    return {"wins": wins, "losses": losses, "draws": draws, "win_rate": win_rate}


def rotate_z_values(z_values: list[float], player: int, num_players: int = 0) -> list[float]:
    if not z_values:
        raise ValueError("z_values is empty — C++ must always populate z_values for every sample")
    n = len(z_values)
    return [z_values[(player + i) % n] for i in range(n)]


def compute_schedule_ratio(
    step: int,
    decay_end_step: int,
    initial_ratio: float,
    hold_steps: int = 0,
) -> float:
    """Three-segment schedule: hold → linear decay → zero.

    - step <= hold_steps                       → initial_ratio
    - step >= decay_end_step                   → 0.0
    - hold_steps < step < decay_end_step       → linear interpolation

    decay_end_step <= 0 disables the schedule (always 0). Configuration
    error if 0 < decay_end_step <= hold_steps — caller must validate.
    """
    if decay_end_step <= 0:
        return 0.0
    if decay_end_step <= hold_steps:
        raise ValueError(
            f"compute_schedule_ratio: decay_end_step ({decay_end_step}) must be "
            f"> hold_steps ({hold_steps})"
        )
    if step <= hold_steps:
        return initial_ratio
    if step >= decay_end_step:
        return 0.0
    span = decay_end_step - hold_steps
    return initial_ratio * (decay_end_step - step) / span


def compute_lr(step: int, total_steps: int, base_lr: float, schedule: dict | None) -> float:
    """LR schedule. Returns base_lr when schedule is None or disabled.

    schedule = {"type": "cosine"|"step"|"constant", "lr_min": float, ...}
    - cosine: half-cosine from base_lr at step=1 to lr_min at step=total_steps
    - step: divide by `gamma` every `step_size` steps (default gamma=10, step_size=total_steps/3)
    - constant / None: base_lr unchanged
    """
    if not schedule or schedule.get("type", "constant") == "constant":
        return base_lr
    sched_type = schedule["type"]
    if sched_type == "cosine":
        lr_min = schedule.get("lr_min", base_lr * 0.01)
        if total_steps <= 1:
            return base_lr
        progress = (step - 1) / (total_steps - 1)
        progress = max(0.0, min(1.0, progress))
        return lr_min + (base_lr - lr_min) * 0.5 * (1.0 + math.cos(math.pi * progress))
    if sched_type == "step":
        gamma = schedule.get("gamma", 10.0)
        step_size = schedule.get("step_size", max(1, total_steps // 3))
        n_drops = (step - 1) // step_size
        return base_lr / (gamma ** n_drops)
    raise ValueError(f"unknown lr_schedule type: {sched_type!r}")


def run_training_loop(
    game_id: str,
    game_config: dict,
    output_dir: Path,
    steps: int = 1000,
    episodes_per_step: int = 200,
    eval_every: int = 50,
    eval_games: int = 40,
    eval_benchmarks: list[str] | None = None,
    max_workers: int = 4,
    batch_size: int = 512,
    learning_rate: float = 0.001,
    seed: int = 20260323,
    save_every: int = 0,
    init_from: str | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    models_dir = output_dir / "models"
    models_dir.mkdir(exist_ok=True)

    train_cfg = game_config["training"]
    action_space = game_config["action_space"]
    feature_dim = game_config["feature_dim"]
    if "num_players" not in game_config:
        raise KeyError(
            "game_config missing 'num_players'. Inject from engine.game_metadata(game_id) "
            "before calling run_training_loop — JSON 'players.max' is not authoritative."
        )
    num_players = game_config["num_players"]
    auxiliary_score = train_cfg.get("auxiliary_score", False)
    auxiliary_score_weight = train_cfg.get("auxiliary_score_weight", 0.5)

    # MCTS profiles drive selfplay / arena (gating) / eval. Top-level
    # max_game_plies is a game property shared across profiles.
    selfplay_profile = resolve_profile(game_id, "selfplay")
    arena_profile = resolve_profile(game_id, "arena")
    eval_profile = resolve_profile(game_id, "eval")
    game_max_plies = max_game_plies(game_id)

    if save_every <= 0:
        save_every = eval_every if eval_every > 0 else 50

    net = create_model_from_config(game_config)
    torch.manual_seed(seed)

    if init_from:
        init_path = Path(init_from)
        if not init_path.exists():
            raise FileNotFoundError(f"--init-from path does not exist: {init_path}")
        ckpt = torch.load(str(init_path), map_location="cpu", weights_only=False)
        if not isinstance(ckpt, dict) or "model_state_dict" not in ckpt:
            raise ValueError(
                f"init_from checkpoint must be a dict with 'model_state_dict' key; "
                f"got top-level: {list(ckpt.keys()) if isinstance(ckpt, dict) else type(ckpt)}")
        net.load_state_dict(ckpt["model_state_dict"])
        logger.info(f"Initialized weights from {init_path} (optimizer state reset)")

    weight_decay = train_cfg.get("weight_decay", 1e-4)
    optimizer = torch.optim.AdamW(net.parameters(), lr=learning_rate, weight_decay=weight_decay)
    lr_schedule = train_cfg.get("lr_schedule")  # None = constant lr

    initial_onnx = models_dir / "model_init.onnx"
    export_onnx(net, initial_onnx, feature_dim)

    current_model_path = str(initial_onnx)
    best_model_path = current_model_path

    # Scheduling parameters (three-segment: hold → linear decay → zero)
    heuristic_guidance_hold_steps = train_cfg.get("heuristic_guidance_hold_steps", 0)
    heuristic_guidance_steps = train_cfg.get("heuristic_guidance_steps", 0)
    heuristic_guidance_initial = train_cfg.get("heuristic_guidance_initial_ratio",
                                                train_cfg.get("heuristic_guidance_ratio", 0.5))
    training_filter_hold_steps = train_cfg.get("training_filter_hold_steps", 0)
    training_filter_steps = train_cfg.get("training_filter_steps", 0)
    training_filter_initial = train_cfg.get("training_filter_initial_ratio", 0.5)

    simulations_start = train_cfg.get("simulations_start", selfplay_profile.simulations)
    simulations_full = selfplay_profile.simulations

    # Opponent pool — opt-in via game.json. Default off (mirror selfplay) so
    # adding the feature didn't change any existing game's training behavior.
    # If `opponent_pool_enabled` is set without `opponent_pool_self_ratio`
    # we still throw — partial config is a bug.
    opponent_pool_enabled = train_cfg.get("opponent_pool_enabled", False)
    if opponent_pool_enabled:
        opponent_pool_self_ratio = train_cfg["opponent_pool_self_ratio"]
    else:
        opponent_pool_self_ratio = 1.0  # unused; full mirror

    replay_buffer_size = episodes_per_step * 50 * 20
    replay_buffer: deque[tuple[list, list, list, list, float]] = deque(maxlen=replay_buffer_size)

    # Belief net plumbing (Plan §2.2 / §2.3). Activated only when the game
    # registers a belief feature extractor AND a 'belief' block exists in
    # game.json. Coup is the only such game today; LL/Splendor will train
    # their own belief nets in future plans (no runtime branch on game id —
    # capability-driven).
    import dinoboard_engine as _engine
    meta = _engine.game_metadata(game_id)
    belief_enabled = meta["has_belief_extractor"] and "belief" in game_config
    belief_net = None
    belief_optimizer = None
    belief_buffer: deque[tuple[list, list, list, list]] | None = None
    belief_lr_schedule = None
    belief_batch_size = 1024
    belief_batches_per_step = 1
    belief_class_count = meta["belief_logit_count"]  # (N-1) * K — used for shape check
    current_belief_model_path = ""
    if belief_enabled:
        belief_cfg = game_config["belief"]
        belief_in = meta["belief_feature_dim"]
        belief_out = meta["belief_logit_count"]
        belief_net = create_belief_model_from_config(
            belief_cfg, input_dim=belief_in, output_dim=belief_out,
        )
        b_train = belief_cfg.get("training", {})
        belief_lr_init = float(b_train.get("lr_schedule", {}).get("lr_max", 3e-4))
        belief_lr_schedule = b_train.get("lr_schedule")
        belief_batch_size = int(b_train.get("batch_size", 1024))
        belief_batches_per_step = int(b_train.get("steps_per_pv_step", 1))
        belief_optimizer = torch.optim.Adam(belief_net.parameters(), lr=belief_lr_init)
        belief_buffer = deque(maxlen=replay_buffer_size)
        # Random-init export so step 1 selfplay can already use the net.
        belief_init_path = models_dir / "belief_init.onnx"
        export_belief_onnx(belief_net, belief_init_path, belief_in)
        current_belief_model_path = str(belief_init_path)
        # Per-opp character count K (kCharacterCount in C++).
        # logit_count = (N-1) * K, so K = logit_count // (N-1).
        belief_K = belief_out // max(1, num_players - 1) if num_players > 1 else belief_out
        logger.info(
            f"  belief: enabled (input_dim={belief_in}, logits={belief_out}, "
            f"K={belief_K}, batch={belief_batch_size}, "
            f"batches/step={belief_batches_per_step})"
        )
    else:
        belief_K = 0
        logger.info(f"  belief: disabled (no extractor or no 'belief' block)")

    logger.info(f"Starting training: game={game_id}, steps={steps}, episodes/step={episodes_per_step}")
    logger.info(
        f"  heuristic_guidance: hold={heuristic_guidance_hold_steps}, "
        f"decay_end={heuristic_guidance_steps}, initial_ratio={heuristic_guidance_initial}"
    )
    logger.info(
        f"  training_filter: hold={training_filter_hold_steps}, "
        f"decay_end={training_filter_steps}, initial_ratio={training_filter_initial}"
    )
    logger.info(f"  simulations: start={simulations_start}, full={simulations_full}")
    logger.info(f"  tail_solve (selfplay): enabled={selfplay_profile.tail_solve_enabled}")
    logger.info(f"  opponent_selection (selfplay): {selfplay_profile.opponent_selection}")
    logger.info(
        f"  opponent_pool: enabled={opponent_pool_enabled}, "
        f"self_ratio={opponent_pool_self_ratio}"
    )
    logger.info(f"  replay_buffer: maxlen={replay_buffer_size}")
    if lr_schedule:
        logger.info(f"  lr_schedule: {lr_schedule} (base_lr={learning_rate})")
    else:
        logger.info(f"  lr: {learning_rate} (constant)")

    for step in range(1, steps + 1):
        t0 = time.perf_counter()

        current_lr = compute_lr(step, steps, learning_rate, lr_schedule)
        for pg in optimizer.param_groups:
            pg["lr"] = current_lr

        # Compute scheduled ratios (three-segment: hold → decay → zero)
        heuristic_ratio = compute_schedule_ratio(
            step,
            heuristic_guidance_steps,
            heuristic_guidance_initial,
            hold_steps=heuristic_guidance_hold_steps,
        )
        filter_ratio = compute_schedule_ratio(
            step,
            training_filter_steps,
            training_filter_initial,
            hold_steps=training_filter_hold_steps,
        )

        # Ramp simulations
        sim_frac = min(1.0, step / max(1, steps * 0.3))
        current_sims = int(simulations_start + (simulations_full - simulations_start) * sim_frac)

        # Action filter is bool-gated by profile; the schedule's effective
        # ratio is zeroed when ai_use_action_filter=false (allows ablation
        # without deleting the schedule config).
        effective_filter_ratio = (
            filter_ratio if selfplay_profile.ai_use_action_filter else 0.0
        )

        selfplay_cfg = profile_as_dict(selfplay_profile)
        selfplay_cfg["simulations"] = current_sims
        selfplay_cfg["max_game_plies"] = game_max_plies
        selfplay_cfg["heuristic_guidance_ratio"] = heuristic_ratio
        # Selfplay heuristic branch uses guidance temperature (high = diverse
        # exploration during warm period). Eval vs heuristic uses the
        # separate `heuristic_temperature` (low = strength benchmark).
        selfplay_cfg["heuristic_temperature"] = train_cfg.get(
            "heuristic_guidance_temperature", 0.0)
        selfplay_cfg["training_filter_ratio"] = effective_filter_ratio
        selfplay_cfg["belief_model_path"] = current_belief_model_path

        # Recompute the pool every step — newly saved checkpoints automatically
        # enrol the step after they land. Empty pool ⇒ all episodes mirror.
        # Disabled via game.json `opponent_pool_enabled: false` ⇒ same.
        pool_paths = _collect_pool_paths(models_dir)
        episodes, sp_stats = run_selfplay_batch(
            game_id, current_model_path, episodes_per_step,
            seed + step * 10000, selfplay_cfg, max_workers,
            pool_paths=pool_paths, num_players=num_players,
            pool_enabled=opponent_pool_enabled,
            self_ratio=opponent_pool_self_ratio)
        logger.info(
            f"  pool: enabled={sp_stats['pool_enabled']}, "
            f"|P|={sp_stats['pool_size']}, "
            f"self={sp_stats['n_self']}, pool={sp_stats['n_pool']}"
        )

        # `step_samples` counts what actually entered the replay buffer
        # (after pool-mode filtering), not the raw episode-output count.
        step_samples = 0
        belief_step_samples = 0
        for ep in episodes:
            latest_seat = ep["latest_seat"]  # int for pool, None for mirror
            # Belief samples (Plan §2.2). One per ply per observer; pool-mode
            # filter applies — only keep observer == latest_seat. (The latest
            # net's belief is what we're training; freezing the pool's net
            # means its belief samples would train against the frozen weights.)
            if belief_enabled:
                for bs in ep.get("belief_samples", []):
                    if latest_seat is not None and bs["observer"] != latest_seat:
                        continue
                    belief_buffer.append((
                        bs["features"],
                        bs["hand_counts"],   # [N-1][K]
                        bs["remaining"],     # [K]
                        bs["alive_per_opp"], # [N-1]
                    ))
                    belief_step_samples += 1
            for sample in ep["samples"]:
                # Pool-mode: keep only the latest model's seat samples — the
                # opp's seat was driven by a frozen historical net and would
                # train against itself if we kept it.
                if latest_seat is not None and sample["player"] != latest_seat:
                    continue
                feats = sample["features"]
                if len(feats) != feature_dim:
                    raise ValueError(
                        f"step {step}: feature dim mismatch: got {len(feats)}, expected {feature_dim}")
                policy = normalize_policy(
                    sample["policy_action_ids"],
                    sample["policy_action_visits"],
                    action_space,
                )
                z_vals = sample["z_values"]
                player = sample["player"]
                z_rotated = rotate_z_values(z_vals, player, num_players)
                mask = sample["legal_mask"]
                aux = float(sample["auxiliary_score"])
                replay_buffer.append((feats, policy, z_rotated, mask, aux))
                step_samples += 1

        winners = [ep["winner"] for ep in episodes]
        per_player_wins = [sum(1 for w in winners if w == p) for p in range(num_players)]
        draws_count = sum(1 for w in winners if w < 0)

        ts_attempts = sum(ep["tail_solve_attempts"] for ep in episodes)
        ts_completed = sum(ep["tail_solve_completed"] for ep in episodes)
        ts_successes = sum(ep["tail_solve_successes"] for ep in episodes)

        if not replay_buffer:
            logger.warning(f"Step {step}: no training samples, skipping")
            continue

        buf = list(replay_buffer)
        all_features = [s[0] for s in buf]
        all_policies = [s[1] for s in buf]
        all_values = [s[2] for s in buf]
        all_masks = [s[3] for s in buf]
        all_aux = [s[4] for s in buf]

        feat_tensor = torch.tensor(all_features, dtype=torch.float32)
        policy_tensor = torch.tensor(all_policies, dtype=torch.float32)
        value_tensor = torch.tensor(all_values, dtype=torch.float32)
        mask_tensor = torch.tensor(all_masks, dtype=torch.float32)
        aux_tensor = torch.tensor(all_aux, dtype=torch.float32) if auxiliary_score else None

        n = feat_tensor.size(0)
        grad_clip = train_cfg.get("grad_clip_norm", 1.0)
        train_batches = train_cfg.get("train_batches_per_step", 3)
        total_loss = 0.0
        for b in range(train_batches):
            idx = torch.randint(n, (min(batch_size, n),))
            metrics = train_step(
                net, optimizer,
                feat_tensor[idx], policy_tensor[idx], value_tensor[idx],
                legal_mask=mask_tensor[idx],
                auxiliary_targets=aux_tensor[idx] if aux_tensor is not None else None,
                auxiliary_weight=auxiliary_score_weight,
                grad_clip_norm=grad_clip,
            )
            total_loss += metrics["loss"]
        avg_loss = total_loss / max(1, train_batches)

        latest_onnx = models_dir / "model_latest.onnx"
        export_onnx(net, latest_onnx, feature_dim)
        current_model_path = str(latest_onnx)

        if step % save_every == 0:
            step_onnx = models_dir / f"model_step_{step:05d}.onnx"
            export_onnx(net, step_onnx, feature_dim)

        # Belief training step (Plan §2.3) — KL(label || pi) where
        # label[i][R] = (hand_counts[i][R] / remaining[R]) / row_norm,
        # masked by remaining > 0 and alive_per_opp[i] > 0. This trains the
        # Wallenius bias factor (label * remaining = q is the truth marginal).
        belief_loss_val = None
        if belief_enabled and belief_buffer:
            belief_lr = compute_lr(step, steps, belief_lr_init, belief_lr_schedule)
            for pg in belief_optimizer.param_groups:
                pg["lr"] = belief_lr
            belief_loss_val = _train_belief_step(
                belief_net, belief_optimizer, belief_buffer,
                num_players=num_players, K=belief_K,
                batch_size=belief_batch_size,
                batches_per_step=belief_batches_per_step,
                grad_clip=train_cfg.get("grad_clip_norm", 1.0),
            )
            if not math.isfinite(belief_loss_val):
                raise RuntimeError(
                    f"belief loss diverged: {belief_loss_val} — abort to "
                    "prevent training-on-NaN drift (Plan §2.4)"
                )
            belief_latest = models_dir / "belief_latest.onnx"
            export_belief_onnx(belief_net, belief_latest, meta["belief_feature_dim"])
            current_belief_model_path = str(belief_latest)
            if step % save_every == 0:
                belief_step_onnx = models_dir / f"belief_step_{step:05d}.onnx"
                export_belief_onnx(
                    belief_net, belief_step_onnx, meta["belief_feature_dim"],
                )

        elapsed = time.perf_counter() - t0
        log_parts = [
            f"Step {step}/{steps}: loss={avg_loss:.4f}",
            f"episodes={len(episodes)}, samples={n}",
            ", ".join(f"p{p}={per_player_wins[p]}" for p in range(num_players)) + f", draw={draws_count}",
            f"sims={current_sims}",
        ]
        if lr_schedule:
            log_parts.append(f"lr={current_lr:.2e}")
        if heuristic_ratio > 0:
            log_parts.append(f"h_ratio={heuristic_ratio:.2f}")
        if filter_ratio > 0:
            log_parts.append(f"f_ratio={filter_ratio:.2f}")
        if ts_attempts > 0:
            log_parts.append(f"ts={ts_successes}/{ts_completed}/{ts_attempts}")
        sps = step_samples / max(0.01, elapsed)
        mps = sps * current_sims
        log_parts.append(f"{sps:.0f}smp/s, {mps:.0f}mcts/s")
        if belief_enabled:
            buf_n = len(belief_buffer) if belief_buffer is not None else 0
            if belief_loss_val is not None:
                log_parts.append(
                    f"b_loss={belief_loss_val:.4f}, b_buf={buf_n}, "
                    f"b_step={belief_step_samples}"
                )
            else:
                log_parts.append(f"b_buf={buf_n}, b_step={belief_step_samples}")
        # Game-defined per-episode stats (averaged over the step's
        # episodes). Splendor's extractor returns {"turns": main_actions
        # / num_players}, which excludes return-token sub-actions and
        # noble-pick sub-actions so the count reflects real game rounds,
        # not raw plies.
        custom_keys: dict[str, list[float]] = {}
        for ep in episodes:
            cs = ep.get("custom_stats")
            if not cs:
                continue
            for k, v in cs.items():
                custom_keys.setdefault(k, []).append(float(v))
        for k, vs in sorted(custom_keys.items()):
            log_parts.append(f"{k}={sum(vs) / len(vs):.1f}")
        log_parts.append(f"time={elapsed:.1f}s")
        logger.info(", ".join(log_parts))

        if eval_every > 0 and step % eval_every == 0:
            eval_model = current_model_path
            h_temp = train_cfg.get("heuristic_temperature", 0.0)
            free_h_temp = train_cfg.get("free_heuristic_temperature", h_temp)
            benchmarks = eval_benchmarks or []

            for bench in benchmarks:
                if bench == "heuristic_constrained":
                    r = run_eval_vs_heuristic(
                        game_id, eval_model, eval_games, seed + step * 100000,
                        eval_profile.simulations, True, h_temp, max_workers,
                        opponent_selection=eval_profile.opponent_selection)
                    logger.info(
                        f"  eval vs heuristic (constrained): win_rate={r['win_rate']:.1%} "
                        f"(W={r['wins']}, L={r['losses']}, D={r['draws']})")
                elif bench == "heuristic_free":
                    r = run_eval_vs_heuristic(
                        game_id, eval_model, eval_games, seed + step * 100000 + 50000,
                        eval_profile.simulations, False, free_h_temp, max_workers,
                        opponent_selection=eval_profile.opponent_selection)
                    logger.info(
                        f"  eval vs heuristic (free): win_rate={r['win_rate']:.1%} "
                        f"(W={r['wins']}, L={r['losses']}, D={r['draws']})")
                else:
                    bench_result = run_eval_batch(
                        game_id, eval_model, bench,
                        eval_games, seed + step * 100000 + 90000,
                        eval_profile.simulations, eval_profile.simulations, max_workers,
                        max_plies=game_max_plies,
                        temperature=eval_profile.temperature,
                        opponent_selection=eval_profile.opponent_selection,
                        tail_solve_enabled=eval_profile.tail_solve_enabled,
                        tail_solve_depth_limit=eval_profile.tail_solve_depth_limit,
                        tail_solve_node_budget=eval_profile.tail_solve_node_budget,
                        tail_solve_margin_weight=eval_profile.tail_solve_margin_weight,
                    )
                    logger.info(
                        f"  eval vs {Path(bench).stem}: win_rate={bench_result['win_rate']:.1%} "
                        f"(W={bench_result['wins']}, L={bench_result['losses']}, D={bench_result['draws']})")

            # Gating: latest vs best (always runs) — uses arena profile so
            # gating and cross-ONNX arena measure the same MCTS behavior.
            gating_result = run_eval_batch(
                game_id, eval_model, best_model_path,
                eval_games, seed + step * 100000 + 80000,
                arena_profile.simulations, arena_profile.simulations, max_workers,
                max_plies=game_max_plies,
                temperature=arena_profile.temperature,
                opponent_selection=arena_profile.opponent_selection,
                tail_solve_enabled=arena_profile.tail_solve_enabled,
                tail_solve_depth_limit=arena_profile.tail_solve_depth_limit,
                tail_solve_node_budget=arena_profile.tail_solve_node_budget,
                tail_solve_margin_weight=arena_profile.tail_solve_margin_weight,
            )
            gating_wr = gating_result["win_rate"]
            logger.info(
                f"  gating vs best: win_rate={gating_wr:.1%} "
                f"(W={gating_result['wins']}, L={gating_result['losses']}, D={gating_result['draws']})")
            gating_threshold = train_cfg.get(
                "gating_accept_win_rate",
                _default_gating_threshold(num_players, eval_games))
            if gating_wr >= gating_threshold:
                best_onnx = models_dir / "model_best.onnx"
                shutil.copy2(eval_model, best_onnx)
                best_model_path = str(best_onnx)
                logger.info(f"  new best model: step {step}, win_rate={gating_wr:.1%}")

        checkpoint = {
            "step": step,
            "model_state_dict": net.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }
        torch.save(checkpoint, output_dir / "checkpoint.pt")

    logger.info(f"Training complete. Best model: {best_model_path}")
