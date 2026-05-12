"""Self-play training pipeline: selfplay -> collect samples -> train -> export -> repeat."""
from __future__ import annotations

import json
import logging
import math
import shutil
import time
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import torch

from .mcts_profile import MctsProfile, max_game_plies, profile_as_dict, resolve_profile
from .model import PVNet, create_model_from_config, export_onnx

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
    )


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


def run_selfplay_batch(
    game_id: str,
    model_path: str,
    num_episodes: int,
    base_seed: int,
    train_cfg: dict,
    max_workers: int,
) -> list[dict[str, Any]]:
    tasks = [
        (game_id, base_seed + i, model_path, train_cfg)
        for i in range(num_episodes)
    ]
    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        for r in pool.map(_worker_selfplay, tasks):
            results.append(r)
    return results


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
    weight_decay = train_cfg.get("weight_decay", 1e-4)
    optimizer = torch.optim.AdamW(net.parameters(), lr=learning_rate, weight_decay=weight_decay)

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

    replay_buffer_size = episodes_per_step * 50 * 20
    replay_buffer: deque[tuple[list, list, list, list, float]] = deque(maxlen=replay_buffer_size)

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
    logger.info(f"  replay_buffer: maxlen={replay_buffer_size}")

    for step in range(1, steps + 1):
        t0 = time.perf_counter()

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

        episodes = run_selfplay_batch(
            game_id, current_model_path, episodes_per_step,
            seed + step * 10000, selfplay_cfg, max_workers)

        for ep in episodes:
            for sample in ep["samples"]:
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

        winners = [ep["winner"] for ep in episodes]
        per_player_wins = [sum(1 for w in winners if w == p) for p in range(num_players)]
        draws_count = sum(1 for w in winners if w < 0)

        step_samples = sum(len(ep["samples"]) for ep in episodes)

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

        elapsed = time.perf_counter() - t0
        log_parts = [
            f"Step {step}/{steps}: loss={avg_loss:.4f}",
            f"episodes={len(episodes)}, samples={n}",
            ", ".join(f"p{p}={per_player_wins[p]}" for p in range(num_players)) + f", draw={draws_count}",
            f"sims={current_sims}",
        ]
        if heuristic_ratio > 0:
            log_parts.append(f"h_ratio={heuristic_ratio:.2f}")
        if filter_ratio > 0:
            log_parts.append(f"f_ratio={filter_ratio:.2f}")
        if ts_attempts > 0:
            log_parts.append(f"ts={ts_successes}/{ts_completed}/{ts_attempts}")
        sps = step_samples / max(0.01, elapsed)
        mps = sps * current_sims
        log_parts.append(f"{sps:.0f}smp/s, {mps:.0f}mcts/s")
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
