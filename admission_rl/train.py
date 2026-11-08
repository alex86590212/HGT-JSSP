"""PPO training of the admission HGT on the gap-free admission model.

Each update collects `rollout_episodes` episodes in parallel worker
processes (arrival rate sampled uniformly from [rate_min, rate_max]), each
paired with an iGreedy run on the same arrivals for the baseline/return and
for logging. A quick paired evaluation against iGreedy on the fixed
eval-scenario stream runs at the start and every `eval_interval` episodes.

Usage:
  PYTHONPATH=. python -m admission_rl.train --config admission_rl/config.yaml \
      --output runs/admission_rl
  PYTHONPATH=. python -m admission_rl.train --resume runs/admission_rl/checkpoint_last.pt \
      --output runs/admission_rl
"""

from __future__ import annotations

import argparse
import csv
import multiprocessing as mp
import os
import random
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Dict, List, Optional

import torch
from torch.utils.tensorboard import SummaryWriter

from admission_rl.common import eval_scenarios, load_config, make_policy, pack, unpack, worker_init
from admission_rl.ppo import compute_gae, ppo_update
from admission_rl.rollout import policy_decide, run_rl_episode
from dynamic_scheduler.data.traffic_generator import TrafficGenerator
from dynamic_scheduler.environment.gapfree_admission import igreedy_admission_select, run_admission_episode

TIE_EPS = 1e-6
BUCKETS = (("low", 0.0, 0.45), ("mid", 0.45, 0.8), ("high", 0.8, float("inf")))

# --- worker side --------------------------------------------------------

_WORKER: Dict[str, Any] = {"version": None, "policy": None}


def _worker_policy(model_cfg: Dict[str, Any], version: int, state: bytes):
    if _WORKER["version"] != version:
        policy = make_policy(model_cfg)
        policy.load_state_dict(unpack(state))
        policy.eval()
        _WORKER["policy"], _WORKER["version"] = policy, version
    return _WORKER["policy"]


def _episode_kwargs(env_cfg: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "episode_duration": env_cfg["episode_duration"],
        "detection_window": env_cfg["detection_window"],
        "commit_window": env_cfg["commit_window"],
        "allow_defer": env_cfg["allow_defer"],
        "defer_step": env_cfg["defer_step"],
        "max_hold": env_cfg.get("max_hold"),
    }


def _igreedy_wait(arrivals, env_cfg) -> float:
    wait, _, _ = run_admission_episode(
        arrivals, env_cfg["episode_duration"], env_cfg["detection_window"],
        igreedy_admission_select, commit_window=env_cfg["commit_window"],
    )
    return wait


def rollout_task(state: bytes, version: int, cfg: Dict[str, Any], episode_seed: int, rate: float) -> bytes:
    policy = _worker_policy(cfg["model"], version, state)
    torch.manual_seed(episode_seed)
    env_cfg, train_cfg = cfg["env"], cfg["train"]
    arrivals = TrafficGenerator(seed=episode_seed).generate_episode_arrivals(
        env_cfg["episode_duration"], arrival_rate=rate)
    ig_wait = _igreedy_wait(arrivals, env_cfg)
    transitions, stats = run_rl_episode(
        policy_decide(policy), arrivals, **_episode_kwargs(env_cfg),
        reward_scale=train_cfg["reward_scale"],
        baseline_wait=ig_wait if train_cfg["igreedy_baseline"] else None,
    )
    compute_gae(transitions, cfg["ppo"]["gamma"], cfg["ppo"]["gae_lambda"])
    stats.update(rate=rate, igreedy_wait=ig_wait)
    return pack((transitions, stats))


def eval_task(state: bytes, version: int, cfg: Dict[str, Any], rate: float, seed: int) -> List[Dict[str, float]]:
    policy = _worker_policy(cfg["model"], version, state)
    env_cfg = cfg["env"]
    out = []
    for arrivals in eval_scenarios(rate, seed, cfg["eval"]["n_scenarios"], env_cfg["episode_duration"]):
        _, stats = run_rl_episode(policy_decide(policy, deterministic=True), arrivals,
                                  **_episode_kwargs(env_cfg), record=False)
        out.append({"hgt": stats["waiting_time"], "igreedy": _igreedy_wait(arrivals, env_cfg),
                    "n_decisions": stats["n_decisions"], "n_defers": stats["n_defers"]})
    return out


# --- main side ----------------------------------------------------------

def episode_seed(cfg: Dict[str, Any], episode: int) -> int:
    # Disjoint from eval seeds (seed*100 + int(rate*1000) < 10^5).
    return 1_000_000 * (1 + cfg["seed"]) + episode


def improvement_pct(hgt: List[float], ig: List[float]) -> float:
    s_ig = sum(ig)
    return (s_ig - sum(hgt)) / s_ig * 100.0 if s_ig > 1e-9 else 0.0


def win_tie_loss(hgt: List[float], ig: List[float]):
    wins = sum(1 for h, g in zip(hgt, ig) if h < g - TIE_EPS)
    losses = sum(1 for h, g in zip(hgt, ig) if h > g + TIE_EPS)
    return wins, len(hgt) - wins - losses, losses


def append_csv(path: str, row: Dict[str, Any]) -> None:
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        if new:
            writer.writeheader()
        writer.writerow(row)


def run_eval(pool, state, version, cfg) -> Dict[str, Any]:
    ecfg = cfg["eval"]
    jobs = {(rate, seed): pool.submit(eval_task, state, version, cfg, rate, seed)
            for rate in ecfg["rates"] for seed in ecfg["seeds"]}
    per_rate = {}
    for rate in ecfg["rates"]:
        res = [r for seed in ecfg["seeds"] for r in jobs[(rate, seed)].result()]
        hgt, ig = [r["hgt"] for r in res], [r["igreedy"] for r in res]
        n_dec = sum(r["n_decisions"] for r in res)
        per_rate[rate] = {
            "hgt": sum(hgt) / len(hgt), "igreedy": sum(ig) / len(ig),
            "improvement_pct": improvement_pct(hgt, ig), "wtl": win_tie_loss(hgt, ig),
            "defer_frac": sum(r["n_defers"] for r in res) / n_dec if n_dec else 0.0,
        }
    mean_hgt = sum(r["hgt"] for r in per_rate.values()) / len(per_rate)
    mean_ig = sum(r["igreedy"] for r in per_rate.values()) / len(per_rate)
    return {"per_rate": per_rate, "mean_hgt": mean_hgt, "mean_igreedy": mean_ig,
            "improvement_pct": (mean_ig - mean_hgt) / mean_ig * 100.0 if mean_ig > 1e-9 else 0.0,
            "worst_rate_pct": min(r["improvement_pct"] for r in per_rate.values())}


def log_eval(ev, episode, writer, out_dir) -> None:
    for rate, r in ev["per_rate"].items():
        w, t, l = r["wtl"]
        writer.add_scalar(f"eval_rate/{rate:.2f}_improvement_pct", r["improvement_pct"], episode)
        writer.add_scalar(f"eval_rate/{rate:.2f}_hgt_wait", r["hgt"], episode)
        append_csv(os.path.join(out_dir, "eval_log.csv"), {
            "episode": episode, "rate": rate, "hgt_wait": r["hgt"], "igreedy_wait": r["igreedy"],
            "improvement_pct": r["improvement_pct"], "wins": w, "ties": t, "losses": l,
            "defer_frac": r["defer_frac"],
        })
    writer.add_scalar("eval/mean_hgt_wait", ev["mean_hgt"], episode)
    writer.add_scalar("eval/mean_igreedy_wait", ev["mean_igreedy"], episode)
    writer.add_scalar("eval/improvement_pct", ev["improvement_pct"], episode)
    writer.add_scalar("eval/worst_rate_improvement_pct", ev["worst_rate_pct"], episode)
    rates = "  ".join(f"{rate:.2f}:{r['improvement_pct']:+.1f}%" for rate, r in ev["per_rate"].items())
    print(f"[eval ep {episode}] mean wait HGT={ev['mean_hgt']:.4f} iGreedy={ev['mean_igreedy']:.4f} "
          f"({ev['improvement_pct']:+.2f}%, worst rate {ev['worst_rate_pct']:+.1f}%)\n    {rates}", flush=True)


def log_train(window: List[Dict[str, float]], losses: Dict[str, float], episode: int, total: int,
              entropy_coef: float, sec_per_ep: float, writer, out_dir: str) -> None:
    hgt = [s["waiting_time"] for s in window]
    ig = [s["igreedy_wait"] for s in window]
    w, t, l = win_tie_loss(hgt, ig)
    n_dec = sum(s["n_decisions"] for s in window)
    defer_frac = sum(s["n_defers"] for s in window) / n_dec if n_dec else 0.0
    imp = improvement_pct(hgt, ig)
    mean_rate = sum(s["rate"] for s in window) / len(window)

    writer.add_scalar("train/hgt_wait", sum(hgt) / len(hgt), episode)
    writer.add_scalar("train/igreedy_wait", sum(ig) / len(ig), episode)
    writer.add_scalar("train/improvement_pct", imp, episode)
    writer.add_scalar("train/win_rate", w / len(window), episode)
    writer.add_scalar("train/defer_frac", defer_frac, episode)
    writer.add_scalar("train/decisions_per_episode", n_dec / len(window), episode)
    writer.add_scalar("train/entropy_coef", entropy_coef, episode)
    writer.add_scalar("train/sec_per_episode", sec_per_ep, episode)
    for k, v in losses.items():
        writer.add_scalar(f"ppo/{k}", v, episode)
    for name, lo, hi in BUCKETS:
        sub = [s for s in window if lo <= s["rate"] < hi]
        if sub:
            writer.add_scalar(f"train_bucket/{name}_improvement_pct",
                              improvement_pct([s["waiting_time"] for s in sub], [s["igreedy_wait"] for s in sub]),
                              episode)

    eta_h = sec_per_ep * (total - episode) / 3600.0
    print(f"ep {episode:>6}/{total}  rate={mean_rate:.2f}  wait HGT={sum(hgt) / len(hgt):.4f} "
          f"iG={sum(ig) / len(ig):.4f} ({imp:+.2f}%)  W/T/L={w}/{t}/{l}  defer={defer_frac:.3f}  "
          f"dec/ep={n_dec / len(window):.0f}  actor={losses['actor_loss']:.4f} "
          f"critic={losses['critic_loss']:.4f} ent={losses['entropy']:.3f} "
          f"kl={losses['approx_kl']:.4f}  {sec_per_ep:.2f}s/ep  ETA {eta_h:.1f}h", flush=True)

    append_csv(os.path.join(out_dir, "train_log.csv"), {
        "episode": episode, "mean_rate": mean_rate, "hgt_wait": sum(hgt) / len(hgt),
        "igreedy_wait": sum(ig) / len(ig), "improvement_pct": imp, "wins": w, "ties": t, "losses": l,
        "defer_frac": defer_frac, "decisions_per_episode": n_dec / len(window),
        **losses, "entropy_coef": entropy_coef, "sec_per_episode": sec_per_ep,
    })


def save_checkpoint(path, policy, optimizer, episode, cfg, best_metric, version) -> None:
    tmp = path + ".tmp"
    torch.save({"policy": policy.state_dict(), "optimizer": optimizer.state_dict(), "episode": episode,
                "config": cfg, "best_metric": best_metric, "version": version}, tmp)
    os.replace(tmp, path)


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None, help="YAML config (default: admission_rl/config.yaml)")
    ap.add_argument("--output", default="runs/admission_rl")
    ap.add_argument("--resume", default=None, help="checkpoint_last.pt to continue from")
    ap.add_argument("--episodes", type=int, default=None, help="override train.num_episodes")
    ap.add_argument("--workers", type=int, default=None, help="override train.num_workers")
    ap.add_argument("--no-defer", action="store_true", help="order-only policy (env.allow_defer=false)")
    args = ap.parse_args(argv)

    ckpt = torch.load(args.resume, weights_only=False) if args.resume else None  # noqa: S301
    if args.config or ckpt is None:
        cfg = load_config(args.config or os.path.join(os.path.dirname(__file__), "config.yaml"))
    else:
        cfg = ckpt["config"]
    if args.episodes is not None:
        cfg["train"]["num_episodes"] = args.episodes
    if args.workers is not None:
        cfg["train"]["num_workers"] = args.workers
    if args.no_defer:
        cfg["env"]["allow_defer"] = False
    tcfg, pcfg = cfg["train"], cfg["ppo"]

    os.makedirs(args.output, exist_ok=True)
    torch.manual_seed(cfg["seed"])
    random.seed(cfg["seed"])
    torch.set_num_threads(tcfg["torch_threads"])

    policy = make_policy(cfg["model"])
    optimizer = torch.optim.Adam(policy.parameters(), lr=pcfg["lr"])
    episode, best_metric, version = 0, float("inf"), 0
    if ckpt is not None:
        policy.load_state_dict(ckpt["policy"])
        optimizer.load_state_dict(ckpt["optimizer"])
        episode, best_metric, version = ckpt["episode"], ckpt["best_metric"], ckpt["version"]
        print(f"Resumed from {args.resume} at episode {episode} (best eval wait {best_metric:.4f})", flush=True)

    total = tcfg["num_episodes"]
    n_params = sum(p.numel() for p in policy.parameters())
    print(f"Training admission HGT: {total} episodes, rate U[{tcfg['rate_min']}, {tcfg['rate_max']}], "
          f"defer={'on' if cfg['env']['allow_defer'] else 'off'}, workers={tcfg['num_workers']}, "
          f"params={n_params:,}, output={args.output}", flush=True)

    writer = SummaryWriter(os.path.join(args.output, "tb"))
    last_path = os.path.join(args.output, "checkpoint_last.pt")
    best_path = os.path.join(args.output, "checkpoint_best.pt")

    def maybe_eval(state):
        nonlocal best_metric
        ev = run_eval(pool, state, version, cfg)
        log_eval(ev, episode, writer, args.output)
        if ev["mean_hgt"] < best_metric:
            best_metric = ev["mean_hgt"]
            save_checkpoint(best_path, policy, optimizer, episode, cfg, best_metric, version)
            print(f"    new best -> {best_path}", flush=True)

    window: List[Dict[str, float]] = []
    losses = dict.fromkeys(("actor_loss", "critic_loss", "entropy", "approx_kl", "clip_frac"), 0.0)
    started = time.time()
    start_episode = episode
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=tcfg["num_workers"], mp_context=ctx, initializer=worker_init) as pool:
        if episode == 0:
            version += 1
            maybe_eval(pack(policy.state_dict()))

        while episode < total:
            n = min(tcfg["rollout_episodes"], total - episode)
            version += 1
            state = pack(policy.state_dict())
            futures = []
            for i in range(n):
                seed = episode_seed(cfg, episode + i)
                rate = random.Random(seed).uniform(tcfg["rate_min"], tcfg["rate_max"])
                futures.append(pool.submit(rollout_task, state, version, cfg, seed, rate))
            transitions = []
            for fut in futures:
                trs, stats = unpack(fut.result())
                transitions.extend(trs)
                window.append(stats)

            frac = episode / max(total, 1)
            entropy_coef = pcfg["entropy_coef"] + frac * (pcfg["entropy_coef_final"] - pcfg["entropy_coef"])
            if transitions:
                policy.train()
                losses = ppo_update(
                    policy, optimizer, transitions,
                    clip_epsilon=pcfg["clip_epsilon"], epochs=pcfg["epochs"],
                    mini_batch_size=pcfg["mini_batch_size"], value_loss_coef=pcfg["value_loss_coef"],
                    entropy_coef=entropy_coef, max_grad_norm=pcfg["max_grad_norm"],
                )
                policy.eval()

            prev, episode = episode, episode + n
            sec_per_ep = (time.time() - started) / (episode - start_episode)
            if episode // tcfg["log_interval"] > prev // tcfg["log_interval"] or episode == total:
                log_train(window, losses, episode, total, entropy_coef, sec_per_ep, writer, args.output)
                window = []
            if episode // tcfg["checkpoint_interval"] > prev // tcfg["checkpoint_interval"] or episode == total:
                save_checkpoint(last_path, policy, optimizer, episode, cfg, best_metric, version)
            if episode // tcfg["eval_interval"] > prev // tcfg["eval_interval"] or episode == total:
                version += 1
                maybe_eval(pack(policy.state_dict()))
                save_checkpoint(last_path, policy, optimizer, episode, cfg, best_metric, version)

    writer.close()
    print(f"Done in {(time.time() - started) / 3600:.2f}h. Best eval mean wait {best_metric:.4f} "
          f"-> {best_path}", flush=True)


if __name__ == "__main__":
    main()
