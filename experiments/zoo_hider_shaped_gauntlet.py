#!/usr/bin/env python3
"""
Compute Forgetting Regret (FR) for each zoo_hider_shaped run via checkpoint gauntlet.

Same logic as zoo_asweep_gauntlet.py but adapted for the hider-shaped sweep
(5 A values instead of 7, different base directory).

Usage:
  # Single run by task ID (for SLURM array)
  python experiments/zoo_hider_shaped_gauntlet.py --task-id $SLURM_ARRAY_TASK_ID

  # Dry-run to see task list
  python experiments/zoo_hider_shaped_gauntlet.py --list

  # Aggregate results after all tasks finish
  python experiments/zoo_hider_shaped_gauntlet.py --aggregate
"""
import argparse
import csv
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from trainer.tag_env import VecTagEnv, TagEnvConfig
from trainer.ppo import PPOAgent, PPOConfig


# ── Sweep axes (same as zoo_hider_shaped_tasks.py) ──────────────────
STPS = [0.005, 0.01, 0.02, 0.05]
HSMS = [1.0, 1.05, 1.1, 1.15, 1.2]
A_VALUES = [0.05, 0.1, 0.2, 0.3, 0.5]
SAMPLING_MODES = ["uniform", "thompson_loss"]
SEEDS = [0, 1, 2]

CKPT_SUBSAMPLE = 4  # take every 4th checkpoint (~12 from 48)
N_EVAL_EPISODES = 20
MAX_STEPS = 200
LAYOUT = "four_corners"
BASE_DIR = Path("experiments/results/zoo_hider_shaped")
OUTPUT_DIR = BASE_DIR / "gauntlet"


def stp_str(stp):
    return f"{stp:.4f}".replace("0.", "").rstrip("0") or "0"


def build_task_table():
    tasks = []
    for stp, hsm, A, sampling, seed in itertools.product(
        STPS, HSMS, A_VALUES, SAMPLING_MODES, SEEDS
    ):
        hsm_s = f"{round(hsm * 100)}"
        config_name = f"STP{stp_str(stp)}_HSM{hsm_s}"
        a_str = f"A{int(A * 100):02d}"
        run_name = f"{config_name}/{a_str}_{sampling}/seed_{seed}"
        tasks.append(dict(
            run_name=run_name,
            config_name=config_name,
            stp=stp,
            hsm=hsm,
            A=A,
            sampling=sampling,
            seed=seed,
        ))
    return tasks


def find_checkpoints(run_dir):
    """Find all seeker/hider checkpoint pairs, return sorted by update number."""
    subdirs = sorted(run_dir.glob("2*"), key=lambda p: p.name)
    if not subdirs:
        return []
    ckpt_dir = subdirs[-1] / "checkpoints"
    if not ckpt_dir.exists():
        return []

    seeker_ckpts = sorted(ckpt_dir.glob("seeker_*.pt"))
    updates = []
    for p in seeker_ckpts:
        update = int(p.stem.split("_")[1])
        hider_path = ckpt_dir / f"hider_{update:05d}.pt"
        if hider_path.exists():
            updates.append((update, str(p), str(hider_path)))
    return updates


def load_policy(path, obs_dim, act_dim):
    cfg = PPOConfig(obs_dim=obs_dim, act_dim=act_dim)
    policy = PPOAgent(cfg)
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    policy.pi.load_state_dict(ckpt["pi"])
    policy.vf.load_state_dict(ckpt["vf"])
    return policy


def batch_act(policy, obs_batch):
    """Sample actions for a batch of observations. Returns numpy actions."""
    with torch.no_grad():
        x = torch.as_tensor(obs_batch, dtype=torch.float32)
        logits = policy.pi(x)
        mean, log_std = torch.chunk(logits, 2, dim=-1)
        log_std = torch.clamp(log_std, -2.0, 1.5)
        std = torch.exp(log_std)
        action = torch.tanh(torch.distributions.Normal(mean, std).sample())
        return action.cpu().numpy()


def evaluate_matchup(seeker, hider, env_config, n_episodes=N_EVAL_EPISODES):
    """Evaluate seeker vs hider using vectorized env for speed."""
    env = VecTagEnv(num_envs=n_episodes, config=env_config)
    obs = env.reset()
    active = np.ones(n_episodes, dtype=bool)
    tagged = np.zeros(n_episodes, dtype=bool)

    for step in range(MAX_STEPS):
        if not active.any():
            break
        s_act = batch_act(seeker, obs["seeker"])
        h_act = batch_act(hider, obs["hider"])
        obs, _, dones, info = env.step({"seeker": s_act, "hider": h_act})
        newly_done = dones & active
        if newly_done.any():
            tagged[newly_done] = info.get("tagged", np.zeros(n_episodes, dtype=bool))[newly_done]
            active[newly_done] = False

    return tagged.sum() / n_episodes


def forgetting_regret(win_matrix):
    W = np.array(win_matrix)
    running_max = np.maximum.accumulate(W, axis=0)
    pr = running_max - W
    return float(pr.mean()), float(pr[-1].mean())


def run_gauntlet(task, dry_run=False):
    run_dir = BASE_DIR / task["run_name"]
    if not run_dir.exists():
        print(f"  SKIP: {run_dir} does not exist")
        return None

    all_ckpts = find_checkpoints(run_dir)
    if not all_ckpts:
        print(f"  SKIP: no checkpoints in {run_dir}")
        return None

    # Subsample
    selected = all_ckpts[::CKPT_SUBSAMPLE]
    if all_ckpts[-1] not in selected:
        selected.append(all_ckpts[-1])
    n = len(selected)

    print(f"  {task['run_name']}: {n} checkpoints, {n*n} matchups")
    if dry_run:
        return None

    env_config = TagEnvConfig(
        layout=LAYOUT,
        hider_speed_mult=task["hsm"],
    )
    env = VecTagEnv(num_envs=1, config=env_config)
    obs_dim, act_dim = env.obs_dim, env.act_dim

    # Load all policies
    seekers = []
    hiders = []
    updates = []
    for update, s_path, h_path in selected:
        seekers.append(load_policy(s_path, obs_dim, act_dim))
        hiders.append(load_policy(h_path, obs_dim, act_dim))
        updates.append(update)

    # Run gauntlet
    win_matrix = np.zeros((n, n))
    total = n * n
    for i in range(n):
        for j in range(n):
            wr = evaluate_matchup(seekers[i], hiders[j], env_config)
            win_matrix[i, j] = wr
            idx = i * n + j + 1
            if idx % 20 == 0 or idx == total:
                print(f"    [{idx}/{total}] seeker_{updates[i]:05d} vs hider_{updates[j]:05d}: {wr:.2f}")

    fr_full, fr_final = forgetting_regret(win_matrix)

    result = dict(
        run_name=task["run_name"],
        config_name=task["config_name"],
        stp=task["stp"],
        hsm=task["hsm"],
        A=task["A"],
        sampling=task["sampling"],
        seed=task["seed"],
        n_checkpoints=n,
        fr_full=fr_full,
        fr_final=fr_final,
        updates=updates,
        win_matrix=win_matrix.tolist(),
    )

    # Save individual result
    out_dir = OUTPUT_DIR / task["run_name"]
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "gauntlet_result.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"  FR_full={fr_full:.4f}  FR_final={fr_final:.4f}")
    return result


def aggregate():
    """Collect all gauntlet results into a single CSV."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results = []

    for jf in sorted(OUTPUT_DIR.rglob("gauntlet_result.json")):
        with open(jf) as f:
            r = json.load(f)
        results.append(r)

    if not results:
        print("No results found.")
        return

    out_csv = OUTPUT_DIR / "fr_summary.csv"
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "config_name", "stp", "hsm", "A", "sampling", "seed",
            "fr_full", "fr_final", "n_checkpoints",
        ])
        writer.writeheader()
        for r in results:
            writer.writerow({k: r[k] for k in writer.fieldnames})

    print(f"Aggregated {len(results)} results -> {out_csv}")

    # Print summary table
    print(f"\n{'Config':<16} {'A':>5} {'Sampling':<15} {'Seed':>4} {'FR_full':>8} {'FR_final':>9}")
    print("-" * 65)
    for r in sorted(results, key=lambda x: (x["config_name"], x["A"], x["sampling"], x["seed"])):
        print(f"{r['config_name']:<16} {r['A']:5.2f} {r['sampling']:<15} "
              f"{r['seed']:4d} {r['fr_full']:8.4f} {r['fr_final']:9.4f}")


def main():
    parser = argparse.ArgumentParser(description="Zoo hider-shaped gauntlet FR computation")
    parser.add_argument("--task-id", type=int, default=None)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--count", action="store_true")
    parser.add_argument("--aggregate", action="store_true")
    args = parser.parse_args()

    if args.aggregate:
        aggregate()
        return

    tasks = build_task_table()

    if args.count:
        print(len(tasks))
        return

    if args.list:
        print(f"Total tasks: {len(tasks)}")
        for i, t in enumerate(tasks):
            print(f"{i:4d}  {t['run_name']}")
        return

    if args.task_id is not None:
        if args.task_id < 0 or args.task_id >= len(tasks):
            print(f"ERROR: task-id {args.task_id} out of range [0, {len(tasks)-1}]",
                  file=sys.stderr)
            sys.exit(1)
        run_gauntlet(tasks[args.task_id], dry_run=args.dry_run)
        return

    parser.print_help()


if __name__ == "__main__":
    main()
