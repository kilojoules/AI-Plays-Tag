#!/usr/bin/env python3
"""
Cross-evaluation for layout generalization experiments.

For each layout, evaluates baseline vs kl_0.2 agents against each other
within that layout. Reports corner camping and strength per layout.
"""
import sys
import json
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from trainer.tag_env import VecTagEnv, TagEnvConfig
from trainer.sac import SACAgent, SACConfig


def load_sac_policy(path, obs_dim, act_dim):
    cfg = SACConfig(obs_dim=obs_dim, act_dim=act_dim)
    agent = SACAgent(cfg)
    agent.load_policy(str(path))
    return agent


def act_batch_sac(agent, obs_batch):
    with torch.no_grad():
        x = torch.as_tensor(obs_batch, dtype=torch.float32)
        actions, _ = agent.actor.sample(x)
        return actions.cpu().numpy()


def find_latest_run(base_dir):
    base = Path(base_dir)
    if not base.exists():
        return None
    runs = sorted(base.glob("2026*"))
    for run in reversed(runs):
        if (run / "policy_seeker_final.pt").exists():
            return run
    return None


def get_final_swr(run_dir):
    csv_path = run_dir / "metrics.csv"
    if not csv_path.exists():
        return 0.5
    lines = csv_path.read_text().strip().split('\n')
    wrs = []
    for line in lines[-10:]:
        parts = line.split(',')
        try:
            wrs.append(float(parts[7]))
        except (IndexError, ValueError):
            continue
    return np.mean(wrs) if wrs else 0.5


def find_best_seed(base_dir):
    best_balance = -1
    best_seed = 0
    for seed in [0, 1, 2]:
        run_dir = find_latest_run(Path(base_dir) / f"seed_{seed}")
        if run_dir is None:
            continue
        swr = get_final_swr(run_dir)
        balance = 1.0 - abs(swr - 0.5) * 2
        if balance > best_balance:
            best_balance = balance
            best_seed = seed
    return best_seed


def evaluate_matchup(seeker_fn, hider_fn, layout, num_episodes=50):
    cfg = TagEnvConfig(layout=layout, hider_speed_mult=1.15)
    env = VecTagEnv(num_envs=num_episodes, config=cfg)
    obs = env.reset()

    max_steps = int(cfg.time_limit / (cfg.dt * cfg.steps_per_action))
    active = np.ones(num_episodes, dtype=bool)
    tagged = np.zeros(num_episodes, dtype=bool)
    lengths = np.zeros(num_episodes, dtype=int)
    wall_samples, corner_samples, speed_samples = [], [], []

    for step in range(max_steps):
        seeker_actions = seeker_fn(obs['seeker'])
        hider_actions = hider_fn(obs['hider'])
        obs, rewards, dones, infos = env.step({
            'seeker': seeker_actions, 'hider': hider_actions})

        wall_samples.append(infos['hider_near_wall_frac'])
        corner_samples.append(infos['hider_corner_frac'])
        speed_samples.append(infos['hider_speed_mean'])

        newly_done = dones & active
        if np.any(newly_done):
            for i in np.where(newly_done)[0]:
                lengths[i] = step + 1
                tagged[i] = infos['tagged'][i]
            active[newly_done] = False
        if not np.any(active):
            break

    for i in range(num_episodes):
        if active[i]:
            lengths[i] = max_steps

    return {
        'seeker_win_rate': float(tagged.mean()),
        'mean_episode_length': float(lengths.mean()),
        'mean_hider_wall_frac': float(np.mean(wall_samples)),
        'mean_hider_corner_frac': float(np.mean(corner_samples)),
        'mean_hider_speed': float(np.mean(speed_samples)),
    }


def main():
    base = Path("experiments/results/layout_generalization")
    output_dir = base / "gauntlet"
    output_dir.mkdir(parents=True, exist_ok=True)

    obs_dim = 87
    act_dim = 3

    layouts = ["empty", "central_cross", "playground"]
    conditions = ["baseline", "kl_02"]

    all_results = {}

    for layout in layouts:
        print(f"\n{'='*60}")
        print(f"Layout: {layout}")
        print(f"{'='*60}")

        # Load agents for this layout
        agents = {}
        for cond in conditions:
            name = f"{layout}_{cond}"
            config_dir = base / name
            best_seed = find_best_seed(config_dir)
            run_dir = find_latest_run(config_dir / f"seed_{best_seed}")
            if run_dir is None:
                print(f"  SKIP {name}: no run found")
                continue

            seeker = load_sac_policy(run_dir / "policy_seeker_final.pt", obs_dim, act_dim)
            hider = load_sac_policy(run_dir / "policy_hider_final.pt", obs_dim, act_dim)
            agents[cond] = (
                lambda obs, s=seeker: act_batch_sac(s, obs),
                lambda obs, h=hider: act_batch_sac(h, obs),
            )
            swr = get_final_swr(run_dir)
            print(f"  Loaded {name} (seed={best_seed}, train_SWR={swr:.1%})")

        if len(agents) < 2:
            print(f"  Not enough agents for layout {layout}")
            continue

        # Cross-evaluate: each seeker vs each hider within this layout
        n = len(conditions)
        layout_results = {}

        for s_cond in conditions:
            if s_cond not in agents:
                continue
            for h_cond in conditions:
                if h_cond not in agents:
                    continue
                key = f"S:{s_cond}_vs_H:{h_cond}"
                result = evaluate_matchup(
                    agents[s_cond][0], agents[h_cond][1],
                    layout=layout, num_episodes=50)
                layout_results[key] = result
                wr = result['seeker_win_rate']
                cr = result['mean_hider_corner_frac']
                sp = result['mean_hider_speed']
                print(f"  {key:<35s} WR={wr:.0%} Corner={cr:.2f} Speed={sp:.1f}")

        all_results[layout] = layout_results

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY: KL responsiveness effect per layout")
    print(f"{'='*60}")

    for layout in layouts:
        if layout not in all_results:
            continue
        lr = all_results[layout]

        # Average hider corner when playing as hider (across both seekers)
        bl_corner = np.mean([lr[k]['mean_hider_corner_frac']
                            for k in lr if 'H:baseline' in k])
        kl_corner = np.mean([lr[k]['mean_hider_corner_frac']
                            for k in lr if 'H:kl_02' in k])

        bl_surv = np.mean([1.0 - lr[k]['seeker_win_rate']
                          for k in lr if 'H:baseline' in k])
        kl_surv = np.mean([1.0 - lr[k]['seeker_win_rate']
                          for k in lr if 'H:kl_02' in k])

        delta_c = kl_corner - bl_corner
        delta_s = kl_surv - bl_surv
        print(f"  {layout:<15s} Corner: {bl_corner:.2f} -> {kl_corner:.2f} ({delta_c:+.2f})  "
              f"Survival: {bl_surv:.1%} -> {kl_surv:.1%} ({delta_s:+.1%})")

    # Save
    out_path = output_dir / "layout_gauntlet_results.json"
    # Convert numpy to python types
    serializable = {}
    for layout, lr in all_results.items():
        serializable[layout] = {k: {kk: float(vv) for kk, vv in v.items()} for k, v in lr.items()}
    with open(out_path, 'w') as f:
        json.dump(serializable, f, indent=2)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
