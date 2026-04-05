#!/usr/bin/env python3
"""Quick cross-eval for all mechanism deep dive conditions."""
import sys, json
from pathlib import Path
import numpy as np, torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from trainer.tag_env import VecTagEnv, TagEnvConfig
from trainer.sac import SACAgent, SACConfig

def load_sac(path):
    cfg = SACConfig(obs_dim=87, act_dim=3)
    a = SACAgent(cfg); a.load_policy(str(path)); return a

def act_batch(agent, obs):
    with torch.no_grad():
        x = torch.as_tensor(obs, dtype=torch.float32)
        actions, _ = agent.actor.sample(x)
        return actions.cpu().numpy()

def evaluate(seeker_fn, hider_fn, n=50):
    cfg = TagEnvConfig(layout="four_corners", hider_speed_mult=1.15)
    env = VecTagEnv(num_envs=n, config=cfg)
    obs = env.reset()
    max_steps = int(cfg.time_limit / (cfg.dt * cfg.steps_per_action))
    active = np.ones(n, dtype=bool); tagged = np.zeros(n, dtype=bool)
    for step in range(max_steps):
        obs, _, dones, infos = env.step({
            'seeker': seeker_fn(obs['seeker']),
            'hider': hider_fn(obs['hider'])})
        nd = dones & active
        if nd.any():
            for i in np.where(nd)[0]: tagged[i] = infos['tagged'][i]
            active[nd] = False
        if not active.any(): break
    return float(tagged.mean())

def load_best_seed(base_dir):
    """Load agents from the seed with most balanced SWR."""
    base = Path(base_dir)
    best_balance, best_run = -1, None
    for seed_dir in sorted(base.glob("seed_*")):
        runs = sorted(seed_dir.glob("2026*"))
        if not runs: continue
        run = runs[-1]
        if not (run / "policy_seeker_final.pt").exists(): continue
        csv_path = run / "metrics.csv"
        if csv_path.exists():
            lines = csv_path.read_text().strip().split('\n')
            wrs = []
            for line in lines[-10:]:
                try: wrs.append(float(line.split(',')[5]))
                except: pass
            swr = np.mean(wrs) if wrs else 0.5
        else:
            swr = 0.5
        balance = 1.0 - abs(swr - 0.5) * 2
        if balance > best_balance:
            best_balance = balance
            best_run = run
    return best_run

def main():
    base = Path("experiments/results/paper_final/mechanism_deep_dive")

    # Collect all conditions
    conditions = {}
    for exp in ["1_transplant_5seed", "3_random_critic", "4_layerwise", "5_freeze_actor"]:
        exp_dir = base / exp
        if not exp_dir.exists(): continue
        for cond_dir in sorted(exp_dir.iterdir()):
            if not cond_dir.is_dir(): continue
            name = f"{exp}/{cond_dir.name}"
            run = load_best_seed(cond_dir)
            if run is None: continue
            s = load_sac(run / "policy_seeker_final.pt")
            h = load_sac(run / "policy_hider_final.pt")
            conditions[name] = (lambda o, s=s: act_batch(s, o),
                               lambda o, h=h: act_batch(h, o))
            print(f"Loaded {name}")

    names = list(conditions.keys())
    n = len(names)
    W = np.zeros((n, n))

    print(f"\nRunning {n}x{n} = {n*n} matchups...")
    for i, sn in enumerate(names):
        for j, hn in enumerate(names):
            W[i, j] = evaluate(conditions[sn][0], conditions[hn][1], n=50)

    print(f"\n{'Condition':<45s} {'Seeker':>7s} {'Surv':>6s} {'Comb':>6s}")
    print("-" * 67)
    for j, name in enumerate(names):
        sk = W[j, :].mean()
        surv = 1.0 - W[:, j].mean()
        comb = (sk + surv) / 2
        print(f"{name:<45s} {sk:>6.1%} {surv:>5.1%} {comb:>5.1%}")

    out = base / "gauntlet_results.json"
    json.dump({'conditions': names, 'win_matrix': W.tolist()}, open(out, 'w'), indent=2)
    print(f"\nSaved {out}")

if __name__ == "__main__":
    main()
