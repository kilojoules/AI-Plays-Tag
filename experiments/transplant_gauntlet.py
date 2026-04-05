#!/usr/bin/env python3
"""Quick 4x4 cross-eval for critic transplant experiment."""
import sys
import json
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from trainer.tag_env import VecTagEnv, TagEnvConfig
from trainer.sac import SACAgent, SACConfig


def load_sac(path, obs_dim, act_dim):
    cfg = SACConfig(obs_dim=obs_dim, act_dim=act_dim)
    a = SACAgent(cfg)
    a.load_policy(str(path))
    return a

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
    active = np.ones(n, dtype=bool)
    tagged = np.zeros(n, dtype=bool)
    for step in range(max_steps):
        s_a = seeker_fn(obs['seeker'])
        h_a = hider_fn(obs['hider'])
        obs, _, dones, infos = env.step({'seeker': s_a, 'hider': h_a})
        nd = dones & active
        if nd.any():
            for i in np.where(nd)[0]:
                tagged[i] = infos['tagged'][i]
            active[nd] = False
        if not active.any():
            break
    return float(tagged.mean())

def main():
    base = Path("experiments/results/paper_final/critic_transplant")
    obs_dim, act_dim = 87, 3

    conditions = ["control_607", "control_005", "transplant_607critic", "transplant_005critic"]
    agents = {}

    for name in conditions:
        runs = sorted((base / name / "seed_0").glob("2026*"))
        if not runs:
            print(f"SKIP {name}")
            continue
        s = load_sac(runs[-1] / "policy_seeker_final.pt", obs_dim, act_dim)
        h = load_sac(runs[-1] / "policy_hider_final.pt", obs_dim, act_dim)
        agents[name] = (lambda o, s=s: act_batch(s, o), lambda o, h=h: act_batch(h, o))
        print(f"Loaded {name}")

    n = len(conditions)
    W = np.zeros((n, n))

    for i, sc in enumerate(conditions):
        if sc not in agents: continue
        for j, hc in enumerate(conditions):
            if hc not in agents: continue
            wr = evaluate(agents[sc][0], agents[hc][1], n=50)
            W[i, j] = wr
            print(f"  S:{sc:<25s} vs H:{hc:<25s} WR={wr:.0%}")

    print(f"\n{'Condition':<25s} {'Seeker':>7s} {'Surv':>6s} {'Combined':>9s}")
    print("-" * 50)
    for j, c in enumerate(conditions):
        sk = W[j, :].mean()
        surv = 1.0 - W[:, j].mean()
        comb = (sk + surv) / 2
        print(f"{c:<25s} {sk:>6.1%} {surv:>5.1%} {comb:>8.1%}")

    out = base / "gauntlet_results.json"
    json.dump({'conditions': conditions, 'win_matrix': W.tolist()}, open(out, 'w'), indent=2)
    print(f"\nSaved {out}")

if __name__ == "__main__":
    main()
