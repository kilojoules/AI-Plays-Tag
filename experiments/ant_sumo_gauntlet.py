#!/usr/bin/env python3
"""Cross-eval gauntlet for Ant Sumo entropy conditions."""
import sys, json
from pathlib import Path
import numpy as np, torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "trainer"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from sac import SACAgent, SACConfig
from ant_sumo import AntSumoEnv


def load_sac(path, obs_dim, act_dim):
    cfg = SACConfig(obs_dim=obs_dim, act_dim=act_dim)
    a = SACAgent(cfg); a.load_policy(str(path)); return a

def act_batch(agent, obs):
    with torch.no_grad():
        x = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        actions, _ = agent.actor.sample(x)
        return actions.squeeze(0).cpu().numpy()

def evaluate(seeker, hider, n_episodes=50):
    """Run episodes sequentially (MuJoCo envs are single-instance)."""
    wins = 0
    total_len = 0
    for _ in range(n_episodes):
        env = AntSumoEnv()
        obs = env.reset()
        done = False
        while not done:
            s_act = act_batch(seeker, obs['seeker'])
            h_act = act_batch(hider, obs['hider'])
            obs, rew, done, info = env.step({'seeker': s_act, 'hider': h_act})
        if info['tagged']:  # agent0 (seeker) lost
            pass
        else:
            wins += 1  # agent1 (hider) lost = seeker wins
        total_len += env._step_count
    # Actually: tagged=True means agent0 lost. Let me re-check the logic.
    # In ant_sumo.py: tagged = out0 or fall0 (agent0 lost)
    # So seeker winning = agent1 out/fell = NOT tagged
    # Wait, that's backward. Let me just count from rewards.
    # Actually let me re-run properly:
    seeker_wins = 0
    lengths = []
    for _ in range(n_episodes):
        env = AntSumoEnv()
        obs = env.reset()
        done = False
        while not done:
            s_act = act_batch(seeker, obs['seeker'])
            h_act = act_batch(hider, obs['hider'])
            obs, rew, done, info = env.step({'seeker': s_act, 'hider': h_act})
        # tagged = agent0 (seeker) lost. So seeker wins when NOT tagged
        # But also need to check if hider lost (out1 or fall1)
        # Actually the reward tells us: if seeker got WIN_REWARD, seeker won
        if rew['seeker'] > 5:  # WIN_REWARD = 10
            seeker_wins += 1
        lengths.append(env._step_count)
    return {
        'seeker_win_rate': seeker_wins / n_episodes,
        'mean_episode_length': np.mean(lengths),
    }


def find_best_seed(base_dir, obs_dim, act_dim):
    """Find seed with most balanced final SWR."""
    best_balance, best_run = -1, None
    for seed_dir in sorted(Path(base_dir).glob("seed_*")):
        runs = sorted(seed_dir.glob("2026*"))
        # Only consider 5M runs (>2000 lines in metrics)
        for run in reversed(runs):
            csv_path = run / "metrics.csv"
            if not csv_path.exists():
                continue
            lines = csv_path.read_text().strip().split('\n')
            if len(lines) < 2000:
                continue
            if not (run / "policy_seeker_final.pt").exists():
                continue
            wrs = []
            for line in lines[-10:]:
                try: wrs.append(float(line.split(',')[5]))
                except: pass
            swr = np.mean(wrs) if wrs else 0.5
            balance = 1.0 - abs(swr - 0.5) * 2
            if balance > best_balance:
                best_balance = balance
                best_run = run
            break
    return best_run


def main():
    base = Path("experiments/results/ant_sumo")
    output_dir = base / "gauntlet"
    output_dir.mkdir(parents=True, exist_ok=True)

    obs_dim = 23
    act_dim = 8

    conditions = ["baseline_02", "optimal_0607", "low_005", "high_20",
                   "fixed_01", "no_entropy"]

    agents = {}
    loaded = []

    for name in conditions:
        run = find_best_seed(base / name, obs_dim, act_dim)
        if run is None:
            print(f"SKIP {name}: no 5M run found")
            continue
        s = load_sac(str(run / "policy_seeker_final.pt"), obs_dim, act_dim)
        h = load_sac(str(run / "policy_hider_final.pt"), obs_dim, act_dim)
        agents[name] = (s, h)
        loaded.append(name)

        # Get training SWR
        lines = (run / "metrics.csv").read_text().strip().split('\n')
        wrs = [float(l.split(',')[5]) for l in lines[-10:] if l.split(',')[5]]
        print(f"Loaded {name} (train_SWR={np.mean(wrs):.1%})")

    n = len(loaded)
    print(f"\n{n} conditions. Running {n}x{n} = {n*n} matchups (30 eps each)...\n")

    W = np.zeros((n, n))
    L = np.zeros((n, n))

    for i, sn in enumerate(loaded):
        for j, hn in enumerate(loaded):
            r = evaluate(agents[sn][0], agents[hn][1], n_episodes=30)
            W[i, j] = r['seeker_win_rate']
            L[i, j] = r['mean_episode_length']
            print(f"  S:{sn:<18s} vs H:{hn:<18s} WR={r['seeker_win_rate']:.0%} "
                  f"EL={r['mean_episode_length']:.0f}")

    print(f"\n{'Condition':<18s} {'Seeker':>7s} {'Surv':>6s} {'Comb':>6s}")
    print("-" * 40)
    for j, name in enumerate(loaded):
        sk = W[j, :].mean()
        surv = 1.0 - W[:, j].mean()
        comb = (sk + surv) / 2
        print(f"{name:<18s} {sk:>6.1%} {surv:>5.1%} {comb:>5.1%}")

    results = {'conditions': loaded, 'win_matrix': W.tolist(),
               'length_matrix': L.tolist(), 'episodes_per_matchup': 30}
    out = output_dir / "gauntlet_results.json"
    with open(out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
