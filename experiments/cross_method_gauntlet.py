#!/usr/bin/env python3
"""
Cross-method gauntlet: compare the best agents from every training paradigm.

For each training method, auto-discovers the best agent(s) by seeker win rate
balance, then runs all seekers vs all hiders in a standardised evaluation.

Methods compared:
  selfplay     - PPO self-play (selfplay_sweep)
  reward/PPO   - Self-play + reward shaping, PPO (reward_sweep)
  reward/SAC   - Self-play + reward shaping, SAC (reward_sweep)
  zoo          - Zoo training, PPO (zoo_asweep)
  zoo_shaped   - Zoo + hider shaping, PPO (zoo_hider_shaped)
  fr/PPO       - Zoo + reward presets, PPO (fr_sweep)
  fr/SAC       - Zoo + reward presets, SAC (fr_sweep)
  fr_v2/PPO    - Optuna-optimised zoo + presets, PPO (fr_sweep_v2)
  fr_v2/SAC    - Optuna-optimised zoo + presets, SAC (fr_sweep_v2)

Usage:
  python experiments/cross_method_gauntlet.py                # run with defaults
  python experiments/cross_method_gauntlet.py --top-k 5      # more agents per method
  python experiments/cross_method_gauntlet.py --list          # dry-run: show selection
  python experiments/cross_method_gauntlet.py --plot          # generate plots
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from trainer.tag_env import VecTagEnv, TagEnvConfig
from trainer.ppo import PPOAgent, PPOConfig
from trainer.sac import SACAgent, SACConfig

# -- Evaluation config -------------------------------------------------
EVAL_LAYOUT = "four_corners"
EVAL_HSM = 1.15
DEFAULT_EPISODES = 50
MAX_EVAL_STEPS = 200
OBS_DIM = 87
ACT_DIM = 3

# ------ Experiment directories ---------------------------------------------------------------------------------------------------------------------
BASE = Path("experiments/results")
OUTPUT_DIR = BASE / "cross_method_gauntlet"

SKIP_DIRS = {"analysis", "gauntlet", "logs", "best_episodes", "showcase",
             "animations", "curves"}


# ------ Helpers ------------------------------------------------------------------------------------------------------------------------------------------------------------------

def get_swr(metrics_path: Path) -> float | None:
    """Read average seeker win rate from last 5 entries of metrics.csv."""
    try:
        text = metrics_path.read_text().strip()
    except Exception:
        return None
    lines = text.split("\n")
    if len(lines) < 2:
        return None
    header = lines[0].split(",")
    try:
        col = header.index("seeker_win_rate")
    except ValueError:
        return None
    vals = []
    for line in lines[-5:]:
        parts = line.split(",")
        try:
            vals.append(float(parts[col]))
        except (IndexError, ValueError):
            continue
    return float(np.mean(vals)) if vals else None


def find_latest_run(parent: Path) -> Path | None:
    """Find most recent timestamped run dir with final policies."""
    if not parent.exists():
        return None
    for d in sorted(parent.glob("2*"), reverse=True):
        if (d / "policy_seeker_final.pt").exists():
            return d
    return None


def find_best_seed(config_dir: Path) -> Path | None:
    """Among seed_* dirs, return the run with highest balance = min(swr, 1-swr)."""
    best, best_bal = None, -1.0
    for seed_dir in sorted(config_dir.iterdir()):
        if not seed_dir.is_dir() or not seed_dir.name.startswith("seed_"):
            continue
        run = find_latest_run(seed_dir)
        if run is None:
            continue
        swr = get_swr(run / "metrics.csv")
        if swr is None:
            continue
        bal = min(swr, 1 - swr)
        if bal > best_bal:
            best_bal = bal
            best = run
    return best


def _agent(method, algo, label, run_dir, swr):
    return {
        "method": method,
        "algo": algo,
        "label": label,
        "seeker_path": run_dir / "policy_seeker_final.pt",
        "hider_path": run_dir / "policy_hider_final.pt",
        "swr": swr,
        "balance": min(swr, 1 - swr),
    }


# ------ Discovery functions ------------------------------------------------------------------------------------------------------------------------------

def discover_selfplay() -> list[dict]:
    """selfplay_sweep: STP{x}_HSM{y}/{timestamp}/ --- PPO, no seeds."""
    d = BASE / "selfplay_sweep"
    agents = []
    if not d.exists():
        return agents
    for cfg in sorted(d.iterdir()):
        if not cfg.is_dir() or cfg.name in SKIP_DIRS or cfg.suffix == ".json":
            continue
        run = find_latest_run(cfg)
        if run is None:
            continue
        swr = get_swr(run / "metrics.csv")
        if swr is not None:
            agents.append(_agent("selfplay", "ppo",
                                 f"selfplay/{cfg.name}", run, swr))
    return agents


def discover_reward_sweep() -> list[dict]:
    """reward_sweep: {Preset}/{algo}/seed_{n}/{timestamp}/ --- PPO + SAC."""
    d = BASE / "reward_sweep"
    agents = []
    if not d.exists():
        return agents
    for preset_dir in sorted(d.iterdir()):
        if not preset_dir.is_dir() or preset_dir.name in SKIP_DIRS:
            continue
        for algo in ("ppo", "sac"):
            algo_dir = preset_dir / algo
            if not algo_dir.exists():
                continue
            run = find_best_seed(algo_dir)
            if run is None:
                continue
            swr = get_swr(run / "metrics.csv")
            if swr is not None:
                agents.append(_agent(f"reward/{algo.upper()}", algo,
                                     f"reward/{preset_dir.name}/{algo}", run, swr))
    return agents


def _discover_zoo(base_dir: Path, method: str) -> list[dict]:
    """Generic zoo discovery: STP{x}_HSM{y}/A{nn}_{sampling}/seed_{n}/ --- PPO."""
    agents = []
    if not base_dir.exists():
        return agents
    for cfg_dir in sorted(base_dir.iterdir()):
        if not cfg_dir.is_dir() or cfg_dir.name in SKIP_DIRS:
            continue
        for a_dir in sorted(cfg_dir.iterdir()):
            if not a_dir.is_dir() or a_dir.suffix == ".json":
                continue
            run = find_best_seed(a_dir)
            if run is None:
                continue
            swr = get_swr(run / "metrics.csv")
            if swr is not None:
                agents.append(_agent(method, "ppo",
                                     f"{method}/{cfg_dir.name}/{a_dir.name}",
                                     run, swr))
    return agents


def discover_zoo() -> list[dict]:
    return _discover_zoo(BASE / "zoo_asweep", "zoo")


def discover_zoo_shaped() -> list[dict]:
    return _discover_zoo(BASE / "zoo_hider_shaped", "zoo_shaped")


def discover_fr(version: str = "v1") -> list[dict]:
    """fr_sweep[_v2]: {Preset}/A{val}_{algo}/seed_{n}/{timestamp}/ --- PPO + SAC."""
    base = BASE / ("fr_sweep" if version == "v1" else "fr_sweep_v2")
    prefix = "fr" if version == "v1" else "fr_v2"
    agents = []
    if not base.exists():
        return agents
    for preset_dir in sorted(base.iterdir()):
        if not preset_dir.is_dir() or preset_dir.name in SKIP_DIRS:
            continue
        for cfg_dir in sorted(preset_dir.iterdir()):
            if not cfg_dir.is_dir():
                continue
            name = cfg_dir.name
            if "_ppo" in name:
                algo = "ppo"
            elif "_sac" in name:
                algo = "sac"
            else:
                continue
            run = find_best_seed(cfg_dir)
            if run is None:
                continue
            swr = get_swr(run / "metrics.csv")
            if swr is not None:
                agents.append(_agent(
                    f"{prefix}/{algo.upper()}", algo,
                    f"{prefix}/{preset_dir.name}/{cfg_dir.name}", run, swr))
    return agents


# ------ Selection ------------------------------------------------------------------------------------------------------------------------------------------------------------

def select_representatives(agents: list[dict], top_k: int = 3) -> list[dict]:
    """Pick top-K per method: most balanced, best seeker, best hider, then fill."""
    by_method: dict[str, list[dict]] = {}
    for a in agents:
        by_method.setdefault(a["method"], []).append(a)

    selected = []
    for method in sorted(by_method):
        pool = by_method[method]
        picked: set[str] = set()

        # 1. Most balanced
        best_bal = max(pool, key=lambda x: x["balance"])
        selected.append(best_bal)
        picked.add(best_bal["label"])

        if top_k >= 2:
            # 2. Best seeker (highest SWR)
            best_s = max(pool, key=lambda x: x["swr"])
            if best_s["label"] not in picked:
                selected.append(best_s)
                picked.add(best_s["label"])

        if top_k >= 3:
            # 3. Best hider (lowest SWR = hider survived most)
            best_h = min(pool, key=lambda x: x["swr"])
            if best_h["label"] not in picked:
                selected.append(best_h)
                picked.add(best_h["label"])

        # Fill remaining with next-best balanced
        for a in sorted(pool, key=lambda x: -x["balance"]):
            if len(picked) >= top_k:
                break
            if a["label"] not in picked:
                selected.append(a)
                picked.add(a["label"])

    return selected


# ------ Loading & evaluation ---------------------------------------------------------------------------------------------------------------------------

def _act_ppo(policy, obs):
    with torch.no_grad():
        x = torch.as_tensor(obs, dtype=torch.float32)
        logits = policy.pi(x)
        mean, log_std = torch.chunk(logits, 2, dim=-1)
        log_std = torch.clamp(log_std, -2.0, 1.5)
        std = torch.exp(log_std)
        return torch.tanh(torch.distributions.Normal(mean, std).sample()).cpu().numpy()


def _act_sac(agent, obs):
    with torch.no_grad():
        x = torch.as_tensor(obs, dtype=torch.float32)
        actions, _ = agent.actor.sample(x)
        return actions.cpu().numpy()


def load_agent(path: Path, algo: str):
    """Return a callable (obs_batch -> action_batch)."""
    if algo == "ppo":
        cfg = PPOConfig(obs_dim=OBS_DIM, act_dim=ACT_DIM)
        policy = PPOAgent(cfg)
        ckpt = torch.load(str(path), map_location="cpu", weights_only=True)
        policy.pi.load_state_dict(ckpt["pi"])
        policy.vf.load_state_dict(ckpt["vf"])
        return lambda obs, p=policy: _act_ppo(p, obs)
    else:
        cfg = SACConfig(obs_dim=OBS_DIM, act_dim=ACT_DIM)
        agent = SACAgent(cfg)
        agent.load_policy(str(path))
        return lambda obs, a=agent: _act_sac(a, obs)


def evaluate_matchup(seeker_fn, hider_fn, n_episodes: int) -> tuple[float, float]:
    """Return (seeker_win_rate, mean_episode_length)."""
    cfg = TagEnvConfig(layout=EVAL_LAYOUT, hider_speed_mult=EVAL_HSM)
    env = VecTagEnv(num_envs=n_episodes, config=cfg)
    obs = env.reset()
    max_steps = int(cfg.time_limit / (cfg.dt * cfg.steps_per_action))

    active = np.ones(n_episodes, dtype=bool)
    tagged = np.zeros(n_episodes, dtype=bool)
    lengths = np.zeros(n_episodes, dtype=int)

    for step in range(max_steps):
        s_acts = seeker_fn(obs["seeker"])
        h_acts = hider_fn(obs["hider"])
        obs, _, dones, infos = env.step({"seeker": s_acts, "hider": h_acts})
        newly_done = dones & active
        if np.any(newly_done):
            for i in np.where(newly_done)[0]:
                lengths[i] = step + 1
                tagged[i] = infos["tagged"][i]
            active[newly_done] = False
        if not np.any(active):
            break

    lengths[active] = max_steps
    return float(tagged.mean()), float(lengths.mean())


# ------ Plotting ---------------------------------------------------------------------------------------------------------------------------------------------------------------

def plot_results(labels, methods, win_matrix, unique_methods, h2h):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plots")
        return

    n = len(labels)
    nm = len(unique_methods)

    # --- 1. Full win matrix heatmap ---
    fig, ax = plt.subplots(figsize=(max(12, n * 0.6), max(10, n * 0.5)))
    im = ax.imshow(win_matrix * 100, cmap="RdYlGn", vmin=0, vmax=100, aspect="auto")

    # Colour-code tick labels by method
    method_colours = {}
    cmap = plt.cm.get_cmap("tab10", nm)
    for mi, m in enumerate(unique_methods):
        method_colours[m] = cmap(mi)

    short_labels = [l.split("/", 1)[-1] if "/" in l else l for l in labels]
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(short_labels, fontsize=7, rotation=90)
    ax.set_yticklabels(short_labels, fontsize=7)

    for i, m in enumerate(methods):
        ax.get_yticklabels()[i].set_color(method_colours[m])
        ax.get_xticklabels()[i].set_color(method_colours[m])

    if n <= 30:
        for i in range(n):
            for j in range(n):
                v = win_matrix[i, j] * 100
                c = "white" if v < 30 or v > 70 else "black"
                ax.text(j, i, f"{v:.0f}", ha="center", va="center",
                        fontsize=6, color=c)

    plt.colorbar(im, ax=ax, label="Seeker Win Rate (%)")
    ax.set_xlabel("Hider")
    ax.set_ylabel("Seeker")
    ax.set_title("Cross-Method Gauntlet: Seeker Win Rate (%)")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "gauntlet_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()

    # --- 2. Method comparison bars ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    seeker_str = win_matrix.mean(axis=1)
    hider_str = 1 - win_matrix.mean(axis=0)

    method_seeker = []
    method_hider = []
    for m in unique_methods:
        idx = [i for i, x in enumerate(methods) if x == m]
        method_seeker.append(np.mean([seeker_str[i] for i in idx]))
        method_hider.append(np.mean([hider_str[i] for i in idx]))

    x = np.arange(nm)
    w = 0.35
    ax = axes[0]
    ax.bar(x - w / 2, method_seeker, w, label="Seeker strength", color="tab:red", alpha=0.8)
    ax.bar(x + w / 2, method_hider, w, label="Hider strength", color="tab:blue", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(unique_methods, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Strength (win/survival rate)")
    ax.set_title("Method Strength Comparison")
    ax.legend()
    ax.set_ylim(0, 1)

    # Method head-to-head
    ax = axes[1]
    im2 = ax.imshow(h2h * 100, cmap="RdYlGn", vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(range(nm))
    ax.set_yticks(range(nm))
    short_m = [m.replace("/", "\n") for m in unique_methods]
    ax.set_xticklabels(short_m, fontsize=7)
    ax.set_yticklabels(short_m, fontsize=7)
    for i in range(nm):
        for j in range(nm):
            v = h2h[i, j] * 100
            c = "white" if v < 30 or v > 70 else "black"
            ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=8, color=c)
    plt.colorbar(im2, ax=ax, label="Seeker Win Rate (%)")
    ax.set_xlabel("Hider method")
    ax.set_ylabel("Seeker method")
    ax.set_title("Method Head-to-Head")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "method_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plots saved to {OUTPUT_DIR}/")


# ------ Main ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Cross-method gauntlet: which training paradigm makes the best opponents?")
    parser.add_argument("--top-k", type=int, default=3,
                        help="Max agents per method (default: 3)")
    parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODES,
                        help="Episodes per matchup (default: 50)")
    parser.add_argument("--list", action="store_true",
                        help="List discovered agents and exit")
    parser.add_argument("--plot", action="store_true",
                        help="Generate plots after gauntlet")
    args = parser.parse_args()

    n_episodes = args.episodes

    # ------ Discovery ------------------------------------------------------------------------------------------------------------------------------------
    print("Discovering agents across training methods...")
    t0 = time.time()

    all_agents = []
    for name, fn in [
        ("selfplay_sweep", discover_selfplay),
        ("reward_sweep", discover_reward_sweep),
        ("zoo_asweep", discover_zoo),
        ("zoo_hider_shaped", discover_zoo_shaped),
        ("fr_sweep", lambda: discover_fr("v1")),
        ("fr_sweep_v2", lambda: discover_fr("v2")),
    ]:
        found = fn()
        print(f"  {name:<20s}: {len(found):4d} candidates")
        all_agents += found

    dt = time.time() - t0
    print(f"  Total: {len(all_agents)} agents discovered in {dt:.1f}s")

    # Summary by method
    by_method: dict[str, list[dict]] = {}
    for a in all_agents:
        by_method.setdefault(a["method"], []).append(a)

    print(f"\n{'Method':<20s} {'Count':>6s} {'SWR min':>8s} {'SWR max':>8s} {'SWR med':>8s}")
    print("-" * 54)
    for m in sorted(by_method):
        swrs = [a["swr"] for a in by_method[m]]
        print(f"  {m:<18s} {len(swrs):>6d} {min(swrs):>7.2f} {max(swrs):>7.2f} "
              f"{np.median(swrs):>7.2f}")

    # ------ Selection ------------------------------------------------------------------------------------------------------------------------------------
    selected = select_representatives(all_agents, top_k=args.top_k)
    print(f"\nSelected {len(selected)} representatives (top-{args.top_k} per method):")
    for a in sorted(selected, key=lambda x: (x["method"], -x["balance"])):
        print(f"  {a['method']:<18s} {a['label']:<55s} SWR={a['swr']:.3f}")

    if args.list:
        return

    # ------ Load policies ------------------------------------------------------------------------------------------------------------------------
    print("\nLoading policies...")
    labels = [a["label"] for a in selected]
    methods = [a["method"] for a in selected]
    seeker_fns = {}
    hider_fns = {}
    for a in selected:
        seeker_fns[a["label"]] = load_agent(a["seeker_path"], a["algo"])
        hider_fns[a["label"]] = load_agent(a["hider_path"], a["algo"])
    print(f"  Loaded {len(selected)} seeker/hider pairs")

    # ------ Gauntlet ---------------------------------------------------------------------------------------------------------------------------------------
    n = len(labels)
    print(f"\nRunning {n}x{n} = {n * n} matchups ({n_episodes} eps each)...\n")
    t0 = time.time()

    win_matrix = np.zeros((n, n), dtype=np.float32)
    len_matrix = np.zeros((n, n), dtype=np.float32)

    for i in range(n):
        for j in range(n):
            wr, el = evaluate_matchup(seeker_fns[labels[i]],
                                       hider_fns[labels[j]], n_episodes)
            win_matrix[i, j] = wr
            len_matrix[i, j] = el
        mean_wr = win_matrix[i, :].mean()
        print(f"  [{i + 1:2d}/{n}] S:{labels[i]:<50s} mean_WR={mean_wr:.1%}")

    dt = time.time() - t0
    print(f"\nGauntlet complete in {dt:.0f}s ({dt / (n * n):.1f}s/matchup)")

    # ------ Save results ---------------------------------------------------------------------------------------------------------------------------
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results = {
        "labels": labels,
        "methods": methods,
        "selected": [
            {k: str(v) if isinstance(v, Path) else v for k, v in a.items()}
            for a in selected
        ],
        "win_matrix": win_matrix.tolist(),
        "length_matrix": len_matrix.tolist(),
        "eval_config": {
            "layout": EVAL_LAYOUT,
            "hider_speed_mult": EVAL_HSM,
            "episodes_per_matchup": n_episodes,
        },
    }
    out_path = OUTPUT_DIR / "gauntlet_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {out_path}")

    # ------ Analysis ---------------------------------------------------------------------------------------------------------------------------------------
    seeker_str = win_matrix.mean(axis=1)
    hider_str = 1 - win_matrix.mean(axis=0)

    print(f"\n{'=' * 80}")
    print("SEEKER STRENGTH (mean win rate as seeker across ALL hiders)")
    print(f"{'=' * 80}")
    for rank, i in enumerate(np.argsort(-seeker_str)):
        print(f"  {rank + 1:2d}. [{methods[i]:<18s}] {labels[i]:<50s} "
              f"WR={seeker_str[i]:.1%}")

    print(f"\n{'=' * 80}")
    print("HIDER STRENGTH (mean survival rate across ALL seekers)")
    print(f"{'=' * 80}")
    for rank, j in enumerate(np.argsort(-hider_str)):
        print(f"  {rank + 1:2d}. [{methods[j]:<18s}] {labels[j]:<50s} "
              f"SURV={hider_str[j]:.1%}")

    # ------ Method aggregates ------------------------------------------------------------------------------------------------------------
    unique_methods = sorted(set(methods))
    nm = len(unique_methods)

    print(f"\n{'=' * 80}")
    print("METHOD COMPARISON (aggregated across representatives)")
    print(f"{'=' * 80}")
    method_stats = []
    for m in unique_methods:
        idx = [i for i, x in enumerate(methods) if x == m]
        s = np.mean([seeker_str[i] for i in idx])
        h = np.mean([hider_str[i] for i in idx])
        combined = (s + h) / 2
        method_stats.append((m, s, h, combined, len(idx)))

    method_stats.sort(key=lambda x: -x[3])
    print(f"  {'Method':<20s} {'Seeker':>8s} {'Hider':>8s} {'Combined':>10s} {'Agents':>7s}")
    print(f"  {'-' * 57}")
    for m, s, h, c, cnt in method_stats:
        print(f"  {m:<20s} {s:>7.1%} {h:>7.1%} {c:>9.1%} {cnt:>7d}")

    # ------ Method head-to-head ------------------------------------------------------------------------------------------------------
    h2h = np.zeros((nm, nm))
    for mi, m1 in enumerate(unique_methods):
        for mj, m2 in enumerate(unique_methods):
            s_idx = [i for i, x in enumerate(methods) if x == m1]
            h_idx = [j for j, x in enumerate(methods) if x == m2]
            wrs = [win_matrix[i, j] for i in s_idx for j in h_idx]
            h2h[mi, mj] = np.mean(wrs)

    print(f"\n{'=' * 80}")
    print("METHOD HEAD-TO-HEAD (seeker WR: row seekers vs col hiders)")
    print(f"{'=' * 80}")
    s_h_label = "S \\ H"
    header = f"  {s_h_label:<18s}"
    for m in unique_methods:
        header += f" {m[:14]:>14s}"
    print(header)
    for mi, m in enumerate(unique_methods):
        row = f"  {m:<18s}"
        for mj in range(nm):
            row += f" {h2h[mi, mj]:>13.0%}"
        print(row)

    # ------ Best opponents ---------------------------------------------------------------------------------------------------------------------
    print(f"\n{'=' * 80}")
    print("BEST OPPONENTS (hardest to beat)")
    print(f"{'=' * 80}")
    # Best seeker = hardest to evade (highest mean WR)
    best_s = int(np.argmax(seeker_str))
    print(f"  Toughest seeker:  {labels[best_s]:<50s} WR={seeker_str[best_s]:.1%}")
    # Best hider = hardest to catch (highest survival)
    best_h = int(np.argmax(hider_str))
    print(f"  Toughest hider:   {labels[best_h]:<50s} SURV={hider_str[best_h]:.1%}")
    # Best overall method
    best_m = method_stats[0]
    print(f"  Best method:      {best_m[0]:<50s} combined={best_m[3]:.1%}")

    # ------ Plots ------------------------------------------------------------------------------------------------------------------------------------------------
    if args.plot:
        plot_results(labels, methods, win_matrix, unique_methods, h2h)


if __name__ == "__main__":
    main()
