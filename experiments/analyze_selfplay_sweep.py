#!/usr/bin/env python3
"""
Analyze self-play sweep results: gauntlet matchups, training curves, and GIF animations.

Subcommands:
  gauntlet  - Cross-checkpoint evaluation heatmaps
  curves    - Training metric line plots
  animate   - GIF animations of final policies
  all       - Run all of the above
"""
import argparse
import csv
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.ticker as ticker
from matplotlib.animation import FuncAnimation, FFMpegWriter

sys.path.insert(0, str(Path(__file__).parent.parent))

from trainer.tag_env import SingleTagEnv, TagEnvConfig, LAYOUTS
from trainer.ppo import PPOAgent, PPOConfig

# ---------------------------------------------------------------------------
# Constants (matching run_selfplay_sweep.py / plot_selfplay_sweep.py)
# ---------------------------------------------------------------------------

SWEEP_DIR = Path("experiments/results/selfplay_sweep")
ANALYSIS_DIR = SWEEP_DIR / "analysis"

STP_VALUES = [0.005, 0.01, 0.02, 0.05]
HSM_VALUES = [1.0, 1.05, 1.1, 1.15, 1.2]

EXPERIMENTS = {}
for stp in STP_VALUES:
    for hsm in HSM_VALUES:
        stp_str = f"{stp:.4f}".replace("0.", "").rstrip("0") or "0"
        hsm_str = f"{round(hsm * 100)}"
        name = f"STP{stp_str}_HSM{hsm_str}"
        EXPERIMENTS[name] = {"stp": stp, "hsm": hsm}

STP_COLORS = {0.005: "#1f77b4", 0.01: "#ff7f0e", 0.02: "#2ca02c", 0.05: "#d62728"}
HSM_STYLES = {1.0: "-", 1.05: "--", 1.1: "-.", 1.15: ":", 1.2: (0, (3, 1, 1, 1))}

N_GAUNTLET_CKPTS = 10   # evenly spaced checkpoints per experiment
N_GAUNTLET_EPISODES = 20
MAX_STEPS = 200

# ---------------------------------------------------------------------------
# Shared helpers (verbatim from checkpoint_gauntlet.py)
# ---------------------------------------------------------------------------

def load_policy(path: str, obs_dim: int, act_dim: int) -> PPOAgent:
    cfg = PPOConfig(obs_dim=obs_dim, act_dim=act_dim)
    policy = PPOAgent(cfg)
    ckpt = torch.load(path, map_location='cpu', weights_only=True)
    policy.pi.load_state_dict(ckpt["pi"])
    policy.vf.load_state_dict(ckpt["vf"])
    return policy


def evaluate_matchup(seeker, hider, env_config, n_episodes=20, max_steps=200):
    """Evaluate seeker vs hider over n_episodes."""
    env = SingleTagEnv(config=env_config)
    wins = 0
    total_steps = []

    for _ in range(n_episodes):
        obs = env.reset()
        done = False
        step = 0

        while not done and step < max_steps:
            with torch.no_grad():
                s_act, _, _ = seeker.act(obs['seeker'])
                h_act, _, _ = hider.act(obs['hider'])

            obs, _, done, info = env.step({
                'seeker': s_act.squeeze(),
                'hider': h_act.squeeze()
            })
            step += 1

        if info.get('tagged', False):
            wins += 1
        total_steps.append(step)

    return wins / n_episodes, np.mean(total_steps)


# ---------------------------------------------------------------------------
# Shared helpers (verbatim from animate_zoo_sweep.py)
# ---------------------------------------------------------------------------

def run_episode(seeker, hider, env_config, max_steps=200):
    env = SingleTagEnv(config=env_config)
    obs = env.reset()
    trajectory = []
    done = False
    step = 0

    while not done and step < max_steps:
        state = env.get_state()
        seeker_idx = state['seeker_idx']
        positions = state['positions']
        trajectory.append({
            'seeker_pos': positions[seeker_idx].copy(),
            'hider_pos': positions[1 - seeker_idx].copy(),
            'step': step,
            'safe_zone_exhausted': state['safe_zone_exhausted'],
        })
        with torch.no_grad():
            s_act, _, _ = seeker.act(obs['seeker'])
            h_act, _, _ = hider.act(obs['hider'])
        obs, _, done, info = env.step({'seeker': s_act.squeeze(), 'hider': h_act.squeeze()})
        step += 1

    # Final frame
    state = env.get_state()
    seeker_idx = state['seeker_idx']
    positions = state['positions']
    trajectory.append({
        'seeker_pos': positions[seeker_idx].copy(),
        'hider_pos': positions[1 - seeker_idx].copy(),
        'step': step,
        'tagged': info.get('tagged', False),
        'safe_zone_exhausted': state['safe_zone_exhausted'],
    })
    return trajectory


def create_animation(trajectory, title, output_path, env_config):
    fig, ax = plt.subplots(figsize=(8, 8))

    layout = LAYOUTS.get(env_config.layout, LAYOUTS['empty'])
    obstacles = layout.get('obstacles', [])
    safe_zone = layout.get('safe_zone', None)
    arena_size = env_config.arena_half

    final = trajectory[-1]
    result = "TAGGED!" if final.get('tagged', False) else "Survived"
    n_frames = len(trajectory)

    def animate(frame_idx):
        ax.clear()
        ax.set_xlim(-arena_size - 1, arena_size + 1)
        ax.set_ylim(-arena_size - 1, arena_size + 1)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.2)

        # Arena
        arena_rect = patches.Rectangle(
            (-arena_size, -arena_size), 2 * arena_size, 2 * arena_size,
            linewidth=2, edgecolor='black', facecolor='#f8f8f8'
        )
        ax.add_patch(arena_rect)

        # Obstacles
        for obs in obstacles:
            rect = patches.Rectangle(
                (obs.x - obs.half_width, obs.y - obs.half_height),
                2 * obs.half_width, 2 * obs.half_height,
                linewidth=1, edgecolor='#555', facecolor='#aaa', alpha=0.8
            )
            ax.add_patch(rect)

        # Safe zone
        if safe_zone:
            frame_data = trajectory[min(frame_idx, n_frames - 1)]
            exhausted = frame_data.get('safe_zone_exhausted', False)
            if exhausted:
                fc, ec = 'lightsalmon', 'red'
            else:
                fc, ec = 'lightgreen', 'green'
            circle = patches.Circle(
                (safe_zone.x, safe_zone.y), safe_zone.radius,
                linewidth=2, edgecolor=ec, facecolor=fc, alpha=0.35
            )
            ax.add_patch(circle)

        frame = trajectory[min(frame_idx, n_frames - 1)]
        seeker_pos = frame['seeker_pos']
        hider_pos = frame['hider_pos']

        # Trails (last 30 frames)
        trail_start = max(0, frame_idx - 30)
        if frame_idx > 0:
            seeker_trail = [trajectory[i]['seeker_pos'] for i in range(trail_start, frame_idx + 1)]
            hider_trail = [trajectory[i]['hider_pos'] for i in range(trail_start, frame_idx + 1)]

            # Fading trail
            for i in range(len(seeker_trail) - 1):
                alpha = 0.1 + 0.5 * (i / max(len(seeker_trail) - 1, 1))
                ax.plot([seeker_trail[i][0], seeker_trail[i+1][0]],
                        [seeker_trail[i][1], seeker_trail[i+1][1]],
                        'r-', alpha=alpha, linewidth=2)
                ax.plot([hider_trail[i][0], hider_trail[i+1][0]],
                        [hider_trail[i][1], hider_trail[i+1][1]],
                        'b-', alpha=alpha, linewidth=2)

        # Agents
        ax.add_patch(patches.Circle(seeker_pos, 0.6, color='#e74c3c', zorder=5))
        ax.add_patch(patches.Circle(hider_pos, 0.6, color='#3498db', zorder=5))
        ax.annotate('S', seeker_pos, ha='center', va='center',
                     fontsize=10, fontweight='bold', color='white', zorder=6)
        ax.annotate('H', hider_pos, ha='center', va='center',
                     fontsize=10, fontweight='bold', color='white', zorder=6)

        # Title
        step_str = f"Step {frame['step']}/{n_frames - 1}"
        if frame_idx == n_frames - 1:
            step_str += f"  |  {result}"
        ax.set_title(f"{title}\n{step_str}", fontsize=11, fontweight='bold')

        return []

    anim = FuncAnimation(fig, animate, frames=n_frames, interval=50, blit=False)
    writer = FFMpegWriter(fps=20)
    anim.save(output_path, writer=writer)
    plt.close()
    print(f"  Saved: {output_path}")


# ---------------------------------------------------------------------------
# Shared helpers (from plot_selfplay_sweep.py)
# ---------------------------------------------------------------------------

def load_metrics(name):
    """Load metrics.csv for a named experiment."""
    exp_dir = SWEEP_DIR / name
    candidates = sorted(exp_dir.glob("2*"), reverse=True)
    for d in candidates:
        csv_path = d / "metrics.csv"
        if csv_path.exists():
            rows = []
            with open(csv_path) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    rows.append({k: float(v) for k, v in row.items()})
            return rows
    return []


def millions(x, _):
    return f"{x / 1e6:.1f}M"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def find_run_dir(exp_name):
    """Find the latest timestamped run directory for an experiment."""
    exp_dir = SWEEP_DIR / exp_name
    if not exp_dir.exists():
        return None
    candidates = sorted(exp_dir.glob("2*"), reverse=True)
    for d in candidates:
        if (d / "metrics.csv").exists():
            return d
    return None


def find_checkpoints(exp_name, n_ckpts=N_GAUNTLET_CKPTS):
    """Find n evenly-spaced checkpoint update numbers for an experiment."""
    run_dir = find_run_dir(exp_name)
    if run_dir is None:
        return None, []
    ckpt_dir = run_dir / "checkpoints"
    if not ckpt_dir.exists():
        return run_dir, []
    seeker_ckpts = sorted(ckpt_dir.glob("seeker_*.pt"))
    if not seeker_ckpts:
        return run_dir, []

    all_updates = [int(p.stem.split('_')[1]) for p in seeker_ckpts]
    step = max(1, len(all_updates) // n_ckpts)
    selected = all_updates[::step]
    if all_updates[-1] not in selected:
        selected.append(all_updates[-1])
    return run_dir, selected


def make_env_config(exp_name):
    """Build TagEnvConfig with correct params for a given experiment."""
    params = EXPERIMENTS[exp_name]
    return TagEnvConfig(
        layout="four_corners",
        seeker_time_penalty=-params["stp"],
        hider_speed_mult=params["hsm"],
    )


# ---------------------------------------------------------------------------
# Subcommand: gauntlet
# ---------------------------------------------------------------------------

def _run_gauntlet_for_experiment(exp_name):
    """Run full gauntlet for one experiment. Returns (exp_name, win_matrix, steps_matrix, updates) or None."""
    run_dir, updates = find_checkpoints(exp_name)
    if not updates:
        print(f"  {exp_name}: no checkpoints, skipping")
        return None

    env_config = make_env_config(exp_name)
    env = SingleTagEnv(config=env_config)
    obs_dim, act_dim = env.obs_dim, env.act_dim

    # Load all checkpoint policies
    ckpt_dir = run_dir / "checkpoints"
    seekers = {}
    hiders = {}
    for u in updates:
        seekers[u] = load_policy(str(ckpt_dir / f"seeker_{u:05d}.pt"), obs_dim, act_dim)
        hiders[u] = load_policy(str(ckpt_dir / f"hider_{u:05d}.pt"), obs_dim, act_dim)

    n = len(updates)
    win_matrix = np.zeros((n, n))
    steps_matrix = np.zeros((n, n))

    for i, s_u in enumerate(updates):
        for j, h_u in enumerate(updates):
            wr, avg_s = evaluate_matchup(
                seekers[s_u], hiders[h_u], env_config,
                n_episodes=N_GAUNTLET_EPISODES, max_steps=MAX_STEPS)
            win_matrix[i, j] = wr
            steps_matrix[i, j] = avg_s

    return exp_name, win_matrix, steps_matrix, updates


def _plot_single_heatmap(exp_name, win_matrix, updates, output_dir):
    """Plot a single heatmap for one experiment."""
    n = len(updates)
    fig, ax = plt.subplots(figsize=(8, 7))

    im = ax.imshow(win_matrix * 100, cmap='RdYlGn', vmin=0, vmax=100, aspect='auto')
    labels = [f"{u//100}k" if u >= 100 else str(u) for u in updates]
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Hider Checkpoint (update)")
    ax.set_ylabel("Seeker Checkpoint (update)")

    params = EXPERIMENTS[exp_name]
    ax.set_title(f"{exp_name}\nstp={params['stp']}, hsm={params['hsm']:.2f}",
                 fontsize=11, fontweight='bold')

    for i in range(n):
        for j in range(n):
            val = win_matrix[i, j] * 100
            color = 'white' if val < 30 or val > 70 else 'black'
            ax.text(j, i, f"{val:.0f}", ha='center', va='center', fontsize=7, color=color)

    plt.colorbar(im, ax=ax, label='Seeker Win Rate (%)')
    plt.tight_layout()
    plt.savefig(output_dir / f"{exp_name}.png", dpi=150)
    plt.close()


def _plot_gauntlet_grid(all_results, output_dir):
    """Plot 4x5 grid of gauntlet heatmaps."""
    fig, axes = plt.subplots(len(STP_VALUES), len(HSM_VALUES),
                             figsize=(4 * len(HSM_VALUES), 3.5 * len(STP_VALUES)))
    fig.suptitle("Self-Play Sweep: Checkpoint Gauntlet\n(Seeker Win Rate %)",
                 fontsize=14, fontweight='bold')

    for si, stp in enumerate(STP_VALUES):
        for hi, hsm in enumerate(HSM_VALUES):
            ax = axes[si, hi]
            stp_str = f"{stp:.4f}".replace("0.", "").rstrip("0") or "0"
            hsm_str = f"{round(hsm * 100)}"
            name = f"STP{stp_str}_HSM{hsm_str}"

            if name in all_results:
                wm = all_results[name]['win_matrix']
                updates = all_results[name]['updates']
                n = len(updates)
                im = ax.imshow(wm * 100, cmap='RdYlGn', vmin=0, vmax=100, aspect='auto')
                labels = [f"{u//100}k" if u >= 100 else str(u) for u in updates]
                ax.set_xticks(range(n))
                ax.set_yticks(range(n))
                ax.set_xticklabels(labels, fontsize=5, rotation=45)
                ax.set_yticklabels(labels, fontsize=5)
            else:
                ax.text(0.5, 0.5, "No data", ha='center', va='center',
                        transform=ax.transAxes, fontsize=10, color='gray')

            ax.set_title(f"stp={stp} hsm={hsm:.2f}", fontsize=8)

            if si == len(STP_VALUES) - 1:
                ax.set_xlabel("Hider ckpt", fontsize=7)
            if hi == 0:
                ax.set_ylabel("Seeker ckpt", fontsize=7)

    plt.tight_layout()
    plt.savefig(output_dir / "gauntlet_grid.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_dir / 'gauntlet_grid.png'}")


def _save_gauntlet_csv(all_results, output_dir):
    """Save gauntlet summary as CSV."""
    csv_path = output_dir / "gauntlet_summary.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'experiment', 'stp', 'hsm',
            'diagonal_wr_mean', 'diagonal_wr_std',
            'best_seeker_update', 'best_seeker_wr',
            'best_hider_update', 'best_hider_survival',
        ])
        for name in sorted(all_results):
            params = EXPERIMENTS[name]
            wm = all_results[name]['win_matrix']
            updates = all_results[name]['updates']
            diag = np.diag(wm)
            row_avg = wm.mean(axis=1)
            col_avg = 1 - wm.mean(axis=0)
            best_s_idx = np.argmax(row_avg)
            best_h_idx = np.argmax(col_avg)
            writer.writerow([
                name, params['stp'], params['hsm'],
                f"{diag.mean():.4f}", f"{diag.std():.4f}",
                updates[best_s_idx], f"{row_avg[best_s_idx]:.4f}",
                updates[best_h_idx], f"{col_avg[best_h_idx]:.4f}",
            ])
    print(f"  Saved: {csv_path}")


def _save_gauntlet_matrices(all_results, output_dir):
    """Save machine-readable gauntlet matrices as JSON and per-experiment CSVs."""
    matrices_dir = output_dir / "matrices"
    matrices_dir.mkdir(parents=True, exist_ok=True)

    combined = {}
    for name in sorted(all_results):
        res = all_results[name]
        params = EXPERIMENTS[name]
        wm = res['win_matrix']
        sm = res['steps_matrix']
        updates = res['updates']

        # Per-experiment CSV: rows=seeker_ckpt, cols=hider_ckpt
        csv_path = matrices_dir / f"{name}_win_matrix.csv"
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["seeker\\hider"] + [str(u) for u in updates])
            for i, u in enumerate(updates):
                writer.writerow([str(u)] + [f"{wm[i, j]:.4f}" for j in range(len(updates))])

        csv_path = matrices_dir / f"{name}_steps_matrix.csv"
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["seeker\\hider"] + [str(u) for u in updates])
            for i, u in enumerate(updates):
                writer.writerow([str(u)] + [f"{sm[i, j]:.1f}" for j in range(len(updates))])

        # JSON entry
        combined[name] = {
            "stp": params['stp'],
            "hsm": params['hsm'],
            "updates": [int(u) for u in updates],
            "win_matrix": wm.tolist(),
            "steps_matrix": sm.tolist(),
        }

    json_path = output_dir / "gauntlet_matrices.json"
    with open(json_path, 'w') as f:
        json.dump(combined, f, indent=2)
    print(f"  Saved: {json_path}")
    print(f"  Saved: {matrices_dir}/ ({len(all_results) * 2} CSVs)")


def cmd_gauntlet(args):
    """Run checkpoint gauntlet for all experiments."""
    output_dir = ANALYSIS_DIR / "gauntlet"
    output_dir.mkdir(parents=True, exist_ok=True)

    cache_path = output_dir / "gauntlet_cache.npz"
    all_results = {}

    # Load cache if exists
    if cache_path.exists() and not args.no_cache:
        print("Loading cached gauntlet results...")
        data = np.load(cache_path, allow_pickle=True)
        for name in data['names']:
            name = str(name)
            all_results[name] = {
                'win_matrix': data[f'{name}_win'],
                'steps_matrix': data[f'{name}_steps'],
                'updates': list(data[f'{name}_updates']),
            }
        print(f"  Loaded {len(all_results)} cached experiments")

    # Find experiments that need evaluation
    missing = [n for n in EXPERIMENTS if n not in all_results]
    if missing:
        print(f"\nRunning gauntlet for {len(missing)} experiments...")
        max_workers = min(args.max_workers, len(missing))

        if max_workers > 1:
            with ProcessPoolExecutor(max_workers=max_workers) as pool:
                futures = {pool.submit(_run_gauntlet_for_experiment, n): n
                           for n in missing}
                for future in as_completed(futures):
                    result = future.result()
                    if result is not None:
                        name, wm, sm, updates = result
                        all_results[name] = {
                            'win_matrix': wm,
                            'steps_matrix': sm,
                            'updates': updates,
                        }
                        print(f"  Done: {name} ({len(updates)} checkpoints)")
        else:
            for name in missing:
                print(f"\n  --- {name} ---")
                result = _run_gauntlet_for_experiment(name)
                if result is not None:
                    _, wm, sm, updates = result
                    all_results[name] = {
                        'win_matrix': wm,
                        'steps_matrix': sm,
                        'updates': updates,
                    }
                    print(f"  Done: {name} ({len(updates)} checkpoints)")

        # Save cache
        cache_data = {'names': np.array(list(all_results.keys()))}
        for name, res in all_results.items():
            cache_data[f'{name}_win'] = res['win_matrix']
            cache_data[f'{name}_steps'] = res['steps_matrix']
            cache_data[f'{name}_updates'] = np.array(res['updates'])
        np.savez(cache_path, **cache_data)
        print(f"\n  Cached results to: {cache_path}")

    if not all_results:
        print("No gauntlet results to plot!")
        return

    # Per-experiment heatmaps
    print("\nPlotting per-experiment heatmaps...")
    for name, res in sorted(all_results.items()):
        _plot_single_heatmap(name, res['win_matrix'], res['updates'], output_dir)
    print(f"  Saved {len(all_results)} individual heatmaps")

    # Grid plot
    print("Plotting 4x5 grid...")
    _plot_gauntlet_grid(all_results, output_dir)

    # CSV summary
    print("Saving CSV summary...")
    _save_gauntlet_csv(all_results, output_dir)

    # Machine-readable matrices
    print("Saving machine-readable matrices...")
    _save_gauntlet_matrices(all_results, output_dir)

    print(f"\nGauntlet complete! Output: {output_dir}")


# ---------------------------------------------------------------------------
# Subcommand: curves
# ---------------------------------------------------------------------------

def cmd_curves(args):
    """Plot training curves from metrics.csv files."""
    output_dir = ANALYSIS_DIR / "curves"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading metrics...")
    all_rows = {}
    for name in EXPERIMENTS:
        rows = load_metrics(name)
        if rows:
            all_rows[name] = rows
            print(f"  {name}: {len(rows)} rows, {rows[-1]['timesteps']:.0f} steps")
        else:
            print(f"  {name}: no data")

    if not all_rows:
        print("No data found!")
        return

    print(f"\nLoaded {len(all_rows)}/{len(EXPERIMENTS)} experiments")

    # --- Training curves (4 subplots) ---
    metrics = ["seeker_win_rate", "episode_length_mean",
               "seeker_reward_mean", "hider_reward_mean"]
    titles = ["Seeker Win Rate", "Episode Length",
              "Seeker Reward", "Hider Reward"]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Self-Play Sweep: Training Curves", fontsize=13, fontweight="bold")

    for ax, metric, title in zip(axes.flat, metrics, titles):
        for name, rows in sorted(all_rows.items()):
            params = EXPERIMENTS[name]
            ts = [r["timesteps"] for r in rows]
            vals = [r[metric] for r in rows]
            # Smooth with rolling mean
            if len(vals) > 20:
                kernel = np.ones(20) / 20
                vals_smooth = np.convolve(vals, kernel, mode='valid')
                ts_smooth = ts[19:]
            else:
                vals_smooth = vals
                ts_smooth = ts
            ax.plot(ts_smooth, vals_smooth,
                    color=STP_COLORS[params["stp"]],
                    ls=HSM_STYLES[params["hsm"]],
                    alpha=0.7, linewidth=1.2)

        ax.set_xlabel("Timesteps")
        ax.set_title(title)
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(millions))
        ax.grid(True, alpha=0.3)

        if metric == "seeker_win_rate":
            ax.set_ylim(0, 1)
            ax.axhline(0.5, color="gray", ls=":", alpha=0.5)
        elif "reward" in metric:
            ax.axhline(0, color="gray", ls=":", alpha=0.5)

    # Build legend entries
    legend_handles = []
    legend_labels = []
    for stp in STP_VALUES:
        h, = axes[0, 0].plot([], [], color=STP_COLORS[stp], linewidth=2)
        legend_handles.append(h)
        legend_labels.append(f"stp={stp}")
    for hsm in HSM_VALUES:
        h, = axes[0, 0].plot([], [], color='gray', ls=HSM_STYLES[hsm], linewidth=2)
        legend_handles.append(h)
        legend_labels.append(f"hsm={hsm:.2f}")
    fig.legend(legend_handles, legend_labels, loc="center right",
               fontsize=8, bbox_to_anchor=(1.12, 0.5))

    plt.tight_layout()
    out_path = output_dir / "training_curves.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")

    # --- Heatmaps of final metrics ---
    all_final = {}
    for name, rows in all_rows.items():
        if not rows:
            continue
        n = max(1, int(len(rows) * 0.1))
        tail = rows[-n:]
        all_final[name] = {
            "seeker_win_rate": np.mean([r["seeker_win_rate"] for r in tail]),
            "episode_length_mean": np.mean([r["episode_length_mean"] for r in tail]),
            "seeker_reward_mean": np.mean([r["seeker_reward_mean"] for r in tail]),
            "hider_reward_mean": np.mean([r["hider_reward_mean"] for r in tail]),
        }

    cmaps = ["RdYlGn", "viridis", "RdYlGn", "RdYlGn_r"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Self-Play Sweep: Final Metrics (last 10%)\n"
                 "seeker_time_penalty x hider_speed_mult",
                 fontsize=13, fontweight="bold")

    for ax, metric, title, cmap in zip(axes.flat, metrics, titles, cmaps):
        grid = np.full((len(STP_VALUES), len(HSM_VALUES)), np.nan)
        for name, params in EXPERIMENTS.items():
            if name not in all_final:
                continue
            si = STP_VALUES.index(params["stp"])
            hi = HSM_VALUES.index(params["hsm"])
            grid[si, hi] = all_final[name][metric]

        im = ax.imshow(grid, aspect="auto", origin="lower", cmap=cmap)
        fig.colorbar(im, ax=ax, shrink=0.8)
        ax.set_xticks(range(len(HSM_VALUES)))
        ax.set_xticklabels([f"{v:.2f}" for v in HSM_VALUES])
        ax.set_yticks(range(len(STP_VALUES)))
        ax.set_yticklabels([f"{v}" for v in STP_VALUES])
        ax.set_xlabel("hider_speed_mult")
        ax.set_ylabel("seeker_time_penalty (abs)")
        ax.set_title(title)

        for si in range(len(STP_VALUES)):
            for hi in range(len(HSM_VALUES)):
                val = grid[si, hi]
                if not np.isnan(val):
                    fmt = f"{val:.2f}" if abs(val) < 100 else f"{val:.0f}"
                    ax.text(hi, si, fmt, ha="center", va="center",
                            fontsize=8, fontweight="bold",
                            color="white" if abs(val) > np.nanmax(np.abs(grid)) * 0.7 else "black")

    plt.tight_layout()
    out_path = output_dir / "heatmaps.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")

    print(f"\nCurves complete! Output: {output_dir}")


# ---------------------------------------------------------------------------
# Subcommand: animate
# ---------------------------------------------------------------------------

def cmd_animate(args):
    """Create MP4 animations of final policies for each experiment."""
    output_dir = ANALYSIS_DIR / "animations"
    output_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(42)
    torch.manual_seed(42)

    print("=" * 60)
    print("SELF-PLAY SWEEP: EPISODE ANIMATIONS")
    print("=" * 60)

    results_summary = []

    for exp_name in sorted(EXPERIMENTS.keys()):
        print(f"\n--- {exp_name} ---")
        run_dir = find_run_dir(exp_name)
        if run_dir is None:
            print(f"  SKIP: no run directory")
            continue

        # Load final policies (prefer policy_*_final.pt, fall back to latest checkpoint)
        seeker_final = run_dir / "policy_seeker_final.pt"
        hider_final = run_dir / "policy_hider_final.pt"

        if seeker_final.exists() and hider_final.exists():
            seeker_path = str(seeker_final)
            hider_path = str(hider_final)
        else:
            ckpt_dir = run_dir / "checkpoints"
            if not ckpt_dir.exists():
                print(f"  SKIP: no checkpoints or final policies")
                continue
            seeker_ckpts = sorted(ckpt_dir.glob("seeker_*.pt"))
            hider_ckpts = sorted(ckpt_dir.glob("hider_*.pt"))
            if not seeker_ckpts or not hider_ckpts:
                print(f"  SKIP: no checkpoints found")
                continue
            seeker_path = str(seeker_ckpts[-1])
            hider_path = str(hider_ckpts[-1])

        env_config = make_env_config(exp_name)
        env = SingleTagEnv(config=env_config)
        obs_dim, act_dim = env.obs_dim, env.act_dim

        print(f"  Seeker: {Path(seeker_path).name}")
        print(f"  Hider:  {Path(hider_path).name}")

        seeker = load_policy(seeker_path, obs_dim, act_dim)
        hider = load_policy(hider_path, obs_dim, act_dim)

        trajectory = run_episode(seeker, hider, env_config)

        final = trajectory[-1]
        tagged = final.get('tagged', False)
        result_str = "Tagged" if tagged else "Survived"
        steps = final['step']
        print(f"  Result: {result_str} in {steps} steps")
        results_summary.append((exp_name, result_str, steps))

        params = EXPERIMENTS[exp_name]
        title = f"{exp_name} (stp={params['stp']}, hsm={params['hsm']:.2f})"
        mp4_path = str(output_dir / f"{exp_name}.mp4")
        create_animation(trajectory, title, mp4_path, env_config)

    # Summary
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    for name, result, steps in results_summary:
        print(f"  {name:20s}: {result:8s} ({steps} steps)")

    print(f"\nAnimations saved to: {output_dir}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Analyze self-play sweep results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", help="Analysis subcommand")

    # gauntlet
    p_gauntlet = sub.add_parser("gauntlet", help="Checkpoint gauntlet evaluation")
    p_gauntlet.add_argument("--max-workers", type=int, default=4,
                            help="Max parallel experiments (default: 4)")
    p_gauntlet.add_argument("--no-cache", action="store_true",
                            help="Ignore cached gauntlet results")

    # curves
    sub.add_parser("curves", help="Training curve plots")

    # animate
    sub.add_parser("animate", help="GIF animations of final policies")

    # all
    p_all = sub.add_parser("all", help="Run all analyses")
    p_all.add_argument("--max-workers", type=int, default=4,
                       help="Max parallel experiments for gauntlet")
    p_all.add_argument("--no-cache", action="store_true",
                       help="Ignore cached gauntlet results")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    if args.command == "gauntlet":
        cmd_gauntlet(args)
    elif args.command == "curves":
        cmd_curves(args)
    elif args.command == "animate":
        cmd_animate(args)
    elif args.command == "all":
        cmd_curves(args)
        cmd_gauntlet(args)
        cmd_animate(args)


if __name__ == "__main__":
    main()
