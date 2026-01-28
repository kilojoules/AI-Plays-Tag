#!/usr/bin/env python3
"""
Checkpoint Gauntlet: Evaluate seeker checkpoint i vs hider checkpoint j.
Creates a heatmap showing win rates across all combinations.
"""
import sys
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))

from trainer.tag_env import SingleTagEnv, TagEnvConfig
from trainer.ppo import PPOAgent, PPOConfig


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


def main():
    # Find the stable training run
    stable_dir = Path("experiments/results/stable_long_training")
    run_dir = sorted(stable_dir.glob("*"))[-1]
    ckpt_dir = run_dir / "checkpoints"

    print(f"Loading checkpoints from: {ckpt_dir}")

    # Get all checkpoint updates
    seeker_ckpts = sorted(ckpt_dir.glob("seeker_*.pt"))
    all_updates = [int(p.stem.split('_')[1]) for p in seeker_ckpts]

    print(f"Found {len(all_updates)} checkpoints: {all_updates[0]} to {all_updates[-1]}")

    # Sample checkpoints at regular intervals (every 10th checkpoint = every 500 updates)
    # This gives us manageable ~8-10 checkpoints for the heatmap
    step = max(1, len(all_updates) // 10)
    selected_updates = all_updates[::step]

    # Always include the latest
    if all_updates[-1] not in selected_updates:
        selected_updates.append(all_updates[-1])

    print(f"Selected {len(selected_updates)} checkpoints for gauntlet: {selected_updates}")

    # Setup environment
    env_config = TagEnvConfig(layout="four_corners")
    env = SingleTagEnv(config=env_config)
    obs_dim, act_dim = env.obs_dim, env.act_dim

    # Load all policies
    print("\nLoading policies...")
    seekers = {}
    hiders = {}
    for update in selected_updates:
        seeker_path = ckpt_dir / f"seeker_{update:05d}.pt"
        hider_path = ckpt_dir / f"hider_{update:05d}.pt"
        seekers[update] = load_policy(str(seeker_path), obs_dim, act_dim)
        hiders[update] = load_policy(str(hider_path), obs_dim, act_dim)
        print(f"  Loaded update {update}")

    # Run gauntlet
    n_ckpts = len(selected_updates)
    win_matrix = np.zeros((n_ckpts, n_ckpts))
    steps_matrix = np.zeros((n_ckpts, n_ckpts))

    total_matchups = n_ckpts * n_ckpts
    print(f"\nRunning {total_matchups} matchups...")

    for i, s_update in enumerate(selected_updates):
        for j, h_update in enumerate(selected_updates):
            matchup_num = i * n_ckpts + j + 1
            print(f"  [{matchup_num}/{total_matchups}] Seeker {s_update} vs Hider {h_update}...", end=" ")

            win_rate, avg_steps = evaluate_matchup(
                seekers[s_update], hiders[h_update], env_config,
                n_episodes=20, max_steps=200
            )
            win_matrix[i, j] = win_rate
            steps_matrix[i, j] = avg_steps
            print(f"WR: {win_rate*100:.0f}%, Steps: {avg_steps:.0f}")

    # Create heatmap
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Win rate heatmap
    ax = axes[0]
    im = ax.imshow(win_matrix * 100, cmap='RdYlGn', vmin=0, vmax=100, aspect='auto')

    # Labels
    labels = [f"{u//100}k" if u >= 100 else str(u) for u in selected_updates]
    ax.set_xticks(range(n_ckpts))
    ax.set_yticks(range(n_ckpts))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Hider Checkpoint (update)", fontsize=11)
    ax.set_ylabel("Seeker Checkpoint (update)", fontsize=11)
    ax.set_title("Seeker Win Rate (%)", fontsize=12, fontweight='bold')

    # Add text annotations
    for i in range(n_ckpts):
        for j in range(n_ckpts):
            val = win_matrix[i, j] * 100
            color = 'white' if val < 30 or val > 70 else 'black'
            ax.text(j, i, f"{val:.0f}", ha='center', va='center', fontsize=8, color=color)

    plt.colorbar(im, ax=ax, label='Win Rate (%)')

    # Average steps heatmap
    ax = axes[1]
    im2 = ax.imshow(steps_matrix, cmap='viridis', aspect='auto')

    ax.set_xticks(range(n_ckpts))
    ax.set_yticks(range(n_ckpts))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Hider Checkpoint (update)", fontsize=11)
    ax.set_ylabel("Seeker Checkpoint (update)", fontsize=11)
    ax.set_title("Average Episode Length (steps)", fontsize=12, fontweight='bold')

    # Add text annotations
    for i in range(n_ckpts):
        for j in range(n_ckpts):
            val = steps_matrix[i, j]
            color = 'white' if val < 100 else 'black'
            ax.text(j, i, f"{val:.0f}", ha='center', va='center', fontsize=8, color=color)

    plt.colorbar(im2, ax=ax, label='Steps')

    # Convert updates to timesteps for title
    max_steps = selected_updates[-1] * 2048
    plt.suptitle(f"Checkpoint Gauntlet: Stable Long Training\n(Updates {selected_updates[0]} to {selected_updates[-1]}, ~{max_steps//1_000_000}M timesteps)",
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    output_path = "experiments/results/stable_training_viz/checkpoint_gauntlet.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {output_path}")

    # Print summary statistics
    print("\n" + "="*60)
    print("GAUNTLET SUMMARY")
    print("="*60)

    # Diagonal = same-epoch matchups (self-play performance)
    diag_wr = np.diag(win_matrix)
    print(f"\nDiagonal (contemporaneous matchups):")
    for i, (update, wr) in enumerate(zip(selected_updates, diag_wr)):
        print(f"  Update {update}: {wr*100:.0f}% seeker WR")
    print(f"  Average: {diag_wr.mean()*100:.1f}%")

    # Row averages = how good is each seeker across all hiders
    row_avg = win_matrix.mean(axis=1)
    print(f"\nSeeker strength (avg WR vs all hiders):")
    best_seeker_idx = np.argmax(row_avg)
    for i, (update, wr) in enumerate(zip(selected_updates, row_avg)):
        marker = " <-- BEST" if i == best_seeker_idx else ""
        print(f"  Update {update}: {wr*100:.1f}%{marker}")

    # Column averages = how good is each hider against all seekers
    col_avg = 1 - win_matrix.mean(axis=0)  # Invert for hider perspective
    print(f"\nHider strength (avg survival vs all seekers):")
    best_hider_idx = np.argmax(col_avg)
    for i, (update, wr) in enumerate(zip(selected_updates, col_avg)):
        marker = " <-- BEST" if i == best_hider_idx else ""
        print(f"  Update {update}: {wr*100:.1f}%{marker}")

    # Cross-temporal analysis
    print(f"\nCross-temporal patterns:")
    # Early seeker vs late hider
    early_idx = 0
    late_idx = -1
    print(f"  Early seeker ({selected_updates[early_idx]}) vs Late hider ({selected_updates[late_idx]}): {win_matrix[early_idx, late_idx]*100:.0f}%")
    print(f"  Late seeker ({selected_updates[late_idx]}) vs Early hider ({selected_updates[early_idx]}): {win_matrix[late_idx, early_idx]*100:.0f}%")


if __name__ == "__main__":
    main()
