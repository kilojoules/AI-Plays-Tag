#!/usr/bin/env python3
"""
README story animations for the Design C reward-shaping study.

Four GIFs (docs/design_c/), one per act of the story:

  01_seed_lottery.gif    Two R7/A=0 seekers, identical config, different
                         seed, vs the SAME reference hider — one hunts,
                         one is lost. The 3.4x seed lottery, visually.
  02_overspecialist.gif  One seeker, two opponents: 100% vs the hider it
                         co-evolved with, 13% vs a reference hider.
  03_sparse_basin.gif    Sparse-reward seeker that never learned pursuit —
                         fails even against its own training partner.
  04_zoo_rescue.gif      Shaping + zoo (A=0.5) champion running the
                         anchor gauntlet with a live score.
  05_coverage_trap.gif   The reward-term mechanism: a coverage-paid,
                         urgency-free seeker patrols the arena instead
                         of chasing; removing the bonus restores pursuit.
  06_zoo_dose.gif        The same seed at A=0 / 0.1 / 0.5 — the lottery
                         ticket gets de-risked by zoo dose.

Policies referenced here are the exact runs quoted in
experiments/design_c_results.md; win rates in captions come from the
anchor panel / 32-policy gauntlet, not from these illustrative episodes.

Usage:
  pixi run python experiments/animate_design_c_story.py            # all
  pixi run python experiments/animate_design_c_story.py --act 1
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation, PillowWriter

sys.path.insert(0, str(Path(__file__).parent.parent))

from trainer.tag_env import SingleTagEnv, TagEnvConfig, LAYOUTS
from trainer.ppo import PPOAgent, PPOConfig

ROOT = Path(__file__).parent.parent
DC = ROOT / "experiments" / "results" / "design_c"
OUT_DIR = ROOT / "docs" / "design_c"

HSM = 1.15
LAYOUT = "four_corners"
MAX_STEPS = 200
FPS = 20
FRAME_STRIDE = 2  # render every 2nd step to keep GIFs light

SEEKER_C = "#d62728"
HIDER_C = "#1f77b4"


def find_run(rel: str) -> Path:
    """Resolve grid/…/seed_N to its timestamped run dir."""
    base = DC / rel
    for ts in sorted(base.glob("2*"), reverse=True):
        if (ts / "policy_seeker_final.pt").exists() or \
           (ts / "policy_hider_final.pt").exists():
            return ts
    raise FileNotFoundError(f"no run under {base}")


def load_policy(path: Path, obs_dim: int, act_dim: int) -> PPOAgent:
    cfg = PPOConfig(obs_dim=obs_dim, act_dim=act_dim)
    agent = PPOAgent(cfg)
    ckpt = torch.load(str(path), map_location="cpu", weights_only=True)
    agent.pi.load_state_dict(ckpt["pi"])
    agent.vf.load_state_dict(ckpt["vf"])
    return agent


def get_action(policy: PPOAgent, obs: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        action, _, _ = policy.act(obs)
        return action.squeeze()


def run_episode(seeker, hider, env_config, rng_seed: int) -> Dict:
    """Play one episode; return role-correct position trace + outcome."""
    np.random.seed(rng_seed)
    torch.manual_seed(rng_seed)
    env = SingleTagEnv(config=env_config)
    obs = env.reset()

    frames = []
    done, step, tagged = False, 0, False
    while not done and step < MAX_STEPS:
        state = env.get_state()
        si = state["seeker_idx"]
        frames.append({
            "seeker_pos": state["positions"][si].copy(),
            "hider_pos": state["positions"][1 - si].copy(),
            "step": step,
        })
        s_act = get_action(seeker, obs["seeker"])
        h_act = get_action(hider, obs["hider"])
        obs, _, done, info = env.step({"seeker": s_act, "hider": h_act})
        step += 1
        if done:
            tagged = bool(info.get("tagged", False))
    # closing frame at the final positions
    state = env.get_state()
    si = state["seeker_idx"]
    frames.append({
        "seeker_pos": state["positions"][si].copy(),
        "hider_pos": state["positions"][1 - si].copy(),
        "step": step,
    })
    return {"frames": frames, "tagged": tagged, "steps": step}


def pick_episode(seeker, hider, env_config, want_tagged: Optional[bool],
                 tries: int = 24, base_seed: int = 0) -> Dict:
    """Sample episodes until the outcome matches the run's typical result
    (illustration should match the statistics, not cherry-pick against
    them). Falls back to the last sample if no episode matches."""
    ep = None
    candidates = []
    for k in range(tries):
        ep = run_episode(seeker, hider, env_config, rng_seed=base_seed + k)
        if want_tagged is None or ep["tagged"] == want_tagged:
            candidates.append(ep)
            if len(candidates) >= 3:
                break
    if candidates:
        # median episode length among matching candidates
        candidates.sort(key=lambda e: e["steps"])
        return candidates[len(candidates) // 2]
    return ep


def draw_arena(ax, env_config):
    layout = LAYOUTS.get(env_config.layout, LAYOUTS["empty"])
    half = env_config.arena_half
    ax.set_xlim(-half - 1, half + 1)
    ax.set_ylim(-half - 1, half + 1)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.add_patch(patches.Rectangle((-half, -half), 2 * half, 2 * half,
                                   linewidth=2, edgecolor="black",
                                   facecolor="white"))
    for obs in layout.get("obstacles", []):
        ax.add_patch(patches.Rectangle(
            (obs.x - obs.half_width, obs.y - obs.half_height),
            2 * obs.half_width, 2 * obs.half_height,
            linewidth=1, edgecolor="black", facecolor="gray", alpha=0.7))
    sz = layout.get("safe_zone", None)
    if sz:
        ax.add_patch(patches.Circle((sz.x, sz.y), sz.radius, linewidth=2,
                                    edgecolor="green",
                                    facecolor="lightgreen", alpha=0.3))


def draw_frame(ax, episode: Dict, frame_idx: int, panel_title: str,
               env_config, trail: int = 24):
    ax.clear()
    draw_arena(ax, env_config)
    frames = episode["frames"]
    i = min(frame_idx, len(frames) - 1)
    fr = frames[i]

    start = max(0, i - trail)
    if i - start > 1:
        s = np.array([frames[j]["seeker_pos"] for j in range(start, i + 1)])
        h = np.array([frames[j]["hider_pos"] for j in range(start, i + 1)])
        ax.plot(s[:, 0], s[:, 1], color=SEEKER_C, alpha=0.45, linewidth=1.4)
        ax.plot(h[:, 0], h[:, 1], color=HIDER_C, alpha=0.45, linewidth=1.4)

    ax.add_patch(patches.Circle(fr["seeker_pos"], 0.55, color=SEEKER_C, zorder=5))
    ax.add_patch(patches.Circle(fr["hider_pos"], 0.55, color=HIDER_C, zorder=5))

    over = i >= len(frames) - 1
    if over and episode["tagged"]:
        ax.add_patch(patches.Circle(fr["hider_pos"], 1.3, fill=False,
                                    color=SEEKER_C, linewidth=2.5, zorder=6))
        status = f"TAGGED @ step {episode['steps']}"
    elif over:
        status = f"hider survives ({episode['steps']} steps)"
    else:
        status = f"step {fr['step']}"
    ax.set_title(f"{panel_title}\n{status}", fontsize=11)


def save_side_by_side(episodes: List[Dict], titles: List[str], suptitle: str,
                      out_path: Path, env_config, hold_frames: int = 24,
                      trail: int = 24):
    """Panels run synchronized; finished panels hold their final frame."""
    n = len(episodes)
    fig, axes = plt.subplots(1, n, figsize=(5.4 * n, 6.0))
    if n == 1:
        axes = [axes]
    longest = max(len(e["frames"]) for e in episodes)
    n_frames = (longest + FRAME_STRIDE - 1) // FRAME_STRIDE + hold_frames

    def animate(k):
        idx = min(k * FRAME_STRIDE, longest - 1)
        for ax, ep, t in zip(axes, episodes, titles):
            draw_frame(ax, ep, idx, t, env_config, trail=trail)
        fig.suptitle(suptitle, fontsize=14, fontweight="bold")
        return []

    anim = FuncAnimation(fig, animate, frames=n_frames, interval=1000 // FPS,
                         blit=False)
    anim.save(str(out_path), writer=PillowWriter(fps=FPS))
    plt.close(fig)
    print(f"saved {out_path}  ({out_path.stat().st_size / 1e6:.1f} MB)")


def save_gauntlet(seeker, opponents, suptitle: str, out_path: Path,
                  env_config, hold_frames: int = 16):
    """Sequential episodes vs a list of (name, hider) with a running score."""
    episodes = []
    for k, (name, hider) in enumerate(opponents):
        ep = pick_episode(seeker, hider, env_config, want_tagged=True,
                          base_seed=100 + 37 * k)
        ep["opp_name"] = name
        episodes.append(ep)

    fig, ax = plt.subplots(figsize=(6.4, 7.0))
    schedule = []  # (episode_idx, frame_idx) per rendered frame
    for e_i, ep in enumerate(episodes):
        steps = list(range(0, len(ep["frames"]), FRAME_STRIDE))
        schedule += [(e_i, j) for j in steps]
        schedule += [(e_i, len(ep["frames"]) - 1)] * hold_frames

    def animate(k):
        e_i, j = schedule[min(k, len(schedule) - 1)]
        ep = episodes[e_i]
        score = sum(int(e["tagged"]) for e in episodes[:e_i]
                    ) + int(ep["tagged"] and j >= len(ep["frames"]) - 1)
        draw_frame(ax, ep, j, f"vs {ep['opp_name']}", env_config)
        ax.text(0.02, 0.02, f"score: {score}/{len(episodes)}",
                transform=ax.transAxes, fontsize=13, fontweight="bold",
                color=SEEKER_C, va="bottom")
        fig.suptitle(suptitle, fontsize=14, fontweight="bold")
        return []

    anim = FuncAnimation(fig, animate, frames=len(schedule),
                         interval=1000 // FPS, blit=False)
    anim.save(str(out_path), writer=PillowWriter(fps=FPS))
    plt.close(fig)
    print(f"saved {out_path}  ({out_path.stat().st_size / 1e6:.1f} MB)")


def coverage_stats(seeker, hider, env_config, n_episodes: int = 30,
                   base_seed: int = 500) -> Dict:
    """Measure what the area-coverage bonus actually paid: unique cells of
    the env's own 6x6 coverage grid visited by the seeker, per episode and
    per 100 steps. Used to back the act-5 'patrols instead of chasing'
    caption with a number instead of an eyeballed trajectory."""
    half = env_config.arena_half
    gsz = 6
    cells_ep, per100, tags = [], [], 0
    for k in range(n_episodes):
        ep = run_episode(seeker, hider, env_config, rng_seed=base_seed + k)
        pos = np.array([f["seeker_pos"] for f in ep["frames"]])
        ij = np.clip(((pos + half) / (2 * half / gsz)).astype(int), 0, gsz - 1)
        ncells = len({(a, b) for a, b in ij})
        cells_ep.append(ncells)
        per100.append(100.0 * ncells / max(ep["steps"], 1))
        tags += int(ep["tagged"])
    return dict(cells=float(np.mean(cells_ep)),
                per100=float(np.mean(per100)),
                wr=tags / n_episodes)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--act", type=int, default=None, choices=[1, 2, 3, 4, 5, 6])
    ap.add_argument("--coverage-stats", action="store_true",
                    help="Print 6x6-grid coverage rates for the act-5 pair "
                         "(tourist vs hunter) instead of rendering.")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    env_config = TagEnvConfig(layout=LAYOUT, hider_speed_mult=HSM)
    probe = SingleTagEnv(config=env_config)
    obs_dim, act_dim = probe.obs_dim, probe.act_dim

    def seeker_of(rel):
        return load_policy(find_run(rel) / "policy_seeker_final.pt",
                           obs_dim, act_dim)

    def hider_of(rel):
        return load_policy(find_run(rel) / "policy_hider_final.pt",
                           obs_dim, act_dim)

    ref_hider = hider_of("anchors/R4_sparse/A00/seed_42")

    if args.coverage_stats:
        tourist = seeker_of("urgency_only_ablation/R7_no_urgency/A00/seed_8")
        hunter = seeker_of("coverage_ablation/R7_no_coverage/A00/seed_7")
        for name, pol in [("tourist (coverage, no urgency)", tourist),
                          ("hunter (no coverage)", hunter)]:
            s = coverage_stats(pol, ref_hider, env_config)
            print(f"{name:32s} cells/episode={s['cells']:.1f}/36  "
                  f"cells/100 steps={s['per100']:.1f}  wr={s['wr']:.2f}")
        return

    if args.act in (None, 1):
        champ = seeker_of("grid/R7_kitchen_sink/A00/seed_2")
        flop = seeker_of("grid/R7_kitchen_sink/A00/seed_4")
        save_side_by_side(
            [pick_episode(champ, ref_hider, env_config, want_tagged=True),
             pick_episode(flop, ref_hider, env_config, want_tagged=False)],
            ["seed 2  —  93% win rate vs the pool",
             "seed 4  —  27% win rate vs the pool"],
            "The seed lottery: identical training, same opponent — only the seed differs",
            OUT_DIR / "01_seed_lottery.gif", env_config)

    if args.act in (None, 2):
        osp_run = "grid/R7_kitchen_sink/A00/seed_12"
        osp = seeker_of(osp_run)
        own = hider_of(osp_run)
        save_side_by_side(
            [pick_episode(osp, own, env_config, want_tagged=True),
             pick_episode(osp, ref_hider, env_config, want_tagged=False)],
            ["vs the hider it trained with  (100%)",
             "vs a reference hider  (13%)"],
            "The over-specialist: it can only beat the opponent it grew up with",
            OUT_DIR / "02_overspecialist.gif", env_config)

    if args.act in (None, 3):
        basin_run = "grid/R4_sparse/A00/seed_1"
        basin = seeker_of(basin_run)
        own = hider_of(basin_run)
        save_side_by_side(
            [pick_episode(basin, own, env_config, want_tagged=False),
             pick_episode(basin, ref_hider, env_config, want_tagged=False)],
            ["vs its own training partner  (17%)",
             "vs a reference hider  (11%)"],
            "The basin: sparse reward, no zoo — pursuit never emerges at all",
            OUT_DIR / "03_sparse_basin.gif", env_config)

    if args.act in (None, 4):
        rescue = seeker_of("grid/R7_kitchen_sink/A50/seed_0")
        opponents = [
            ("sparse-trained hider", ref_hider),
            ("shaped hider", hider_of("anchors/R7_kitchen_sink/A00/seed_42")),
            ("shaped + zoo hider", hider_of("anchors/R7_kitchen_sink/A50/seed_42")),
        ]
        save_gauntlet(
            rescue, opponents,
            "Shaping + zoo (A=0.5):\nthis seeker beats 99% of the pool",
            OUT_DIR / "04_zoo_rescue.gif", env_config)

    if args.act in (None, 5):
        tourist = seeker_of("urgency_only_ablation/R7_no_urgency/A00/seed_8")
        hunter = seeker_of("coverage_ablation/R7_no_coverage/A00/seed_7")
        save_side_by_side(
            [pick_episode(tourist, ref_hider, env_config, want_tagged=False),
             pick_episode(hunter, ref_hider, env_config, want_tagged=True)],
            ["coverage bonus, no urgency  (12% vs anchors)",
             "coverage bonus removed  (89% vs anchors)"],
            "The coverage trap: trained with an exploration bonus, it sweeps past the hider",
            OUT_DIR / "05_coverage_trap.gif", env_config, trail=60)

    if args.act in (None, 6):
        doses = [
            ("grid/R7_kitchen_sink/A00/seed_0", "A = 0  —  over-specialist (0.49)", False),
            ("a_sweep/R7_kitchen_sink/A10/seed_0", "A = 0.1  —  healthy (0.92)", True),
            ("grid/R7_kitchen_sink/A50/seed_0", "A = 0.5  —  healthy (1.00)", True),
        ]
        save_side_by_side(
            [pick_episode(seeker_of(rel), ref_hider, env_config, want_tagged=w)
             for rel, _, w in doses],
            [t for _, t, _ in doses],
            "Same seed, same reward — only the zoo dose differs",
            OUT_DIR / "06_zoo_dose.gif", env_config)


if __name__ == "__main__":
    main()
