AI Tag Game (2D Python RL)

Overview
- 2D reinforcement learning tag game with vectorized Python environment.
- Self-play training where seeker and hider agents learn simultaneously.
- Open-source stack: Python, PyTorch, numpy, matplotlib.

Structure
- `trainer/`: RL training, vectorized 2D environment, and visualization tools.
- `data/`: Workspace-local runtime artifacts (trajectories, frame dumps). Override with `AI_DATA_ROOT` if embedding the project elsewhere.
- `scripts/`: Automation helpers (debug artifacts, plotting, encoding).
- `experiments/`: Self-play vs SCRO comparison experiments.

Quick Start: Training

1. Train agents (takes ~1 minute for 500K steps):
   - `pixi run train`              # Default: 500K timesteps, 64 parallel envs
   - `pixi run train-quick`        # Quick test: 100K timesteps
   - `pixi run train-full`         # Full training: 1M timesteps

2. Visualize outcomes (generates trajectory plots and statistics):
   - `pixi run visualize`          # Run 50 eval episodes, generate charts
   - `pixi run visualize-anim`     # Also create animated MP4s (slower)

3. Train with obstacles:
   - `pixi run train-obstacles`        # four_corners layout
   - `pixi run train-obstacles-full`   # 1M timesteps with obstacles

Output locations:
- Trained policies: `trainer/policy_seeker.pt`, `trainer/policy_hider.pt`
- Training logs: `trainer/logs/fast_train/<run_id>/`
- Visualizations: `trainer/visualizations/<timestamp>/`

The training uses self-play where both seeker and hider learn simultaneously.

Tag Rules
- One agent starts as "it". If the "it" agent's tag area touches another agent that is not immune, the "it" status is transferred.
- Newly tagged agents gain short immunity to prevent immediate tag-back.

Python Environment (Pixi preferred)
- Why Pixi: pip on Python 3.13 lacks PyTorch wheels; Pixi pins a compatible Python and installs from conda-forge/pytorch channels.
- Install Pixi (see https://pixi.sh for platform-specific install), then from repo root:
  - `pixi run train`              # start training
  - `pixi run -e train plot`      # render charts for the latest training run
  - `pixi run collect-debug`      # gather logs/metrics into `debug/<timestamp>/`
  - `pixi run monitor`            # build dashboards for the most recent run of each approach
  - `pixi run monitor-all`        # aggregate charts across every recorded run

Runtime Data Directory
- All training/eval artifacts land inside `data/` by default:
  - `data/trajectories/ep_*.jsonl` — evaluation traces.
  - `data/frames/frame_*.png` — recorded frames.
- Shell helpers (`scripts/lib/data_paths.sh`) expose the shared paths.
- Override `AI_DATA_ROOT=/custom/path` when you need an alternate workspace.

Training Monitoring
- Every training session is grouped under `trainer/logs/runs/<approach>/<run_id>/`. The approach defaults to the training mode, and you can override it with `TRAIN_APPROACH=custom-label`.
- Each run directory contains `metrics.csv` (per-episode stats including per-role rewards, win outcomes, and episode duration), rotating policy checkpoints, TensorBoard logs, and a `metadata.json` snapshot of environment overrides.
- `pixi run monitor` produces refreshed dashboards (`charts/*.png`) and a `run_overview.csv` summarising the latest run per approach. `pixi run monitor-all` compares every stored run for side-by-side analysis.
- `pixi run -e train plot` remains a quick way to generate charts for a specific run; pass `--output-dir <path>` to save them outside the run directory.

Reward Notes
- Reward-shaping decisions, open questions, and tuning history live in `REWARD_NOTES.md`. Update it whenever you adjust parameters in `trainer/tag_env.py`.

Debug Artifacts
- Use `bash scripts/collect_debug_artifacts.sh [dest]` (or `pixi run collect-debug`) to bundle logs manually; pass `--server-log` to add extra files.
- Collected bundles include `metadata.txt` with git revision, making bug triage reproducible.
- Trainer metrics record PPO diagnostics alongside per-role rewards, win outcomes, and episode duration. Use `pixi run monitor` to regenerate reward/win-rate charts when triaging a run.

Open-Source Only
- Uses only open-source tools; no proprietary dependencies.

Version Control
- A `.gitignore` is included. Initialize your repo with `git init` in the project root.
