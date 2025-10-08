AI Tag Game (Godot + Python RL)

Overview
- Godot 4 project with a 3D tag arena, controllable agents, and core tag mechanics.
- Phase 2 introduces a Python WebSocket bridge and RL training stubs (PyTorch-ready).
- Entirely open-source stack: Godot, Python, PyTorch/TensorFlow optional, websockets, Blender/Krita/GIMP.

Structure
- `godot/`: Godot 4 project with scenes and scripts.
- `trainer/`: Python bridge and RL training scaffolding.
- `data/`: Workspace-local runtime artifacts (trajectories, frame dumps, imported legacy data). Override with `AI_DATA_ROOT` if embedding the project elsewhere.

Quick Start (Phase 1: Manual Control)
- Open `godot/` in Godot 4.x.
- Run the main scene `scenes/Main.tscn`.
- Controls: WASD to move, Space to jump. Tab switches camera target.

Tag Rules
- One agent starts as "it". If the "it" agent’s tag area touches another agent that is not immune, the "it" status is transferred.
- Newly tagged agents gain short immunity to prevent immediate tag-back.

Phase 2 Bridge (Preview)
- Godot side can be wired to a WebSocket client (`scripts/rl_client.gd`).
- Python side launches a WebSocket server (`trainer/server.py`) that serves `act`/`act_batch` requests and consumes `transition_batch` payloads streamed from Godot.
- PPO/training scaffolding included (`trainer/ppo.py`) as a starting point.

End-to-End Training with Godot (Live)
- The server can learn from Godot by collecting transitions and updating a policy online (simple PPO-style update).
- In Godot, select the `RLEnv` node and set:
  - `training_mode = true`
  - `control_all_agents = true` (both agents are controlled by the policy; they learn together)
  - `ai_is_it = true` or `false` picks which agent starts as "it" on reset (roles still change through tagging).
  - Optional: adjust `step_tick_interval`, `tag_bonus`, `progress_reward_scale`.
  - Optional: `legacy_act_fallback` sends one-agent `act` requests for older Python bridges (default disabled).
- Start the training server with PyTorch:
  - `pixi run -e train server`
- Run the Godot scene `scenes/Main.tscn` and watch the learning progress. The server periodically saves `trainer/policy.pt`.

Python Environment (Pixi preferred)
- Why Pixi: pip on Python 3.13 lacks PyTorch wheels; Pixi pins a compatible Python and installs from conda-forge/pytorch channels.
- Install Pixi (see https://pixi.sh for platform-specific install), then from repo root:
  - `pixi run -e default server`  # runs `trainer/server.py` with Python 3.11, websockets, numpy
  - `pixi run -e train server`    # starts the server and loads `policy.pt` for inference
  - `pixi run -e train plot`      # regenerate charts from `trainer/logs/metrics.csv`
  - `pixi run tests`              # headless Godot test suite (uses `scripts/run_godot_tests.sh`)
  - `pixi run collect-debug`      # gather logs/metrics into `debug/<timestamp>/`
  - `pixi run eval-episode`       # run a single headless evaluation, log the trajectory, and render a path PNG

Pip (optional, server-only)
- If you only need the WebSocket server (no PyTorch), you can also do:
  - `cd trainer && python3 -m venv .venv && source .venv/bin/activate`
  - `pip install websockets numpy`  # skip torch for now
  - `python server.py`

Wire Godot to the Server
- The main scene now includes an `RLClient` (auto-connects to `ws://127.0.0.1:8765`) and an `RLEnv` that requests actions.
- Start the Python server first (`pixi run -e default server`), then run the Godot scene `scenes/Main.tscn`.
- With `control_all_agents=true`, both agents use the shared policy (self-play). With training enabled, the server updates the policy over time; otherwise it uses random or a saved `policy.pt`.

Runtime Data Directory
- All training/eval artifacts now land inside `data/` by default:
  - `data/trajectories/ep_*.jsonl` — headless rollouts and evaluation traces.
  - `data/frames/frame_*.png` — GUI recordings captured by `recorder.gd`.
  - `data/_imported/...` — optional backups copied from legacy `app_userdata` locations.
- Shell helpers (`scripts/lib/data_paths.sh`) expose the shared paths so scripts and Godot stay in sync.
- Override `AI_DATA_ROOT=/custom/path` when you need an alternate workspace.
- Run `pixi run migrate-user-data` once to copy existing `app_userdata` trajectories/frames into `data/`.

Recording an Animation (Open-Source)
- Add a `Node` to the scene and attach `scripts/recorder.gd`.
- Set `enabled=true` to dump frames into `data/frames` while the scene runs.
- Encode to video:
  - `bash scripts/encode_frames.sh` uses the workspace frames directory and falls back to any legacy `app_userdata` cache if present.

Minimal “Learning” Demo
- Enable `RLEnv.training_mode`, run `pixi run -e train server`, then run the Godot scene. Policy updates online and is saved as `trainer/policy.pt`.
- Record frames with `recorder.gd` or screen capture; stitch multiple runs to show progression.
- Role-specific checkpoints are saved as `trainer/policy_seeker.pt` and `trainer/policy_hider.pt`; the debug collector copies both for post-run analysis.

Reward Notes
- Reward-shaping decisions, open questions, and tuning history live in `REWARD_NOTES.md`. Update it whenever you adjust parameters in `godot/scripts/rl_env.gd`.

Additional Docs
- `docs/trajectory_rendering.md` walks through replay generation with `scripts/render_trajectory.sh` and `Replay.tscn`.
- `docs/ws_protocol.md` captures the JSON contract between Godot and the Python trainer.

Debug Artifacts
- `bash scripts/train.sh live-seeker` and `live-hider` now stream Godot/server logs to `debug/<timestamp>/` dirs and automatically snapshot metrics, policies, and encodes.
- Use `bash scripts/collect_debug_artifacts.sh [dest]` (or `pixi run collect-debug`) to bundle logs manually; pass `--server-log/--godot-log` to add extra files.
- Collected bundles include `metadata.txt` with git revision, making bug triage reproducible.
- Trainer metrics now track advantage statistics and PPO losses; rerun `pixi run -e train plot` to render the new charts (`advantage_mean.png`, `policy_loss.png`, etc.).
- Self-play runs also capture `self_play_rounds.csv` and per-round `godot_round*.log` files inside the debug directory for postmortem analysis.

Shell Script (one command training)
- Use `scripts/train.sh` to orchestrate server and Godot headless:
- `bash scripts/train.sh live-seeker`  # shared policy; first agent starts as the seeker
- `bash scripts/train.sh live-hider`   # shared policy; second agent starts as the seeker
- `bash scripts/train.sh self-play`    # runs alternating seeker/hider rounds in a single session (self-play pipeline)
- The script requires Pixi and attempts to launch `godot4` or `godot` headless. If not found, it keeps the server alive and asks you to open Godot manually.
- Environment overrides for headless tuning (optional):
  - `AI_DISTANCE_REWARD_SCALE`, `AI_SEEKER_TIME_PENALTY`, `AI_WIN_BONUS`, `AI_STEP_TICK_INTERVAL`
  - `AI_TRAIN_DURATION` (seconds to run the headless client before shutting down)
  - `AI_LOG_TRAJECTORIES=1` to dump JSONL rollouts into `data/trajectories` for later rendering with `render_trajectory.sh`
  - Self-play extras: `SELF_PLAY_ROUNDS` (even count of alternating rounds), `SELF_PLAY_DURATION` (seconds per round), `SELF_PLAY_START_ROLE` (`seeker` or `hider`)

Record With GUI
- If you want to capture frames, run the GUI build (headless cannot capture):
  - `GODOT_BIN="/Users/julianquick/Downloads/Godot.app/Contents/MacOS/Godot" bash scripts/record_gui.sh chaser`
  - Close Godot when you’ve captured enough frames. Then encode:
    - `bash scripts/encode_frames.sh`

About the WebSocket Error
- Seeing `opening handshake failed ... invalid Connection header: keep-alive` means something tried to open the port with plain HTTP (e.g., a browser). It’s normal.
- Run the Godot scene (which uses a proper WebSocket client) to establish a valid connection.

Open-Source Only
- Uses only open-source tools; no proprietary dependencies.

Version Control
- A `.gitignore` is included. Initialize your repo with `git init` in the project root.
