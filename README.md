AI Tag Game (Godot + Python RL)

Overview
- Godot 4 project with a 3D tag arena, controllable agents, and core tag mechanics.
- Phase 2 introduces a Python WebSocket bridge and RL training stubs (PyTorch-ready).
- Entirely open-source stack: Godot, Python, PyTorch/TensorFlow optional, websockets, Blender/Krita/GIMP.

Structure
- `godot/`: Godot 4 project with scenes and scripts.
- `trainer/`: Python bridge and RL training scaffolding.

Quick Start (Phase 1: Manual Control)
- Open `godot/` in Godot 4.x.
- Run the main scene `scenes/Main.tscn`.
- Controls: WASD to move, Space to jump. Tab switches camera target.

Tag Rules
- One agent starts as "it". If the "it" agent’s tag area touches another agent that is not immune, the "it" status is transferred.
- Newly tagged agents gain short immunity to prevent immediate tag-back.

Phase 2 Bridge (Preview)
- Godot side can be wired to a WebSocket client (`scripts/rl_client.gd`).
- Python side launches a WebSocket server (`trainer/server.py`) with a simple JSON protocol: reset/step producing observation, reward, done, info.
- PPO/training scaffolding included (`trainer/ppo.py`) as a starting point.

End-to-End Training with Godot (Live)
- The server can learn from Godot by collecting transitions and updating a policy online (simple PPO-style update).
- In Godot, select the `RLEnv` node and set:
  - `training_mode = true`
  - `control_all_agents = true` (both agents are controlled by the policy; they learn together)
  - `ai_is_it = true` or `false` picks which agent starts as "it" on reset (roles still change through tagging).
  - Optional: adjust `step_tick_interval`, `tag_bonus`, `progress_reward_scale`.
- Start the training server with PyTorch:
  - `pixi run -e train server`
- Run the Godot scene `scenes/Main.tscn` and watch the learning progress. The server periodically saves `trainer/policy.pt`.

Python Environment (Pixi preferred)
- Why Pixi: pip on Python 3.13 lacks PyTorch wheels; Pixi pins a compatible Python and installs from conda-forge/pytorch channels.
- Install Pixi (see https://pixi.sh for platform-specific install), then from repo root:
  - `pixi run -e default server`  # runs `trainer/server.py` with Python 3.11, websockets, numpy
  - `pixi run -e train train`     # trains a toy PPO policy on a stub env and saves `trainer/policy.pt`
  - `pixi run -e train server`    # starts the server and loads `policy.pt` for inference

Pip (optional, server-only)
- If you only need the WebSocket server (no PyTorch), you can also do:
  - `cd trainer && python3 -m venv .venv && source .venv/bin/activate`
  - `pip install websockets numpy`  # skip torch for now
  - `python server.py`

Wire Godot to the Server
- The main scene now includes an `RLClient` (auto-connects to `ws://127.0.0.1:8765`) and an `RLEnv` that requests actions.
- Start the Python server first (`pixi run -e default server`), then run the Godot scene `scenes/Main.tscn`.
- With `control_all_agents=true`, both agents use the shared policy (self-play). With training enabled, the server updates the policy over time; otherwise it uses random or a saved `policy.pt`.

Recording an Animation (Open-Source)
- Add a `Node` to the scene and attach `scripts/recorder.gd`.
- Set `enabled=true` to dump frames into `user://frames` while the scene runs.
- Encode to video:
  - macOS/Linux: `ffmpeg -r 60 -i $HOME/Library/Application\ Support/Godot/app_userdata/AI\ Tag\ Game/frames/frame_%05d.png -c:v libx264 -pix_fmt yuv420p out.mp4`
  - Windows: adjust the `app_userdata` path under `%APPDATA%/Godot/app_userdata/AI Tag Game/frames`.

Minimal “Learning” Demo
- Option A (live in Godot): enable `RLEnv.training_mode`, run `pixi run -e train server`, then run the Godot scene. Policy updates online and is saved as `trainer/policy.pt`.
- Option B (offline stub): `pixi run -e train train` to produce a toy `policy.pt`, then `pixi run -e train server` and run Godot to see that policy.
- Record frames with `recorder.gd` or screen capture; stitch multiple runs to show progression.

Shell Script (one command training)
- Use `scripts/train.sh` to orchestrate server and Godot headless:
  - `bash scripts/train.sh live-chaser`  # self-play; first agent starts as "it"
  - `bash scripts/train.sh live-runner`  # self-play; second agent starts as "it"
  - `bash scripts/train.sh stub`         # offline toy PPO training (no Godot)
- The script requires Pixi and attempts to launch `godot4` or `godot` headless. If not found, it starts the server and asks you to open Godot manually.
- Environment overrides for headless tuning (optional):
  - `AI_PROGRESS_REWARD_SCALE`, `AI_TIME_PENALTY`, `AI_TAG_BONUS`, `AI_STEP_TICK_INTERVAL`

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
