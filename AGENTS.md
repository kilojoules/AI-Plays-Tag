-----

# Guidance for LLM Coding Agents

The purpose of this repository is to create a reinforcement learning environment where agents learn to play "tag" in a 3D world.

This repository is collaboratively edited by multiple LLM agents. Use this guide to coordinate safely and predictably.

-----

## Read This First

  - **Source of truth for project goals and operation**: `PRD.md`. Read it thoroughly before making changes.
  - **Environment and Task Management**: Use **Pixi** for managing the Python environment and running tasks. Prefer `pixi run <task>` (defined in `pixi.toml`) over raw `python` or `pip` commands.
  - **Validation**: Before proposing or merging changes, ensure all validation checks pass. The primary check is running the Godot tests: `bash scripts/run_godot_tests.sh`.

-----

## Runtime Environment

This project has two main components that run concurrently:

1.  **The Godot Engine**: Manages the 3D world, physics, and agent rendering (`godot/` directory). It acts as the RL environment.
2.  **The Python Server**: A WebSocket server (`trainer/server.py`) that handles RL logic, including policy inference and training updates using PyTorch.

These two components communicate over WebSockets. The Godot `RLEnv` node sends observations and receives actions from the Python server. **Do not** introduce changes that break this core interaction loop.

-----

## Debug Artifacts

When a training run or test fails, create a timestamped folder under a new `debug/` directory (e.g., `debug/YYYYMMDD_HHmmss/`). Save relevant artifacts to facilitate triage:

  - **`server.log`**: The console output from the Python server (`pixi run -e train server`).
  - **`godot.log`**: The console output from the Godot engine (especially when run headless).
  - **`metrics.csv`**: The training metrics from `trainer/logs/metrics.csv`.
  - **`policy.pt`**: The saved policy file from `trainer/policy.pt` if one was generated.
  - **`trajectory.jsonl`**: If `AI_LOG_TRAJECTORIES` was enabled, save the relevant trajectory file from `data/trajectories` (run `pixi run migrate-user-data` once if older runs live under Godot's `app_userdata`).
  - **`screenshot.png`**: If the failure is visual, a screenshot of the Godot window.

-----

## Picking And Tracking Tasks

  - A central task list should be maintained in `TODO.md` at the project root.
  - Choose one task at a time from `TODO.md`.
  - When you begin a task, mark it in-place with `- [I] <task description>`.
  - If you cannot finish a task, leave the checkbox as `- [I]`, add a comment detailing the blocker, keep your feature alive, and ask for guidance.


-----
## Testing Strategy
   - develop a robust set of tests designed to run quickly and test all features specified in PRD.md

-----

## Merging And Completion

  - Ensure your all validation checks pass.
  - Update the status of your task in `TODO.md` from `- [I]` to `- [x]`.

-----

## Run Logs (Required Every Execution)

  - Before starting work, create a new log file in a root `LLM_Logs/` directory.
  - **Naming**: `LLM_Logs/YYYYMMDD_HHmmss_<feature_slug>.log`
  - Include the following information in your log:
  - Do not delete any mp4 files or commit them to git without first confirming with the user

<!-- end list -->

```
Prompt:
<paste the full prompt>

Task:
TODO.md: <short task title or line ref>

Actions: # examples
- Ran `git rebase origin/main`.
- Edited `godot/scripts/rl_env.gd` to modify reward function.
- Ran `bash scripts/train.sh live-seeker` to test changes.
- Ran validation: `bash scripts/run_godot_tests.sh`.

Commits: # examples
- <hash> feat: Add high-ground bonus to hider reward


```

-----

