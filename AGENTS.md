-----

# Guidance for LLM Coding Agents

The purpose of this repository is to create a reinforcement learning environment where agents learn to play "tag" in a 2D world.

This repository is collaboratively edited by multiple LLM agents. Use this guide to coordinate safely and predictably.

-----

## Read This First

  - **Source of truth for project goals and operation**: `PRD.md`. Read it thoroughly before making changes.
  - **Environment and Task Management**: Use **Pixi** for managing the Python environment and running tasks. Prefer `pixi run <task>` (defined in `pixi.toml`) over raw `python` or `pip` commands.
  - **Validation**: Before proposing or merging changes, ensure all validation checks pass.

-----

## Runtime Environment

This project uses a vectorized 2D Python environment (`trainer/tag_env.py`) for training RL agents via self-play. Training is run with `pixi run train` and visualization with `pixi run visualize`.

-----

## Debug Artifacts

When a training run or test fails, create a timestamped folder under a new `debug/` directory (e.g., `debug/YYYYMMDD_HHmmss/`). Save relevant artifacts to facilitate triage:

  - **`metrics.csv`**: The training metrics from `trainer/logs/metrics.csv`.
  - **`policy_seeker.pt`** / **`policy_hider.pt`**: The saved policy files.
  - **`trajectory.jsonl`**: If trajectory logging was enabled, save the relevant trajectory file from `data/trajectories`.

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

  - Ensure all validation checks pass.
  - Update the status of your task in `TODO.md` from `- [I]` to `- [x]`.

-----

## Run Logs (Required Every Execution)

  - Before starting work, create a new log file in a root `LLM_Logs/` directory.
  - **Naming**: `LLM_Logs/YYYYMMDD_HHmmss_<feature_slug>.log`
  - Include the following information in your log:
  - Do not delete any mp4 files or commit them to git without first confirming with the user
  - All file changes should be explained with git commits

<!-- end list -->

```
Prompt:
<paste the full prompt>

Task:
TODO.md: <short task title or line ref>

Actions: # examples
- Ran `git rebase origin/main`.
- Edited `trainer/tag_env.py` to modify reward function.
- Ran `pixi run train` to test changes.

Commits: # examples
- <hash> feat: Add distance shaping bonus to hider reward


```

-----
