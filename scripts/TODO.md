# Scripts TODO

- [x] Extend `run_godot_tests.sh` and headless test scenes to cover physics and observation sanity checks (`tests/run_tests.gd` now validates obs length + ray masks).
- [x] Add a helper to collect debug artifacts under `debug/YYYYMMDD_HHmmss/` after failures (server and Godot logs). Implemented `scripts/collect_debug_artifacts.sh` with bundling support.
- [x] Provide a Pixi task (e.g., `pixi run plot`) that regenerates charts from `trainer/logs/metrics.csv`. Added `plot`, `tests`, `collect-debug` tasks to `pixi.toml`.
- [x] Teach `train.sh` to capture server stdout/stderr to `debug/` when runs are interrupted (logs now streamed to per-run directories with automatic bundling).
- [I] Generate a PNG map of seeker/hider paths for a single evaluation episode using trained policies.
- [x] Establish a repo-local `data/` root (or similar) and expose a shared shell helper so scripts can resolve trajectories, frames, and debug artifacts without touching `$HOME`-scoped Godot paths.
- [x] Refactor `train.sh`, `eval_episode.sh`, `collect_debug_artifacts.sh`, `render_trajectory.sh`, and plotting utilities to honor the new data root while keeping backwards compatibility or offering a one-time migration.
- [x] Add a maintenance script (and Pixi task) that migrates legacy `app_userdata` artifacts into the workspace directory and warns if stale data remains outside the repo.
- [x] Fix GitHub CI test failures (`pixi run tests` / `scripts/run_godot_tests.sh` regressions).
