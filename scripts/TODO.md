# Scripts TODO

- [x] Extend `run_godot_tests.sh` and headless test scenes to cover physics and observation sanity checks (`tests/run_tests.gd` now validates obs length + ray masks).
- [x] Add a helper to collect debug artifacts under `debug/YYYYMMDD_HHmmss/` after failures (server and Godot logs). Implemented `scripts/collect_debug_artifacts.sh` with bundling support.
- [x] Provide a Pixi task (e.g., `pixi run plot`) that regenerates charts from `trainer/logs/metrics.csv`. Added `plot`, `tests`, `collect-debug` tasks to `pixi.toml`.
- [x] Teach `train.sh` to capture server stdout/stderr to `debug/` when runs are interrupted (logs now streamed to per-run directories with automatic bundling).
