# Scripts TODO

- [x] Add a helper to collect debug artifacts under `debug/YYYYMMDD_HHmmss/` after failures. Implemented `scripts/collect_debug_artifacts.sh` with bundling support.
- [x] Provide a Pixi task (e.g., `pixi run plot`) that regenerates charts from `trainer/logs/metrics.csv`. Added `plot`, `collect-debug` tasks to `pixi.toml`.
- [I] Generate a PNG map of seeker/hider paths for a single evaluation episode using trained policies.

- [x] Establish a repo-local `data/` root and expose a shared shell helper so scripts can resolve trajectories, frames, and debug artifacts.
- [x] Refactor `collect_debug_artifacts.sh` and plotting utilities to honor the new data root.
