# Trainer TODO

- [x] Replace the placeholder L2 "PPO" update with a clipped-surrogate PPO loss so learning matches PRD expectations. *(trainer/ppo.py + trainer/server.py now implement clipped PPO with entropy/KL metrics.)*
- [x] Split seeker/hider policies (`policy_seeker.pt`, `policy_hider.pt`) and update load/save paths in `server.py`. *(server saves/loads per-role policies; debug bundle copies new files.)*
- [x] Expand metrics logging (reward mean/sum, episode length) and ensure plotting scripts stay in sync. *(trainer/server.py now tracks advantage stats & losses; `trainer/plot_metrics.py` renders new charts.)*
- [x] Harden reset, timeout, and tag-event handling when consuming Godot transitions (avoid stale buffers). *(server caches per-agent actions, clears on episode end, and requeues batches when updates roll back; covered by `pixi run tests`).*
- [x] Implement checkpoints/rollbacks for policy updates so failed runs can be debugged safely. *(policies snapshot before PPO updates, roll back on NaN/KL spikes, and persist rotating checkpoints & legacy copies).*

- [x] Develop self-play training pipeline (Codex completed 2025-10-07).
- [x] Teach `trainer/metrics_from_trajectories.py` (and related helpers) to look inside the workspace data directory by default while still allowing custom overrides.
- [x] Add regression coverage to ensure training/eval scripts can find trajectories after the storage move (Pixi task or unit test).
- [x] Develop a comprehensive training monitoring suite for multi-approach hider/seeker experiments (metrics aggregation, dashboards, comparisons).
- [ ] Add automated coverage (unit + CLI smoke) for the monitoring suite and wire summary regressions into CI.
