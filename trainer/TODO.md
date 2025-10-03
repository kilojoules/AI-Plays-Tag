# Trainer TODO

- [x] Replace the placeholder L2 "PPO" update with a clipped-surrogate PPO loss so learning matches PRD expectations. *(trainer/ppo.py + trainer/server.py now implement clipped PPO with entropy/KL metrics.)*
- [x] Split seeker/hider policies (`policy_seeker.pt`, `policy_hider.pt`) and update load/save paths in `server.py`. *(server saves/loads per-role policies; debug bundle copies new files.)*
- [x] Expand metrics logging (reward mean/sum, episode length) and ensure plotting scripts stay in sync. *(trainer/server.py now tracks advantage stats & losses; `trainer/plot_metrics.py` renders new charts.)*
- [ ] Harden reset, timeout, and tag-event handling when consuming Godot transitions (avoid stale buffers). *(needs integration test once Godot NPC audit completes)*
- [ ] Implement checkpoints/rollbacks for policy updates so failed runs can be debugged safely. *(design pending debug directory conventions)*
