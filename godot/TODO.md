# Godot TODO

- [x] Harden arena collision and gravity; add regression coverage in `tests/` for wall, floor, and fall-through cases. (`pixi run tests` now green: `run_godot_tests.sh` covers floor + wall constraints.)
- [x] Confirm observation pipeline matches PRD vision requirements (`scripts/rl_env.gd`, `scripts/eyes.gd`) — verified normalized kinematics, opponent diff, forward vector, 36-ray vision with agent mask; server now consumes batched obs only.
- [I] Verify camera framing and distance scaling for showcase recordings (`scenes/Main.tscn`, `scripts/camera_rig.gd`) — pending capture pass to validate near/far agent scale.
- [x] Streamline WebSocket requests in `RLEnv` once server-side compatibility fallback (`act` + `act_batch`) is no longer needed (legacy path now behind `legacy_act_fallback`).
- [ ] Audit NPC fallback behavior for non-controlled agents so seeker/hider roles stay faithful to PRD timing rules.
