# Reward Notes

The core reward shaping lives in `godot/scripts/rl_env.gd`. Each episode awards:

- **Seeker shaping**
  - Distance closure scaled by `distance_reward_scale` (default `0.14`).
  - Per-step time penalty `seeker_time_penalty` (default `-0.005`) to keep pressure on chasing without overwhelming the agent.
  - `jump_near_bonus` when the seeker jumps within 3m of the target for dynamic play.
  - `win_bonus` when a tag occurs (applied via `_on_tag_event`).
  - Timeout now applies `timeout_seeker_penalty` (default `6.0`) instead of reusing the full `win_bonus` so the loss is softer than a missed tag.

- **Hider shaping**
  - Distance expansion scaled by `distance_reward_scale` (sign inverted).
  - Per-step survival bonus `runner_survival_bonus` (default `0.01`).
  - `high_ground_bonus` when the hider remains above 1.5m to promote vertical evasion.
  - Timeouts award `timeout_hider_bonus` (default `6.0`) so the hider still profits from running out the clock but with less disparity versus a tag event.

## Open Items

- Tune bonus magnitudes once seeker/hider policies are trained separately (see `trainer/TODO.md`).
- Consider extra shaping for maintaining line-of-sight or strategic use of obstacles once the arena layout stabilises.
- Logging: ensure `AI_LOG_TRAJECTORIES=1` is used during tuning so reward side-effects can be replayed via `Replay.tscn`.

## Change Log

- 2025-10-03: Documented current shaping parameters prior to policy split work. — auto-builder
