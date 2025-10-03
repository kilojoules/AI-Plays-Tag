# Reward Notes

The core reward shaping lives in `godot/scripts/rl_env.gd`. Each episode awards:

- **Seeker shaping**
  - Distance closure scaled by `distance_reward_scale`.
  - Per-step time penalty `seeker_time_penalty` to encourage faster tags.
  - `jump_near_bonus` when the seeker jumps within 3m of the target for dynamic play.
  - `win_bonus` when a tag occurs (applied via `_on_tag_event`).
  - Negative `win_bonus` if the timer expires (seeker failure).

- **Hider shaping**
  - Distance expansion scaled by `distance_reward_scale` (sign inverted).
  - Per-step survival bonus `runner_survival_bonus`.
  - `high_ground_bonus` when the hider remains above 1.5m to promote vertical evasion.
  - Positive `win_bonus` when the time limit is reached without being tagged.

## Open Items

- Tune bonus magnitudes once seeker/hider policies are trained separately (see `trainer/TODO.md`).
- Consider extra shaping for maintaining line-of-sight or strategic use of obstacles once the arena layout stabilises.
- Logging: ensure `AI_LOG_TRAJECTORIES=1` is used during tuning so reward side-effects can be replayed via `Replay.tscn`.

## Change Log

- 2025-10-03: Documented current shaping parameters prior to policy split work. — auto-builder
