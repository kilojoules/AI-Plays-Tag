# Reward Notes

The core reward shaping lives in `trainer/tag_env.py`. Each episode awards:

- **Seeker shaping**
  - Distance closure scaled by `distance_reward_scale` (default `0.14`).
  - Per-step time penalty `seeker_time_penalty` (default `-0.005`) to keep pressure on chasing without overwhelming the agent.
  - `win_bonus` when a tag occurs.
  - Timeout applies `timeout_seeker_penalty` (default `6.0`) so the loss is softer than a missed tag.

- **Hider shaping**
  - Distance expansion scaled by `distance_reward_scale` (sign inverted).
  - Per-step survival bonus `runner_survival_bonus` (default `0.01`).
  - Timeouts award `timeout_hider_bonus` (default `6.0`) so the hider still profits from running out the clock but with less disparity versus a tag event.

## Open Items

- Tune bonus magnitudes once seeker/hider policies are trained separately (see `trainer/TODO.md`).
- Consider extra shaping for maintaining line-of-sight or strategic use of obstacles once the arena layout stabilises.

## Change Log

- 2025-10-03: Documented current shaping parameters prior to policy split work. — auto-builder
