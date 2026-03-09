"""
Named reward presets for the reward shaping study.

Each preset maps to TagEnvConfig overrides. All use four_corners layout
with hider_speed_mult=1.15 (consistent with zoo experiments).
"""

# Shared game config across all presets
GAME_CONFIG = {
    "layout": "four_corners",
    "hider_speed_mult": 1.15,
}

PRESETS = {
    # R0: Minimal baseline — only terminal rewards + small time penalty
    "R0_baseline": {
        "seeker_time_penalty": -0.005,
        "distance_reward_scale": 0.0,
        "hider_dist_reward_scale": 0.0,
        "hider_abs_dist_reward_scale": 0.0,
        "runner_survival_bonus": 0.01,
    },

    # R1: Seeker-focused — strong pursuit signal (distance shaping + steeper time penalty)
    "R1_seeker_pursuit": {
        "seeker_time_penalty": -0.02,
        "distance_reward_scale": 0.2,
        "hider_dist_reward_scale": 0.0,
        "hider_abs_dist_reward_scale": 0.0,
        "runner_survival_bonus": 0.01,
    },

    # R2: Hider-focused — active evasion (distance + wall penalty + speed bonus)
    "R2_hider_active": {
        "seeker_time_penalty": -0.005,
        "distance_reward_scale": 0.14,
        "hider_dist_reward_scale": 0.14,
        "hider_abs_dist_reward_scale": 0.1,
        "hider_wall_prox_penalty": -0.02,
        "hider_min_speed_reward": 0.005,
        "runner_survival_bonus": 0.01,
    },

    # R3: Both shaped — seeker pursuit + hider active evasion
    "R3_both_shaped": {
        "seeker_time_penalty": -0.02,
        "distance_reward_scale": 0.2,
        "hider_dist_reward_scale": 0.14,
        "hider_abs_dist_reward_scale": 0.1,
        "hider_wall_prox_penalty": -0.02,
        "hider_min_speed_reward": 0.005,
        "runner_survival_bonus": 0.01,
    },

    # R4: Sparse terminal — only win/loss/timeout, no shaping at all
    "R4_sparse": {
        "seeker_time_penalty": 0.0,
        "distance_reward_scale": 0.0,
        "hider_dist_reward_scale": 0.0,
        "hider_abs_dist_reward_scale": 0.0,
        "runner_survival_bonus": 0.0,
    },

    # R5: Escalating urgency — seeker time pressure doubles over episode
    "R5_escalating": {
        "seeker_time_penalty": -0.01,
        "seeker_escalating_urgency": True,
        "distance_reward_scale": 0.14,
        "hider_dist_reward_scale": 0.0,
        "hider_abs_dist_reward_scale": 0.1,
        "runner_survival_bonus": 0.01,
    },

    # R6: Coverage exploration — both agents rewarded for exploring the arena
    "R6_coverage": {
        "seeker_time_penalty": -0.005,
        "distance_reward_scale": 0.14,
        "hider_dist_reward_scale": 0.0,
        "hider_abs_dist_reward_scale": 0.05,
        "area_coverage_bonus": 0.1,
        "runner_survival_bonus": 0.01,
    },

    # R7: Kitchen sink — all shaping terms combined
    "R7_kitchen_sink": {
        "seeker_time_penalty": -0.015,
        "seeker_escalating_urgency": True,
        "distance_reward_scale": 0.2,
        "hider_dist_reward_scale": 0.14,
        "hider_abs_dist_reward_scale": 0.1,
        "hider_wall_prox_penalty": -0.02,
        "hider_min_speed_reward": 0.005,
        "area_coverage_bonus": 0.05,
        "runner_survival_bonus": 0.01,
    },
}


def get_preset_cli_args(preset_name: str) -> list:
    """Convert a preset to CLI argument list for train_selfplay.py or train_selfplay_sac.py."""
    if preset_name not in PRESETS:
        raise ValueError(f"Unknown preset: {preset_name}. Available: {list(PRESETS.keys())}")

    cfg = PRESETS[preset_name]
    args = []

    # Game config
    args += ["--layout", GAME_CONFIG["layout"]]
    args += ["--hider-speed-mult", str(GAME_CONFIG["hider_speed_mult"])]

    # Reward params
    mapping = {
        "seeker_time_penalty": "--seeker-time-penalty",
        "distance_reward_scale": "--distance-reward-scale",
        "hider_dist_reward_scale": "--hider-dist-reward",
        "hider_abs_dist_reward_scale": "--hider-abs-dist-reward",
        "hider_wall_prox_penalty": "--hider-wall-prox-penalty",
        "hider_min_speed_reward": "--hider-min-speed-reward",
        "area_coverage_bonus": "--area-coverage-bonus",
        "runner_survival_bonus": "--runner-survival-bonus",
        "seeker_escalating_urgency": "--seeker-escalating-urgency",
    }

    for key, cli_flag in mapping.items():
        if key in cfg and cli_flag is not None:
            val = cfg[key]
            if isinstance(val, bool):
                if val:
                    args.append(cli_flag)
            else:
                args += [cli_flag, str(val)]

    return args


def get_preset_env_overrides(preset_name: str) -> dict:
    """Get TagEnvConfig field overrides for a preset (for direct Python use)."""
    if preset_name not in PRESETS:
        raise ValueError(f"Unknown preset: {preset_name}. Available: {list(PRESETS.keys())}")

    overrides = dict(GAME_CONFIG)
    overrides.update(PRESETS[preset_name])
    return overrides
