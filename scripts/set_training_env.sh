#!/usr/bin/env bash
# Usage:
#   source scripts/set_training_env.sh [seeker|hider]
#
# Exports environment variables used by the Godot training scene.
# Must be sourced to persist in your current shell.

_ROLE=${1:-seeker}  # seeker (AI starts as it) or hider

export AI_TRAINING_MODE=1
if [[ "$_ROLE" == "hider" ]]; then
  export AI_IS_IT=0
else
  export AI_IS_IT=1
fi

# Tunables (override as needed before launching Godot)
export AI_DISTANCE_REWARD_SCALE=${AI_DISTANCE_REWARD_SCALE:-0.1}
export AI_SEEKER_TIME_PENALTY=${AI_SEEKER_TIME_PENALTY:- -0.001}
export AI_WIN_BONUS=${AI_WIN_BONUS:-5.0}
export AI_STEP_TICK_INTERVAL=${AI_STEP_TICK_INTERVAL:-3}

echo "[env] AI_TRAINING_MODE=$AI_TRAINING_MODE"
echo "[env] AI_IS_IT=$AI_IS_IT ($_ROLE)"
echo "[env] AI_DISTANCE_REWARD_SCALE=$AI_DISTANCE_REWARD_SCALE"
echo "[env] AI_SEEKER_TIME_PENALTY=$AI_SEEKER_TIME_PENALTY"
echo "[env] AI_WIN_BONUS=$AI_WIN_BONUS"
echo "[env] AI_STEP_TICK_INTERVAL=$AI_STEP_TICK_INTERVAL"
echo
echo "Environment set. To launch headless Godot (example macOS):"
echo "  GODOT_BIN=\"/Applications/Godot.app/Contents/MacOS/Godot\" \\\" \\
      bash scripts/train.sh live-seeker"
