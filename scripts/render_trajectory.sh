#!/usr/bin/env bash
set -euo pipefail

# Render a trajectory JSONL to frames + MP4 using the GUI runtime.
# Usage: GODOT_BIN=/path/to/Godot AI_REPLAY_PATH=/path/to/file.jsonl ./scripts/render_trajectory.sh

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")"/.. && pwd)"

if [[ -z "${GODOT_BIN:-}" ]]; then
  echo "Set GODOT_BIN to your Godot 4 binary (e.g., /Users/julianquick/Downloads/Godot.app/Contents/MacOS/Godot)" >&2
  exit 1
fi

if [[ -z "${AI_REPLAY_PATH:-}" ]]; then
  echo "Set AI_REPLAY_PATH to a trajectory (user://trajectories/ep_XXXXX.jsonl)." >&2
  echo "Tip: after training, files are under: \"$HOME/Library/Application Support/Godot/app_userdata/AI Tag Game/trajectories\"" >&2
  exit 1
fi

echo "Launching Replay scene with recording enabled"
AI_RECORD=1 \
AI_RECORD_FPS=60 \
"$GODOT_BIN" --path "$ROOT_DIR/godot" "res://scenes/Replay.tscn"

echo "Encoding frames..."
bash "$ROOT_DIR/scripts/encode_frames.sh"

