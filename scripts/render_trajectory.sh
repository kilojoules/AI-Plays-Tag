#!/usr/bin/env bash
set -euo pipefail

# Render a trajectory JSONL to frames + MP4 using the GUI runtime.
# Usage: GODOT_BIN=/path/to/Godot AI_REPLAY_PATH=/path/to/file.jsonl ./scripts/render_trajectory.sh [low|high]
#        (defaults to high; pass "low" or set AI_RENDER_QUALITY=low for faster preview renders.)

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")"/.. && pwd)"

normalize_quality() {
  printf '%s' "$1" | tr '[:upper:]' '[:lower:]'
}

QUALITY="${AI_RENDER_QUALITY:-high}"
QUALITY="$(normalize_quality "$QUALITY")"

if [[ $# -gt 0 ]]; then
  case "$(normalize_quality "$1")" in
    low|high)
      QUALITY="$(normalize_quality "$1")"
      shift
      ;;
    --quality)
      if [[ $# -lt 2 ]]; then
        echo "Missing value for --quality" >&2
        exit 1
      fi
      QUALITY="$(normalize_quality "$2")"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
fi

if [[ "$QUALITY" != "low" && "$QUALITY" != "high" ]]; then
  echo "Quality must be 'low' or 'high' (got '$QUALITY')" >&2
  exit 1
fi

if [[ -z "${GODOT_BIN:-}" ]]; then
  echo "Set GODOT_BIN to your Godot 4 binary (e.g., /Users/julianquick/Downloads/Godot.app/Contents/MacOS/Godot)" >&2
  exit 1
fi

if [[ -z "${AI_REPLAY_PATH:-}" ]]; then
  echo "Set AI_REPLAY_PATH to a trajectory (user://trajectories/ep_XXXXX.jsonl)." >&2
  echo "Tip: after training, files are under: \"$HOME/Library/Application Support/Godot/app_userdata/AI Tag Game/trajectories\"" >&2
  exit 1
fi

echo "Launching Replay scene with recording enabled (quality: $QUALITY)"
AI_RENDER_QUALITY="$QUALITY" \
AI_RECORD=1 \
AI_RECORD_FPS=60 \
"$GODOT_BIN" --path "$ROOT_DIR/godot" "res://scenes/Replay.tscn"

echo "Encoding frames..."
bash "$ROOT_DIR/scripts/encode_frames.sh"
