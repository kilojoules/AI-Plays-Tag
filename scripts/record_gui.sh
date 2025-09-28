#!/usr/bin/env bash
set -euo pipefail

# Launch Godot with recording enabled (GUI, not headless).
# Usage: GODOT_BIN=/path/to/Godot ./scripts/record_gui.sh [chaser|runner]

ROLE=${1:-chaser}
ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")"/.. && pwd)"

if [[ -z "${GODOT_BIN:-}" ]]; then
  echo "Set GODOT_BIN to your Godot 4 binary path (e.g., /Users/julianquick/Downloads/Godot.app/Contents/MacOS/Godot)" >&2
  exit 1
fi

AI_IS_IT=1
if [[ "$ROLE" == "runner" ]]; then AI_IS_IT=0; fi

echo "Launching Godot GUI with recording on (role: $ROLE)"
AI_TRAINING_MODE=1 \
AI_IS_IT=$AI_IS_IT \
AI_CONTROL_ALL_AGENTS=1 \
AI_RECORD=1 \
AI_RECORD_FPS=60 \
"$GODOT_BIN" --path "$ROOT_DIR/godot"

echo "If frames were captured, encode with:"
echo "  bash scripts/encode_frames.sh"

