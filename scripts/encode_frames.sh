#!/usr/bin/env bash
set -euo pipefail

# Encode frames dumped by recorder.gd into an MP4 using ffmpeg.

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")"/.. && pwd)"

if [[ "$(uname)" == "Darwin" ]]; then
  FRAMES_DIR="$HOME/Library/Application Support/Godot/app_userdata/AI Tag Game/frames"
else
  FRAMES_DIR="$HOME/.local/share/godot/app_userdata/AI Tag Game/frames"
fi

if [[ ! -d "$FRAMES_DIR" ]]; then
  echo "Frames directory not found: $FRAMES_DIR" >&2
  exit 1
fi

USE_PIXI=0
if command -v ffmpeg >/dev/null 2>&1; then
  FFMPEG_BIN="ffmpeg"
else
  if command -v pixi >/dev/null 2>&1; then
    USE_PIXI=1
  else
    echo "ffmpeg not found; please install it (e.g., brew install ffmpeg)" >&2
    exit 1
  fi
fi

OUT="$ROOT_DIR/learn_progress-$(date +%Y%m%d-%H%M%S).mp4"
echo "Encoding frames from $FRAMES_DIR to $OUT"
if [[ "$USE_PIXI" == "1" ]]; then
  pixi run -e train ffmpeg -y -r 60 -i "$FRAMES_DIR/frame_%05d.png" -c:v libx264 -pix_fmt yuv420p "$OUT"
else
  ffmpeg -y -r 60 -i "$FRAMES_DIR/frame_%05d.png" -c:v libx264 -pix_fmt yuv420p "$OUT"
fi
echo "Saved: $OUT"
