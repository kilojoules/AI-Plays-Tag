#!/usr/bin/env bash
set -euo pipefail

# Encode frames dumped by recorder.gd into an MP4 using ffmpeg.

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")"/.. && pwd)"
source "$ROOT_DIR/scripts/lib/data_paths.sh"
ai_ensure_data_dirs

FRAMES_DIR="$AI_FRAMES_DIR"
LEGACY_FRAMES_DIRS=()
if [[ -n "${AI_LEGACY_FRAMES_DIRS:-}" ]]; then
  IFS="$AI_PATHSEP" read -r -a LEGACY_FRAMES_DIRS <<< "$AI_LEGACY_FRAMES_DIRS"
fi
if [[ ! -d "$FRAMES_DIR" ]]; then
  for legacy in "${LEGACY_FRAMES_DIRS[@]}"; do
    if [[ -d "$legacy" ]]; then
      echo "[encode] Falling back to legacy frames directory: $legacy"
      FRAMES_DIR="$legacy"
      break
    fi
  done
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
if [[ "${AI_PRESERVE_FRAMES:-0}" != "1" ]]; then
  if compgen -G "$FRAMES_DIR/frame_*.png" >/dev/null; then
    echo "[encode] Cleaning up raw frames in $FRAMES_DIR"
    rm -f "$FRAMES_DIR"/frame_*.png
  fi
else
  echo "[encode] Preserving raw frames (AI_PRESERVE_FRAMES=$AI_PRESERVE_FRAMES)"
fi
