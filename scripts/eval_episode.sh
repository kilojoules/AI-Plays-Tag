#!/usr/bin/env bash
set -euo pipefail

# Run a single evaluation episode using the trained seeker/hider policies,
# capture the resulting trajectory JSONL, and render a PNG of both paths.
#
# Usage:
#   bash scripts/eval_episode.sh [--duration 15] [--start-role seeker|hider] [--output charts/eval_paths.png]
#
# Environment variables:
#   EVAL_RUN_ID     Optional run identifier (defaults to timestamp).
#   GODOT_BIN       Explicit path to the Godot 4 binary.
#   KEEP_SERVER     Set to 1 to leave the server running after evaluation.
#   EVAL_USE_PIXI   Defaults to 1. Set to 0 to run server/plotting with system Python env.

PORT=8765
ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")"/.. && pwd)"
source "$ROOT_DIR/scripts/lib/data_paths.sh"
ai_ensure_data_dirs
RUN_ID="${EVAL_RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
DEBUG_DIR="$ROOT_DIR/debug/${RUN_ID}_eval"
SERVER_LOG_PATH="$DEBUG_DIR/server.log"
GODOT_LOG_PATH="$DEBUG_DIR/godot.log"
DEFAULT_OUTPUT="$ROOT_DIR/charts/eval_paths_${RUN_ID}.png"
START_ROLE="seeker"
DURATION=15
OUTPUT_PATH="$DEFAULT_OUTPUT"
TITLE="Trained Agents Paths (${RUN_ID})"
KEEP_SERVER_FLAG="${KEEP_SERVER:-0}"
USE_PIXI="${EVAL_USE_PIXI:-1}"
RECORD_FRAMES=0
RECORD_FPS=60
FINAL_VIDEO=""

usage() {
  cat <<'EOF'
Usage: eval_episode.sh [options]

Options:
  --duration <seconds>     Duration to run Godot headless (default: 15).
  --start-role <role>      Starting role for Agent1 (seeker|hider, default: seeker).
  --output <path>          Output PNG path (default: charts/eval_paths_<timestamp>.png).
  --title <title>          Plot title override.
  --record                 Enable frame capture for animation (default: off).
  --record-fps <fps>       Frame rate when recording (default: 60).
  -h, --help               Show this help message.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --duration)
      shift
      [[ $# -gt 0 ]] || { echo "Missing value for --duration" >&2; exit 1; }
      if ! [[ "$1" =~ ^[0-9]+$ ]]; then
        echo "Duration must be integer seconds (got '$1')." >&2
        exit 1
      fi
      DURATION="$1"
      ;;
    --start-role)
      shift
      [[ $# -gt 0 ]] || { echo "Missing value for --start-role" >&2; exit 1; }
      case "$1" in
        seeker|Seeker) START_ROLE="seeker" ;;
        hider|Hider) START_ROLE="hider" ;;
        *)
          echo "Unknown start role '$1'. Use seeker or hider." >&2
          exit 1
          ;;
      esac
      ;;
    --output)
      shift
      [[ $# -gt 0 ]] || { echo "Missing value for --output" >&2; exit 1; }
      OUTPUT_PATH="$1"
      ;;
    --title)
      shift
      [[ $# -gt 0 ]] || { echo "Missing value for --title" >&2; exit 1; }
      TITLE="$1"
      ;;
    --record)
      RECORD_FRAMES=1
      ;;
    --record-fps)
      shift
      [[ $# -gt 0 ]] || { echo "Missing value for --record-fps" >&2; exit 1; }
      if ! [[ "$1" =~ ^[0-9]+$ ]]; then
        echo "record-fps must be an integer (got '$1')." >&2
        exit 1
      fi
      RECORD_FPS="$1"
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
  shift
done

mkdir -p "$DEBUG_DIR"
mkdir -p "$(dirname "$OUTPUT_PATH")"
LEGACY_TRAJECTORY_DIRS=()
while IFS= read -r legacy_dir; do
  LEGACY_TRAJECTORY_DIRS+=("$legacy_dir")
done < <(ai_legacy_trajectory_dirs)
legacy_joined=""
for legacy_dir in "${LEGACY_TRAJECTORY_DIRS[@]}"; do
  if [[ -z "$legacy_joined" ]]; then
    legacy_joined="$legacy_dir"
  else
    legacy_joined="${legacy_joined}${AI_PATHSEP:-:}$legacy_dir"
  fi
done
export AI_LEGACY_TRAJECTORY_DIRS="$legacy_joined"

export AI_TRAJECTORIES_DIR
export AI_DATA_ROOT

need_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing required command: $1" >&2
    exit 1
  fi
}

if [[ "$USE_PIXI" == "1" ]]; then
  need_cmd pixi
else
  echo "[eval] EVAL_USE_PIXI=0: using system Python environment for server and plotting."
fi
need_cmd python3
need_cmd lsof

if [[ ! -f "$ROOT_DIR/trainer/policy_seeker.pt" || ! -f "$ROOT_DIR/trainer/policy_hider.pt" ]]; then
  echo "Expected trained policies at trainer/policy_seeker.pt and trainer/policy_hider.pt." >&2
  echo "Run training first or place the checkpoints before running evaluation." >&2
  exit 1
fi

latest_trajectory_meta() {
  python3 - <<'PY'
from pathlib import Path
import os
import sys

def iter_directories():
    primary = os.environ.get("AI_TRAJECTORIES_DIR")
    if primary:
        yield Path(primary)
    legacy_raw = os.environ.get("AI_LEGACY_TRAJECTORY_DIRS", "")
    if legacy_raw:
        for entry in legacy_raw.split(":"):
            entry = entry.strip()
            if entry:
                yield Path(entry)

for candidate in iter_directories():
    if not candidate.is_dir():
        continue
    files = sorted(candidate.glob("*.jsonl"), key=lambda p: p.stat().st_mtime)
    if files:
        latest = files[-1]
        stat = latest.stat()
        print(f"{latest}|{stat.st_mtime}|{stat.st_size}")
        sys.exit(0)
PY
}

latest_video_path() {
  python3 - "$ROOT_DIR" <<'PY'
import pathlib, sys
root = pathlib.Path(sys.argv[1])
candidates = sorted(root.glob("learn_progress-*.mp4"), key=lambda p: p.stat().st_mtime)
if candidates:
    print(candidates[-1])
PY
}

SERVER_STARTED=0
SERVER_PID=""
GODOT_PID=""

cleanup() {
  if [[ -n "$GODOT_PID" ]]; then
    kill "$GODOT_PID" 2>/dev/null || true
    wait "$GODOT_PID" 2>/dev/null || true
    GODOT_PID=""
  fi
  if [[ "$SERVER_STARTED" -eq 1 && -n "$SERVER_PID" ]]; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
    SERVER_PID=""
  fi
}
trap cleanup EXIT

start_server() {
  if lsof -i TCP:$PORT -sTCP:LISTEN >/dev/null 2>&1; then
    echo "[eval] Server already running on port $PORT; will reuse."
    return
  fi
  local -a CMD
  if [[ "$USE_PIXI" == "1" ]]; then
    echo "[eval] Starting training server via Pixi (logs: $SERVER_LOG_PATH)"
    CMD=(pixi run -e train server)
  else
    echo "[eval] Starting training server with system python (logs: $SERVER_LOG_PATH)"
    CMD=(python3 "$ROOT_DIR/trainer/server.py")
  fi
  (
    cd "$ROOT_DIR"
    exec "${CMD[@]}"
  ) >"$SERVER_LOG_PATH" 2>&1 &
  SERVER_PID=$!
  SERVER_STARTED=1
  echo "[eval] Server PID: $SERVER_PID"
}

wait_for_port() {
  echo "[eval] Waiting for ws://127.0.0.1:$PORT ..."
  python3 - "$PORT" <<'PY'
import socket, sys, time
port = int(sys.argv[1])
deadline = time.time() + 60
while time.time() < deadline:
    s = socket.socket()
    s.settimeout(1.0)
    try:
        s.connect(("127.0.0.1", port))
    except Exception:
        time.sleep(0.5)
    else:
        s.close()
        print("ready")
        sys.exit(0)
print("timeout")
sys.exit(1)
PY
}

detect_godot() {
  local godot_bin="${GODOT_BIN:-}"
  if [[ -n "$godot_bin" && -x "$godot_bin" ]]; then
    printf '%s\n' "$godot_bin"
    return
  fi
  if command -v godot4 >/dev/null 2>&1; then
    printf '%s\n' "$(command -v godot4)"
    return
  fi
  if command -v godot >/dev/null 2>&1; then
    printf '%s\n' "$(command -v godot)"
    return
  fi
  local mac_paths=(
    "/Applications/Godot.app/Contents/MacOS/Godot"
    "/Applications/Godot4.app/Contents/MacOS/Godot"
    "/Users/$(whoami)/Downloads/Godot.app/Contents/MacOS/Godot"
  )
  for candidate in "${mac_paths[@]}"; do
    if [[ -x "$candidate" ]]; then
      printf '%s\n' "$candidate"
      return
    fi
  done
  printf '%s\n' ""  # not found
}

run_godot() {
  local godot_bin
  godot_bin="$(detect_godot)"
  if [[ -z "$godot_bin" ]]; then
    cat <<'MSG'
[eval] Godot 4 binary not found.
[eval] Set GODOT_BIN to your executable, e.g.:
[eval]   export GODOT_BIN="/Applications/Godot.app/Contents/MacOS/Godot"
[eval] Evaluation aborted.
MSG
    exit 1
  fi
  echo "[eval] Launching Godot headless: $godot_bin"
  local ai_is_it=1
  if [[ "$START_ROLE" == "hider" ]]; then
    ai_is_it=0
  fi
  (
    cd "$ROOT_DIR"
    if [[ -n "${AI_TIME_LIMIT_SEC:-}" ]]; then
      export AI_TIME_LIMIT_SEC
    fi
    env \
      AI_DATA_ROOT="$AI_DATA_ROOT" \
      AI_TRAINING_MODE=1 \
      AI_CONTROL_ALL_AGENTS=1 \
      AI_IS_IT=$ai_is_it \
      AI_LOG_TRAJECTORIES=1 \
      AI_RECORD=$RECORD_FRAMES \
      AI_RECORD_FPS=$RECORD_FPS \
      AI_TRAIN_DURATION=$DURATION \
      "$godot_bin" --headless --path "$ROOT_DIR/godot"
  ) >"$GODOT_LOG_PATH" 2>&1 &
  GODOT_PID=$!
  echo "[eval] Godot PID: $GODOT_PID (duration ${DURATION}s)"
  sleep "$DURATION"
  kill "$GODOT_PID" 2>/dev/null || true
  wait "$GODOT_PID" 2>/dev/null || true
  GODOT_PID=""
}

echo "[eval] Debug artifacts directory: $DEBUG_DIR"

if [[ "$RECORD_FRAMES" -eq 1 && "${AI_PRESERVE_FRAMES:-0}" != "1" ]]; then
  if [[ -d "$AI_FRAMES_DIR" ]] && compgen -G "$AI_FRAMES_DIR/frame_*.png" >/dev/null; then
    echo "[eval] Clearing stale frames from $AI_FRAMES_DIR"
    rm -f "$AI_FRAMES_DIR"/frame_*.png
  fi
fi

LATEST_BEFORE_INFO="$(latest_trajectory_meta || true)"
LATEST_BEFORE_PATH=""
LATEST_BEFORE_MTIME=""
LATEST_BEFORE_SIZE=""
if [[ -n "$LATEST_BEFORE_INFO" ]]; then
  IFS='|' read -r LATEST_BEFORE_PATH LATEST_BEFORE_MTIME LATEST_BEFORE_SIZE <<<"$LATEST_BEFORE_INFO"
fi
start_server
wait_for_port
run_godot
sleep 1
LATEST_AFTER_INFO="$(latest_trajectory_meta || true)"
LATEST_AFTER_PATH=""
LATEST_AFTER_MTIME=""
LATEST_AFTER_SIZE=""
if [[ -n "$LATEST_AFTER_INFO" ]]; then
  IFS='|' read -r LATEST_AFTER_PATH LATEST_AFTER_MTIME LATEST_AFTER_SIZE <<<"$LATEST_AFTER_INFO"
fi

if [[ -z "$LATEST_AFTER_PATH" ]]; then
  echo "[eval] No trajectory file detected after run. Check Godot logs at $GODOT_LOG_PATH." >&2
  exit 1
fi

if [[ -n "$LATEST_BEFORE_PATH" && "$LATEST_AFTER_PATH" == "$LATEST_BEFORE_PATH" && "$LATEST_AFTER_MTIME" == "$LATEST_BEFORE_MTIME" && "$LATEST_AFTER_SIZE" == "$LATEST_BEFORE_SIZE" ]]; then
  echo "[eval] No new trajectory produced (latest did not change)." >&2
  exit 1
fi

echo "[eval] Latest trajectory: $LATEST_AFTER_PATH"

if [[ -f "$LATEST_AFTER_PATH" ]]; then
  cp "$LATEST_AFTER_PATH" "$DEBUG_DIR/trajectory.jsonl"
fi

if [[ "$RECORD_FRAMES" -eq 1 ]]; then
  if compgen -G "$AI_FRAMES_DIR/frame_*.png" >/dev/null; then
    echo "[eval] Encoding recorded frames into MP4"
    prev_mp4="$(latest_video_path || true)"
    AI_PRESERVE_FRAMES=0 bash "$ROOT_DIR/scripts/encode_frames.sh"
    new_mp4="$(latest_video_path || true)"
    if [[ -n "$new_mp4" && "$new_mp4" != "$prev_mp4" ]]; then
      cp "$new_mp4" "$DEBUG_DIR/$(basename "$new_mp4")"
      FINAL_VIDEO="$new_mp4"
    fi
  else
    echo "[eval] No frames captured; skipping encoding." >&2
  fi
fi

echo "[eval] Rendering path plot to $OUTPUT_PATH"
if [[ "$USE_PIXI" == "1" ]]; then
  pixi run -e train -- python scripts/plot_paths_from_trajectory.py \
    --trajectory "$LATEST_AFTER_PATH" \
    --output "$OUTPUT_PATH" \
    --title "$TITLE"
else
  python3 "$ROOT_DIR/scripts/plot_paths_from_trajectory.py" \
    --trajectory "$LATEST_AFTER_PATH" \
    --output "$OUTPUT_PATH" \
    --title "$TITLE"
fi

if [[ -f "$OUTPUT_PATH" ]]; then
  cp "$OUTPUT_PATH" "$DEBUG_DIR/$(basename "$OUTPUT_PATH")"
fi

echo "[eval] Plot created at $OUTPUT_PATH"
if [[ -n "$FINAL_VIDEO" ]]; then
  echo "[eval] Video saved: $FINAL_VIDEO"
fi
echo "[eval] Godot log:   $GODOT_LOG_PATH"
echo "[eval] Server log:  $SERVER_LOG_PATH"

if [[ "$KEEP_SERVER_FLAG" == "1" ]]; then
  echo "[eval] KEEP_SERVER=1: leaving server running (PID ${SERVER_PID:-<reused>})."
  SERVER_STARTED=0
fi
