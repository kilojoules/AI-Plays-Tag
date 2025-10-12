#!/usr/bin/env bash
set -euo pipefail

# Train the agents using the Pixi environment.
# Modes:
#   live-seeker (default): Run server (PyTorch) and Godot headless with AI as seeker
#   live-hider:            Run server (PyTorch) and Godot headless with AI as hider

MODE=${1:-live-seeker}
ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")"/.. && pwd)"
source "$ROOT_DIR/scripts/lib/data_paths.sh"
ai_ensure_data_dirs
TRAIN_USE_PIXI="${TRAIN_USE_PIXI:-1}"
TRAIN_ENV_DIR="${TRAIN_ENV_DIR:-$ROOT_DIR/.pixi/envs/train}"
TRAIN_BIN_DIR="${TRAIN_BIN_DIR:-$TRAIN_ENV_DIR/bin}"
TRAIN_PYTHON="${TRAIN_PYTHON:-$TRAIN_BIN_DIR/python}"
if [[ "$TRAIN_USE_PIXI" != "0" ]]; then
  echo "[train.sh] Using Pixi to execute train environment tasks."
else
  echo "[train.sh] Pixi execution disabled (TRAIN_USE_PIXI=0); using binaries under $TRAIN_BIN_DIR"
fi
PORT=8765
RUN_ID="${TRAIN_RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
DEBUG_DIR="$ROOT_DIR/debug/$RUN_ID"
SERVER_LOG_PATH="$DEBUG_DIR/server.log"
GODOT_LOG_PATH="$DEBUG_DIR/godot.log"
COLLECTED_DEBUG=0

if [[ "${TRAIN_CLEAN_FRAMES:-1}" == "1" && "${AI_PRESERVE_FRAMES:-0}" != "1" ]]; then
  if [[ -d "$AI_FRAMES_DIR" ]] && compgen -G "$AI_FRAMES_DIR/frame_*.png" >/dev/null; then
    echo "[train.sh] Clearing stale frames from $AI_FRAMES_DIR"
    rm -f "$AI_FRAMES_DIR"/frame_*.png
  fi
fi

mkdir -p "$DEBUG_DIR"
echo "[train.sh] Debug artifacts directory: $DEBUG_DIR"
echo "$DEBUG_DIR" > "$ROOT_DIR/.server.debugdir"

abort() { echo "[train.sh] $*" >&2; exit 1; }

need_cmd() { command -v "$1" >/dev/null 2>&1 || abort "Missing required command: $1"; }

start_server_bg() {
  echo "[train.sh] Starting training server (Pixi env: train) ..."
  local approach="${TRAIN_APPROACH:-$MODE}"
  export TRAIN_RUN_ID="$RUN_ID"
  if [[ -z "${TRAIN_APPROACH:-}" ]]; then
    export TRAIN_APPROACH="$approach"
  fi
  # If port already in use, skip starting a duplicate server
  if lsof -i TCP:$PORT -sTCP:LISTEN >/dev/null 2>&1; then
    echo "[train.sh] Port $PORT already in use; assuming server is running."
    return 0
  fi
  echo "[train.sh] Streaming server output to $SERVER_LOG_PATH"
  echo "[train.sh] Tip: run 'tail -f $SERVER_LOG_PATH' for live logs"
  (
    cd "$ROOT_DIR"
    if [[ "$TRAIN_USE_PIXI" != "0" ]]; then
      exec env \
        TRAIN_RUN_ID="$RUN_ID" \
        TRAIN_APPROACH="$approach" \
        TRAIN_VARIANT="$MODE" \
        pixi run -e train server
    else
      if [[ ! -x "$TRAIN_PYTHON" ]]; then
        abort "TRAIN_PYTHON not found at $TRAIN_PYTHON (set TRAIN_USE_PIXI=1 or point TRAIN_PYTHON to Pixi env python)."
      fi
      PATH="$TRAIN_BIN_DIR:$PATH" exec env \
        TRAIN_RUN_ID="$RUN_ID" \
        TRAIN_APPROACH="$approach" \
        TRAIN_VARIANT="$MODE" \
        "$TRAIN_PYTHON" trainer/server.py
    fi
  ) >"$SERVER_LOG_PATH" 2>&1 &
  SERVER_PID=$!
  echo "$SERVER_PID" > "$ROOT_DIR/.server.pid"
  echo "[train.sh] Server PID: $SERVER_PID"
}

wait_for_port() {
  echo "[train.sh] Waiting for ws://127.0.0.1:$PORT ..."
  python3 - "$PORT" <<'PY'
import socket, sys, time
port=int(sys.argv[1])
deadline=time.time()+600  # allow long time on first pixi environment build
while time.time()<deadline:
    s=socket.socket(); s.settimeout(1.0)
    try:
        s.connect(("127.0.0.1", port)); s.close(); print("ready"); sys.exit(0)
    except Exception:
        time.sleep(0.5)
print("timeout")
sys.exit(1)
PY
}

run_godot_headless() {
  local role=$1 # seeker|hider
  local log_path=${2:-$GODOT_LOG_PATH}
  local godot_bin="${GODOT_BIN:-}"
  # Try PATH names
  if [[ -z "$godot_bin" ]] && command -v godot4 >/dev/null 2>&1; then godot_bin="godot4"; fi
  if [[ -z "$godot_bin" ]] && command -v godot >/dev/null 2>&1; then godot_bin="godot"; fi
  # Try common macOS app bundle locations
  if [[ -z "$godot_bin" ]] && [[ -x "/Applications/Godot.app/Contents/MacOS/Godot" ]]; then godot_bin="/Applications/Godot.app/Contents/MacOS/Godot"; fi
  if [[ -z "$godot_bin" ]] && [[ -x "/Applications/Godot4.app/Contents/MacOS/Godot" ]]; then godot_bin="/Applications/Godot4.app/Contents/MacOS/Godot"; fi
  # Try user-provided Downloads path (macOS)
  if [[ -z "$godot_bin" ]] && [[ -x "/Users/julianquick/Downloads/Godot.app/Contents/MacOS/Godot" ]]; then godot_bin="/Users/julianquick/Downloads/Godot.app/Contents/MacOS/Godot"; fi
  # If still not found, keep server running and prompt user
  if [[ -z "$godot_bin" ]]; then
    cat <<MSG
[train.sh] Godot 4 binary not found in PATH.
[train.sh] Start the Editor manually and run the project:
[train.sh]   Project path: $ROOT_DIR/godot
[train.sh] Or set GODOT_BIN to your binary path, e.g.:
[train.sh]   export GODOT_BIN="/Applications/Godot.app/Contents/MacOS/Godot"
[train.sh] Press ENTER when done training to stop the server.
MSG
    read -r _
    return 0
  fi
  echo "[train.sh] Launching Godot headless ($godot_bin) with AI role: $role"
  local AI_IS_IT=1
  if [[ "$role" == "hider" ]]; then AI_IS_IT=0; fi
  local DURATION="${AI_TRAIN_DURATION:-0}"
  echo "[train.sh] Streaming Godot output to $log_path"
  (
    cd "$ROOT_DIR"
    if [[ -n "${AI_TIME_LIMIT_SEC:-}" ]]; then
      export AI_TIME_LIMIT_SEC
    fi
    env \
      AI_DATA_ROOT="$AI_DATA_ROOT" \
      AI_TRAINING_MODE=1 \
      AI_IS_IT=$AI_IS_IT \
      AI_CONTROL_ALL_AGENTS=1 \
      AI_MAX_STEPS_PER_EPISODE=${AI_MAX_STEPS_PER_EPISODE:-300} \
      AI_STEP_TICK_INTERVAL=${AI_STEP_TICK_INTERVAL:-1} \
      AI_RECORD=0 \
      AI_RECORD_FPS=0 \
      AI_LOG_TRAJECTORIES=${AI_LOG_TRAJECTORIES:-0} \
      "$godot_bin" --headless --path "$ROOT_DIR/godot"
  ) >"$log_path" 2>&1 &
  GODO_PID=$!
  if [[ "$DURATION" == "0" ]]; then
    echo "[train.sh] Godot PID: $GODO_PID (press Ctrl+C when you want to stop)"
    wait "$GODO_PID"
  else
    echo "[train.sh] Godot PID: $GODO_PID (will run for ${DURATION}s)"
    sleep "$DURATION"
    kill "$GODO_PID" 2>/dev/null || true
    wait "$GODO_PID" 2>/dev/null || true
  fi
}

run_self_play() {
  local rounds_raw="${SELF_PLAY_ROUNDS:-4}"
  local rounds="$rounds_raw"
  if ! [[ "$rounds" =~ ^[0-9]+$ ]] || [[ "$rounds" -eq 0 ]]; then
    echo "[train.sh] Invalid SELF_PLAY_ROUNDS='$rounds_raw'; defaulting to 4 rounds (2 per role)."
    rounds=4
  fi
  if (( rounds % 2 == 1 )); then
    echo "[train.sh] SELF_PLAY_ROUNDS=$rounds is odd; adding one extra round to complete a role pair."
    rounds=$((rounds + 1))
  fi

  local duration_raw="${SELF_PLAY_DURATION:-}"
  local duration="$duration_raw"
  if [[ -z "$duration" ]]; then
    duration="${AI_TRAIN_DURATION:-0}"
  fi
  if ! [[ "$duration" =~ ^[0-9]+$ ]] || [[ "$duration" -le 0 ]]; then
    duration=180
    echo "[train.sh] SELF_PLAY_DURATION not set; defaulting to ${duration}s per round. Set SELF_PLAY_DURATION to override."
  fi

  local start_role="${SELF_PLAY_START_ROLE:-seeker}"
  if [[ "$start_role" != "hider" ]]; then
    start_role="seeker"
  fi

  local summary_path="$DEBUG_DIR/self_play_rounds.csv"
  if [[ ! -f "$summary_path" ]]; then
    echo "round_index,role,start_ts,end_ts,log_path" >"$summary_path"
  fi

  local original_duration="${AI_TRAIN_DURATION:-}"
  local -a round_logs=()
  local role="$start_role"

  echo "[train.sh] Starting self-play pipeline: ${rounds} rounds, ${duration}s per round (starting role: $role)."

  local round_index
  for ((round_index = 1; round_index <= rounds; round_index++)); do
    local round_log="$DEBUG_DIR/godot_round$(printf '%02d' "$round_index").log"
    round_logs+=("$round_log")
    local start_ts
    start_ts="$(date -Iseconds)"
    echo "[train.sh] --- Self-play round $round_index/$rounds (role=$role, ${duration}s) ---"
    AI_TRAIN_DURATION="$duration"
    run_godot_headless "$role" "$round_log"
    local end_ts
    end_ts="$(date -Iseconds)"
    echo "$round_index,$role,$start_ts,$end_ts,$round_log" >>"$summary_path"
    if [[ "$role" == "seeker" ]]; then
      role="hider"
    else
      role="seeker"
    fi
  done

  if [[ -n "$original_duration" ]]; then
    AI_TRAIN_DURATION="$original_duration"
  else
    unset AI_TRAIN_DURATION
  fi

  : >"$GODOT_LOG_PATH"
  local idx=1
  for log_file in "${round_logs[@]}"; do
    if [[ -f "$log_file" ]]; then
      printf '===== Self-play round %02d (%s) =====\n' "$idx" "$(basename "$log_file")" >>"$GODOT_LOG_PATH"
      cat "$log_file" >>"$GODOT_LOG_PATH"
      printf '\n' >>"$GODOT_LOG_PATH"
    fi
    ((idx++))
  done
}

make_video_from_frames() {
  local frames_dir="$AI_FRAMES_DIR"
  local legacy_dirs=()
  if [[ -n "${AI_LEGACY_FRAMES_DIRS:-}" ]]; then
    IFS="$AI_PATHSEP" read -r -a legacy_dirs <<< "$AI_LEGACY_FRAMES_DIRS"
  fi
  if [[ ! -d "$frames_dir" ]]; then
    for legacy in "${legacy_dirs[@]}"; do
      if [[ -d "$legacy" ]]; then
        echo "[train.sh] Falling back to legacy frames directory: $legacy"
        frames_dir="$legacy"
        break
      fi
    done
  fi
  if [[ ! -d "$frames_dir" ]]; then
    echo "[train.sh] No frames found. Checked workspace and legacy directories."
    return 0
  fi
  local FFMPEG_BIN="ffmpeg"
  if ! command -v ffmpeg >/dev/null 2>&1; then
    if [[ "$TRAIN_USE_PIXI" != "0" ]] && command -v pixi >/dev/null 2>&1; then
      FFMPEG_BIN="pixi run -e train ffmpeg"
      echo "[train.sh] Using ffmpeg from Pixi environment"
    elif [[ -x "$TRAIN_BIN_DIR/ffmpeg" ]]; then
      FFMPEG_BIN="$TRAIN_BIN_DIR/ffmpeg"
      echo "[train.sh] Using ffmpeg from TRAIN_BIN_DIR ($TRAIN_BIN_DIR)"
    else
      echo "[train.sh] ffmpeg not found; skipping video encoding. Frames are in: $frames_dir"
      return 0
    fi
  fi
  local out="${ROOT_DIR}/learn_progress-$(date +%Y%m%d-%H%M%S).mp4"
  echo "[train.sh] Encoding video to $out"
  eval $FFMPEG_BIN -y -r 30 -i "$frames_dir/frame_%05d.png" -c:v libx264 -pix_fmt yuv420p "$out" >/dev/null 2>&1 || true
  echo "[train.sh] Video saved: $out"
  if [[ -f "$out" ]]; then
    cp "$out" "$DEBUG_DIR/$(basename "$out")"
  fi
  if [[ "${AI_PRESERVE_FRAMES:-0}" != "1" ]]; then
    if compgen -G "$frames_dir/frame_*.png" >/dev/null; then
      echo "[train.sh] Cleaning up raw frames in $frames_dir"
      rm -f "$frames_dir"/frame_*.png
    fi
  else
    echo "[train.sh] Preserving raw frames (AI_PRESERVE_FRAMES=$AI_PRESERVE_FRAMES)"
  fi
}

cleanup() {
  if [[ -f "$ROOT_DIR/.server.pid" ]]; then
    local pid; pid="$(cat "$ROOT_DIR/.server.pid" || true)"
    if [[ -n "${pid}" ]] && ps -p "$pid" >/dev/null 2>&1; then
      echo "[train.sh] Stopping server (PID $pid)"
      kill "$pid" 2>/dev/null || true
    fi
    rm -f "$ROOT_DIR/.server.pid"
  fi
  if [[ $COLLECTED_DEBUG -eq 0 ]]; then
    bash "$ROOT_DIR/scripts/collect_debug_artifacts.sh" "$DEBUG_DIR" --server-log "$SERVER_LOG_PATH" --godot-log "$GODOT_LOG_PATH" || true
  fi
  rm -f "$ROOT_DIR/.server.debugdir"
  echo "[train.sh] Debug bundle: $DEBUG_DIR"
}
trap cleanup EXIT

case "$MODE" in
  live-seeker)
    if [[ "$TRAIN_USE_PIXI" != "0" ]]; then
      need_cmd pixi
      echo "[train.sh] Pre-warming Pixi environment (train) ..."
      ( cd "$ROOT_DIR" && pixi run -e train python -c "import sys" )
    else
      if [[ ! -x "$TRAIN_PYTHON" ]]; then
        abort "TRAIN_PYTHON not found at $TRAIN_PYTHON (set TRAIN_USE_PIXI=1 or point TRAIN_PYTHON to Pixi env python)."
      fi
      echo "[train.sh] Using Pixi env binaries directly from $TRAIN_BIN_DIR"
      ( cd "$ROOT_DIR" && PATH="$TRAIN_BIN_DIR:$PATH" "$TRAIN_PYTHON" -c "import sys" )
    fi
    start_server_bg
    wait_for_port
    run_godot_headless "seeker"
    make_video_from_frames
    bash "$ROOT_DIR/scripts/collect_debug_artifacts.sh" "$DEBUG_DIR" --server-log "$SERVER_LOG_PATH" --godot-log "$GODOT_LOG_PATH"
    COLLECTED_DEBUG=1
    ;;
  live-hider)
    if [[ "$TRAIN_USE_PIXI" != "0" ]]; then
      need_cmd pixi
      echo "[train.sh] Pre-warming Pixi environment (train) ..."
      ( cd "$ROOT_DIR" && pixi run -e train python -c "import sys" )
    else
      if [[ ! -x "$TRAIN_PYTHON" ]]; then
        abort "TRAIN_PYTHON not found at $TRAIN_PYTHON (set TRAIN_USE_PIXI=1 or point TRAIN_PYTHON to Pixi env python)."
      fi
      echo "[train.sh] Using Pixi env binaries directly from $TRAIN_BIN_DIR"
      ( cd "$ROOT_DIR" && PATH="$TRAIN_BIN_DIR:$PATH" "$TRAIN_PYTHON" -c "import sys" )
    fi
    start_server_bg
    wait_for_port
    run_godot_headless "hider"
    make_video_from_frames
    bash "$ROOT_DIR/scripts/collect_debug_artifacts.sh" "$DEBUG_DIR" --server-log "$SERVER_LOG_PATH" --godot-log "$GODOT_LOG_PATH"
    COLLECTED_DEBUG=1
    ;;
  self-play)
    if [[ "$TRAIN_USE_PIXI" != "0" ]]; then
      need_cmd pixi
      echo "[train.sh] Pre-warming Pixi environment (train) ..."
      ( cd "$ROOT_DIR" && pixi run -e train python -c "import sys" )
    else
      if [[ ! -x "$TRAIN_PYTHON" ]]; then
        abort "TRAIN_PYTHON not found at $TRAIN_PYTHON (set TRAIN_USE_PIXI=1 or point TRAIN_PYTHON to Pixi env python)."
      fi
      echo "[train.sh] Using Pixi env binaries directly from $TRAIN_BIN_DIR"
      ( cd "$ROOT_DIR" && PATH="$TRAIN_BIN_DIR:$PATH" "$TRAIN_PYTHON" -c "import sys" )
    fi
    start_server_bg
    wait_for_port
    run_self_play
    make_video_from_frames
    bash "$ROOT_DIR/scripts/collect_debug_artifacts.sh" "$DEBUG_DIR" --server-log "$SERVER_LOG_PATH" --godot-log "$GODOT_LOG_PATH"
    COLLECTED_DEBUG=1
    ;;
  *)
    echo "Usage: $0 [live-seeker|live-hider|self-play]"; exit 1;
    ;;
esac
