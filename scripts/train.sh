#!/usr/bin/env bash
set -euo pipefail

# Train the agents using the Pixi environment.
# Modes:
#   live-seeker (default): Run server (PyTorch) and Godot headless with AI as seeker
#   live-hider:            Run server (PyTorch) and Godot headless with AI as hider
#   stub:                  Offline toy PPO training (no Godot), saves trainer/policy.pt

MODE=${1:-live-seeker}
ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")"/.. && pwd)"
PORT=8765

abort() { echo "[train.sh] $*" >&2; exit 1; }

need_cmd() { command -v "$1" >/dev/null 2>&1 || abort "Missing required command: $1"; }

start_server_bg() {
  echo "[train.sh] Starting training server (Pixi env: train) ..."
  # If port already in use, skip starting a duplicate server
  if lsof -i TCP:$PORT -sTCP:LISTEN >/dev/null 2>&1; then
    echo "[train.sh] Port $PORT already in use; assuming server is running."
    return 0
  fi
  (
    cd "$ROOT_DIR"
    exec pixi run -e train server
  ) &
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
  local DURATION="${AI_TRAIN_DURATION:-120}"
  (
    cd "$ROOT_DIR"
    AI_TRAINING_MODE=1 \
    AI_IS_IT=$AI_IS_IT \
    AI_CONTROL_ALL_AGENTS=1 \
    AI_MAX_STEPS_PER_EPISODE=${AI_MAX_STEPS_PER_EPISODE:-300} \
    AI_CONTROL_ALL_AGENTS=1 \
    AI_RECORD=1 \
    AI_RECORD_FPS=30 \
    AI_LOG_TRAJECTORIES=1 \
    "$godot_bin" --headless --path "$ROOT_DIR/godot" &
    GODOt_PID=$!
    echo "[train.sh] Godot PID: $GODOt_PID (will run for ${DURATION}s)"
    sleep "$DURATION"
    kill "$GODOt_PID" 2>/dev/null || true
    wait "$GODOt_PID" 2>/dev/null || true
  )
}

make_video_from_frames() {
  local frames_dir
  if [[ "$(uname)" == "Darwin" ]]; then
    frames_dir="$HOME/Library/Application Support/Godot/app_userdata/AI Tag Game/frames"
  else
    frames_dir="$HOME/.local/share/godot/app_userdata/AI Tag Game/frames"
  fi
  if [[ ! -d "$frames_dir" ]]; then
    echo "[train.sh] No frames found at: $frames_dir"
    return 0
  fi
  local FFMPEG_BIN="ffmpeg"
  if ! command -v ffmpeg >/dev/null 2>&1; then
    if command -v pixi >/dev/null 2>&1; then
      FFMPEG_BIN="pixi run -e train ffmpeg"
      echo "[train.sh] Using ffmpeg from Pixi environment"
    else
      echo "[train.sh] ffmpeg not found; skipping video encoding. Frames are in: $frames_dir"
      return 0
    fi
  fi
  local out="${ROOT_DIR}/learn_progress-$(date +%Y%m%d-%H%M%S).mp4"
  echo "[train.sh] Encoding video to $out"
  eval $FFMPEG_BIN -y -r 30 -i "$frames_dir/frame_%05d.png" -c:v libx264 -pix_fmt yuv420p "$out" >/dev/null 2>&1 || true
  echo "[train.sh] Video saved: $out"
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
}
trap cleanup EXIT

case "$MODE" in
  live-seeker)
    need_cmd pixi
    echo "[train.sh] Pre-warming Pixi environment (train) ..."
    ( cd "$ROOT_DIR" && pixi run -e train python -c "import sys" )
    start_server_bg
    wait_for_port
    run_godot_headless "seeker"
    make_video_from_frames
    ;;
  live-hider)
    need_cmd pixi
    echo "[train.sh] Pre-warming Pixi environment (train) ..."
    ( cd "$ROOT_DIR" && pixi run -e train python -c "import sys" )
    start_server_bg
    wait_for_port
    run_godot_headless "hider"
    make_video_from_frames
    ;;
  stub)
    need_cmd pixi
    echo "[train.sh] Running offline stub training (no Godot) ..."
    (
      cd "$ROOT_DIR"
      exec pixi run -e train train
    )
    ;;
  *)
    echo "Usage: $0 [live-seeker|live-hider|stub]"; exit 1;
    ;;
esac
