#!/usr/bin/env bash
set -euo pipefail

# Demo trained agents in Godot with full 3D visualization.
# This script runs the trained policies through the WebSocket bridge
# and launches Godot to visualize the agent behavior.
#
# Usage:
#   bash scripts/demo_trained.sh [num_episodes] [--record]
#
# Options:
#   num_episodes  Number of episodes to run (default: 5)
#   --record      Record frames and encode to video

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")"/.. && pwd)"
source "$ROOT_DIR/scripts/lib/data_paths.sh"
ai_ensure_data_dirs

NUM_EPISODES=${1:-5}
RECORD_MODE=0

for arg in "$@"; do
  if [[ "$arg" == "--record" ]]; then
    RECORD_MODE=1
  fi
done

DEMO_ID="demo_$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="$ROOT_DIR/trainer/demos/$DEMO_ID"
mkdir -p "$OUTPUT_DIR"

echo "=== Tag Agent Demo ==="
echo "Episodes: $NUM_EPISODES"
echo "Output: $OUTPUT_DIR"
echo ""

# Check for trained policies
SEEKER_POLICY="$ROOT_DIR/trainer/policy_seeker.pt"
HIDER_POLICY="$ROOT_DIR/trainer/policy_hider.pt"

if [[ ! -f "$SEEKER_POLICY" ]]; then
  echo "Warning: Seeker policy not found at $SEEKER_POLICY"
  echo "Run 'pixi run train' first to train the agents."
fi

if [[ ! -f "$HIDER_POLICY" ]]; then
  echo "Warning: Hider policy not found at $HIDER_POLICY"
  echo "Run 'pixi run train' first to train the agents."
fi

# Find Godot binary
godot_bin="${GODOT_BIN:-}"
if [[ -z "$godot_bin" ]] && command -v godot4 >/dev/null 2>&1; then godot_bin="godot4"; fi
if [[ -z "$godot_bin" ]] && command -v godot >/dev/null 2>&1; then godot_bin="godot"; fi
if [[ -z "$godot_bin" ]] && [[ -x "/Applications/Godot.app/Contents/MacOS/Godot" ]]; then
  godot_bin="/Applications/Godot.app/Contents/MacOS/Godot"
fi

if [[ -z "$godot_bin" ]]; then
  echo "Error: Godot 4 not found. Set GODOT_BIN environment variable."
  echo ""
  echo "Alternative: Use the Python-only visualizer:"
  echo "  pixi run visualize"
  exit 1
fi

PORT=8765
SERVER_LOG="$OUTPUT_DIR/server.log"
GODOT_LOG="$OUTPUT_DIR/godot.log"

# Start the training server (which serves the trained policies)
echo "Starting policy server..."
if lsof -i TCP:$PORT -sTCP:LISTEN >/dev/null 2>&1; then
  echo "Port $PORT already in use; assuming server is running."
else
  (
    cd "$ROOT_DIR"
    pixi run -e train server
  ) >"$SERVER_LOG" 2>&1 &
  SERVER_PID=$!
  echo "$SERVER_PID" > "$ROOT_DIR/.server.pid"
  echo "Server PID: $SERVER_PID"

  # Wait for server
  echo "Waiting for server..."
  python3 - "$PORT" <<'PY'
import socket, sys, time
port=int(sys.argv[1])
deadline=time.time()+30
while time.time()<deadline:
    s=socket.socket(); s.settimeout(1.0)
    try:
        s.connect(("127.0.0.1", port)); s.close(); print("ready"); sys.exit(0)
    except Exception:
        time.sleep(0.5)
print("timeout")
sys.exit(1)
PY
fi

# Configure recording if requested
RECORD_ARGS=""
if [[ "$RECORD_MODE" == "1" ]]; then
  echo "Recording enabled"
  RECORD_ARGS="AI_RECORD=1 AI_RECORD_FPS=30"

  # Clear old frames
  if [[ -d "$AI_FRAMES_DIR" ]]; then
    rm -f "$AI_FRAMES_DIR"/frame_*.png 2>/dev/null || true
  fi
fi

# Calculate demo duration (each episode ~10s max, add buffer)
DEMO_DURATION=$((NUM_EPISODES * 12))

echo ""
echo "Launching Godot demo (${DEMO_DURATION}s)..."
echo "  Press Ctrl+C to stop early"
echo ""

# Run Godot in demo mode
(
  cd "$ROOT_DIR"
  env \
    AI_DATA_ROOT="$AI_DATA_ROOT" \
    AI_TRAINING_MODE=1 \
    AI_CONTROL_ALL_AGENTS=1 \
    AI_LOG_TRAJECTORIES=1 \
    AI_MAX_STEPS_PER_EPISODE=300 \
    $RECORD_ARGS \
    timeout "$DEMO_DURATION" "$godot_bin" --path "$ROOT_DIR/godot" 2>&1 || true
) | tee "$GODOT_LOG" &
GODOT_PID=$!

# Monitor and wait
wait "$GODOT_PID" 2>/dev/null || true

# Stop server
if [[ -f "$ROOT_DIR/.server.pid" ]]; then
  pid="$(cat "$ROOT_DIR/.server.pid" || true)"
  if [[ -n "$pid" ]] && ps -p "$pid" >/dev/null 2>&1; then
    echo "Stopping server..."
    kill "$pid" 2>/dev/null || true
  fi
  rm -f "$ROOT_DIR/.server.pid"
fi

# Encode video if recording was enabled
if [[ "$RECORD_MODE" == "1" ]]; then
  if compgen -G "$AI_FRAMES_DIR/frame_*.png" >/dev/null 2>&1; then
    echo ""
    echo "Encoding video..."
    VIDEO_PATH="$OUTPUT_DIR/demo.mp4"
    ffmpeg -y -r 30 -i "$AI_FRAMES_DIR/frame_%05d.png" \
      -c:v libx264 -pix_fmt yuv420p "$VIDEO_PATH" 2>/dev/null || true

    if [[ -f "$VIDEO_PATH" ]]; then
      echo "Video saved: $VIDEO_PATH"
    fi

    # Clean up frames
    rm -f "$AI_FRAMES_DIR"/frame_*.png
  fi
fi

# Copy trajectory files
if compgen -G "$AI_TRAJECTORIES_DIR/ep_*.jsonl" >/dev/null 2>&1; then
  mkdir -p "$OUTPUT_DIR/trajectories"
  cp "$AI_TRAJECTORIES_DIR"/ep_*.jsonl "$OUTPUT_DIR/trajectories/" 2>/dev/null || true
  echo "Trajectories saved to: $OUTPUT_DIR/trajectories/"
fi

echo ""
echo "Demo complete! Output saved to: $OUTPUT_DIR"
echo ""
echo "To visualize trajectories:"
echo "  pixi run plot-paths $OUTPUT_DIR/trajectories/ep_00001.jsonl"
