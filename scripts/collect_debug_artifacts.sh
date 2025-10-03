#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")"/.. && pwd)"
DEFAULT_TRAJECTORY_DIR="$HOME/Library/Application Support/Godot/app_userdata/AI Tag Game/trajectories"
DEFAULT_FRAMES_DIR="$HOME/Library/Application Support/Godot/app_userdata/AI Tag Game/frames"
LINUX_TRAJECTORY_DIR="$HOME/.local/share/godot/app_userdata/AI Tag Game/trajectories"
LINUX_FRAMES_DIR="$HOME/.local/share/godot/app_userdata/AI Tag Game/frames"

if [[ ! -d "$DEFAULT_TRAJECTORY_DIR" && -d "$LINUX_TRAJECTORY_DIR" ]]; then
  DEFAULT_TRAJECTORY_DIR="$LINUX_TRAJECTORY_DIR"
fi
if [[ ! -d "$DEFAULT_FRAMES_DIR" && -d "$LINUX_FRAMES_DIR" ]]; then
  DEFAULT_FRAMES_DIR="$LINUX_FRAMES_DIR"
fi

usage() {
  cat <<USAGE
Usage: ${0##*/} [destination] [--server-log path] [--godot-log path] [--trajectory path]

Without a destination argument the script creates debug/<timestamp>/ under the repo root.
It copies available artifacts (metrics, policies, charts, logs, trajectories, metadata).
USAGE
}

DEST=""
SERVER_LOG=""
GODOT_LOG=""
TRAJECTORY_PATH=""
FRAMES_DIR=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --server-log)
      SERVER_LOG="${2:-}"
      shift 2
      ;;
    --godot-log)
      GODOT_LOG="${2:-}"
      shift 2
      ;;
    --trajectory)
      TRAJECTORY_PATH="${2:-}"
      shift 2
      ;;
    --frames-dir)
      FRAMES_DIR="${2:-}"
      shift 2
      ;;
    *)
      if [[ -z "$DEST" ]]; then
        DEST="$1"
        shift
      else
        echo "Unknown argument: $1" >&2
        usage
        exit 1
      fi
      ;;
  esac
done

if [[ -z "$DEST" ]]; then
  DEST="$ROOT_DIR/debug/$(date +%Y%m%d_%H%M%S)"
fi

mkdir -p "$DEST"

copy_if_exists() {
  local src="$1"
  local name="${2:-}"
  if [[ -z "$name" ]]; then
    name="$(basename "$src")"
  fi
  if [[ -f "$src" ]]; then
    cp "$src" "$DEST/$name"
    echo "[collect] copied $(basename "$src")"
  fi
}

copy_dir_if_exists() {
  local src="$1"
  local name="${2:-}"
  if [[ -z "$name" ]]; then
    name="$(basename "$src")"
  fi
  if [[ -d "$src" ]]; then
    mkdir -p "$DEST/$name"
    cp -R "$src"/. "$DEST/$name/"
    echo "[collect] copied directory $(basename "$src")"
  fi
}

# Logs provided via flags
if [[ -n "$SERVER_LOG" ]]; then
  copy_if_exists "$SERVER_LOG" "server.log"
fi
if [[ -n "$GODOT_LOG" ]]; then
  copy_if_exists "$GODOT_LOG" "godot.log"
fi

# Trainer artifacts
copy_if_exists "$ROOT_DIR/trainer/logs/metrics.csv" "metrics.csv"
copy_if_exists "$ROOT_DIR/trainer/policy.pt" "policy.pt"
copy_if_exists "$ROOT_DIR/trainer/policy_seeker.pt" "policy_seeker.pt"
copy_if_exists "$ROOT_DIR/trainer/policy_hider.pt" "policy_hider.pt"
copy_dir_if_exists "$ROOT_DIR/trainer/charts" "charts"

# Historical seeker logs (if present)
copy_if_exists "$ROOT_DIR/trainer/seeker_server.log"
copy_if_exists "$ROOT_DIR/trainer/seeker_godot.log"

# Trajectories
if [[ -n "$TRAJECTORY_PATH" ]]; then
  copy_if_exists "$TRAJECTORY_PATH" "trajectory.jsonl"
else
  if [[ -d "$DEFAULT_TRAJECTORY_DIR" ]]; then
    latest="$(ls -t "$DEFAULT_TRAJECTORY_DIR"/*.jsonl 2>/dev/null | head -n 1 || true)"
    if [[ -n "$latest" ]]; then
      copy_if_exists "$latest" "$(basename "$latest")"
    fi
  fi
fi

# Frame dumps
if [[ -n "$FRAMES_DIR" ]]; then
  copy_dir_if_exists "$FRAMES_DIR" "frames"
else
  if [[ -d "$DEFAULT_FRAMES_DIR" ]]; then
    copy_dir_if_exists "$DEFAULT_FRAMES_DIR" "frames"
  fi
fi

# Metadata (git, environment)
GIT_STATUS="$(git -C "$ROOT_DIR" rev-parse HEAD 2>/dev/null || echo unknown)"
GIT_BRANCH="$(git -C "$ROOT_DIR" rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)"
cat <<META > "$DEST/metadata.txt"
created: $(date -Iseconds)
repo: $ROOT_DIR
git_head: $GIT_STATUS
git_branch: $GIT_BRANCH
hostname: $(hostname)
user: ${USER:-unknown}
META

echo "[collect] Artifacts stored in: $DEST"
