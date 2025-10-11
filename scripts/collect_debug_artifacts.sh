#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")"/.. && pwd)"
source "$ROOT_DIR/scripts/lib/data_paths.sh"
ai_ensure_data_dirs

DEFAULT_TRAJECTORY_DIR="$AI_TRAJECTORIES_DIR"
DEFAULT_FRAMES_DIR="$AI_FRAMES_DIR"

LEGACY_TRAJECTORY_DIRS=()
while IFS= read -r _legacy_traj; do
  [[ -n "$_legacy_traj" ]] && LEGACY_TRAJECTORY_DIRS+=("$_legacy_traj")
done < <(ai_legacy_trajectory_dirs)

LEGACY_FRAMES_DIRS=()
while IFS= read -r _legacy_frame; do
  [[ -n "$_legacy_frame" ]] && LEGACY_FRAMES_DIRS+=("$_legacy_frame")
done < <(ai_legacy_frames_dirs)

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
  godot_dir="$(dirname "$GODOT_LOG")"
  shopt -s nullglob
  for extra_log in "$godot_dir"/godot_round*.log; do
    if [[ -f "$extra_log" ]]; then
      copy_if_exists "$extra_log" "$(basename "$extra_log")"
    fi
  done
  shopt -u nullglob
  if [[ -f "$godot_dir/self_play_rounds.csv" ]]; then
    copy_if_exists "$godot_dir/self_play_rounds.csv" "self_play_rounds.csv"
  fi
fi

# Trainer artifacts
copy_if_exists "$ROOT_DIR/trainer/logs/metrics.csv" "metrics.csv"
copy_if_exists "$ROOT_DIR/trainer/policy.pt" "policy.pt"
copy_if_exists "$ROOT_DIR/trainer/policy_seeker.pt" "policy_seeker.pt"
copy_if_exists "$ROOT_DIR/trainer/policy_hider.pt" "policy_hider.pt"
copy_dir_if_exists "$ROOT_DIR/trainer/charts" "charts"
copy_dir_if_exists "$ROOT_DIR/trainer/checkpoints" "checkpoints"

# Historical seeker logs (if present)
copy_if_exists "$ROOT_DIR/trainer/seeker_server.log"
copy_if_exists "$ROOT_DIR/trainer/seeker_godot.log"

# Trajectories
if [[ -n "$TRAJECTORY_PATH" ]]; then
  copy_if_exists "$TRAJECTORY_PATH" "trajectory.jsonl"
else
  latest=""
  searched=()
  if [[ -d "$DEFAULT_TRAJECTORY_DIR" ]]; then
    searched+=("$DEFAULT_TRAJECTORY_DIR")
    latest="$(ls -t "$DEFAULT_TRAJECTORY_DIR"/*.jsonl 2>/dev/null | head -n 1 || true)"
    if [[ -n "$latest" ]]; then
      copy_if_exists "$latest" "$(basename "$latest")"
    fi
  fi
  if [[ -z "$latest" ]]; then
    for legacy_dir in "${LEGACY_TRAJECTORY_DIRS[@]}"; do
      if [[ -d "$legacy_dir" ]]; then
        searched+=("$legacy_dir")
        latest="$(ls -t "$legacy_dir"/*.jsonl 2>/dev/null | head -n 1 || true)"
        if [[ -n "$latest" ]]; then
          copy_if_exists "$latest" "$(basename "$latest")"
          echo "[collect] (legacy) copied from $legacy_dir"
          break
        fi
      fi
    done
    if [[ -z "$latest" ]]; then
      echo "[collect] No trajectory files found. Checked: ${searched[*]}" >&2
    fi
  fi
fi

# Frame dumps
if [[ -n "$FRAMES_DIR" ]]; then
  copy_dir_if_exists "$FRAMES_DIR" "frames"
else
  copied=0
  if [[ -d "$DEFAULT_FRAMES_DIR" ]]; then
    copy_dir_if_exists "$DEFAULT_FRAMES_DIR" "frames"
    copied=1
  fi
  if [[ "$copied" -eq 0 ]]; then
    for legacy_dir in "${LEGACY_FRAMES_DIRS[@]}"; do
      if [[ -d "$legacy_dir" ]]; then
        copy_dir_if_exists "$legacy_dir" "frames"
        echo "[collect] (legacy) copied frames from $legacy_dir"
        copied=1
        break
      fi
    done
  fi
  if [[ "$copied" -eq 0 ]]; then
    echo "[collect] No frames directory found. Checked workspace + legacy locations." >&2
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
