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
INCLUDE_FRAMES=0
FRAMES_LIMIT=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --include-frames)
      INCLUDE_FRAMES=1
      shift
      ;;
    --frames-limit)
      FRAMES_LIMIT="${2:-}"
      shift 2
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
      INCLUDE_FRAMES=1
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

if [[ -n "$FRAMES_LIMIT" && ! "$FRAMES_LIMIT" =~ ^[0-9]+$ ]]; then
  echo "frames-limit must be a non-negative integer (got '$FRAMES_LIMIT')" >&2
  exit 1
fi

copy_if_exists() {
  local src="$1"
  local name="${2:-}"
  if [[ -z "$name" ]]; then
    name="$(basename "$src")"
  fi
  if [[ -f "$src" ]]; then
    local dest_path="$DEST/$name"
    # Skip self-copy to avoid cp errors when DEST already hosts the source file.
    if [[ -f "$dest_path" && -e "$src" ]] && [[ "$src" -ef "$dest_path" ]]; then
      return
    fi
    cp "$src" "$dest_path"
    echo "[collect] copied $(basename "$src")"
  fi
}

copy_frame_subset() {
  local src="$1"
  local name="$2"
  local limit="$3"
  local dest_dir="$DEST/$name"
  if [[ ! -d "$src" ]]; then
    echo "[collect] No frames directory at $src" >&2
    return
  fi
  if ! compgen -G "$src/frame_*.png" > /dev/null; then
    echo "[collect] No frame_*.png files found in $src" >&2
    return
  fi
  mkdir -p "$dest_dir"
  local copied=0
  while IFS= read -r rel; do
    [[ -z "$rel" ]] && continue
    cp "$src/$rel" "$dest_dir/"
    ((copied++))
  done < <(cd "$src" && ls -t frame_*.png 2>/dev/null | head -n "$limit")
  if (( copied > 0 )); then
    echo "[collect] copied ${copied} frame(s) into $name (latest first, limit=$limit)"
  else
    echo "[collect] No frames selected for copy (limit=$limit)" >&2
  fi
}

copy_dir_if_exists() {
  local src="$1"
  local name="${2:-}"
  if [[ -z "$name" ]]; then
    name="$(basename "$src")"
  fi
  if [[ -d "$src" ]]; then
    local dest_dir="$DEST/$name"
    if [[ -d "$dest_dir" ]] && [[ "$src" -ef "$dest_dir" ]]; then
      return
    fi
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
if [[ "$INCLUDE_FRAMES" -eq 1 ]]; then
  selected_dir=""
  if [[ -n "$FRAMES_DIR" ]]; then
    selected_dir="$FRAMES_DIR"
  elif [[ -d "$DEFAULT_FRAMES_DIR" ]]; then
    selected_dir="$DEFAULT_FRAMES_DIR"
  else
    for legacy_dir in "${LEGACY_FRAMES_DIRS[@]}"; do
      if [[ -d "$legacy_dir" ]]; then
        selected_dir="$legacy_dir"
        echo "[collect] (legacy) selecting frames from $legacy_dir"
        break
      fi
    done
  fi
  if [[ -n "$selected_dir" ]]; then
    if [[ -n "$FRAMES_LIMIT" && "$FRAMES_LIMIT" -gt 0 ]]; then
      copy_frame_subset "$selected_dir" "frames" "$FRAMES_LIMIT"
    else
      copy_dir_if_exists "$selected_dir" "frames"
    fi
  else
    echo "[collect] No frames directory found. Skipping frames copy." >&2
  fi
else
  echo "[collect] frames not copied (pass --include-frames to bundle them)"
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
