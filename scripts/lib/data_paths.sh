#!/usr/bin/env bash
# Shared helpers for locating runtime data directories inside the repository.
# Shell scripts should source this file after defining ROOT_DIR (falls back to repo root).

set -euo pipefail

_data_paths_guard="${_DATA_PATHS_SH_LOADED:-0}"
if [[ "$_data_paths_guard" == "1" ]]; then
  return 0
fi
export _DATA_PATHS_SH_LOADED=1

scripts_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
: "${ROOT_DIR:="$(cd "$scripts_dir/.." && pwd)"}"

case "${OSTYPE:-}" in
  msys*|cygwin*|win32*|mingw*)
    AI_PATHSEP=";"
    ;;
  *)
    AI_PATHSEP=":"
    ;;
esac
export AI_PATHSEP

# Normalize AI_DATA_ROOT to an absolute path inside the repo.
if [[ -z "${AI_DATA_ROOT:-}" ]]; then
  AI_DATA_ROOT="$ROOT_DIR/data"
elif [[ "${AI_DATA_ROOT}" != /* ]]; then
  AI_DATA_ROOT="$ROOT_DIR/${AI_DATA_ROOT}"
fi
export AI_DATA_ROOT

: "${AI_TRAJECTORIES_DIR:="$AI_DATA_ROOT/trajectories"}"
if [[ "${AI_TRAJECTORIES_DIR}" != /* ]]; then
  AI_TRAJECTORIES_DIR="$ROOT_DIR/${AI_TRAJECTORIES_DIR}"
fi
export AI_TRAJECTORIES_DIR

: "${AI_FRAMES_DIR:="$AI_DATA_ROOT/frames"}"
if [[ "${AI_FRAMES_DIR}" != /* ]]; then
  AI_FRAMES_DIR="$ROOT_DIR/${AI_FRAMES_DIR}"
fi
export AI_FRAMES_DIR

: "${AI_MIGRATION_BACKUP_DIR:="$AI_DATA_ROOT/_imported"}"
if [[ "${AI_MIGRATION_BACKUP_DIR}" != /* ]]; then
  AI_MIGRATION_BACKUP_DIR="$ROOT_DIR/${AI_MIGRATION_BACKUP_DIR}"
fi
export AI_MIGRATION_BACKUP_DIR

declare -ga _AI_LEGACY_TRAJECTORY_DIRS=()
declare -ga _AI_LEGACY_FRAMES_DIRS=()
_AI_LEGACY_TRAJECTORY_DIRS+=("$HOME/Library/Application Support/Godot/app_userdata/AI Tag Game/trajectories")
_AI_LEGACY_TRAJECTORY_DIRS+=("$HOME/.local/share/godot/app_userdata/AI Tag Game/trajectories")
_AI_LEGACY_FRAMES_DIRS+=("$HOME/Library/Application Support/Godot/app_userdata/AI Tag Game/frames")
_AI_LEGACY_FRAMES_DIRS+=("$HOME/.local/share/godot/app_userdata/AI Tag Game/frames")
if [[ -n "${APPDATA:-}" ]]; then
  _AI_LEGACY_TRAJECTORY_DIRS+=("$APPDATA/Godot/app_userdata/AI Tag Game/trajectories")
  _AI_LEGACY_FRAMES_DIRS+=("$APPDATA/Godot/app_userdata/AI Tag Game/frames")
fi

_ai_join_legacy() {
  local joined=""
  for dir in "$@"; do
    if [[ -z "$dir" ]]; then
      continue
    fi
    if [[ -z "$joined" ]]; then
      joined="$dir"
    else
      joined="${joined}${AI_PATHSEP}$dir"
    fi
  done
  printf '%s' "$joined"
}

if [[ -z "${AI_LEGACY_TRAJECTORY_DIRS:-}" ]]; then
  AI_LEGACY_TRAJECTORY_DIRS="$(_ai_join_legacy "${_AI_LEGACY_TRAJECTORY_DIRS[@]}")"
fi
if [[ -z "${AI_LEGACY_FRAMES_DIRS:-}" ]]; then
  AI_LEGACY_FRAMES_DIRS="$(_ai_join_legacy "${_AI_LEGACY_FRAMES_DIRS[@]}")"
fi
export AI_LEGACY_TRAJECTORY_DIRS
export AI_LEGACY_FRAMES_DIRS

ai_data_root() {
  printf '%s\n' "$AI_DATA_ROOT"
}

ai_trajectories_dir() {
  printf '%s\n' "$AI_TRAJECTORIES_DIR"
}

ai_frames_dir() {
  printf '%s\n' "$AI_FRAMES_DIR"
}

ai_ensure_data_dirs() {
  mkdir -p "$AI_TRAJECTORIES_DIR" "$AI_FRAMES_DIR"
}

ai_print_data_context() {
  cat <<EOF
AI_DATA_ROOT=$AI_DATA_ROOT
AI_TRAJECTORIES_DIR=$AI_TRAJECTORIES_DIR
AI_FRAMES_DIR=$AI_FRAMES_DIR
EOF
}

ai_legacy_trajectory_dirs() {
  printf '%s\n' "${_AI_LEGACY_TRAJECTORY_DIRS[@]}"
}

ai_legacy_frames_dirs() {
  printf '%s\n' "${_AI_LEGACY_FRAMES_DIRS[@]}"
}
