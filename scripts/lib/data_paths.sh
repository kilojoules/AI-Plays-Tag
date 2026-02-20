#!/usr/bin/env bash
# Shared helpers for locating runtime data directories inside the repository.
# Shell scripts should source this file after defining ROOT_DIR (falls back to repo root).

set -euo pipefail

if [[ "${_DATA_PATHS_SH_LOADED:-0}" == "1" ]]; then
  return 0
fi

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

_DATA_PATHS_SH_LOADED=1
