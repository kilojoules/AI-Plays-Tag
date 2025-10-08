#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")"/.. && pwd)"
source "$ROOT_DIR/scripts/lib/data_paths.sh"
ai_ensure_data_dirs

DRY_RUN=0
ARCHIVE_LEGACY=0

usage() {
  cat <<'USAGE'
Migrate Godot runtime artifacts from legacy app_userdata directories into the repo-local data folder.

Usage: scripts/migrate_user_data.sh [--dry-run] [--archive-legacy]

  --dry-run         Show what would be copied without touching files.
  --archive-legacy  Move legacy directories into data/_imported/<timestamp>/ after migration.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --archive-legacy)
      ARCHIVE_LEGACY=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

mapfile -t LEGACY_TRAJECTORY_DIRS < <(ai_legacy_trajectory_dirs)
mapfile -t LEGACY_FRAMES_DIRS < <(ai_legacy_frames_dirs)

timestamp="$(date +%Y%m%d_%H%M%S)"
archive_root="$AI_MIGRATION_BACKUP_DIR/$timestamp"

copy_dir_contents() {
  local src="$1"
  local dest="$2"
  local label="$3"

  if [[ ! -d "$src" ]]; then
    return
  fi
  if [[ "$src" -ef "$dest" ]]; then
    echo "[migrate] $label already points to workspace directory ($src); skipping."
    return
  fi
  if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "[migrate][dry-run] Would copy contents of '$src' to '$dest'."
    return
  fi
  mkdir -p "$dest"
  echo "[migrate] Copying $label from '$src' to '$dest'."
  cp -R "$src"/. "$dest"/
  if [[ "$ARCHIVE_LEGACY" -eq 1 ]]; then
    local backup="$archive_root/$(basename "$(dirname "$src")")/$(basename "$src")"
    mkdir -p "$(dirname "$backup")"
    echo "[migrate] Archiving legacy $label to '$backup'."
    cp -R "$src" "$backup"
    rm -rf "$src"
  fi
}

for legacy in "${LEGACY_TRAJECTORY_DIRS[@]}"; do
  copy_dir_contents "$legacy" "$AI_TRAJECTORIES_DIR" "trajectories"
done

for legacy in "${LEGACY_FRAMES_DIRS[@]}"; do
  copy_dir_contents "$legacy" "$AI_FRAMES_DIR" "frames"
done

if [[ "$DRY_RUN" -eq 0 && "$ARCHIVE_LEGACY" -eq 0 ]]; then
  leftovers=()
  for legacy in "${LEGACY_TRAJECTORY_DIRS[@]}" "${LEGACY_FRAMES_DIRS[@]}"; do
    if [[ -d "$legacy" && ! "$legacy" -ef "$AI_TRAJECTORIES_DIR" && ! "$legacy" -ef "$AI_FRAMES_DIR" ]]; then
      leftovers+=("$legacy")
    fi
  done
  if [[ "${#leftovers[@]}" -gt 0 ]]; then
    echo "[migrate] Warning: legacy directories still exist outside the workspace:"
    for l in "${leftovers[@]}"; do
      echo "  - $l"
    done
    echo "[migrate] Re-run with --archive-legacy once you verify the copied data."
  fi
fi

if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "[migrate] Dry run complete; no files were copied."
else
  echo "[migrate] Migration finished. Current workspace data layout:"
  ai_print_data_context
fi
