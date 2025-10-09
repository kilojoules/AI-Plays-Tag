#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")"/.. && pwd)"

python3 "$ROOT_DIR/scripts/tests/test_workspace_data.py"

BIN="${GODOT_BIN:-}"
if [[ -z "$BIN" ]] && command -v godot4 >/dev/null 2>&1; then BIN="godot4"; fi
if [[ -z "$BIN" ]] && [[ -x "/Applications/Godot.app/Contents/MacOS/Godot" ]]; then BIN="/Applications/Godot.app/Contents/MacOS/Godot"; fi
if [[ -z "$BIN" ]] && [[ -x "/Users/julianquick/Downloads/Godot.app/Contents/MacOS/Godot" ]]; then BIN="/Users/julianquick/Downloads/Godot.app/Contents/MacOS/Godot"; fi
if [[ -z "$BIN" ]] && [[ -x "$ROOT_DIR/Godot.app/Contents/MacOS/Godot" ]]; then BIN="$ROOT_DIR/Godot.app/Contents/MacOS/Godot"; fi
if [[ -z "$BIN" ]]; then
  echo "Set GODOT_BIN or install godot4 in PATH" >&2
  exit 1
fi

export ROOT_DIR BIN TEST_TIMEOUT
TEST_TIMEOUT="${TEST_TIMEOUT:-90}"

python3 - "$@" <<'PY'
import os, subprocess, sys
root = os.environ['ROOT_DIR']
bin = os.environ['BIN']
timeout = float(os.environ.get('TEST_TIMEOUT','90'))
scenes = [
    'res://tests/TestRunner.tscn',
    'res://tests/GroundingStressTest.tscn',
]
for scene in scenes:
    cmd = [bin, '--headless', '--path', f"{root}/godot", scene]
    try:
        print(f"[tests] Running {scene} ...")
        proc = subprocess.run(cmd, timeout=timeout)
    except subprocess.TimeoutExpired:
        print(f"[tests] {scene} timed out after {timeout}s; killing Godot", file=sys.stderr)
        sys.exit(124)
    if proc.returncode != 0:
        sys.exit(proc.returncode)
sys.exit(0)
PY
