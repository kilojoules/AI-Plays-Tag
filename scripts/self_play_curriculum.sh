#!/usr/bin/env bash
set -euo pipefail

# Orchestrate a seeker-biased self-play curriculum.
# Steps:
#   1. Run a set of short live-seeker warm-up sessions.
#   2. Launch an alternating self-play cycle with seeker-heavy weighting.
#   3. Snapshot hider opponents into a small replay pool.
#   4. (Optional) Evaluate the seeker against recent hider snapshots.
#
# Configuration knobs (env vars):
#   SELF_PLAY_SESSION_ID        Stable identifier for run grouping (default: timestamp).
#   SELF_PLAY_APPROACH          Trainer "approach" label (default: self-play-curriculum).
#   WARM_START_RUNS             Number of warm-up live-seeker runs (default: 2).
#   WARM_START_DURATION         Seconds per warm-up run (default: 90).
#   CURRICULUM_SELF_ROUNDS      Self-play rounds per cycle (default: 6).
#   CURRICULUM_SELF_DURATION    Seconds per self-play round (default: 150).
#   CURRICULUM_SEEKER_PENALTY   Seeker per-step time penalty (default: -0.005).
#   CURRICULUM_DISTANCE_SCALE   Distance reward scale (default: 0.15).
#   CURRICULUM_TARGET_WIN       Required seeker win-rate threshold (default: 0.55).
#   CURRICULUM_CHECK_EPISODES   Recent episodes considered when computing win-rate (default: 20).
#   CURRICULUM_POOL_DIR         Directory for archived hider policies (default: trainer/policy_pool/hider).
#   CURRICULUM_POOL_KEEP        Number of hider snapshots to retain (default: 4).
#   CURRICULUM_POOL_EVAL        If 1, run evaluation against recent pool entries (default: 1).
#   CURRICULUM_POOL_EVAL_COUNT  Opponents drawn from the pool when evaluating (default: 2).
#   CURRICULUM_POOL_EVAL_EPIS   Episodes per opponent during evaluation (default: 5).
#   CURRICULUM_POOL_EVAL_DUR    Seconds per evaluation episode (default: 12).
#
# The script exits with status 0 on success (seeker cleared thresholds), otherwise 1.

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")"/.. && pwd)"
source "$ROOT_DIR/scripts/lib/data_paths.sh"
ai_ensure_data_dirs

log() {
  echo "[self-play-curriculum] $*"
}

need_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing required command: $1" >&2
    exit 1
  fi
}

need_cmd bash
need_cmd python3

SESSION_ID="${SELF_PLAY_SESSION_ID:-$(date +%Y%m%d_%H%M%S)}"
APPROACH="${SELF_PLAY_APPROACH:-self-play-curriculum}"
WARM_START_RUNS="${WARM_START_RUNS:-2}"
WARM_START_DURATION="${WARM_START_DURATION:-90}"
CURRICULUM_SELF_ROUNDS="${CURRICULUM_SELF_ROUNDS:-6}"
CURRICULUM_SELF_DURATION="${CURRICULUM_SELF_DURATION:-150}"
CURRICULUM_SEEKER_PENALTY="${CURRICULUM_SEEKER_PENALTY:--0.005}"
CURRICULUM_DISTANCE_SCALE="${CURRICULUM_DISTANCE_SCALE:-0.15}"
CURRICULUM_TARGET_WIN="${CURRICULUM_TARGET_WIN:-0.55}"
CURRICULUM_CHECK_EPISODES="${CURRICULUM_CHECK_EPISODES:-20}"
CURRICULUM_POOL_DIR="${CURRICULUM_POOL_DIR:-$ROOT_DIR/trainer/policy_pool/hider}"
CURRICULUM_POOL_KEEP="${CURRICULUM_POOL_KEEP:-4}"
CURRICULUM_POOL_EVAL="${CURRICULUM_POOL_EVAL:-1}"
CURRICULUM_POOL_EVAL_COUNT="${CURRICULUM_POOL_EVAL_COUNT:-2}"
CURRICULUM_POOL_EVAL_EPIS="${CURRICULUM_POOL_EVAL_EPIS:-5}"
CURRICULUM_POOL_EVAL_DUR="${CURRICULUM_POOL_EVAL_DUR:-12}"

mkdir -p "$CURRICULUM_POOL_DIR"

log "session=${SESSION_ID} approach=${APPROACH}"
log "warm_start_runs=${WARM_START_RUNS} warm_duration=${WARM_START_DURATION}s"
log "self_play_rounds=${CURRICULUM_SELF_ROUNDS} self_play_duration=${CURRICULUM_SELF_DURATION}s"
log "seeker_penalty=${CURRICULUM_SEEKER_PENALTY} distance_scale=${CURRICULUM_DISTANCE_SCALE}"

metrics_path_for() {
  local run_id="$1"
  printf '%s\n' "$ROOT_DIR/trainer/logs/runs/$APPROACH/$run_id/metrics.csv"
}

run_warm_start() {
  local idx="$1"
  local run_id="${SESSION_ID}_warm${idx}"
  log "Warm start run $idx/$WARM_START_RUNS (run_id=$run_id)"
  TRAIN_APPROACH="$APPROACH" \
  TRAIN_RUN_ID="$run_id" \
  AI_TRAIN_DURATION="$WARM_START_DURATION" \
  AI_DISTANCE_REWARD_SCALE="$CURRICULUM_DISTANCE_SCALE" \
  AI_SEEKER_TIME_PENALTY="$CURRICULUM_SEEKER_PENALTY" \
  AI_LOG_TRAJECTORIES="${AI_LOG_TRAJECTORIES:-0}" \
  bash "$ROOT_DIR/scripts/train.sh" live-seeker
  local metrics_file
  metrics_file="$(metrics_path_for "$run_id")"
  if [[ ! -f "$metrics_file" ]]; then
    log "warning: metrics file missing for warm run ($metrics_file)"
  fi
}

run_self_play_cycle() {
  local run_id="${SESSION_ID}_selfplay"
  log "Self-play cycle (run_id=$run_id)"
  TRAIN_APPROACH="$APPROACH" \
  TRAIN_RUN_ID="$run_id" \
  AI_DISTANCE_REWARD_SCALE="$CURRICULUM_DISTANCE_SCALE" \
  AI_SEEKER_TIME_PENALTY="$CURRICULUM_SEEKER_PENALTY" \
  SELF_PLAY_ROUNDS="$CURRICULUM_SELF_ROUNDS" \
  SELF_PLAY_DURATION="$CURRICULUM_SELF_DURATION" \
  SELF_PLAY_START_ROLE="seeker" \
  bash "$ROOT_DIR/scripts/train.sh" self-play
  metrics_path_for "$run_id"
}

compute_recent_stats() {
  local metrics_file="$1"
  if [[ ! -f "$metrics_file" ]]; then
    echo "0 0 0 0 0"
    return
  fi
  python3 - "$metrics_file" "$CURRICULUM_CHECK_EPISODES" <<'PY'
import csv
import sys

path, limit_raw = sys.argv[1], sys.argv[2]
try:
    limit = int(limit_raw)
except ValueError:
    limit = 0
rows = []
with open(path, newline="") as fh:
    reader = csv.DictReader(fh)
    for row in reader:
        rows.append(row)
if limit > 0 and len(rows) > limit:
    rows = rows[-limit:]
total = len(rows)
if total == 0:
    print("0 0 0 0 0")
    sys.exit(0)
wins = sum(1 for row in rows if row.get("winner", "").strip().lower() == "seeker")
def safe_float(val):
    try:
        return float(val)
    except Exception:
        return 0.0
reward_values = [safe_float(row.get("seeker_reward_mean", 0.0)) for row in rows]
distance_values = [safe_float(row.get("seeker_avg_distance", 0.0)) for row in rows]
reward_avg = sum(reward_values) / len(reward_values) if reward_values else 0.0
distance_avg = sum(distance_values) / len(distance_values) if distance_values else 0.0
win_rate = wins / total if total else 0.0
print(f"{wins} {total} {win_rate:.4f} {reward_avg:.4f} {distance_avg:.4f}")
PY
}

snapshot_hider_to_pool() {
  local source="$ROOT_DIR/trainer/policy_hider.pt"
  if [[ ! -f "$source" ]]; then
    log "warning: hider policy missing; skipping pool snapshot."
    return
  fi
  local dest="$CURRICULUM_POOL_DIR/policy_hider_${SESSION_ID}.pt"
  cp "$source" "$dest"
  log "Snapshot hider policy -> $(basename "$dest")"
}

prune_pool() {
  local keep="$CURRICULUM_POOL_KEEP"
  if [[ "$keep" -le 0 ]]; then
    return
  fi
  local snapshots=()
  while IFS= read -r snapshot_path; do
    [[ -n "$snapshot_path" ]] && snapshots+=("$snapshot_path")
  done < <(ls -1t "$CURRICULUM_POOL_DIR"/policy_hider_*.pt 2>/dev/null || true)
  if (( ${#snapshots[@]} <= keep )); then
    return
  fi
  log "Pruning hider pool to keep ${keep} latest snapshots."
  for old in "${snapshots[@]:keep}"; do
    rm -f -- "$old"
  done
}

extract_winner_from_trajectory() {
  local trajectory_path="$1"
  if [[ ! -f "$trajectory_path" ]]; then
    echo ""
    return
  fi
  python3 - "$trajectory_path" <<'PY'
import json
import sys

winner = ""
try:
    with open(sys.argv[1], "r") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            info = obj.get("info") or {}
            if "winner" in info and info["winner"]:
                winner = str(info["winner"])
except FileNotFoundError:
    winner = ""
print(winner)
PY
}

evaluate_against_pool() {
  if [[ "$CURRICULUM_POOL_EVAL" != "1" ]]; then
    log "Pool evaluation disabled (CURRICULUM_POOL_EVAL!=1)."
    return 0
  fi
  local hider_path="$ROOT_DIR/trainer/policy_hider.pt"
  if [[ ! -f "$hider_path" ]]; then
    log "warning: hider policy missing; skipping pool evaluation."
    return 0
  fi
  local opponents=()
  while IFS= read -r opponent_path; do
    [[ -n "$opponent_path" ]] && opponents+=("$opponent_path")
  done < <(ls -1t "$CURRICULUM_POOL_DIR"/policy_hider_*.pt 2>/dev/null || true)
  if (( ${#opponents[@]} == 0 )); then
    log "No opponents in pool yet; skipping evaluation."
    return 0
  fi
  local limit="$CURRICULUM_POOL_EVAL_COUNT"
  if (( limit > 0 && ${#opponents[@]} > limit )); then
    opponents=("${opponents[@]:0:limit}")
  fi
  local backup="${hider_path}.curriculum.bak"
  cp "$hider_path" "$backup"
  trap 'mv "$backup" "$hider_path" >/dev/null 2>&1 || true' RETURN
  local total_eps=0
  local seeker_wins=0
  local idx=0
  for opponent in "${opponents[@]}"; do
    idx=$((idx + 1))
    log "Evaluating against pool opponent ${idx}/${#opponents[@]} ($(basename "$opponent"))"
    cp "$opponent" "$hider_path"
    local episodes="$CURRICULUM_POOL_EVAL_EPIS"
    local ep=1
    while (( ep <= episodes )); do
      local eval_run_id="${SESSION_ID}_opp${idx}_ep$(printf '%02d' "$ep")"
      local charts_dir="$ROOT_DIR/charts/self_play_curriculum"
      mkdir -p "$charts_dir"
      local output_png="$charts_dir/${eval_run_id}.png"
      EVAL_RUN_ID="$eval_run_id" \
      bash "$ROOT_DIR/scripts/eval_episode.sh" \
        --duration "$CURRICULUM_POOL_EVAL_DUR" \
        --start-role seeker \
        --output "$output_png" \
        --title "Curriculum Eval $eval_run_id"
      local traj="$ROOT_DIR/debug/${eval_run_id}_eval/trajectory.jsonl"
      local winner
      winner="$(extract_winner_from_trajectory "$traj")"
      if [[ -z "$winner" ]]; then
        log "warning: could not determine winner for ${eval_run_id} (trajectory: $traj)"
      else
        log "  episode result: winner=${winner}"
        if [[ "${winner,,}" == "seeker" ]]; then
          seeker_wins=$((seeker_wins + 1))
        fi
      fi
      total_eps=$((total_eps + 1))
      ep=$((ep + 1))
    done
  done
  trap - RETURN
  mv "$backup" "$hider_path"
  if (( total_eps == 0 )); then
    log "warning: pool evaluation produced no completed episodes."
    return 0
  fi
  local win_rate
  win_rate=$(python3 - <<PY "$seeker_wins" "$total_eps"
import sys
wins = int(sys.argv[1])
total = int(sys.argv[2])
rate = wins / total if total else 0.0
print(f"{rate:.4f}")
PY
)
  log "Pool evaluation seeker wins=${seeker_wins}/${total_eps} (rate=${win_rate})"
  python3 - "$ROOT_DIR/trainer/logs/runs/$APPROACH" "$SESSION_ID" "$seeker_wins" "$total_eps" "$win_rate" <<'PY'
import json
import os
import sys

root, session_id, wins_raw, total_raw, rate = sys.argv[1:]
wins = int(wins_raw)
total = int(total_raw)
summary_dir = os.path.join(root, f"{session_id}_summary")
os.makedirs(summary_dir, exist_ok=True)
with open(os.path.join(summary_dir, "pool_eval.json"), "w") as fh:
    json.dump(
        {"session": session_id, "wins": wins, "episodes": total, "win_rate": float(rate)},
        fh,
        indent=2,
    )
PY
  return 0
}

main() {
  local i=1
  while (( i <= WARM_START_RUNS )); do
    run_warm_start "$i"
    i=$((i + 1))
  done

  local self_metrics
  self_metrics="$(run_self_play_cycle)"

  snapshot_hider_to_pool
  prune_pool

  read wins total rate reward_avg distance_avg < <(compute_recent_stats "$self_metrics")
  log "Seeker recent stats: wins=${wins}/${total} rate=${rate} reward_mean=${reward_avg} distance_avg=${distance_avg}"

  python3 - "$ROOT_DIR/trainer/logs/runs/$APPROACH" "$SESSION_ID" "$self_metrics" "$wins" "$total" "$rate" "$reward_avg" "$distance_avg" <<'PY'
import json
import os
import sys

root, session, metrics_path, wins, total, rate, reward, dist = sys.argv[1:]
summary_dir = os.path.join(root, f"{session}_summary")
os.makedirs(summary_dir, exist_ok=True)
with open(os.path.join(summary_dir, "metrics.json"), "w") as fh:
    json.dump(
        {
            "session": session,
            "metrics_csv": os.path.relpath(metrics_path, root),
            "recent": {
                "wins": int(wins),
                "episodes": int(total),
                "win_rate": float(rate),
                "reward_mean": float(reward),
                "distance_avg": float(dist),
            },
        },
        fh,
        indent=2,
    )
PY

  local status=0
  python3 - <<'PY' "$rate" "$CURRICULUM_TARGET_WIN"
import sys
rate = float(sys.argv[1])
target = float(sys.argv[2])
if rate + 1e-6 < target:
    sys.exit(1)
PY
  status=$?

  evaluate_against_pool || true

  if [[ "$status" -ne 0 ]]; then
    log "FAIL: seeker win rate ${rate} below target ${CURRICULUM_TARGET_WIN}"
    exit 1
  fi

  log "SUCCESS: seeker win rate ${rate} cleared target ${CURRICULUM_TARGET_WIN}"
}

main "$@"
