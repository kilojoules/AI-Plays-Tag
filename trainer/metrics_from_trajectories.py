#!/usr/bin/env python3
import glob
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


def _repo_default_dir() -> str:
    project_root = Path(__file__).resolve().parents[1]
    return str(project_root / "data" / "trajectories")


def _legacy_dirs_from_env(env_key: str) -> Iterable[str]:
    raw = os.environ.get(env_key, "")
    if not raw:
        return []
    sep = os.environ.get("AI_PATHSEP", os.pathsep)
    return [entry.strip() for entry in raw.split(sep) if entry.strip()]


def candidate_trajectory_dirs() -> List[str]:
    candidates: List[str] = []
    env_dir = os.environ.get("AI_TRAJECTORIES_DIR")
    if env_dir:
        candidates.append(env_dir)
    candidates.append(_repo_default_dir())
    candidates.extend(_legacy_dirs_from_env("AI_LEGACY_TRAJECTORY_DIRS"))

    # Fallback to OS-specific defaults to support pre-migration artifacts.
    home = Path.home()
    candidates.append(str(home / "Library" / "Application Support" / "Godot" / "app_userdata" / "AI Tag Game" / "trajectories"))
    candidates.append(str(home / ".local" / "share" / "godot" / "app_userdata" / "AI Tag Game" / "trajectories"))
    appdata = os.environ.get("APPDATA")
    if appdata:
        candidates.append(str(Path(appdata) / "Godot" / "app_userdata" / "AI Tag Game" / "trajectories"))

    seen = set()
    ordered: List[str] = []
    for path in candidates:
        if not path:
            continue
        try:
            resolved = str(Path(path).expanduser().resolve())
        except OSError:
            continue
        if resolved not in seen:
            seen.add(resolved)
            ordered.append(resolved)
    return ordered


def find_first_dir_with_trajectories() -> Tuple[str, List[str]]:
    for directory in candidate_trajectory_dirs():
        if not os.path.isdir(directory):
            continue
        files = sorted(glob.glob(os.path.join(directory, "*.jsonl")))
        if files:
            return directory, files
    return "", []


def parse_episode(path: str) -> Tuple[int, Dict[str, int]]:
    steps = 0
    scores: Dict[str, int] = {}
    with open(path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            typ = obj.get("type")
            if typ == "step":
                steps += 1
            elif typ == "tag":
                atk = obj.get("attacker")
                tgt = obj.get("target")
                if atk:
                    scores[atk] = scores.get(atk, 0) + 1
                if tgt:
                    scores[tgt] = scores.get(tgt, 0) - 1
    return steps, scores


def main():
    tdir, files = find_first_dir_with_trajectories()
    if not tdir:
        print("No trajectory files found in known directories.")
        return
    print(f"Using trajectories from: {tdir}")
    out_dir = os.path.join(os.path.dirname(__file__), "logs")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "metrics.csv")
    with open(csv_path, "w") as f:
        f.write("episode,reward_mean,reward_sum,steps,updates\n")
        for i, fp in enumerate(files, start=1):
            steps, scores = parse_episode(fp)
            # zero-sum; choose a differential: Agent1 - Agent2 if present; else total sum
            if scores:
                agents = list(scores.keys())
                if len(agents) >= 2:
                    diff = scores.get("Agent1", 0) - scores.get("Agent2", 0)
                else:
                    diff = sum(scores.values())
            else:
                diff = 0
            f.write(f"{i},{float(diff)},{float(diff)},{steps},0\n")
    print("Wrote:", csv_path)


if __name__ == "__main__":
    main()
