#!/usr/bin/env python3
import os
import json
import glob
from typing import Dict, List, Tuple


def trajectories_dir() -> str:
    home = os.path.expanduser("~")
    # macOS default
    mac = os.path.join(home, "Library", "Application Support", "Godot", "app_userdata", "AI Tag Game", "trajectories")
    if os.path.isdir(mac):
        return mac
    # Linux default
    lin = os.path.join(home, ".local", "share", "godot", "app_userdata", "AI Tag Game", "trajectories")
    if os.path.isdir(lin):
        return lin
    # Windows default
    win = os.path.join(os.environ.get("APPDATA", os.path.join(home, "AppData", "Roaming")), "Godot", "app_userdata", "AI Tag Game", "trajectories")
    if os.path.isdir(win):
        return win
    return ""


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
    tdir = trajectories_dir()
    if not tdir:
        print("No trajectories directory found.")
        return
    files = sorted(glob.glob(os.path.join(tdir, "ep_*.jsonl")))
    if not files:
        print("No trajectory files found in", tdir)
        return
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

