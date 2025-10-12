#!/usr/bin/env python3
"""Plot seeker/hider paths from a trajectory JSONL file."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple



def repo_default_trajectory_dir() -> Path:
    project_root = Path(__file__).resolve().parents[1]
    return project_root / "data" / "trajectories"


def legacy_trajectory_dirs() -> Iterable[Path]:
    home = Path.home()
    yield home / "Library" / "Application Support" / "Godot" / "app_userdata" / "AI Tag Game" / "trajectories"
    yield home / ".local" / "share" / "godot" / "app_userdata" / "AI Tag Game" / "trajectories"
    appdata = os.environ.get("APPDATA")
    if appdata:
        yield Path(appdata) / "Godot" / "app_userdata" / "AI Tag Game" / "trajectories"


def candidate_trajectory_dirs() -> List[Path]:
    candidates: List[Path] = []
    env_dir = os.environ.get("AI_TRAJECTORIES_DIR")
    if env_dir:
        candidates.append(Path(env_dir))
    candidates.append(repo_default_trajectory_dir())
    legacy_env = os.environ.get("AI_LEGACY_TRAJECTORY_DIRS", "")
    if legacy_env:
        for entry in legacy_env.split(os.pathsep):
            entry = entry.strip()
            if entry:
                candidates.append(Path(entry))
    candidates.extend(list(legacy_trajectory_dirs()))

    seen: set[Path] = set()
    unique: List[Path] = []
    for path in candidates:
        try:
            resolved = path.resolve()
        except OSError:
            continue
        if resolved not in seen:
            seen.add(resolved)
            unique.append(path)
    return unique


def find_latest_trajectory(allow_missing: bool = False) -> Optional[Path]:
    for directory in candidate_trajectory_dirs():
        if not directory.is_dir():
            continue
        files = sorted(directory.glob("*.jsonl"))
        if files:
            return files[-1]
    if allow_missing:
        return None
    raise SystemExit("No trajectory files found in known directories. Set --trajectory explicitly.")


def load_paths(path: Path) -> Dict[str, Dict[str, Any]]:
    paths: Dict[str, Dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if obj.get("type") != "step":
                continue
            agent = str(obj.get("agent", ""))
            pos = obj.get("pos")
            if not agent or not isinstance(pos, list) or len(pos) < 3:
                continue
            x, _, z = pos
            entry = paths.setdefault(agent, {"coords": [], "initial_is_it": None})
            entry["coords"].append((float(x), float(z)))
            if entry["initial_is_it"] is None:
                is_it = obj.get("is_it")
                if isinstance(is_it, bool):
                    entry["initial_is_it"] = is_it
    return paths


def plot_paths(paths: Dict[str, Dict[str, Any]], output: Path, title: str, dpi: int) -> None:
    import matplotlib.pyplot as plt  # Lazy import to keep tests lightweight
    plt.style.use("seaborn-v0_8")
    fig, ax = plt.subplots(figsize=(6, 6))
    role_colors = {"seeker": "tab:red", "hider": "tab:blue"}
    for agent, info in paths.items():
        coords = info.get("coords", [])
        if len(coords) < 2:
            continue
        xs, zs = zip(*coords)
        initial_is_it = info.get("initial_is_it")
        role = "unknown"
        if isinstance(initial_is_it, bool):
            role = "seeker" if initial_is_it else "hider"
        color = role_colors.get(role, None)
        label_agent = agent
        if role != "unknown":
            label_agent = f"{role.title()} ({agent})"
        ax.plot(xs, zs, label=label_agent, color=color, linewidth=2)
        start_label = f"{role.title()} start" if role != "unknown" else f"{agent} start"
        end_label = f"{role.title()} end" if role != "unknown" else f"{agent} end"
        ax.scatter([xs[0]], [zs[0]], color=color or "black", marker="o", s=60, label=start_label)
        ax.scatter([xs[-1]], [zs[-1]], color=color or "black", marker="X", s=70, label=end_label)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("X position (m)")
    ax.set_ylabel("Z position (m)")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    handles, labels = ax.get_legend_handles_labels()
    seen = set()
    uniq_handles = []
    uniq_labels = []
    for h, l in zip(handles, labels):
        if l in seen:
            continue
        seen.add(l)
        uniq_handles.append(h)
        uniq_labels.append(l)
    ax.legend(uniq_handles, uniq_labels, loc="best")
    ax.set_title(title)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output, dpi=dpi)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot seeker/hider paths from a trajectory JSONL file.")
    parser.add_argument("--trajectory", type=Path, help="Path to trajectory JSONL (defaults to latest).", default=None)
    parser.add_argument("--output", type=Path, help="PNG output path", default=Path("charts/trajectory_paths.png"))
    parser.add_argument("--title", type=str, default="Agent Paths")
    parser.add_argument("--dpi", type=int, default=150, help="Output image DPI (default: 150).")
    parser.add_argument("--print-latest", action="store_true", help="Print the latest trajectory path and exit.")
    args = parser.parse_args()

    if args.print_latest:
        traj = find_latest_trajectory(allow_missing=True)
        if traj:
            print(traj)
        return

    traj = args.trajectory or find_latest_trajectory()
    if traj is None:
        raise SystemExit("No trajectory found; provide --trajectory explicitly.")
    paths = load_paths(traj)
    if not paths:
        raise SystemExit(f"No path data extracted from {traj}")
    dpi = max(50, args.dpi)
    plot_paths(paths, args.output, args.title, dpi)
    print(f"Saved path plot to {args.output}")


if __name__ == "__main__":  # pragma: no cover - script entry point guard
    main()
