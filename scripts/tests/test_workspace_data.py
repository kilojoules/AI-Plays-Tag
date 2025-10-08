#!/usr/bin/env python3
"""
Sanity checks for the workspace-local data directory wiring.

Ensures helper scripts see trajectories inside `data/` without
depending on platform-specific Godot app_userdata folders.
"""

from __future__ import annotations

import importlib
import os
import subprocess
import sys
from pathlib import Path


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    data_root = repo_root / "data"
    trajectories_dir = data_root / "trajectories"
    trajectories_dir.mkdir(parents=True, exist_ok=True)

    sample_path = trajectories_dir / "ep_99999_workspace_sanity.jsonl"
    sample_payload = (
        '{"type":"episode_start","episode":99999}\n'
        '{"type":"step","agent":"Agent1","pos":[0.0,1.5,0.0],"is_it":true}\n'
        '{"type":"step","agent":"Agent2","pos":[1.0,1.5,0.0],"is_it":false}\n'
        '{"type":"episode_end","episode":99999}\n'
    )
    sample_path.write_text(sample_payload, encoding="utf-8")

    env = os.environ.copy()
    env["AI_DATA_ROOT"] = str(data_root)
    env["AI_TRAJECTORIES_DIR"] = str(trajectories_dir)
    env["AI_PATHSEP"] = os.pathsep
    env["MPLBACKEND"] = "Agg"
    # Reset legacy entries so fallbacks do not mask bugs.
    env.pop("AI_LEGACY_TRAJECTORY_DIRS", None)

    try:
        # Reload the metrics module to ensure it picks up the new environment.
        if "trainer.metrics_from_trajectories" in sys.modules:
            del sys.modules["trainer.metrics_from_trajectories"]
        sys.path.insert(0, str(repo_root))
        metrics = importlib.import_module("trainer.metrics_from_trajectories")
        directory, files = metrics.find_first_dir_with_trajectories()
        assert Path(directory) == trajectories_dir.resolve(), (
            f"metrics_from_trajectories should prefer workspace data dir "
            f"(expected {trajectories_dir}, got {directory})"
        )
        assert Path(files[-1]).resolve() == sample_path.resolve(), (
            "metrics_from_trajectories returned unexpected file"
        )

        # Ensure the plotting helper resolves the same workspace trajectory.
        cmd = [
            sys.executable,
            "scripts/plot_paths_from_trajectory.py",
            "--print-latest",
        ]
        latest = subprocess.check_output(cmd, cwd=repo_root, env=env, text=True).strip()
        assert Path(latest).resolve() == sample_path.resolve(), (
            "plot_paths_from_trajectory did not pick workspace trajectory"
        )
        print("[workspace-data-test] OK")
    finally:
        try:
            sample_path.unlink()
        except FileNotFoundError:
            pass


if __name__ == "__main__":
    main()
