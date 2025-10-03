# Git & Collaboration Guidelines

This repository is shared by multiple LLM and human collaborators. To keep the history reviewable while enabling parallel work, follow the workflow below.

## Branching
- Base your work from the latest `master`.
- Create a focused feature branch per task, e.g. `feature/godot-obs-tests`, `docs/reward-notes`.
- For stacked work, branch off the commit that contains the prerequisite change and open PRs against that base (GitHub/GitLab supports stacked PRs).

## Commits
- Keep commits small and self-contained. Reference the relevant TODO entry (`TODO.md`, `godot/TODO.md`, etc.) in the commit message.
- Suggested prefixes: `feat`, `fix`, `chore`, `docs`, `test`.
- Avoid including generated artifacts (charts, logs, videos, `debug/` dumps); they are ignored.

## Validation
- Run `pixi run tests` before opening or merging a PR.
- For training changes, run `bash scripts/train.sh live-seeker` long enough to ensure the server loop stays healthy and attach logs if issues appear.

## PR Checklist
1. Rebase on `origin/master` and resolve conflicts.
2. Ensure TODO status updates are included in the branch when a task is completed or blocked.
3. Provide a short summary, list of validation commands, and indicate any blockers or follow-up TODOs.

## Logs and Debugging
- Each session should log activities in `LLM_Logs/YYYYMMDD_HHmmss_<slug>.log` with prompt, task, actions, tests, and commit hashes.
- Use `scripts/collect_debug_artifacts.sh` (or `pixi run collect-debug`) to bundle logs when capturing failures.

## Merging
- Prefer fast-forward merges when history is linear.
- Squash only if the intermediate commits are noisy or experimental; otherwise keep the structured commits for traceability.
- After a PR merges, delete the branch locally/remotely and update `TODO.md` checkboxes on `master`.

## Code Review Focus
- Prioritise correctness, safety, and adherence to PRD.md.
- Highlight missing tests or validation gaps.
- Keep reviews centered on findings; use TODO files to note follow-up items if a change is approved with follow-up work required.
