## PRD — AI Tag Game RL Environment

### Executive Summary
This project creates an open-source framework for training reinforcement learning agents to play a 2D game of "tag." The system uses a vectorized Python environment for fast simulation and PyTorch for agent training via self-play. The project emphasizes a clean, modular architecture and includes automation scripts for training, evaluation, and visualization.

---
### Goals & Scope
- **Primary Goal**: Develop a functional end-to-end system where an agent can learn to play "tag" against another agent via self-play. The agents perform in a timed 2D environment with walls and optional obstacles. After initialization, the seeker has 10 seconds to catch the hider.
- **Demonstrable Learning**: The trained agent's performance, measured by metrics like episode reward and length, must show a clear positive trend.
- **Reproducibility**: The entire environment and training setup must be easily reproducible using Pixi across multiple platforms (macOS, Linux, Windows).
- **Visualization**: Provide tools to generate trajectory plots, statistics charts, and animated MP4s of agent behavior.

---
### Reward Function Design
Maintain `REWARD_NOTES.md` to note the current reward strategy and possible pitfalls or improvements. Make sure to remove redundant, obvious, or outdated notes.

---
### Key Deliverables & Outputs
The primary outputs of a successful training and evaluation cycle are:

* **`trainer/policy_seeker.pt`**: The saved PyTorch policy network trained for the seeker role.
* **`trainer/policy_hider.pt`**: The saved PyTorch policy network trained for the hider role.
* **`trainer/logs/metrics.csv`**: A CSV file that logs episode-level metrics like reward and episode length to track learning progress.
* **`charts/*.png`**: PNG images of the learning curves generated from the metrics.
* **`trainer/visualizations/<timestamp>/`**: Trajectory plots, statistics, and optional animated MP4s.

---
### End-to-End Workflow
1.  **Environment Setup**: Install all dependencies using `pixi install`.
2.  **Start Training**: `pixi run train` to start a training session with the vectorized 2D environment.
3.  **Monitor Progress**: Check training logs or run `pixi run monitor` to visualize learning curves.
4.  **Evaluate & Visualize**: Run `pixi run visualize` to generate trajectory plots and statistics from trained policies.

---
### Agent & Environment Design
-   **Observation Space**: The agent perceives the world through a flattened vector containing:
    -   Its own normalized position and velocity.
    -   A discretized ray "vision cone" that returns the distance and type (`wall`, `obstacle`, `agent`) of the nearest object in each direction.
    -   Role flags indicating who is the "seeker" and who is the "hider."
    -   Its forward direction vector.
    -   Safe zone state (if applicable).
-   **Action Space**: The agent outputs a continuous 2-element vector controlling `(move_x, move_z)`.
-   **Reward Function**: The reward is shaped to encourage effective play:
    -   **Seeker**: Receives a positive reward for reducing distance to the hider and a small time penalty each step. A large bonus is awarded for a successful tag.
    -   **Hider**: Receives a positive reward for increasing distance from the seeker and a small survival bonus each step. A large bonus is awarded for surviving until the time limit.

---
### Tooling & Compliance
- **Open-Source Stack**: The entire project relies exclusively on open-source software: Python, PyTorch, and various libraries available via conda-forge.
- **Environment Management**: **Pixi** is used to manage the Python environment, ensuring that the correct versions of Python, PyTorch, and other dependencies are installed consistently across all supported platforms.

---
### Error Handling & Observability
- **Training Analysis**: If an agent fails to learn, its behavior can be closely analyzed by enabling trajectory logging.
- **Logging**: The training script produces console logs essential for debugging issues during training runs.
- **Timeouts**: Include handling of cases where commands result in unexpectedly long runs. Instead of hanging, throw an error so execution can continue without intervention.
