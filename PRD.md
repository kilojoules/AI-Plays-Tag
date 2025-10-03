## PRD — AI Tag Game RL Environment

### Executive Summary
This project will create a complete, open-source framework for training reinforcement learning agents to play a 3D game of "tag." The system uses the Godot 4 game engine for the simulation environment and a Python server with PyTorch for agent training. The two components communicate via a WebSocket bridge, allowing for live, end-to-end training where agents visibly improve their chasing and evasion strategies. The project emphasizes a clean, modular architecture and includes a suite of automation scripts for training, evaluation, and visualization. The project is developed with the goal of making a visually pleasing animation for a youtube video, where the agents can usually be seen, are approximately the same size close to and far away from the camera, and play in an enclosed environment. 

---
### Goals & Scope
- **Primary Goal**: Develop a functional end-to-end system where an agent can learn to play "tag" against another agent (either scripted or self-play). This necesitates high-quality physics simulations, where there is realistic agnet sensing/seeing and movements, and walls are absolutely non-permeable by agents
  The idea is that we have a hider and seeker agent. Each agent is a cube that can move itself and jump. They have "eyes" that can see a limitted amount of information.
  The agents perform in a timed environment. After initialization, the seeker has 10 seconds to chase the hider. Otherwise, the seeker looses.
- **Demonstrable Learning**: The trained agent's performance, measured by metrics like episode reward and length, must show a clear positive trend.
- **Reproducibility**: The entire environment and training setup must be easily reproducible using Pixi across multiple platforms (macOS, Linux, Windows).
- **Visualization**: Provide tools to record videos of agent behavior and plot learning curves from training logs.

---
### Reward Function Design
Maintain `REWARD_NOTES.md` to note the current reward strategy and possible pitfalls or improvements. Make sure to remove redundant, obvious, or outdated notes. 

---
### Key Deliverables & Outputs
The primary outputs of a successful training and evaluation cycle are:

* **`trainer/policy_seeker.pt`**: The saved PyTorch policy network trained specifically for the chaser/seeker role.
* **`trainer/policy_hider.pt`**: The saved PyTorch policy network trained specifically for the hider/runner role.
* **`trainer/logs/metrics.csv`**: A CSV file that logs episode-level metrics like reward and episode length to track learning progress.
* **`charts/*.png`**: PNG images of the learning curves generated from the `metrics.csv` file.
* **`learn_progress-*.mp4`**: An MP4 video demonstrating an agent's learned behavior. After a satisfactory policy has been trained, this video is created by running the simulation in GUI mode to capture frames, which are then encoded.
* **`user://trajectories/ep_*.jsonl`**: (Optional) Detailed, per-step logs of agent states and actions during an episode. These files are used for fine-grained analysis and for rendering replay videos.

---
### End-to-End Workflow
1.  **Environment Setup**: A user installs all dependencies for both the `server` and `train` environments using a single command: `pixi install`.
2.  **Testing**: The user tests the setup, ensuring the project is high quality, including things like:
    - agents can't go through walls
    - agents can't fall through floor
    - agents will interact in inital policy?
    - agents see correctly
    - agent movement behaves as expected
    - gravity is respected
    - camera capture is as desired
2.  **Start Training**: The user initiates a complete training session with a single command, like `bash scripts/train.sh live-seeker`. This script automatically starts the Python training server and launches the Godot simulation in headless mode.
3.  **Monitor Progress**: The user can monitor the server's console output for training updates and periodically check the `trainer/logs/metrics.csv` file or run `bash scripts/plot_metrics.sh` to visualize learning curves.
4.  **Evaluate & Record**: Once a satisfactory policy is trained (`trainer/policy.pt`), the user stops the training script and runs `GODOT_BIN=/path/to/Godot bash scripts/record_gui.sh seeker` to launch the Godot GUI and record the agent's performance.
5.  **Encode Video**: After closing the Godot window, the user runs `bash scripts/encode_frames.sh` to convert the recorded image frames into a shareable MP4 video.

---
### Agent & Environment Design
-   **Observation Space**: The agent perceives the world through a flattened vector containing:
    -   Its own normalized position and velocity.
    -   A sensibly discreitzed ray "vision cone" that returns the distance and type (`wall`, `agent`, `none`) of the nearest object in each direction.
    -   Role flags indicating who is the "seeker" and who is the "hider."
    -   Its forward direction vector.
-   **Action Space**: The agent outputs a continuous 3-element vector controlling `(move_x, move_z, jump)`.
-   **Reward Function**: The reward is shaped to encourage effective play:
    -   **Seeker**: Receives a positive reward for reducing its distance to the hider and a small time penalty each step. A large bonus is awarded for a successful tag.
    -   **Hider**: Receives a positive reward for increasing its distance from the seeker and a small survival bonus each step. A large bonus is awarded for surviving until the time limit.

---
### Tooling & Compliance
- **Open-Source Stack**: The entire project relies exclusively on open-source software: Godot, Python, PyTorch, and various libraries available via conda-forge.
- **Environment Management**: **Pixi** is used to manage the Python environment, ensuring that the correct versions of Python, PyTorch, and other dependencies are installed consistently across all supported platforms.

---
### Error Handling & Observability
- **Connection Errors**: The `README.md` documents the common WebSocket handshake error that occurs when a non-WebSocket client (like a browser) attempts to connect to the server port.
- **Training Analysis**: If an agent fails to learn, its behavior can be closely analyzed by enabling trajectory logging 
- **Logging**: The Python server and Godot engine both produce console logs that are essential for debugging issues during training runs.
- **timeouts**: include handling of cases where your command results in unexpectadly long runs. Instead of hanging, you shoul throw an error so you can continue without my intervention.

