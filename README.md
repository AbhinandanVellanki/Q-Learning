# HW4 — Q-Learning in a GridWorld

Tabular Q-learning agent trained in an 11x11 gridworld with bombs (-500 reward) and a gold goal (+500 reward), benchmarked against a random baseline. Assignment for 16-662 (Robot Autonomy), CMU.

## Contents

- `code/environment.py` — `GridWorld`: defines the grid, bomb/gold terminal states, stochastic transitions (`rho`), step/reset API, and rendering/video-saving for visualization.
- `code/agents.py` — `RandomAgent` (uniform random action) and `QAgent` (epsilon-greedy tabular Q-learning with a `q_table` keyed by grid cell, plus a `visualize_q_values` plot of the learned table).
- `code/q_learning.py` — entry point: benchmarks the random agent, trains `QAgent` for 1000 trials, plots reward-per-trial, benchmarks the trained agent, and renders sample rollout videos for both agents.
- `visualizations/` — rendered MP4 rollouts (`random_agent_*.mp4`, `q_agent_*.mp4`).
- `q_learning_training.png`, `q_agent_initial.png`, `Best Q Values.png` — training curve and learned Q-value heatmaps.
- `HW4.pdf` — assignment spec; `abhinanv_HW4.pdf` — submitted write-up.

## Running

```bash
pip install numpy matplotlib opencv-python
cd code
python q_learning.py
```

Outputs land in `code/visualizations/` and as PNGs in `code/` (`q_learning_training.png`, etc.) — move/compare against the top-level copies if regenerating.
