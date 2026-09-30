# Mobile Robot Path Planning with CNNs and Reinforcement Learning

Learning-based path planning for a mobile robot. A **convolutional neural network (CNN)** and a **reinforcement learning (RL)** agent are trained to find paths on 2D grid maps, first on random mazes and then on real city street maps. Their results are then compared.

📑 **[Project presentation (PDF)](docs/Path_Planning_CNN_RL_Presentation.pdf)**

---

## Overview

- **Maps.** Planning is first tested on 2D grid-based mazes. Real **city maps** (New York street maps at 256/512/1024 px) are then converted to occupancy grids and used as the environment.
- **CNN planner.** Convolutional and max-pooling layers extract spatial information from the map, and a fully connected head makes the high-level decision, i.e. the path. It needs no pre-processing of the map data. Ground-truth shortest paths from **A\*** are used for training and evaluation.
- **RL planner.** A reinforcement learning agent learns to navigate the same grids.
- **Comparison.** The two approaches are compared on the same maps and the better model is selected.

## Results

| | CNN | Reinforcement learning (Q-learning) |
|---|---|---|
| **Accuracy** | 100% | 70% |
| **Training time** | Very long (a couple of hours, depending on input size) | No separate training dataset needed |
| **Planning time** | Very fast (about 0.2 s) | Long, but converges for small mazes |
| **Strengths** | Fast and efficient for the cases it was trained on | No labelled dataset required; generalizes well |
| **Weaknesses** | Not versatile; needs labelled training data; computationally expensive | Can get stuck in local optima |

## Skills and tools

`Python` · `Keras / TensorFlow` · `CNN` · `Reinforcement learning` · `A* search` · `Path planning` · `Occupancy grids`

## Repository contents

All code is in `PROJECT1/`:

| File(s) | What it does |
|---|---|
| `random_maze.py`, `random_shape_maze.py` | Generate random maze datasets |
| `Streetmap.py`, `gridmap.py`, `new_york/`, `NewYork_*.png` | Convert city street maps to occupancy grids |
| `a_star.py` | A* shortest-path baseline and label generation |
| `Testing_CNN_on_City.py` | Evaluate the trained CNN on city maps |
| `cnn_model*.json`, `cnn_model.h5`, `nn_model.*` | Trained model architectures and weights |
| `mazes*.pkl`, `paths*.pkl`, `pathshortest*.pkl` | Generated mazes and path labels |
| `utils.py`, `unpickleddata*.py` | Data loading utilities |

## Context

Joint project with Vinita Narayanamurthi, M.S. Electrical Engineering, Rochester Institute of Technology. Also on [Portfolium](https://portfolium.com/entry/mobile-robot-path-planning).

## License

This project is released under the [PolyForm Noncommercial License 1.0.0](LICENSE). You may use, modify and share it for **noncommercial purposes**, including academic research, teaching and personal study. Commercial use needs separate permission from the author.

Required Notice: Copyright (c) 2020 Sriparvathi Shaji Bhattathiri

Developed together with Vinita Narayanamurthi.
