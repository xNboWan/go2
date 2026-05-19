---

# Unitree Go2 RL Training with Isaac Lab

[![Isaac Lab](https://img.shields.io/badge/Powered%20by-Isaac%20Lab-blue.svg)](https://isaac-sim.github.io/IsaacLab/)
[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)

A standalone repository and template for training the Unitree Go2 quadruped robot using reinforcement learning based on **Isaac Lab**. This project allows you to develop in an isolated environment, outside of the core Isaac Lab repository, while leveraging its powerful simulation capabilities.

## 🌟 Features
- **Out-of-Tree Extension:** Clean and isolated development environment based on Isaac Lab.
- **Reinforcement Learning:** Integrated with `rsl_rl` for fast PPO training.
- **Sim-to-Sim Verification:** MuJoCo deployment scripts to verify trained policies outside of Isaac Sim.

---

## 🛠️ Installation

### 1. Prerequisites
Install Isaac Lab by following their [official installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html). We highly recommend using a `conda` or `uv` virtual environment, as it simplifies calling Python scripts from the terminal.

### 2. Clone the Repository
Clone this repository separately from your Isaac Lab installation (i.e., outside the `IsaacLab` directory):

```bash
git clone https://github.com/xNboWan/go2.git
cd go2

```

### 3. Install the Extension

Using the Python interpreter that has Isaac Lab installed, install this library in editable mode:

```bash
# Note: Use 'isaaclab.sh -p' instead of 'python' if Isaac Lab is not installed in a Python venv or conda env.
python -m pip install -e source/go2

```

### 4. Verify Installation

Check if the environments are correctly registered by listing available tasks:

```bash
python scripts/list_envs.py

```

*(Note: If your task name changes in the future, you may need to update the search pattern in `scripts/list_envs.py` so it appears in the list.)*

---

## 🚀 Usage

### Training and Playing (`rsl_rl`)

To train the Go2 robot, run the training script and specify your target task name:

```bash
python scripts/rsl_rl/train.py --task=<TASK_NAME>

```

To play/evaluate a trained policy:

```bash
python scripts/rsl_rl/play.py --task=<TASK_NAME>

```

### Debugging with Dummy Agents

You can run tasks with dummy agents to ensure your environments and observations are configured correctly before running full RL training:

* **Zero-action agent:**
```bash
python scripts/zero_agent.py --task=<TASK_NAME>

```


* **Random-action agent:**
```bash
python scripts/random_agent.py --task=<TASK_NAME>

```



---

## 🤖 Sim-to-Sim Verification (MuJoCo)

The `Deploy` folder provides simple scripts to test your trained models (exported as ONNX policies) in a lightweight MuJoCo simulation. This is a crucial step to verify your policy's robustness before real-world deployment.

```bash
cd Deploy
# For rough terrain testing
python play_mujoco.py

# For flat terrain testing
python play_mujoco_flat.py

```

*(Ensure you have your trained `policy.onnx` correctly placed in the `policy/` directory and that MuJoCo is installed in your Python environment).*

---

## 💻 IDE Setup (VSCode)

To enable intelligent code completion and module indexing in VSCode:

1. Press `Ctrl+Shift+P` and select **`Tasks: Run Task`**.
2. Run **`setup_python_env`** from the drop-down menu.
3. When prompted, enter the absolute path to your Isaac Sim installation.

This will generate a `.python.env` file in the `.vscode` directory containing the paths to all Omniverse and Isaac Sim extensions.

### Setup as an Omniverse Extension (Optional)

We provide an example UI extension (`source/go2/go2/ui_extension_example.py`) that loads inside Isaac Sim.

1. Open Isaac Sim and go to `Window` -> `Extensions`.
2. Click the **Gear/Hamburger Icon** -> `Settings`.
3. Under **Extension Search Paths**, add the absolute path to the `source` directory of this repository.
4. Refresh, search for your extension under the `Third Party` category, and toggle it to enable.

---

## 🧹 Code Formatting

We use `pre-commit` to ensure consistent code styling. To set it up:

```bash
pip install pre-commit
pre-commit install

```

To run formatting manually on all files:

```bash
pre-commit run --all-files

```

---

## 🐛 Troubleshooting

### Pylance Cannot Correctly Resolve Python Code

If VsCode's Pylance cannot find your modules, update the `pyproject.toml` file in the root directory to include your specific paths. Replace the paths below with the absolute paths on your machine:

```toml
[tool.pyright]
include = ["source", "scripts"]
exclude = [
    "**/__pycache__",
    "**/_isaac_sim",
    "**/docs",
    "**/logs",
    ".git",
    ".vscode",
    "**/node_modules",  
    "**/.*"            
]
extraPaths = [
    # Replace with your actual project path
    "/path/to/your/workplace/source/<your-project-name>",
    
    # Replace with your actual Isaac Lab path
    "/path/to/IsaacLab/source/isaaclab", 
    "/path/to/IsaacLab/source/isaaclab_assets",
    "/path/to/IsaacLab/source/isaaclab_rl",
    "/path/to/IsaacLab/source/isaaclab_tasks",
    
    # Add Omniverse python packages (replace with your conda env path)
    "/path/to/miniconda3/envs/isaaclab/lib/python3.11/site-packages/extscache/omni.kit.*",
    "/path/to/miniconda3/envs/isaaclab/lib/python3.11/site-packages/extscache/omni.graph.*",
    "/path/to/miniconda3/envs/isaaclab/lib/python3.11/site-packages/extscache/omni.services.*"
]

