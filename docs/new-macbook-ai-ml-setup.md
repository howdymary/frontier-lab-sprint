# New MacBook AI/ML Setup

Use this guide to dedicate a new Apple Silicon MacBook Pro to learning artificial intelligence and machine learning.

The Mac is excellent for learning, prototyping, small model training, local inference, MLX experiments, PyTorch MPS practice, JAX basics, and paper-reproduction work. For large training runs, CUDA kernels, or multi-hour transformer experiments, use cloud GPUs/TPUs.

## 1. System Setup

Update macOS first, then install Xcode command-line tools:

```bash
xcode-select --install
```

Install Homebrew from https://brew.sh, then install core tools:

```bash
brew install git gh uv wget curl cmake pkg-config ffmpeg htop tree
```

Optional but useful:

```bash
brew install --cask visual-studio-code
brew install --cask iterm2
```

Set up Git identity:

```bash
git config --global user.name "howdymary"
git config --global user.email "YOUR_EMAIL@example.com"
```

Log in to GitHub:

```bash
gh auth login
```

## 2. Folder Layout

Create a predictable workspace:

```bash
mkdir -p ~/Projects
mkdir -p ~/ml/data ~/ml/models ~/ml/runs ~/ml/cache
```

Use:

- `~/Projects` for code repos.
- `~/ml/data` for datasets.
- `~/ml/models` for model weights.
- `~/ml/runs` for experiment logs.
- `~/ml/cache` for temporary downloads.

Keep datasets, model weights, and checkpoints out of Git.

## 3. Clone This Curriculum Repo

```bash
cd ~/Projects
git clone https://github.com/howdymary/frontier-lab-sprint.git
cd frontier-lab-sprint
```

Install and verify the repo:

```bash
./scripts/bootstrap_project.sh
```

Manual equivalent:

```bash
uv sync
uv run python scripts/check_env.py
uv run pytest
uv run python scripts/generate_addition_sample.py --n 10
uv run python scripts/simulate_lif.py
```

## 4. Main Learning Stacks

### JAX

Use JAX for the frontier-lab sprint:

- Transformer from scratch.
- Scaling-law experiments.
- Dense vs MoE analysis.
- Pallas/kernel experiments.

This repo already installs JAX, Flax, and Optax through `uv`.

### PyTorch

Use PyTorch for broad ML learning because most courses, tutorials, and papers use it.

Create a separate sandbox:

```bash
cd ~/Projects
mkdir -p ml-lab
cd ml-lab
uv init --python 3.12
uv add torch torchvision torchaudio
uv add numpy pandas matplotlib scikit-learn jupyterlab ipykernel tqdm rich
uv add datasets transformers accelerate safetensors tensorboard
uv add pytest ruff
```

Check Apple Metal/MPS support:

```bash
uv run python - <<'PY'
import torch
print("torch", torch.__version__)
print("mps built:", torch.backends.mps.is_built())
print("mps available:", torch.backends.mps.is_available())
device = "mps" if torch.backends.mps.is_available() else "cpu"
x = torch.ones(5, device=device)
print(device, x * 2)
PY
```

### MLX

Use MLX for Apple-native local model experiments:

```bash
cd ~/Projects/ml-lab
uv add mlx mlx-lm
```

Verify:

```bash
uv run python - <<'PY'
import mlx.core as mx
x = mx.ones((5,))
print(x + 1)
print("default device:", mx.default_device())
PY
```

## 5. Best Practices

Use one Python environment per project. Do not create one huge global ML environment.

Use `uv` instead of manually juggling Python packages:

```bash
uv sync
uv add package-name
uv run python script.py
```

Start every model with a tiny overfit test:

- Can it overfit 32 examples?
- Does loss go down?
- Are labels aligned?
- Are gradients nonzero?
- Does the metric match the task?

Track every experiment:

- Git commit hash.
- Dataset version.
- Model config.
- Random seed.
- Hardware used.
- Runtime.
- Final metrics.
- What changed from the previous run.

Prefer scripts once an experiment matters:

- Notebooks are good for exploration.
- Training should become `train.py`.
- Evaluation should become `eval.py`.
- Dataset code should live in `data.py`.

Keep this out of Git:

```text
.venv/
data/
models/
checkpoints/
runs/
wandb/
__pycache__/
```

## 6. What To Train Locally

Good local projects:

- Linear regression and logistic regression from scratch.
- Tiny MLP on XOR or MNIST.
- CNN on small image datasets.
- Tiny transformer on synthetic addition.
- Leaky integrate-and-fire spiking neuron simulation.
- Small LoRA/fine-tuning experiments.
- Local LLM inference and prompting with MLX.

Use cloud GPUs/TPUs for:

- Larger transformer training.
- Long fine-tuning runs.
- CUDA-specific kernels.
- Multi-GPU experiments.
- Anything that makes the Mac swap heavily.

## 7. First Weekend Plan

Day 1:

- Finish system setup.
- Clone this repo.
- Run `./scripts/bootstrap_project.sh`.
- Read `docs/lesson-plan.md`.
- Run `uv run python scripts/generate_addition_sample.py --n 10`.
- Run `uv run python scripts/simulate_lif.py`.

Day 2:

- Work through basic JAX arrays, `jit`, `grad`, and `vmap`.
- Implement a tiny MLP in JAX.
- Write a short note: "What backprop is actually doing."
- Draft the target role memo from the lesson plan.

## 8. Daily Operating Rhythm

Minimum useful day:

- 90 minutes building or training.
- 60 minutes math/JAX/PyTorch exercises.
- 45 minutes paper reading or writing.
- 20 minutes README, notes, or experiment-log cleanup.

Weekly review:

- What did I understand this week that I could not explain before?
- What model did I train?
- What broke?
- What did I measure?
- What should I cut next week to preserve focus?

