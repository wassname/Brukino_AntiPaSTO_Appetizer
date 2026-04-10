# Brukino's AntiPaSTO Appetizer

Testing whether the Frenet-Serret extrinsic curvature ($\kappa$) of a model's hidden state trajectory can predict structural shifts in the model's persona or criterion (e.g., eval-awareness, preference changes) without needing behavioral labels.

## Setup

This project is managed by `uv`.

### Requirements
- Python 3.11+
- `uv` installed

### Installation

1. Clone this repository.
2. The dependencies are specified in `pyproject.toml` and lockfile. `uv` handles them automatically.

To sync the environment:
```bash
uv sync
```

## Running the Experiment

You can explore the experiment either via the Jupyter Notebook or by running the generated Python script directly.

### Via Notebook
To spin up Jupyter Lab/Notebooks:
```bash
uv run jupyter notebook
```
Then open `experiment.ipynb` and run the cells.

### Via Script
To run the python script directly (converted from the notebook via `jupytext`):
```bash
uv run python experiment.py
```
*(Note: Ensure you have your X11/Wayland display setup to see the matplotlib plot, or run with `MPLBACKEND=Agg` if headless).*

## How it Works

We use the **Guided CoT trick**:
1. Generate ~32 tokens of Chain of Thought reasoning (`n_think`) using greedy decoding.
2. Force the model to transition to an answer by appending a specific suffix (`\nI should answer now.\nMy choice: **`).
3. Run a single forward pass over the full sequence.
4. Extract the final-layer hidden states during the reasoning step.
5. Calculate the Frenet-Serret extrinsic curvature $\kappa(t) = \|\gamma''(t)\| / \|\gamma'(t)\|^3$ of these states using finite differences.
6. Compare $\kappa(t)$ between opposite personas ("honest" vs. "dishonest" vs. "neutral baseline") on daily dilemmas.

## Model
The default script uses `Qwen/Qwen2.5-0.5B-Instruct` as it fits comfortably on small GPUs or CPUs. You can easily scale this up by changing `MODEL_NAME` in `experiment.ipynb`/`experiment.py`.