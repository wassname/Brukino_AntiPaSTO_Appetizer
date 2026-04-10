# Brukino's AntiPaSTO Appetizer

Testing whether the Frenet-Serret extrinsic curvature ($\kappa$) of a model's hidden state trajectory can predict structural shifts in the model's persona or criterion (e.g., eval-awareness, preference changes) without needing behavioral labels.

## Concepts & Motivation

- **Guided Chain-of-Thought (CoT) with Logprobs:** Standard teacher-forced evaluation only measures the effect of an intervention on a single token, missing how the reasoning process itself changes. Full on-policy generation captures reasoning but is slow and hard to parse. The *Guided CoT* trick strikes a balance: we let the model generate a short reasoning trace (~32 tokens) greedily, then append a fixed suffix (e.g., `\nI should answer now.\nMy choice: **`) to force a decision. By running a single forward pass over this combined sequence, we extract both the hidden state trajectory of the reasoning *and* calibrated log-probabilities (`log P(Yes) - log P(No)`) at the final position. This provides a clean, bounded uncertainty estimate while capturing how personas or interventions alter the actual reasoning path.
- **Daily Dilemmas (Self-Honesty Subset):** The dataset used here comes from `wassname/daily_dilemmas-self-honesty`, originally adapted from the Reddit *AmITheAsshole* subreddit. These are 1,360 unseen moral dilemmas where honesty explicitly conflicts with other values (like kindness or loyalty). Simple prompting (e.g., "You are honest") often struggles to steer models reliably in these complex, out-of-distribution format shifts. By testing opposite personas on these dilemmas, we create a challenging environment to observe if structural shifts in reasoning (captured by $\kappa$) correlate with actual preference flipping.

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