# tinywmcraft-task

Train an **autoregressive world model** for **Craftax-Classic Pixels** using
[tinywmcraft_500k](https://huggingface.co/datasets/maharishiva/tinywmcraft_500k) (500k transitions).

## Rules

- Edit `train.py`; keep `eval.py` unchanged.
- Add training dependencies to `pyproject.toml` without changing the evaluator's pinned versions.
- Commit each configuration before training.
- Train from scratch on `data/tinywmcraft_500k/train.npz` and improve the model's evaluation scores.
- Training budget: **10 minutes wall-clock per run on 1× A100**, including model-specific preprocessing and all training.
- Log all scores, brief experiment notes, commit IDs, and measured training times in `ledger.jsonl`.
- Submit code, a checkpoint, and commands to reproduce training and evaluation.

**Evaluation** (frozen `eval.py`): 2048 windows, 8 context frames, 16 generated frames; evaluation seed 0.

**Metrics**: gFVD ↓, AF ↑.

Recommendations: start with a simple model; compare runs using the same 10-minute compute budget;
check conclusions across multiple seeds and inspect generated videos.
Codex, Claude Code, and other coding agents are allowed! Cite reused code and sources.

## Setup

```bash
uv venv --python 3.11
uv pip install -r pyproject.toml --extra cuda
source .venv/bin/activate
python eval.py --download
python eval.py
```

```bash
python eval.py --model runs/e001/checkpoint --tag e001 \
  --note "What changed and why" --gif
```
