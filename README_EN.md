# Dynamic Target Attack

[中文说明](README.md)

This repository is a trimmed and sanitized research codebase for **Dynamic Target Attack (DTA)**. It keeps only the core attack runner, ablation runner, and reusable harmfulness-evaluation pipeline needed to understand and reproduce the main workflow of the simplified project.

## Highlights

- repository-relative paths instead of personal machine paths
- environment-variable-based API configuration
- a single top-level dispatcher in `src/main.py`
- a smaller file set focused on the core research workflow

## Repository Scope

This version intentionally removes legacy attacker variants, one-off analysis scripts, and baseline-specific evaluation helpers. The primary maintained code paths are:

- `src/combined_attacker.py`: combined DTA pipeline
- `src/experiments_ablation.py`: ablation experiment runner
- `src/eval/harmfulness/`: reusable harmfulness-evaluation toolkit
- `src/attacker_v3.py`: main attacker implementation used by the retained runners

## Repository Layout

```text
.
├── README.md
├── README_EN.md
├── README_zh.md
├── .env.example
├── requirements.txt
├── pyproject.toml
├── environment.yaml
├── data/
│   ├── README.md
│   └── raw/
├── scripts/
│   ├── run_ablation_A1.sh
│   ├── run_ablation_F_batch.sh
│   └── run_combined_attacker01.sh
└── src/
    ├── main.py
    ├── combined_attacker.py
    ├── experiments_ablation.py
    ├── attacker_v3.py
    └── eval/harmfulness/
```

## Main Entry Points

Use the dispatcher if you want one stable CLI:

```bash
python src/main.py combined --help
python src/main.py ablation --help
python src/main.py harmfulness-eval --help
```

You can also call the underlying modules directly:

```bash
python src/combined_attacker.py --help
python src/experiments_ablation.py --help
python src/eval/run_harmfulness_eval.py --help
```

## Installation

### Option 1: uv

```bash
uv sync
```

### Option 2: pip

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Option 3: conda

```bash
conda env create -f environment.yaml
conda activate dta
```

## Environment Variables

Copy the template first:

```bash
cp .env.example .env
```

Common variables:

- `OPENAI_API_KEY`: required for OpenAI-compatible judge / generation backends
- `OPENAI_API_BASE`: optional custom OpenAI-compatible base URL
- `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`, `DASHSCOPE_API_KEY`, `TOGETHER_API_KEY`: optional provider-specific backends
- `DTA_DATA_DIR`: override the default `./data` directory
- `DTA_LOG_DIR`: override the default `./logs` directory
- `DTA_MODEL_LLAMA3`, `DTA_MODEL_QWEN25`, `DTA_GPTFUZZ_MODEL`, etc.: override default model ids with local mirrors if needed

## Data Layout

By default, the project expects data under `data/`:

```text
data/
  raw/
    advbench_100.csv
    harmBench_100.csv
```

Expected CSV schema:

- a `goal` column, or
- a `harmful` column

Generated outputs are typically written to:

- `data/combined/`
- `data/DTA_ablation/`
- `data/eval/`
- `logs/`

See [data/README.md](data/README.md) for the expected layout.

## Common Workflows

### 1. Combined attacker

Run the core DTA pipeline:

```bash
python src/main.py combined --target-llm Llama3 --help
```

For the retained batch example:

```bash
bash scripts/run_combined_attacker01.sh
```

### 2. Ablation experiments

Run one ablation family:

```bash
python src/main.py ablation --experiment A1 --help
```

Example batch script:

```bash
bash scripts/run_ablation_A1.sh
```

### 3. Harmfulness evaluation

Run the unified harmfulness evaluation CLI:

```bash
python src/main.py harmfulness-eval --input-path path/to/results.jsonl --help
```

## Quick Sanity Checks

After installing dependencies, these commands should work:

```bash
python src/main.py --help
python src/main.py combined --help
python src/main.py ablation --help
python src/main.py harmfulness-eval --help
```

## Responsible Use

This project is intended for safety research, red teaming, defense evaluation, and reproducibility-oriented experimentation. It does not ship datasets, model weights, or private experiment outputs. Please follow your institution's policy, model provider terms, and applicable laws before running any experiment.
