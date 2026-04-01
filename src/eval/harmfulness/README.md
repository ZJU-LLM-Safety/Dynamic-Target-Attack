# Harmfulness Evaluation Framework

This folder contains a standalone harmfulness-evaluation framework for `src/eval`.

Chinese documentation:

- `README_zh.md`

Main pieces:

- `config.py`: load API provider settings from `.env`
- `backends.py`: pluggable generation backends for local HuggingFace models, OpenAI-compatible APIs, or arbitrary callables
- `evaluators.py`: heuristic, LLM-as-a-judge, GPTFuzzer, and LlamaGuard3 evaluators
- `pipeline.py`: dataset loading, JSONL output, and aggregate unsafe-rate summaries

Recommended `.env` locations:

1. project root: `.env`
2. current working directory: `.env`
3. `src/.env` for backward compatibility with the current repo layout

Minimal usage:

```python
from src.eval.harmfulness import (
    OpenAICompatibleBackend,
    build_openai_policy_evaluator,
    HarmfulnessEvaluationPipeline,
)

backend = OpenAICompatibleBackend.from_env(
    provider="openai",
    model_name="gpt-4o-mini",
)
evaluator = build_openai_policy_evaluator(backend, name="gpt4_policy")
pipeline = HarmfulnessEvaluationPipeline({"gpt4_policy": evaluator})
```

CLI usage:

```bash
python -m src.eval.harmfulness \
  --input-path data/sample.jsonl \
  --output-path data/sample_eval.jsonl \
  --evaluators refusal,openai_policy \
  --evaluation-mode prompt_response \
  --llm-backend api \
  --llm-provider openai \
  --llm-model gpt-4o-mini
```

Or call the wrapper script directly:

```bash
python src/eval/run_harmfulness_eval.py \
  --input-path data/sample.jsonl \
  --evaluators refusal,gptfuzzer
```

Create a `.env` file from `.env.example` and keep secrets out of git.
