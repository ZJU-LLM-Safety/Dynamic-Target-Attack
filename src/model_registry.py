"""Canonical model ids and lightweight alias resolution."""

from __future__ import annotations

import os
from typing import Dict


def _env_or_default(env_key: str, default: str) -> str:
    return os.getenv(env_key, default)


ATTACK_MODEL_PATHS: Dict[str, str] = {
    "Llama3": _env_or_default(
        "DTA_MODEL_LLAMA3",
        "meta-llama/Meta-Llama-3-8B-Instruct",
    ),
    "Qwen2.5": _env_or_default(
        "DTA_MODEL_QWEN25",
        "Qwen/Qwen2.5-7B-Instruct",
    ),
    "Mistralv0.3": _env_or_default(
        "DTA_MODEL_MISTRALV03",
        "mistralai/Mistral-7B-Instruct-v0.3",
    ),
    "Gemma7b": _env_or_default(
        "DTA_MODEL_GEMMA7B",
        "google/gemma-7b-it",
    ),
    "Vicunav1.5": _env_or_default(
        "DTA_MODEL_VICUNA15",
        "lmsys/vicuna-7b-v1.5",
    ),
}

GPTFUZZ_MODEL_PATH = _env_or_default(
    "DTA_GPTFUZZ_MODEL",
    "hubert233/GPTFuzz",
)
LLAMAGUARD3_MODEL_PATH = _env_or_default(
    "DTA_LLAMAGUARD3_MODEL",
    "meta-llama/Llama-Guard-3-8B",
)
WILDGUARD_MODEL_PATH = _env_or_default(
    "DTA_WILDGUARD_MODEL",
    "allenai/wildguard",
)
HARMBENCH_CLASSIFIER_MODEL_PATH = _env_or_default(
    "DTA_HARMBENCH_CLASSIFIER_MODEL",
    "cais/HarmBench-Llama-2-13b-cls",
)

MODEL_ALIASES: Dict[str, str] = {
    "Llama-3-8b-instruct": ATTACK_MODEL_PATHS["Llama3"],
    "llama-3-8B-Instruct": ATTACK_MODEL_PATHS["Llama3"],
    "Llama-3-8B-Instruct": ATTACK_MODEL_PATHS["Llama3"],
    "Qwen-2.5-7b": ATTACK_MODEL_PATHS["Qwen2.5"],
    "Qwen2.5-7B-Instruct": ATTACK_MODEL_PATHS["Qwen2.5"],
    "Mistral-7b": ATTACK_MODEL_PATHS["Mistralv0.3"],
    "Mistral-7B-Instruct": ATTACK_MODEL_PATHS["Mistralv0.3"],
    "Gemma-7b": ATTACK_MODEL_PATHS["Gemma7b"],
    "Vicuna-7b-v1.5": ATTACK_MODEL_PATHS["Vicunav1.5"],
    "GPTFuzz": GPTFUZZ_MODEL_PATH,
    "LlamaGuard3": LLAMAGUARD3_MODEL_PATH,
    "WildGuard": WILDGUARD_MODEL_PATH,
}


def resolve_model_name_or_path(name_or_path: str) -> str:
    if name_or_path in ATTACK_MODEL_PATHS:
        return ATTACK_MODEL_PATHS[name_or_path]
    return MODEL_ALIASES.get(name_or_path, name_or_path)
