"""Shared project path helpers for portable local and public use."""

from __future__ import annotations

import os
from pathlib import Path


_SRC_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = _SRC_ROOT.parent.resolve()


def _resolve_root(env_key: str, default: Path) -> Path:
    value = os.getenv(env_key)
    if value:
        return Path(value).expanduser().resolve()
    return default.resolve()


DATA_ROOT = _resolve_root("DTA_DATA_DIR", PROJECT_ROOT / "data")
LOG_ROOT = _resolve_root("DTA_LOG_DIR", PROJECT_ROOT / "logs")
EXTERNAL_ROOT = _resolve_root("DTA_EXTERNAL_ROOT", PROJECT_ROOT.parent)
COLD_ATTACK_ROOT = _resolve_root(
    "DTA_COLD_ATTACK_ROOT",
    EXTERNAL_ROOT / "COLD-Attack",
)


def project_path(*parts: str) -> Path:
    return PROJECT_ROOT.joinpath(*parts)


def data_path(*parts: str) -> Path:
    return DATA_ROOT.joinpath(*parts)


def log_path(*parts: str) -> Path:
    return LOG_ROOT.joinpath(*parts)


def cold_attack_path(*parts: str) -> Path:
    return COLD_ATTACK_ROOT.joinpath(*parts)


RAW_DATA_DIR = data_path("raw")
ADVBENCH_PATH = data_path("raw", "advbench_100.csv")
HARMBENCH_PATH = data_path("raw", "harmBench_100.csv")
ABLATION_DIR = data_path("DTA_ablation")
COMBINED_RESULTS_DIR = data_path("combined")
CUSTOM_DATASET_DIR = data_path("custom_dataset")
DTA_MAIN_RESULTS_DIR = data_path("DTA_paper_main_results")
EVAL_RESULTS_DIR = data_path("eval")
