"""Configuration helpers for harmfulness evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional
import os

import torch
from dotenv import dotenv_values


_PACKAGE_PATH = Path(__file__).resolve()
_SRC_ROOT = _PACKAGE_PATH.parents[2]
_PROJECT_ROOT = _PACKAGE_PATH.parents[3]
_DEFAULT_ENV_SEARCH_ORDER = (
    Path.cwd() / ".env",
    _PROJECT_ROOT / ".env",
    _SRC_ROOT / ".env",
)

_DEFAULT_PROVIDER_BASE_URLS = {
    "dashscope": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "deepseek": "https://api.deepseek.com/v1",
}


@dataclass(frozen=True)
class ApiBackendSettings:
    """Resolved API configuration for an OpenAI-compatible backend."""

    provider: str
    model_name: str
    api_key: str
    base_url: Optional[str] = None
    timeout: float = 60.0
    env_prefix: Optional[str] = None
    env_file: Optional[Path] = None


def find_env_file(env_file: Optional[os.PathLike[str] | str] = None) -> Optional[Path]:
    """Locate a usable .env file.

    Search order:
    1. Explicit ``env_file``.
    2. ``HARMFULNESS_EVAL_ENV_FILE`` environment variable.
    3. ``.env`` under the current working directory.
    4. Project-root ``.env``.
    5. ``src/.env`` for backward compatibility with the current repo.
    """

    if env_file is not None:
        candidate = Path(env_file).expanduser().resolve()
        if not candidate.exists():
            raise FileNotFoundError(f".env file not found: {candidate}")
        return candidate

    env_override = os.getenv("HARMFULNESS_EVAL_ENV_FILE")
    if env_override:
        candidate = Path(env_override).expanduser().resolve()
        if not candidate.exists():
            raise FileNotFoundError(f".env file not found: {candidate}")
        return candidate

    seen = set()
    for candidate in _DEFAULT_ENV_SEARCH_ORDER:
        resolved = candidate.expanduser().resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.exists():
            return resolved
    return None


def read_env_values(
    env_file: Optional[os.PathLike[str] | str] = None,
) -> Dict[str, str]:
    """Read key/value pairs from the discovered .env file and ``os.environ``.

    Values from ``os.environ`` take precedence over values in the .env file.
    """

    resolved_env = find_env_file(env_file)
    values: Dict[str, str] = {}

    if resolved_env is not None:
        for key, value in dotenv_values(resolved_env).items():
            if value is not None:
                values[key] = value

    for key, value in os.environ.items():
        values[key] = value

    return values


def _build_env_prefix(provider: str, env_prefix: Optional[str] = None) -> str:
    prefix = env_prefix or provider
    normalized = prefix.replace("-", "_").replace(" ", "_").upper()
    if not normalized:
        raise ValueError("env_prefix and provider cannot both be empty.")
    return normalized


def load_api_backend_settings(
    provider: str,
    model_name: str,
    *,
    env_file: Optional[os.PathLike[str] | str] = None,
    env_prefix: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    timeout: float = 60.0,
) -> ApiBackendSettings:
    """Resolve API backend settings from a .env file and runtime overrides."""

    prefix = _build_env_prefix(provider, env_prefix)
    env_values = read_env_values(env_file)

    resolved_api_key = api_key or env_values.get(f"{prefix}_API_KEY")
    if not resolved_api_key:
        raise ValueError(
            f"Missing API key for provider '{provider}'. Expected "
            f"'{prefix}_API_KEY' in .env or environment variables."
        )

    resolved_base_url = (
        base_url
        or env_values.get(f"{prefix}_API_BASE")
        or env_values.get(f"{prefix}_BASE_URL")
        or _DEFAULT_PROVIDER_BASE_URLS.get(provider.lower())
    )

    return ApiBackendSettings(
        provider=provider,
        model_name=model_name,
        api_key=resolved_api_key,
        base_url=resolved_base_url,
        timeout=timeout,
        env_prefix=prefix,
        env_file=find_env_file(env_file),
    )


def parse_torch_dtype(dtype: Optional[str | torch.dtype]) -> Optional[torch.dtype]:
    """Parse string dtypes into ``torch.dtype`` values."""

    if dtype is None or isinstance(dtype, torch.dtype):
        return dtype

    normalized = dtype.strip().lower()
    mapping = {
        "float": torch.float32,
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "half": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    if normalized not in mapping:
        raise ValueError(
            "Unsupported dtype. Expected one of: float32, float16, bfloat16."
        )
    return mapping[normalized]
