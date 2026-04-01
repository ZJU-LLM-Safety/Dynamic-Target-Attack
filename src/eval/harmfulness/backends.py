"""Model and API backends for harmfulness evaluation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Optional

from openai import OpenAI
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from .config import ApiBackendSettings, load_api_backend_settings, parse_torch_dtype


@dataclass(frozen=True)
class GenerationOptions:
    """Shared text-generation options."""

    max_new_tokens: int = 256
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = 50
    num_return_sequences: int = 1


class BaseGenerationBackend(ABC):
    """Abstract text-generation backend."""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        *,
        options: Optional[GenerationOptions] = None,
    ) -> str:
        """Generate a single text completion for ``prompt``."""


class CallableGenerationBackend(BaseGenerationBackend):
    """Adapter that turns an arbitrary callable into a backend."""

    def __init__(
        self,
        generator: Callable[[str], str],
        *,
        name: str = "callable",
    ):
        self.generator = generator
        self.name = name

    def generate(
        self,
        prompt: str,
        *,
        options: Optional[GenerationOptions] = None,
    ) -> str:
        del options
        return self.generator(prompt)

    def __repr__(self) -> str:
        return f"CallableGenerationBackend(name={self.name})"


class OpenAICompatibleBackend(BaseGenerationBackend):
    """Backend for OpenAI-compatible chat-completions APIs."""

    def __init__(
        self,
        settings: ApiBackendSettings,
        *,
        client: Optional[OpenAI] = None,
        default_options: Optional[GenerationOptions] = None,
    ):
        self.settings = settings
        self.client = client or OpenAI(
            api_key=settings.api_key,
            base_url=settings.base_url,
        )
        self.default_options = default_options or GenerationOptions()

    @classmethod
    def from_env(
        cls,
        provider: str,
        model_name: str,
        *,
        env_file: Optional[str] = None,
        env_prefix: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = 60.0,
        default_options: Optional[GenerationOptions] = None,
    ) -> "OpenAICompatibleBackend":
        settings = load_api_backend_settings(
            provider=provider,
            model_name=model_name,
            env_file=env_file,
            env_prefix=env_prefix,
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
        )
        return cls(settings=settings, default_options=default_options)

    def generate(
        self,
        prompt: str,
        *,
        options: Optional[GenerationOptions] = None,
    ) -> str:
        options = options or self.default_options
        response = self.client.chat.completions.create(
            model=self.settings.model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=options.max_new_tokens,
            temperature=options.temperature,
            top_p=options.top_p,
            n=1,
            timeout=self.settings.timeout,
        )
        content = response.choices[0].message.content
        return content or ""

    def __repr__(self) -> str:
        return (
            "OpenAICompatibleBackend("
            f"provider={self.settings.provider}, model={self.settings.model_name})"
        )


def _move_batch_to_device(batch, device: str):
    if hasattr(batch, "to"):
        return batch.to(device)
    if isinstance(batch, dict):
        moved = {}
        for key, value in batch.items():
            moved[key] = value.to(device) if hasattr(value, "to") else value
        return moved
    return batch


def _extract_input_ids(batch):
    if isinstance(batch, dict):
        return batch["input_ids"]
    return batch.input_ids


class HuggingFaceGenerationBackend(BaseGenerationBackend):
    """Backend for local HuggingFace causal language models."""

    def __init__(
        self,
        model_name_or_path: str,
        *,
        device: str = "cpu",
        dtype: Optional[str | torch.dtype] = None,
        model=None,
        tokenizer=None,
        default_options: Optional[GenerationOptions] = None,
    ):
        self.model_name_or_path = model_name_or_path
        self.device = device
        self.dtype = parse_torch_dtype(dtype)
        self.default_options = default_options or GenerationOptions()

        self.model = model or AutoModelForCausalLM.from_pretrained(model_name_or_path)
        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(model_name_or_path)

        if hasattr(self.model, "eval"):
            self.model.eval()
        if hasattr(self.model, "to"):
            self.model = self.model.to(device)
            if self.dtype is not None:
                self.model = self.model.to(self.dtype)

        if getattr(self.tokenizer, "pad_token", None) is None:
            eos_token = getattr(self.tokenizer, "eos_token", None)
            if eos_token is not None:
                self.tokenizer.pad_token = eos_token

    def generate(
        self,
        prompt: str,
        *,
        options: Optional[GenerationOptions] = None,
    ) -> str:
        options = options or self.default_options
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = _move_batch_to_device(inputs, self.device)

        input_ids = _extract_input_ids(inputs)
        input_length = input_ids.shape[1]
        do_sample = options.temperature > 0
        generation_config = GenerationConfig(
            max_new_tokens=options.max_new_tokens,
            do_sample=do_sample,
            temperature=options.temperature if do_sample else 1.0,
            top_p=options.top_p if do_sample else 1.0,
            top_k=options.top_k if do_sample else 50,
            num_return_sequences=1,
            eos_token_id=getattr(self.tokenizer, "eos_token_id", None),
            pad_token_id=getattr(self.tokenizer, "pad_token_id", None),
        )

        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                generation_config=generation_config,
            )

        completion_ids = outputs[:, input_length:]
        decoded = self.tokenizer.batch_decode(
            completion_ids,
            skip_special_tokens=True,
        )
        return decoded[0] if decoded else ""

    def __repr__(self) -> str:
        return (
            "HuggingFaceGenerationBackend("
            f"model={self.model_name_or_path}, device={self.device})"
        )
