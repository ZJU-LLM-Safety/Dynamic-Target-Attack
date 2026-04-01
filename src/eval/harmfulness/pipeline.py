"""Dataset-oriented harmfulness evaluation pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence
import json

from .evaluators import BaseHarmfulnessEvaluator, HarmfulnessScore


def load_jsonl(path: str | Path) -> list[Dict[str, Any]]:
    """Load a JSONL file into memory."""

    records: list[Dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                records.append(json.loads(stripped))
    return records


def save_jsonl(records: Iterable[Dict[str, Any]], path: str | Path) -> None:
    """Save records to a JSONL file."""

    with Path(path).open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


@dataclass
class HarmfulnessSample:
    """Single sample to evaluate."""

    response: str
    prompt: Optional[str] = None
    sample_id: Optional[Any] = None
    record: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_record(
        cls,
        record: Dict[str, Any],
        *,
        response_key: str = "response",
        prompt_key: Optional[str] = "prompt",
        id_key: Optional[str] = "id",
    ) -> "HarmfulnessSample":
        if response_key not in record:
            raise KeyError(f"Missing response key: {response_key}")

        prompt = record.get(prompt_key) if prompt_key else None
        sample_id = record.get(id_key) if id_key else None
        return cls(
            response=record[response_key],
            prompt=prompt,
            sample_id=sample_id,
            record=dict(record),
        )


@dataclass
class SampleEvaluation:
    """Evaluation output for one sample."""

    sample: HarmfulnessSample
    results: Dict[str, HarmfulnessScore]

    def to_dict(self, *, include_raw_output: bool = False) -> Dict[str, Any]:
        payload = dict(self.sample.record)
        if "response" not in payload:
            payload["response"] = self.sample.response
        if self.sample.prompt is not None and "prompt" not in payload:
            payload["prompt"] = self.sample.prompt
        if self.sample.sample_id is not None and "id" not in payload:
            payload["id"] = self.sample.sample_id

        for evaluator_name, result in self.results.items():
            payload[f"{evaluator_name}_score"] = result.score
            payload[f"is_unsafe_{evaluator_name}"] = result.unsafe
            if result.details:
                payload[f"{evaluator_name}_details"] = result.details
            if include_raw_output:
                payload[f"{evaluator_name}_raw_output"] = result.raw_output
        return payload


@dataclass
class EvaluationSummary:
    """Aggregate summary across a dataset."""

    total: int
    unsafe_counts: Dict[str, int]
    unsafe_rates: Dict[str, float]


class HarmfulnessEvaluationPipeline:
    """Pipeline that coordinates multiple harmfulness evaluators."""

    def __init__(
        self,
        evaluators: Optional[Dict[str, BaseHarmfulnessEvaluator]] = None,
    ):
        self.evaluators = evaluators or {}

    def add_evaluator(self, evaluator: BaseHarmfulnessEvaluator) -> None:
        self.evaluators[evaluator.name] = evaluator

    def get_evaluator_names(
        self,
        evaluator_names: Optional[Sequence[str]] = None,
    ) -> list[str]:
        names = list(evaluator_names) if evaluator_names is not None else list(self.evaluators)
        missing = [name for name in names if name not in self.evaluators]
        if missing:
            raise KeyError(f"Unknown evaluator(s): {', '.join(missing)}")
        return names

    def evaluate_sample(
        self,
        sample: HarmfulnessSample,
        *,
        evaluator_names: Optional[Sequence[str]] = None,
    ) -> SampleEvaluation:
        results: Dict[str, HarmfulnessScore] = {}
        for evaluator_name in self.get_evaluator_names(evaluator_names):
            evaluator = self.evaluators[evaluator_name]
            results[evaluator_name] = evaluator.evaluate(
                sample.response,
                prompt=sample.prompt,
            )
        return SampleEvaluation(sample=sample, results=results)

    def evaluate_records(
        self,
        records: Iterable[Dict[str, Any]],
        *,
        response_key: str = "response",
        prompt_key: Optional[str] = "prompt",
        id_key: Optional[str] = "id",
        evaluator_names: Optional[Sequence[str]] = None,
    ) -> list[SampleEvaluation]:
        evaluations = []
        for record in records:
            sample = HarmfulnessSample.from_record(
                record,
                response_key=response_key,
                prompt_key=prompt_key,
                id_key=id_key,
            )
            evaluations.append(
                self.evaluate_sample(sample, evaluator_names=evaluator_names)
            )
        return evaluations

    def evaluate_jsonl(
        self,
        input_path: str | Path,
        *,
        output_path: Optional[str | Path] = None,
        response_key: str = "response",
        prompt_key: Optional[str] = "prompt",
        id_key: Optional[str] = "id",
        evaluator_names: Optional[Sequence[str]] = None,
        include_raw_output: bool = False,
    ) -> list[SampleEvaluation]:
        records = load_jsonl(input_path)
        evaluations = self.evaluate_records(
            records,
            response_key=response_key,
            prompt_key=prompt_key,
            id_key=id_key,
            evaluator_names=evaluator_names,
        )
        if output_path is not None:
            save_jsonl(
                [
                    evaluation.to_dict(include_raw_output=include_raw_output)
                    for evaluation in evaluations
                ],
                output_path,
            )
        return evaluations

    def summarize(
        self,
        evaluations: Sequence[SampleEvaluation],
        *,
        evaluator_names: Optional[Sequence[str]] = None,
    ) -> EvaluationSummary:
        names = self.get_evaluator_names(evaluator_names)
        total = len(evaluations)
        unsafe_counts = {name: 0 for name in names}

        for evaluation in evaluations:
            for evaluator_name in names:
                result = evaluation.results.get(evaluator_name)
                if result is not None and result.unsafe:
                    unsafe_counts[evaluator_name] += 1

        unsafe_rates = {
            name: (unsafe_counts[name] / total if total else 0.0) for name in names
        }
        return EvaluationSummary(
            total=total,
            unsafe_counts=unsafe_counts,
            unsafe_rates=unsafe_rates,
        )
