from __future__ import annotations

# --- migrated from src/prism/benchmarks/calvin/eval_summary.py ---
from collections.abc import Sequence
from dataclasses import asdict, dataclass, is_dataclass
import json
from pathlib import Path
from typing import Any, Mapping

from prism.eval.runner import build_run_metadata


@dataclass(frozen=True)
class SequenceResult:
    sequence_id: int
    initial_state: str
    subtasks: list[str]
    successful_subtasks: int
    success: bool
    decision_steps: int
    control_steps: int
    failed_subtask: str = ""
    failure_reason: str = ""
    video_paths: list[str] | None = None


def summarize_sequence_results(results: Sequence[SequenceResult | Mapping[str, Any]]) -> dict[str, Any]:
    sequences = [_sequence_to_dict(result) for result in results]
    total_sequences = len(sequences)
    successful_sequences = sum(1 for sequence in sequences if sequence["success"])
    successful_counts = [int(sequence["successful_subtasks"]) for sequence in sequences]
    total_subtasks = sum(len(sequence["subtasks"]) for sequence in sequences)
    successful_subtasks = sum(successful_counts)
    return {
        "total_sequences": total_sequences,
        "successful_sequences": successful_sequences,
        "failed_sequences": total_sequences - successful_sequences,
        "sequence_success_rate": successful_sequences / total_sequences if total_sequences else 0.0,
        "average_successful_subtasks": _mean(successful_counts),
        "total_subtasks": total_subtasks,
        "successful_subtasks": successful_subtasks,
        "subtask_success_rate": successful_subtasks / total_subtasks if total_subtasks else 0.0,
        "chain_success_rates": _chain_success_rates(successful_counts, max_chain_length=_max_sequence_length(sequences)),
        "average_decision_steps": _mean([int(sequence["decision_steps"]) for sequence in sequences]),
        "average_control_steps": _mean([int(sequence["control_steps"]) for sequence in sequences]),
        "task_info": _task_info(sequences),
        "successful_sequence_ids": [sequence["sequence_id"] for sequence in sequences if sequence["success"]],
    }


def write_result_summary(
    path: str | Path,
    *,
    config: Any,
    results: Sequence[SequenceResult | Mapping[str, Any]],
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    result_path = Path(path).expanduser()
    result_path.parent.mkdir(parents=True, exist_ok=True)
    sequences = [_sequence_to_dict(result) for result in results]
    payload = {
        "config": _serialize_config(config),
        "metadata": dict(metadata) if metadata is not None else build_run_metadata(),
        "summary": summarize_sequence_results(sequences),
        "sequences": sequences,
    }
    with result_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return result_path


def _chain_success_rates(successful_counts: Sequence[int], *, max_chain_length: int) -> dict[str, float]:
    if not successful_counts:
        return {str(index): 0.0 for index in range(1, max_chain_length + 1)}
    return {
        str(index): sum(1 for count in successful_counts if count >= index) / len(successful_counts)
        for index in range(1, max_chain_length + 1)
    }


def _task_info(sequences: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, int | float]]:
    task_success: dict[str, int] = {}
    task_total: dict[str, int] = {}
    for sequence in sequences:
        successful_subtasks = int(sequence["successful_subtasks"])
        subtasks = [str(task) for task in sequence["subtasks"]]
        for index, subtask in enumerate(subtasks):
            task_total[subtask] = task_total.get(subtask, 0) + 1
            if index < successful_subtasks:
                task_success[subtask] = task_success.get(subtask, 0) + 1
            else:
                task_success.setdefault(subtask, 0)
    return {
        task: {
            "success": task_success.get(task, 0),
            "total": total,
            "success_rate": task_success.get(task, 0) / total if total else 0.0,
        }
        for task, total in sorted(task_total.items())
    }


def _max_sequence_length(sequences: Sequence[Mapping[str, Any]]) -> int:
    if not sequences:
        return 5
    return max(len(sequence["subtasks"]) for sequence in sequences)


def _mean(values: Sequence[int]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _sequence_to_dict(result: SequenceResult | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(result, SequenceResult):
        payload = asdict(result)
    else:
        payload = dict(result)
    subtasks = payload.get("subtasks") or []
    if not isinstance(subtasks, Sequence) or isinstance(subtasks, (str, bytes, bytearray)):
        raise ValueError("CALVIN sequence result subtasks must be a list")
    successful_subtasks = int(payload["successful_subtasks"])
    payload["sequence_id"] = int(payload["sequence_id"])
    payload["initial_state"] = str(payload.get("initial_state") or "")
    payload["subtasks"] = [str(task) for task in subtasks]
    payload["successful_subtasks"] = successful_subtasks
    payload["success"] = bool(payload.get("success", successful_subtasks >= len(payload["subtasks"])))
    payload["decision_steps"] = int(payload["decision_steps"])
    payload["control_steps"] = int(payload["control_steps"])
    payload["failed_subtask"] = str(payload.get("failed_subtask") or "")
    payload["failure_reason"] = str(payload.get("failure_reason") or "")
    payload["video_paths"] = [str(path) for path in (payload.get("video_paths") or [])]
    return payload


def _serialize_config(config: Any) -> dict[str, Any]:
    if is_dataclass(config):
        return asdict(config)
    if isinstance(config, Mapping):
        return dict(config)
    if hasattr(config, "__dict__"):
        return {
            key: value
            for key, value in vars(config).items()
            if not key.startswith("_")
        }
    return {"repr": repr(config)}

