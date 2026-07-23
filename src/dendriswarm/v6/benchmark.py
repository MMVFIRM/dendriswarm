from __future__ import annotations

import math
from typing import Any

import numpy as np

from dendriswarm.core.crypto import content_hash
from dendriswarm.v6.native10 import Native10Dendritron, encode_array

BASELINE_FORMAT = "dendriswarm.external-baseline-reference.v1"
EVALUATION_FORMAT = "dendriswarm.native10-evaluation.v1"
COMPARISON_FORMAT = "dendriswarm.native10-baseline-comparison.v1"


def _hash_without_sha(value: dict[str, Any]) -> str:
    return content_hash({key: item for key, item in value.items() if key != "sha256"})


def _require_sha256(value: str, field: str) -> str:
    text = str(value).lower()
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ValueError(f"{field} must be a lowercase SHA-256 hex digest")
    return text


def make_baseline_reference(
    *,
    dataset: str,
    split: str,
    metric: str,
    value: float,
    model: str,
    source: str,
    evidence_sha256: str,
    higher_is_better: bool = True,
    notes: str = "",
) -> dict[str, Any]:
    """Create an import-only reference to an already-established baseline.

    This format deliberately stores no baseline trainer or model weights.  The
    evidence hash must identify the external report, log, or artifact from which
    the value was taken.
    """
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError("baseline value must be finite")
    for field, text, limit in (
        ("dataset", dataset, 256), ("split", split, 128), ("metric", metric, 128),
        ("model", model, 256), ("source", source, 1024), ("notes", notes, 4096),
    ):
        if not str(text).strip() or len(str(text)) > limit:
            if field == "notes" and not str(text):
                continue
            raise ValueError(f"{field} is empty or too long")
    artifact: dict[str, Any] = {
        "format": BASELINE_FORMAT,
        "dataset": str(dataset),
        "split": str(split),
        "metric": str(metric),
        "value": numeric,
        "higher_is_better": bool(higher_is_better),
        "model": str(model),
        "source": str(source),
        "evidence_sha256": _require_sha256(evidence_sha256, "evidence_sha256"),
        "notes": str(notes),
        "includes_training_code": False,
        "includes_model_weights": False,
    }
    artifact["sha256"] = _hash_without_sha(artifact)
    return artifact


def validate_baseline_reference(artifact: dict[str, Any]) -> dict[str, Any]:
    if artifact.get("format") != BASELINE_FORMAT:
        raise ValueError("unsupported baseline reference format")
    rebuilt = make_baseline_reference(
        dataset=artifact["dataset"], split=artifact["split"], metric=artifact["metric"],
        value=artifact["value"], model=artifact["model"], source=artifact["source"],
        evidence_sha256=artifact["evidence_sha256"],
        higher_is_better=artifact.get("higher_is_better", True), notes=artifact.get("notes", ""),
    )
    if rebuilt != artifact:
        raise ValueError("baseline reference body or hash is inconsistent")
    return rebuilt


def evaluate_checkpoint(
    model: Native10Dendritron,
    inputs: np.ndarray,
    labels: np.ndarray,
    *,
    dataset: str,
    split: str,
    source: str,
) -> dict[str, Any]:
    x = np.asarray(inputs, dtype=np.float32)
    y = np.asarray(labels, dtype=np.int64)
    if x.ndim != 2 or x.shape[1] != model.config.input_width or not len(x):
        raise ValueError("evaluation inputs have the wrong shape")
    if y.ndim != 1 or len(y) != len(x):
        raise ValueError("evaluation labels are empty or misaligned")
    if not np.isfinite(x).all() or np.any(y < 0) or np.any(y >= model.config.classes):
        raise ValueError("evaluation data contains invalid values")
    predictions = model.predict(x)
    correct = int((predictions == y).sum())
    counts = np.bincount(y, minlength=model.config.classes).astype(int)
    correct_by_class = np.bincount(y[predictions == y], minlength=model.config.classes).astype(int)
    data_commitment = content_hash({"inputs": encode_array(x), "labels": encode_array(y)})
    report: dict[str, Any] = {
        "format": EVALUATION_FORMAT,
        "model_root": model.root,
        "parameter_count": model.parameter_count,
        "dataset": str(dataset),
        "split": str(split),
        "source": str(source),
        "data_sha256": data_commitment,
        "metric": "accuracy",
        "sample_count": int(len(y)),
        "correct": correct,
        "value": correct / len(y),
        "samples_by_class": counts.tolist(),
        "correct_by_class": correct_by_class.tolist(),
        "training_performed": False,
    }
    report["sha256"] = _hash_without_sha(report)
    return report


def validate_evaluation_report(report: dict[str, Any]) -> dict[str, Any]:
    if report.get("format") != EVALUATION_FORMAT:
        raise ValueError("unsupported Native10 evaluation report format")
    if report.get("sha256") != _hash_without_sha(report):
        raise ValueError("Native10 evaluation report hash mismatch")
    _require_sha256(report["model_root"], "model_root")
    _require_sha256(report["data_sha256"], "data_sha256")
    if report.get("metric") != "accuracy" or report.get("training_performed") is not False:
        raise ValueError("evaluation report claim boundary is invalid")
    sample_count = int(report["sample_count"])
    correct = int(report["correct"])
    if sample_count < 1 or not 0 <= correct <= sample_count:
        raise ValueError("evaluation counts are invalid")
    if abs(float(report["value"]) - correct / sample_count) > 1e-15:
        raise ValueError("evaluation accuracy is inconsistent with counts")
    if sum(int(value) for value in report["samples_by_class"]) != sample_count:
        raise ValueError("evaluation class counts are inconsistent")
    if len(report["samples_by_class"]) != len(report["correct_by_class"]):
        raise ValueError("evaluation class vectors are misaligned")
    if any(not 0 <= int(c) <= int(n) for c, n in zip(report["correct_by_class"], report["samples_by_class"], strict=True)):
        raise ValueError("evaluation class correctness is invalid")
    return dict(report)


def compare_with_baseline(evaluation: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    evaluation = validate_evaluation_report(evaluation)
    baseline = validate_baseline_reference(baseline)
    if evaluation["dataset"] != baseline["dataset"] or evaluation["split"] != baseline["split"]:
        raise ValueError("evaluation and baseline refer to different dataset splits")
    if evaluation["metric"] != baseline["metric"]:
        raise ValueError("evaluation and baseline metrics differ")
    difference = float(evaluation["value"]) - float(baseline["value"])
    favorable = difference >= 0 if baseline["higher_is_better"] else difference <= 0
    report: dict[str, Any] = {
        "format": COMPARISON_FORMAT,
        "evaluation_sha256": evaluation["sha256"],
        "baseline_sha256": baseline["sha256"],
        "model_root": evaluation["model_root"],
        "dataset": evaluation["dataset"],
        "split": evaluation["split"],
        "metric": evaluation["metric"],
        "swarm_value": float(evaluation["value"]),
        "baseline_value": float(baseline["value"]),
        "difference": difference,
        "meets_or_exceeds_baseline": bool(favorable),
        "baseline_training_included": False,
    }
    report["sha256"] = _hash_without_sha(report)
    return report
