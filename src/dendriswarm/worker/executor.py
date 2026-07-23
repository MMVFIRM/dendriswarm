from __future__ import annotations

import platform
from contextlib import nullcontext
from typing import Any

import numpy as np
from threadpoolctl import threadpool_limits

from dendriswarm.core.models import TaskKind
from dendriswarm.v5.native10 import Native10Dendritron, execute_mutation, verify_mutation_full
from dendriswarm.v5.validation import decode_global_validation_artifact
from dendriswarm.v6.native10 import (
    Native10Dendritron as Native10V6Dendritron,
    decode_array as decode_v6_array, decode_training_tensor,
    execute_mutation as execute_v6_mutation,
    verify_mutation_full as verify_v6_mutation_full,
)
from dendriswarm.v6.validation import decode_global_validation_artifact as decode_v6_global_validation_artifact
from dendriswarm.tissues.reference import (
    ReferenceDendritron,
    TissueConfig,
    artifact_hash,
    dataset_split,
    dataset_split_budgeted,
    train_and_validation,
)


def execute_task(kind: TaskKind, payload: dict[str, Any], cpu_threads: int | None = None) -> dict[str, Any]:
    """Execute an approved built-in task under an optional local thread budget."""
    dataset = payload.get("_dataset")
    artifact = payload.get("_artifact")
    limiter = threadpool_limits(limits=max(1, int(cpu_threads))) if cpu_threads else nullcontext()

    with limiter:
        if kind == TaskKind.EXPLORATION:
            if dataset is None:
                raise ValueError("exploration task is missing its dataset")
            cfg = TissueConfig(**payload["config"])
            sample_budget = payload.get("sample_budget")
            subset_seed = int(payload.get("subset_seed", cfg.seed))
            x_train, y_train = dataset_split_budgeted(dataset, "train", sample_budget, subset_seed)
            x_val, y_val = dataset_split_budgeted(dataset, "validation", sample_budget, subset_seed + 1)
            model = ReferenceDendritron.train(x_train, y_train, cfg)
            validation_predictions = model.predict(x_val)
            validation_correct = int((validation_predictions == y_val).sum())
            output = {
                "config": cfg.as_dict(),
                "validation_accuracy": validation_correct / len(y_val),
                "sample_count": int(len(y_val)),
                "correct_count": validation_correct,
            }

        elif kind == TaskKind.TRAINING:
            if dataset is None:
                raise ValueError("training task is missing its dataset")
            cfg = TissueConfig(**payload["config"])
            x_train, y_train = train_and_validation(dataset)
            model = ReferenceDendritron.train(x_train, y_train, cfg)
            tissue = model.artifact(cfg.as_dict(), payload["dataset_hash"])
            train_predictions = model.predict(x_train)
            train_correct = int((train_predictions == y_train).sum())
            output = {
                "artifact": tissue,
                "train_accuracy": train_correct / len(y_train),
                "sample_count": int(len(y_train)),
                "correct_count": train_correct,
            }

        elif kind == TaskKind.VERIFICATION:
            if dataset is None or artifact is None:
                raise ValueError("verification task is missing data or artifact")
            model = ReferenceDendritron.from_artifact(artifact)
            x_test, y_test = dataset_split(dataset, "test")
            predictions = model.predict(x_test)
            correct = int((predictions == y_test).sum())
            output = {
                "artifact_hash": artifact_hash(artifact),
                "test_accuracy": correct / len(y_test),
                "sample_count": int(len(y_test)),
                "correct_count": correct,
            }

        elif kind == TaskKind.DENDRITRON_MUTATION:
            train_y = np.asarray(payload["train_labels"], dtype=np.int64)
            if payload.get("engine") == "dendriswarm.native10-trainable.v6":
                raw_train = payload["train_data"]
                train_data = decode_training_tensor(raw_train)
                output = execute_v6_mutation(
                    payload["bundle"], train_data, train_y, train_data, train_y,
                    subset_seed=int(payload.get("search_seed", 7)),
                    optimizer_steps=int(payload.get("optimizer_steps", 24)),
                    learning_rate=float(payload.get("learning_rate", 0.04)),
                    search_recipe=dict(payload.get("search_recipe") or {}),
                )
            else:
                train_x = np.asarray(payload["train_representations"], dtype=np.float32)
                output = execute_mutation(
                    payload["bundle"], train_x, train_y, train_x, train_y,
                    subset_seed=int(payload.get("subset_seed", 7)),
                )

        elif kind == TaskKind.DENDRITRON_VERIFICATION:
            checkpoint = payload.get("_native10_checkpoint")
            validation = payload.get("_native10_validation")
            if checkpoint is None:
                raise ValueError("Dendritron verification is missing the canonical checkpoint")
            if validation is None:
                raise ValueError("Dendritron verification is missing coordinator-held global validation")
            if payload.get("engine") == "dendriswarm.native10-trainable.v6":
                model_config = Native10V6Dendritron.from_artifact(checkpoint).config
                validation_x, validation_y, policy = decode_v6_global_validation_artifact(validation, expected_config=model_config)
                output = verify_v6_mutation_full(
                    checkpoint, payload["bundle"], payload["delta"], validation_x, validation_y,
                    validation_hash_value=validation["sha256"], validation_policy=policy,
                )
            else:
                model_config = Native10Dendritron.from_artifact(checkpoint).config
                validation_x, validation_y, _ = decode_global_validation_artifact(validation, expected_config=model_config)
                output = verify_mutation_full(
                    checkpoint, payload["bundle"], payload["delta"], validation_x, validation_y,
                    validation_hash_value=validation["sha256"],
                )

        elif kind == TaskKind.INFERENCE:
            if artifact is None:
                raise ValueError("inference task is missing its artifact")
            model = ReferenceDendritron.from_artifact(artifact)
            scores = model.scores(np.asarray(payload["features"], dtype=np.float64))
            output = {
                "prediction": int(np.argmax(scores)),
                "confidence": float(np.max(scores)),
                "scores": scores.tolist(),
                "active_branches": min(model.top_k, len(model.centers)),
                "total_branches": int(len(model.centers)),
                "activation_fraction": model.activation_fraction,
            }

        else:
            raise ValueError(f"unsupported task kind: {kind}")

    output["runtime"] = {
        "backend": "numpy-cpu",
        "machine": platform.machine().lower() or "unknown",
        "python": platform.python_version(),
        "cpu_threads": None if cpu_threads is None else max(1, int(cpu_threads)),
    }
    return output
