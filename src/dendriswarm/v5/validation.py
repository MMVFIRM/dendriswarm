from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from dendriswarm.core.crypto import content_hash
from dendriswarm.v5.native10 import Native10Config, decode_array, encode_array

VALIDATION_FORMAT = "dendriswarm.native10-global-validation.v1"


@dataclass(frozen=True)
class GlobalValidationPolicy:
    min_samples_per_class: int = 5
    minimum_net_wins: int = 1
    max_loss_per_class: int = 1
    max_loss_rate_per_class: float = 0.20
    max_candidate_evaluations: int = 40

    def __post_init__(self) -> None:
        if self.min_samples_per_class < 1:
            raise ValueError("min_samples_per_class must be positive")
        if self.minimum_net_wins < 1:
            raise ValueError("minimum_net_wins must be positive")
        if self.max_loss_per_class < 0:
            raise ValueError("max_loss_per_class cannot be negative")
        if not 0.0 <= self.max_loss_rate_per_class <= 1.0:
            raise ValueError("max_loss_rate_per_class must be in [0,1]")
        if self.max_candidate_evaluations < 1:
            raise ValueError("max_candidate_evaluations must be positive")

    def as_dict(self) -> dict[str, Any]:
        return {
            "min_samples_per_class": self.min_samples_per_class,
            "minimum_net_wins": self.minimum_net_wins,
            "max_loss_per_class": self.max_loss_per_class,
            "max_loss_rate_per_class": self.max_loss_rate_per_class,
            "max_candidate_evaluations": self.max_candidate_evaluations,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "GlobalValidationPolicy":
        return cls(**value)


def validation_hash(value: dict[str, Any]) -> str:
    material = {key: item for key, item in value.items() if key != "sha256"}
    return content_hash(material)


def make_global_validation_artifact(
    config: Native10Config,
    representations: np.ndarray,
    labels: np.ndarray,
    *,
    source: str,
    split: str = "validation",
    policy: GlobalValidationPolicy | None = None,
    protocol_fixture_only: bool = False,
) -> dict[str, Any]:
    policy = policy or GlobalValidationPolicy()
    x = np.asarray(representations, dtype=np.float32)
    y = np.asarray(labels, dtype=np.int64)
    if x.ndim != 2 or x.shape[1] != config.representation_width or not len(x):
        raise ValueError("global validation representations have the wrong shape")
    if y.ndim != 1 or len(y) != len(x):
        raise ValueError("global validation labels are empty or misaligned")
    if not np.isfinite(x).all():
        raise ValueError("global validation representations contain non-finite values")
    if np.any(y < 0) or np.any(y >= config.classes):
        raise ValueError("global validation labels are outside the model class range")
    counts = np.bincount(y, minlength=config.classes)
    if len(counts) != config.classes or np.any(counts < policy.min_samples_per_class):
        missing = np.flatnonzero(counts < policy.min_samples_per_class).astype(int).tolist()
        raise ValueError(
            "global validation must cover every class with at least "
            f"{policy.min_samples_per_class} samples; insufficient classes={missing[:20]}"
        )
    value: dict[str, Any] = {
        "format": VALIDATION_FORMAT,
        "config": config.as_dict(),
        "representation_width": config.representation_width,
        "classes": config.classes,
        "sample_count": int(len(y)),
        "counts_by_class": counts.astype(int).tolist(),
        "representations": encode_array(x),
        "labels": encode_array(y),
        "source": str(source),
        "split": str(split),
        "trainer_visible": False,
        "protocol_fixture_only": bool(protocol_fixture_only),
        "policy": policy.as_dict(),
    }
    value["sha256"] = validation_hash(value)
    return value


def decode_global_validation_artifact(
    artifact: dict[str, Any],
    *,
    expected_config: Native10Config | None = None,
) -> tuple[np.ndarray, np.ndarray, GlobalValidationPolicy]:
    if artifact.get("format") != VALIDATION_FORMAT:
        raise ValueError("unsupported global validation artifact format")
    if artifact.get("sha256") != validation_hash(artifact):
        raise ValueError("global validation artifact hash mismatch")
    config = Native10Config.from_dict(dict(artifact["config"]))
    if expected_config is not None and config.as_dict() != expected_config.as_dict():
        raise ValueError("global validation config does not match the canonical model")
    policy = GlobalValidationPolicy.from_dict(dict(artifact["policy"]))
    x = np.asarray(decode_array(artifact["representations"]), dtype=np.float32)
    y = np.asarray(decode_array(artifact["labels"]), dtype=np.int64)
    rebuilt = make_global_validation_artifact(
        config,
        x,
        y,
        source=str(artifact.get("source", "")),
        split=str(artifact.get("split", "validation")),
        policy=policy,
        protocol_fixture_only=bool(artifact.get("protocol_fixture_only", False)),
    )
    if rebuilt["sha256"] != artifact["sha256"]:
        raise ValueError("global validation metadata is inconsistent")
    return x, y, policy


def synthetic_global_validation_fixture(
    config: Native10Config,
    *,
    per_class: int = 20,
    seed: int = 20260723,
    policy: GlobalValidationPolicy | None = None,
) -> dict[str, Any]:
    """Deterministic all-class protocol fixture; never a performance benchmark."""
    policy = policy or GlobalValidationPolicy(min_samples_per_class=min(5, per_class))
    if per_class < policy.min_samples_per_class:
        raise ValueError("per_class is below the committed validation policy")
    rows: list[np.ndarray] = []
    labels: list[int] = []
    for category in range(config.categories):
        axis_rng = np.random.default_rng(seed + category * 1009)
        sample_rng = np.random.default_rng(seed + 10_000_019 + category * 65537)
        category_axis = np.zeros(config.representation_width, dtype=np.float32)
        category_axis[category % config.representation_width] = 1.0
        class_axes = axis_rng.normal(
            0, 1, size=(config.classes_per_category, config.representation_width)
        ).astype(np.float32)
        class_axes[:, : config.categories] = 0.0
        class_axes /= np.maximum(np.linalg.norm(class_axes, axis=1, keepdims=True), 1e-8)
        start = category * config.classes_per_category
        for local, axis in enumerate(class_axes):
            values = (
                category_axis[None, :] * 3.0
                + axis[None, :] * 2.0
                + sample_rng.normal(0, 0.25, size=(per_class, config.representation_width))
            )
            rows.append(values.astype(np.float32))
            labels.extend([start + local] * per_class)
    x = np.concatenate(rows, axis=0)
    y = np.asarray(labels, dtype=np.int64)
    order = np.random.default_rng(seed ^ 0x5A17).permutation(len(y))
    return make_global_validation_artifact(
        config,
        x[order],
        y[order],
        source="synthetic-all-class-protocol-fixture",
        split="coordinator-private-global-validation",
        policy=policy,
        protocol_fixture_only=True,
    )
