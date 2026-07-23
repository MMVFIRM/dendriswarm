from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from dendriswarm.core.crypto import content_hash
from dendriswarm.v6.native10 import Native10Config, Native10Dendritron, decode_array, encode_array

VALIDATION_FORMAT = "dendriswarm.native10-global-validation.v2"


@dataclass(frozen=True)
class GlobalValidationPolicy:
    """Committed statistical and harm policy for one trainer-invisible holdout."""

    min_samples_per_class: int = 10
    familywise_alpha: float = 0.05
    max_candidate_evaluations: int = 20
    max_search_rounds: int = 1
    min_discordant: int = 20
    minimum_net_wins: int = 2
    minimum_effect_rate: float = 0.002
    max_loss_per_class: int = 1
    max_loss_rate_per_class: float = 0.10

    def __post_init__(self) -> None:
        if self.min_samples_per_class < 1:
            raise ValueError("min_samples_per_class must be positive")
        if not 0.0 < self.familywise_alpha < 1.0:
            raise ValueError("familywise_alpha must be in (0,1)")
        if self.max_candidate_evaluations < 1:
            raise ValueError("max_candidate_evaluations must be positive")
        if self.max_search_rounds != 1:
            raise ValueError("v0.6 validation artifacts are one-tournament only")
        if self.min_discordant < 1:
            raise ValueError("min_discordant must be positive")
        if self.minimum_net_wins < 1:
            raise ValueError("minimum_net_wins must be positive")
        if not 0.0 <= self.minimum_effect_rate <= 1.0:
            raise ValueError("minimum_effect_rate must be in [0,1]")
        if self.max_loss_per_class < 0:
            raise ValueError("max_loss_per_class cannot be negative")
        if not 0.0 <= self.max_loss_rate_per_class <= 1.0:
            raise ValueError("max_loss_rate_per_class must be in [0,1]")

    @property
    def corrected_alpha(self) -> float:
        return self.familywise_alpha / self.max_candidate_evaluations

    def as_dict(self) -> dict[str, Any]:
        return {
            "min_samples_per_class": self.min_samples_per_class,
            "familywise_alpha": self.familywise_alpha,
            "max_candidate_evaluations": self.max_candidate_evaluations,
            "max_search_rounds": self.max_search_rounds,
            "min_discordant": self.min_discordant,
            "minimum_net_wins": self.minimum_net_wins,
            "minimum_effect_rate": self.minimum_effect_rate,
            "max_loss_per_class": self.max_loss_per_class,
            "max_loss_rate_per_class": self.max_loss_rate_per_class,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "GlobalValidationPolicy":
        return cls(**value)


def exact_one_sided_mcnemar(wins: int, losses: int) -> float:
    """Exact P[X >= wins] for X~Binomial(wins+losses, 0.5)."""
    wins = int(wins)
    losses = int(losses)
    n = wins + losses
    if n == 0 or wins <= losses:
        return 1.0
    # Recurrence avoids huge integer-to-float conversions and SciPy dependency.
    log_p = -n * math.log(2.0) + math.lgamma(n + 1) - math.lgamma(wins + 1) - math.lgamma(n - wins + 1)
    term = math.exp(log_p)
    total = term
    for k in range(wins, n):
        term *= (n - k) / (k + 1)
        total += term
    return min(1.0, float(total))


def paired_evidence(pre_predictions: np.ndarray, post_predictions: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    pre = np.asarray(pre_predictions) == np.asarray(labels)
    post = np.asarray(post_predictions) == np.asarray(labels)
    wins = int((~pre & post).sum())
    losses = int((pre & ~post).sum())
    discordant = wins + losses
    return {
        "wins": wins,
        "losses": losses,
        "discordant": discordant,
        "net_wins": wins - losses,
        "effect_rate": (wins - losses) / max(1, len(pre)),
        "mcnemar_p_value": exact_one_sided_mcnemar(wins, losses),
    }


def validation_hash(value: dict[str, Any]) -> str:
    return content_hash({key: item for key, item in value.items() if key != "sha256"})


def make_global_validation_artifact(
    config: Native10Config,
    inputs: np.ndarray,
    labels: np.ndarray,
    *,
    source: str,
    split: str = "validation",
    policy: GlobalValidationPolicy | None = None,
    protocol_fixture_only: bool = False,
) -> dict[str, Any]:
    policy = policy or GlobalValidationPolicy()
    x = np.asarray(inputs, dtype=np.float32)
    y = np.asarray(labels, dtype=np.int64)
    if x.ndim != 2 or x.shape[1] != config.input_width or not len(x):
        raise ValueError("global validation inputs have the wrong shape")
    if y.ndim != 1 or len(y) != len(x):
        raise ValueError("global validation labels are empty or misaligned")
    if not np.isfinite(x).all():
        raise ValueError("global validation inputs contain non-finite values")
    if np.any(y < 0) or np.any(y >= config.classes):
        raise ValueError("global validation labels are outside the model class range")
    counts = np.bincount(y, minlength=config.classes)
    if np.any(counts < policy.min_samples_per_class):
        missing = np.flatnonzero(counts < policy.min_samples_per_class).astype(int).tolist()
        raise ValueError(f"global validation lacks committed class coverage: {missing[:20]}")
    value: dict[str, Any] = {
        "format": VALIDATION_FORMAT,
        "config": config.as_dict(),
        "input_width": config.input_width,
        "classes": config.classes,
        "sample_count": int(len(y)),
        "counts_by_class": counts.astype(int).tolist(),
        "inputs": encode_array(x),
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
    artifact: dict[str, Any], *, expected_config: Native10Config | None = None
) -> tuple[np.ndarray, np.ndarray, GlobalValidationPolicy]:
    if artifact.get("format") != VALIDATION_FORMAT:
        raise ValueError("unsupported global validation artifact format")
    if artifact.get("sha256") != validation_hash(artifact):
        raise ValueError("global validation artifact hash mismatch")
    config = Native10Config.from_dict(dict(artifact["config"]))
    if expected_config is not None and config.as_dict() != expected_config.as_dict():
        raise ValueError("global validation config does not match the canonical model")
    policy = GlobalValidationPolicy.from_dict(dict(artifact["policy"]))
    x = np.asarray(decode_array(artifact["inputs"]), dtype=np.float32)
    y = np.asarray(decode_array(artifact["labels"]), dtype=np.int64)
    rebuilt = make_global_validation_artifact(
        config, x, y, source=str(artifact.get("source", "")), split=str(artifact.get("split", "validation")),
        policy=policy, protocol_fixture_only=bool(artifact.get("protocol_fixture_only", False)),
    )
    if rebuilt["sha256"] != artifact["sha256"]:
        raise ValueError("global validation metadata is inconsistent")
    return x, y, policy


def synthetic_raw_samples(
    config: Native10Config, *, per_class: int, prototype_seed: int = 20260723, sample_seed: int = 20260724
) -> tuple[np.ndarray, np.ndarray]:
    prototype_rng = np.random.default_rng(prototype_seed)
    sample_rng = np.random.default_rng(sample_seed)
    prototypes = prototype_rng.normal(0, 1, size=(config.classes, config.input_width)).astype(np.float32)
    for class_id in range(config.classes):
        category = class_id // config.classes_per_category
        start = category * config.field_block_width % config.input_width
        stop = min(config.input_width, start + config.field_block_width)
        prototypes[class_id, start:stop] += 2.5
        prototypes[class_id] /= max(float(np.linalg.norm(prototypes[class_id])), 1e-8)
    rows, labels = [], []
    for class_id in range(config.classes):
        noise = sample_rng.normal(0, 0.18, size=(per_class, config.input_width)).astype(np.float32)
        rows.append(prototypes[class_id][None, :] * 3.0 + noise)
        labels.extend([class_id] * per_class)
    x = np.concatenate(rows, axis=0)
    y = np.asarray(labels, dtype=np.int64)
    order = sample_rng.permutation(len(y))
    return x[order], y[order]


def synthetic_global_validation_fixture(
    model_or_config: Native10Dendritron | Native10Config,
    *,
    per_class: int = 12,
    seed: int = 20260723,
    policy: GlobalValidationPolicy | None = None,
) -> dict[str, Any]:
    """Deterministic raw-input protocol fixture; never a performance benchmark."""
    config = model_or_config.config if isinstance(model_or_config, Native10Dendritron) else model_or_config
    policy = policy or GlobalValidationPolicy(
        min_samples_per_class=min(5, per_class), min_discordant=min(10, max(1, config.classes)),
        minimum_net_wins=1, minimum_effect_rate=0.0, max_loss_per_class=max(1, per_class // 4),
        max_loss_rate_per_class=0.25,
    )
    if per_class < policy.min_samples_per_class:
        raise ValueError("per_class is below the committed validation policy")
    x, y = synthetic_raw_samples(config, per_class=per_class, prototype_seed=seed, sample_seed=seed + 1)
    return make_global_validation_artifact(
        config, x, y, source="synthetic-raw-all-class-protocol-fixture",
        split="coordinator-private-global-validation", policy=policy, protocol_fixture_only=True,
    )

