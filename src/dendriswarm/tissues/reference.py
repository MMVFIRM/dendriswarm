from __future__ import annotations

from dataclasses import dataclass
import warnings
from typing import Any

import numpy as np

from dendriswarm.core.crypto import content_hash


@dataclass(frozen=True)
class TissueConfig:
    branches: int = 160
    top_k: int = 3
    temperature: float = 0.18
    iterations: int = 15
    seed: int = 7

    def as_dict(self) -> dict[str, Any]:
        return {
            "branches": self.branches,
            "top_k": self.top_k,
            "temperature": self.temperature,
            "iterations": self.iterations,
            "seed": self.seed,
        }


class ReferenceDendritron:
    """Small sparse branch-owned prototype tissue.

    Each branch is owned by one class and stores a local prototype. Inference
    activates only the nearest top-k branches and aggregates radial evidence.
    It is intentionally inspectable and deterministic, not a claim of a full
    production Dendritron implementation.
    """

    def __init__(self, centers: np.ndarray, owners: np.ndarray, top_k: int, temperature: float):
        self.centers = np.asarray(centers, dtype=np.float64)
        self.owners = np.asarray(owners, dtype=np.int64)
        self.top_k = int(top_k)
        self.temperature = float(temperature)
        if self.centers.ndim != 2 or len(self.centers) == 0:
            raise ValueError("centers must be a non-empty 2D matrix")
        if len(self.owners) != len(self.centers):
            raise ValueError("owners must align with centers")
        self.classes = int(self.owners.max()) + 1

    @classmethod
    def train(cls, x: np.ndarray, y: np.ndarray, config: TissueConfig) -> "ReferenceDendritron":
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.int64)
        if x.ndim != 2 or y.ndim != 1 or len(x) != len(y) or not len(x):
            raise ValueError("training arrays are invalid")
        rng = np.random.default_rng(config.seed)
        classes = np.unique(y)
        per_class = max(1, config.branches // len(classes))
        centers: list[np.ndarray] = []
        owners: list[int] = []

        for label in classes:
            points = x[y == label]
            count = min(per_class, len(points))
            idx = rng.choice(len(points), size=count, replace=False)
            c = points[idx].copy()
            for _ in range(config.iterations):
                distances = ((points[:, None, :] - c[None, :, :]) ** 2).sum(axis=2)
                assignment = distances.argmin(axis=1)
                for branch in range(count):
                    selected = points[assignment == branch]
                    if len(selected):
                        c[branch] = selected.mean(axis=0)
            c = np.round(c, 12)
            centers.extend(c)
            owners.extend([int(label)] * count)

        return cls(np.asarray(centers), np.asarray(owners), config.top_k, config.temperature)

    def scores(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        single = x.ndim == 1
        if single:
            x = x[None, :]
        if x.ndim != 2 or x.shape[1] != self.centers.shape[1]:
            raise ValueError(f"expected feature width {self.centers.shape[1]}")
        d2 = ((x[:, None, :] - self.centers[None, :, :]) ** 2).sum(axis=2)
        k = min(max(1, self.top_k), self.centers.shape[0])
        active = np.argpartition(d2, kth=k - 1, axis=1)[:, :k]
        out = np.zeros((len(x), self.classes), dtype=np.float64)
        for row in range(len(x)):
            idx = active[row]
            evidence = np.exp(-d2[row, idx] / max(self.temperature, 1e-8))
            for branch, weight in zip(idx, evidence, strict=True):
                out[row, self.owners[branch]] += weight
        denom = out.sum(axis=1, keepdims=True)
        out = out / np.where(denom == 0, 1.0, denom)
        return out[0] if single else out

    def predict(self, x: np.ndarray) -> np.ndarray:
        scores = self.scores(x)
        if scores.ndim == 1:
            return np.asarray(int(scores.argmax()))
        return scores.argmax(axis=1)

    def accuracy(self, x: np.ndarray, y: np.ndarray) -> float:
        return float((self.predict(x) == np.asarray(y)).mean())

    @property
    def activation_fraction(self) -> float:
        return min(self.top_k, len(self.centers)) / len(self.centers)

    def artifact(self, config: dict[str, Any], dataset_hash: str) -> dict[str, Any]:
        value = {
            "format": "dendriswarm.reference-tissue.v2",
            "centers": np.round(self.centers, 10).tolist(),
            "owners": self.owners.tolist(),
            "top_k": self.top_k,
            "temperature": self.temperature,
            "feature_width": int(self.centers.shape[1]),
            "classes": self.classes,
            "dataset_hash": dataset_hash,
            "config": config,
        }
        value["sha256"] = artifact_hash(value)
        return value

    @classmethod
    def from_artifact(cls, artifact: dict[str, Any]) -> "ReferenceDendritron":
        if artifact.get("format") != "dendriswarm.reference-tissue.v2":
            raise ValueError("unsupported tissue artifact format")
        expected = artifact.get("sha256")
        if not expected or expected != artifact_hash(artifact):
            raise ValueError("tissue artifact hash mismatch")
        model = cls(
            np.asarray(artifact["centers"], dtype=np.float64),
            np.asarray(artifact["owners"], dtype=np.int64),
            int(artifact["top_k"]),
            float(artifact["temperature"]),
        )
        if model.centers.shape[1] != int(artifact["feature_width"]):
            raise ValueError("artifact feature width mismatch")
        if model.centers.shape[0] > 4096 or model.centers.shape[1] > 4096:
            raise ValueError("artifact exceeds reference runtime dimensions")
        if not np.isfinite(model.centers).all() or not np.isfinite(model.temperature):
            raise ValueError("artifact contains non-finite values")
        if model.top_k < 1 or model.top_k > len(model.centers):
            raise ValueError("artifact top_k is invalid")
        if (model.owners < 0).any() or (model.owners >= model.classes).any():
            raise ValueError("artifact owners are invalid")
        return model


def artifact_hash(artifact: dict[str, Any]) -> str:
    return content_hash(artifact)


def artifact_consensus_hash(artifact: dict[str, Any], decimals: int = 8) -> str:
    """Cross-architecture semantic fingerprint for independently trained artifacts.

    The exact artifact hash remains the content-addressed identity. Consensus is
    intentionally computed over a coarser, versioned quantization so harmless
    BLAS/architecture rounding differences do not reject honest replicas.
    """
    value = {key: item for key, item in artifact.items() if key != "sha256"}
    value["centers"] = np.round(np.asarray(artifact["centers"], dtype=np.float64), decimals).tolist()
    value["temperature"] = round(float(artifact["temperature"]), 12)
    return content_hash({
        "format": "dendriswarm.reference-tissue-consensus.v1",
        "quantization_decimals": int(decimals),
        "artifact": value,
    })


def make_digits_dataset(seed: int = 42) -> dict[str, Any]:
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split

    digits = load_digits()
    features = digits.data.astype(np.float64) / 16.0
    labels = digits.target.astype(np.int64)
    all_indices = np.arange(len(features))
    train_idx, temp_idx = train_test_split(
        all_indices, test_size=0.4, random_state=seed, stratify=labels
    )
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.5, random_state=seed, stratify=labels[temp_idx]
    )
    artifact: dict[str, Any] = {
        "format": "dendriswarm.dataset.v1",
        "name": "sklearn-digits-8x8-v1",
        "source": "scikit-learn load_digits; derived from Alpaydin & Kaynak, Optical Recognition of Handwritten Digits, UCI ML Repository, DOI 10.24432/C50P49",
        "license": "CC BY 4.0 (upstream UCI dataset)",
        "description": "1,797 normalized 8x8 handwritten digit images with deterministic stratified train/validation/test splits.",
        "features": features.tolist(),
        "labels": labels.tolist(),
        "splits": {
            "train": train_idx.tolist(),
            "validation": val_idx.tolist(),
            "test": test_idx.tolist(),
        },
        "feature_width": int(features.shape[1]),
        "classes": 10,
        "seed": seed,
    }
    artifact["sha256"] = dataset_hash(artifact)
    return artifact


def dataset_hash(dataset: dict[str, Any]) -> str:
    return content_hash(dataset)


def dataset_split(dataset: dict[str, Any], split: str) -> tuple[np.ndarray, np.ndarray]:
    if dataset.get("sha256") != dataset_hash(dataset):
        raise ValueError("dataset hash mismatch")
    indices = np.asarray(dataset["splits"][split], dtype=np.int64)
    x = np.asarray(dataset["features"], dtype=np.float64)[indices]
    y = np.asarray(dataset["labels"], dtype=np.int64)[indices]
    return x, y


def _budgeted_source_indices(
    dataset: dict[str, Any], split: str, sample_budget: int | None, seed: int
) -> np.ndarray:
    split_indices = np.asarray(dataset["splits"][split], dtype=np.int64)
    if sample_budget is None or sample_budget >= len(split_indices):
        return split_indices
    all_labels = np.asarray(dataset["labels"], dtype=np.int64)
    split_labels = all_labels[split_indices]
    unique_labels = np.unique(split_labels)
    if sample_budget < len(unique_labels):
        raise ValueError("sample budget must cover every class")
    rng = np.random.default_rng(seed)
    selected_positions: list[int] = []
    base = sample_budget // len(unique_labels)
    remainder = sample_budget % len(unique_labels)
    for position, label in enumerate(unique_labels):
        candidates = np.flatnonzero(split_labels == label)
        count = min(len(candidates), base + (1 if position < remainder else 0))
        selected_positions.extend(rng.choice(candidates, size=count, replace=False).tolist())
    return split_indices[np.asarray(sorted(selected_positions), dtype=np.int64)]


def dataset_split_budgeted(
    dataset: dict[str, Any], split: str, sample_budget: int | None, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Materialize only selected rows instead of converting the full feature matrix."""
    if dataset.get("sha256") != dataset_hash(dataset):
        raise ValueError("dataset hash mismatch")
    source_indices = _budgeted_source_indices(dataset, split, sample_budget, seed)
    features = np.asarray([dataset["features"][int(index)] for index in source_indices], dtype=np.float64)
    labels = np.asarray([dataset["labels"][int(index)] for index in source_indices], dtype=np.int64)
    return features, labels


def budgeted_dataset_artifact(
    dataset: dict[str, Any], train_budget: int, validation_budget: int, seed: int
) -> dict[str, Any]:
    """Build a compact signed scouting dataset containing only assigned rows."""
    if dataset.get("sha256") != dataset_hash(dataset):
        raise ValueError("dataset hash mismatch")
    train_sources = _budgeted_source_indices(dataset, "train", train_budget, seed)
    validation_sources = _budgeted_source_indices(dataset, "validation", validation_budget, seed + 1)
    source_indices = np.concatenate([train_sources, validation_sources])
    artifact: dict[str, Any] = {
        "format": "dendriswarm.dataset.v1",
        "name": f"{dataset.get('name', 'dataset')}-budgeted-{train_budget}-{validation_budget}",
        "source": dataset.get("source", ""),
        "license": dataset.get("license", ""),
        "description": "Coordinator-generated deterministic compact scouting shard.",
        "features": [dataset["features"][int(index)] for index in source_indices],
        "labels": [int(dataset["labels"][int(index)]) for index in source_indices],
        "splits": {
            "train": list(range(len(train_sources))),
            "validation": list(range(len(train_sources), len(source_indices))),
            "test": [],
        },
        "feature_width": int(dataset["feature_width"]),
        "classes": int(dataset["classes"]),
        "seed": int(seed),
        "parent_dataset_hash": dataset["sha256"],
        "source_indices_hash": content_hash({"indices": source_indices.astype(int).tolist()}),
    }
    artifact["sha256"] = dataset_hash(artifact)
    return artifact


def train_and_validation(dataset: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    train = np.asarray(dataset["splits"]["train"], dtype=np.int64)
    validation = np.asarray(dataset["splits"]["validation"], dtype=np.int64)
    indices = np.concatenate([train, validation])
    x = np.asarray(dataset["features"], dtype=np.float64)[indices]
    y = np.asarray(dataset["labels"], dtype=np.int64)[indices]
    return x, y


def perturbed_audit_split(dataset: dict[str, Any], seed: int = 20260318) -> tuple[np.ndarray, np.ndarray]:
    """Publicly reconstructable perturbed replica of the test split.

    v0.2.1 correction: this was previously named "hidden" and duplicated as
    ``hidden_challenge``. The seed and transformation are public, so this is a
    reproducibility and artifact-integrity check, NOT a hidden generalization
    test. Genuinely private challenges are provided by the v0.3 committed
    challenge epochs (``dendriswarm.leverage.epoch``)."""
    x, y = dataset_split(dataset, "test")
    rng = np.random.default_rng(seed)
    return np.clip(x + rng.normal(0.0, 0.015, size=x.shape), 0.0, 1.0), y


hidden_audit_split = perturbed_audit_split  # deprecated alias, kept for v0.2 compatibility


def reference_benchmark() -> dict[str, Any]:
    from sklearn.dummy import DummyClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier, NearestCentroid

    dataset = make_digits_dataset()
    x_train, y_train = train_and_validation(dataset)
    x_test, y_test = dataset_split(dataset, "test")
    config = TissueConfig(branches=160, top_k=1, temperature=0.18, iterations=15, seed=7)
    tissue = ReferenceDendritron.train(x_train, y_train, config)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        nearest_centroid = NearestCentroid().fit(x_train, y_train).score(x_test, y_test)
    baselines = {
        "dendriswarm_reference_tissue": tissue.accuracy(x_test, y_test),
        "knn_3": KNeighborsClassifier(n_neighbors=3).fit(x_train, y_train).score(x_test, y_test),
        "logistic_regression": LogisticRegression(max_iter=2000, random_state=7).fit(x_train, y_train).score(x_test, y_test),
        "nearest_centroid": nearest_centroid,
        "majority": DummyClassifier(strategy="most_frequent").fit(x_train, y_train).score(x_test, y_test),
    }
    return {
        "dataset": dataset["name"], "samples": len(dataset["features"]),
        "split_seed": dataset["seed"], "config": config.as_dict(),
        "accuracy": {k: float(v) for k, v in baselines.items()},
        "active_branches": config.top_k, "total_branches": len(tissue.centers),
        "activation_fraction": tissue.activation_fraction,
        "claim_boundary": "Accuracy and structural sparsity only; no wall-clock speedup claim.",
    }
