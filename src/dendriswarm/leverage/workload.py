"""Synthetic protocol workload with independent public and secret splits.

DISCLOSURE: this is NOT CIFAR-100 and NOT the Native8 tissue. It is a synthetic
100-class Gaussian-mixture workload used to falsify protocol mechanics. Search
may access only training and public data. Admission, canary, and final
replication use independently seeded splits that are not supplied to search.
"""
from __future__ import annotations

from dataclasses import dataclass
import secrets

import numpy as np

from dendriswarm.leverage.tissue import TerritoryTissue

CLASSES = 100
FEATURES = 8
MODES = 3


@dataclass
class SurrogateWorkload:
    x_train: np.ndarray
    y_train: np.ndarray
    x_public: np.ndarray
    y_public: np.ndarray
    x_private: np.ndarray
    y_private: np.ndarray
    x_canary: np.ndarray
    y_canary: np.ndarray
    x_replication: np.ndarray
    y_replication: np.ndarray
    private_seed: int
    canary_seed: int
    replication_seed: int


def make_surrogate_workload(
    seed: int = 20260722,
    train_per_class: int = 100,
    public_per_class: int = 40,
    private_per_class: int = 60,
    canary_per_class: int = 40,
    replication_per_class: int = 80,
    private_seed: int | None = None,
    canary_seed: int | None = None,
    replication_seed: int | None = None,
) -> SurrogateWorkload:
    """Create public material plus independent admission/canary/replication data."""
    world_rng = np.random.default_rng(seed)
    class_means = world_rng.normal(0.0, 1.0, size=(CLASSES, FEATURES))
    sub_means = class_means[:, None, :] + world_rng.normal(0.0, 1.6, size=(CLASSES, MODES, FEATURES))
    spread = 0.8

    def sample(count_per_class: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        xs, ys = [], []
        for label in range(CLASSES):
            modes = rng.integers(0, MODES, size=count_per_class)
            xs.append(sub_means[label, modes] + rng.normal(0.0, spread, size=(count_per_class, FEATURES)))
            ys.extend([label] * count_per_class)
        return np.concatenate(xs), np.asarray(ys, dtype=np.int64)

    train_rng = np.random.default_rng(seed ^ 0x545241494E)
    public_rng = np.random.default_rng(seed ^ 0x5055424C4943)
    private_value = int(private_seed) if private_seed is not None else secrets.randbits(128)
    canary_value = int(canary_seed) if canary_seed is not None else secrets.randbits(128)
    replication_value = int(replication_seed) if replication_seed is not None else secrets.randbits(128)

    x_train, y_train = sample(train_per_class, train_rng)
    x_public, y_public = sample(public_per_class, public_rng)
    x_private, y_private = sample(private_per_class, np.random.default_rng(private_value))
    x_canary, y_canary = sample(canary_per_class, np.random.default_rng(canary_value))
    x_replication, y_replication = sample(replication_per_class, np.random.default_rng(replication_value))
    return SurrogateWorkload(
        x_train, y_train, x_public, y_public,
        x_private, y_private, x_canary, y_canary, x_replication, y_replication,
        private_value, canary_value, replication_value,
    )


def kmeans_prototypes(points: np.ndarray, count: int, iterations: int,
                      rng: np.random.Generator) -> np.ndarray:
    count = min(count, len(points))
    idx = rng.choice(len(points), size=count, replace=False)
    centers = points[idx].copy()
    for _ in range(iterations):
        d2 = ((points[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        assignment = d2.argmin(axis=1)
        for branch in range(count):
            selected = points[assignment == branch]
            if len(selected):
                centers[branch] = selected.mean(axis=0)
    return np.round(centers, 12)


def train_parent(workload: SurrogateWorkload, per_class: int = 2,
                 iterations: int = 1, seed: int = 7,
                 top_k: int = 3, temperature: float = 2.0) -> TerritoryTissue:
    rng = np.random.default_rng(seed)
    centers, owners = [], []
    for label in range(CLASSES):
        points = workload.x_train[workload.y_train == label]
        prototypes = kmeans_prototypes(points, per_class, iterations, rng)
        centers.extend(prototypes)
        owners.extend([label] * len(prototypes))
    return TerritoryTissue(np.asarray(centers), np.asarray(owners), top_k, temperature)


def honest_delta(parent: TerritoryTissue, workload: SurrogateWorkload,
                 categories: tuple[int, ...], per_class: int = 6,
                 iterations: int = 25, seed: int = 11,
                 ) -> tuple[dict[int, tuple[np.ndarray, int]], list[tuple[np.ndarray, int]]]:
    """Training-only local refit; no protected split is consulted."""
    rng = np.random.default_rng(seed)
    replaced: dict[int, tuple[np.ndarray, int]] = {}
    added: list[tuple[np.ndarray, int]] = []
    for label in categories:
        points = workload.x_train[workload.y_train == label]
        prototypes = kmeans_prototypes(points, per_class, iterations, rng)
        owned = [i for i in range(len(parent.centers)) if parent.owners[i] == label]
        for slot, index in enumerate(owned):
            replaced[index] = (prototypes[slot], label)
        for extra in prototypes[len(owned):]:
            added.append((extra, label))
    return replaced, added


def variance_farm_delta(parent: TerritoryTissue, categories: tuple[int, ...],
                        seed: int) -> tuple[dict[int, tuple[np.ndarray, int]], list[tuple[np.ndarray, int]]]:
    rng = np.random.default_rng(seed)
    replaced: dict[int, tuple[np.ndarray, int]] = {}
    for label in categories:
        owned = [i for i in range(len(parent.centers)) if parent.owners[i] == label]
        for index in owned:
            jitter = rng.normal(0.0, 0.08, size=parent.centers.shape[1])
            replaced[index] = (parent.centers[index] + jitter, label)
    return replaced, []


def escape_delta(parent: TerritoryTissue, workload: SurrogateWorkload,
                 declared: tuple[int, ...], target: int,
                 seed: int = 13) -> tuple[dict[int, tuple[np.ndarray, int]], list[tuple[np.ndarray, int]]]:
    rng = np.random.default_rng(seed)
    points = workload.x_train[workload.y_train == target]
    intruders = kmeans_prototypes(points, 3, 10, rng)
    return {}, [(center, declared[0]) for center in intruders]


def sleeper_delta(
    parent: TerritoryTissue,
    workload: SurrogateWorkload,
    categories: tuple[int, ...],
    *,
    target_label: int = 0,
    wrong_owner: int = 1,
    seed: int = 11,
) -> tuple[
    dict[int, tuple[np.ndarray, int]],
    list[tuple[np.ndarray, int]],
    np.ndarray,
    np.ndarray,
]:
    """Training-only improvement plus a dormant off-distribution trigger.

    The trigger center is selected using only the parent and public training
    geometry: it lies in the target route region and is correctly classified by
    the parent, but is far from training samples. Three exact wrong-owner
    branches make the candidate fail on a concrete labeled canary batch while
    remaining dormant on ordinary admission data.
    """
    if target_label not in categories or wrong_owner not in categories:
        raise ValueError("sleeper labels must be inside the declared route regions")
    replaced, added = honest_delta(parent, workload, categories, seed=seed)
    rng = np.random.default_rng(seed ^ 0x51EE)
    target_points = workload.x_train[workload.y_train == target_label]
    base = parent.centers[np.flatnonzero(parent.owners == target_label)[0]]
    candidates: list[np.ndarray] = []
    for scale in np.linspace(2.0, 10.0, 17):
        for _ in range(64):
            direction = rng.normal(size=parent.centers.shape[1])
            direction /= max(np.linalg.norm(direction), 1e-12)
            point = base + direction * scale
            if int(parent.canonical_categories(point[None, :])[0]) != target_label:
                continue
            if int(parent.predict(point[None, :])[0]) != target_label:
                continue
            candidates.append(point)
    if not candidates:
        raise RuntimeError("could not construct a parent-correct sleeper trigger")
    # Prefer a point least represented by training data, reducing accidental
    # admission-challenge activation without consulting protected data.
    trigger_center = max(
        candidates,
        key=lambda point: float(((target_points - point) ** 2).sum(axis=1).min()),
    )
    for offset in (-0.002, 0.0, 0.002):
        center = trigger_center.copy()
        center[1] += offset
        added.append((center, wrong_owner))

    generated: list[np.ndarray] = []
    while len(generated) < 120:
        batch = trigger_center + rng.normal(0.0, 0.001, size=(240, parent.centers.shape[1]))
        keep = (parent.canonical_categories(batch) == target_label) & (parent.predict(batch) == target_label)
        generated.extend(batch[keep])
    trigger_x = np.asarray(generated[:120], dtype=np.float64)
    trigger_y = np.full(len(trigger_x), target_label, dtype=np.int64)
    return replaced, added, trigger_x, trigger_y
