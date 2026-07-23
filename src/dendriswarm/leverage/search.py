"""Reference untrusted search harness using public data only.

This module is intentionally outside the trusted service. It receives no
challenge, canary, or replication arrays. Its report makes the missing economic
predicate measurable: how often public-data search discovers deltas that later
survive trusted admission and independent replication.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from dendriswarm.leverage.stats import paired_comparison
from dendriswarm.leverage.tissue import Territory, TerritoryTissue, build_eval_cache, evaluate_candidate
from dendriswarm.leverage.workload import kmeans_prototypes


@dataclass(frozen=True)
class SearchCandidate:
    categories: tuple[int, ...]
    seed: int
    replaced: dict[int, tuple[np.ndarray, int]]
    added: list[tuple[np.ndarray, int]]
    public_net_wins: int
    public_p_value: float
    public_delta_active_share: float
    untrusted_distance_ops: int


@dataclass(frozen=True)
class PublicSearchReport:
    candidates_generated: int
    candidates_with_positive_public_net_wins: int
    best_public_net_wins: int
    total_untrusted_distance_ops: int
    protected_split_accessed: bool
    candidates: tuple[SearchCandidate, ...]

    def as_dict(self) -> dict:
        return {
            "format": "dendriswarm.public-search-report.v3.2",
            "candidates_generated": self.candidates_generated,
            "candidates_with_positive_public_net_wins": self.candidates_with_positive_public_net_wins,
            "best_public_net_wins": self.best_public_net_wins,
            "total_untrusted_distance_ops": self.total_untrusted_distance_ops,
            "protected_split_accessed": self.protected_split_accessed,
            "candidate_summaries": [
                {
                    "categories": list(candidate.categories),
                    "seed": candidate.seed,
                    "public_net_wins": candidate.public_net_wins,
                    "public_p_value": candidate.public_p_value,
                    "public_delta_active_share": candidate.public_delta_active_share,
                    "untrusted_distance_ops": candidate.untrusted_distance_ops,
                }
                for candidate in self.candidates
            ],
        }


def _training_delta(
    parent: TerritoryTissue,
    x_train: np.ndarray,
    y_train: np.ndarray,
    categories: tuple[int, ...],
    *,
    seed: int,
    per_class: int,
    iterations: int,
) -> tuple[dict[int, tuple[np.ndarray, int]], list[tuple[np.ndarray, int]]]:
    rng = np.random.default_rng(seed)
    replaced: dict[int, tuple[np.ndarray, int]] = {}
    added: list[tuple[np.ndarray, int]] = []
    for label in categories:
        points = x_train[y_train == label]
        if len(points) == 0:
            continue
        prototypes = kmeans_prototypes(points, per_class, iterations, rng)
        owned = [index for index, owner in enumerate(parent.owners) if int(owner) == label]
        for slot, index in enumerate(owned):
            if slot < len(prototypes):
                replaced[index] = (prototypes[slot], label)
        for extra in prototypes[len(owned):]:
            added.append((extra, label))
    return replaced, added


def search_local_refits(
    parent: TerritoryTissue,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_public: np.ndarray,
    y_public: np.ndarray,
    category_blocks: Iterable[tuple[int, ...]],
    *,
    seeds: tuple[int, ...] = (11, 29, 47),
    per_class: int = 6,
    iterations: int = 20,
    max_route_share: float = 0.25,
) -> PublicSearchReport:
    """Generate and rank local refits using only explicitly supplied public data."""
    x_train = np.asarray(x_train, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=np.int64)
    x_public = np.asarray(x_public, dtype=np.float64)
    y_public = np.asarray(y_public, dtype=np.int64)
    if x_train.ndim != 2 or x_public.ndim != 2 or x_train.shape[1] != x_public.shape[1]:
        raise ValueError("training and public feature matrices must be compatible")
    if y_train.shape != (len(x_train),) or y_public.shape != (len(x_public),):
        raise ValueError("training and public labels must align")

    cache = build_eval_cache(parent, x_public)
    parent_correct = cache.predictions == y_public
    candidates: list[SearchCandidate] = []
    total_ops = 0
    for raw_categories in category_blocks:
        categories = tuple(sorted(int(value) for value in raw_categories))
        territory = Territory(categories, max_route_share)
        for seed in seeds:
            replaced, added = _training_delta(
                parent, x_train, y_train, categories,
                seed=seed, per_class=per_class, iterations=iterations,
            )
            replacement_centers = {index: center for index, (center, _) in replaced.items()}
            evaluation = evaluate_candidate(parent, cache, replacement_centers, added, territory)
            paired = paired_comparison(parent_correct, evaluation.predictions == y_public)
            total_ops += evaluation.distance_ops
            candidates.append(SearchCandidate(
                categories=categories,
                seed=seed,
                replaced=replaced,
                added=added,
                public_net_wins=paired.net_wins,
                public_p_value=paired.p_value,
                public_delta_active_share=evaluation.delta_active_share,
                untrusted_distance_ops=evaluation.distance_ops,
            ))
    candidates.sort(key=lambda candidate: (candidate.public_net_wins, -candidate.public_p_value), reverse=True)
    return PublicSearchReport(
        candidates_generated=len(candidates),
        candidates_with_positive_public_net_wins=sum(candidate.public_net_wins > 0 for candidate in candidates),
        best_public_net_wins=max((candidate.public_net_wins for candidate in candidates), default=0),
        total_untrusted_distance_ops=total_ops,
        protected_split_accessed=False,
        candidates=tuple(candidates),
    )
