"""Territory-bounded sparse tissue and incremental guarded evaluation.

Canonical branches carry immutable parent-route activation masks. Replacements
split a branch structurally: the old artifact remains active outside the
territory and the replacement artifact is active only inside the intersection
of the old mask and declared route regions. Anchors remain frozen across a
lineage, so route-region membership cannot drift after promotion.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from dendriswarm.core.crypto import content_hash

EVAL_KERNEL_VERSION = "dendriswarm.eval-kernel.v4.1"
COMPOSITION_KERNEL_VERSION = "dendriswarm.compose-kernel.v4.1"


def _rounded_vector(value: np.ndarray) -> list[float]:
    return [float(v) for v in np.round(np.asarray(value, dtype=np.float64), 12)]


def branch_artifact(center: np.ndarray, owner: int) -> dict[str, Any]:
    value = {
        "format": "dendriswarm.branch.v3",
        "center": _rounded_vector(center),
        "owner": int(owner),
    }
    value["sha256"] = content_hash(value)
    return value


@dataclass(frozen=True)
class Territory:
    """Declared frozen parent-route regions where a delta may affect routing."""

    permitted_categories: tuple[int, ...]
    max_route_share: float

    def __post_init__(self) -> None:
        normalized = tuple(int(value) for value in self.permitted_categories)
        if not normalized:
            raise ValueError("territory must name at least one route region")
        if len(set(normalized)) != len(normalized):
            raise ValueError("territory route regions must be unique")
        if not 0 < float(self.max_route_share) <= 1:
            raise ValueError("max_route_share must be in (0, 1]")
        object.__setattr__(self, "permitted_categories", tuple(sorted(normalized)))
        object.__setattr__(self, "max_route_share", float(self.max_route_share))

    @property
    def permitted_route_regions(self) -> tuple[int, ...]:
        return self.permitted_categories

    def as_dict(self) -> dict[str, Any]:
        return {
            "permitted_categories": list(self.permitted_categories),
            "permitted_route_regions": list(self.permitted_categories),
            "max_route_share": self.max_route_share,
        }


class TerritoryTissue:
    """Sparse class-owned prototype tissue with guarded branch activation."""

    def __init__(
        self,
        centers: np.ndarray,
        owners: np.ndarray,
        top_k: int,
        temperature: float,
        *,
        active_regions: np.ndarray | None = None,
        anchors: np.ndarray | None = None,
    ) -> None:
        self.centers = np.asarray(centers, dtype=np.float64).copy()
        self.owners = np.asarray(owners, dtype=np.int64).copy()
        self.top_k = int(top_k)
        self.temperature = float(temperature)
        if self.centers.ndim != 2 or len(self.centers) == 0:
            raise ValueError("centers must be a non-empty matrix")
        if self.owners.shape != (len(self.centers),):
            raise ValueError("owners must align with centers")
        if not np.isfinite(self.centers).all():
            raise ValueError("centers contain non-finite values")
        if (self.owners < 0).any():
            raise ValueError("owners must be non-negative")
        if self.top_k <= 0:
            raise ValueError("top_k must be positive")
        if not np.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("temperature must be finite and positive")

        self.classes = int(self.owners.max()) + 1
        for label in range(self.classes):
            if not (self.owners == label).any():
                raise ValueError(f"class {label} has no owning branch")

        if anchors is None:
            anchor_matrix = np.zeros((self.classes, self.centers.shape[1]), dtype=np.float64)
            for label in range(self.classes):
                anchor_matrix[label] = self.centers[self.owners == label].mean(axis=0)
        else:
            anchor_matrix = np.asarray(anchors, dtype=np.float64)
            if anchor_matrix.shape != (self.classes, self.centers.shape[1]):
                raise ValueError("anchors have incompatible shape")
            if not np.isfinite(anchor_matrix).all():
                raise ValueError("anchors contain non-finite values")
        self.anchors = anchor_matrix.copy()

        if active_regions is None:
            region_matrix = np.ones((len(self.centers), self.classes), dtype=bool)
        else:
            region_matrix = np.asarray(active_regions, dtype=bool)
            if region_matrix.shape != (len(self.centers), self.classes):
                raise ValueError("active_regions have incompatible shape")
        if not region_matrix.any(axis=1).all():
            raise ValueError("every canonical branch must be active in at least one route region")
        self.active_regions = region_matrix.copy()

    def branch_hashes(self) -> list[str]:
        return [
            branch_artifact(self.centers[index], int(self.owners[index]))["sha256"]
            for index in range(len(self.centers))
        ]

    def representation_root(self) -> str:
        return content_hash({
            "format": "dendriswarm.representation.v3.2",
            "kernel": EVAL_KERNEL_VERSION,
            "feature_width": int(self.centers.shape[1]),
            "classes": self.classes,
            "anchor_hash": content_hash({"anchors": [_rounded_vector(row) for row in self.anchors]}),
        })

    def root_manifest(self) -> dict[str, Any]:
        """Full state digest used for genesis and persistence integrity.

        Candidate lineage roots are cheaper compositional hashes maintained by
        ``ModelStore``. This full digest is not recomputed during rejection-path
        binding and is not part of per-candidate challenge evaluation cost.
        """
        value = {
            "format": "dendriswarm.tissue-state.v3.2",
            "branch_hashes": self.branch_hashes(),
            "owners": [int(value) for value in self.owners],
            "active_route_regions": [
                np.flatnonzero(mask).astype(int).tolist() for mask in self.active_regions
            ],
            "anchors": [_rounded_vector(row) for row in self.anchors],
            "top_k": self.top_k,
            "temperature": self.temperature,
            "representation_root": self.representation_root(),
            "classes": self.classes,
        }
        value["sha256"] = content_hash(value)
        return value

    def canonical_categories(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != self.centers.shape[1]:
            raise ValueError("input matrix has incompatible feature width")
        d2 = ((x[:, None, :] - self.anchors[None, :, :]) ** 2).sum(axis=2)
        return d2.argmin(axis=1)

    def raw_distance_matrix(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != self.centers.shape[1]:
            raise ValueError("input matrix has incompatible feature width")
        return ((x[:, None, :] - self.centers[None, :, :]) ** 2).sum(axis=2)

    def distance_matrix(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        canonical = self.canonical_categories(x)
        d2 = self.raw_distance_matrix(x)
        active = self.active_regions[:, canonical].T
        d2[~active] = np.inf
        return d2

    def _predict_from_distances(self, d2: np.ndarray, owners: np.ndarray) -> np.ndarray:
        k = min(max(1, self.top_k), d2.shape[1])
        order = np.argsort(d2, axis=1, kind="stable")[:, :k]
        selected = np.take_along_axis(d2, order, axis=1)
        n = d2.shape[0]
        out = np.zeros((n, self.classes), dtype=np.float64)
        rows = np.arange(n)
        evidence = np.exp(-selected / self.temperature)
        evidence[~np.isfinite(selected)] = 0.0
        for slot in range(k):
            np.add.at(out, (rows, owners[order[:, slot]]), evidence[:, slot])
        return out.argmax(axis=1)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self._predict_from_distances(self.distance_matrix(x), self.owners)

    def accuracy(self, x: np.ndarray, y: np.ndarray) -> float:
        return float((self.predict(x) == np.asarray(y)).mean())


@dataclass(frozen=True)
class EvalCache:
    x: np.ndarray
    d2: np.ndarray
    canonical: np.ndarray
    predictions: np.ndarray
    distance_ops: int
    bytes_materialized: int
    selection_items: int


def build_eval_cache(parent: TerritoryTissue, x: np.ndarray) -> EvalCache:
    x = np.asarray(x, dtype=np.float64)
    canonical = parent.canonical_categories(x)
    raw = parent.raw_distance_matrix(x)
    active = parent.active_regions[:, canonical].T
    raw[~active] = np.inf
    return EvalCache(
        x=x,
        d2=raw,
        canonical=canonical,
        predictions=parent._predict_from_distances(raw, parent.owners),
        distance_ops=int(raw.shape[0] * raw.shape[1]),
        bytes_materialized=int(raw.nbytes),
        selection_items=int(raw.shape[0] * raw.shape[1]),
    )


@dataclass(frozen=True)
class CandidateEvaluation:
    predictions: np.ndarray
    candidate_cache: EvalCache
    candidate_owners: np.ndarray
    delta_active_share: float
    outside_discordant: int
    distance_ops: int
    guard_masked_inputs: int
    bytes_copied: int
    selection_items: int
    aggregation_items: int
    candidate_branch_count: int


def evaluate_candidate(
    parent: TerritoryTissue,
    cache: EvalCache,
    replaced: dict[int, np.ndarray],
    added: Sequence[tuple[np.ndarray, int]],
    territory: Territory,
) -> CandidateEvaluation:
    """Incrementally evaluate and construct the exact next-root challenge cache.

    Only touched branch distances are recomputed. The returned cache has the
    exact branch ordering and route masks of deterministic canonical
    composition, so an accepted candidate becomes the next parent without a
    second full challenge evaluation.
    """
    n, branches = cache.d2.shape
    permitted = np.zeros(parent.classes, dtype=bool)
    for category in territory.permitted_categories:
        if category < 0 or category >= parent.classes:
            raise ValueError("territory names an unknown route region")
        permitted[category] = True
    inside = permitted[cache.canonical]

    base = cache.d2.copy()
    operations = 0
    guard_masked = int((~inside).sum()) if (replaced or added) else 0
    region_mask = permitted
    old_keep = np.ones(branches, dtype=bool)
    appended_columns: list[np.ndarray] = []
    appended_owners: list[int] = []

    def distances(centers: np.ndarray) -> np.ndarray:
        return ((cache.x[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)

    replacement_indices = sorted(int(index) for index in replaced)
    if replacement_indices:
        index_array = np.asarray(replacement_indices, dtype=np.int64)
        if (index_array < 0).any() or (index_array >= branches).any():
            raise ValueError("replaced branch index out of range")
        centers = np.stack([np.asarray(replaced[index], dtype=np.float64) for index in replacement_indices])
        if centers.shape[1] != parent.centers.shape[1] or not np.isfinite(centers).all():
            raise ValueError("replacement centers have invalid shape or values")
        fresh = distances(centers)
        operations += fresh.size
        for column, index in enumerate(replacement_indices):
            old_regions = parent.active_regions[index]
            replacement_regions = old_regions & region_mask
            old_regions_after = old_regions & ~region_mask
            old_keep[index] = bool(old_regions_after.any())
            active_samples = replacement_regions[cache.canonical]
            base[inside & old_regions[cache.canonical], index] = np.inf
            replacement_d2 = fresh[:, column].copy()
            replacement_d2[~active_samples] = np.inf
            if replacement_regions.any():
                appended_columns.append(replacement_d2)
                appended_owners.append(int(parent.owners[index]))

    if added:
        centers = np.stack([np.asarray(center, dtype=np.float64) for center, _ in added])
        if centers.shape[1] != parent.centers.shape[1] or not np.isfinite(centers).all():
            raise ValueError("added centers have invalid shape or values")
        added_owners = np.asarray([int(owner) for _, owner in added], dtype=np.int64)
        if (added_owners < 0).any() or (added_owners >= parent.classes).any():
            raise ValueError("added owner outside model classes")
        fresh = distances(centers)
        operations += fresh.size
        fresh[~inside, :] = np.inf
        for column, owner in enumerate(added_owners):
            appended_columns.append(fresh[:, column].copy())
            appended_owners.append(int(owner))

    canonical_d2 = base[:, old_keep]
    canonical_owners = parent.owners[old_keep].copy()
    if appended_columns:
        canonical_d2 = np.concatenate([canonical_d2, np.stack(appended_columns, axis=1)], axis=1)
        canonical_owners = np.concatenate([canonical_owners, np.asarray(appended_owners, dtype=np.int64)])

    predictions = parent._predict_from_distances(canonical_d2, canonical_owners)
    k = min(max(1, parent.top_k), canonical_d2.shape[1])
    candidate_order = np.argsort(canonical_d2, axis=1, kind="stable")[:, :k]

    # Route-share is based on any delta-caused routing or behavioral change, not
    # merely whether an appended branch appears in top-k. Canonical columns for
    # inherited branches retain their original identity even when earlier
    # branches are masked or removed; appended replacement/addition identities
    # are distinct. This catches the removal case where an untouched inherited
    # branch enters top-k because a replaced branch disappeared.
    parent_k = min(max(1, parent.top_k), cache.d2.shape[1])
    parent_order = np.argsort(cache.d2, axis=1, kind="stable")[:, :parent_k]
    inherited_ids = np.flatnonzero(old_keep).astype(np.int64)
    appended_count = canonical_d2.shape[1] - len(inherited_ids)
    candidate_ids = np.concatenate([
        inherited_ids,
        np.arange(branches, branches + appended_count, dtype=np.int64),
    ])
    candidate_route_ids = candidate_ids[candidate_order]
    if parent_order.shape[1] == candidate_route_ids.shape[1]:
        route_changed = (parent_order != candidate_route_ids).any(axis=1)
    else:
        route_changed = np.ones(n, dtype=bool)
    behavior_changed = predictions != cache.predictions
    delta_active = inside & (route_changed | behavior_changed)
    outside_discordant = int((predictions[~inside] != cache.predictions[~inside]).sum())
    candidate_cache = EvalCache(
        x=cache.x,
        d2=canonical_d2,
        canonical=cache.canonical,
        predictions=predictions,
        distance_ops=cache.distance_ops + int(operations),
        bytes_materialized=int(canonical_d2.nbytes),
        selection_items=int(canonical_d2.shape[0] * canonical_d2.shape[1]),
    )
    return CandidateEvaluation(
        predictions=predictions,
        candidate_cache=candidate_cache,
        candidate_owners=canonical_owners,
        delta_active_share=float(delta_active.mean()),
        outside_discordant=outside_discordant,
        distance_ops=int(operations),
        guard_masked_inputs=guard_masked,
        bytes_copied=int(base.nbytes + sum(column.nbytes for column in appended_columns)),
        selection_items=int(canonical_d2.shape[0] * canonical_d2.shape[1]),
        aggregation_items=int(canonical_d2.shape[0] * k),
        candidate_branch_count=int(canonical_d2.shape[1]),
    )
