"""Candidate manifests: exact artifact commitments and route-region contracts."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from dendriswarm.core.crypto import content_hash
from dendriswarm.leverage.tissue import Territory, TerritoryTissue, branch_artifact


@dataclass(frozen=True)
class CandidateManifest:
    parent_root: str
    representation_root: str
    replaced: dict[int, dict[str, Any]]
    added: tuple[dict[str, Any], ...]
    territory: Territory
    contributor: str

    def touched_count(self) -> int:
        return len(self.replaced) + len(self.added)

    def as_dict(self, *, include_artifacts: bool = False) -> dict[str, Any]:
        value: dict[str, Any] = {
            "format": "dendriswarm.candidate-manifest.v3.2",
            "parent_root": self.parent_root,
            "representation_root": self.representation_root,
            "replaced": {str(index): artifact["sha256"] for index, artifact in sorted(self.replaced.items())},
            "added": [artifact["sha256"] for artifact in self.added],
            "territory": self.territory.as_dict(),
            "contributor": self.contributor,
        }
        if include_artifacts:
            value["artifacts"] = {
                "replaced": {str(index): dict(artifact) for index, artifact in sorted(self.replaced.items())},
                "added": [dict(artifact) for artifact in self.added],
            }
        commitment_basis = {key: item for key, item in value.items() if key != "artifacts"}
        value["sha256"] = content_hash(commitment_basis)
        return value

    @property
    def commitment(self) -> str:
        return self.as_dict()["sha256"]

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "CandidateManifest":
        artifacts = value.get("artifacts")
        if not isinstance(artifacts, dict):
            raise ValueError("manifest wire form must include branch artifacts")
        territory_value = value.get("territory", {})
        regions = territory_value.get("permitted_route_regions", territory_value.get("permitted_categories"))
        manifest = cls(
            parent_root=str(value["parent_root"]),
            representation_root=str(value["representation_root"]),
            replaced={int(index): dict(artifact) for index, artifact in artifacts.get("replaced", {}).items()},
            added=tuple(dict(artifact) for artifact in artifacts.get("added", [])),
            territory=Territory(tuple(int(item) for item in regions), float(territory_value["max_route_share"])),
            contributor=str(value["contributor"]),
        )
        if value.get("sha256") and value["sha256"] != manifest.commitment:
            raise ValueError("candidate manifest hash mismatch")
        expected_replaced = {str(index): artifact["sha256"] for index, artifact in sorted(manifest.replaced.items())}
        if value.get("replaced") != expected_replaced or value.get("added") != [a["sha256"] for a in manifest.added]:
            raise ValueError("manifest artifact index does not match embedded artifacts")
        for artifact in list(manifest.replaced.values()) + list(manifest.added):
            basis = {key: item for key, item in artifact.items() if key != "sha256"}
            if content_hash(basis) != artifact.get("sha256"):
                raise ValueError("embedded branch artifact hash mismatch")
        return manifest


def build_manifest(
    parent_root: str,
    representation_root: str,
    contributor: str,
    replaced: dict[int, tuple[np.ndarray, int]],
    added: list[tuple[np.ndarray, int]],
    territory: Territory,
) -> CandidateManifest:
    return CandidateManifest(
        parent_root=parent_root,
        representation_root=representation_root,
        replaced={int(index): branch_artifact(center, owner) for index, (center, owner) in replaced.items()},
        added=tuple(branch_artifact(center, owner) for center, owner in added),
        territory=territory,
        contributor=contributor,
    )


def validate_manifest_artifacts(
    manifest: CandidateManifest,
    parent: TerritoryTissue,
    *,
    expected_parent_root: str | None = None,
) -> None:
    """Validate every committed artifact and its exact relationship to parent."""
    if not manifest.contributor:
        raise ValueError("contributor must be non-empty")
    if manifest.touched_count() == 0:
        raise ValueError("candidate delta is empty")
    expected_root = expected_parent_root or parent.root_manifest()["sha256"]
    if manifest.parent_root != expected_root:
        raise ValueError("stale parent root")
    if manifest.representation_root != parent.representation_root():
        raise ValueError("representation root mismatch")

    feature_width = parent.centers.shape[1]
    for index, artifact in sorted(manifest.replaced.items()):
        if index < 0 or index >= len(parent.centers):
            raise ValueError("replaced branch index out of range")
        _validate_artifact(artifact, manifest.territory, feature_width)
        if int(artifact["owner"]) != int(parent.owners[index]):
            raise ValueError("replacement owner must match the parent branch owner")

    for artifact in manifest.added:
        _validate_artifact(artifact, manifest.territory, feature_width)


def _validate_artifact(artifact: dict[str, Any], territory: Territory, feature_width: int) -> None:
    required = {"format", "center", "owner", "sha256"}
    if set(artifact) != required:
        raise ValueError("branch artifact has unexpected or missing fields")
    expected = artifact["sha256"]
    basis = {key: value for key, value in artifact.items() if key != "sha256"}
    if content_hash(basis) != expected:
        raise ValueError("branch artifact hash mismatch")
    if artifact["format"] != "dendriswarm.branch.v3":
        raise ValueError("unsupported branch artifact format")
    center = np.asarray(artifact["center"], dtype=np.float64)
    if center.shape != (feature_width,) or not np.isfinite(center).all():
        raise ValueError("branch artifact has invalid center")
    if int(artifact["owner"]) not in territory.permitted_categories:
        raise ValueError("branch owner outside declared territory")


def manifest_delta(manifest: CandidateManifest) -> tuple[dict[int, np.ndarray], list[tuple[np.ndarray, int]]]:
    """Materialize the exact delta from committed artifacts only."""
    replaced = {
        index: np.asarray(artifact["center"], dtype=np.float64)
        for index, artifact in manifest.replaced.items()
    }
    added = [
        (np.asarray(artifact["center"], dtype=np.float64), int(artifact["owner"]))
        for artifact in manifest.added
    ]
    return replaced, added
