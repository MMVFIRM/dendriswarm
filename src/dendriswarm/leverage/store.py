"""Content-addressed model store with deterministic compositional lineage roots."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from dendriswarm.core.crypto import content_hash
from dendriswarm.leverage.manifest import CandidateManifest, validate_manifest_artifacts
from dendriswarm.leverage.tissue import COMPOSITION_KERNEL_VERSION, TerritoryTissue


@dataclass(frozen=True)
class LineageRecord:
    root: str
    parent_root: str | None
    candidate_manifest: str | None
    representation_root: str
    composition_certificate: str | None


def candidate_lineage_root(manifest: CandidateManifest) -> str:
    """Delta-proportional identity of the exact deterministic composition."""
    return content_hash({
        "format": "dendriswarm.compositional-root.v4.1",
        "composition_kernel": COMPOSITION_KERNEL_VERSION,
        "parent_root": manifest.parent_root,
        "candidate_manifest": manifest.commitment,
        "representation_root": manifest.representation_root,
    })


class ModelStore:
    def __init__(self) -> None:
        self._tissues: dict[str, TerritoryTissue] = {}
        self._lineage: dict[str, LineageRecord] = {}

    def register_genesis(self, tissue: TerritoryTissue) -> str:
        root = tissue.root_manifest()["sha256"]
        existing = self._tissues.get(root)
        if existing is not None and existing.root_manifest() != tissue.root_manifest():
            raise ValueError("root collision")
        self._tissues[root] = tissue
        self._lineage.setdefault(
            root,
            LineageRecord(root, None, None, tissue.representation_root(), None),
        )
        return root

    def expected_candidate_root(self, manifest: CandidateManifest) -> str:
        if manifest.parent_root not in self._tissues:
            raise ValueError("unknown parent root")
        return candidate_lineage_root(manifest)

    def composition_certificate(
        self,
        manifest: CandidateManifest,
        tissue: TerritoryTissue,
    ) -> dict[str, Any]:
        root = self.expected_candidate_root(manifest)
        value: dict[str, Any] = {
            "format": "dendriswarm.composition-certificate.v4.1",
            "composition_kernel": COMPOSITION_KERNEL_VERSION,
            "parent_root": manifest.parent_root,
            "candidate_manifest": manifest.commitment,
            "candidate_root": root,
            "representation_root": manifest.representation_root,
            "territory": manifest.territory.as_dict(),
            "replacement_artifacts": [
                manifest.replaced[index]["sha256"] for index in sorted(manifest.replaced)
            ],
            "added_artifacts": [artifact["sha256"] for artifact in manifest.added],
            "result_branch_count": int(len(tissue.centers)),
            "binding_method": (
                "parent compositional root + manifest-only branch artifacts + "
                "versioned deterministic compose kernel; no behavioral replay"
            ),
        }
        value["sha256"] = content_hash(value)
        return value

    def register_candidate(
        self,
        tissue: TerritoryTissue,
        manifest: CandidateManifest,
        certificate: dict[str, Any],
    ) -> str:
        root = self.expected_candidate_root(manifest)
        if certificate.get("candidate_root") != root:
            raise ValueError("composition certificate names the wrong candidate root")
        basis = {key: value for key, value in certificate.items() if key != "sha256"}
        if content_hash(basis) != certificate.get("sha256"):
            raise ValueError("composition certificate hash mismatch")
        if int(certificate.get("result_branch_count", -1)) != len(tissue.centers):
            raise ValueError("composition certificate branch count mismatch")
        if tissue.representation_root() != manifest.representation_root:
            raise ValueError("materialized tissue representation mismatch")
        # Do not trust a caller-supplied tissue merely because its branch count
        # and representation match. Recompose from the manifest and compare the
        # complete deterministic state before registration.
        expected_tissue = self.materialize(manifest)
        if expected_tissue.root_manifest() != tissue.root_manifest():
            raise ValueError("supplied tissue is not the deterministic manifest composition")
        self._tissues[root] = tissue
        self._lineage[root] = LineageRecord(
            root,
            manifest.parent_root,
            manifest.commitment,
            manifest.representation_root,
            certificate["sha256"],
        )
        return root

    def get(self, root: str) -> TerritoryTissue:
        try:
            return self._tissues[root]
        except KeyError as exc:
            raise ValueError("unknown model root") from exc

    def lineage(self, root: str) -> LineageRecord:
        try:
            return self._lineage[root]
        except KeyError as exc:
            raise ValueError("unknown model root") from exc

    def roots(self) -> tuple[str, ...]:
        return tuple(self._tissues)

    def materialize(self, manifest: CandidateManifest) -> TerritoryTissue:
        """Apply exactly the manifest delta using the committed compose kernel."""
        parent = self.get(manifest.parent_root)
        validate_manifest_artifacts(manifest, parent, expected_parent_root=manifest.parent_root)
        region_mask = np.zeros(parent.classes, dtype=bool)
        region_mask[list(manifest.territory.permitted_categories)] = True

        centers = [row.copy() for row in parent.centers]
        owners = [int(value) for value in parent.owners]
        masks = [row.copy() for row in parent.active_regions]

        for index, artifact in sorted(manifest.replaced.items()):
            old_mask = masks[index].copy()
            new_mask = old_mask & region_mask
            masks[index] = old_mask & ~region_mask
            if new_mask.any():
                centers.append(np.asarray(artifact["center"], dtype=np.float64))
                owners.append(int(artifact["owner"]))
                masks.append(new_mask)

        for artifact in manifest.added:
            centers.append(np.asarray(artifact["center"], dtype=np.float64))
            owners.append(int(artifact["owner"]))
            masks.append(region_mask.copy())

        keep = np.asarray([mask.any() for mask in masks], dtype=bool)
        return TerritoryTissue(
            np.asarray(centers, dtype=np.float64)[keep],
            np.asarray(owners, dtype=np.int64)[keep],
            parent.top_k,
            parent.temperature,
            active_regions=np.asarray(masks, dtype=bool)[keep],
            anchors=parent.anchors,
        )

    def compose(self, manifest: CandidateManifest) -> tuple[TerritoryTissue, dict[str, Any]]:
        tissue = self.materialize(manifest)
        return tissue, self.composition_certificate(manifest, tissue)

    def to_dict(self) -> dict[str, Any]:
        models: dict[str, Any] = {}
        for root, tissue in self._tissues.items():
            state = tissue.root_manifest()
            lineage = self._lineage[root]
            models[root] = {
                "centers": np.round(tissue.centers, 12).tolist(),
                "owners": tissue.owners.astype(int).tolist(),
                "active_regions": tissue.active_regions.astype(bool).tolist(),
                "anchors": np.round(tissue.anchors, 12).tolist(),
                "top_k": tissue.top_k,
                "temperature": tissue.temperature,
                "state_hash": state["sha256"],
                "lineage": {
                    "parent_root": lineage.parent_root,
                    "candidate_manifest": lineage.candidate_manifest,
                    "representation_root": lineage.representation_root,
                    "composition_certificate": lineage.composition_certificate,
                },
            }
        return {"format": "dendriswarm.model-store.v4.1", "models": models}

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ModelStore":
        if value.get("format") not in (None, "dendriswarm.model-store.v4.1"):
            raise ValueError("unsupported model store format")
        store = cls()
        for expected_root, model in value.get("models", {}).items():
            tissue = TerritoryTissue(
                np.asarray(model["centers"], dtype=np.float64),
                np.asarray(model["owners"], dtype=np.int64),
                int(model["top_k"]),
                float(model["temperature"]),
                active_regions=np.asarray(model["active_regions"], dtype=bool),
                anchors=np.asarray(model["anchors"], dtype=np.float64),
            )
            state_hash = tissue.root_manifest()["sha256"]
            if model.get("state_hash", state_hash) != state_hash:
                raise ValueError("persisted model state hash mismatch")
            lineage = model["lineage"]
            parent_root = lineage.get("parent_root")
            manifest_hash = lineage.get("candidate_manifest")
            representation_root = lineage.get("representation_root", tissue.representation_root())
            if parent_root is None:
                if expected_root != state_hash:
                    raise ValueError("persisted genesis root mismatch")
            else:
                expected_composed = content_hash({
                    "format": "dendriswarm.compositional-root.v4.1",
                    "composition_kernel": COMPOSITION_KERNEL_VERSION,
                    "parent_root": parent_root,
                    "candidate_manifest": manifest_hash,
                    "representation_root": representation_root,
                })
                if expected_root != expected_composed:
                    raise ValueError("persisted compositional root mismatch")
            store._tissues[expected_root] = tissue
            store._lineage[expected_root] = LineageRecord(
                expected_root,
                parent_root,
                manifest_hash,
                representation_root,
                lineage.get("composition_certificate"),
            )
        return store

    def snapshot(self) -> dict[str, Any]:
        return {
            "roots": len(self._tissues),
            "lineage": {
                root: {
                    "parent_root": record.parent_root,
                    "candidate_manifest": record.candidate_manifest,
                    "composition_certificate": record.composition_certificate,
                }
                for root, record in self._lineage.items()
            },
        }
