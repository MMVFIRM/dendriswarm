"""Locality Leverage orchestration with measurable predicates and final replication."""
from __future__ import annotations

import json
import os
import tempfile
import threading
import time
from contextlib import contextmanager
from functools import wraps
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from dendriswarm.core.crypto import content_hash
from dendriswarm.leverage.epoch import ChallengeEpoch
from dendriswarm.leverage.ledger import LeverageLedger
from dendriswarm.leverage.manifest import CandidateManifest, manifest_delta, validate_manifest_artifacts
from dendriswarm.leverage.stats import (
    GatePolicy,
    paired_comparison,
    subgroup_comparisons,
    worst_subgroup_net_wins,
)
from dendriswarm.leverage.store import ModelStore
from dendriswarm.leverage.tissue import CandidateEvaluation, EvalCache, TerritoryTissue, build_eval_cache, evaluate_candidate


def atomic_state_method(method):
    """Serialize and roll back a complete leverage mutation on failure."""
    @wraps(method)
    def wrapped(self, *args, **kwargs):
        with self._atomic_mutation():
            return method(self, *args, **kwargs)
    return wrapped


@dataclass
class CandidateRecord:
    manifest: CandidateManifest
    verdict: str
    detail: dict[str, Any]
    status: str
    parent_root: str
    candidate_root: str | None = None
    reward_locked: int = 0
    bond_locked: int = 0
    canary_observations: list[dict[str, Any]] = field(default_factory=list)
    canary_violation: str = ""

    @property
    def canary_events(self) -> int:
        return len(self.canary_observations)


@dataclass
class LeverageService:
    parent: TerritoryTissue
    epoch: ChallengeEpoch
    policy: GatePolicy | None = None
    ledger: LeverageLedger = field(default_factory=LeverageLedger)
    store: ModelStore = field(default_factory=ModelStore)
    state_path: str | Path | None = None
    candidates: dict[str, CandidateRecord] = field(default_factory=dict)
    audit: list[dict[str, Any]] = field(default_factory=list)
    canonical_root: str = ""
    genesis_root: str = ""
    cache: EvalCache | None = None
    active_canary_id: str | None = None
    funded_principals: dict[str, str] = field(default_factory=dict)
    contributor_principals: dict[str, str] = field(default_factory=dict)
    trusted_wall_seconds: float = 0.0
    evaluation_costs: list[dict[str, Any]] = field(default_factory=list)
    promotion_costs: list[dict[str, Any]] = field(default_factory=list)
    canary_costs: list[dict[str, Any]] = field(default_factory=list)
    replication_report: dict[str, Any] | None = None
    search_evidence: dict[str, Any] | None = None
    _cache_by_root: dict[str, EvalCache] = field(default_factory=dict, repr=False)
    _state_lock: threading.RLock = field(default_factory=threading.RLock, repr=False, compare=False)
    _mutation_depth: int = field(default=0, repr=False, compare=False)
    _dirty: bool = field(default=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.policy is None:
            self.policy = self.epoch.policy
        elif self.policy.registration_hash != self.epoch.policy.registration_hash:
            raise ValueError("service policy does not match the committed epoch policy")
        self.policy = self.epoch.policy
        self.genesis_root = self.store.register_genesis(self.parent)
        self.canonical_root = self.genesis_root
        self._rebuild_cache(reason="genesis")
        self._append_audit("epoch_opened", {
            "commitment": self.epoch.commitment,
            "policy_hash": self.policy.registration_hash,
            "genesis_root": self.genesis_root,
            "identity_assumption": self.policy.identity_assumption,
            "replication_holdout_committed": self.epoch.replication_hash,
        })
        self._persist()

    @property
    def canonical_tissue(self) -> TerritoryTissue:
        return self.store.get(self.canonical_root)

    # ---------------------------------------------------------------- audit

    def _append_audit(self, kind: str, payload: dict[str, Any]) -> str:
        previous = self.audit[-1]["hash"] if self.audit else "genesis"
        event = {"kind": kind, "payload": payload, "previous": previous}
        event["hash"] = content_hash(event)
        self.audit.append(event)
        return event["hash"]

    def validate_audit_chain(self) -> bool:
        previous = "genesis"
        for event in self.audit:
            expected = content_hash({key: value for key, value in event.items() if key != "hash"})
            if event["previous"] != previous or event["hash"] != expected:
                return False
            previous = event["hash"]
        return True

    def audit_checkpoint(self) -> dict[str, Any]:
        return {
            "format": "dendriswarm.leverage-audit-checkpoint.v3.2",
            "valid": self.validate_audit_chain(),
            "events": len(self.audit),
            "head": self.audit[-1]["hash"] if self.audit else "genesis",
            "canonical_root": self.canonical_root,
            "epoch_closed": self.epoch.closed,
        }

    # ---------------------------------------------------------------- identity and funding

    @atomic_state_method
    def register_contributor(self, contributor: str) -> None:
        """Create an account without minting credits.

        Permissionless strings do not receive grants. Testnet grants require a
        separately verified principal through ``fund_contributor``.
        """
        if not contributor:
            raise ValueError("contributor must be non-empty")
        self.ledger.account(contributor)
        self._append_audit("contributor_registered", {"contributor": contributor, "grant": 0})
        self._persist()

    @atomic_state_method
    def fund_contributor(
        self,
        contributor: str,
        verified_principal: str,
        units: int | None = None,
    ) -> None:
        """Issue at most one grant per externally verified principal."""
        if not contributor or not verified_principal:
            raise ValueError("contributor and verified principal must be non-empty")
        existing_contributor = self.funded_principals.get(verified_principal)
        if existing_contributor is not None and existing_contributor != contributor:
            raise ValueError("verified principal has already funded another contributor")
        existing_principal = self.contributor_principals.get(contributor)
        if existing_principal is not None and existing_principal != verified_principal:
            raise ValueError("contributor has already used another verified principal")
        grant = self.policy.grant_units if units is None else int(units)
        if grant < 0:
            raise ValueError("grant units must be non-negative")
        self.ledger.grant(f"verified-grant:{verified_principal}", contributor, grant)
        self.funded_principals[verified_principal] = contributor
        self.contributor_principals[contributor] = verified_principal
        self._append_audit("contributor_funded", {
            "contributor": contributor,
            "principal_hash": content_hash({"principal": verified_principal}),
            "units": grant,
            "assumption": self.policy.identity_assumption,
        })
        self._persist()

    # ---------------------------------------------------------------- submission

    @atomic_state_method
    def submit(self, manifest: CandidateManifest, authenticated_contributor: str) -> tuple[str, str]:
        candidate_id = manifest.commitment
        existing = self.candidates.get(candidate_id)
        if existing is not None:
            # A stale result is contextual, not permanent. If rollback later
            # restores its parent, the same committed candidate may be admitted.
            if existing.status == "stale" and manifest.parent_root == self.canonical_root:
                del self.candidates[candidate_id]
            else:
                return candidate_id, existing.verdict
        if self.epoch.closed:
            raise ValueError("epoch is closed")
        if authenticated_contributor != manifest.contributor:
            raise ValueError("authenticated contributor does not match manifest contributor")
        if self.active_canary_id is not None:
            verdict = "rejected:canary-window-active"
            self._append_audit("candidate_deferred", {
                "candidate": candidate_id,
                "active_canary": self.active_canary_id,
                "verdict": verdict,
                "charged": False,
            })
            self._persist()
            return candidate_id, verdict

        # Staleness is a race condition, not evidence of malicious behavior.
        # It consumes no protected challenge test and never locks or burns a bond.
        if manifest.parent_root != self.canonical_root:
            contributor = manifest.contributor
            stale_fee = self.policy.stale_submission_fee_units
            if stale_fee:
                if self.ledger.account(contributor).available < stale_fee:
                    raise ValueError("insufficient available balance for stale-submission fee")
                self.ledger.burn_fee(f"stale-fee:{candidate_id}", contributor, stale_fee)
            self.epoch.note_stale_submission()
            verdict = "rejected:stale-parent"
            self.candidates[candidate_id] = CandidateRecord(
                manifest, verdict,
                {"reason": "parent root is no longer canonical", "bond_burned": 0, "fee": stale_fee},
                "stale", manifest.parent_root,
            )
            self._append_audit("candidate_stale", {
                "candidate": candidate_id,
                "submitted_parent": manifest.parent_root,
                "canonical_parent": self.canonical_root,
                "fee": stale_fee,
                "bond_burned": 0,
            })
            self._persist()
            return candidate_id, verdict

        contributor = manifest.contributor
        fee = self.epoch.next_fee(contributor)
        account = self.ledger.account(contributor)
        if account.available < fee + self.policy.bond_units:
            raise ValueError("insufficient available balance for fee and bond")

        try:
            charged_fee = self.epoch.admit_submission(contributor)
        except ValueError as error:
            verdict = (
                "rejected:contributor-budget-exhausted"
                if "contributor" in str(error)
                else "rejected:submission-budget-exhausted"
            )
            self.candidates[candidate_id] = CandidateRecord(
                manifest, verdict, {"reason": str(error)}, "rejected", manifest.parent_root
            )
            self._append_audit("candidate_rejected", {"candidate": candidate_id, "verdict": verdict})
            self._persist()
            return candidate_id, verdict

        self.ledger.burn_fee(f"fee:{candidate_id}", contributor, charged_fee)
        self.ledger.lock_bond(f"bond:{candidate_id}", contributor, self.policy.bond_units)

        verdict, detail, evaluation = self._evaluate(manifest)
        accepted = verdict == "accepted:promotion-candidate"
        record = CandidateRecord(
            manifest=manifest,
            verdict=verdict,
            detail=detail,
            status="promoted" if accepted else "rejected",
            parent_root=manifest.parent_root,
            bond_locked=self.policy.bond_units,
        )
        self.candidates[candidate_id] = record

        if accepted:
            assert evaluation is not None
            compose_started = time.perf_counter()
            candidate_tissue, certificate = self.store.compose(manifest)
            compose_wall = time.perf_counter() - compose_started
            if len(candidate_tissue.centers) != evaluation.candidate_branch_count:
                raise RuntimeError("incremental cache and deterministic composition disagree on branch count")
            if not np.array_equal(candidate_tissue.owners, evaluation.candidate_owners):
                raise RuntimeError("incremental cache and deterministic composition disagree on branch owners")
            candidate_root = certificate["candidate_root"]
            stored_root = self.store.register_candidate(candidate_tissue, manifest, certificate)
            if stored_root != candidate_root:
                raise RuntimeError("candidate root changed during registration")

            record.candidate_root = candidate_root
            record.detail.update({
                "candidate_root": candidate_root,
                "artifact_binding_verified": True,
                "binding_method": certificate["binding_method"],
                "composition_certificate": certificate,
                "behavioral_binding_full_passes": 0,
            })
            self.ledger.lock_reward(f"reward:{candidate_id}", contributor, self.policy.reward_units)
            record.reward_locked = self.policy.reward_units
            self.canonical_root = candidate_root
            self.cache = evaluation.candidate_cache
            self._cache_by_root[candidate_root] = self.cache
            self.active_canary_id = candidate_id
            promotion_cost = {
                "candidate": candidate_id,
                "composition_wall_seconds": compose_wall,
                "behavioral_binding_full_passes": 0,
                "challenge_cache_rebuild_distance_ops": 0,
                "cache_source": "incremental accepted-candidate cache",
            }
            self.promotion_costs.append(promotion_cost)
            self.trusted_wall_seconds += compose_wall
            self._append_audit("candidate_promoted", {
                "candidate": candidate_id,
                "contributor": contributor,
                "parent_root": record.parent_root,
                "new_root": self.canonical_root,
                "composition_certificate": certificate["sha256"],
                "binding_full_inference_passes": 0,
                "canary_window_events": self.policy.canary_window_events,
                "reward_state": "locked-pending-canary-and-final-replication",
            })
        else:
            self.ledger.burn_locked(f"bond-burn:{candidate_id}", contributor, self.policy.bond_units)
            record.bond_locked = 0
            self._append_audit("candidate_rejected", {
                "candidate": candidate_id,
                "verdict": verdict,
                "reason": detail.get("reason", ""),
            })
        self._persist()
        return candidate_id, verdict

    def _evaluate(self, manifest: CandidateManifest) -> tuple[str, dict[str, Any], CandidateEvaluation | None]:
        assert self.cache is not None
        parent = self.canonical_tissue
        detail: dict[str, Any] = {}
        try:
            validate_manifest_artifacts(manifest, parent, expected_parent_root=self.canonical_root)
        except ValueError as error:
            return "rejected:territorial-violation", {"reason": str(error)}, None

        territory = manifest.territory
        if len(territory.permitted_categories) > self.policy.max_territory_categories:
            return "rejected:territorial-violation", {"reason": "territory exceeds proposal class"}, None
        if manifest.touched_count() > self.policy.max_touched_branches:
            return "rejected:territorial-violation", {"reason": "delta exceeds touched-branch ceiling"}, None
        if territory.max_route_share > self.policy.max_route_share:
            return "rejected:route-share-violation", {"reason": "declared share above policy ceiling"}, None

        replaced, added = manifest_delta(manifest)
        started = time.perf_counter()
        evaluation = evaluate_candidate(parent, self.cache, replaced, added, territory)
        wall = time.perf_counter() - started
        self.trusted_wall_seconds += wall

        counterfactual_ops = int(len(self.epoch.x) * evaluation.candidate_branch_count)
        cost = {
            "candidate": manifest.commitment,
            "trusted_distance_ops": evaluation.distance_ops,
            "counterfactual_full_candidate_distance_ops": counterfactual_ops,
            "distance_ratio": evaluation.distance_ops / counterfactual_ops if counterfactual_ops else 0.0,
            "bytes_copied": evaluation.bytes_copied,
            "selection_items": evaluation.selection_items,
            "aggregation_items": evaluation.aggregation_items,
            "trusted_verification_wall_seconds": wall,
            "behavioral_binding_full_passes": 0,
            "binding_cost_boundary": "versioned deterministic composition; no second prediction pass",
        }
        self.evaluation_costs.append(cost)
        detail.update({
            "delta_active_share": evaluation.delta_active_share,
            "distance_ops": evaluation.distance_ops,
            "outside_discordant": evaluation.outside_discordant,
            "cost": cost,
        })

        if evaluation.outside_discordant != 0:
            return "rejected:territorial-violation", detail, None
        if evaluation.delta_active_share > territory.max_route_share:
            return "rejected:route-share-violation", detail, None

        parent_correct = self.cache.predictions == self.epoch.y
        candidate_correct = evaluation.predictions == self.epoch.y
        paired = paired_comparison(parent_correct, candidate_correct)
        subgroup = subgroup_comparisons(
            self.epoch.y,
            parent_correct,
            candidate_correct,
            min_samples=self.policy.min_subgroup_samples,
        )
        worst_group, worst_net = worst_subgroup_net_wins(subgroup)
        detail["paired"] = paired.as_dict()
        detail["subgroups"] = subgroup
        detail["worst_subgroup"] = {"id": worst_group, "net_wins": worst_net}

        if paired.net_wins < 0:
            return "rejected:protected-challenge-regression", detail, None
        if worst_net < -self.policy.max_challenge_subgroup_net_loss:
            detail["reason"] = "challenge improvement concentrates excessive net loss in a subgroup"
            return "rejected:subgroup-harm", detail, None
        if paired.p_value > self.policy.per_test_alpha:
            return "rejected:no-significant-improvement", detail, None
        if paired.net_wins < self.policy.min_net_wins:
            return "rejected:below-effect-size-floor", detail, None
        return "accepted:promotion-candidate", detail, evaluation

    # ---------------------------------------------------------------- canary

    @atomic_state_method
    def record_canary_batch(
        self,
        candidate_id: str,
        x: np.ndarray,
        y: np.ndarray,
        *,
        source_id: str,
        source_kind: str,
        subgroup_ids: np.ndarray | None = None,
    ) -> str:
        """Evaluate a concrete, hash-committed labeled observation batch.

        Cleanliness is computed from parent-versus-candidate outcomes under the
        committed aggregate and subgroup thresholds. The caller supplies data,
        not a Boolean verdict. Every batch and source identifier is auditable.
        """
        if candidate_id != self.active_canary_id:
            raise ValueError("candidate is not the active canary")
        record = self.candidates[candidate_id]
        if record.status != "promoted" or record.candidate_root is None:
            raise ValueError("candidate is not in a canary window")
        if source_kind not in self.policy.canary_source_kinds:
            raise ValueError("unsupported canary source kind")
        if not source_id:
            raise ValueError("canary source_id must be non-empty")
        features = np.asarray(x, dtype=np.float64)
        labels = np.asarray(y, dtype=np.int64)
        if features.ndim != 2 or labels.shape != (len(features),):
            raise ValueError("canary features and labels must align")
        if len(features) < self.policy.canary_min_batch_samples:
            raise ValueError("canary batch is below the committed minimum sample count")
        candidate_classes = self.store.get(record.candidate_root).classes
        if features.shape[1] != self.canonical_tissue.centers.shape[1] or not np.isfinite(features).all():
            raise ValueError("canary batch has invalid features")
        if (labels < 0).any() or (labels >= candidate_classes).any():
            raise ValueError("canary labels are outside the committed model class schema")
        groups = labels if subgroup_ids is None else np.asarray(subgroup_ids)
        if groups.shape != labels.shape:
            raise ValueError("canary subgroup ids must align with labels")

        data_basis = {
            "format": "dendriswarm.canary-data.v3.2",
            "features": np.round(features, 12).tolist(),
            "labels": labels.astype(int).tolist(),
            "subgroups": groups.tolist(),
        }
        data_hash = content_hash(data_basis)
        observation_basis = {
            "format": "dendriswarm.canary-observation.v3.2",
            "candidate": candidate_id,
            "source_id": source_id,
            "source_kind": source_kind,
            "data_hash": data_hash,
        }
        observation_hash = content_hash(observation_basis)
        if any(item.get("data_hash") == data_hash for item in record.canary_observations):
            raise ValueError("duplicate canary observation data")

        parent = self.store.get(record.parent_root)
        candidate = self.store.get(record.candidate_root)
        started = time.perf_counter()
        parent_predictions = parent.predict(features)
        candidate_predictions = candidate.predict(features)
        wall = time.perf_counter() - started
        self.trusted_wall_seconds += wall
        parent_correct = parent_predictions == labels
        candidate_correct = candidate_predictions == labels
        paired = paired_comparison(parent_correct, candidate_correct)
        subgroup = subgroup_comparisons(
            groups,
            parent_correct,
            candidate_correct,
            min_samples=self.policy.min_subgroup_samples,
        )
        worst_group, worst_net = worst_subgroup_net_wins(subgroup)
        violation_reasons: list[str] = []
        informative_predictions = int((parent_correct | candidate_correct).sum())
        discordant = int(paired.wins + paired.losses)
        if informative_predictions < self.policy.canary_min_informative_predictions:
            violation_reasons.append("non-informative canary: neither model is correct often enough")
        if discordant < self.policy.canary_min_discordant:
            violation_reasons.append("non-informative canary: insufficient parent-candidate discordance")
        if paired.net_wins < self.policy.canary_min_net_wins:
            violation_reasons.append("aggregate canary regression")
        if worst_net < -self.policy.canary_max_subgroup_net_loss:
            violation_reasons.append(f"subgroup {worst_group} canary regression")

        observation = {
            "observation_hash": observation_hash,
            "data_hash": data_hash,
            "source_id": source_id,
            "source_kind": source_kind,
            "samples": int(len(features)),
            "paired": paired.as_dict(),
            "informative_predictions": informative_predictions,
            "discordant_predictions": discordant,
            "subgroups": subgroup,
            "worst_subgroup": {"id": worst_group, "net_wins": worst_net},
            "clean": not violation_reasons,
            "violations": violation_reasons,
        }
        record.canary_observations.append(observation)
        cost = {
            "candidate": candidate_id,
            "observation_hash": observation_hash,
            "distance_ops": int(len(features) * (len(parent.centers) + len(candidate.centers))),
            "wall_seconds": wall,
        }
        self.canary_costs.append(cost)
        self._append_audit("canary_observation", {
            "candidate": candidate_id,
            "observation": observation,
            "predicate": {
                "min_net_wins": self.policy.canary_min_net_wins,
                "min_informative_predictions": self.policy.canary_min_informative_predictions,
                "min_discordant": self.policy.canary_min_discordant,
                "max_subgroup_net_loss": self.policy.canary_max_subgroup_net_loss,
            },
        })

        if violation_reasons:
            record.canary_violation = "; ".join(violation_reasons)
            self._revoke(record, candidate_id)
            self._persist()
            return "revoked"
        if record.canary_events >= self.policy.canary_window_events:
            record.status = "canary-passed"
            self.active_canary_id = None
            self._append_audit("candidate_canary_passed", {
                "candidate": candidate_id,
                "observations": record.canary_events,
                "reward_state": "locked-pending-final-replication",
            })
            self._persist()
            return "canary-passed"
        self._persist()
        return "pending"

    def record_canary_event(self, *args: Any, **kwargs: Any) -> str:
        raise TypeError(
            "Boolean canary events were removed in v0.3.2; use record_canary_batch "
            "with a concrete labeled, source-identified observation batch"
        )

    def _revoke(self, record: CandidateRecord, candidate_id: str) -> None:
        contributor = record.manifest.contributor
        self.ledger.slash(
            f"slash:{candidate_id}", contributor,
            record.bond_locked + record.reward_locked,
        )
        record.status = "revoked"
        self.canonical_root = record.parent_root
        self.active_canary_id = None
        restored = self._cache_by_root.get(self.canonical_root)
        if restored is None:
            self._rebuild_cache(reason="canary-rollback")
        else:
            self.cache = restored
        self._append_audit("candidate_revoked", {
            "candidate": candidate_id,
            "reason": record.canary_violation,
            "restored_root": self.canonical_root,
        })

    # ---------------------------------------------------------------- final replication

    @atomic_state_method
    def close_epoch(self) -> dict[str, Any]:
        """Run the one-shot genesis-versus-final replication and settle rewards."""
        if self.epoch.closed:
            if self.replication_report is None:
                raise ValueError("epoch is closed without a replication report")
            return self.replication_report
        if self.active_canary_id is not None:
            raise ValueError("cannot close epoch while a canary window is active")

        genesis = self.store.get(self.genesis_root)
        final = self.canonical_tissue
        started = time.perf_counter()
        genesis_predictions = genesis.predict(self.epoch.replication_x)
        final_predictions = final.predict(self.epoch.replication_x)
        wall = time.perf_counter() - started
        self.trusted_wall_seconds += wall
        genesis_correct = genesis_predictions == self.epoch.replication_y
        final_correct = final_predictions == self.epoch.replication_y
        paired = paired_comparison(genesis_correct, final_correct)
        informative_predictions = int((genesis_correct | final_correct).sum())
        discordant = int(paired.wins + paired.losses)
        subgroup = subgroup_comparisons(
            self.epoch.replication_y,
            genesis_correct,
            final_correct,
            min_samples=self.policy.min_subgroup_samples,
        )
        worst_group, worst_net = worst_subgroup_net_wins(subgroup)
        passed = (
            paired.net_wins >= self.policy.replication_min_net_wins
            and paired.p_value <= self.policy.replication_alpha
            and worst_net >= -self.policy.replication_max_subgroup_net_loss
        )
        eligible = [
            (candidate_id, record)
            for candidate_id, record in self.candidates.items()
            if record.status == "canary-passed"
        ]
        if passed:
            for candidate_id, record in eligible:
                contributor = record.manifest.contributor
                self.ledger.vest(f"vest-bond:{candidate_id}", contributor, record.bond_locked)
                self.ledger.vest(f"vest-reward:{candidate_id}", contributor, record.reward_locked)
                record.status = "vested"
        else:
            for candidate_id, record in eligible:
                contributor = record.manifest.contributor
                self.ledger.refund_bond(f"refund-bond:{candidate_id}", contributor, record.bond_locked)
                self.ledger.cancel_reward(f"cancel-reward:{candidate_id}", contributor, record.reward_locked)
                record.status = "replication-rejected"
            self.canonical_root = self.genesis_root
            self._rebuild_cache(reason="replication-rollback")

        report = {
            "format": "dendriswarm.final-replication.v4.1",
            "genesis_root": self.genesis_root,
            "submitted_final_root": final.root_manifest()["sha256"],
            "canonical_lineage_root_before_settlement": (
                eligible[-1][1].candidate_root if eligible else self.canonical_root
            ),
            "canonical_root_after_settlement": self.canonical_root,
            "paired": paired.as_dict(),
            "informative_predictions": informative_predictions,
            "discordant_predictions": discordant,
            "subgroups": subgroup,
            "worst_subgroup": {"id": worst_group, "net_wins": worst_net},
            "policy": {
                "min_net_wins": self.policy.replication_min_net_wins,
                "alpha": self.policy.replication_alpha,
                "max_subgroup_net_loss": self.policy.replication_max_subgroup_net_loss,
            },
            "passed": passed,
            "settled_candidates": [candidate_id for candidate_id, _ in eligible],
            "wall_seconds": wall,
            "distance_ops": int(len(self.epoch.replication_x) * (len(genesis.centers) + len(final.centers))),
        }
        self.replication_report = report
        self.epoch.close(report)
        self._append_audit("epoch_final_replication", report)
        self._persist()
        return report

    @atomic_state_method
    def reveal_epoch(self) -> dict[str, Any]:
        disclosure = self.epoch.reveal()
        self._append_audit("epoch_revealed", {
            "commitment": disclosure["commitment"],
            "replication_passed": disclosure["replication_result"]["passed"],
        })
        self._persist()
        return disclosure

    # ---------------------------------------------------------------- cache and persistence

    def _rebuild_cache(self, *, reason: str) -> None:
        started = time.perf_counter()
        self.cache = build_eval_cache(self.canonical_tissue, self.epoch.x)
        wall = time.perf_counter() - started
        self.trusted_wall_seconds += wall
        self._cache_by_root[self.canonical_root] = self.cache
        if reason != "genesis":
            self.promotion_costs.append({
                "candidate": None,
                "reason": reason,
                "composition_wall_seconds": 0.0,
                "behavioral_binding_full_passes": 0,
                "challenge_cache_rebuild_distance_ops": self.cache.distance_ops,
                "cache_rebuild_wall_seconds": wall,
            })

    @atomic_state_method
    def record_search_evidence(self, evidence: dict[str, Any]) -> None:
        """Attach a public-data-only untrusted search report to the audit trail."""
        self.search_evidence = dict(evidence)
        self._append_audit("untrusted_search_evidence", self.search_evidence)
        self._persist()

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": "dendriswarm.leverage-state.v3.2",
            "epoch": {
                "x": np.round(self.epoch.x, 12).tolist(),
                "y": self.epoch.y.astype(int).tolist(),
                "replication_x": np.round(self.epoch.replication_x, 12).tolist(),
                "replication_y": self.epoch.replication_y.astype(int).tolist(),
                "policy": self.policy.as_dict(),
                "salt": self.epoch.salt,
                "submissions": dict(self.epoch.submissions),
                "tests_spent": self.epoch.tests_spent,
                "stale_submissions": self.epoch.stale_submissions,
                "closed": self.epoch.closed,
                "revealed": self.epoch.revealed,
                "replication_result": self.epoch.replication_result,
            },
            "store": self.store.to_dict(),
            "ledger": self.ledger.to_dict(),
            "candidates": {
                candidate_id: {
                    "manifest": record.manifest.as_dict(include_artifacts=True),
                    "verdict": record.verdict,
                    "detail": record.detail,
                    "status": record.status,
                    "parent_root": record.parent_root,
                    "candidate_root": record.candidate_root,
                    "reward_locked": record.reward_locked,
                    "bond_locked": record.bond_locked,
                    "canary_observations": record.canary_observations,
                    "canary_violation": record.canary_violation,
                }
                for candidate_id, record in self.candidates.items()
            },
            "audit": self.audit,
            "canonical_root": self.canonical_root,
            "genesis_root": self.genesis_root,
            "active_canary_id": self.active_canary_id,
            "funded_principals": self.funded_principals,
            "contributor_principals": self.contributor_principals,
            "trusted_wall_seconds": self.trusted_wall_seconds,
            "evaluation_costs": self.evaluation_costs,
            "promotion_costs": self.promotion_costs,
            "canary_costs": self.canary_costs,
            "replication_report": self.replication_report,
            "search_evidence": self.search_evidence,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any], state_path: str | Path | None = None) -> "LeverageService":
        if value.get("format") != "dendriswarm.leverage-state.v3.2":
            raise ValueError("unsupported leverage state format")
        epoch_value = value["epoch"]
        policy = GatePolicy.from_dict(epoch_value["policy"])
        epoch = ChallengeEpoch(
            np.asarray(epoch_value["x"], dtype=np.float64),
            np.asarray(epoch_value["y"], dtype=np.int64),
            np.asarray(epoch_value["replication_x"], dtype=np.float64),
            np.asarray(epoch_value["replication_y"], dtype=np.int64),
            policy,
            salt=str(epoch_value["salt"]),
        )
        epoch.submissions = {str(name): int(count) for name, count in epoch_value.get("submissions", {}).items()}
        epoch.tests_spent = int(epoch_value.get("tests_spent", 0))
        epoch.stale_submissions = int(epoch_value.get("stale_submissions", 0))
        epoch.closed = bool(epoch_value.get("closed", False))
        epoch.revealed = bool(epoch_value.get("revealed", False))
        epoch.replication_result = epoch_value.get("replication_result")
        store = ModelStore.from_dict(value["store"])
        genesis_root = str(value["genesis_root"])
        service = cls(
            parent=store.get(genesis_root),
            epoch=epoch,
            policy=policy,
            ledger=LeverageLedger.from_dict(value["ledger"]),
            store=store,
            state_path=None,
        )
        service.candidates = {
            candidate_id: CandidateRecord(
                manifest=CandidateManifest.from_dict(record["manifest"]),
                verdict=record["verdict"],
                detail=record.get("detail", {}),
                status=record["status"],
                parent_root=record["parent_root"],
                candidate_root=record.get("candidate_root"),
                reward_locked=int(record.get("reward_locked", 0)),
                bond_locked=int(record.get("bond_locked", 0)),
                canary_observations=list(record.get("canary_observations", [])),
                canary_violation=record.get("canary_violation", ""),
            )
            for candidate_id, record in value.get("candidates", {}).items()
        }
        service.audit = list(value.get("audit", []))
        service.genesis_root = genesis_root
        service.canonical_root = str(value["canonical_root"])
        service.active_canary_id = value.get("active_canary_id")
        service.funded_principals = dict(value.get("funded_principals", {}))
        service.contributor_principals = dict(value.get("contributor_principals", {}))
        service.trusted_wall_seconds = float(value.get("trusted_wall_seconds", 0.0))
        service.evaluation_costs = list(value.get("evaluation_costs", []))
        service.promotion_costs = list(value.get("promotion_costs", []))
        service.canary_costs = list(value.get("canary_costs", []))
        service.replication_report = value.get("replication_report")
        service.search_evidence = value.get("search_evidence")
        service._cache_by_root = {}
        service._rebuild_cache(reason="state-load")
        service.state_path = state_path
        if not service.validate_audit_chain():
            raise ValueError("persisted leverage audit chain is invalid")
        service._persist()
        return service

    @classmethod
    def load(cls, path: str | Path) -> "LeverageService":
        path = Path(path)
        return cls.from_dict(json.loads(path.read_text()), state_path=path)

    @contextmanager
    def _atomic_mutation(self):
        with self._state_lock:
            outer = self._mutation_depth == 0
            snapshot = self.to_dict() if outer else None
            self._mutation_depth += 1
            try:
                yield
            except Exception:
                self._mutation_depth -= 1
                if outer and snapshot is not None:
                    self._restore_snapshot(snapshot)
                    self._dirty = False
                raise
            else:
                self._mutation_depth -= 1
                if outer and self._dirty:
                    self._persist_now()
                    self._dirty = False

    def _restore_snapshot(self, snapshot: dict[str, Any]) -> None:
        state_path = self.state_path
        restored = type(self).from_dict(snapshot, state_path=None)
        for name in (
            "parent", "epoch", "policy", "ledger", "store", "candidates", "audit",
            "canonical_root", "genesis_root", "cache", "active_canary_id",
            "funded_principals", "contributor_principals", "trusted_wall_seconds",
            "evaluation_costs", "promotion_costs", "canary_costs", "replication_report",
            "search_evidence", "_cache_by_root",
        ):
            setattr(self, name, getattr(restored, name))
        self.state_path = state_path

    def _persist(self) -> None:
        if self._mutation_depth:
            self._dirty = True
            return
        self._persist_now()

    def _persist_now(self) -> None:
        if self.state_path is None:
            return
        path = Path(self.state_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(self.to_dict(), handle, sort_keys=True)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_name, path)
            try:
                directory_fd = os.open(path.parent, os.O_DIRECTORY)
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
            except (AttributeError, OSError):
                pass
        finally:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass

    # ---------------------------------------------------------------- metrics

    def metrics(self) -> dict[str, Any]:
        costs = self.evaluation_costs
        trusted_candidate_ops = sum(item["trusted_distance_ops"] for item in costs)
        counterfactual_ops = sum(item["counterfactual_full_candidate_distance_ops"] for item in costs)
        canary_ops = sum(item["distance_ops"] for item in self.canary_costs)
        replication_ops = int(self.replication_report.get("distance_ops", 0)) if self.replication_report else 0
        rebuild_ops = sum(int(item.get("challenge_cache_rebuild_distance_ops", 0)) for item in self.promotion_costs)
        total_trusted_ops = trusted_candidate_ops + canary_ops + replication_ops + rebuild_ops
        replicated_net = (
            int(self.replication_report["paired"]["net_wins"])
            if self.replication_report and self.replication_report.get("passed") else 0
        )
        ratios = [item["distance_ratio"] for item in costs]
        return {
            "verification_locality": {
                "mean_candidate_distance_ratio": sum(ratios) / len(ratios) if ratios else 0.0,
                "trusted_candidate_distance_ops": trusted_candidate_ops,
                "counterfactual_full_candidate_distance_ops": counterfactual_ops,
                "behavioral_binding_full_passes": sum(
                    int(item["behavioral_binding_full_passes"]) for item in costs
                ),
                "total_bytes_copied": sum(item["bytes_copied"] for item in costs),
                "total_selection_items": sum(item["selection_items"] for item in costs),
                "claim_boundary": (
                    "Admission recomputes touched distances and carries forward the exact next-root cache. "
                    "Binding is structural and performs zero full inference replays. Copying and selection "
                    "remain full-width costs and are reported; generic hardware speedup is not assumed."
                ),
            },
            "replicated_leverage": {
                "final_replication_passed": bool(self.replication_report and self.replication_report.get("passed")),
                "replicated_net_wins": replicated_net,
                "total_trusted_distance_ops_including_canary_replication_and_rebuild": total_trusted_ops,
                "replicated_net_wins_per_million_total_trusted_distance_ops": (
                    replicated_net / total_trusted_ops * 1e6 if total_trusted_ops else 0.0
                ),
                "untrusted_search_evidence": self.search_evidence,
            },
            "identity_economics": {
                "assumption": self.policy.identity_assumption,
                "permissionless_registration_grant_units": 0,
                "verified_principals_funded": len(self.funded_principals),
                "anti_farming_claim_is_conditional": True,
            },
            "trusted_wall_seconds": round(self.trusted_wall_seconds, 6),
            "submissions": self.epoch.tests_spent,
            "stale_submissions": self.epoch.stale_submissions,
            "ledger": self.ledger.snapshot(),
            "audit_events": len(self.audit),
            "audit_valid": self.validate_audit_chain(),
            "canonical_root": self.canonical_root,
            "genesis_root": self.genesis_root,
            "active_canary": self.active_canary_id,
            "model_store": self.store.snapshot(),
        }
