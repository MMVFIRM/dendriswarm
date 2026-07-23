"""Pre-registered statistical admission, canary, and replication gates.

The frozen policy is the single source of truth for an epoch. Every field that
changes admission, accounting, canary interpretation, or final replication is
included in the registration hash.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from math import comb
from typing import Any, Iterable

import numpy as np

from dendriswarm.core.crypto import content_hash


def mcnemar_exact_one_sided(wins: int, losses: int) -> float:
    """P(X >= wins | n = wins + losses, p = 0.5), exact."""
    if wins < 0 or losses < 0:
        raise ValueError("counts must be non-negative")
    n = wins + losses
    if n == 0:
        return 1.0
    tail = sum(comb(n, k) for k in range(wins, n + 1))
    return tail / 2 ** n


@dataclass(frozen=True)
class GatePolicy:
    """Pre-registered evaluation-epoch policy, frozen before submissions open."""

    alpha_epoch: float = 0.05
    max_submissions_per_epoch: int = 40
    max_submissions_per_contributor: int = 8
    min_net_wins: int = 25
    max_challenge_subgroup_net_loss: int = 5
    min_subgroup_samples: int = 20
    max_territory_categories: int = 8
    max_route_share: float = 0.25
    max_touched_branches: int = 64
    submission_fee_units: int = 200
    stale_submission_fee_units: int = 0
    fee_escalation_numerator: int = 3
    fee_escalation_denominator: int = 2
    bond_units: int = 1000
    reward_units: int = 6000
    grant_units: int = 8000
    identity_assumption: str = "externally-verified-one-grant-per-principal"
    canary_window_events: int = 2
    canary_min_batch_samples: int = 50
    canary_min_net_wins: int = -5
    canary_min_informative_predictions: int = 1
    canary_min_discordant: int = 0
    canary_max_subgroup_net_loss: int = 5
    canary_source_kinds: tuple[str, ...] = field(default=(
        "audited-inference-batch",
        "heldout-canary-batch",
    ))
    replication_min_net_wins: int = 25
    replication_max_subgroup_net_loss: int = 8
    replication_alpha: float = 0.05
    verdict_bands: tuple[str, ...] = field(default=(
        "rejected:no-significant-improvement",
        "rejected:below-effect-size-floor",
        "rejected:territorial-violation",
        "rejected:route-share-violation",
        "rejected:protected-challenge-regression",
        "rejected:subgroup-harm",
        "rejected:stale-parent",
        "rejected:submission-budget-exhausted",
        "rejected:contributor-budget-exhausted",
        "rejected:canary-window-active",
        "accepted:promotion-candidate",
    ))

    def __post_init__(self) -> None:
        if not 0 < self.alpha_epoch < 1:
            raise ValueError("alpha_epoch must be between zero and one")
        if not 0 < self.replication_alpha < 1:
            raise ValueError("replication_alpha must be between zero and one")
        for name in (
            "max_submissions_per_epoch", "max_submissions_per_contributor",
            "max_territory_categories", "max_touched_branches",
            "fee_escalation_numerator", "fee_escalation_denominator",
            "canary_window_events", "canary_min_batch_samples",
            "canary_min_informative_predictions",
            "min_subgroup_samples",
        ):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        for name in (
            "min_net_wins", "max_challenge_subgroup_net_loss",
            "submission_fee_units", "stale_submission_fee_units", "bond_units",
            "reward_units", "grant_units", "canary_max_subgroup_net_loss",
            "replication_min_net_wins", "replication_max_subgroup_net_loss", "canary_min_discordant",
        ):
            if int(getattr(self, name)) < 0:
                raise ValueError(f"{name} must be non-negative")
        if not 0 < self.max_route_share <= 1:
            raise ValueError("max_route_share must be in (0, 1]")
        if not self.identity_assumption:
            raise ValueError("identity_assumption must be non-empty")
        if not self.canary_source_kinds or len(set(self.canary_source_kinds)) != len(self.canary_source_kinds):
            raise ValueError("canary source kinds must be non-empty and unique")
        if len(set(self.verdict_bands)) != len(self.verdict_bands):
            raise ValueError("verdict bands must be unique")

    @property
    def per_test_alpha(self) -> float:
        return self.alpha_epoch / self.max_submissions_per_epoch

    def escalated_fee(self, prior_attempts: int) -> int:
        if prior_attempts < 0:
            raise ValueError("prior_attempts must be non-negative")
        fee = self.submission_fee_units
        for _ in range(prior_attempts):
            fee = fee * self.fee_escalation_numerator // self.fee_escalation_denominator
        return fee

    def as_dict(self) -> dict[str, Any]:
        return {
            "alpha_epoch": self.alpha_epoch,
            "per_test_alpha": self.per_test_alpha,
            "max_submissions_per_epoch": self.max_submissions_per_epoch,
            "max_submissions_per_contributor": self.max_submissions_per_contributor,
            "min_net_wins": self.min_net_wins,
            "max_challenge_subgroup_net_loss": self.max_challenge_subgroup_net_loss,
            "min_subgroup_samples": self.min_subgroup_samples,
            "max_territory_categories": self.max_territory_categories,
            "max_route_share": self.max_route_share,
            "max_touched_branches": self.max_touched_branches,
            "submission_fee_units": self.submission_fee_units,
            "stale_submission_fee_units": self.stale_submission_fee_units,
            "fee_escalation_numerator": self.fee_escalation_numerator,
            "fee_escalation_denominator": self.fee_escalation_denominator,
            "bond_units": self.bond_units,
            "reward_units": self.reward_units,
            "grant_units": self.grant_units,
            "identity_assumption": self.identity_assumption,
            "canary_window_events": self.canary_window_events,
            "canary_min_batch_samples": self.canary_min_batch_samples,
            "canary_min_net_wins": self.canary_min_net_wins,
            "canary_min_informative_predictions": self.canary_min_informative_predictions,
            "canary_min_discordant": self.canary_min_discordant,
            "canary_max_subgroup_net_loss": self.canary_max_subgroup_net_loss,
            "canary_source_kinds": list(self.canary_source_kinds),
            "replication_min_net_wins": self.replication_min_net_wins,
            "replication_max_subgroup_net_loss": self.replication_max_subgroup_net_loss,
            "replication_alpha": self.replication_alpha,
            "verdict_bands": list(self.verdict_bands),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "GatePolicy":
        fields = dict(value)
        fields.pop("per_test_alpha", None)
        fields["verdict_bands"] = tuple(fields.get("verdict_bands", ()))
        fields["canary_source_kinds"] = tuple(fields.get("canary_source_kinds", ()))
        return cls(**fields)

    @property
    def registration_hash(self) -> str:
        return content_hash({"format": "dendriswarm.gate-policy.v4.1", **self.as_dict()})


@dataclass(frozen=True)
class PairedResult:
    wins: int
    losses: int
    concordant: int
    p_value: float

    @property
    def net_wins(self) -> int:
        return self.wins - self.losses

    def as_dict(self) -> dict[str, Any]:
        return {
            "wins": self.wins,
            "losses": self.losses,
            "concordant": self.concordant,
            "net_wins": self.net_wins,
            "p_value": self.p_value,
        }


def paired_comparison(parent_correct: Iterable[bool], candidate_correct: Iterable[bool]) -> PairedResult:
    """Exact paired comparison from two aligned boolean correctness vectors."""
    wins = losses = concordant = 0
    for parent_ok, candidate_ok in zip(parent_correct, candidate_correct, strict=True):
        if candidate_ok and not parent_ok:
            wins += 1
        elif parent_ok and not candidate_ok:
            losses += 1
        else:
            concordant += 1
    return PairedResult(wins, losses, concordant, mcnemar_exact_one_sided(wins, losses))


def subgroup_comparisons(
    subgroup_ids: np.ndarray,
    parent_correct: np.ndarray,
    candidate_correct: np.ndarray,
    *,
    min_samples: int = 1,
) -> dict[str, dict[str, Any]]:
    """Return paired outcomes for every sufficiently represented subgroup."""
    subgroup_ids = np.asarray(subgroup_ids)
    parent_correct = np.asarray(parent_correct, dtype=bool)
    candidate_correct = np.asarray(candidate_correct, dtype=bool)
    if subgroup_ids.shape != parent_correct.shape or parent_correct.shape != candidate_correct.shape:
        raise ValueError("subgroup ids and correctness vectors must align")
    report: dict[str, dict[str, Any]] = {}
    for subgroup in np.unique(subgroup_ids):
        mask = subgroup_ids == subgroup
        samples = int(mask.sum())
        if samples < min_samples:
            continue
        result = paired_comparison(parent_correct[mask], candidate_correct[mask])
        report[str(subgroup)] = {"samples": samples, **result.as_dict()}
    return report


def worst_subgroup_net_wins(report: dict[str, dict[str, Any]]) -> tuple[str | None, int]:
    if not report:
        return None, 0
    subgroup, detail = min(report.items(), key=lambda item: int(item[1]["net_wins"]))
    return subgroup, int(detail["net_wins"])
