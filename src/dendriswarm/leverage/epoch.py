"""Committed challenge epochs with an independent final replication holdout."""
from __future__ import annotations

import secrets
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from dendriswarm.core.crypto import content_hash
from dendriswarm.leverage.stats import GatePolicy


def _freeze_split(x: np.ndarray, y: np.ndarray, name: str) -> tuple[np.ndarray, np.ndarray]:
    features = np.asarray(x, dtype=np.float64).copy()
    labels = np.asarray(y, dtype=np.int64).copy()
    if features.ndim != 2 or len(features) == 0:
        raise ValueError(f"{name} features must be a non-empty matrix")
    if labels.shape != (len(features),):
        raise ValueError(f"{name} labels must align with features")
    if not np.isfinite(features).all():
        raise ValueError(f"{name} contains non-finite features")
    features.setflags(write=False)
    labels.setflags(write=False)
    return features, labels


def _split_artifact(x: np.ndarray, y: np.ndarray, kind: str) -> dict[str, Any]:
    value = {
        "format": f"dendriswarm.{kind}.v4.1",
        "features": np.round(x, 12).tolist(),
        "labels": [int(item) for item in y],
    }
    value["sha256"] = content_hash(value)
    return value


@dataclass
class ChallengeEpoch:
    x: np.ndarray
    y: np.ndarray
    replication_x: np.ndarray
    replication_y: np.ndarray
    policy: GatePolicy
    salt: str = field(default_factory=lambda: secrets.token_hex(32))
    submissions: dict[str, int] = field(default_factory=dict)
    tests_spent: int = 0
    stale_submissions: int = 0
    closed: bool = False
    revealed: bool = False
    replication_result: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        self.x, self.y = _freeze_split(self.x, self.y, "challenge")
        self.replication_x, self.replication_y = _freeze_split(
            self.replication_x, self.replication_y, "replication"
        )
        if self.x.shape[1] != self.replication_x.shape[1]:
            raise ValueError("challenge and replication feature widths must match")

    @property
    def challenge_artifact(self) -> dict[str, Any]:
        return _split_artifact(self.x, self.y, "challenge")

    @property
    def replication_artifact(self) -> dict[str, Any]:
        return _split_artifact(self.replication_x, self.replication_y, "replication-holdout")

    @property
    def challenge_hash(self) -> str:
        return self.challenge_artifact["sha256"]

    @property
    def replication_hash(self) -> str:
        return self.replication_artifact["sha256"]

    @property
    def commitment(self) -> str:
        return content_hash({
            "format": "dendriswarm.epoch-commitment.v4.1",
            "challenge": self.challenge_hash,
            "replication": self.replication_hash,
            "salt_hash": content_hash({"salt": self.salt}),
            "policy": self.policy.registration_hash,
        })

    def next_fee(self, contributor: str) -> int:
        return self.policy.escalated_fee(self.submissions.get(contributor, 0))

    def admit_submission(self, contributor: str) -> int:
        if self.closed:
            raise ValueError("epoch is closed")
        if self.tests_spent >= self.policy.max_submissions_per_epoch:
            raise ValueError("epoch test budget exhausted")
        prior = self.submissions.get(contributor, 0)
        if prior >= self.policy.max_submissions_per_contributor:
            raise ValueError("contributor test budget exhausted")
        fee = self.policy.escalated_fee(prior)
        self.submissions[contributor] = prior + 1
        self.tests_spent += 1
        return fee

    def note_stale_submission(self) -> None:
        if self.closed:
            raise ValueError("epoch is closed")
        self.stale_submissions += 1

    def close(self, replication_result: dict[str, Any]) -> None:
        if self.closed:
            raise ValueError("epoch is already closed")
        self.replication_result = dict(replication_result)
        self.closed = True

    def reveal(self) -> dict[str, Any]:
        if not self.closed or self.replication_result is None:
            raise ValueError("cannot reveal before final replication")
        self.revealed = True
        return {
            "salt": self.salt,
            "challenge": self.challenge_artifact,
            "challenge_hash": self.challenge_hash,
            "replication": self.replication_artifact,
            "replication_hash": self.replication_hash,
            "replication_result": self.replication_result,
            "commitment": self.commitment,
            "policy": self.policy.as_dict(),
            "policy_hash": self.policy.registration_hash,
        }

    @staticmethod
    def verify_disclosure(disclosure: dict[str, Any]) -> bool:
        def verify_artifact(name: str) -> str | None:
            artifact = dict(disclosure[name])
            artifact_hash = artifact.pop("sha256", None)
            if artifact_hash is None or content_hash(artifact) != artifact_hash:
                return None
            if artifact_hash != disclosure[f"{name}_hash"]:
                return None
            return artifact_hash

        challenge_hash = verify_artifact("challenge")
        replication_hash = verify_artifact("replication")
        if challenge_hash is None or replication_hash is None:
            return False
        try:
            disclosed_policy = GatePolicy.from_dict(dict(disclosure["policy"]))
        except (KeyError, TypeError, ValueError):
            return False
        if disclosed_policy.registration_hash != disclosure.get("policy_hash"):
            return False
        salt_hash = content_hash({"salt": disclosure["salt"]})
        expected = content_hash({
            "format": "dendriswarm.epoch-commitment.v4.1",
            "challenge": challenge_hash,
            "replication": replication_hash,
            "salt_hash": salt_hash,
            "policy": disclosed_policy.registration_hash,
        })
        return expected == disclosure["commitment"]
