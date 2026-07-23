from __future__ import annotations

import json
import math
import secrets
import time
from pathlib import Path
from typing import Any

import numpy as np

from dendriswarm.coordinator.db import Database
from dendriswarm.core.limits import MAX_RESULT_OUTPUT_BYTES
from dendriswarm.core.crypto import Identity, content_hash, node_id_from_public_key, verify
from dendriswarm.core.models import (
    ExplorationOutput, InferenceOutput, NodeRegistration, SeedPolicy, TaskBody,
    TaskKind, TaskRequirements, TrainingOutput, VerificationOutput,
)
from dendriswarm.core.resources import estimate_reference_requirements, requirements_from_value
from dendriswarm.v5.service import Native10Coordinator
from dendriswarm.v6.service import Native10Coordinator as Native10V6Coordinator
from dendriswarm.v7.campaign import CIFAR100Campaign
from dendriswarm.tissues.reference import (
    ReferenceDendritron,
    TissueConfig,
    artifact_hash,
    artifact_consensus_hash,
    dataset_split,
    budgeted_dataset_artifact,
    dataset_split_budgeted,
    hidden_audit_split,
    make_digits_dataset,
    reference_benchmark,
)

CREDIT_SCALE = 1000
REWARD_UNITS = {
    TaskKind.EXPLORATION: 1000,
    TaskKind.TRAINING: 6000,
    TaskKind.VERIFICATION: 2000,
    TaskKind.INFERENCE: 800,
    TaskKind.DENDRITRON_MUTATION: 4000,
    TaskKind.DENDRITRON_VERIFICATION: 2000,
}
INFERENCE_COST_UNITS = 1000
EXPLORATION_QUORUM = 2
TRAINING_QUORUM = 2
VERIFICATION_QUORUM = 2
MIN_PROMOTION_ACCURACY = 0.90


class CommittedResultError(ValueError):
    """Failure whose task/accounting/audit records must commit atomically."""

    commit_transaction = True


class CoordinatorService:
    def __init__(self, state_dir: Path, lease_seconds: float = 60.0, inference_audit_rate: float = 0.2):
        self.state_dir = state_dir
        self.identity = Identity.load_or_create(state_dir / "keys")
        self.db = Database(state_dir / "dendriswarm.sqlite3")
        self.native10 = Native10Coordinator(state_dir / "native10", self.db)
        self.native10_v6 = Native10V6Coordinator(state_dir / "native10-v6", self.db)
        self.cifar100 = CIFAR100Campaign(state_dir / "cifar100", self.native10_v6)
        self.lease_seconds = float(lease_seconds)
        self.inference_audit_rate = float(inference_audit_rate)
        if not 0.0 <= self.inference_audit_rate <= 1.0:
            raise ValueError("inference_audit_rate must be between zero and one")
        audit_secret_path = state_dir / "keys" / "inference-audit-secret"
        audit_secret_path.parent.mkdir(parents=True, exist_ok=True)
        if audit_secret_path.exists():
            self._inference_audit_secret = audit_secret_path.read_text().strip()
        else:
            self._inference_audit_secret = secrets.token_hex(32)
            audit_secret_path.write_text(self._inference_audit_secret)
            try:
                audit_secret_path.chmod(0o600)
            except OSError:
                pass
        if self.db.validate_audit_chain()[1] == 0:
            self.db.append_audit("genesis", {"schema": 2})

    @staticmethod
    def _fresh(timestamp: int) -> bool:
        return abs(time.time() - int(timestamp)) <= 300

    def register_node(self, registration: NodeRegistration) -> None:
        # Verify over exactly the fields supplied by the client so older seed
        # packages remain compatible when new capability fields gain defaults.
        signed_value = registration.model_dump(mode="json", exclude_unset=True)
        signature = signed_value.pop("signature")
        if not self._fresh(signed_value["timestamp"]):
            raise ValueError("stale registration")
        if node_id_from_public_key(signed_value["public_key"]) != signed_value["node_id"]:
            raise ValueError("node id does not derive from public key")
        if not verify(signed_value["public_key"], signed_value, signature):
            raise ValueError("registration proof of key possession failed")
        if not self.db.consume_nonce(signed_value["node_id"], "register", signed_value["nonce"]):
            raise ValueError("replayed registration")
        policy = registration.policy or SeedPolicy(cpu_percent=100, memory_percent=100, disk_limit_mb=1_000_000, max_task_seconds=86400, allow_on_battery=True, max_system_cpu_percent=100)
        self.db.register_node(
            registration.node_id,
            registration.public_key,
            registration.capabilities.model_dump(mode="json"),
            policy.model_dump(mode="json"),
        )
        self.db.append_audit(
            "node_registered",
            {
                "node_id": registration.node_id,
                "machine": registration.capabilities.machine,
                "backend": registration.capabilities.supported_backends,
                "cpu_percent": policy.cpu_percent,
                "paused": policy.paused,
            },
        )

    def verify_node_request(self, node_id: str, timestamp: int, nonce: str, signature: str, action: str) -> bool:
        node = self.db.node(node_id)
        if not node or not self._fresh(timestamp):
            return False
        value = {"action": action, "node_id": node_id, "timestamp": timestamp, "nonce": nonce}
        if not verify(node["public_key"], value, signature):
            return False
        return self.db.consume_nonce(node_id, action, nonce)

    def verify_signed_object(self, value: dict[str, Any], action: str, consume: bool = True) -> bool:
        signature = value.get("signature", "")
        node_id = value.get("node_id", "")
        timestamp = int(value.get("timestamp", 0))
        nonce_value = str(value.get("nonce", ""))
        node = self.db.node(node_id)
        if not node or not self._fresh(timestamp):
            return False
        body = {"action": action, **{k: v for k, v in value.items() if k != "signature"}}
        if not verify(node["public_key"], body, signature):
            return False
        return self.db.consume_nonce(node_id, action, nonce_value) if consume else True

    def signed_task(self, row: Any) -> dict[str, Any]:
        requirements = requirements_from_value(json.loads(row["requirements"] or "{}"), TaskKind(row["kind"]))
        body = TaskBody(
            id=row["id"], kind=TaskKind(row["kind"]), payload=json.loads(row["payload"]),
            requirements=requirements,
            reward=float(row["reward_units"]) / CREDIT_SCALE, priority=int(row["priority"]),
            created_at=float(row["created_at"]), assigned_to=row["assigned_to"],
            lease_token=row["lease_token"], lease_expires_at=float(row["lease_expires_at"]),
            lease_deadline_at=float(row["lease_deadline_at"]),
        ).model_dump(mode="json")
        return {"task": body, "coordinator_public_key": self.identity.public_key_b64, "signature": self.identity.sign(body)}

    def _verify_result_signature(self, result: dict[str, Any]) -> None:
        node = self.db.node(result["node_id"])
        if not node:
            raise ValueError("unknown worker")
        signed = {k: result[k] for k in ("node_id", "task_id", "lease_token", "duration_ms", "output")}
        if not verify(node["public_key"], signed, result["signature"]):
            raise ValueError("invalid worker result signature")

    def bootstrap(self, experiments: int = 10) -> dict[str, Any]:
        dataset_row = self.db.approved_dataset()
        if dataset_row:
            return {"already_bootstrapped": True, "dataset_hash": dataset_row["content_hash"]}
        dataset = make_digits_dataset()
        self.db.add_dataset(dataset, "dendriswarm-bootstrap", status="approved")
        self.db.set_metadata("benchmark", reference_benchmark())
        configs = [
            TissueConfig(branches=branches, top_k=top_k, temperature=temp, iterations=15, seed=7 + branches + top_k).as_dict()
            for branches in (40, 80, 120, 160, 240, 320)
            for top_k, temp in ((1, 0.18), (3, 0.18))
        ]
        task_ids: list[str] = []
        for config in configs[:experiments]:
            sample_budget = 320 if int(config["branches"]) <= 80 else None
            task_dataset = dataset
            if sample_budget is not None:
                task_dataset = budgeted_dataset_artifact(dataset, sample_budget, sample_budget, int(config["seed"]))
                self.db.add_dataset(task_dataset, "dendriswarm-budget-sharder", status="approved")
            work_key = f"explore:{dataset['sha256']}:{content_hash(config)}"
            payload = {
                "dataset_hash": task_dataset["sha256"],
                "promotion_dataset_hash": dataset["sha256"],
                "config": config,
                "work_key": work_key,
                "required_tags": ["deterministic-v2", "portable-numpy-v1"],
            }
            requirements = estimate_reference_requirements(
                TaskKind.EXPLORATION,
                samples=len(task_dataset["splits"]["train"]) + len(task_dataset["splits"]["validation"]),
                features=int(task_dataset["feature_width"]),
                branches=int(config["branches"]),
                iterations=int(config["iterations"]),
                artifact_bytes=len(json.dumps(task_dataset).encode("utf-8")),
            )
            for slot in range(EXPLORATION_QUORUM):
                task_ids.append(self.db.add_task(
                    TaskKind.EXPLORATION, payload,
                    REWARD_UNITS[TaskKind.EXPLORATION], 20,
                    dedupe_key=f"{work_key}:replica:{slot}",
                    requirements=requirements,
                ))
        self.db.append_audit("bootstrap", {
            "dataset_hash": dataset["sha256"],
            "logical_experiments": min(experiments, len(configs)),
            "replicated_tasks": len(task_ids),
            "exploration_quorum": EXPLORATION_QUORUM,
            "coordinator_replay": False,
        })
        return {"dataset_hash": dataset["sha256"], "exploration_tasks": task_ids}

    @staticmethod
    def _score(value: Any, name: str) -> float:
        score = float(value)
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError(f"invalid {name}")
        return score

    @staticmethod
    def _validate_artifact(artifact: dict[str, Any]) -> ReferenceDendritron:
        model = ReferenceDendritron.from_artifact(artifact)
        if artifact_hash(artifact) != artifact["sha256"]:
            raise ValueError("artifact hash mismatch")
        if len(model.centers) > 4096 or model.centers.shape[1] > 4096:
            raise ValueError("artifact exceeds reference limits")
        return model

    def _receipt(self, node_id: str, task_id: str, reward_units: int, event_hash: str, details: dict[str, Any]) -> dict[str, Any]:
        receipt = {
            "format": "dendriswarm.work-receipt.v2", "node_id": node_id, "task_id": task_id,
            "reward_units": reward_units, "reward_credits": reward_units / CREDIT_SCALE,
            "audit_event_hash": event_hash, "details": details, "issued_at": time.time(),
        }
        return {"receipt": receipt, "coordinator_public_key": self.identity.public_key_b64,
                "signature": self.identity.sign(receipt)}

    @staticmethod
    def _output_size(output: dict[str, Any]) -> int:
        return len(json.dumps(output, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8"))

    def _reject_worker_output(
        self,
        *,
        task: Any,
        result: dict[str, Any],
        output: dict[str, Any],
        reason: str,
        event_type: str,
    ) -> None:
        """Commit a hostile/malformed worker-result rejection atomically.

        Coordinator-owned failures (for example a missing approved dataset) are
        deliberately handled outside this helper.  Only evidence supplied by
        the worker reaches this path, so exclusion, quarantine, and any claim-
        bond loss are attributable to the submitting identity.
        """
        status = self.db.reject_assigned_task(
            str(task["id"]), str(result["node_id"]), str(result["lease_token"]), output, reason,
        )
        self.db.append_audit(event_type, {
            "task_id": task["id"],
            "kind": task["kind"],
            "node_id": result["node_id"],
            "next_status": status,
            "reason": reason,
            "claim_bond_slashed_units": int(task["claim_bond_units"] or 0),
        })
        raise CommittedResultError(reason)

    @staticmethod
    def _consensus_reports(reports: list[Any], key_fn, quorum: int) -> list[Any]:
        groups: dict[Any, list[Any]] = {}
        for report in reports:
            value = json.loads(report["output"])
            groups.setdefault(key_fn(value), []).append(report)
        winners = [group for group in groups.values() if len(group) >= quorum]
        if not winners:
            return []
        winners.sort(key=lambda group: (-len(group), min(float(row["created_at"]) for row in group)))
        return winners[0]

    def _enqueue_tiebreaker(
        self,
        *,
        kind: TaskKind,
        payload: dict[str, Any],
        requirements: TaskRequirements,
        reward_units: int,
        priority: int,
        work_key: str,
        reports: list[Any],
        max_replicas: int = 3,
        parent_task: str | None = None,
    ) -> None:
        if len(reports) >= max_replicas:
            return
        excluded = [str(row["node_id"]) for row in reports]
        slot = len(reports)
        self.db.add_task(
            kind, payload, reward_units, priority,
            parent_task=parent_task,
            excluded_nodes=excluded,
            dedupe_key=f"{work_key}:replica:{slot}",
            requirements=requirements,
        )

    def _validate_inference_output(self, output: dict[str, Any], artifact: dict[str, Any]) -> InferenceOutput:
        validated = InferenceOutput.model_validate(output)
        classes = int(artifact["classes"])
        branches = len(artifact["centers"])
        active = min(int(artifact["top_k"]), branches)
        scores = np.asarray(validated.scores, dtype=np.float64)
        if len(scores) != classes or not np.isfinite(scores).all():
            raise ValueError("inference scores have invalid shape or values")
        if (scores < 0).any() or (scores > 1).any() or abs(float(scores.sum()) - 1.0) > 1e-6:
            raise ValueError("inference scores are not a probability vector")
        if validated.prediction != int(scores.argmax()) or not math.isclose(
            validated.confidence, float(scores.max()), rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError("inference prediction/confidence disagree with scores")
        if not 0 <= validated.prediction < classes:
            raise ValueError("inference prediction is outside class range")
        if validated.total_branches != branches or validated.active_branches != active:
            raise ValueError("inference branch metadata does not match the artifact")
        if abs(validated.activation_fraction - active / branches) > 1e-12:
            raise ValueError("inference activation fraction is inconsistent")
        return validated

    def process_result(self, result: dict[str, Any]) -> dict[str, Any]:
        # Serialize result validation, deterministic replay, side effects, and
        # final completion against task claiming. This prevents an active lease
        # from being requeued between verification and completion.
        with self.db.lock:
            return self._process_result_locked(result)

    def _process_result_locked(self, result: dict[str, Any]) -> dict[str, Any]:
        task = self.db.task(str(result.get("task_id", "")))
        is_native = bool(task and TaskKind(task["kind"]) in {
            TaskKind.DENDRITRON_MUTATION, TaskKind.DENDRITRON_VERIFICATION
        })
        if is_native:
            payload = json.loads(task["payload"])
            native = self.native10_v6 if payload.get("engine") == "dendriswarm.native10-trainable.v6" else self.native10
            with native.store.transaction():
                with self.db.transaction():
                    return self._process_result_transactional(result)
        with self.db.transaction():
            return self._process_result_transactional(result)

    def _process_result_transactional(self, result: dict[str, Any]) -> dict[str, Any]:
        self._verify_result_signature(result)
        if self._output_size(result["output"]) > MAX_RESULT_OUTPUT_BYTES:
            raise ValueError("task result output exceeds the coordinator limit")
        task = self.db.task(result["task_id"])
        if not task:
            raise ValueError("unknown task")
        if task["status"] == "completed":
            if task["assigned_to"] == result["node_id"] and task["lease_token"] == result["lease_token"] and json.loads(task["output"] or "{}") == result["output"]:
                return {"accepted": True, "duplicate": True, "stage": task["kind"]}
            raise ValueError("completed task replay does not match stored result")
        if task["status"] != "assigned" or task["assigned_to"] != result["node_id"]:
            raise ValueError("task is not assigned to this worker")
        now = time.time()
        if task["lease_token"] != result["lease_token"] or float(task["lease_expires_at"]) < now or float(task["lease_deadline_at"]) < now:
            raise ValueError("invalid or expired task lease")
        kind = TaskKind(task["kind"])
        payload = json.loads(task["payload"])
        output = result["output"]
        reward_units = 0
        details: dict[str, Any] = {"stage": kind.value, "verification_mode": "replicated-consensus"}
        dataset_row = self.db.dataset_by_hash(payload.get("dataset_hash", ""), approved_only=True) if payload.get("dataset_hash") else None
        dataset = json.loads(dataset_row["artifact"]) if dataset_row else None

        if kind in {TaskKind.DENDRITRON_MUTATION, TaskKind.DENDRITRON_VERIFICATION}:
            native = self.native10_v6 if payload.get("engine") == "dendriswarm.native10-trainable.v6" else self.native10
            reward_units, native_details = native.handle_result(task, result, payload, output)
            details.update(native_details)

        elif kind == TaskKind.EXPLORATION:
            if dataset is None:
                raise ValueError("approved dataset unavailable")
            try:
                validated = ExplorationOutput.model_validate(output)
                config = TissueConfig(**validated.config)
                if config.as_dict() != payload["config"]:
                    raise ValueError("exploration config mismatch")
            except (ValueError, TypeError, KeyError) as error:
                self._reject_worker_output(
                    task=task, result=result, output=output,
                    reason=f"invalid exploration proof: {error}",
                    event_type="exploration_proof_rejected",
                )
            work_key = str(payload["work_key"])
            self.db.record_work_report(work_key, kind.value, result["node_id"], task["id"], validated.model_dump(mode="json"))
            reports = self.db.work_reports(work_key, kind.value)
            consensus = self._consensus_reports(
                reports,
                lambda value: (content_hash(value["config"]), int(value["sample_count"]), int(value["correct_count"])),
                EXPLORATION_QUORUM,
            )
            if consensus:
                reported = float(json.loads(consensus[0]["output"])["validation_accuracy"])
                for report in consensus:
                    credited = self.db.credit(
                        f"reward:{report['task_id']}", report["node_id"], int(task["reward_units"]),
                        "replicated exploration consensus", report["task_id"],
                    )
                    if report["task_id"] == task["id"] and credited:
                        reward_units = int(task["reward_units"])
                if reported >= 0.90:
                    full_dataset_hash = str(payload.get("promotion_dataset_hash") or payload["dataset_hash"])
                    full_row = self.db.dataset_by_hash(full_dataset_hash, approved_only=True)
                    if not full_row:
                        raise ValueError("promotion dataset unavailable")
                    full_dataset = json.loads(full_row["artifact"])
                    training_work_key = f"train:{full_dataset_hash}:{content_hash(config.as_dict())}"
                    training_payload = {
                        "dataset_hash": full_dataset_hash,
                        "config": config.as_dict(),
                        "work_key": training_work_key,
                        "required_tags": ["deterministic-v2", "portable-numpy-v1"],
                    }
                    training_requirements = estimate_reference_requirements(
                        TaskKind.TRAINING,
                        samples=len(full_dataset["splits"]["train"]) + len(full_dataset["splits"]["validation"]),
                        features=int(full_dataset["feature_width"]),
                        branches=config.branches,
                        iterations=config.iterations,
                        artifact_bytes=len(json.dumps(full_dataset).encode("utf-8")),
                    )
                    for slot in range(TRAINING_QUORUM):
                        self.db.add_task(
                            TaskKind.TRAINING, training_payload,
                            REWARD_UNITS[TaskKind.TRAINING], 30, parent_task=task["id"],
                            dedupe_key=f"{training_work_key}:replica:{slot}",
                            requirements=training_requirements,
                        )
                details.update({"validation_accuracy": reported, "consensus": True, "reports": len(consensus)})
            else:
                requirements = requirements_from_value(json.loads(task["requirements"] or "{}"), kind)
                self._enqueue_tiebreaker(
                    kind=kind, payload=payload, requirements=requirements,
                    reward_units=int(task["reward_units"]), priority=int(task["priority"]),
                    work_key=work_key, reports=reports,
                )
                details.update({"consensus": False, "reports": len(reports)})

        elif kind == TaskKind.TRAINING:
            if dataset is None:
                raise ValueError("approved dataset unavailable")
            try:
                validated = TrainingOutput.model_validate(output)
                artifact = validated.artifact
                model = self._validate_artifact(artifact)
                if artifact["dataset_hash"] != dataset["sha256"] or artifact["config"] != payload["config"]:
                    raise ValueError("artifact provenance mismatch")
            except (ValueError, TypeError, KeyError) as error:
                self._reject_worker_output(
                    task=task, result=result, output=output,
                    reason=f"invalid training proof: {error}",
                    event_type="training_proof_rejected",
                )
            work_key = str(payload["work_key"])
            normalized = validated.model_dump(mode="json")
            self.db.record_work_report(work_key, kind.value, result["node_id"], task["id"], normalized)
            reports = self.db.work_reports(work_key, kind.value)
            consensus = self._consensus_reports(
                reports,
                lambda value: (
                    artifact_consensus_hash(value["artifact"]),
                    int(value["sample_count"]),
                    int(value["correct_count"]),
                ),
                TRAINING_QUORUM,
            )
            if consensus:
                chosen = json.loads(consensus[0]["output"])
                artifact = chosen["artifact"]
                trainers = [str(row["node_id"]) for row in consensus]
                for report in consensus:
                    credited = self.db.credit(
                        f"reward:{report['task_id']}", report["node_id"], int(task["reward_units"]),
                        "replicated training artifact consensus", report["task_id"],
                    )
                    if report["task_id"] == task["id"] and credited:
                        reward_units = int(task["reward_units"])
                candidate_id = self.db.add_candidate(
                    artifact, artifact["config"], dataset["sha256"], trainers[0], work_key,
                    float(chosen["train_accuracy"]), 0.0, 0.0, VERIFICATION_QUORUM,
                    trainer_nodes=trainers,
                )
                verification_payload = {
                    "candidate_id": candidate_id,
                    "artifact_hash": artifact["sha256"],
                    "dataset_hash": dataset["sha256"],
                    "work_key": f"verify:{candidate_id}",
                    "required_tags": ["deterministic-v2", "portable-numpy-v1"],
                }
                verification_requirements = estimate_reference_requirements(
                    TaskKind.VERIFICATION,
                    samples=len(dataset["splits"]["test"]),
                    features=int(dataset["feature_width"]),
                    branches=len(model.centers), iterations=1,
                    artifact_bytes=len(json.dumps(dataset).encode("utf-8")) + len(json.dumps(artifact).encode("utf-8")),
                )
                for slot in range(VERIFICATION_QUORUM):
                    self.db.add_task(
                        TaskKind.VERIFICATION, verification_payload,
                        REWARD_UNITS[TaskKind.VERIFICATION], 40, parent_task=task["id"],
                        excluded_nodes=trainers, dedupe_key=f"verify:{candidate_id}:replica:{slot}",
                        requirements=verification_requirements,
                    )
                details.update({"candidate_id": candidate_id, "artifact_hash": artifact["sha256"], "consensus": True})
            else:
                requirements = requirements_from_value(json.loads(task["requirements"] or "{}"), kind)
                self._enqueue_tiebreaker(
                    kind=kind, payload=payload, requirements=requirements,
                    reward_units=int(task["reward_units"]), priority=int(task["priority"]),
                    work_key=work_key, reports=reports,
                )
                details.update({"consensus": False, "reports": len(reports)})

        elif kind == TaskKind.VERIFICATION:
            candidate = self.db.candidate(payload["candidate_id"])
            if not candidate or dataset is None:
                raise ValueError("unknown candidate or dataset")
            trainers = set(json.loads(candidate["trainer_nodes"] or "[]")) | {str(candidate["trainer_node"])}
            expected_samples = len(dataset["splits"]["test"])
            try:
                if result["node_id"] in trainers:
                    raise ValueError("trainer cannot verify its own candidate")
                validated = VerificationOutput.model_validate(output)
                if validated.artifact_hash != candidate["artifact_hash"]:
                    raise ValueError("verification hash mismatch")
                if validated.sample_count != expected_samples:
                    raise ValueError("verification sample count mismatch")
            except (ValueError, TypeError, KeyError) as error:
                self._reject_worker_output(
                    task=task, result=result, output=output,
                    reason=f"invalid verification proof: {error}",
                    event_type="verification_proof_rejected",
                )
            inserted = self.db.record_verification(
                candidate["id"], result["node_id"], task["id"], validated.test_accuracy
            )
            reports = self.db.candidate_verifications(candidate["id"])
            values: dict[tuple[int, int], list[Any]] = {}
            # Integer counts are represented through accuracy and the committed sample count.
            for report in reports:
                correct = int(round(float(report["accuracy"]) * expected_samples))
                values.setdefault((correct, expected_samples), []).append(report)
            consensus = max(values.values(), key=len) if values else []
            finalized = False
            if len(consensus) >= VERIFICATION_QUORUM:
                for report in consensus:
                    self.db.credit(
                        f"reward:{report['task_id']}", report["verifier_node"], int(task["reward_units"]),
                        "independent verification quorum", report["task_id"],
                    )
                if inserted and any(report["task_id"] == task["id"] for report in consensus):
                    reward_units = int(task["reward_units"])
                benchmark = self.db.get_metadata("benchmark") or {"accuracy": {}}
                promotion_gate = max(0.95, float(benchmark.get("accuracy", {}).get("logistic_regression", MIN_PROMOTION_ACCURACY)))
                finalized = self.db.finalize_candidate(candidate["id"], promotion_gate, tolerance=1e-12)
            elif len(reports) >= VERIFICATION_QUORUM:
                requirements = requirements_from_value(json.loads(task["requirements"] or "{}"), kind)
                self._enqueue_tiebreaker(
                    kind=kind, payload=payload, requirements=requirements,
                    reward_units=int(task["reward_units"]), priority=int(task["priority"]),
                    work_key=str(payload["work_key"]), reports=[
                        {"node_id": row["verifier_node"], "task_id": row["task_id"], "created_at": row["created_at"]}
                        for row in reports
                    ], parent_task=task["parent_task"],
                )
            details.update({
                "candidate_id": candidate["id"], "test_accuracy": validated.test_accuracy,
                "verifications": len(reports), "finalized": finalized,
            })

        elif kind == TaskKind.INFERENCE:
            candidate = self.db.candidate_by_hash(payload["artifact_hash"])
            if not candidate or candidate["status"] not in ("canonical", "verified"):
                raise ValueError("inference artifact is not trusted")
            artifact = json.loads(candidate["artifact"])
            try:
                validated = self._validate_inference_output(output, artifact)
            except (ValueError, TypeError, KeyError) as error:
                status = self.db.reject_assigned_task(
                    task["id"], result["node_id"], result["lease_token"], output,
                    f"invalid inference proof: {error}",
                )
                self.db.append_audit("inference_proof_rejected", {
                    "task_id": task["id"], "node_id": result["node_id"],
                    "next_status": status, "reason": str(error),
                    "claim_bond_slashed_units": int(task["claim_bond_units"] or 0),
                })
                raise CommittedResultError(f"invalid inference proof: {error}") from error
            prediction = validated.prediction
            token = content_hash({"task_id": task["id"], "secret": self._inference_audit_secret})
            audited = int(token[:16], 16) / 16**16 < self.inference_audit_rate
            if audited:
                model = self._validate_artifact(artifact)
                expected = int(model.predict(np.asarray(payload["features"], dtype=np.float64)))
                if prediction != expected:
                    status = self.db.reject_assigned_task(
                        task["id"], result["node_id"], result["lease_token"], output,
                        "audited inference mismatch",
                    )
                    self.db.append_audit("inference_audit_failed", {
                        "task_id": task["id"], "node_id": result["node_id"], "next_status": status,
                        "claim_bond_slashed_units": int(task["claim_bond_units"] or 0),
                    })
                    raise CommittedResultError("audited inference result is incorrect")
            reward_units = int(task["reward_units"])
            self.db.credit(f"reward:{task['id']}", result["node_id"], reward_units, "inference execution", task["id"])
            details.update({"prediction": prediction, "audited": audited, "proof_carrying_schema": True})

        if not self.db.complete_task(task["id"], result["node_id"], result["lease_token"], output):
            raise ValueError("task completion lost its active lease")
        event_hash = self.db.append_audit("task_completed", {
            "task_id": task["id"], "kind": kind.value, "node_id": result["node_id"],
            "output_hash": content_hash(output), "reward_units": reward_units,
        })
        return {"accepted": True, **details, **self._receipt(result["node_id"], task["id"], reward_units, event_hash, details)}

    def renew_lease(self, value: dict[str, Any]) -> dict[str, Any]:
        if not self.verify_signed_object(value, "renew-lease"):
            raise ValueError("invalid, stale, or replayed lease-renewal signature")
        task = self.db.task(str(value["task_id"]))
        if not task:
            raise ValueError("unknown task")
        requirements = requirements_from_value(json.loads(task["requirements"] or "{}"), TaskKind(task["kind"]))
        extension = max(self.lease_seconds, requirements.estimated_runtime_seconds * 1.5)
        expires_at = self.db.renew_task_lease(
            str(value["task_id"]), str(value["node_id"]), str(value["lease_token"]), extension
        )
        if expires_at is None:
            raise ValueError("task lease is no longer renewable")
        return {"renewed": True, "lease_expires_at": expires_at}

    def abandon_lease(self, value: dict[str, Any]) -> dict[str, Any]:
        if not self.verify_signed_object(value, "abandon-task"):
            raise ValueError("invalid, stale, or replayed task-abandon signature")
        abandoned = self.db.abandon_task(
            str(value["task_id"]), str(value["node_id"]), str(value["lease_token"]), str(value.get("reason", "local-policy-change"))
        )
        if not abandoned:
            raise ValueError("task is no longer abandonable")
        self.db.append_audit("task_abandoned", {
            "task_id": value["task_id"], "node_id": value["node_id"], "reason": value.get("reason", "")
        })
        return {"abandoned": True}

    def enqueue_inference(self, value: dict[str, Any]) -> dict[str, Any]:
        if not self.verify_signed_object(value, "inference", consume=False):
            raise ValueError("invalid or stale inference signature")
        request_key = f"{value['node_id']}:{value['request_id']}"
        existing = self.db.inference_request(request_key)
        if existing:
            return {"task_id": existing["task_id"], "created": False,
                    "cost_units": int(existing["cost_units"]),
                    "cost_credits": int(existing["cost_units"]) / CREDIT_SCALE}
        if not self.db.consume_nonce(value["node_id"], "inference", value["nonce"]):
            raise ValueError("replayed inference signature")
        canonical = self.db.canonical_candidate()
        if not canonical:
            raise ValueError("no canonical model")
        artifact = json.loads(canonical["artifact"])
        features = [float(v) for v in value["features"]]
        if len(features) != int(artifact["feature_width"]) or not all(math.isfinite(v) for v in features):
            raise ValueError(f"inference requires {artifact['feature_width']} finite features")
        task_id, created = self.db.create_inference_task(
            request_key, value["node_id"], canonical["artifact_hash"], features,
            INFERENCE_COST_UNITS, REWARD_UNITS[TaskKind.INFERENCE],
        )
        if created:
            self.db.append_audit("inference_requested", {"task_id": task_id, "node_id": value["node_id"],
                                                          "cost_units": INFERENCE_COST_UNITS})
        return {"task_id": task_id, "created": created, "cost_units": INFERENCE_COST_UNITS,
                "cost_credits": INFERENCE_COST_UNITS / CREDIT_SCALE}

    def audit_checkpoint(self) -> dict[str, Any]:
        valid, events, head = self.db.validate_audit_chain()
        checkpoint = {"format": "dendriswarm.audit-checkpoint.v1", "valid": valid,
                      "events": events, "head": head, "issued_at": time.time()}
        return {"checkpoint": checkpoint, "coordinator_public_key": self.identity.public_key_b64,
                "signature": self.identity.sign(checkpoint)}
