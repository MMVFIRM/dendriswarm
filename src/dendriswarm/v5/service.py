from __future__ import annotations

import json
import os
import tempfile
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np

from dendriswarm.core.crypto import content_hash
from dendriswarm.core.models import (
    DendritronMutationOutput,
    DendritronVerificationOutput,
    TaskKind,
)
from dendriswarm.core.resources import derive_payload_requirements
from dendriswarm.v5.native10 import (
    Native10Config,
    Native10Dendritron,
    delta_hash,
    delta_consensus_hash,
    synthetic_representation_shard,
    validate_delta,
)
from dendriswarm.v5.validation import (
    decode_global_validation_artifact,
    synthetic_global_validation_fixture,
)

MUTATION_QUORUM = 2
VERIFICATION_QUORUM = 2
MUTATION_REWARD_UNITS = 4_000
VERIFICATION_REWARD_UNITS = 2_000
PROMOTABLE_OPERATIONS = {"expert_refit", "repair", "branch_lifecycle", "scout_refit", "memory_update"}


class Native10Store:
    """Atomic local store for the canonical Dendritron and contribution lineage."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.checkpoint_path = self.path / "canonical-native10.json"
        self.validation_path = self.path / "global-validation.json"
        self.state_path = self.path / "state.json"
        self.lock = threading.RLock()
        if not self.state_path.exists():
            self._write_json(self.state_path, {
                "format": "dendriswarm.native10-coordinator-state.v5.1",
                "canonical_root": None,
                "global_validation_hash": None,
                "global_validation_evaluations": 0,
                "active_round": None,
                "candidates": {},
                "contributions": [],
                "created_at": time.time(),
            })

    @staticmethod
    def _write_json(path: Path, value: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(value, handle, sort_keys=True, separators=(",", ":"), allow_nan=False)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass

    @contextmanager
    def transaction(self):
        """Rollback canonical and protocol files if the enclosing DB transaction fails."""
        with self.lock:
            state_before = self.state_path.read_bytes() if self.state_path.exists() else None
            checkpoint_before = self.checkpoint_path.read_bytes() if self.checkpoint_path.exists() else None
            validation_before = self.validation_path.read_bytes() if self.validation_path.exists() else None
            try:
                yield
            except Exception:
                if state_before is None:
                    self.state_path.unlink(missing_ok=True)
                else:
                    fd, temporary = tempfile.mkstemp(prefix=f".{self.state_path.name}.", dir=self.state_path.parent)
                    with os.fdopen(fd, "wb") as handle:
                        handle.write(state_before)
                        handle.flush()
                        os.fsync(handle.fileno())
                    os.replace(temporary, self.state_path)
                if checkpoint_before is None:
                    self.checkpoint_path.unlink(missing_ok=True)
                else:
                    fd, temporary = tempfile.mkstemp(prefix=f".{self.checkpoint_path.name}.", dir=self.checkpoint_path.parent)
                    with os.fdopen(fd, "wb") as handle:
                        handle.write(checkpoint_before)
                        handle.flush()
                        os.fsync(handle.fileno())
                    os.replace(temporary, self.checkpoint_path)
                if validation_before is None:
                    self.validation_path.unlink(missing_ok=True)
                else:
                    fd, temporary = tempfile.mkstemp(prefix=f".{self.validation_path.name}.", dir=self.validation_path.parent)
                    with os.fdopen(fd, "wb") as handle:
                        handle.write(validation_before)
                        handle.flush()
                        os.fsync(handle.fileno())
                    os.replace(temporary, self.validation_path)
                raise

    def state(self) -> dict[str, Any]:
        with self.lock:
            return json.loads(self.state_path.read_text())

    def save_state(self, state: dict[str, Any]) -> None:
        with self.lock:
            self._write_json(self.state_path, state)

    def initialized(self) -> bool:
        return self.checkpoint_path.exists()

    def initialize(self, config: Native10Config, *, seed: int | None = None, replace: bool = False) -> dict[str, Any]:
        with self.lock:
            if self.state().get("active_round") is not None:
                raise ValueError("cannot initialize or replace the canonical checkpoint during an active round")
            if self.checkpoint_path.exists() and not replace:
                raise ValueError("a canonical Native10 checkpoint already exists")
            model = Native10Dendritron.initialize(config, seed=seed)
            self._write_json(self.checkpoint_path, model.artifact())
            self.validation_path.unlink(missing_ok=True)
            state = self.state()
            state.update({
                "canonical_root": model.root,
                "global_validation_hash": None,
                "global_validation_evaluations": 0,
                "active_round": None,
                "candidates": {},
                "contributions": [],
                "initialized_at": time.time(),
            })
            self.save_state(state)
            return self.status()

    def import_checkpoint(self, artifact: dict[str, Any], *, replace: bool = False) -> dict[str, Any]:
        with self.lock:
            if self.state().get("active_round") is not None:
                raise ValueError("cannot import or replace the canonical checkpoint during an active round")
            if self.checkpoint_path.exists() and not replace:
                raise ValueError("a canonical Native10 checkpoint already exists")
            model = Native10Dendritron.from_artifact(artifact)
            self._write_json(self.checkpoint_path, model.artifact())
            self.validation_path.unlink(missing_ok=True)
            state = self.state()
            state.update({
                "canonical_root": model.root,
                "global_validation_hash": None,
                "global_validation_evaluations": 0,
                "active_round": None,
                "candidates": {},
                "contributions": [],
                "imported_at": time.time(),
            })
            self.save_state(state)
            return self.status()

    def set_global_validation(self, artifact: dict[str, Any], *, replace: bool = False) -> dict[str, Any]:
        with self.lock:
            if self.state().get("active_round") is not None:
                raise ValueError("cannot replace global validation during an active round")
            model = self.model()
            if self.validation_path.exists() and not replace:
                raise ValueError("a global Native10 validation artifact already exists")
            _, _, policy = decode_global_validation_artifact(artifact, expected_config=model.config)
            self._write_json(self.validation_path, artifact)
            state = self.state()
            state["global_validation_hash"] = artifact["sha256"]
            state["global_validation_evaluations"] = 0
            state["global_validation_policy"] = policy.as_dict()
            state["global_validation_source"] = artifact.get("source")
            self.save_state(state)
            return self.validation_status()

    def global_validation(self) -> dict[str, Any]:
        if not self.validation_path.exists():
            raise ValueError("coordinator-held global Native10 validation is not configured")
        artifact = json.loads(self.validation_path.read_text())
        decode_global_validation_artifact(artifact, expected_config=self.model().config)
        return artifact

    def validation_status(self) -> dict[str, Any]:
        if not self.validation_path.exists():
            return {"configured": False, "sha256": None}
        artifact = self.global_validation()
        state = self.state()
        return {
            "configured": True,
            "sha256": artifact["sha256"],
            "sample_count": artifact["sample_count"],
            "candidate_evaluations": int(state.get("global_validation_evaluations", 0)),
            "max_candidate_evaluations": int(artifact["policy"]["max_candidate_evaluations"]),
            "counts_by_class": artifact["counts_by_class"],
            "source": artifact["source"],
            "split": artifact["split"],
            "trainer_visible": False,
            "protocol_fixture_only": artifact.get("protocol_fixture_only", False),
            "policy": artifact["policy"],
        }

    def model(self) -> Native10Dendritron:
        if not self.checkpoint_path.exists():
            raise ValueError("Native10 canonical checkpoint is not initialized")
        return Native10Dendritron.from_artifact(json.loads(self.checkpoint_path.read_text()))

    def status(self) -> dict[str, Any]:
        state = self.state()
        if not self.initialized():
            return {
                "initialized": False,
                "canonical_root": None,
                "active_round": state.get("active_round"),
                "contribution_count": len(state.get("contributions", [])),
                "global_validation": self.validation_status(),
            }
        model = self.model()
        return {
            "initialized": True,
            "canonical_root": model.root,
            "parameter_count": model.parameter_count,
            "config": model.config.as_dict(),
            "active_round": state.get("active_round"),
            "candidate_count": len(state.get("candidates", {})),
            "contribution_count": len(state.get("contributions", [])),
            "latest_contribution": state.get("contributions", [])[-1] if state.get("contributions") else None,
            "global_validation": self.validation_status(),
        }

    def begin_round(
        self,
        round_value: dict[str, Any],
        *,
        validation_hash_value: str,
        max_candidate_evaluations: int,
    ) -> None:
        with self.lock:
            state = self.state()
            if state.get("active_round") is not None:
                raise ValueError("one Native10 contribution round is already active")
            if round_value["base_root"] != state.get("canonical_root"):
                raise ValueError("round parent is not canonical")
            if state.get("global_validation_hash") != validation_hash_value:
                raise ValueError("round validation artifact is not the installed coordinator artifact")
            evaluations = int(state.get("global_validation_evaluations", 0))
            if evaluations >= int(max_candidate_evaluations):
                raise ValueError(
                    "global validation evaluation budget is exhausted; rotate the coordinator-held artifact"
                )
            round_value = {**round_value, "validation_evaluation_index": evaluations + 1}
            state["global_validation_evaluations"] = evaluations + 1
            state["active_round"] = round_value
            self.save_state(state)

    def candidate(self, candidate_id: str) -> dict[str, Any] | None:
        return self.state().get("candidates", {}).get(candidate_id)

    def record_candidate(self, candidate_id: str, value: dict[str, Any]) -> None:
        with self.lock:
            state = self.state()
            existing = state.setdefault("candidates", {}).get(candidate_id)
            if existing and existing != value:
                raise ValueError("conflicting Native10 candidate record")
            state["candidates"][candidate_id] = value
            self.save_state(state)

    def promote(self, candidate_id: str, verification: dict[str, Any]) -> dict[str, Any]:
        with self.lock:
            state = self.state()
            candidate = state.get("candidates", {}).get(candidate_id)
            if not candidate:
                raise ValueError("unknown Native10 candidate")
            model = self.model()
            if candidate["base_root"] != model.root:
                raise ValueError("candidate became stale before promotion")
            before = model.root
            contribution = {
                "candidate_id": candidate_id,
                "trainer_nodes": candidate["trainer_nodes"],
                "verifier_nodes": verification["verifier_nodes"],
                "net_wins": int(verification["net_wins"]),
                "sample_count": int(verification["sample_count"]),
                "validation_hash": verification["validation_hash"],
                "promoted_at": time.time(),
            }
            updated = model.apply_delta(candidate["delta"], contribution=contribution)
            after = updated.root
            self._write_json(self.checkpoint_path, updated.artifact())
            record = {
                **contribution,
                "root_before": before,
                "root_after": after,
                "delta_hash": candidate["delta"]["sha256"],
                "operation": candidate["operation"],
                "category": candidate["category"],
                "active_experts": candidate.get("active_experts", []),
                "rotation_phase_before": candidate.get("rotation_phase_before"),
                "rotation_phase_after": candidate.get("rotation_phase_after"),
                "losses_by_class": verification["losses_by_class"],
                "loss_rates_by_class": verification["loss_rates_by_class"],
                "samples_by_class": verification["samples_by_class"],
            }
            state["canonical_root"] = after
            state["contributions"].append(record)
            state["active_round"] = None
            state["candidates"][candidate_id]["status"] = "promoted"
            state["candidates"][candidate_id]["root_after"] = after
            self.save_state(state)
            return record

    def reject_round(self, candidate_id: str, reason: str) -> None:
        with self.lock:
            state = self.state()
            if candidate_id in state.get("candidates", {}):
                state["candidates"][candidate_id]["status"] = "rejected"
                state["candidates"][candidate_id]["reason"] = reason
            state["active_round"] = None
            self.save_state(state)


class Native10Coordinator:
    """Coordinator-side state machine for Native10-derived tissue contributions."""

    def __init__(self, state_dir: Path, db: Any):
        self.store = Native10Store(Path(state_dir))
        self.db = db
        self.lock = threading.RLock()

    def initialize(self, profile: str = "compact", *, input_width: int | None = None, seed: int = 7, replace: bool = False) -> dict[str, Any]:
        if profile == "compact":
            config = Native10Config.compact_demo(seed=seed)
            if input_width is not None and input_width != config.input_width:
                config = Native10Config(**{**config.as_dict(), "input_width": int(input_width)})
        elif profile == "native10":
            config = Native10Config(input_width=int(input_width or 3072), seed=seed)
        else:
            raise ValueError("profile must be compact or native10")
        return self.store.initialize(config, seed=seed, replace=replace)

    def queue_mutation(
        self,
        shard: dict[str, Any],
        *,
        operation: str = "expert_refit",
        category: int | None = None,
        subset_seed: int = 7,
        mutation_quorum: int = MUTATION_QUORUM,
        verification_quorum: int = VERIFICATION_QUORUM,
    ) -> dict[str, Any]:
        # Lock order matches result processing: database → model store →
        # coordinator state machine. The file and SQLite transactions roll back
        # together if task creation or state persistence fails.
        with self.db.lock, self.store.transaction(), self.db.transaction(), self.lock:
            model = self.store.model()
            if operation not in PROMOTABLE_OPERATIONS:
                raise ValueError(
                    "unsupported v0.5.1 Native10 mutation operation"
                )
            category_value = int(shard.get("category") if category is None else category)
            if int(shard.get("representation_width", -1)) != model.config.representation_width:
                raise ValueError("representation shard width does not match the canonical Dendritron")
            train_x = np.asarray(shard["train_representations"], dtype=np.float32)
            train_y = np.asarray(shard["train_labels"], dtype=np.int64)
            if train_x.ndim != 2 or not len(train_x) or train_x.shape[1] != model.config.representation_width:
                raise ValueError("representation shard is invalid")
            if train_y.ndim != 1 or len(train_x) != len(train_y):
                raise ValueError("representation shard arrays are misaligned")
            start_class = category_value * model.config.classes_per_category
            stop_class = start_class + model.config.classes_per_category
            if np.any(train_y < start_class) or np.any(train_y >= stop_class):
                raise ValueError("trainer shard contains labels outside its assigned category")
            validation = self.store.global_validation()
            _, _, validation_policy = decode_global_validation_artifact(
                validation, expected_config=model.config
            )
            bundle = model.component_bundle(operation, category_value)
            shard_hash = content_hash({
                "category": category_value,
                "train_representations": shard["train_representations"],
                "train_labels": shard["train_labels"],
            })
            work_key = (
                f"native10:{bundle['base_root']}:{bundle['sha256']}:{shard_hash}:"
                f"{validation['sha256']}:{subset_seed}"
            )
            round_id = uuid.uuid4().hex
            payload = {
                "engine": "dendriswarm.native10-derived.v5",
                "round_id": round_id,
                "work_key": work_key,
                "bundle": bundle,
                "train_representations": shard["train_representations"],
                "train_labels": shard["train_labels"],
                "global_validation_hash": validation["sha256"],
                "trainer_objective_scope": "trainer-visible-training-diagnostic-not-promotion-evidence",
                "subset_seed": int(subset_seed),
                "mutation_quorum": int(mutation_quorum),
                "verification_quorum": int(verification_quorum),
                "required_tags": ["portable-numpy-v1", "deterministic-v2"],
            }
            requirements = derive_payload_requirements(TaskKind.DENDRITRON_MUTATION, payload)
            self.store.begin_round(
                {
                    "round_id": round_id,
                    "base_root": model.root,
                    "operation": operation,
                    "category": category_value,
                    "work_key": work_key,
                    "global_validation_hash": validation["sha256"],
                    "status": "mutation-quorum",
                    "started_at": time.time(),
                },
                validation_hash_value=validation["sha256"],
                max_candidate_evaluations=validation_policy.max_candidate_evaluations,
            )
            task_ids = [
                self.db.add_task(
                    TaskKind.DENDRITRON_MUTATION,
                    payload,
                    MUTATION_REWARD_UNITS,
                    60,
                    dedupe_key=f"{work_key}:mutation:{slot}",
                    requirements=requirements,
                    max_attempts=5,
                )
                for slot in range(int(mutation_quorum))
            ]
            self.db.append_audit("native10_round_started", {
                "round_id": round_id,
                "base_root": model.root,
                "operation": operation,
                "category": category_value,
                "mutation_tasks": task_ids,
                "baseline_training_included": False,
                "global_validation_hash": validation["sha256"],
                "trainer_received_global_validation": False,
            })
            return {
                "round_id": round_id,
                "base_root": model.root,
                "operation": operation,
                "category": category_value,
                "mutation_tasks": task_ids,
                "work_key": work_key,
            }

    def queue_demo_round(self, category: int = 0, operation: str = "expert_refit") -> dict[str, Any]:
        model = self.store.model()
        if not self.store.validation_path.exists():
            fixture = synthetic_global_validation_fixture(
                model.config, per_class=20 if model.config.classes <= 20 else 5
            )
            self.store.set_global_validation(fixture)
        shard = synthetic_representation_shard(model.config, category)
        return self.queue_mutation(shard, operation=operation, category=category)

    @staticmethod
    def _consensus(rows: list[Any], key_fn: Any, quorum: int) -> list[Any]:
        groups: dict[str, list[Any]] = {}
        for row in rows:
            output = json.loads(row["output"])
            key = json.dumps(key_fn(output), sort_keys=True, separators=(",", ":"))
            groups.setdefault(key, []).append(row)
        return max(groups.values(), key=len) if groups and max(map(len, groups.values())) >= quorum else []

    def handle_result(self, task: Any, result: dict[str, Any], payload: dict[str, Any], output: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        with self.lock:
            kind = TaskKind(task["kind"])
            if kind == TaskKind.DENDRITRON_MUTATION:
                validated = DendritronMutationOutput.model_validate(output)
                validate_delta(validated.delta)
                if validated.delta["base_root"] != payload["bundle"]["base_root"]:
                    raise ValueError("Dendritron mutation parent mismatch")
                if validated.delta["bundle_hash"] != payload["bundle"]["sha256"]:
                    raise ValueError("Dendritron mutation is not bound to its assigned component bundle")
                if validated.sample_count != len(payload["train_labels"]):
                    raise ValueError("Dendritron mutation diagnostic sample count mismatch")
                normalized = validated.model_dump(mode="json")
                self.db.record_work_report(payload["work_key"], kind.value, result["node_id"], task["id"], normalized)
                rows = self.db.work_reports(payload["work_key"], kind.value)
                quorum = int(payload.get("mutation_quorum", MUTATION_QUORUM))
                consensus = self._consensus(
                    rows,
                    lambda value: {
                        "delta_consensus_hash": delta_consensus_hash(value["delta"]),
                    },
                    quorum,
                )
                details: dict[str, Any] = {
                    "stage": kind.value,
                    "round_id": payload["round_id"],
                    "reports": len(rows),
                    "quorum": quorum,
                    "delta_hash": validated.delta["sha256"],
                }
                if consensus:
                    chosen = json.loads(consensus[0]["output"])
                    candidate_id = chosen["delta"]["sha256"]
                    existing = self.store.candidate(candidate_id)
                    if not existing:
                        trainer_reports = sorted(
                            ({"node_id": str(row["node_id"]), "task_id": str(row["task_id"])} for row in consensus),
                            key=lambda item: (item["node_id"], item["task_id"]),
                        )
                        trainer_nodes = [item["node_id"] for item in trainer_reports]
                        trainer_tasks = [item["task_id"] for item in trainer_reports]
                        candidate = {
                            "candidate_id": candidate_id,
                            "round_id": payload["round_id"],
                            "base_root": payload["bundle"]["base_root"],
                            "global_validation_hash": payload["global_validation_hash"],
                            "bundle": payload["bundle"],
                            "delta": chosen["delta"],
                            "operation": payload["bundle"]["operation"],
                            "category": int(payload["bundle"]["category"]),
                            "trainer_nodes": trainer_nodes,
                            "trainer_tasks": trainer_tasks,
                            "trainer_reports": trainer_reports,
                            "active_experts": chosen.get("active_experts", []),
                            "rotation_phase_before": chosen.get("rotation_phase_before"),
                            "rotation_phase_after": chosen.get("rotation_phase_after"),
                            "status": "verification-quorum",
                            "created_at": time.time(),
                        }
                        self.store.record_candidate(candidate_id, candidate)
                        verify_work_key = f"native10-verify:{candidate_id}"
                        verification_payload = {
                            "engine": "dendriswarm.native10-derived.v5",
                            "round_id": payload["round_id"],
                            "candidate_id": candidate_id,
                            "work_key": verify_work_key,
                            "bundle": payload["bundle"],
                            "delta": chosen["delta"],
                            "native10_checkpoint_root": payload["bundle"]["base_root"],
                            "global_validation_hash": payload["global_validation_hash"],
                            "verification_quorum": int(payload.get("verification_quorum", VERIFICATION_QUORUM)),
                            "required_tags": ["portable-numpy-v1", "deterministic-v2"],
                        }
                        requirements = derive_payload_requirements(TaskKind.DENDRITRON_VERIFICATION, verification_payload)
                        checkpoint_bytes = self.store.checkpoint_path.stat().st_size
                        validation_bytes = self.store.validation_path.stat().st_size
                        checkpoint_memory_mb = max(
                            256,
                            int(np.ceil(self.store.model().parameter_count * 12 / (1024 * 1024))) + 128,
                        )
                        requirements = requirements.model_copy(update={
                            "max_artifact_bytes": min(
                                2 * 1024 * 1024 * 1024,
                                checkpoint_bytes + validation_bytes + requirements.max_artifact_bytes + 256 * 1024,
                            ),
                            "min_disk_mb": max(
                                requirements.min_disk_mb,
                                int(np.ceil((checkpoint_bytes + validation_bytes + requirements.max_artifact_bytes) / (1024 * 1024))) + 1,
                            ),
                            "min_memory_mb": max(requirements.min_memory_mb, checkpoint_memory_mb),
                            "max_memory_mb": max(int(requirements.max_memory_mb or 0), checkpoint_memory_mb * 2),
                        })
                        verification_tasks = [
                            self.db.add_task(
                                TaskKind.DENDRITRON_VERIFICATION,
                                verification_payload,
                                VERIFICATION_REWARD_UNITS,
                                70,
                                parent_task=task["id"],
                                excluded_nodes=trainer_nodes,
                                dedupe_key=f"{verify_work_key}:verification:{slot}",
                                requirements=requirements,
                                max_attempts=5,
                            )
                            for slot in range(int(payload.get("verification_quorum", VERIFICATION_QUORUM)))
                        ]
                        self.db.append_audit("native10_candidate_created", {
                            "candidate_id": candidate_id,
                            "round_id": payload["round_id"],
                            "delta_hash": candidate_id,
                            "trainer_nodes": trainer_nodes,
                            "verification_tasks": verification_tasks,
                        })
                        details.update({"candidate_id": candidate_id, "verification_tasks": verification_tasks})
                    else:
                        details.update({"candidate_id": candidate_id, "verification_tasks": []})
                return 0, details

            if kind == TaskKind.DENDRITRON_VERIFICATION:
                validated = DendritronVerificationOutput.model_validate(output)
                candidate = self.store.candidate(payload["candidate_id"])
                if not candidate:
                    raise ValueError("unknown Dendritron candidate")
                if result["node_id"] in set(candidate["trainer_nodes"]):
                    raise ValueError("a mutation trainer cannot verify the same Dendritron candidate")
                if validated.delta_hash != candidate["delta"]["sha256"]:
                    raise ValueError("Dendritron verification delta hash mismatch")
                if validated.base_root != candidate["base_root"]:
                    raise ValueError("Dendritron verification parent mismatch")
                if validated.validation_hash != candidate["global_validation_hash"]:
                    raise ValueError("Dendritron verification used the wrong global validation artifact")
                normalized = validated.model_dump(mode="json")
                self.db.record_work_report(payload["work_key"], kind.value, result["node_id"], task["id"], normalized)
                rows = self.db.work_reports(payload["work_key"], kind.value)
                quorum = int(payload.get("verification_quorum", VERIFICATION_QUORUM))
                consensus = self._consensus(
                    rows,
                    lambda value: {
                        "delta_hash": value["delta_hash"],
                        "pre_correct": value["pre_correct"],
                        "post_correct": value["post_correct"],
                        "sample_count": value["sample_count"],
                        "validation_hash": value["validation_hash"],
                        "samples_by_class": value["samples_by_class"],
                        "losses_by_class": value["losses_by_class"],
                        "loss_rates_by_class": value["loss_rates_by_class"],
                    },
                    quorum,
                )
                details = {
                    "stage": kind.value,
                    "round_id": payload["round_id"],
                    "candidate_id": payload["candidate_id"],
                    "reports": len(rows),
                    "quorum": quorum,
                }
                if consensus:
                    chosen = json.loads(consensus[0]["output"])
                    validation_artifact = self.store.global_validation()
                    _, _, validation_policy = decode_global_validation_artifact(
                        validation_artifact, expected_config=self.store.model().config
                    )
                    if chosen["validation_hash"] != validation_artifact["sha256"]:
                        raise ValueError("verification consensus is not bound to the current global holdout")
                    coverage_ok = all(
                        int(count) >= validation_policy.min_samples_per_class
                        for count in chosen["samples_by_class"]
                    )
                    acceptable = (
                        bool(chosen["informative"])
                        and coverage_ok
                        and int(chosen["net_wins"]) >= validation_policy.minimum_net_wins
                        and max(chosen["losses_by_class"] or [0]) <= validation_policy.max_loss_per_class
                        and max(chosen["loss_rates_by_class"] or [0.0]) <= validation_policy.max_loss_rate_per_class
                    )
                    if not acceptable:
                        self.store.reject_round(payload["candidate_id"], "verification gate failed")
                        self.db.append_audit("native10_candidate_rejected", {
                            "candidate_id": payload["candidate_id"],
                            "net_wins": chosen["net_wins"],
                            "losses_by_class": chosen["losses_by_class"],
                            "loss_rates_by_class": chosen["loss_rates_by_class"],
                            "samples_by_class": chosen["samples_by_class"],
                            "validation_hash": chosen["validation_hash"],
                        })
                        details.update({"promoted": False, "reason": "verification-gate-failed"})
                    else:
                        verifier_nodes = sorted({str(row["node_id"]) for row in consensus})
                        record = self.store.promote(payload["candidate_id"], {
                            **chosen,
                            "verifier_nodes": verifier_nodes,
                        })
                        for report in candidate.get("trainer_reports", []):
                            self.db.credit(
                                f"reward:{report['task_id']}", report["node_id"], MUTATION_REWARD_UNITS,
                                "verified Native10 tissue mutation", report["task_id"],
                            )
                        for row in consensus:
                            self.db.credit(
                                f"reward:{row['task_id']}", row["node_id"], VERIFICATION_REWARD_UNITS,
                                "independent Native10 mutation verification", row["task_id"],
                            )
                        self.db.append_audit("native10_tissue_promoted", record)
                        details.update({"promoted": True, "contribution": record})
                return 0, details

            raise ValueError("task is not a Native10 v5 task")
