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
    DendritronV6MutationOutput,
    DendritronV6VerificationOutput,
    TaskKind,
)
from dendriswarm.core.resources import derive_payload_requirements
from dendriswarm.v6.native10 import (
    Native10Config,
    Native10Dendritron,
    parameter_reachability,
    synthetic_representation_shard,
    canonical_operation,
    decode_array,
    decode_training_tensor,
    validate_delta,
)
from dendriswarm.v6.validation import (
    GlobalValidationPolicy,
    decode_global_validation_artifact,
    make_global_validation_artifact,
    synthetic_global_validation_fixture,
    synthetic_raw_samples,
)

SEARCH_CANDIDATES = 4
VERIFICATION_QUORUM = 2
MUTATION_REWARD_UNITS = 4_000
VERIFICATION_REWARD_UNITS = 2_000
PROMOTABLE_OPERATIONS = {"expert_train", "branch_train", "repair", "scout_train", "memory_train", "field_train"}


class Native10Store:
    """Atomic local store for the canonical Dendritron and contribution lineage."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.checkpoint_path = self.path / "canonical-native10.json"
        self.validation_path = self.path / "selection-validation.json"
        self.replication_path = self.path / "replication-validation.json"
        self.baseline_reference_path = self.path / "baseline-reference.json"
        self.state_path = self.path / "state.json"
        self.lock = threading.RLock()
        if not self.state_path.exists():
            self._write_json(self.state_path, {
                "format": "dendriswarm.native10-coordinator-state.v6",
                "canonical_root": None,
                "global_validation_hash": None,
                "global_validation_evaluations": 0,
                "global_validation_rounds": 0,
                "replication_validation_hash": None,
                "replication_validation_evaluations": 0,
                "replication_validation_rounds": 0,
                "active_round": None,
                "search_rounds_completed": 0,
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
            replication_before = self.replication_path.read_bytes() if self.replication_path.exists() else None
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
                if replication_before is None:
                    self.replication_path.unlink(missing_ok=True)
                else:
                    fd, temporary = tempfile.mkstemp(prefix=f".{self.replication_path.name}.", dir=self.replication_path.parent)
                    with os.fdopen(fd, "wb") as handle:
                        handle.write(replication_before)
                        handle.flush()
                        os.fsync(handle.fileno())
                    os.replace(temporary, self.replication_path)
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
            self.replication_path.unlink(missing_ok=True)
            state = self.state()
            state.update({
                "canonical_root": model.root,
                "global_validation_hash": None,
                "global_validation_evaluations": 0,
                "global_validation_rounds": 0,
                "replication_validation_hash": None,
                "replication_validation_evaluations": 0,
                "replication_validation_rounds": 0,
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
            self.replication_path.unlink(missing_ok=True)
            state = self.state()
            state.update({
                "canonical_root": model.root,
                "global_validation_hash": None,
                "global_validation_evaluations": 0,
                "global_validation_rounds": 0,
                "replication_validation_hash": None,
                "replication_validation_evaluations": 0,
                "replication_validation_rounds": 0,
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
            state["global_validation_rounds"] = 0
            state["global_validation_policy"] = policy.as_dict()
            state["global_validation_source"] = artifact.get("source")
            self.save_state(state)
            return self.validation_status()

    def global_validation(self) -> dict[str, Any]:
        if not self.validation_path.exists():
            raise ValueError("coordinator-held selection validation is not configured")
        artifact = json.loads(self.validation_path.read_text())
        decode_global_validation_artifact(artifact, expected_config=self.model().config)
        return artifact

    def set_replication_validation(self, artifact: dict[str, Any], *, replace: bool = False) -> dict[str, Any]:
        """Install a trainer-invisible holdout used only after candidate selection and replay."""
        with self.lock:
            if self.state().get("active_round") is not None:
                raise ValueError("cannot replace replication validation during an active round")
            model = self.model()
            if self.replication_path.exists() and not replace:
                raise ValueError("a replication validation artifact already exists")
            _, _, policy = decode_global_validation_artifact(artifact, expected_config=model.config)
            if self.validation_path.exists() and artifact["sha256"] == self.global_validation()["sha256"]:
                raise ValueError("selection and replication validation artifacts must be distinct")
            self._write_json(self.replication_path, artifact)
            state = self.state()
            state["replication_validation_hash"] = artifact["sha256"]
            state["replication_validation_evaluations"] = 0
            state["replication_validation_rounds"] = 0
            state["replication_validation_policy"] = policy.as_dict()
            state["replication_validation_source"] = artifact.get("source")
            self.save_state(state)
            return self.replication_validation_status()

    def replication_validation(self) -> dict[str, Any]:
        if not self.replication_path.exists():
            raise ValueError("coordinator-held final replication validation is not configured")
        artifact = json.loads(self.replication_path.read_text())
        decode_global_validation_artifact(artifact, expected_config=self.model().config)
        return artifact

    def validation_by_hash(self, value: str) -> dict[str, Any]:
        for getter in (self.global_validation, self.replication_validation):
            try:
                artifact = getter()
            except ValueError:
                continue
            if artifact.get("sha256") == value:
                return artifact
        raise ValueError("unknown Native10 v0.6 validation artifact")

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
            "search_rounds": int(state.get("global_validation_rounds", 0)),
            "max_search_rounds": int(artifact["policy"].get("max_search_rounds", 1)),
            "max_candidate_evaluations": int(artifact["policy"]["max_candidate_evaluations"]),
            "counts_by_class": artifact["counts_by_class"],
            "source": artifact["source"],
            "split": artifact["split"],
            "trainer_visible": False,
            "protocol_fixture_only": artifact.get("protocol_fixture_only", False),
            "policy": artifact["policy"],
        }

    def replication_validation_status(self) -> dict[str, Any]:
        if not self.replication_path.exists():
            return {"configured": False, "sha256": None}
        artifact = self.replication_validation()
        state = self.state()
        return {
            "configured": True,
            "sha256": artifact["sha256"],
            "sample_count": artifact["sample_count"],
            "candidate_evaluations": int(state.get("replication_validation_evaluations", 0)),
            "search_rounds": int(state.get("replication_validation_rounds", 0)),
            "max_search_rounds": int(artifact["policy"].get("max_search_rounds", 1)),
            "max_candidate_evaluations": int(artifact["policy"]["max_candidate_evaluations"]),
            "counts_by_class": artifact["counts_by_class"],
            "source": artifact["source"],
            "split": artifact["split"],
            "trainer_visible": False,
            "protocol_fixture_only": artifact.get("protocol_fixture_only", False),
            "policy": artifact["policy"],
        }


    def set_baseline_reference(self, artifact: dict[str, Any], *, replace: bool = False) -> dict[str, Any]:
        """Install an external, provenance-bound baseline result without training it."""
        from dendriswarm.v6.benchmark import validate_baseline_reference
        with self.lock:
            validated = validate_baseline_reference(dict(artifact))
            if self.baseline_reference_path.exists() and not replace:
                raise ValueError("an external baseline reference is already installed")
            self._write_json(self.baseline_reference_path, validated)
            return self.baseline_reference_status()

    def baseline_reference(self) -> dict[str, Any]:
        from dendriswarm.v6.benchmark import validate_baseline_reference
        if not self.baseline_reference_path.exists():
            raise ValueError("no external baseline reference is installed")
        return validate_baseline_reference(json.loads(self.baseline_reference_path.read_text()))

    def baseline_reference_status(self) -> dict[str, Any]:
        if not self.baseline_reference_path.exists():
            return {"configured": False, "sha256": None}
        artifact = self.baseline_reference()
        return {
            "configured": True,
            "sha256": artifact["sha256"],
            "dataset": artifact["dataset"],
            "split": artifact["split"],
            "metric": artifact["metric"],
            "value": artifact["value"],
            "model": artifact["model"],
            "source": artifact["source"],
            "evidence_sha256": artifact["evidence_sha256"],
            "training_code_included": False,
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
                "replication_validation": self.replication_validation_status(),
                "baseline_reference": self.baseline_reference_status(),
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
            "replication_validation": self.replication_validation_status(),
            "baseline_reference": self.baseline_reference_status(),
            "parameter_reachability": parameter_reachability(model.config),
            "search_rounds_completed": int(state.get("search_rounds_completed", 0)),
        }

    def begin_round(
        self,
        round_value: dict[str, Any],
        *,
        validation_hash_value: str,
        max_candidate_evaluations: int,
        max_selection_rounds: int,
        replication_hash_value: str,
        max_replication_evaluations: int,
        max_replication_rounds: int,
        candidate_evaluations: int = 1,
    ) -> None:
        with self.lock:
            state = self.state()
            if state.get("active_round") is not None:
                raise ValueError("one Native10 contribution round is already active")
            if round_value["base_root"] != state.get("canonical_root"):
                raise ValueError("round parent is not canonical")
            if state.get("global_validation_hash") != validation_hash_value:
                raise ValueError("round selection artifact is not the installed coordinator artifact")
            if state.get("replication_validation_hash") != replication_hash_value:
                raise ValueError("round replication artifact is not the installed coordinator artifact")
            evaluations = int(state.get("global_validation_evaluations", 0))
            replication_evaluations = int(state.get("replication_validation_evaluations", 0))
            selection_rounds = int(state.get("global_validation_rounds", 0))
            replication_rounds = int(state.get("replication_validation_rounds", 0))
            requested = max(1, int(candidate_evaluations))
            if selection_rounds + 1 > int(max_selection_rounds):
                raise ValueError(
                    "selection validation tournament is exhausted; rotate the coordinator-held artifact"
                )
            if replication_rounds + 1 > int(max_replication_rounds):
                raise ValueError(
                    "final replication artifact is one-shot; rotate it before another round"
                )
            if evaluations + requested > int(max_candidate_evaluations):
                raise ValueError(
                    "selection validation candidate budget is exhausted; rotate the coordinator-held artifact"
                )
            if replication_evaluations + 1 > int(max_replication_evaluations):
                raise ValueError(
                    "final replication evaluation budget is exhausted; rotate the coordinator-held artifact"
                )
            round_value = {
                **round_value,
                "validation_evaluation_start": evaluations + 1,
                "validation_evaluation_count": requested,
                "replication_evaluation_index": replication_evaluations + 1,
                "replication_validation_hash": replication_hash_value,
            }
            state["global_validation_evaluations"] = evaluations + requested
            state["global_validation_rounds"] = selection_rounds + 1
            state["replication_validation_evaluations"] = replication_evaluations + 1
            state["replication_validation_rounds"] = replication_rounds + 1
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

    def update_candidate(self, candidate_id: str, **changes: Any) -> dict[str, Any]:
        with self.lock:
            state = self.state()
            candidate = state.get("candidates", {}).get(candidate_id)
            if not candidate:
                raise ValueError("unknown Native10 candidate")
            candidate.update(changes)
            self.save_state(state)
            return dict(candidate)

    def set_round_candidates(self, candidate_ids: list[str]) -> None:
        with self.lock:
            state = self.state()
            active = state.get("active_round")
            if active is None:
                raise ValueError("no active Native10 search round")
            active["candidate_ids"] = list(candidate_ids)
            active["status"] = "blind-global-verification"
            self.save_state(state)

    def close_without_promotion(self, reason: str) -> None:
        with self.lock:
            state = self.state()
            active = state.get("active_round") or {}
            for candidate_id in active.get("candidate_ids", []):
                if candidate_id in state.get("candidates", {}) and state["candidates"][candidate_id].get("status") not in {"accepted", "rejected"}:
                    state["candidates"][candidate_id]["status"] = "rejected"
                    state["candidates"][candidate_id]["reason"] = reason
            state["active_round"] = None
            state["search_rounds_completed"] = int(state.get("search_rounds_completed", 0)) + 1
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
                "wins": int(verification["wins"]),
                "losses": int(verification["losses"]),
                "mcnemar_p_value": float(verification["mcnemar_p_value"]),
                "effect_rate": float(verification["effect_rate"]),
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
            state["search_rounds_completed"] = int(state.get("search_rounds_completed", 0)) + 1
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
        operation: str = "expert_train",
        category: int | None = None,
        subset_seed: int = 7,
        search_candidates: int = SEARCH_CANDIDATES,
        verification_quorum: int = VERIFICATION_QUORUM,
        optimizer_steps: int = 24,
        learning_rate: float = 0.04,
        search_recipes: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Queue independent local search trajectories for one bounded tissue."""
        with self.db.lock, self.store.transaction(), self.db.transaction(), self.lock:
            model = self.store.model()
            operation = canonical_operation(operation)
            target = int(shard.get("category", 0) if category is None else category)
            if not 2 <= int(search_candidates) <= 32:
                raise ValueError("search_candidates must be between 2 and 32")
            if not 2 <= int(verification_quorum) <= 8:
                raise ValueError("verification_quorum must be between 2 and 8")
            labels = np.asarray(shard["train_labels"], dtype=np.int64)
            if labels.ndim != 1 or not len(labels) or np.any(labels < 0) or np.any(labels >= model.config.classes):
                raise ValueError("training labels are invalid")
            if operation == "field_train":
                data_key = "train_inputs"
                raw_data = shard[data_key]
                data = decode_training_tensor(raw_data)
                expected_width = model.config.input_width
                if not 0 <= target < model.config.field_blocks:
                    raise ValueError("field target is invalid")
            else:
                data_key = "train_representations"
                raw_data = shard[data_key]
                data = decode_training_tensor(raw_data)
                expected_width = model.config.representation_width
                if not 0 <= target < model.config.categories:
                    raise ValueError("category target is invalid")
            if data.ndim != 2 or data.shape[1] != expected_width or len(data) != len(labels) or not np.isfinite(data).all():
                raise ValueError("training shard is invalid")
            if operation in {"expert_train", "branch_train", "repair", "memory_train"}:
                start = target * model.config.classes_per_category
                stop = start + model.config.classes_per_category
                if np.any(labels < start) or np.any(labels >= stop):
                    raise ValueError("local tissue shard contains labels outside its category")
            if operation == "scout_train":
                membership = labels // model.config.classes_per_category == target
                if not membership.any() or membership.all():
                    raise ValueError("scout training requires positive and negative categories")
            validation = self.store.global_validation()
            replication = self.store.replication_validation()
            _, _, validation_policy = decode_global_validation_artifact(validation, expected_config=model.config)
            _, _, replication_policy = decode_global_validation_artifact(replication, expected_config=model.config)
            bundle = model.component_bundle(operation, target)
            shard_hash = content_hash({"target": target, "operation": operation, "data": shard[data_key], "labels": shard["train_labels"]})
            recipes = [dict(value) for value in (search_recipes or [])]
            if recipes and len(recipes) != int(search_candidates):
                raise ValueError("search_recipes must match search_candidates")
            round_id = uuid.uuid4().hex
            common_work_key = f"native10-v6-search:{round_id}"
            self.store.begin_round(
                {
                    "round_id": round_id, "base_root": model.root, "operation": operation, "category": target,
                    "work_key": common_work_key, "global_validation_hash": validation["sha256"],
                    "replication_validation_hash": replication["sha256"],
                    "status": "independent-search", "expected_search_reports": int(search_candidates),
                    "candidate_ids": [], "started_at": time.time(), "shard_hash": shard_hash,
                },
                validation_hash_value=validation["sha256"],
                max_candidate_evaluations=validation_policy.max_candidate_evaluations,
                max_selection_rounds=validation_policy.max_search_rounds,
                replication_hash_value=replication["sha256"],
                max_replication_evaluations=replication_policy.max_candidate_evaluations,
                max_replication_rounds=replication_policy.max_search_rounds,
                candidate_evaluations=int(search_candidates),
            )
            task_ids: list[str] = []
            for slot in range(int(search_candidates)):
                search_seed = int(subset_seed) + slot * 1009
                step_multiplier = (0.75, 1.0, 1.25, 1.5)[slot % 4]
                rate_multiplier = (1.15, 1.0, 0.85, 0.70)[slot % 4]
                recipe = recipes[slot] if recipes else {}
                candidate_steps = max(4, int(round(int(optimizer_steps) * float(recipe.get("steps_multiplier", step_multiplier)))))
                candidate_rate = float(learning_rate) * float(recipe.get("learning_rate_multiplier", rate_multiplier))
                payload = {
                    "engine": "dendriswarm.native10-trainable.v6", "round_id": round_id,
                    "work_key": common_work_key, "bundle": bundle, "train_data": shard[data_key],
                    "train_labels": shard["train_labels"], "global_validation_hash": validation["sha256"],
                    "search_seed": search_seed, "optimizer_steps": candidate_steps,
                    "learning_rate": candidate_rate, "search_recipe": recipe,
                    "search_candidates": int(search_candidates),
                    "replication_validation_hash": replication["sha256"],
                    "verification_quorum": int(verification_quorum),
                    "trainer_objective_scope": "trainer-visible-training-diagnostic-not-promotion-evidence",
                    "required_tags": ["portable-numpy-v1", "independent-search-v1"],
                }
                requirements = derive_payload_requirements(TaskKind.DENDRITRON_MUTATION, payload)
                task_ids.append(self.db.add_task(
                    TaskKind.DENDRITRON_MUTATION, payload, MUTATION_REWARD_UNITS, 60,
                    dedupe_key=f"{common_work_key}:search:{slot}", requirements=requirements, max_attempts=5,
                ))
            self.db.append_audit("native10_v6_search_started", {
                "round_id": round_id, "base_root": model.root, "operation": operation, "target": target,
                "search_tasks": task_ids, "independent_search_seeds": [int(subset_seed)+slot*1009 for slot in range(int(search_candidates))],
                "global_validation_hash": validation["sha256"], "replication_validation_hash": replication["sha256"],
                "trainer_received_global_validation": False, "trainer_received_replication_validation": False,
            })
            return {"round_id": round_id, "base_root": model.root, "operation": operation, "category": target, "search_tasks": task_ids, "work_key": common_work_key}

    def queue_demo_round(self, category: int = 0, operation: str = "expert_train") -> dict[str, Any]:
        """Queue a protocol fixture using a training split disjoint from hidden validation."""
        model = self.store.model()
        per_class = 32 if model.config.classes <= 20 else 8
        if not self.store.validation_path.exists():
            self.store.set_global_validation(synthetic_global_validation_fixture(model, per_class=per_class))
        if not self.store.replication_path.exists():
            replication_x, replication_y = synthetic_raw_samples(
                model.config,
                per_class=per_class,
                prototype_seed=20260723,
                sample_seed=20260726,
            )
            replication_policy = GlobalValidationPolicy(
                min_samples_per_class=min(5, per_class),
                familywise_alpha=0.05,
                max_candidate_evaluations=20,
                min_discordant=min(10, max(1, model.config.classes)),
                minimum_net_wins=1,
                minimum_effect_rate=0.0,
                max_loss_per_class=max(1, per_class // 4),
                max_loss_rate_per_class=0.25,
            )
            self.store.set_replication_validation(
                make_global_validation_artifact(
                    model.config,
                    replication_x,
                    replication_y,
                    source="synthetic-raw-all-class-final-replication-fixture",
                    split="coordinator-private-final-replication",
                    policy=replication_policy,
                    protocol_fixture_only=True,
                )
            )
        operation = canonical_operation(operation)
        # Same latent class prototypes, independent sample noise. The generated
        # training rows are never taken from the coordinator-held validation artifact.
        raw, labels = synthetic_raw_samples(
            model.config,
            per_class=40 if model.config.classes <= 20 else 10,
            prototype_seed=20260723,
            sample_seed=20260725,
        )
        if operation == "field_train":
            shard = {"category": int(category), "train_inputs": raw.tolist(), "train_labels": labels.tolist()}
        else:
            representations = model.encode(raw)
            if operation == "scout_train":
                selected_x, selected_y = representations, labels
            else:
                mask = labels // model.config.classes_per_category == int(category)
                selected_x, selected_y = representations[mask], labels[mask]
            shard = {
                "category": int(category),
                "train_representations": selected_x.tolist(),
                "train_labels": selected_y.tolist(),
            }
        return self.queue_mutation(
            shard,
            operation=operation,
            category=category,
            optimizer_steps=60,
            learning_rate=0.035,
        )


    @staticmethod
    def _consensus(rows: list[Any], key_fn: Any, quorum: int) -> list[Any]:
        groups: dict[str, list[Any]] = {}
        for row in rows:
            output = json.loads(row["output"])
            key = json.dumps(key_fn(output), sort_keys=True, separators=(",", ":"))
            groups.setdefault(key, []).append(row)
        return max(groups.values(), key=len) if groups and max(map(len, groups.values())) >= quorum else []

    def _queue_verification_tasks(self, candidate: dict[str, Any], parent_task_id: str, verification_quorum: int) -> list[str]:
        candidate_id = candidate["candidate_id"]
        payload = {
            "engine": "dendriswarm.native10-trainable.v6",
            "mode": "global-verification",
            "round_id": candidate["round_id"],
            "candidate_id": candidate_id,
            "work_key": f"native10-v6-verify:{candidate_id}",
            "bundle": candidate["bundle"],
            "delta": candidate["delta"],
            "native10_checkpoint_root": candidate["base_root"],
            "global_validation_hash": candidate["global_validation_hash"],
            "validation_sample_count": int(self.store.global_validation()["sample_count"]),
            "verification_quorum": int(verification_quorum),
            "required_tags": ["portable-numpy-v1", "blind-global-verification-v2"],
        }
        requirements = derive_payload_requirements(TaskKind.DENDRITRON_VERIFICATION, payload)
        checkpoint_bytes = self.store.checkpoint_path.stat().st_size
        validation_bytes = self.store.validation_path.stat().st_size
        checkpoint_memory_mb = max(256, int(np.ceil(self.store.model().parameter_count * 12 / (1024 * 1024))) + 128)
        requirements = requirements.model_copy(update={
            "max_artifact_bytes": min(2 * 1024 * 1024 * 1024, checkpoint_bytes + validation_bytes + requirements.max_artifact_bytes + 256 * 1024),
            "min_disk_mb": max(requirements.min_disk_mb, int(np.ceil((checkpoint_bytes + validation_bytes + requirements.max_artifact_bytes) / (1024 * 1024))) + 1),
            "min_memory_mb": max(requirements.min_memory_mb, checkpoint_memory_mb),
            "max_memory_mb": max(int(requirements.max_memory_mb or 0), checkpoint_memory_mb * 2),
        })
        excluded = sorted(set(candidate["trainer_nodes"]))
        return [
            self.db.add_task(
                TaskKind.DENDRITRON_VERIFICATION, payload, VERIFICATION_REWARD_UNITS, 70,
                parent_task=parent_task_id, excluded_nodes=excluded,
                dedupe_key=f"{payload['work_key']}:slot:{slot}", requirements=requirements, max_attempts=5,
            )
            for slot in range(int(verification_quorum))
        ]

    def _queue_replay_task(self, candidate: dict[str, Any]) -> str:
        source_task = self.db.task(candidate["source_task_id"])
        if source_task is None:
            raise ValueError("candidate source task no longer exists")
        source_payload = json.loads(source_task["payload"])
        replay_payload = {
            **source_payload,
            "mode": "replay-audit",
            "candidate_id": candidate["candidate_id"],
            "work_key": f"native10-v6-replay:{candidate['candidate_id']}",
            "search_candidates": 1,
        }
        requirements = derive_payload_requirements(TaskKind.DENDRITRON_MUTATION, replay_payload)
        task_id = self.db.add_task(
            TaskKind.DENDRITRON_MUTATION, replay_payload, VERIFICATION_REWARD_UNITS, 75,
            parent_task=candidate["source_task_id"], excluded_nodes=sorted(set(candidate["trainer_nodes"])),
            dedupe_key=f"native10-v6-replay:{candidate['candidate_id']}", requirements=requirements, max_attempts=5,
        )
        self.store.update_candidate(candidate["candidate_id"], status="replay-audit", replay_task_id=task_id)
        return task_id

    def _queue_replication_tasks(self, candidate: dict[str, Any], replay_task_id: str, verification_quorum: int) -> list[str]:
        payload = {
            "engine": "dendriswarm.native10-trainable.v6",
            "mode": "final-replication",
            "round_id": candidate["round_id"],
            "candidate_id": candidate["candidate_id"],
            "work_key": f"native10-v6-replication:{candidate['candidate_id']}",
            "bundle": candidate["bundle"],
            "delta": candidate["delta"],
            "native10_checkpoint_root": candidate["base_root"],
            "global_validation_hash": candidate["replication_validation_hash"],
            "validation_sample_count": int(self.store.replication_validation()["sample_count"]),
            "verification_quorum": int(verification_quorum),
            "required_tags": ["portable-numpy-v1", "blind-global-verification-v2"],
        }
        requirements = derive_payload_requirements(TaskKind.DENDRITRON_VERIFICATION, payload)
        checkpoint_bytes = self.store.checkpoint_path.stat().st_size
        validation_bytes = self.store.replication_path.stat().st_size
        checkpoint_memory_mb = max(256, int(np.ceil(self.store.model().parameter_count * 12 / (1024 * 1024))) + 128)
        requirements = requirements.model_copy(update={
            "max_artifact_bytes": min(2 * 1024 * 1024 * 1024, checkpoint_bytes + validation_bytes + requirements.max_artifact_bytes + 256 * 1024),
            "min_disk_mb": max(requirements.min_disk_mb, int(np.ceil((checkpoint_bytes + validation_bytes + requirements.max_artifact_bytes) / (1024 * 1024))) + 1),
            "min_memory_mb": max(requirements.min_memory_mb, checkpoint_memory_mb),
            "max_memory_mb": max(int(requirements.max_memory_mb or 0), checkpoint_memory_mb * 2),
        })
        excluded = sorted(set(candidate.get("trainer_nodes", [])) | set(candidate.get("verifier_nodes", [])) | {str(candidate.get("replay_node", ""))})
        return [
            self.db.add_task(
                TaskKind.DENDRITRON_VERIFICATION,
                payload,
                VERIFICATION_REWARD_UNITS,
                80,
                parent_task=replay_task_id,
                excluded_nodes=[node for node in excluded if node],
                dedupe_key=f"{payload['work_key']}:slot:{slot}",
                requirements=requirements,
                max_attempts=5,
            )
            for slot in range(int(verification_quorum))
        ]

    def _ranked_accepted_candidates(self) -> list[dict[str, Any]]:
        state = self.store.state()
        active = state.get("active_round") or {}
        candidates = [state["candidates"][candidate_id] for candidate_id in active.get("candidate_ids", []) if candidate_id in state.get("candidates", {})]
        accepted = [candidate for candidate in candidates if candidate.get("status") in {"accepted", "replay-failed"}]
        return sorted(
            accepted,
            key=lambda candidate: (
                -int(candidate["verification"]["net_wins"]),
                float(candidate["verification"]["mcnemar_p_value"]),
                int(candidate["delta"]["changed_parameters"]),
                candidate["candidate_id"],
            ),
        )

    def _advance_round_after_verification(self) -> dict[str, Any]:
        state = self.store.state()
        active = state.get("active_round") or {}
        candidate_ids = list(active.get("candidate_ids", []))
        if not candidate_ids:
            return {"ready": False}
        candidates = [state.get("candidates", {}).get(candidate_id) for candidate_id in candidate_ids]
        if any(candidate is None or candidate.get("status") not in {"accepted", "rejected", "replay-failed", "replay-audit"} for candidate in candidates):
            return {"ready": False}
        if any(candidate.get("status") == "replay-audit" for candidate in candidates if candidate):
            return {"ready": False}
        ranked = [candidate for candidate in self._ranked_accepted_candidates() if candidate.get("status") == "accepted"]
        if not ranked:
            self.store.close_without_promotion("no statistically valid candidate")
            self.db.append_audit("native10_v6_round_rejected", {"round_id": active.get("round_id"), "reason": "no-statistically-valid-candidate"})
            return {"ready": True, "promoted": False}
        replay_task = self._queue_replay_task(ranked[0])
        return {"ready": True, "promoted": False, "replay_task": replay_task, "selected_candidate": ranked[0]["candidate_id"]}

    def _promote_after_replication(self, candidate: dict[str, Any]) -> dict[str, Any]:
        selection = dict(candidate["verification"])
        replication = dict(candidate["replication_verification"])
        promotion_evidence = {
            **replication,
            "validation_hash": replication["validation_hash"],
            "verifier_nodes": list(candidate["replication_verifier_nodes"]),
        }
        state_before = self.store.state()
        active_before = state_before.get("active_round") or {}
        round_candidates = [
            state_before.get("candidates", {}).get(candidate_id)
            for candidate_id in active_before.get("candidate_ids", [])
        ]
        round_candidates = [value for value in round_candidates if value]
        search_reports = [report for value in round_candidates for report in value.get("trainer_reports", [])]
        selection_reports = [report for value in round_candidates for report in value.get("verifier_reports", [])]
        replication_reports = list(candidate.get("replication_verifier_reports", []))
        worker_ms = sum(int(report.get("duration_ms", 0)) for report in search_reports + selection_reports + replication_reports)
        worker_ms += int(candidate.get("replay_duration_ms", 0))
        round_compute = {
            "candidate_searches": len(round_candidates),
            "search_reports": len(search_reports),
            "selection_verification_reports": len(selection_reports),
            "replay_reports": 1 if candidate.get("replay_task_id") else 0,
            "replication_verification_reports": len(replication_reports),
            "contributed_worker_seconds": worker_ms / 1000.0,
            "contributed_worker_hours": worker_ms / 3_600_000.0,
            "accepted_candidates": sum(1 for value in round_candidates if value.get("status") in {"accepted", "replication-verification", "replicated"}),
            "promoted_candidates": 1,
            "search_yield": 1.0 / max(1, len(round_candidates)),
        }
        record = self.store.promote(candidate["candidate_id"], promotion_evidence)
        record["round_compute"] = round_compute
        record["selection_evidence"] = {
            key: selection[key]
            for key in ("validation_hash", "sample_count", "wins", "losses", "net_wins", "mcnemar_p_value", "effect_rate")
        }
        record["replication_evidence"] = {
            key: replication[key]
            for key in ("validation_hash", "sample_count", "wins", "losses", "net_wins", "mcnemar_p_value", "effect_rate")
        }
        # Persist the full two-stage evidence rather than returning an enriched
        # transient view while storing only the replication summary.
        state = self.store.state()
        if state.get("contributions"):
            state["contributions"][-1] = dict(record)
            self.store.save_state(state)
        for report in candidate.get("trainer_reports", []):
            self.db.credit(f"reward:{report['task_id']}", report["node_id"], MUTATION_REWARD_UNITS, "selected trainable Native10 search candidate", report["task_id"])
        for report in candidate.get("verifier_reports", []):
            self.db.credit(f"reward:{report['task_id']}", report["node_id"], VERIFICATION_REWARD_UNITS, "independent selection verification", report["task_id"])
        self.db.credit(
            f"reward:{candidate['replay_task_id']}",
            candidate["replay_node"],
            VERIFICATION_REWARD_UNITS,
            "independent Dendritron replay audit",
            candidate["replay_task_id"],
        )
        for report in candidate.get("replication_verifier_reports", []):
            self.db.credit(f"reward:{report['task_id']}", report["node_id"], VERIFICATION_REWARD_UNITS, "fresh final Dendritron replication", report["task_id"])
        self.db.append_audit("native10_v6_tissue_promoted", record)
        return record

    @staticmethod
    def _verification_consensus_key(value: dict[str, Any]) -> dict[str, Any]:
        """Return every evidence field that can affect admission, excluding runtime only.

        Quorum agreement must cover the full integer evidence vector.  Derived
        floating-point fields are checked again by the coordinator below.
        """
        return {
            key: value[key]
            for key in (
                "delta_hash", "validation_hash", "base_root", "operation", "category",
                "sample_count", "pre_correct", "post_correct", "net_wins", "wins",
                "losses", "discordant", "effect_rate", "mcnemar_p_value",
                "corrected_alpha", "statistically_significant",
                "pre_correct_by_class", "post_correct_by_class",
                "samples_by_class", "losses_by_class", "loss_rates_by_class",
                "informative", "write_set",
            )
        }

    def _verification_gate(
        self, chosen: dict[str, Any], artifact: dict[str, Any], candidate: dict[str, Any] | None = None
    ) -> bool:
        _, artifact_labels, policy = decode_global_validation_artifact(
            artifact, expected_config=self.store.model().config
        )
        from dendriswarm.v6.validation import exact_one_sided_mcnemar

        config = self.store.model().config
        expected_counts = np.bincount(artifact_labels, minlength=config.classes).astype(int).tolist()
        vector_names = (
            "pre_correct_by_class", "post_correct_by_class", "samples_by_class",
            "losses_by_class", "loss_rates_by_class",
        )
        if any(len(chosen[name]) != config.classes for name in vector_names):
            raise ValueError("verifier class evidence does not cover every model class")
        if int(chosen["sample_count"]) != len(artifact_labels):
            raise ValueError("verifier sample count does not match the committed artifact")
        if [int(value) for value in chosen["samples_by_class"]] != expected_counts:
            raise ValueError("verifier class counts do not match the committed artifact")
        if int(chosen["pre_correct"]) != sum(int(value) for value in chosen["pre_correct_by_class"]):
            raise ValueError("verifier pre-correct total is inconsistent with class evidence")
        if int(chosen["post_correct"]) != sum(int(value) for value in chosen["post_correct_by_class"]):
            raise ValueError("verifier post-correct total is inconsistent with class evidence")

        expected_losses: list[int] = []
        expected_rates: list[float] = []
        for pre, post, count in zip(
            chosen["pre_correct_by_class"],
            chosen["post_correct_by_class"],
            chosen["samples_by_class"],
            strict=True,
        ):
            pre_i, post_i, count_i = int(pre), int(post), int(count)
            if not (0 <= pre_i <= count_i and 0 <= post_i <= count_i):
                raise ValueError("verifier class-correct counts exceed class sample counts")
            loss = max(0, pre_i - post_i)
            expected_losses.append(loss)
            expected_rates.append(loss / count_i if count_i else 0.0)
        if [int(value) for value in chosen["losses_by_class"]] != expected_losses:
            raise ValueError("verifier class losses are inconsistent with class evidence")
        if any(
            abs(float(reported) - expected) > 1e-12
            for reported, expected in zip(chosen["loss_rates_by_class"], expected_rates, strict=True)
        ):
            raise ValueError("verifier class loss rates are inconsistent with class evidence")

        expected_wins = int(chosen["wins"])
        expected_losses_total = int(chosen["losses"])
        expected_discordant = expected_wins + expected_losses_total
        expected_net = expected_wins - expected_losses_total
        if int(chosen["discordant"]) != expected_discordant:
            raise ValueError("verifier discordant count is inconsistent")
        if int(chosen["net_wins"]) != expected_net:
            raise ValueError("verifier net wins are inconsistent")
        if int(chosen["post_correct"]) - int(chosen["pre_correct"]) != expected_net:
            raise ValueError("verifier aggregate correctness is inconsistent with paired counts")
        expected_effect = expected_net / max(1, int(chosen["sample_count"]))
        if abs(float(chosen["effect_rate"]) - expected_effect) > 1e-12:
            raise ValueError("verifier effect rate is inconsistent with paired counts")

        expected_p = exact_one_sided_mcnemar(expected_wins, expected_losses_total)
        if abs(expected_p - float(chosen["mcnemar_p_value"])) > 1e-12:
            raise ValueError("verifier McNemar probability is inconsistent with paired counts")
        if abs(float(chosen["corrected_alpha"]) - policy.corrected_alpha) > 1e-15:
            raise ValueError("verifier used an uncommitted significance threshold")
        expected_informative = expected_discordant > 0
        if bool(chosen["informative"]) != expected_informative:
            raise ValueError("verifier informative flag is inconsistent")
        expected_significant = bool(
            expected_discordant >= policy.min_discordant
            and expected_net >= policy.minimum_net_wins
            and expected_effect >= policy.minimum_effect_rate
            and expected_p <= policy.corrected_alpha
        )
        if bool(chosen["statistically_significant"]) != expected_significant:
            raise ValueError("verifier significance flag is inconsistent")

        if candidate is not None:
            if str(chosen["operation"]) != str(candidate["operation"]):
                raise ValueError("verifier operation is not bound to the candidate")
            if int(chosen["category"]) != int(candidate["category"]):
                raise ValueError("verifier target is not bound to the candidate")

        coverage_ok = all(count >= policy.min_samples_per_class for count in expected_counts)
        return bool(
            expected_informative
            and expected_significant
            and coverage_ok
            and max(expected_losses or [0]) <= policy.max_loss_per_class
            and max(expected_rates or [0.0]) <= policy.max_loss_rate_per_class
        )

    def _try_next_candidate_after_failure(self, failed_candidate_id: str, reason: str) -> dict[str, Any]:
        self.store.update_candidate(failed_candidate_id, status="rejected", rejection_reason=reason)
        ranked = [
            item for item in self._ranked_accepted_candidates()
            if item["candidate_id"] != failed_candidate_id and item.get("status") == "accepted"
        ]
        if ranked:
            replay_task = self._queue_replay_task(ranked[0])
            return {"promoted": False, "next_candidate": ranked[0]["candidate_id"], "replay_task": replay_task}
        self.store.close_without_promotion(reason)
        return {"promoted": False}

    def handle_result(self, task: Any, result: dict[str, Any], payload: dict[str, Any], output: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        with self.lock:
            kind = TaskKind(task["kind"])
            if kind == TaskKind.DENDRITRON_MUTATION:
                validated = DendritronV6MutationOutput.model_validate(output)
                validate_delta(validated.delta, payload["bundle"])
                if validated.search_seed != int(payload.get("search_seed", -1)):
                    raise ValueError("Dendritron search seed mismatch")
                if validated.sample_count != len(payload["train_labels"]):
                    raise ValueError("Dendritron trainer diagnostic count mismatch")
                if payload.get("mode") == "replay-audit":
                    candidate = self.store.candidate(payload["candidate_id"])
                    if not candidate:
                        raise ValueError("unknown replay candidate")
                    from dendriswarm.v6.native10 import delta_consensus_hash
                    replay_ok = delta_consensus_hash(validated.delta) == delta_consensus_hash(candidate["delta"])
                    if replay_ok:
                        candidate = self.store.update_candidate(
                            candidate["candidate_id"],
                            status="replication-verification",
                            replay_node=result["node_id"],
                            replay_task_id=task["id"],
                            replay_duration_ms=int(result.get("duration_ms", 0)),
                        )
                        quorum = int(payload.get("verification_quorum", VERIFICATION_QUORUM))
                        replication_tasks = self._queue_replication_tasks(candidate, task["id"], quorum)
                        self.store.update_candidate(candidate["candidate_id"], replication_tasks=replication_tasks)
                        self.db.append_audit("native10_v6_replay_passed", {
                            "candidate_id": candidate["candidate_id"],
                            "replay_node": result["node_id"],
                            "replication_tasks": replication_tasks,
                        })
                        return 0, {
                            "stage": "replay-audit",
                            "replay_ok": True,
                            "promoted": False,
                            "replication_tasks": replication_tasks,
                        }
                    self.store.update_candidate(candidate["candidate_id"], status="replay-failed", replay_node=result["node_id"])
                    self.db.append_audit("native10_v6_replay_failed", {"candidate_id": candidate["candidate_id"], "node_id": result["node_id"]})
                    # Try the next statistically valid candidate.
                    ranked = [item for item in self._ranked_accepted_candidates() if item["candidate_id"] != candidate["candidate_id"] and item.get("status") == "accepted"]
                    if ranked:
                        replay_task = self._queue_replay_task(ranked[0])
                        return 0, {"stage": "replay-audit", "replay_ok": False, "next_candidate": ranked[0]["candidate_id"], "replay_task": replay_task}
                    self.store.close_without_promotion("all statistically valid candidates failed replay")
                    return 0, {"stage": "replay-audit", "replay_ok": False, "promoted": False}

                normalized = validated.model_dump(mode="json")
                normalized["worker_duration_ms"] = int(result.get("duration_ms", 0))
                self.db.record_work_report(payload["work_key"], kind.value, result["node_id"], task["id"], normalized)
                rows = self.db.work_reports(payload["work_key"], kind.value)
                expected = int(payload.get("search_candidates", SEARCH_CANDIDATES))
                details: dict[str, Any] = {"stage": "independent-search", "round_id": payload["round_id"], "reports": len(rows), "expected": expected, "delta_hash": validated.delta["sha256"]}
                state = self.store.state()
                active = state.get("active_round") or {}
                if len(rows) >= expected and active.get("status") == "independent-search":
                    grouped: dict[str, list[Any]] = {}
                    for row in rows:
                        value = json.loads(row["output"])
                        grouped.setdefault(value["delta"]["sha256"], []).append(row)
                    candidate_ids: list[str] = []
                    all_tasks: list[str] = []
                    for candidate_id, group in sorted(grouped.items()):
                        chosen = json.loads(group[0]["output"])
                        trainer_reports = sorted((
                            {
                                "node_id": str(row["node_id"]),
                                "task_id": str(row["task_id"]),
                                "duration_ms": int(json.loads(row["output"]).get("worker_duration_ms", 0)),
                            }
                            for row in group
                        ), key=lambda item: (item["node_id"], item["task_id"]))
                        candidate = {
                            "candidate_id": candidate_id, "round_id": payload["round_id"], "base_root": payload["bundle"]["base_root"],
                            "global_validation_hash": payload["global_validation_hash"],
                            "replication_validation_hash": payload["replication_validation_hash"],
                            "bundle": payload["bundle"], "delta": chosen["delta"],
                            "operation": payload["bundle"]["operation"], "category": int(payload["bundle"]["category"]),
                            "trainer_nodes": [item["node_id"] for item in trainer_reports], "trainer_reports": trainer_reports,
                            "source_task_id": str(group[0]["task_id"]), "search_seed": int(chosen["search_seed"]),
                            "active_experts": chosen.get("active_experts", []), "rotation_phase_before": chosen.get("rotation_phase_before"),
                            "rotation_phase_after": chosen.get("rotation_phase_after"), "status": "global-verification", "created_at": time.time(),
                        }
                        self.store.record_candidate(candidate_id, candidate)
                        verification_tasks = self._queue_verification_tasks(candidate, task["id"], int(payload.get("verification_quorum", VERIFICATION_QUORUM)))
                        self.store.update_candidate(candidate_id, verification_tasks=verification_tasks)
                        candidate_ids.append(candidate_id)
                        all_tasks.extend(verification_tasks)
                    self.store.set_round_candidates(candidate_ids)
                    self.db.append_audit("native10_v6_candidates_created", {"round_id": payload["round_id"], "candidate_ids": candidate_ids, "verification_tasks": all_tasks, "search_reports": len(rows)})
                    details.update({"candidate_ids": candidate_ids, "verification_tasks": all_tasks})
                return 0, details

            if kind == TaskKind.DENDRITRON_VERIFICATION:
                validated = DendritronV6VerificationOutput.model_validate(output)
                candidate = self.store.candidate(payload["candidate_id"])
                if not candidate:
                    raise ValueError("unknown Dendritron candidate")
                disallowed = set(candidate.get("trainer_nodes", []))
                if payload.get("mode") == "final-replication":
                    disallowed |= set(candidate.get("verifier_nodes", []))
                    if candidate.get("replay_node"):
                        disallowed.add(candidate["replay_node"])
                if result["node_id"] in disallowed:
                    raise ValueError("verification identity is not independent from candidate generation or selection")
                if validated.delta_hash != candidate["delta"]["sha256"] or validated.base_root != candidate["base_root"]:
                    raise ValueError("Dendritron verification binding mismatch")
                expected_hash = (
                    candidate["replication_validation_hash"]
                    if payload.get("mode") == "final-replication"
                    else candidate["global_validation_hash"]
                )
                if validated.validation_hash != expected_hash:
                    raise ValueError("Dendritron verification used the wrong hidden holdout")
                if validated.write_set != candidate["delta"]["write_set"]:
                    raise ValueError("Dendritron verification write-set mismatch")
                normalized = validated.model_dump(mode="json")
                normalized["worker_duration_ms"] = int(result.get("duration_ms", 0))
                self.db.record_work_report(payload["work_key"], kind.value, result["node_id"], task["id"], normalized)
                rows = self.db.work_reports(payload["work_key"], kind.value)
                quorum = int(payload.get("verification_quorum", VERIFICATION_QUORUM))
                consensus = self._consensus(
                    rows, self._verification_consensus_key, quorum
                )
                is_replication = payload.get("mode") == "final-replication"
                stage = "fresh-final-replication" if is_replication else "blind-selection-verification"
                details: dict[str, Any] = {
                    "stage": stage,
                    "candidate_id": payload["candidate_id"],
                    "reports": len(rows),
                    "quorum": quorum,
                }
                expected_status = "replication-verification" if is_replication else "global-verification"
                if consensus and candidate.get("status") == expected_status:
                    chosen = json.loads(consensus[0]["output"])
                    artifact = self.store.validation_by_hash(expected_hash)
                    acceptable = self._verification_gate(chosen, artifact, candidate)
                    verifier_reports = sorted(
                        (
                            {
                                "node_id": str(row["node_id"]),
                                "task_id": str(row["task_id"]),
                                "duration_ms": int(json.loads(row["output"]).get("worker_duration_ms", 0)),
                            }
                            for row in consensus
                        ),
                        key=lambda item: (item["node_id"], item["task_id"]),
                    )
                    if is_replication:
                        self.store.update_candidate(
                            candidate["candidate_id"],
                            status="replicated" if acceptable else "replication-failed",
                            replication_verification=chosen,
                            replication_verifier_nodes=[item["node_id"] for item in verifier_reports],
                            replication_verifier_reports=verifier_reports,
                            rejection_reason=None if acceptable else "fresh-replication-gate-failed",
                        )
                        self.db.append_audit(
                            "native10_v6_final_replication_evaluated",
                            {
                                "candidate_id": candidate["candidate_id"],
                                "accepted": acceptable,
                                "wins": chosen["wins"],
                                "losses": chosen["losses"],
                                "mcnemar_p_value": chosen["mcnemar_p_value"],
                                "effect_rate": chosen["effect_rate"],
                            },
                        )
                        if acceptable:
                            refreshed = self.store.candidate(candidate["candidate_id"])
                            record = self._promote_after_replication(refreshed)
                            details.update({"accepted": True, "promoted": True, "contribution": record})
                        else:
                            # The final replication artifact is one-shot.  Do not
                            # adaptively test another selected candidate on the
                            # same holdout after observing this failure.
                            self.store.update_candidate(
                                candidate["candidate_id"],
                                status="rejected",
                                rejection_reason="fresh-replication-gate-failed",
                            )
                            self.store.close_without_promotion("fresh-replication-gate-failed")
                            details.update({"accepted": False, "promoted": False, "round_closed": True})
                    else:
                        self.store.update_candidate(
                            candidate["candidate_id"],
                            status="accepted" if acceptable else "rejected",
                            verification=chosen,
                            verifier_nodes=[item["node_id"] for item in verifier_reports],
                            verifier_reports=verifier_reports,
                            rejection_reason=None if acceptable else "selection-statistical-or-harm-gate-failed",
                        )
                        self.db.append_audit(
                            "native10_v6_candidate_evaluated",
                            {
                                "candidate_id": candidate["candidate_id"],
                                "accepted": acceptable,
                                "wins": chosen["wins"],
                                "losses": chosen["losses"],
                                "mcnemar_p_value": chosen["mcnemar_p_value"],
                                "corrected_alpha": chosen["corrected_alpha"],
                                "effect_rate": chosen["effect_rate"],
                            },
                        )
                        details.update({"accepted": acceptable, **self._advance_round_after_verification()})
                return 0, details

            raise ValueError("task is not a Native10 v0.6 task")
