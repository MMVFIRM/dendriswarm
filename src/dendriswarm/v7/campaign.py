from __future__ import annotations

import json
import math
import os
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

from dendriswarm.core.crypto import content_hash
from dendriswarm.v6.benchmark import EVALUATION_FORMAT, compare_with_baseline
from dendriswarm.v6.service import Native10Coordinator
from dendriswarm.v6.validation import GlobalValidationPolicy, make_global_validation_artifact
from dendriswarm.v7.cifar100 import CIFAR100DatasetStore
from dendriswarm.v7.routing import plan_next_round, routing_gap_report

CAMPAIGN_FORMAT = "dendriswarm.cifar100-swarm-campaign.v1"
TEST_REPORT_FORMAT = "dendriswarm.cifar100-final-test-report.v1"


class CIFAR100Campaign:
    """Orchestrate real CIFAR-100 training and routing-gap search.

    The official test split is never used by the planner or promotion gate.
    Each selection/replication fold is consumed once.  Operators must install a
    new trainer-invisible validation bank after the committed fold budget is
    exhausted rather than adaptively reusing observed evidence.
    """

    def __init__(self, state_dir: str | os.PathLike[str], native: Native10Coordinator):
        self.path = Path(state_dir)
        self.path.mkdir(parents=True, exist_ok=True)
        self.dataset = CIFAR100DatasetStore(self.path / "dataset")
        self.state_path = self.path / "campaign.json"
        self.reports_path = self.path / "reports"
        self.reports_path.mkdir(parents=True, exist_ok=True)
        self.native = native
        if not self.state_path.exists():
            self._write({
                "format": CAMPAIGN_FORMAT,
                "dataset_sha256": None,
                "round_index": 0,
                "holdout_per_class": 5,
                "max_rounds": 5,
                "active_native_round": None,
                "history": [],
                "test_evaluations": 0,
                "created_at": time.time(),
            })

    @staticmethod
    def _atomic_json(path: Path, value: dict[str, Any]) -> None:
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

    def _read(self) -> dict[str, Any]:
        value = json.loads(self.state_path.read_text())
        if value.get("format") != CAMPAIGN_FORMAT:
            raise ValueError("unsupported CIFAR-100 campaign state")
        return value

    def _write(self, value: dict[str, Any]) -> None:
        self._atomic_json(self.state_path, value)

    def prepare_dataset(
        self,
        source: str | os.PathLike[str],
        *,
        seed: int = 20260723,
        holdout_per_class: int = 5,
        replace: bool = False,
    ) -> dict[str, Any]:
        if not 2 <= int(holdout_per_class) <= 25:
            raise ValueError("holdout_per_class must be between 2 and 25")
        dataset_status = self.dataset.prepare(source, seed=seed, replace=replace)
        state = self._read()
        state.update({
            "dataset_sha256": dataset_status["sha256"],
            "round_index": 0,
            "holdout_per_class": int(holdout_per_class),
            "max_rounds": 25 // int(holdout_per_class),
            "active_native_round": None,
            "history": [],
            "test_evaluations": 0,
            "prepared_at": time.time(),
        })
        self._write(state)
        return self.status()

    def initialize_model(self, *, seed: int = 7, checkpoint: dict[str, Any] | None = None, replace: bool = False) -> dict[str, Any]:
        if checkpoint is None:
            return self.native.initialize(profile="native10", input_width=3072, seed=seed, replace=replace)
        return self.native.store.import_checkpoint(checkpoint, replace=replace)

    def _fold_rows(self, split: str, round_index: int, per_class: int) -> np.ndarray:
        labels = np.asarray(self.dataset.labels(split), dtype=np.int64)
        rows: list[np.ndarray] = []
        start = round_index * per_class
        stop = start + per_class
        for class_id in range(100):
            class_rows = np.flatnonzero(labels == class_id)
            if stop > len(class_rows):
                raise ValueError("the committed CIFAR-100 validation bank is exhausted")
            rows.append(class_rows[start:stop])
        return np.concatenate(rows).astype(np.int64)

    def _install_round_holdouts(self, round_index: int, search_candidates: int) -> dict[str, Any]:
        state = self._read()
        per_class = int(state["holdout_per_class"])
        selection_rows = self._fold_rows("selection", round_index, per_class)
        replication_rows = self._fold_rows("replication", round_index, per_class)
        selection_x, selection_y = self.dataset.normalized_rows("selection", selection_rows)
        replication_x, replication_y = self.dataset.normalized_rows("replication", replication_rows)
        sample_count = len(selection_y)
        minimum_net = max(3, int(math.ceil(sample_count * 0.008)))
        policy = GlobalValidationPolicy(
            min_samples_per_class=per_class,
            familywise_alpha=0.05,
            max_candidate_evaluations=int(search_candidates),
            max_search_rounds=1,
            min_discordant=max(10, int(math.ceil(sample_count * 0.02))),
            minimum_net_wins=minimum_net,
            minimum_effect_rate=minimum_net / sample_count,
            max_loss_per_class=max(1, int(math.floor(per_class * 0.20))),
            max_loss_rate_per_class=0.20,
        )
        dataset_hash = self.dataset.manifest()["sha256"]
        selection = make_global_validation_artifact(
            self.native.store.model().config, selection_x, selection_y,
            source=f"CIFAR-100:{dataset_hash}:selection-fold:{round_index}",
            split=f"campaign-selection-{round_index}", policy=policy, protocol_fixture_only=False,
        )
        replication = make_global_validation_artifact(
            self.native.store.model().config, replication_x, replication_y,
            source=f"CIFAR-100:{dataset_hash}:replication-fold:{round_index}",
            split=f"campaign-replication-{round_index}", policy=policy, protocol_fixture_only=False,
        )
        self.native.store.set_global_validation(selection, replace=True)
        self.native.store.set_replication_validation(replication, replace=True)
        return {
            "selection_hash": selection["sha256"],
            "replication_hash": replication["sha256"],
            "sample_count_each": sample_count,
            "samples_per_class": per_class,
            "policy": policy.as_dict(),
        }

    def _diagnostic_report(self, *, per_class: int = 10, seed: int = 20260723) -> dict[str, Any]:
        rows = self.dataset.balanced_indices("train", per_class=per_class, seed=seed)
        x, y = self.dataset.normalized_rows("train", rows)
        report = routing_gap_report(
            self.native.store.model(), x, y,
            dataset_sha256=self.dataset.manifest()["sha256"],
            split="campaign-train-diagnostic",
            sample_source=f"balanced-{per_class}-per-class-seed-{seed}",
        )
        self._atomic_json(self.reports_path / f"routing-{report['model_root'][:12]}.json", report)
        return report

    @staticmethod
    def _routing_summary(report: dict[str, Any]) -> dict[str, Any]:
        return {
            "report_sha256": report["sha256"],
            "model_root": report["model_root"],
            "actual_accuracy": float(report["actual_accuracy"]),
            "oracle_category_accuracy": float(report["oracle_category_accuracy"]),
            "oracle_routing_gap": float(report["oracle_routing_gap"]),
            "top4_category_recall": float(report["topk_category_recall"]["4"]),
            "expanded_category_recall": float(report["expanded_category_recall"]),
            "conditional_accuracy_when_routed": float(report["conditional_accuracy_when_routed"]),
            "route_miss_count": int(report["route_miss_count"]),
        }

    def reconcile(self) -> dict[str, Any]:
        state = self._read()
        active_id = state.get("active_native_round")
        if not active_id:
            return state
        native_state = self.native.store.state()
        active = native_state.get("active_round")
        if active and active.get("round_id") == active_id:
            return state
        history = state.get("history", [])
        current = next((item for item in reversed(history) if item.get("native_round_id") == active_id), None)
        if current is not None and current.get("status") == "queued":
            contributions = [
                value for value in native_state.get("contributions", [])
                if value.get("root_before") == current.get("base_root")
            ]
            if contributions:
                contribution = contributions[-1]
                after_report = self._diagnostic_report(seed=20260723 + int(current["campaign_round"]) + 100_000)
                before = dict(current.get("plan", {}).get("routing_snapshot") or {})
                after = self._routing_summary(after_report)
                current.update({
                    "status": "promoted",
                    "root_after": contribution["root_after"],
                    "delta_hash": contribution["delta_hash"],
                    "net_wins": contribution["net_wins"],
                    "selection_evidence": contribution.get("selection_evidence"),
                    "replication_evidence": contribution.get("replication_evidence"),
                    "round_compute": contribution.get("round_compute", {}),
                    "routing_before": before,
                    "routing_after": after,
                    "routing_change": {
                        key: float(after.get(key, 0.0)) - float(before.get(key, 0.0))
                        for key in (
                            "actual_accuracy", "oracle_category_accuracy", "oracle_routing_gap",
                            "top4_category_recall", "expanded_category_recall",
                            "conditional_accuracy_when_routed",
                        )
                    },
                    "completed_at": time.time(),
                })
            else:
                current.update({"status": "closed-without-promotion", "completed_at": time.time()})
        state["round_index"] = int(state.get("round_index", 0)) + 1
        state["active_native_round"] = None
        self._write(state)
        return state

    def plan_next(self, *, search_candidates: int = 8, sample_budget: int = 640) -> dict[str, Any]:
        state = self.reconcile()
        if not self.dataset.prepared() or not self.native.store.initialized():
            raise ValueError("CIFAR-100 dataset and canonical model must be initialized")
        if state.get("active_native_round"):
            raise ValueError("a CIFAR-100 training round is already active")
        if int(state["round_index"]) >= int(state["max_rounds"]):
            raise ValueError("the one-shot CIFAR-100 validation bank is exhausted")
        report = self._diagnostic_report(seed=20260723 + int(state["round_index"]))
        return plan_next_round(
            report,
            round_index=int(state["round_index"]),
            search_candidates=int(search_candidates),
            sample_budget=int(sample_budget),
        )

    def queue_next(
        self,
        *,
        search_candidates: int = 8,
        sample_budget: int = 640,
        optimizer_steps: int = 36,
        learning_rate: float = 0.03,
        verification_quorum: int = 2,
    ) -> dict[str, Any]:
        plan = self.plan_next(search_candidates=search_candidates, sample_budget=sample_budget)
        state = self._read()
        round_index = int(state["round_index"])
        holdouts = self._install_round_holdouts(round_index, search_candidates)
        model = self.native.store.model()
        shard = self.dataset.training_shard(
            model,
            operation=plan["operation"],
            target=int(plan["target"]),
            sample_budget=int(plan["sample_budget"]),
            seed=20260723 + round_index * 1009,
            hard_negative_fraction=float(plan["hard_negative_fraction"]),
        )
        expected_shard_hash = content_hash({key: value for key, value in shard.items() if key != "sha256"})
        if shard.get("sha256") != expected_shard_hash:
            raise ValueError("CIFAR-100 training shard hash mismatch")
        if shard.get("dataset_sha256") != self.dataset.manifest()["sha256"] or shard.get("model_root") != model.root:
            raise ValueError("CIFAR-100 training shard is not bound to the active dataset and model")
        queued = self.native.queue_mutation(
            shard,
            operation=plan["operation"],
            category=int(plan["target"]),
            subset_seed=20260723 + round_index * 1009,
            search_candidates=int(plan["search_candidates"]),
            verification_quorum=int(verification_quorum),
            optimizer_steps=int(optimizer_steps),
            learning_rate=float(learning_rate),
            search_recipes=list(plan["recipes"]),
        )
        entry = {
            "campaign_round": round_index,
            "native_round_id": queued["round_id"],
            "base_root": queued["base_root"],
            "plan": plan,
            "training_shard_sha256": shard["sha256"],
            "holdouts": holdouts,
            "status": "queued",
            "queued_at": time.time(),
        }
        state = self._read()
        state["active_native_round"] = queued["round_id"]
        state.setdefault("history", []).append(entry)
        self._write(state)
        return {**entry, "search_tasks": queued["search_tasks"]}

    def evaluate_test(self, *, source: str = "official-cifar100-test", batch_size: int = 128) -> dict[str, Any]:
        """Evaluate the untouched official test split without feeding results to the planner."""
        model = self.native.store.model()
        correct = 0
        sample_count = 0
        predictions_by_class = np.zeros(model.config.classes, dtype=np.int64)
        correct_by_class = np.zeros(model.config.classes, dtype=np.int64)
        samples_by_class = np.zeros(model.config.classes, dtype=np.int64)
        for x, y in self.dataset.iter_normalized_batches("test", batch_size=batch_size):
            predictions = model.predict(x)
            correct_mask = predictions == y
            correct += int(correct_mask.sum())
            sample_count += int(len(y))
            samples_by_class += np.bincount(y, minlength=model.config.classes)
            predictions_by_class += np.bincount(predictions, minlength=model.config.classes)
            correct_by_class += np.bincount(y[correct_mask], minlength=model.config.classes)
        evaluation: dict[str, Any] = {
            "format": EVALUATION_FORMAT,
            "model_root": model.root,
            "parameter_count": model.parameter_count,
            "dataset": "CIFAR-100",
            "split": "official-test",
            "source": source,
            "data_sha256": content_hash({"dataset_manifest_sha256": self.dataset.manifest()["sha256"], "split": "official-test"}),
            "metric": "accuracy",
            "value": correct / max(1, sample_count),
            "sample_count": sample_count,
            "correct": correct,
            "samples_by_class": samples_by_class.astype(int).tolist(),
            "correct_by_class": correct_by_class.astype(int).tolist(),
            "predictions_by_class": predictions_by_class.astype(int).tolist(),
            "training_performed": False,
            "test_only": True,
        }
        evaluation["sha256"] = content_hash(evaluation)
        manifest = self.dataset.manifest()
        report: dict[str, Any] = {
            "format": TEST_REPORT_FORMAT,
            "dataset_manifest_sha256": manifest["sha256"],
            "model_root": evaluation["model_root"],
            "sample_count": evaluation["sample_count"],
            "correct": evaluation["correct"],
            "accuracy": evaluation["value"],
            "native_label_names": manifest["native_label_names"],
            "native_to_official": manifest["native_to_official"],
            "evaluation": evaluation,
            "used_for_training_or_selection": False,
            "evaluated_at": time.time(),
        }
        report["sha256"] = content_hash(report)
        self._atomic_json(self.reports_path / f"test-{report['model_root'][:12]}.json", report)
        state = self._read()
        state["test_evaluations"] = int(state.get("test_evaluations", 0)) + 1
        state["last_test_report"] = report["sha256"]
        self._write(state)
        return report

    def compare_test_with_baseline(self, test_report: dict[str, Any]) -> dict[str, Any]:
        return compare_with_baseline(test_report["evaluation"], self.native.store.baseline_reference())

    def status(self, *, native_status: dict[str, Any] | None = None) -> dict[str, Any]:
        state = self.reconcile()
        dataset = self.dataset.status()
        native = native_status if native_status is not None else self.native.store.status()
        history = list(state.get("history", []))
        promoted_items = [item for item in history if item.get("status") == "promoted"]
        promoted = len(promoted_items)
        worker_seconds = sum(float(item.get("round_compute", {}).get("contributed_worker_seconds", 0.0)) for item in promoted_items)
        search_candidates = sum(int(item.get("round_compute", {}).get("candidate_searches", 0)) for item in promoted_items)
        latest_routing = promoted_items[-1].get("routing_after") if promoted_items else None
        return {
            "format": CAMPAIGN_FORMAT,
            "dataset": dataset,
            "model": native,
            "round_index": int(state.get("round_index", 0)),
            "max_rounds": int(state.get("max_rounds", 0)),
            "remaining_one_shot_rounds": max(0, int(state.get("max_rounds", 0)) - int(state.get("round_index", 0))),
            "active_native_round": state.get("active_native_round"),
            "rounds_promoted": promoted,
            "rounds_completed": sum(1 for item in history if item.get("status") != "queued"),
            "contributed_worker_seconds": worker_seconds,
            "contributed_worker_hours": worker_seconds / 3600.0,
            "candidate_searches_completed": search_candidates,
            "promotions_per_candidate_search": promoted / max(1, search_candidates),
            "latest_routing": latest_routing,
            "test_evaluations": int(state.get("test_evaluations", 0)),
            "latest_round": history[-1] if history else None,
            "training_data": "official CIFAR-100 fine labels grouped by the official 20 coarse labels",
            "routing_search": True,
            "test_used_for_planning": False,
        }
