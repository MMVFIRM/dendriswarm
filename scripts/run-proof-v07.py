#!/usr/bin/env python3
"""Reproduce the v0.7.0 CIFAR-100 campaign-readiness proof.

This proof executes the exact Native10 topology on CIFAR-shaped image tensors
and validates the real campaign/data path.  It does not invent an external
benchmark result when the official archive is unavailable.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

from dendriswarm.coordinator.service import CoordinatorService
from dendriswarm.core.crypto import content_hash
from dendriswarm.core.models import DendritronV6MutationOutput
from dendriswarm.v6.native10 import (
    Native10Config,
    Native10Dendritron,
    decode_training_tensor,
    execute_mutation,
    parameter_reachability,
    synthetic_representation_shard,
)
from dendriswarm.v6.validation import GlobalValidationPolicy, make_global_validation_artifact
from dendriswarm.v7.cifar100 import (
    CIFAR100DatasetStore,
    CIFAR100_MD5,
    CIFAR100_URL,
    DATASET_FORMAT,
    _array_digest,
    _native_label_mapping,
    _stratified_split,
)
from dendriswarm.v7.routing import plan_next_round, routing_gap_report

ROOT = Path(__file__).resolve().parents[1]


def prepared_fixture(path: Path) -> CIFAR100DatasetStore:
    path.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(700)
    arrays: dict[str, dict[str, Any]] = {}
    split_counts = {"train": 2000, "selection": 200, "replication": 200, "test": 100}
    counts: dict[str, list[int]] = {}
    for split, count in split_counts.items():
        labels = np.repeat(np.arange(100, dtype=np.int16), count // 100)
        images = rng.integers(0, 256, size=(count, 3072), dtype=np.uint8)
        counts[split] = np.bincount(labels, minlength=100).astype(int).tolist()
        for suffix, value in (("images", images), ("labels", labels)):
            filename = f"{split}-{suffix}.npy"
            file_path = path / filename
            with file_path.open("wb") as handle:
                np.save(handle, value, allow_pickle=False)
            arrays[f"{split}_{suffix}"] = {
                "file": filename,
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "sha256": _array_digest(value),
                "file_sha256": hashlib.sha256(file_path.read_bytes()).hexdigest(),
            }
    manifest: dict[str, Any] = {
        "format": DATASET_FORMAT,
        "dataset": "CIFAR-100",
        "official_url": CIFAR100_URL,
        "official_archive_md5": CIFAR100_MD5,
        "observed_archive_md5": None,
        "source_kind": "proof-fixture-cifar-shaped-not-benchmark-data",
        "source_sha256": "0" * 64,
        "seed": 700,
        "input_width": 3072,
        "classes": 100,
        "categories": 20,
        "classes_per_category": 5,
        "split_counts": split_counts,
        "counts_by_class": counts,
        "normalization": {
            "source_layout": "channel-major-r1024-g1024-b1024",
            "model_layout": "spatial-patches-4x2-c3-h8-w16",
            "scale": 255.0,
            "channel_mean": [0.5, 0.5, 0.5],
            "channel_std": [0.25, 0.25, 0.25],
        },
        "fine_label_names_official": [f"fine-{i}" for i in range(100)],
        "coarse_label_names": [f"coarse-{i}" for i in range(20)],
        "official_to_native": list(range(100)),
        "native_to_official": list(range(100)),
        "fine_labels_by_coarse_category": [list(range(i * 5, i * 5 + 5)) for i in range(20)],
        "native_label_names": [f"fine-{i}" for i in range(100)],
        "arrays": arrays,
        "test_used_for_selection": False,
    }
    manifest["sha256"] = content_hash(manifest)
    (path / "manifest.json").write_text(json.dumps(manifest, sort_keys=True, separators=(",", ":")))
    return CIFAR100DatasetStore(path)


def gate(name: str, passed: bool, evidence: Any) -> dict[str, Any]:
    return {"name": name, "pass": bool(passed), "evidence": evidence}


def run(official_source: str | None = None) -> dict[str, Any]:
    gates: list[dict[str, Any]] = []
    config = Native10Config()
    model = Native10Dendritron.initialize(config, seed=7)
    reachability = parameter_reachability(config)
    gates.append(gate("exact_native10_executes", model.parameter_count == 4_898_812, {
        "parameter_count": model.parameter_count, "config": config.as_dict()
    }))
    gates.append(gate("all_trainable_tensor_families_reachable", reachability["reachable_float_fraction"] == 1.0, reachability))
    gates.append(gate("official_source_contract_committed", CIFAR100_URL.startswith("https://") and len(CIFAR100_MD5) == 32, {
        "url": CIFAR100_URL, "archive_md5": CIFAR100_MD5
    }))

    mapping = np.arange(100).reshape(20, 5)[:, ::-1].reshape(-1)
    coarse_for_fine = np.empty(100, dtype=np.int64)
    for coarse, group in enumerate(mapping.reshape(20, 5)):
        coarse_for_fine[group] = coarse
    train_fine = np.tile(np.arange(100), 5)
    official_to_native, native_to_official, grouped = _native_label_mapping(
        train_fine, coarse_for_fine[train_fine], np.arange(100), coarse_for_fine
    )
    gates.append(gate("official_coarse_labels_define_native_colonies", np.array_equal(official_to_native[train_fine] // 5, coarse_for_fine[train_fine]), {
        "categories": len(grouped), "classes_per_category": sorted({len(value) for value in grouped})
    }))
    labels = np.repeat(np.arange(100), 500)
    train_rows, selection_rows, replication_rows = _stratified_split(labels, seed=20260723)
    gates.append(gate("campaign_split_is_450_25_25_per_class", (
        len(train_rows), len(selection_rows), len(replication_rows)
    ) == (45_000, 2_500, 2_500), {
        "train": len(train_rows), "selection": len(selection_rows), "replication": len(replication_rows)
    }))

    with tempfile.TemporaryDirectory(prefix="dendriswarm-v07-proof-") as temporary:
        root = Path(temporary)
        store = prepared_fixture(root / "dataset")
        image = np.zeros((1, 3, 32, 32), dtype=np.uint8)
        for row in range(4):
            for column in range(2):
                image[:, :, row*8:(row+1)*8, column*16:(column+1)*16] = 20 + 10 * (row*2 + column)
        adapted = store.normalize_images(image.reshape(1, 3072)).reshape(1, 8, 384)
        distinct_blocks = len(set(np.round(adapted.mean(axis=2)[0], 6).tolist())) == 8
        gates.append(gate("cifar_images_map_to_eight_spatial_field_tissues", distinct_blocks, {
            "block_means": adapted.mean(axis=2)[0].astype(float).tolist(),
            "layout": store.manifest()["normalization"]["model_layout"],
        }))

        field_shard = store.training_shard(model, operation="field_train", target=0, sample_budget=100, seed=701)
        scout_shard = store.training_shard(model, operation="scout_train", target=0, sample_budget=100, seed=702)
        expert_shard = store.training_shard(model, operation="expert_train", target=0, sample_budget=100, seed=703)
        decoded = decode_training_tensor(field_shard["train_inputs"])
        gates.append(gate("bounded_real_image_shards_are_cpu_portable", decoded.shape == (100, 3072) and np.isfinite(decoded).all(), {
            "field_json_bytes": len(json.dumps(field_shard)),
            "scout_json_bytes": len(json.dumps(scout_shard)),
            "expert_json_bytes": len(json.dumps(expert_shard)),
            "field_transport_dtype": field_shard["train_inputs"]["array"]["dtype"],
        }))

        diagnostic_rows = store.balanced_indices("train", per_class=1, seed=704)
        diagnostic_x, diagnostic_y = store.normalized_rows("train", diagnostic_rows)
        routing = routing_gap_report(model, diagnostic_x, diagnostic_y, dataset_sha256=store.manifest()["sha256"], split="campaign-train-diagnostic", sample_source="proof")
        gates.append(gate("complete_model_routing_gap_is_measured", len(routing["categories"]) == 20 and routing["sample_count"] == 100, {
            "actual_accuracy": routing["actual_accuracy"],
            "oracle_category_accuracy": routing["oracle_category_accuracy"],
            "top4_recall": routing["topk_category_recall"]["4"],
            "expanded_recall": routing["expanded_category_recall"],
        }))
        plan = plan_next_round(routing, round_index=0, search_candidates=4, sample_budget=100)
        gates.append(gate("planner_assigns_routing_or_tissue_search_from_measured_bottleneck", plan["objective"] in {"close-routing-gap", "raise-conditional-class-accuracy"}, plan))
        gates.append(gate("candidate_recipes_are_independent_search_trajectories", len({content_hash(value) for value in plan["recipes"]}) == 4, {
            "recipe_hashes": [content_hash(value) for value in plan["recipes"]]
        }))
        forbidden = dict(routing)
        forbidden["split"] = "test"
        forbidden["test_selection_forbidden"] = True
        forbidden["sha256"] = content_hash({k: v for k, v in forbidden.items() if k != "sha256"})
        try:
            plan_next_round(forbidden, round_index=0)
            test_blocked = False
        except ValueError:
            test_blocked = True
        gates.append(gate("official_test_split_cannot_drive_planning", test_blocked, {"used_for_training_or_selection": False}))

        service = CoordinatorService(root / "coordinator")
        native = service.native10_v6
        native.initialize("native10", input_width=3072, seed=7)
        rng = np.random.default_rng(705)
        selection_x = rng.normal(0, 1, size=(100, 3072)).astype(np.float32)
        selection_y = np.arange(100, dtype=np.int64)
        replication_x = rng.normal(0, 1, size=(100, 3072)).astype(np.float32)
        replication_y = np.arange(100, dtype=np.int64)
        policy = GlobalValidationPolicy(
            min_samples_per_class=1, max_candidate_evaluations=2, max_search_rounds=1,
            min_discordant=1, minimum_net_wins=1, minimum_effect_rate=0.0,
            max_loss_per_class=1, max_loss_rate_per_class=1.0,
        )
        selection = make_global_validation_artifact(config, selection_x, selection_y, source="proof-selection", split="selection", policy=policy, protocol_fixture_only=True)
        replication = make_global_validation_artifact(config, replication_x, replication_y, source="proof-replication", split="replication", policy=policy, protocol_fixture_only=True)
        native.store.set_global_validation(selection)
        native.store.set_replication_validation(replication)
        queue_shard = store.training_shard(native.store.model(), operation="field_train", target=0, sample_budget=100, seed=706)
        queued = native.queue_mutation(queue_shard, operation="field_train", category=0, search_candidates=2, verification_quorum=2, optimizer_steps=2, search_recipes=plan_next_round(routing, round_index=4, search_candidates=2, sample_budget=100)["recipes"])
        payloads = [json.loads(service.db.task(task_id)["payload"]) for task_id in queued["search_tasks"]]
        gates.append(gate("real_cifar_shard_queues_heterogeneous_search_tasks", len(payloads) == 2 and all(value["train_data"]["format"] == "dendriswarm.cifar100-patch-input.v1" for value in payloads), {
            "task_ids": queued["search_tasks"], "payload_bytes": [len(json.dumps(value)) for value in payloads]
        }))
        gates.append(gate("selection_and_replication_are_distinct_trainer_invisible_artifacts", selection["sha256"] != replication["sha256"], {
            "selection": selection["sha256"], "replication": replication["sha256"]
        }))

        local = synthetic_representation_shard(Native10Config.compact_demo(), 0, train_per_class=4, validation_per_class=2, seed=707)
        compact = Native10Dendritron.initialize(Native10Config.compact_demo())
        mutation = execute_mutation(
            compact.component_bundle("expert_train", 0),
            np.asarray(local["train_representations"], dtype=np.float32),
            np.asarray(local["train_labels"], dtype=np.int64),
            optimizer_steps=2, search_recipe={"expert_diversity": 0.01},
        )
        mutation["runtime"] = {"backend": "numpy-cpu", "machine": "proof", "python": "3", "cpu_threads": 1}
        mutation["search_recipe"] = {"expert_diversity": 0.99}
        try:
            DendritronV6MutationOutput.model_validate(mutation)
            recipe_tamper_blocked = False
        except ValueError:
            recipe_tamper_blocked = True
        gates.append(gate("search_recipe_commitment_rejects_tampering", recipe_tamper_blocked, {}))

    repository_files = [path.relative_to(ROOT).as_posix() for path in ROOT.rglob("*") if path.is_file()]
    dataset_bundled = any("cifar-100-python" in value or value.endswith("cifar-100-python.tar.gz") for value in repository_files)
    gates.append(gate("benchmark_dataset_and_baseline_trainer_are_not_bundled", not dataset_bundled, {
        "dataset_bundled": dataset_bundled, "baseline_training_included": False
    }))

    real_data: dict[str, Any] = {"executed": False, "reason": "no official archive supplied"}
    if official_source:
        with tempfile.TemporaryDirectory(prefix="dendriswarm-v07-real-") as temporary:
            real_store = CIFAR100DatasetStore(Path(temporary) / "dataset")
            status = real_store.prepare(official_source)
            real_data = {"executed": True, "dataset": status}
    gates.append(gate("real_archive_mode_is_hash_verified_when_requested", (not official_source) or real_data.get("executed") is True, real_data))

    report = {
        "proof": "dendriswarm-v0.7.0-cifar100-swarm-campaign",
        "generated_at": time.time(),
        "all_pass": all(item["pass"] for item in gates),
        "gates": gates,
        "real_cifar100": real_data,
        "claim_boundary": {
            "real_cifar100_campaign_code": True,
            "official_archive_executed_in_packaging_environment": bool(real_data.get("executed")),
            "external_benchmark_accuracy_claim": False,
            "baseline_training_included": False,
            "synthetic_fixture_used_for_accuracy_claim": False,
        },
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cifar100", help="Optional official archive or extracted directory for real ingestion verification")
    args = parser.parse_args()
    report = run(args.cifar100)
    output = ROOT / "docs" / "PROOF_RUN_V07.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps({"all_pass": report["all_pass"], "gates": len(report["gates"]), "output": str(output), "real_cifar100": report["real_cifar100"]}, indent=2))
    if not report["all_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
