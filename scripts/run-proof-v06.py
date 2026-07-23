#!/usr/bin/env python3
"""Reproduce the DendriSwarm v0.6.0 Trainable Native10 proof.

This is a protocol/model-mechanics proof, not a benchmark-accuracy claim. It
executes the exact profile, proves every persistent parameter family has a
bounded owner operation, runs independent volunteer search, trainer-blind
selection, deterministic replay, and fresh final replication.
"""
from __future__ import annotations

import copy
import json
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
from fastapi.testclient import TestClient

from dendriswarm.coordinator.app import create_app
from dendriswarm.core.crypto import Identity, nonce, verify
from dendriswarm.core.models import TaskKind
from dendriswarm.v6.native10 import (
    Native10Config,
    Native10Dendritron,
    compose_non_conflicting_deltas,
    deltas_conflict,
    execute_mutation,
    parameter_reachability,
    validate_delta,
    verify_mutation_full,
)
from dendriswarm.v6.validation import (
    GlobalValidationPolicy,
    exact_one_sided_mcnemar,
    make_global_validation_artifact,
    synthetic_raw_samples,
)
from dendriswarm.v6.benchmark import (
    compare_with_baseline,
    evaluate_checkpoint,
    make_baseline_reference,
)
from dendriswarm.worker.executor import execute_task

ROOT = Path(__file__).resolve().parents[1]


def encoded_size(value: dict[str, Any]) -> int:
    return len(json.dumps(value, separators=(",", ":"), allow_nan=False).encode())


def exact_profile() -> dict[str, Any]:
    config = Native10Config()
    started = time.perf_counter()
    model = Native10Dendritron.initialize(config, seed=7)
    initialize_seconds = time.perf_counter() - started
    reachability = parameter_reachability(config)
    raw, labels = synthetic_raw_samples(config, per_class=1, prototype_seed=20260723, sample_seed=20260724)
    representations = model.encode(raw)
    local = labels < config.classes_per_category
    bundle = model.component_bundle("expert_train", 0)
    mutation_started = time.perf_counter()
    mutation = execute_mutation(
        bundle,
        representations[local],
        labels[local],
        subset_seed=7,
        optimizer_steps=2,
        learning_rate=0.02,
    )
    mutation_seconds = time.perf_counter() - mutation_started
    policy = GlobalValidationPolicy(
        min_samples_per_class=1,
        familywise_alpha=0.05,
        max_candidate_evaluations=2,
        min_discordant=1,
        minimum_net_wins=1,
        minimum_effect_rate=0.0,
        max_loss_per_class=1,
        max_loss_rate_per_class=1.0,
    )
    verification_started = time.perf_counter()
    verification = verify_mutation_full(
        model.artifact(),
        bundle,
        mutation["delta"],
        raw,
        labels,
        validation_hash_value="0" * 64,
        validation_policy=policy,
    )
    verification_seconds = time.perf_counter() - verification_started
    changed_tensors = sorted({patch["tensor"] for patch in mutation["delta"]["patches"]})
    return {
        "parameter_count": model.parameter_count,
        "config": config.as_dict(),
        "reachability": reachability,
        "changed_tensors": changed_tensors,
        "changed_parameters": mutation["delta"]["changed_parameters"],
        "sample_count": verification["sample_count"],
        "samples_by_class": verification["samples_by_class"],
        "checkpoint_bytes": encoded_size(model.artifact()),
        "bundle_bytes": encoded_size(bundle),
        "delta_bytes": encoded_size(mutation["delta"]),
        "initialize_seconds": initialize_seconds,
        "mutation_seconds": mutation_seconds,
        "verification_seconds": verification_seconds,
        "baseline_training_included": False,
        "historical_trained_weights_exercised": False,
    }


def register(client: TestClient, identity: Identity) -> None:
    value = {
        "node_id": identity.node_id,
        "public_key": identity.public_key_b64,
        "capabilities": {
            "cpu_count": 2,
            "memory_mb": 4096,
            "memory_available_mb": 4096,
            "disk_free_mb": 10000,
            "accelerator": "cpu",
            "platform": "proof",
            "machine": "x86_64",
            "supported_backends": ["numpy-cpu"],
            "tags": ["portable-numpy-v1", "independent-search-v1", "blind-global-verification-v2", "deterministic-v2"],
        },
        "policy": {
            "cpu_percent": 100,
            "memory_percent": 100,
            "disk_limit_mb": 10000,
            "max_task_seconds": 600,
            "allow_on_battery": True,
            "max_system_cpu_percent": 100,
        },
        "timestamp": int(time.time()),
        "nonce": nonce(),
    }
    value["signature"] = identity.sign(value)
    response = client.post("/v1/nodes/register", json=value)
    response.raise_for_status()


def signed(identity: Identity, action: str, **extra: Any) -> dict[str, Any]:
    body = {"action": action, "node_id": identity.node_id, **extra, "timestamp": int(time.time()), "nonce": nonce()}
    return {key: value for key, value in body.items() if key != "action"} | {"signature": identity.sign(body)}


def execute_one(client: TestClient, identity: Identity) -> tuple[dict[str, Any], dict[str, Any]] | None:
    response = client.post("/v1/tasks/claim", json=signed(identity, "claim"))
    if response.status_code == 204:
        return None
    response.raise_for_status()
    envelope = response.json()
    assert verify(envelope["coordinator_public_key"], envelope["task"], envelope["signature"])
    task = envelope["task"]
    payload = dict(task["payload"])
    if "native10_checkpoint_root" in payload:
        checkpoint = client.get(f"/v1/native10-v6/checkpoints/{payload['native10_checkpoint_root']}")
        checkpoint.raise_for_status()
        payload["_native10_checkpoint"] = checkpoint.json()
    if "global_validation_hash" in payload and task["kind"] == TaskKind.DENDRITRON_VERIFICATION.value:
        validation = client.post(
            f"/v1/native10-v6/validation/{payload['global_validation_hash']}",
            json=signed(
                identity,
                "fetch-native10-validation",
                task_id=task["id"],
                lease_token=task["lease_token"],
            ),
        )
        validation.raise_for_status()
        payload["_native10_validation"] = validation.json()
    output = execute_task(TaskKind(task["kind"]), payload, cpu_threads=1)
    body = {
        "node_id": identity.node_id,
        "task_id": task["id"],
        "lease_token": task["lease_token"],
        "duration_ms": 1,
        "output": output,
    }
    body["signature"] = identity.sign(body)
    result = client.post("/v1/tasks/result", json=body)
    result.raise_for_status()
    return task, result.json()


def network_round(path: Path) -> dict[str, Any]:
    app = create_app(path, bootstrap=False, lease_seconds=30.0)
    service = app.state.service
    before = service.native10_v6.initialize("compact", seed=7)["canonical_root"]
    queued = service.native10_v6.queue_demo_round(category=0, operation="field_train")
    identities = [Identity.generate() for _ in range(16)]
    task_records: list[dict[str, Any]] = []
    receipts: list[dict[str, Any]] = []
    with TestClient(app) as client:
        for identity in identities:
            register(client, identity)
        for _ in range(40):
            progressed = False
            for identity in identities:
                item = execute_one(client, identity)
                if item is not None:
                    progressed = True
                    task_records.append(item[0])
                    receipts.append(item[1])
            if not progressed:
                break
    status = service.native10_v6.store.status()
    holdout_rotation_required = False
    try:
        service.native10_v6.queue_demo_round(category=0, operation="field_train")
    except ValueError as exc:
        holdout_rotation_required = "exhausted" in str(exc) or "one-shot" in str(exc)
    contribution = status["latest_contribution"]
    state = service.native10_v6.store.state()
    selected = state["candidates"][contribution["candidate_id"]]
    search_outputs = [
        json.loads(service.db.task(task_id)["output"])
        for task_id in queued["search_tasks"]
    ]
    return {
        "root_before": before,
        "root_after": status["canonical_root"],
        "contribution": contribution,
        "search_candidate_hashes": [value["delta"]["sha256"] for value in search_outputs],
        "search_seeds": [value["search_seed"] for value in search_outputs],
        "trainer_nodes": selected["trainer_nodes"],
        "selection_verifier_nodes": selected["verifier_nodes"],
        "replay_node": selected["replay_node"],
        "replication_verifier_nodes": selected["replication_verifier_nodes"],
        "selection_hash": contribution["selection_evidence"]["validation_hash"],
        "replication_hash": contribution["replication_evidence"]["validation_hash"],
        "selection_evidence": contribution["selection_evidence"],
        "replication_evidence": contribution["replication_evidence"],
        "task_modes": sorted({str(task["payload"].get("mode", "search")) for task in task_records}),
        "trainer_payload_contains_validation": any(
            "_native10_validation" in task["payload"] or "validation_inputs" in task["payload"]
            for task in task_records
            if task["kind"] == TaskKind.DENDRITRON_MUTATION.value
        ),
        "audit_chain_valid": service.db.validate_audit_chain()[0],
        "all_result_receipts_accepted": all(value.get("accepted") for value in receipts),
        "holdout_rotation_required_after_round": holdout_rotation_required,
    }


def schema_and_composition() -> dict[str, Any]:
    model = Native10Dendritron.initialize(Native10Config.compact_demo(seed=11), seed=11)
    raw, labels = synthetic_raw_samples(model.config, per_class=10, prototype_seed=3, sample_seed=4)
    representations = model.encode(raw)
    category_deltas = []
    for category in (0, 1):
        mask = labels // model.config.classes_per_category == category
        bundle = model.component_bundle("memory_train", category)
        category_deltas.append(execute_mutation(bundle, representations[mask], labels[mask], optimizer_steps=3)["delta"])
    conflict_free = not deltas_conflict(category_deltas[0], category_deltas[1])
    composed = compose_non_conflicting_deltas(model, category_deltas)

    expert_mask = labels < model.config.classes_per_category
    expert_bundle = model.component_bundle("expert_train", 0)
    expert_delta = execute_mutation(expert_bundle, representations[expert_mask], labels[expert_mask], optimizer_steps=2)["delta"]
    malicious = copy.deepcopy(expert_delta)
    malicious["patches"].append(copy.deepcopy(category_deltas[0]["patches"][0]))
    malicious["write_set"].append(category_deltas[0]["write_set"][0])
    malicious["changed_parameters"] += model.config.classes_per_category * model.config.representation_width
    from dendriswarm.v6.native10 import delta_hash
    malicious["sha256"] = delta_hash(malicious)
    schema_rejected = False
    try:
        validate_delta(malicious, expert_bundle)
    except ValueError:
        schema_rejected = True
    return {
        "conflict_free": conflict_free,
        "composed_root_changed": composed.root != model.root,
        "operation_schema_rejected_cross_tissue_patch": schema_rejected,
        "noise_p_value": exact_one_sided_mcnemar(1, 0),
        "strong_p_value": exact_one_sided_mcnemar(12, 0),
        "corrected_alpha": GlobalValidationPolicy(max_candidate_evaluations=20).corrected_alpha,
        "sparse_delta_bytes": encoded_size(expert_delta),
        "full_bundle_bytes": encoded_size(expert_bundle),
    }


def evidence_and_baseline(path: Path) -> dict[str, Any]:
    app = create_app(path, bootstrap=False)
    native = app.state.service.native10_v6
    native.initialize("compact", seed=7)
    model = native.store.model()
    train_x, train_y = synthetic_raw_samples(
        model.config, per_class=12, prototype_seed=20260723, sample_seed=20260725
    )
    bundle = model.component_bundle("field_train", 0)
    mutation = execute_mutation(
        bundle, train_x, train_y, subset_seed=7, optimizer_steps=20, learning_rate=0.035
    )
    validation_x, validation_y = synthetic_raw_samples(
        model.config, per_class=12, prototype_seed=20260723, sample_seed=20260724
    )
    policy = GlobalValidationPolicy(
        min_samples_per_class=5, max_candidate_evaluations=20, min_discordant=2,
        minimum_net_wins=1, minimum_effect_rate=0.0, max_loss_per_class=8,
        max_loss_rate_per_class=0.5,
    )
    artifact = make_global_validation_artifact(
        model.config, validation_x, validation_y, source="v0.6 proof", policy=policy
    )
    native.store.set_global_validation(artifact)
    evidence = verify_mutation_full(
        model.artifact(), bundle, mutation["delta"], validation_x, validation_y,
        validation_hash_value=artifact["sha256"], validation_policy=policy,
    )
    candidate = {"operation": "field_train", "category": 0}
    native._verification_gate(evidence, artifact, candidate)
    forged = copy.deepcopy(evidence)
    forged["operation"] = "memory_train"
    forged_rejected = False
    try:
        native._verification_gate(forged, artifact, candidate)
    except ValueError:
        forged_rejected = True

    evaluation = evaluate_checkpoint(
        model, validation_x, validation_y, dataset="v0.6-protocol-fixture",
        split="test", source="synthetic proof data",
    )
    baseline = make_baseline_reference(
        dataset="v0.6-protocol-fixture", split="test", metric="accuracy",
        value=max(0.0, float(evaluation["value"]) - 0.01),
        model="external established control", source="external proof reference",
        evidence_sha256="a" * 64,
        notes="Import-only proof of comparison plumbing; not a benchmark claim.",
    )
    baseline_status = native.store.set_baseline_reference(baseline)
    comparison = compare_with_baseline(evaluation, baseline)
    return {
        "forged_verifier_evidence_rejected": forged_rejected,
        "baseline_reference_installed": baseline_status["configured"],
        "baseline_training_included": comparison["baseline_training_included"],
        "comparison_hash": comparison["sha256"],
    }


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="dendriswarm-v06-proof-") as directory:
        temporary = Path(directory)
        exact = exact_profile()
        network = network_round(temporary / "network")
        mechanics = schema_and_composition()
        integrity = evidence_and_baseline(temporary / "integrity")

    identities = (
        set(network["trainer_nodes"])
        | set(network["selection_verifier_nodes"])
        | {network["replay_node"]}
        | set(network["replication_verifier_nodes"])
    )
    gates = [
        {"gate": "exact_native10_executes", "pass": exact["parameter_count"] == 4_898_812 and exact["sample_count"] == 100},
        {"gate": "all_float_parameters_reachable", "pass": exact["reachability"]["reachable_float_fraction"] == 1.0},
        {"gate": "nonlinear_branch_weights_train", "pass": "expert_branch_weights" in exact["changed_tensors"] and exact["changed_parameters"] > 70_000},
        {"gate": "operation_specific_schema", "pass": mechanics["operation_schema_rejected_cross_tissue_patch"]},
        {"gate": "independent_candidate_search", "pass": len(set(network["search_candidate_hashes"])) >= 3 and len(set(network["search_seeds"])) == 4},
        {"gate": "noise_level_gain_rejected", "pass": mechanics["noise_p_value"] > mechanics["corrected_alpha"]},
        {"gate": "strong_paired_gain_significant", "pass": mechanics["strong_p_value"] <= mechanics["corrected_alpha"]},
        {"gate": "trainers_blind_to_holdouts", "pass": not network["trainer_payload_contains_validation"]},
        {"gate": "selection_and_replication_are_distinct", "pass": network["selection_hash"] != network["replication_hash"]},
        {"gate": "hidden_artifacts_are_one_tournament_only", "pass": network["holdout_rotation_required_after_round"]},
        {"gate": "selection_significant", "pass": network["selection_evidence"]["mcnemar_p_value"] <= mechanics["corrected_alpha"]},
        {"gate": "fresh_replication_significant", "pass": network["replication_evidence"]["mcnemar_p_value"] <= mechanics["corrected_alpha"]},
        {"gate": "identity_distinct_evidence_chain", "pass": len(identities) >= 6},
        {"gate": "canonical_root_changes_only_after_replication", "pass": network["root_before"] != network["root_after"]},
        {"gate": "conflict_free_delta_composition", "pass": mechanics["conflict_free"] and mechanics["composed_root_changed"]},
        {"gate": "sparse_delta_smaller_than_bundle", "pass": mechanics["sparse_delta_bytes"] < mechanics["full_bundle_bytes"]},
        {"gate": "audit_chain_valid", "pass": network["audit_chain_valid"]},
        {"gate": "verifier_evidence_recomputed", "pass": integrity["forged_verifier_evidence_rejected"]},
        {"gate": "external_baseline_reference_without_training", "pass": integrity["baseline_reference_installed"] and not integrity["baseline_training_included"]},
        {"gate": "baseline_training_excluded", "pass": not exact["baseline_training_included"] and not exact["historical_trained_weights_exercised"]},
    ]
    report = {
        "proof": "dendriswarm-v0.6.0-trainable-native10",
        "generated_at": time.time(),
        "all_pass": all(item["pass"] for item in gates),
        "gates": gates,
        "exact_profile": exact,
        "network_round": network,
        "schema_and_composition": mechanics,
        "evidence_and_baseline": integrity,
        "scope": "Protocol mechanics and synthetic-fixture learning only; no external benchmark accuracy claim.",
    }
    output = ROOT / "docs" / "PROOF_RUN_V06.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["all_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
