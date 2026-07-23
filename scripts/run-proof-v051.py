#!/usr/bin/env python3
"""Reproduce the v0.5.1 Native10 topology contribution proof.

The proof separates trainer-visible category tissue from a coordinator-held,
all-class promotion holdout. It executes the exact 4,898,712-parameter profile,
performs a tensor-archive import round trip, demonstrates a valid compact
promotion, demonstrates rejection of a locally positive but globally harmful
candidate, and exercises the database-enforced one-active-lease invariant.

No baseline trainer or baseline benchmark is included.
"""
from __future__ import annotations

import json
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
from fastapi.testclient import TestClient

from dendriswarm.coordinator.app import create_app
from dendriswarm.coordinator.db import Database
from dendriswarm.core.crypto import Identity, nonce, verify
from dendriswarm.core.limits import MAX_CONTROL_REQUEST_BYTES, MAX_HTTP_BODY_BYTES, MAX_RESULT_OUTPUT_BYTES
from dendriswarm.core.models import NodeCapabilities, SeedPolicy, TaskKind
from dendriswarm.v5.native10 import (
    Native10Config,
    Native10Dendritron,
    execute_mutation,
    load_external_checkpoint,
    synthetic_representation_shard,
    verify_mutation_full,
)
from dendriswarm.v5.validation import (
    GlobalValidationPolicy,
    decode_global_validation_artifact,
    synthetic_global_validation_fixture,
)
from dendriswarm.worker.executor import execute_task

ROOT = Path(__file__).resolve().parents[1]


def encoded_size(value: dict[str, Any]) -> int:
    return len(json.dumps(value, separators=(",", ":"), allow_nan=False).encode("utf-8"))


def register(client: TestClient, identity: Identity) -> None:
    value = {
        "node_id": identity.node_id,
        "public_key": identity.public_key_b64,
        "capabilities": {
            "cpu_count": 2,
            "memory_mb": 2048,
            "memory_available_mb": 1536,
            "disk_free_mb": 8192,
            "accelerator": "cpu",
            "accelerators": ["cpu"],
            "platform": "proof",
            "machine": "x86_64",
            "python_version": "proof",
            "supported_backends": ["numpy-cpu"],
            "tags": ["portable-numpy-v1", "deterministic-v2"],
        },
        "timestamp": int(time.time()),
        "nonce": nonce(),
    }
    value["signature"] = identity.sign(value)
    response = client.post("/v1/nodes/register", json=value)
    response.raise_for_status()


def signed_action(identity: Identity, action: str, **extra: object) -> dict[str, object]:
    body: dict[str, object] = {
        "action": action,
        "node_id": identity.node_id,
        **extra,
        "timestamp": int(time.time()),
        "nonce": nonce(),
    }
    return {key: value for key, value in body.items() if key != "action"} | {
        "signature": identity.sign(body)
    }


def materialize_private_task(
    client: TestClient, identity: Identity, task: dict[str, Any]
) -> dict[str, Any]:
    payload = dict(task["payload"])
    checkpoint_root = payload.get("native10_checkpoint_root")
    if checkpoint_root:
        checkpoint = client.get(f"/v1/native10/checkpoints/{checkpoint_root}")
        checkpoint.raise_for_status()
        payload["_native10_checkpoint"] = checkpoint.json()
    validation_hash = payload.get("global_validation_hash")
    if validation_hash and task["kind"] == TaskKind.DENDRITRON_VERIFICATION.value:
        response = client.post(
            f"/v1/native10/validation/{validation_hash}",
            json=signed_action(
                identity,
                "fetch-native10-validation",
                task_id=task["id"],
                lease_token=task["lease_token"],
            ),
        )
        response.raise_for_status()
        payload["_native10_validation"] = response.json()
    return payload


def claim_execute_submit(
    client: TestClient, identity: Identity
) -> tuple[dict[str, Any], dict[str, Any]]:
    claimed = client.post("/v1/tasks/claim", json=signed_action(identity, "claim"))
    claimed.raise_for_status()
    envelope = claimed.json()
    assert verify(envelope["coordinator_public_key"], envelope["task"], envelope["signature"])
    task = envelope["task"]
    payload = materialize_private_task(client, identity, task)
    output = execute_task(TaskKind(task["kind"]), payload, cpu_threads=1)
    body = {
        "node_id": identity.node_id,
        "task_id": task["id"],
        "lease_token": task["lease_token"],
        "duration_ms": 1,
        "output": output,
    }
    body["signature"] = identity.sign(body)
    response = client.post("/v1/tasks/result", json=body)
    response.raise_for_status()
    return task, response.json()


def exact_profile_evidence(temporary: Path) -> dict[str, Any]:
    config = Native10Config()
    started = time.perf_counter()
    model = Native10Dendritron.initialize(config)
    initialize_seconds = time.perf_counter() - started

    archive = temporary / "native10-exact-profile.npz"
    np.savez(archive, **model.tensors)
    imported = load_external_checkpoint(archive, config=config)

    shard = synthetic_representation_shard(
        config, 0, train_per_class=2, validation_per_class=1
    )
    bundle = imported.component_bundle("expert_refit", 0)
    mutation_started = time.perf_counter()
    mutation = execute_mutation(
        bundle,
        np.asarray(shard["train_representations"], dtype=np.float32),
        np.asarray(shard["train_labels"], dtype=np.int64),
        np.asarray(shard["train_representations"], dtype=np.float32),
        np.asarray(shard["train_labels"], dtype=np.int64),
    )
    mutation_seconds = time.perf_counter() - mutation_started

    validation = synthetic_global_validation_fixture(
        config,
        per_class=1,
        policy=GlobalValidationPolicy(
            min_samples_per_class=1,
            max_loss_per_class=1,
            max_loss_rate_per_class=1.0,
            max_candidate_evaluations=2,
        ),
    )
    x_global, y_global, _ = decode_global_validation_artifact(
        validation, expected_config=config
    )
    verification_started = time.perf_counter()
    verification = verify_mutation_full(
        imported.artifact(),
        bundle,
        mutation["delta"],
        x_global,
        y_global,
        validation_hash_value=validation["sha256"],
    )
    verification_seconds = time.perf_counter() - verification_started

    checkpoint = imported.artifact()
    return {
        "parameter_count": imported.parameter_count,
        "config": config.as_dict(),
        "initialized_root": model.root,
        "imported_root": imported.root,
        "archive_import_root_equal": imported.root == model.root,
        "checkpoint_bytes": encoded_size(checkpoint),
        "trainer_bundle_bytes": encoded_size(bundle),
        "delta_bytes": encoded_size(mutation["delta"]),
        "validation_bytes": encoded_size(validation),
        "validation_samples": verification["sample_count"],
        "samples_by_class": verification["samples_by_class"],
        "class_metric_count": len(verification["losses_by_class"]),
        "composed_root_changed": imported.apply_delta(mutation["delta"]).root != imported.root,
        "initialize_seconds": initialize_seconds,
        "mutation_seconds": mutation_seconds,
        "full_global_verification_seconds": verification_seconds,
        "baseline_training_included": False,
        "historical_trained_weights_exercised": False,
    }


def successful_compact_round(temporary: Path) -> dict[str, Any]:
    app = create_app(temporary / "positive", bootstrap=False, lease_seconds=10.0)
    with TestClient(app) as client:
        service = app.state.service
        initialized = service.native10.initialize("compact", seed=7)
        queued = service.native10.queue_demo_round(category=0, operation="expert_refit")
        identities = [Identity.generate() for _ in range(4)]
        for identity in identities:
            register(client, identity)
        receipts = [claim_execute_submit(client, identity) for identity in identities]
        status = service.native10.store.status()
        contribution = status["latest_contribution"]
        mutation_tasks = [task for task, _ in receipts if task["kind"] == TaskKind.DENDRITRON_MUTATION.value]
        verification_tasks = [task for task, _ in receipts if task["kind"] == TaskKind.DENDRITRON_VERIFICATION.value]
        return {
            "root_before": initialized["canonical_root"],
            "root_after": status["canonical_root"],
            "queued": queued,
            "contribution": contribution,
            "mutation_payload_keys": sorted(mutation_tasks[0]["payload"]),
            "verification_payload_keys": sorted(verification_tasks[0]["payload"]),
            "mutation_task_count": len(mutation_tasks),
            "verification_task_count": len(verification_tasks),
            "validation_status": status["global_validation"],
            "all_receipts_accepted": all(receipt.get("accepted") for _, receipt in receipts),
            "audit_chain_valid": service.db.validate_audit_chain()[0],
            "int8_source_root": service.native10.store.model().export_int8()["source_root"],
        }


def harmful_candidate_rejection(temporary: Path) -> dict[str, Any]:
    app = create_app(temporary / "negative", bootstrap=False, lease_seconds=10.0)
    with TestClient(app) as client:
        service = app.state.service
        root_before = service.native10.initialize("compact", seed=7)["canonical_root"]
        config = service.native10.store.model().config
        validation = synthetic_global_validation_fixture(config, per_class=10, seed=1)
        service.native10.store.set_global_validation(validation)
        shard = synthetic_representation_shard(config, 0)
        service.native10.queue_mutation(shard, operation="expert_refit", category=0)
        identities = [Identity.generate() for _ in range(4)]
        for identity in identities:
            register(client, identity)
        receipts = [claim_execute_submit(client, identity) for identity in identities]
        state = service.native10.store.state()
        candidate = next(iter(state["candidates"].values()))
        local_reports = [
            json.loads(service.db.task(task_id)["output"])
            for task_id in candidate["trainer_tasks"]
        ]
        verification_reports = service.db.work_reports(
            f"native10-verify:{candidate['candidate_id']}",
            TaskKind.DENDRITRON_VERIFICATION.value,
        )
        global_reports = [json.loads(row["output"]) for row in verification_reports]
        final_receipt = receipts[-1][1]
        return {
            "root_before": root_before,
            "root_after": service.native10.store.model().root,
            "candidate_status": candidate["status"],
            "local_trainer_net_wins": [report["net_wins"] for report in local_reports],
            "global_verifier_net_wins": [report["net_wins"] for report in global_reports],
            "global_max_class_loss": [max(report["losses_by_class"]) for report in global_reports],
            "promoted": bool(final_receipt.get("promoted")),
            "reason": final_receipt.get("reason"),
        }


def lease_atomicity(temporary: Path) -> dict[str, Any]:
    path = temporary / "lease-atomic.sqlite3"
    db1 = Database(path)
    db2 = Database(path)
    node_id = "lease-proof-node-0001"
    capabilities = NodeCapabilities(
        cpu_count=2,
        memory_mb=1024,
        memory_available_mb=1024,
        disk_free_mb=4096,
        tags=["portable-numpy-v1", "deterministic-v2"],
    ).model_dump(mode="json")
    policy = SeedPolicy(
        cpu_percent=100,
        memory_percent=100,
        disk_limit_mb=4096,
        max_task_seconds=600,
        allow_on_battery=True,
        max_system_cpu_percent=100,
    ).model_dump(mode="json")
    db1.register_node(node_id, "proof-public-key", capabilities, policy)
    for slot in range(2):
        db1.add_task(
            TaskKind.EXPLORATION,
            {"work_key": f"atomic-lease-{slot}"},
            0,
            1,
            dedupe_key=f"atomic-lease-{slot}",
        )
    barrier = threading.Barrier(2)
    claims: list[str | None] = []

    def claim(db: Database) -> None:
        barrier.wait()
        row = db.claim_task(node_id, 60.0)
        claims.append(None if row is None else str(row["id"]))

    threads = [threading.Thread(target=claim, args=(database,)) for database in (db1, db2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    active = int(
        db1.conn.execute(
            "SELECT COUNT(*) AS count FROM tasks WHERE status='assigned' AND assigned_to=?",
            (node_id,),
        ).fetchone()["count"]
    )
    return {
        "successful_claims": sum(value is not None for value in claims),
        "active_leases": active,
        "partial_unique_index": bool(
            db1.conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='index' AND name='idx_tasks_one_active_per_node'"
            ).fetchone()
        ),
    }


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="dendriswarm-v051-proof-") as value:
        temporary = Path(value)
        exact = exact_profile_evidence(temporary)
        positive = successful_compact_round(temporary)
        negative = harmful_candidate_rejection(temporary)
        lease = lease_atomicity(temporary)
        contribution = positive["contribution"]
        gate_values: dict[str, dict[str, Any]] = {
            "exact_native10_profile_executed": {
                "pass": exact["parameter_count"] == 4_898_712
                and exact["config"]["field_blocks"] == 8
                and exact["config"]["scouts_per_category"] * exact["config"]["categories"] == 1000
                and exact["config"]["experts_per_category"] == 45,
                "evidence": exact,
            },
            "exact_tensor_archive_import_conformance": {
                "pass": exact["archive_import_root_equal"],
                "evidence": {
                    "initialized_root": exact["initialized_root"],
                    "imported_root": exact["imported_root"],
                    "historical_trained_weights_exercised": False,
                },
            },
            "exact_profile_full_global_verification_executed": {
                "pass": exact["validation_samples"] == 100
                and exact["samples_by_class"] == [1] * 100
                and exact["class_metric_count"] == 100,
                "evidence": {
                    "validation_samples": exact["validation_samples"],
                    "class_metric_count": exact["class_metric_count"],
                    "seconds": exact["full_global_verification_seconds"],
                },
            },
            "trainer_invisible_holdout": {
                "pass": "global_validation_hash" in positive["mutation_payload_keys"]
                and "validation_representations" not in positive["mutation_payload_keys"]
                and "validation_labels" not in positive["mutation_payload_keys"]
                and positive["validation_status"]["trainer_visible"] is False,
                "evidence": {
                    "mutation_payload_keys": positive["mutation_payload_keys"],
                    "validation_status": positive["validation_status"],
                },
            },
            "all_class_promotion_evidence": {
                "pass": contribution["sample_count"] == 240
                and contribution["samples_by_class"] == [20] * 12
                and len(contribution["losses_by_class"]) == 12,
                "evidence": {
                    "sample_count": contribution["sample_count"],
                    "samples_by_class": contribution["samples_by_class"],
                    "losses_by_class": contribution["losses_by_class"],
                },
            },
            "independent_quorum": {
                "pass": positive["mutation_task_count"] == 2
                and positive["verification_task_count"] == 2
                and len(set(contribution["trainer_nodes"])) == 2
                and len(set(contribution["verifier_nodes"])) == 2
                and set(contribution["trainer_nodes"]).isdisjoint(contribution["verifier_nodes"]),
                "evidence": {
                    "trainers": contribution["trainer_nodes"],
                    "verifiers": contribution["verifier_nodes"],
                },
            },
            "globally_beneficial_candidate_promoted": {
                "pass": contribution["net_wins"] > 0
                and max(contribution["losses_by_class"]) <= 1
                and positive["root_before"] != positive["root_after"],
                "evidence": {
                    "net_wins": contribution["net_wins"],
                    "max_class_loss": max(contribution["losses_by_class"]),
                    "root_before": positive["root_before"],
                    "root_after": positive["root_after"],
                },
            },
            "locally_positive_globally_harmful_candidate_rejected": {
                "pass": all(value > 0 for value in negative["local_trainer_net_wins"])
                and any(value < 0 for value in negative["global_verifier_net_wins"])
                and negative["candidate_status"] == "rejected"
                and negative["root_before"] == negative["root_after"]
                and negative["promoted"] is False,
                "evidence": negative,
            },
            "validation_reuse_is_bounded": {
                "pass": positive["validation_status"]["max_candidate_evaluations"] == 40
                and positive["validation_status"]["candidate_evaluations"] == 1,
                "evidence": positive["validation_status"],
            },
            "atomic_one_active_lease": {
                "pass": lease["successful_claims"] == 1
                and lease["active_leases"] == 1
                and lease["partial_unique_index"],
                "evidence": lease,
            },
            "request_result_size_boundaries_align": {
                "pass": MAX_HTTP_BODY_BYTES == MAX_CONTROL_REQUEST_BYTES
                and MAX_HTTP_BODY_BYTES > MAX_RESULT_OUTPUT_BYTES,
                "evidence": {
                    "result_output_bytes": MAX_RESULT_OUTPUT_BYTES,
                    "http_body_bytes": MAX_HTTP_BODY_BYTES,
                    "worker_request_bytes": MAX_CONTROL_REQUEST_BYTES,
                    "envelope_headroom_bytes": MAX_HTTP_BODY_BYTES - MAX_RESULT_OUTPUT_BYTES,
                },
            },
            "traceable_exact_composition": {
                "pass": contribution["root_after"] == positive["root_after"]
                and contribution["delta_hash"]
                and positive["int8_source_root"] == positive["root_after"],
                "evidence": {
                    "delta_hash": contribution["delta_hash"],
                    "root_after": positive["root_after"],
                    "int8_source_root": positive["int8_source_root"],
                },
            },
            "transfer_costs_disclosed": {
                "pass": exact["checkpoint_bytes"] > 20 * 1024 * 1024
                and exact["trainer_bundle_bytes"] < 2 * 1024 * 1024
                and exact["delta_bytes"] < 2 * 1024 * 1024,
                "evidence": {
                    "checkpoint_bytes": exact["checkpoint_bytes"],
                    "trainer_bundle_bytes": exact["trainer_bundle_bytes"],
                    "delta_bytes": exact["delta_bytes"],
                    "validation_bytes": exact["validation_bytes"],
                    "two_cold_verifier_checkpoint_bytes": exact["checkpoint_bytes"] * 2,
                    "checkpoint_cache_key": "content-addressed model root",
                },
            },
            "claim_scope_is_precise": {
                "pass": exact["baseline_training_included"] is False
                and exact["historical_trained_weights_exercised"] is False,
                "evidence": {
                    "demonstrated": "Native10 topology, tensor mutation, all-class promotion gate, deterministic composition, provenance",
                    "not_demonstrated": "historical trained checkpoint accuracy or baseline retraining",
                },
            },
            "audit_chain_and_receipts": {
                "pass": positive["all_receipts_accepted"] and positive["audit_chain_valid"],
                "evidence": {
                    "all_receipts_accepted": positive["all_receipts_accepted"],
                    "audit_chain_valid": positive["audit_chain_valid"],
                },
            },
        }
        report = {
            "format": "dendriswarm.proof.v051",
            "version": "0.5.1",
            "purpose": "Prove a trainer-invisible, all-class promotion gate for real Native10 topology mutations.",
            "baseline_training_included": False,
            "historical_trained_weights_exercised": False,
            "gates": gate_values,
            "passed": all(value["pass"] for value in gate_values.values()),
            "gate_count": len(gate_values),
        }
        output = ROOT / "docs" / "PROOF_RUN_V051.json"
        output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
