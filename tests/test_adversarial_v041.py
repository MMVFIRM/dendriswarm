from __future__ import annotations

import copy
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient

from dendriswarm.cli import _policy_changes
from dendriswarm.coordinator.app import MAX_HTTP_BODY_BYTES, create_app
from dendriswarm.coordinator.db import Database
from dendriswarm.core.crypto import Identity, content_hash, nonce
from dendriswarm.core.models import NodeCapabilities, SeedPolicy, TaskKind, TaskRequirements
from dendriswarm.core.resources import contract_covers, derive_payload_requirements, node_can_run
from dendriswarm.leverage.epoch import ChallengeEpoch
from dendriswarm.leverage.manifest import build_manifest
from dendriswarm.leverage.service import LeverageService
from dendriswarm.leverage.stats import GatePolicy
from dendriswarm.leverage.store import ModelStore
from dendriswarm.leverage.tissue import Territory, TerritoryTissue
from dendriswarm.leverage.workload import honest_delta, make_surrogate_workload, train_parent
from dendriswarm.tissues.reference import (
    artifact_consensus_hash,
    budgeted_dataset_artifact,
    make_digits_dataset,
)
from dendriswarm.worker.config import SeedPolicyStore
from dendriswarm.worker.isolation import TaskExecutionCancelled, execute_task_isolated
from dendriswarm.worker.node import SeedNode
from tests.helpers import register, signed_action


def _fund(service: LeverageService, contributor: str) -> None:
    service.register_contributor(contributor)
    service.fund_contributor(contributor, f"principal:{contributor}")


def test_route_share_counts_removed_inherited_branch_influence():
    parent = TerritoryTissue(
        np.asarray([[0.0], [1.0]]), np.asarray([0, 1]), top_k=1, temperature=1.0,
        anchors=np.asarray([[0.0], [1.0]]),
    )
    x = np.zeros((100, 1), dtype=np.float64)
    y = np.ones(100, dtype=np.int64)
    policy = GatePolicy(min_net_wins=25, min_subgroup_samples=10)
    service = LeverageService(parent, ChallengeEpoch(x, y, x + 0.01, y, policy, salt="route-share"), policy)
    _fund(service, "attacker")
    manifest = build_manifest(
        service.canonical_root,
        parent.representation_root(),
        "attacker",
        {0: (np.asarray([100.0]), 0)},
        [],
        Territory((0,), 0.01),
    )
    candidate_id, verdict = service.submit(manifest, "attacker")
    assert verdict == "rejected:route-share-violation"
    assert service.candidates[candidate_id].detail["delta_active_share"] == pytest.approx(1.0)


def test_one_node_cannot_hoard_or_reclaim_expired_task(tmp_path):
    db = Database(tmp_path / "leases.sqlite3")
    capabilities = NodeCapabilities(tags=["portable-numpy-v1", "deterministic-v2"]).model_dump(mode="json")
    policy = SeedPolicy(cpu_percent=100, memory_percent=100, max_task_seconds=300).model_dump(mode="json")
    db.register_node("attacker", "key-a", capabilities, policy)
    db.register_node("honest", "key-h", capabilities, policy)
    first_id = db.add_task(TaskKind.EXPLORATION, {"required_tags": ["portable-numpy-v1"]}, 1, 10)
    second_id = db.add_task(TaskKind.EXPLORATION, {"required_tags": ["portable-numpy-v1"]}, 1, 9)
    first = db.claim_task("attacker", 0.01)
    assert first is not None and first["id"] == first_id
    assert db.claim_task("attacker", 0.01) is None  # one active lease per identity
    time.sleep(0.02)
    db.requeue_expired_leases()
    assert db.claim_task("attacker", 0.01) is None  # quarantined and excluded
    recovered = db.claim_task("honest", 30.0)
    assert recovered is not None and recovered["id"] == first_id
    assert second_id != first_id and "attacker" in json.loads(db.task(first_id)["excluded_nodes"])


def test_lease_renewal_never_exceeds_absolute_deadline(tmp_path):
    db = Database(tmp_path / "renewal.sqlite3")
    caps = NodeCapabilities(tags=["portable-numpy-v1", "deterministic-v2"]).model_dump(mode="json")
    policy = SeedPolicy(cpu_percent=100, memory_percent=100, max_task_seconds=60).model_dump(mode="json")
    db.register_node("node", "key", caps, policy)
    requirements = TaskRequirements(estimated_runtime_seconds=5, hard_timeout_seconds=20)
    task_id = db.add_task(TaskKind.EXPLORATION, {}, 0, 1, requirements=requirements)
    row = db.claim_task("node", 1.0)
    assert row is not None and row["id"] == task_id
    deadline = float(row["lease_deadline_at"])
    expiry = float(row["lease_expires_at"])
    for _ in range(100):
        renewed = db.renew_task_lease(task_id, "node", row["lease_token"], 10_000)
        assert renewed is not None
        expiry = renewed
    assert expiry <= deadline


def test_scheduler_scans_past_256_incompatible_tasks(tmp_path):
    db = Database(tmp_path / "scheduler.sqlite3")
    caps = NodeCapabilities(tags=["portable-numpy-v1", "deterministic-v2"]).model_dump(mode="json")
    policy = SeedPolicy(cpu_percent=100, memory_percent=100, max_task_seconds=300).model_dump(mode="json")
    db.register_node("portable", "key", caps, policy)
    incompatible = TaskRequirements(required_tags=["unsupported-special-backend"])
    for index in range(256):
        db.add_task(TaskKind.EXPLORATION, {}, 0, 1000 - index, requirements=incompatible)
    eligible_id = db.add_task(
        TaskKind.EXPLORATION, {}, 0, 1,
        requirements=TaskRequirements(required_tags=["portable-numpy-v1"]),
    )
    claimed = db.claim_task("portable", 30.0)
    assert claimed is not None and claimed["id"] == eligible_id


def test_one_identity_cannot_supply_multiple_replicas_of_same_work(tmp_path):
    db = Database(tmp_path / "replica-independence.sqlite3")
    caps = NodeCapabilities(tags=["portable-numpy-v1", "deterministic-v2"]).model_dump(mode="json")
    policy = SeedPolicy(cpu_percent=100, memory_percent=100, max_task_seconds=300).model_dump(mode="json")
    db.register_node("first", "key-first", caps, policy)
    db.register_node("second", "key-second", caps, policy)
    payload = {"work_key": "explore:shared", "required_tags": ["portable-numpy-v1"]}
    first_id = db.add_task(TaskKind.EXPLORATION, payload, 0, 10, dedupe_key="shared:0")
    second_id = db.add_task(TaskKind.EXPLORATION, payload, 0, 10, dedupe_key="shared:1")
    first = db.claim_task("first", 30.0)
    assert first is not None and first["id"] == first_id
    db.record_work_report("explore:shared", TaskKind.EXPLORATION.value, "first", first_id, {"ok": True})
    assert db.complete_task(first_id, "first", first["lease_token"], {"ok": True})
    assert db.claim_task("first", 30.0) is None
    second = db.claim_task("second", 30.0)
    assert second is not None and second["id"] == second_id


def test_leverage_submit_is_serialized_and_persistence_remains_valid(tmp_path):
    workload = make_surrogate_workload(private_seed=42, canary_seed=43, replication_seed=44)
    parent = train_parent(workload)
    policy = GatePolicy()
    state_path = tmp_path / "leverage.json"
    service = LeverageService(
        parent,
        ChallengeEpoch(workload.x_private, workload.y_private, workload.x_replication, workload.y_replication, policy, salt="race"),
        policy,
        state_path=state_path,
    )
    manifests = []
    for contributor, categories, seed in (("alice", tuple(range(8)), 11), ("bob", tuple(range(8, 16)), 29)):
        _fund(service, contributor)
        replaced, added = honest_delta(parent, workload, categories, seed=seed)
        manifests.append(build_manifest(
            service.canonical_root, parent.representation_root(), contributor,
            replaced, added, Territory(categories, 0.25),
        ))
    barrier = threading.Barrier(2)

    def submit(item):
        barrier.wait()
        return service.submit(item, item.contributor)

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(submit, manifests))
    accepted = [result for result in results if result[1] == "accepted:promotion-candidate"]
    deferred = [result for result in results if result[1] == "rejected:canary-window-active"]
    assert len(accepted) == 1 and len(deferred) == 1
    assert service.active_canary_id == accepted[0][0]
    assert sum(record.status == "promoted" for record in service.candidates.values()) == 1
    assert service.validate_audit_chain()
    loaded = LeverageService.load(state_path)
    assert loaded.canonical_root == service.canonical_root and loaded.validate_audit_chain()


def test_resource_contract_is_derived_from_materialized_payload():
    dataset = make_digits_dataset()
    compact = budgeted_dataset_artifact(dataset, 320, 320, 7)
    payload = {
        "config": {"branches": 320, "top_k": 3, "temperature": 0.18, "iterations": 1000, "seed": 7},
        "_dataset": compact,
    }
    derived = derive_payload_requirements(TaskKind.EXPLORATION, payload)
    understated = TaskRequirements(
        min_memory_mb=32,
        max_memory_mb=64,
        min_disk_mb=0,
        estimated_runtime_seconds=1,
        hard_timeout_seconds=5,
        max_artifact_bytes=1024,
    )
    covered, reason = contract_covers(understated, derived)
    assert covered is False and reason.startswith("understated-")


def test_budgeted_dataset_artifact_contains_only_assigned_rows():
    dataset = make_digits_dataset()
    shard = budgeted_dataset_artifact(dataset, 120, 80, 7)
    assert len(shard["features"]) == 200
    assert len(shard["labels"]) == 200
    assert len(shard["features"]) < len(dataset["features"])
    assert len(json.dumps(shard)) < len(json.dumps(dataset))


def test_zero_available_memory_is_never_treated_as_unknown():
    capabilities = NodeCapabilities(memory_mb=1024, memory_available_mb=0, disk_free_mb=1024)
    policy = SeedPolicy(cpu_percent=100, memory_percent=100, disk_limit_mb=1024)
    requirement = TaskRequirements(min_memory_mb=64, max_memory_mb=128)
    assert node_can_run(TaskKind.EXPLORATION, requirement, capabilities, policy)[1] == "memory-budget-too-small"


def test_declared_artifact_ceiling_must_fit_local_control_plane_memory():
    capabilities = NodeCapabilities(
        memory_mb=256, memory_available_mb=256, disk_free_mb=4096,
        tags=["portable-numpy-v1", "deterministic-v2"],
    )
    policy = SeedPolicy(cpu_percent=100, memory_percent=100, disk_limit_mb=4096, max_task_seconds=300)
    requirement = TaskRequirements(
        min_memory_mb=64,
        max_memory_mb=128,
        max_artifact_bytes=128 * 1024 * 1024,
        hard_timeout_seconds=120,
    )
    assert node_can_run(TaskKind.EXPLORATION, requirement, capabilities, policy)[1] == "artifact-exceeds-control-plane-memory-budget"


def test_active_task_stops_after_hot_reloaded_pause(tmp_path):
    policy_store = SeedPolicyStore(tmp_path / "policy.json")
    policy_store.save(SeedPolicy(cpu_percent=100, memory_percent=100, max_task_seconds=60, allow_on_battery=True))
    dataset = {
        "format": "dendriswarm.dataset.v1",
        "name": "slow",
        "source": "test",
        "license": "test",
        "description": "test",
        "features": [[float(i % 7), float(i % 5)] for i in range(200)],
        "labels": [i % 2 for i in range(200)],
        "splits": {"train": list(range(160)), "validation": list(range(160, 200)), "test": []},
        "feature_width": 2,
        "classes": 2,
        "seed": 1,
    }
    from dendriswarm.tissues.reference import dataset_hash
    dataset["sha256"] = dataset_hash(dataset)
    payload = {
        "config": {"branches": 20, "top_k": 1, "temperature": 0.18, "iterations": 100_000, "seed": 7},
        "_dataset": dataset,
    }
    capabilities = NodeCapabilities(
        cpu_count=2, memory_mb=2048, memory_available_mb=1500, disk_free_mb=2048,
        tags=["portable-numpy-v1", "deterministic-v2"],
    )
    requirements = TaskRequirements(
        min_memory_mb=64, max_memory_mb=512, estimated_runtime_seconds=10,
        hard_timeout_seconds=60, required_tags=["portable-numpy-v1"],
    )

    def pause():
        time.sleep(0.2)
        policy_store.update(paused=True)

    thread = threading.Thread(target=pause, daemon=True)
    thread.start()
    started = time.monotonic()
    with pytest.raises(TaskExecutionCancelled, match="live-policy-change:seed-paused"):
        execute_task_isolated(
            TaskKind.EXPLORATION, payload, cpu_threads=1, memory_limit_mb=512,
            timeout_seconds=60, requirements=requirements, capabilities=capabilities,
            policy_store=policy_store, poll_seconds=0.05,
        )
    assert time.monotonic() - started < 5.0


def test_large_isolated_task_result_does_not_deadlock_queue(tmp_path):
    policy_store = SeedPolicyStore(tmp_path / "large-result-policy.json")
    policy_store.save(SeedPolicy(
        cpu_percent=100,
        memory_percent=100,
        max_task_seconds=20,
        allow_on_battery=True,
    ))
    width = 256
    sample_count = 200
    dataset = {
        "features": [
            [float((row + column) % 17) for column in range(width)]
            for row in range(sample_count)
        ],
        "labels": [row % 2 for row in range(sample_count)],
        "splits": {
            "train": list(range(160)),
            "validation": list(range(160, sample_count)),
            "test": [],
        },
    }
    payload = {
        "config": {
            "branches": 160,
            "top_k": 3,
            "temperature": 0.18,
            "iterations": 1,
            "seed": 7,
        },
        "dataset_hash": "a" * 64,
        "_dataset": dataset,
    }
    capabilities = NodeCapabilities(
        cpu_count=2,
        memory_mb=2_048,
        memory_available_mb=1_500,
        disk_free_mb=2_048,
        tags=["reference-runtime-v2", "deterministic-v2"],
    )
    requirements = TaskRequirements(
        min_memory_mb=64,
        max_memory_mb=1_024,
        estimated_runtime_seconds=5,
        hard_timeout_seconds=20,
        required_tags=["reference-runtime-v2"],
    )

    started = time.monotonic()
    result = execute_task_isolated(
        TaskKind.TRAINING,
        payload,
        cpu_threads=1,
        memory_limit_mb=1_024,
        timeout_seconds=20,
        requirements=requirements,
        capabilities=capabilities,
        policy_store=policy_store,
        poll_seconds=0.05,
    )

    assert len(json.dumps(result)) > 64 * 1024
    assert time.monotonic() - started < 10.0


def test_seed_status_clears_stale_error_after_recovery(tmp_path):
    node = SeedNode("http://127.0.0.1:8787", tmp_path / "status-recovery")
    node._status(state="error", reason="timed out", current_task=None, last_error="timed out")
    node._status(state="working", reason="task-active", current_task={"id": "next-task"})

    status = json.loads(node.status_path.read_text())
    assert status["state"] == "working"
    assert status["last_error"] is None


def test_canary_rejects_invalid_and_noninformative_labels():
    workload = make_surrogate_workload(private_seed=42, canary_seed=43, replication_seed=44)
    parent = train_parent(workload)
    policy = GatePolicy(canary_min_informative_predictions=1)
    service = LeverageService(
        parent,
        ChallengeEpoch(workload.x_private, workload.y_private, workload.x_replication, workload.y_replication, policy, salt="canary"),
        policy,
    )
    _fund(service, "builder")
    categories = tuple(range(8))
    replaced, added = honest_delta(parent, workload, categories)
    manifest = build_manifest(service.canonical_root, parent.representation_root(), "builder", replaced, added, Territory(categories, 0.25))
    candidate_id, verdict = service.submit(manifest, "builder")
    assert verdict == "accepted:promotion-candidate"
    x = workload.x_canary[:100]
    with pytest.raises(ValueError, match="outside the committed model class schema"):
        service.record_canary_batch(candidate_id, x, np.full(100, -1), source_id="bad", source_kind="heldout-canary-batch")


def test_disclosure_policy_body_is_cryptographically_bound():
    x = np.asarray([[0.0], [1.0]])
    y = np.asarray([0, 1])
    policy = GatePolicy()
    epoch = ChallengeEpoch(x, y, x + 0.1, y, policy, salt="policy")
    epoch.close({"passed": True})
    disclosure = epoch.reveal()
    assert ChallengeEpoch.verify_disclosure(disclosure)
    tampered = copy.deepcopy(disclosure)
    tampered["policy"]["bond_units"] = 999_999_999
    assert not ChallengeEpoch.verify_disclosure(tampered)


def test_request_body_limit_runs_before_endpoint_parsing(tmp_path):
    app = create_app(tmp_path)
    with TestClient(app) as client:
        response = client.post(
            "/v1/nodes/register",
            content=b"{" + b" " * MAX_HTTP_BODY_BYTES,
            headers={"content-type": "application/json"},
        )
        assert response.status_code == 413


def test_malformed_exploration_result_is_committed_as_worker_failure(tmp_path):
    app = create_app(tmp_path / "malformed-result", bootstrap=True)
    with TestClient(app) as client:
        identity = Identity.generate()
        register(client, identity)
        envelope = client.post("/v1/tasks/claim", json=signed_action(identity, "claim")).json()
        task = envelope["task"]
        body = {
            "node_id": identity.node_id,
            "task_id": task["id"],
            "lease_token": task["lease_token"],
            "duration_ms": 1,
            "output": {
                "config": {"branches": 999},
                "validation_accuracy": 1.0,
                "sample_count": 1,
                "correct_count": 1,
                "runtime": {"backend": "numpy-cpu", "machine": "x86_64", "python": "3.13"},
            },
        }
        body["signature"] = identity.sign(body)
        response = client.post("/v1/tasks/result", json=body)
        assert response.status_code == 400
        row = app.state.service.db.task(task["id"])
        node = app.state.service.db.node(identity.node_id)
        assert row["status"] == "queued"
        assert identity.node_id in json.loads(row["excluded_nodes"])
        assert node["failed"] == 1 and float(node["quarantine_until"]) > time.time()


def test_terminal_inference_failure_refunds_requester_once(tmp_path):
    db = Database(tmp_path / "inference-refund.sqlite3")
    caps = NodeCapabilities(tags=["portable-numpy-v1", "deterministic-v2"]).model_dump(mode="json")
    policy = SeedPolicy(cpu_percent=100, memory_percent=100, max_task_seconds=300).model_dump(mode="json")
    requester = "requester"
    workers = [f"worker-{index}" for index in range(3)]
    db.register_node(requester, "requester-key", caps, policy)
    for worker in workers:
        db.register_node(worker, f"{worker}-key", caps, policy)
        db.credit(f"bond-funding:{worker}", worker, 4_000, "test bond funding")
    db.credit("request-funding", requester, 1_000, "test request funding")
    task_id, created = db.create_inference_task(
        "requester:one", requester, "a" * 64, [0.0], 1_000, 800,
    )
    assert created and db.node(requester)["credit_units"] == 0
    for worker in workers:
        claimed = db.claim_task(worker, 30.0)
        assert claimed is not None and claimed["id"] == task_id
        db.reject_assigned_task(task_id, worker, claimed["lease_token"], {"invalid": True}, "invalid proof")
    assert db.task(task_id)["status"] == "failed"
    assert db.node(requester)["credit_units"] == 1_000
    refund_entries = db.conn.execute(
        "SELECT COUNT(*) AS count FROM ledger WHERE entry_key='inference-refund:requester:one'"
    ).fetchone()["count"]
    assert refund_entries == 1


def test_worker_rejects_invalid_or_oversized_declared_response_length(tmp_path):
    node = SeedNode("http://127.0.0.1:8787", tmp_path / "response-limit")

    def malformed(_request):
        return httpx.Response(200, headers={"content-length": "not-a-number"}, content=b"{}")

    node.client.close()
    node.client = httpx.Client(transport=httpx.MockTransport(malformed))
    with pytest.raises(RuntimeError, match="invalid Content-Length"):
        node._request_json("GET", "http://127.0.0.1/test")

    def oversized(_request):
        return httpx.Response(200, headers={"content-length": "1048577"}, content=b"{}")

    node.client.close()
    node.client = httpx.Client(transport=httpx.MockTransport(oversized))
    with pytest.raises(RuntimeError, match="declared byte limit"):
        node._request_json("GET", "http://127.0.0.1/test", max_bytes=1_048_576)


def test_result_outbox_survives_network_failure(tmp_path, monkeypatch):
    node = SeedNode("http://127.0.0.1:8787", tmp_path / "seed")
    body = {
        "node_id": node.identity.node_id,
        "task_id": "a" * 32,
        "lease_token": "b" * 32,
        "duration_ms": 1,
        "output": {},
        "signature": "c" * 64,
    }
    path = node._queue_result(body)

    def fail(*args, **kwargs):
        raise httpx.ConnectError("offline")

    monkeypatch.setattr(node, "_request_json", fail)
    with pytest.raises(httpx.ConnectError):
        node._flush_outbox()
    assert path.exists()
    monkeypatch.setattr(node, "_request_json", lambda *args, **kwargs: (200, {"accepted": True}))
    assert node._flush_outbox() == 1
    assert not path.exists()
    assert (node.receipts_dir / f"{body['task_id']}.json").exists()


def test_expired_signed_task_is_rejected_before_materialization(tmp_path):
    node = SeedNode("http://127.0.0.1:8787", tmp_path / "seed")
    task = {
        "id": "a" * 32,
        "kind": "exploration",
        "payload": {"dataset_hash": "b" * 64},
        "requirements": TaskRequirements().model_dump(mode="json"),
        "lease_expires_at": time.time() - 1,
        "lease_deadline_at": time.time() + 30,
    }
    with pytest.raises(TaskExecutionCancelled, match="already expired"):
        node._execute_signed_task(task)


def test_remote_http_requires_opt_in_and_fingerprint_mismatch_fails(tmp_path):
    with pytest.raises(ValueError, match="require HTTPS"):
        SeedNode("http://example.com", tmp_path / "remote")
    node = SeedNode(
        "https://example.com", tmp_path / "pinned",
        expected_coordinator_fingerprint="00" * 32,
    )
    identity = Identity.generate()
    with pytest.raises(RuntimeError, match="does not match"):
        node._pin_coordinator(identity.public_key_b64)


def test_identity_creation_race_converges_on_one_key(tmp_path):
    directory = tmp_path / "keys"
    with ThreadPoolExecutor(max_workers=8) as pool:
        identities = list(pool.map(lambda _: Identity.load_or_create(directory), range(16)))
    assert len({identity.public_key_b64 for identity in identities}) == 1


def test_share_flag_has_exact_cpu_and_memory_semantics():
    class Args:
        share = 1
        cpu_percent = memory_percent = memory_mb = disk_mb = None
        max_task_minutes = task_types = min_battery_percent = max_system_cpu_percent = None
        battery = paused = None

    assert _policy_changes(Args()) == {"cpu_percent": 1, "memory_percent": 1}
    Args.share = 100
    assert _policy_changes(Args()) == {"cpu_percent": 100, "memory_percent": 100}


def test_cross_architecture_artifact_consensus_tolerates_harmless_rounding():
    dataset = make_digits_dataset()
    from dendriswarm.tissues.reference import ReferenceDendritron, TissueConfig, train_and_validation
    x, y = train_and_validation(dataset)
    model = ReferenceDendritron.train(x, y, TissueConfig(branches=40, top_k=1, iterations=2, seed=7))
    artifact = model.artifact(TissueConfig(branches=40, top_k=1, iterations=2, seed=7).as_dict(), dataset["sha256"])
    perturbed = copy.deepcopy(artifact)
    perturbed["centers"][0][0] += 1e-10
    perturbed["sha256"] = content_hash(perturbed)
    assert artifact["sha256"] != perturbed["sha256"]
    assert artifact_consensus_hash(artifact) == artifact_consensus_hash(perturbed)


def test_model_store_rejects_caller_supplied_non_composition():
    parent = TerritoryTissue(np.asarray([[0.0], [1.0]]), np.asarray([0, 1]), 1, 1.0)
    store = ModelStore()
    root = store.register_genesis(parent)
    manifest = build_manifest(root, parent.representation_root(), "builder", {0: (np.asarray([0.2]), 0)}, [], Territory((0,), 1.0))
    tissue, certificate = store.compose(manifest)
    tampered = TerritoryTissue(tissue.centers + 0.5, tissue.owners, tissue.top_k, tissue.temperature, active_regions=tissue.active_regions, anchors=tissue.anchors)
    with pytest.raises(ValueError, match="deterministic manifest composition|representation mismatch"):
        store.register_candidate(tampered, manifest, certificate)
