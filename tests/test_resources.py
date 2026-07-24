from __future__ import annotations

import json
import time

from fastapi.testclient import TestClient

from dendriswarm.coordinator.app import create_app
from dendriswarm.coordinator.db import Database
from dendriswarm.core.crypto import Identity, nonce
from dendriswarm.core.models import (
    NodeCapabilities,
    ResourceClass,
    SeedPolicy,
    TaskKind,
    TaskRequirements,
)
from dendriswarm.core.resources import derive_payload_requirements, effective_limits, node_can_run
from dendriswarm.worker.config import SeedPolicyStore
from dendriswarm.worker.node import SeedNode
from tests.helpers import claim_and_execute


def register_with_policy(client, identity, capabilities, policy):
    body = {
        "node_id": identity.node_id,
        "public_key": identity.public_key_b64,
        "capabilities": capabilities,
        "policy": policy,
        "timestamp": int(time.time()),
        "nonce": nonce(),
    }
    body["signature"] = identity.sign(body)
    response = client.post("/v1/nodes/register", json=body)
    assert response.status_code == 200, response.text


def signed_renewal(identity, task):
    body = {
        "action": "renew-lease",
        "node_id": identity.node_id,
        "task_id": task["id"],
        "lease_token": task["lease_token"],
        "timestamp": int(time.time()),
        "nonce": nonce(),
    }
    return {key: value for key, value in body.items() if key != "action"} | {
        "signature": identity.sign(body)
    }


def cifar100_native10_payload(optimizer_steps=27):
    return {
        "engine": "dendriswarm.native10-trainable.v6",
        "work_key": "native10-v6-search:resource-contract-regression",
        "bundle": {
            "operation": "scout_train",
            "config": {
                "input_width": 3072,
                "representation_width": 96,
                "categories": 20,
                "classes": 100,
                "max_routed_categories": 8,
                "experts_per_category": 45,
                "active_experts_per_update": 15,
                "expert_branches": 4,
                "branch_width": 12,
            },
        },
        "train_data": {"shape": [640, 96]},
        "train_labels": [0] * 640,
        "optimizer_steps": optimizer_steps,
        "required_tags": ["portable-numpy-v1", "independent-search-v1"],
    }


def test_resource_matching_respects_cpu_memory_disk_time_and_task_kind():
    capabilities = NodeCapabilities(
        cpu_count=8,
        memory_mb=16_000,
        memory_available_mb=8_000,
        disk_free_mb=20_000,
        machine="x86_64",
        supported_backends=["numpy-cpu"],
        tags=["portable-numpy-v1"],
    )
    policy = SeedPolicy(
        cpu_percent=25,
        memory_percent=10,
        disk_limit_mb=512,
        max_task_seconds=120,
        allowed_task_kinds=[TaskKind.EXPLORATION, TaskKind.VERIFICATION],
    )
    small = TaskRequirements(
        resource_class=ResourceClass.SMALL,
        min_cpu_threads=2,
        preferred_cpu_threads=2,
        min_memory_mb=1_000,
        min_disk_mb=128,
        estimated_runtime_seconds=60,
        required_tags=["portable-numpy-v1"],
    )
    assert node_can_run(TaskKind.EXPLORATION, small, capabilities, policy) == (True, "eligible")
    assert node_can_run(TaskKind.TRAINING, small, capabilities, policy)[1] == "task-kind-disabled"
    assert node_can_run(
        TaskKind.EXPLORATION,
        small.model_copy(update={"min_cpu_threads": 3}),
        capabilities,
        policy,
    )[1] == "cpu-budget-too-small"
    assert node_can_run(
        TaskKind.EXPLORATION,
        small.model_copy(update={"min_memory_mb": 2_000}),
        capabilities,
        policy,
    )[1] == "memory-budget-too-small"
    assert node_can_run(
        TaskKind.EXPLORATION,
        small.model_copy(update={"min_disk_mb": 1_000}),
        capabilities,
        policy,
    )[1] == "disk-budget-too-small"
    assert node_can_run(
        TaskKind.EXPLORATION,
        small.model_copy(update={"estimated_runtime_seconds": 121}),
        capabilities,
        policy,
    )[1] == "task-time-budget-too-small"


def test_portable_numpy_tasks_match_arm_and_x86_without_gpu():
    requirements = TaskRequirements(
        backend="numpy-cpu",
        supported_machines=[],
        min_memory_mb=128,
        min_disk_mb=8,
        estimated_runtime_seconds=10,
    )
    policy = SeedPolicy(cpu_percent=50, memory_percent=50, disk_limit_mb=256)
    for machine in ("x86_64", "amd64", "aarch64", "arm64"):
        capabilities = NodeCapabilities(
            cpu_count=4,
            memory_mb=4_096,
            memory_available_mb=3_000,
            disk_free_mb=5_000,
            machine=machine,
            accelerator="cpu",
            accelerators=["cpu"],
            supported_backends=["numpy-cpu"],
        )
        assert node_can_run(TaskKind.EXPLORATION, requirements, capabilities, policy)[0]


def test_scheduler_skips_paused_or_ineligible_nodes(tmp_path):
    db = Database(tmp_path / "resources.sqlite3")
    capabilities = NodeCapabilities(
        cpu_count=2,
        memory_mb=2_048,
        memory_available_mb=1_500,
        disk_free_mb=2_000,
        tags=["portable-numpy-v1"],
    ).model_dump(mode="json")
    db.register_node("paused", "key-paused", capabilities, SeedPolicy(paused=True).model_dump(mode="json"))
    db.register_node(
        "small",
        "key-small",
        capabilities,
        SeedPolicy(cpu_percent=50, memory_percent=25, disk_limit_mb=128).model_dump(mode="json"),
    )
    db.register_node(
        "capable",
        "key-capable",
        capabilities,
        SeedPolicy(cpu_percent=100, memory_percent=90, disk_limit_mb=1_000).model_dump(mode="json"),
    )
    requirements = TaskRequirements(
        min_cpu_threads=2,
        preferred_cpu_threads=2,
        min_memory_mb=800,
        min_disk_mb=256,
        estimated_runtime_seconds=30,
        required_tags=["portable-numpy-v1"],
    )
    task_id = db.add_task(TaskKind.EXPLORATION, {}, 100, 1, requirements=requirements)
    assert db.claim_task("paused", 30) is None
    assert db.claim_task("small", 30) is None
    claimed = db.claim_task("capable", 30)
    assert claimed is not None and claimed["id"] == task_id


def test_seed_policy_store_hot_reload_and_pause_resume(tmp_path):
    store = SeedPolicyStore(tmp_path / "seed-config.json")
    initial = store.load()
    assert initial.cpu_percent == 25 and initial.paused is False
    changed = store.update(
        cpu_percent=60,
        memory_percent=40,
        allowed_task_kinds=["exploration", "verification"],
        paused=True,
    )
    assert changed.cpu_percent == 60
    assert changed.allowed_task_kinds == [TaskKind.EXPLORATION, TaskKind.VERIFICATION]
    assert json.loads(store.path.read_text())["paused"] is True
    assert store.update(paused=False).paused is False


def test_running_seed_reloads_policy_without_restart(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "dendriswarm.worker.node.detect_capabilities",
        lambda state, policy: NodeCapabilities(
            cpu_count=8,
            memory_mb=8_000,
            memory_available_mb=6_000,
            disk_free_mb=10_000,
            machine="test",
            benchmark_units_per_second=1.0,
        ),
    )
    node = SeedNode("http://127.0.0.1:1", tmp_path / "seed")
    assert node.policy.cpu_percent == 25
    node.policy_store.update(cpu_percent=75, memory_percent=50)
    assert node._reload_policy() is True
    assert node.policy.cpu_percent == 75
    assert effective_limits(node.capabilities(), node.policy)["cpu_threads"] == 6


def test_low_resource_cpu_seed_can_complete_model_scouting(tmp_path):
    app = create_app(tmp_path / "coordinator", bootstrap=True)
    with TestClient(app) as client:
        identity = Identity.generate()
        register_with_policy(
            client,
            identity,
            {
                "cpu_count": 1,
                "physical_cpu_count": 1,
                "memory_mb": 768,
                "memory_available_mb": 640,
                "disk_free_mb": 1_024,
                "accelerator": "cpu",
                "accelerators": ["cpu"],
                "platform": "low-resource-test",
                "machine": "arm64",
                "python_version": "3.11",
                "supported_backends": ["numpy-cpu"],
                "benchmark_units_per_second": 100_000_000.0,
                "tags": ["reference-runtime-v2", "deterministic-v2", "portable-numpy-v1"],
            },
            {
                "cpu_percent": 100,
                "memory_percent": 75,
                "disk_limit_mb": 512,
                "max_task_seconds": 900,
                "allowed_task_kinds": ["exploration"],
                "allow_on_battery": True,
            },
        )
        result = claim_and_execute(client, identity)
        assert result is not None
        task, _, response = result
        assert task["kind"] == "exploration"
        assert task["requirements"]["backend"] == "numpy-cpu"
        assert response.status_code == 200
        account = client.get(f"/v1/nodes/{identity.node_id}").json()
        assert account["completed"] == 1
        assert account["credit_units"] == 0  # rewards unlock only after independent replica consensus

        replica = Identity.generate()
        register_with_policy(
            client, replica,
            {
                "cpu_count": 1, "memory_mb": 768, "memory_available_mb": 640,
                "disk_free_mb": 1_024, "machine": "aarch64",
                "supported_backends": ["numpy-cpu"],
                "tags": ["reference-runtime-v2", "deterministic-v2", "portable-numpy-v1"],
            },
            {
                "cpu_percent": 100, "memory_percent": 75, "disk_limit_mb": 512,
                "max_task_seconds": 900, "allowed_task_kinds": ["exploration"],
                "allow_on_battery": True,
            },
        )
        replica_result = claim_and_execute(client, replica)
        assert replica_result is not None and replica_result[2].status_code == 200
        assert client.get(f"/v1/nodes/{identity.node_id}").json()["credit_units"] > 0


def test_lease_renewal_supports_slow_volunteer_nodes(tmp_path):
    app = create_app(tmp_path / "coordinator", bootstrap=True, lease_seconds=1.0)
    with TestClient(app) as client:
        identity = Identity.generate()
        register_with_policy(
            client,
            identity,
            {
                "cpu_count": 2,
                "memory_mb": 2_048,
                "memory_available_mb": 1_500,
                "disk_free_mb": 2_000,
                "tags": ["reference-runtime-v2", "deterministic-v2"],
            },
            SeedPolicy(cpu_percent=100, memory_percent=90, disk_limit_mb=1_000).model_dump(mode="json"),
        )
        claim_body = {
            "action": "claim",
            "node_id": identity.node_id,
            "timestamp": int(time.time()),
            "nonce": nonce(),
        }
        signed_claim = {key: value for key, value in claim_body.items() if key != "action"} | {
            "signature": identity.sign(claim_body)
        }
        task = client.post("/v1/tasks/claim", json=signed_claim).json()["task"]
        before = task["lease_expires_at"]
        app.state.service.db.conn.execute(
            "UPDATE nodes SET last_seen=? WHERE id=?",
            (time.time() - 120, identity.node_id),
        )
        assert client.get("/v1/stats").json()["active_nodes"] == 0
        response = client.post("/v1/tasks/renew", json=signed_renewal(identity, task))
        assert response.status_code == 200, response.text
        assert response.json()["lease_expires_at"] > before
        assert client.get("/v1/stats").json()["active_nodes"] == 1


def test_coordinator_advertises_no_accelerator_requirement(tmp_path):
    with TestClient(create_app(tmp_path / "coordinator", bootstrap=True)) as client:
        meta = client.get("/v1/meta").json()
        assert meta["required_accelerator"] is None
        assert meta["portable_backend"] == "numpy-cpu"
        assert meta["live_seed_policy"] is True


def test_best_fit_scheduler_reserves_small_work_for_small_nodes(tmp_path):
    db = Database(tmp_path / "best-fit.sqlite3")
    capabilities = NodeCapabilities(
        cpu_count=8,
        memory_mb=16_000,
        memory_available_mb=12_000,
        disk_free_mb=20_000,
        tags=["portable-numpy-v1"],
    ).model_dump(mode="json")
    db.register_node(
        "large-node",
        "key",
        capabilities,
        SeedPolicy(cpu_percent=100, memory_percent=90, disk_limit_mb=10_000).model_dump(mode="json"),
    )
    small_id = db.add_task(
        TaskKind.EXPLORATION,
        {},
        1,
        10,
        requirements=TaskRequirements(min_memory_mb=128, preferred_cpu_threads=1),
    )
    large_id = db.add_task(
        TaskKind.EXPLORATION,
        {},
        1,
        10,
        requirements=TaskRequirements(min_memory_mb=4_000, preferred_cpu_threads=4),
    )
    claimed = db.claim_task("large-node", 30)
    assert claimed is not None
    assert claimed["id"] == large_id
    assert claimed["id"] != small_id


def test_seed_cli_import_does_not_load_coordinator_dependencies():
    import os
    import subprocess
    import sys

    environment = dict(os.environ)
    environment["PYTHONPATH"] = "src"
    script = (
        "import sys; import dendriswarm.cli; "
        "assert 'fastapi' not in sys.modules; assert 'sklearn' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", script], check=True, env=environment)


def test_default_quarter_share_on_one_gb_machine_can_scout():
    from dendriswarm.core.resources import estimate_reference_requirements

    capabilities = NodeCapabilities(
        cpu_count=2,
        memory_mb=1024,
        memory_available_mb=900,
        disk_free_mb=2048,
        tags=["deterministic-v2", "portable-numpy-v1"],
    )
    policy = SeedPolicy()
    requirements = estimate_reference_requirements(
        TaskKind.EXPLORATION,
        samples=320,
        features=64,
        branches=40,
        iterations=15,
        required_tags=["deterministic-v2", "portable-numpy-v1"],
    )
    assert effective_limits(capabilities, policy)["memory_mb"] == 256
    assert requirements.min_memory_mb <= 256
    assert node_can_run(TaskKind.EXPLORATION, requirements, capabilities, policy)[0]


def test_cifar100_native10_search_contract_fits_default_worker_without_early_timeout():
    expected = [(27, 150, 450), (36, 200, 600), (45, 249, 747), (54, 299, 897)]
    for steps, estimated, hard_timeout in expected:
        requirements = derive_payload_requirements(
            TaskKind.DENDRITRON_MUTATION, cifar100_native10_payload(steps)
        )
        assert requirements.estimated_runtime_seconds == estimated
        assert requirements.hard_timeout_seconds == hard_timeout
        assert requirements.hard_timeout_seconds <= SeedPolicy().max_task_seconds


def test_native10_verification_contract_uses_private_sample_count_hint():
    payload = cifar100_native10_payload()
    payload.pop("train_data")
    payload.pop("train_labels")
    payload["validation_sample_count"] = 500
    requirements = derive_payload_requirements(TaskKind.DENDRITRON_VERIFICATION, payload)
    assert requirements.estimated_runtime_seconds == 208
    assert requirements.hard_timeout_seconds == 624
    assert requirements.hard_timeout_seconds <= SeedPolicy().max_task_seconds


def test_coordinator_upgrades_and_rehabilitates_stale_queued_search_contracts(tmp_path):
    state = tmp_path / "coordinator"
    db = Database(state / "dendriswarm.sqlite3")
    task_id = db.add_task(
        TaskKind.DENDRITRON_MUTATION,
        cifar100_native10_payload(),
        4000,
        60,
        requirements=TaskRequirements(
            min_memory_mb=96,
            max_memory_mb=256,
            min_disk_mb=1,
            estimated_runtime_seconds=30,
            hard_timeout_seconds=75,
            max_artifact_bytes=400_000,
            required_tags=["portable-numpy-v1", "independent-search-v1"],
        ),
    )
    db.conn.execute(
        "UPDATE tasks SET attempts=2,excluded_nodes=? WHERE id=?",
        (json.dumps(["timed-out-worker"]), task_id),
    )
    db.conn.close()

    app = create_app(state)
    row = app.state.service.db.task(task_id)
    requirements = json.loads(row["requirements"])
    assert requirements["estimated_runtime_seconds"] == 150
    assert requirements["hard_timeout_seconds"] == 450
    assert row["attempts"] == 0
    assert json.loads(row["excluded_nodes"]) == []
    assert app.state.service.db.validate_audit_chain()[0]


def test_cache_budget_evicts_old_content_addressed_artifacts(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "dendriswarm.worker.node.detect_capabilities",
        lambda state, policy: NodeCapabilities(
            cpu_count=2,
            memory_mb=2048,
            memory_available_mb=1500,
            disk_free_mb=10_000,
            benchmark_units_per_second=100_000_000,
        ),
    )
    node = SeedNode("http://127.0.0.1:1", tmp_path / "seed")
    node.policy = SeedPolicy(disk_limit_mb=64)
    first = node.cache_dir / "dataset-first.json"
    second = node.cache_dir / "dataset-second.json"
    with first.open("wb") as handle:
        handle.truncate(40 * 1024 * 1024)
    time.sleep(0.01)
    with second.open("wb") as handle:
        handle.truncate(20 * 1024 * 1024)
    target = node.cache_dir / "dataset-new.json"
    node._make_cache_room(30 * 1024 * 1024, target)
    assert not first.exists()
    assert second.exists()
