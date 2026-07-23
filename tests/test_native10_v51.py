from __future__ import annotations

import json
import threading
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

from dendriswarm.coordinator.app import create_app
from dendriswarm.coordinator.db import Database
from dendriswarm.core.crypto import Identity, nonce
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
    make_global_validation_artifact,
    synthetic_global_validation_fixture,
)
from tests.helpers import claim_and_execute, register, signed_action


def _signed_fetch(identity: Identity, task: dict) -> dict:
    body = {
        "action": "fetch-native10-validation",
        "node_id": identity.node_id,
        "task_id": task["id"],
        "lease_token": task["lease_token"],
        "timestamp": int(time.time()),
        "nonce": nonce(),
    }
    return {key: value for key, value in body.items() if key != "action"} | {
        "signature": identity.sign(body)
    }


def test_global_validation_artifact_requires_every_class():
    config = Native10Config.compact_demo()
    x = np.zeros((20, config.representation_width), dtype=np.float32)
    y = np.zeros(20, dtype=np.int64)
    with pytest.raises(ValueError, match="cover every class"):
        make_global_validation_artifact(
            config,
            x,
            y,
            source="invalid-one-class",
            policy=GlobalValidationPolicy(min_samples_per_class=1),
        )


def test_trainer_payload_excludes_holdout_and_private_fetch_is_denied(tmp_path):
    app = create_app(tmp_path / "private-validation", bootstrap=False, lease_seconds=5.0)
    with TestClient(app) as client:
        service = app.state.service
        service.native10.initialize("compact", seed=7)
        queued = service.native10.queue_demo_round(category=0, operation="expert_refit")
        identity = Identity.generate()
        register(client, identity)
        response = client.post("/v1/tasks/claim", json=signed_action(identity, "claim"))
        assert response.status_code == 200
        task = response.json()["task"]
        assert task["id"] in queued["mutation_tasks"]
        assert "validation_representations" not in task["payload"]
        assert "validation_labels" not in task["payload"]
        assert "global_validation_hash" in task["payload"]
        denied = client.post(
            f"/v1/native10/validation/{task['payload']['global_validation_hash']}",
            json=_signed_fetch(identity, task),
        )
        assert denied.status_code == 403


def test_global_gate_rejects_local_gain_that_harms_all_class_holdout(tmp_path):
    app = create_app(tmp_path / "global-reject", bootstrap=False, lease_seconds=5.0)
    with TestClient(app) as client:
        service = app.state.service
        root_before = service.native10.initialize("compact", seed=7)["canonical_root"]
        config = service.native10.store.model().config
        shifted = synthetic_global_validation_fixture(config, per_class=10, seed=1)
        service.native10.store.set_global_validation(shifted)
        shard = synthetic_representation_shard(config, 0)
        service.native10.queue_mutation(shard, operation="expert_refit", category=0)
        identities = [Identity.generate() for _ in range(4)]
        for identity in identities:
            register(client, identity)
        receipts = [claim_and_execute(client, identity) for identity in identities]
        assert all(item is not None and item[2].status_code == 200 for item in receipts)
        final = receipts[-1][2].json()
        assert final["promoted"] is False
        assert final["reason"] == "verification-gate-failed"
        assert service.native10.store.model().root == root_before
        state = service.native10.store.state()
        candidate = next(iter(state["candidates"].values()))
        assert candidate["status"] == "rejected"
        mutation_rows = [
            service.db.task(task_id) for task_id in candidate["trainer_tasks"]
        ]
        assert all(json.loads(row["output"])["net_wins"] > 0 for row in mutation_rows)


def test_exact_profile_executes_full_global_verification_and_npz_import(tmp_path):
    config = Native10Config()
    model = Native10Dendritron.initialize(config)
    archive = tmp_path / "native10-exact.npz"
    np.savez(archive, **model.tensors)
    imported = load_external_checkpoint(archive, config=config)
    assert imported.root == model.root

    shard = synthetic_representation_shard(
        config, 0, train_per_class=2, validation_per_class=1
    )
    bundle = imported.component_bundle("expert_refit", 0)
    mutation = execute_mutation(
        bundle,
        np.asarray(shard["train_representations"], dtype=np.float32),
        np.asarray(shard["train_labels"], dtype=np.int64),
        np.asarray(shard["train_representations"], dtype=np.float32),
        np.asarray(shard["train_labels"], dtype=np.int64),
    )
    validation = synthetic_global_validation_fixture(
        config,
        per_class=1,
        policy=GlobalValidationPolicy(
            min_samples_per_class=1,
            max_loss_per_class=1,
            max_loss_rate_per_class=1.0,
        ),
    )
    x, y, _ = decode_global_validation_artifact(validation, expected_config=config)
    result = verify_mutation_full(
        imported.artifact(),
        bundle,
        mutation["delta"],
        x,
        y,
        validation_hash_value=validation["sha256"],
    )
    assert imported.parameter_count == 4_898_712
    assert result["sample_count"] == 100
    assert result["samples_by_class"] == [1] * 100
    assert len(result["losses_by_class"]) == 100
    assert imported.apply_delta(mutation["delta"]).root != imported.root
    checkpoint_bytes = len(json.dumps(imported.artifact(), separators=(",", ":")).encode())
    bundle_bytes = len(json.dumps(bundle, separators=(",", ":")).encode())
    delta_bytes = len(json.dumps(mutation["delta"], separators=(",", ":")).encode())
    assert checkpoint_bytes > 20 * 1024 * 1024
    assert bundle_bytes < 2 * 1024 * 1024
    assert delta_bytes < 2 * 1024 * 1024


def test_one_active_lease_is_atomic_across_database_instances(tmp_path):
    path = tmp_path / "shared.sqlite3"
    db1 = Database(path)
    db2 = Database(path)
    node_id = "n" * 20
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
    db1.register_node(node_id, "public-key", capabilities, policy)
    for slot in range(2):
        db1.add_task(
            TaskKind.EXPLORATION,
            {"work_key": f"lease-race-{slot}"},
            0,
            1,
            dedupe_key=f"lease-race-{slot}",
        )

    barrier = threading.Barrier(2)
    claimed: list[str | None] = []

    def run(db: Database) -> None:
        barrier.wait()
        row = db.claim_task(node_id, 60.0)
        claimed.append(None if row is None else str(row["id"]))

    threads = [threading.Thread(target=run, args=(db,)) for db in (db1, db2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert sum(value is not None for value in claimed) == 1
    active = db1.conn.execute(
        "SELECT COUNT(*) AS count FROM tasks WHERE status='assigned' AND assigned_to=?",
        (node_id,),
    ).fetchone()["count"]
    assert active == 1


def test_result_body_limits_leave_signed_envelope_headroom():
    from dendriswarm.core.limits import (
        MAX_CONTROL_REQUEST_BYTES, MAX_HTTP_BODY_BYTES, MAX_RESULT_OUTPUT_BYTES,
    )

    assert MAX_HTTP_BODY_BYTES == MAX_CONTROL_REQUEST_BYTES
    assert MAX_HTTP_BODY_BYTES == MAX_RESULT_OUTPUT_BYTES + 256 * 1024


def test_global_validation_budget_requires_rotation(tmp_path):
    app = create_app(tmp_path / "validation-budget", bootstrap=False)
    with TestClient(app):
        service = app.state.service
        service.native10.initialize("compact", seed=7)
        config = service.native10.store.model().config
        artifact = synthetic_global_validation_fixture(
            config,
            per_class=5,
            policy=GlobalValidationPolicy(
                min_samples_per_class=5,
                max_candidate_evaluations=1,
            ),
        )
        service.native10.store.set_global_validation(artifact)
        shard = synthetic_representation_shard(config, 0)
        service.native10.queue_mutation(shard, category=0)
        with pytest.raises(ValueError, match="round is already active"):
            service.native10.queue_mutation(shard, category=0)
        assert service.native10.store.validation_status()["candidate_evaluations"] == 1
        service.native10.store.reject_round("not-yet-a-candidate", "test cleanup")
        with pytest.raises(ValueError, match="evaluation budget is exhausted"):
            service.native10.queue_mutation(shard, category=0)


def test_checkpoint_and_validation_cannot_change_during_active_round(tmp_path):
    app = create_app(tmp_path / "active-round-immutability", bootstrap=False)
    with TestClient(app):
        service = app.state.service
        service.native10.initialize("compact", seed=7)
        config = service.native10.store.model().config
        validation = synthetic_global_validation_fixture(config, per_class=5)
        service.native10.store.set_global_validation(validation)
        service.native10.queue_mutation(
            synthetic_representation_shard(config, 0), category=0
        )
        replacement = synthetic_global_validation_fixture(config, per_class=5, seed=999)
        with pytest.raises(ValueError, match="active round"):
            service.native10.store.set_global_validation(replacement, replace=True)
        with pytest.raises(ValueError, match="active round"):
            service.native10.store.import_checkpoint(
                service.native10.store.model().artifact(), replace=True
            )
