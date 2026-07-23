from __future__ import annotations

import json

import numpy as np
from fastapi.testclient import TestClient

from dendriswarm.coordinator.app import create_app
from dendriswarm.core.crypto import Identity
from dendriswarm.v5.native10 import (
    Native10Config,
    Native10Dendritron,
    execute_mutation,
    synthetic_representation_shard,
    verify_mutation,
    verify_mutation_full,
)
from dendriswarm.v5.validation import (
    GlobalValidationPolicy,
    decode_global_validation_artifact,
    synthetic_global_validation_fixture,
)
from tests.helpers import claim_and_execute, materialize, register


def test_native10_exact_profile_preserves_defining_topology():
    config = Native10Config()
    model = Native10Dendritron.initialize(config)
    assert config.field_blocks == 8
    assert config.representation_width == 96
    assert config.scout_count == 1000
    assert config.top_categories == 4
    assert config.categories == 20
    assert config.experts_per_category == 45
    assert config.active_experts_per_update == 15
    assert config.rotation_groups == 3
    assert config.expert_branches == 4
    assert model.parameter_count > 4_800_000
    assert model.lineage[0]["event"] == "topology-initialized"
    assert "No baseline training" in model.lineage[0]["note"]


def test_real_expert_colony_refit_improves_and_composes_exact_delta():
    config = Native10Config.compact_demo()
    model = Native10Dendritron.initialize(config)
    shard = synthetic_representation_shard(config, 0)
    bundle = model.component_bundle("expert_refit", 0)
    result = execute_mutation(
        bundle,
        shard["train_representations"],
        shard["train_labels"],
        shard["validation_representations"],
        shard["validation_labels"],
    )
    validation = synthetic_global_validation_fixture(config, per_class=20)
    validation_x, validation_y, _ = decode_global_validation_artifact(
        validation, expected_config=config
    )
    verified = verify_mutation_full(
        model.artifact(),
        bundle,
        result["delta"],
        validation_x,
        validation_y,
        validation_hash_value=validation["sha256"],
    )
    assert verified["net_wins"] > 0
    assert max(verified["losses_by_class"]) <= 1
    assert result["active_experts"] == [0, 1]
    assert result["rotation_phase_before"] == 0
    assert result["rotation_phase_after"] == 1
    promoted = model.apply_delta(result["delta"], contribution={"trainer_nodes": ["alice"]})
    assert promoted.root != model.root
    assert promoted.lineage[-1]["operation"] == "expert_refit"
    assert promoted.lineage[-1]["trainer_nodes"] == ["alice"]


def test_rotation_covers_entire_exact_colony_in_three_updates():
    config = Native10Config()
    model = Native10Dendritron.initialize(config)
    rng = np.random.default_rng(3)
    start = 0
    train_x = rng.normal(size=(25, config.representation_width)).astype(np.float32)
    train_y = np.repeat(np.arange(config.classes_per_category), 5) + start
    val_x = train_x.copy()
    val_y = train_y.copy()
    seen = set()
    for expected_phase in range(3):
        bundle = model.component_bundle("expert_refit", 0)
        result = execute_mutation(bundle, train_x, train_y, val_x, val_y, subset_seed=11)
        assert result["rotation_phase_before"] == expected_phase
        seen.update(result["active_experts"])
        model = model.apply_delta(result["delta"])
    assert seen == set(range(45))
    assert int(model.tensors["rotation_phase"][0]) == 0


def test_int8_export_is_bound_to_canonical_root():
    model = Native10Dendritron.initialize(Native10Config.compact_demo())
    exported = model.export_int8()
    assert exported["format"] == "dendriswarm.native10-int8.v1"
    assert exported["source_root"] == model.root
    assert exported["accumulation"] == "int32"
    assert exported["boundaries"] == "fp32"
    assert exported["sha256"]


def test_four_volunteer_identities_promote_real_dendritron_tissue(tmp_path):
    app = create_app(tmp_path / "native10-v5", bootstrap=False, lease_seconds=5.0)
    with TestClient(app) as client:
        service = app.state.service
        initialized = service.native10.initialize("compact", seed=7)
        root_before = initialized["canonical_root"]
        queued = service.native10.queue_demo_round(category=0, operation="expert_refit")
        assert len(queued["mutation_tasks"]) == 2

        seeds = [Identity.generate() for _ in range(4)]
        for seed in seeds:
            register(client, seed)

        completed = []
        for seed in seeds:
            result = claim_and_execute(client, seed)
            assert result is not None
            task, _, response = result
            assert response.status_code == 200, response.text
            completed.append((task["kind"], response.json()))

        status = client.get("/v1/native10/status").json()
        assert status["canonical_root"] != root_before
        assert status["contribution_count"] == 1
        contribution = status["latest_contribution"]
        assert contribution["operation"] == "expert_refit"
        assert contribution["category"] == 0
        assert contribution["net_wins"] > 0
        assert contribution["sample_count"] == 240
        assert contribution["samples_by_class"] == [20] * 12
        assert len(contribution["losses_by_class"]) == 12
        assert max(contribution["losses_by_class"]) <= 1
        assert len(contribution["trainer_nodes"]) == 2
        assert len(contribution["verifier_nodes"]) == 2
        assert set(contribution["trainer_nodes"]).isdisjoint(contribution["verifier_nodes"])
        assert contribution["active_experts"] == [0, 1]
        assert any(kind == "dendritron-mutation" for kind, _ in completed)
        assert any(kind == "dendritron-verification" for kind, _ in completed)

        checkpoint = service.native10.store.model()
        assert checkpoint.root == status["canonical_root"]
        assert checkpoint.lineage[-1]["event"] == "verified-tissue-promotion"
        audit = service.db.validate_audit_chain()
        assert audit[0] is True


def test_native10_checkpoint_round_trip(tmp_path):
    model = Native10Dendritron.initialize(Native10Config.compact_demo())
    path = tmp_path / "checkpoint.json"
    path.write_text(json.dumps(model.artifact()))
    restored = Native10Dendritron.from_artifact(json.loads(path.read_text()))
    assert restored.root == model.root
    np.testing.assert_allclose(restored.tensors["expert_branch_weights"], model.tensors["expert_branch_weights"])


def test_native10_file_state_rolls_back_when_db_completion_fails(tmp_path, monkeypatch):
    app = create_app(tmp_path / "native10-atomic", bootstrap=False, lease_seconds=5.0)
    with TestClient(app) as client:
        service = app.state.service
        root_before = service.native10.initialize("compact", seed=7)["canonical_root"]
        service.native10.queue_demo_round(category=0, operation="expert_refit")
        seeds = [Identity.generate() for _ in range(4)]
        for seed in seeds:
            register(client, seed)
        for seed in seeds[:3]:
            _, _, response = claim_and_execute(client, seed)
            assert response.status_code == 200
        claimed = client.post(
            "/v1/tasks/claim",
            json=__import__("tests.helpers", fromlist=["signed_action"]).signed_action(seeds[3], "claim"),
        ).json()["task"]
        from dendriswarm.core.models import TaskKind
        from dendriswarm.worker.executor import execute_task
        output = execute_task(
            TaskKind(claimed["kind"]), materialize(client, claimed["payload"], seeds[3], claimed)
        )
        body = {
            "node_id": seeds[3].node_id,
            "task_id": claimed["id"],
            "lease_token": claimed["lease_token"],
            "duration_ms": 1,
            "output": output,
        }
        body["signature"] = seeds[3].sign(body)
        monkeypatch.setattr(service.db, "complete_task", lambda *args, **kwargs: False)
        response = client.post("/v1/tasks/result", json=body)
        assert response.status_code == 400
        assert service.native10.store.model().root == root_before
        assert service.native10.store.status()["contribution_count"] == 0


def test_npz_import_adapter_round_trip(tmp_path):
    from dendriswarm.v5.native10 import load_external_checkpoint

    model = Native10Dendritron.initialize(Native10Config.compact_demo())
    path = tmp_path / "native10.npz"
    np.savez(path, **model.tensors)
    restored = load_external_checkpoint(path, config=model.config)
    assert restored.root == model.root
    assert restored.lineage[-1]["event"] == "external-checkpoint-imported"


def test_delta_consensus_tolerates_subquantization_architecture_noise():
    from dendriswarm.v5.native10 import decode_array, delta_consensus_hash, delta_hash, encode_array

    config = Native10Config.compact_demo()
    model = Native10Dendritron.initialize(config)
    shard = synthetic_representation_shard(config, 0)
    result = execute_mutation(
        model.component_bundle("expert_refit", 0),
        shard["train_representations"], shard["train_labels"],
        shard["validation_representations"], shard["validation_labels"],
    )
    original = result["delta"]
    noisy = json.loads(json.dumps(original))
    name = "expert_branch_centers"
    value = decode_array(noisy["tensors"][name]).astype(np.float32)
    value.flat[0] += np.float32(1e-7)
    noisy["tensors"][name] = encode_array(value)
    noisy["sha256"] = delta_hash(noisy)
    assert noisy["sha256"] != original["sha256"]
    assert delta_consensus_hash(noisy) == delta_consensus_hash(original)
