from __future__ import annotations

import copy
import json
import time

import numpy as np
import pytest
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
    load_external_checkpoint,
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
from dendriswarm.worker.executor import execute_task


def _compact(seed: int = 7) -> Native10Dendritron:
    return Native10Dendritron.initialize(Native10Config.compact_demo(seed=seed), seed=seed)


def _training(model: Native10Dendritron, per_class: int = 12):
    raw, labels = synthetic_raw_samples(
        model.config,
        per_class=per_class,
        prototype_seed=20260723,
        sample_seed=20260725,
    )
    return raw, model.encode(raw), labels


def test_exact_profile_and_all_parameter_families_are_reachable():
    config = Native10Config()
    model = Native10Dendritron.initialize(config)
    report = parameter_reachability(config)
    assert model.parameter_count == 4_898_812
    assert report["trainable_float_parameters"] == model.parameter_count
    assert report["reachable_float_parameters"] == model.parameter_count
    assert report["reachable_float_fraction"] == 1.0
    assert report["all_trainable_parameter_families_reachable"] is True
    assert report["all_persistent_tensor_families_owned"] is True
    assert "expert_train" in report["tensors"]["expert_branch_weights"]["operations"]
    assert "scout_train" in report["tensors"]["scout_weights"]["operations"]
    assert "field_train" in report["tensors"]["field_weights"]["operations"]
    assert "memory_train" in report["tensors"]["associative_strength"]["operations"]


@pytest.mark.parametrize(
    ("operation", "changed_tensor"),
    [
        ("field_train", "field_weights"),
        ("scout_train", "scout_weights"),
        ("expert_train", "expert_branch_weights"),
        ("branch_train", "expert_branch_weights"),
        ("repair", "expert_branch_weights"),
        ("memory_train", "associative_strength"),
    ],
)
def test_each_operation_changes_its_real_parameter_family(operation: str, changed_tensor: str):
    model = _compact()
    raw, representations, labels = _training(model, per_class=8)
    target = 0
    if operation == "field_train":
        data, selected_labels = raw, labels
    elif operation == "scout_train":
        data, selected_labels = representations, labels
    else:
        mask = labels < model.config.classes_per_category
        data, selected_labels = representations[mask], labels[mask]
    bundle = model.component_bundle(operation, target)
    output = execute_mutation(
        bundle,
        data,
        selected_labels,
        subset_seed=17,
        optimizer_steps=5,
        learning_rate=0.025,
    )
    validate_delta(output["delta"], bundle)
    assert changed_tensor in {patch["tensor"] for patch in output["delta"]["patches"]}
    assert output["final_loss"] <= output["initial_loss"] + 1e-6
    assert output["changed_parameters"] > 0
    assert model.apply_delta(output["delta"]).root != model.root


def test_operation_specific_schema_rejects_cross_tissue_injection():
    model = _compact()
    raw, representations, labels = _training(model)
    mask = labels < model.config.classes_per_category
    bundle = model.component_bundle("expert_train", 0)
    output = execute_mutation(bundle, representations[mask], labels[mask], optimizer_steps=3)
    malicious = copy.deepcopy(output["delta"])
    memory_bundle = model.component_bundle("memory_train", 0)
    memory_output = execute_mutation(memory_bundle, representations[mask], labels[mask], optimizer_steps=2)
    malicious["patches"].append(copy.deepcopy(memory_output["delta"]["patches"][0]))
    malicious["write_set"].append(memory_output["delta"]["write_set"][0])
    malicious["changed_parameters"] += model.config.classes_per_category * model.config.representation_width
    from dendriswarm.v6.native10 import delta_hash
    malicious["sha256"] = delta_hash(malicious)
    with pytest.raises(ValueError, match="not permitted|schema|tensor"):
        validate_delta(malicious, bundle)


def test_independent_search_seeds_and_hyperparameters_generate_distinct_candidates():
    model = _compact()
    raw, _, labels = _training(model, per_class=20)
    bundle = model.component_bundle("field_train", 0)
    candidates = [
        execute_mutation(bundle, raw, labels, subset_seed=seed, optimizer_steps=steps, learning_rate=rate)["delta"]["sha256"]
        for seed, steps, rate in [(7, 18, 0.04), (1016, 24, 0.035), (2025, 30, 0.03)]
    ]
    assert len(set(candidates)) == len(candidates)


def test_mcnemar_gate_rejects_noise_level_gain_and_accepts_strong_gain():
    assert exact_one_sided_mcnemar(1, 0) == 0.5
    assert exact_one_sided_mcnemar(2, 1) == pytest.approx(0.5)
    assert exact_one_sided_mcnemar(12, 0) < 0.001
    policy = GlobalValidationPolicy(max_candidate_evaluations=20)
    assert exact_one_sided_mcnemar(1, 0) > policy.corrected_alpha
    assert exact_one_sided_mcnemar(12, 0) <= policy.corrected_alpha


def test_full_verification_uses_raw_all_class_inputs_and_reports_every_class():
    model = _compact()
    raw, _, labels = _training(model, per_class=20)
    bundle = model.component_bundle("field_train", 0)
    output = execute_mutation(bundle, raw, labels, subset_seed=7, optimizer_steps=60, learning_rate=0.035)
    validation_x, validation_y = synthetic_raw_samples(
        model.config, per_class=16, prototype_seed=20260723, sample_seed=20260724
    )
    policy = GlobalValidationPolicy(
        min_samples_per_class=5,
        max_candidate_evaluations=20,
        min_discordant=5,
        minimum_net_wins=1,
        minimum_effect_rate=0.0,
        max_loss_per_class=8,
        max_loss_rate_per_class=0.5,
    )
    evidence = verify_mutation_full(
        model.artifact(),
        bundle,
        output["delta"],
        validation_x,
        validation_y,
        validation_hash_value="0" * 64,
        validation_policy=policy,
    )
    assert evidence["sample_count"] == model.config.classes * 16
    assert evidence["samples_by_class"] == [16] * model.config.classes
    assert len(evidence["losses_by_class"]) == model.config.classes
    assert evidence["mcnemar_p_value"] == pytest.approx(
        exact_one_sided_mcnemar(evidence["wins"], evidence["losses"])
    )


def test_non_conflicting_category_deltas_compose_and_overlap_is_rejected():
    model = _compact()
    _, representations, labels = _training(model, per_class=10)
    deltas = []
    for category in (0, 1):
        mask = labels // model.config.classes_per_category == category
        bundle = model.component_bundle("memory_train", category)
        deltas.append(execute_mutation(bundle, representations[mask], labels[mask], optimizer_steps=3)["delta"])
    assert not deltas_conflict(deltas[0], deltas[1])
    composed = compose_non_conflicting_deltas(model, deltas)
    assert composed.root != model.root
    with pytest.raises(ValueError, match="conflict|overlapping"):
        compose_non_conflicting_deltas(model, [deltas[0], deltas[0]])


def test_sparse_delta_is_smaller_than_full_component_bundle():
    model = _compact()
    _, representations, labels = _training(model, per_class=10)
    mask = labels < model.config.classes_per_category
    bundle = model.component_bundle("expert_train", 0)
    delta = execute_mutation(bundle, representations[mask], labels[mask], optimizer_steps=4)["delta"]
    bundle_size = len(json.dumps(bundle, separators=(",", ":")).encode())
    delta_size = len(json.dumps(delta, separators=(",", ":")).encode())
    assert delta_size < bundle_size


def test_v5_checkpoint_adapter_adds_learnable_memory_strength(tmp_path):
    from dendriswarm.v5.native10 import Native10Config as V5Config, Native10Dendritron as V5Model

    old = V5Model.initialize(V5Config.compact_demo(seed=7), seed=7)
    path = tmp_path / "v5.json"
    path.write_text(json.dumps(old.artifact()))
    converted = load_external_checkpoint(path)
    assert converted.config.classes == old.config.classes
    assert converted.tensors["associative_strength"].shape == (converted.config.classes,)
    assert np.allclose(converted.tensors["associative_strength"], converted.config.memory_strength_init)


def _register(client: TestClient, identity: Identity) -> None:
    value = {
        "node_id": identity.node_id,
        "public_key": identity.public_key_b64,
        "capabilities": {
            "cpu_count": 2,
            "memory_mb": 4096,
            "memory_available_mb": 4096,
            "disk_free_mb": 10000,
            "accelerator": "cpu",
            "platform": "pytest",
            "machine": "x86_64",
            "supported_backends": ["numpy-cpu"],
            "tags": [
                "portable-numpy-v1",
                "independent-search-v1",
                "blind-global-verification-v2",
                "deterministic-v2",
            ],
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
    assert response.status_code == 200, response.text


def _signed(identity: Identity, action: str) -> dict:
    body = {"action": action, "node_id": identity.node_id, "timestamp": int(time.time()), "nonce": nonce()}
    return {key: value for key, value in body.items() if key != "action"} | {"signature": identity.sign(body)}


def _execute_one(client: TestClient, identity: Identity):
    response = client.post("/v1/tasks/claim", json=_signed(identity, "claim"))
    if response.status_code == 204:
        return None
    assert response.status_code == 200, response.text
    envelope = response.json()
    assert verify(envelope["coordinator_public_key"], envelope["task"], envelope["signature"])
    task = envelope["task"]
    payload = dict(task["payload"])
    if "native10_checkpoint_root" in payload:
        checkpoint = client.get(f"/v1/native10-v6/checkpoints/{payload['native10_checkpoint_root']}")
        assert checkpoint.status_code == 200, checkpoint.text
        payload["_native10_checkpoint"] = checkpoint.json()
    if "global_validation_hash" in payload and task["kind"] == TaskKind.DENDRITRON_VERIFICATION.value:
        action = {
            "action": "fetch-native10-validation",
            "node_id": identity.node_id,
            "task_id": task["id"],
            "lease_token": task["lease_token"],
            "timestamp": int(time.time()),
            "nonce": nonce(),
        }
        request = {key: value for key, value in action.items() if key != "action"}
        request["signature"] = identity.sign(action)
        validation = client.post(
            f"/v1/native10-v6/validation/{payload['global_validation_hash']}", json=request
        )
        assert validation.status_code == 200, validation.text
        payload["_native10_validation"] = validation.json()
    output = execute_task(TaskKind(task["kind"]), payload)
    body = {
        "node_id": identity.node_id,
        "task_id": task["id"],
        "lease_token": task["lease_token"],
        "duration_ms": 1,
        "output": output,
    }
    body["signature"] = identity.sign(body)
    result = client.post("/v1/tasks/result", json=body)
    assert result.status_code == 200, result.text
    return task, result.json()


def test_end_to_end_independent_search_selection_replay_and_fresh_replication(tmp_path):
    app = create_app(tmp_path)
    service = app.state.service
    service.native10_v6.initialize("compact", seed=7)
    root_before = service.native10_v6.store.model().root
    service.native10_v6.queue_demo_round(category=0, operation="field_train")
    identities = [Identity.generate() for _ in range(16)]
    with TestClient(app) as client:
        for identity in identities:
            _register(client, identity)
        observed = []
        for _ in range(40):
            progressed = False
            for identity in identities:
                item = _execute_one(client, identity)
                if item is not None:
                    progressed = True
                    observed.append(item[1])
            if not progressed:
                break
    status = service.native10_v6.store.status()
    assert status["canonical_root"] != root_before
    assert status["contribution_count"] == 1
    contribution = status["latest_contribution"]
    assert contribution["selection_evidence"]["validation_hash"] != contribution["replication_evidence"]["validation_hash"]
    assert contribution["selection_evidence"]["mcnemar_p_value"] <= 0.05 / 20
    assert contribution["replication_evidence"]["mcnemar_p_value"] <= 0.05 / 20
    state = service.native10_v6.store.state()
    selected = state["candidates"][contribution["candidate_id"]]
    assert selected["trainer_nodes"]
    assert set(selected["trainer_nodes"]).isdisjoint(selected["verifier_nodes"])
    assert selected["replay_node"] not in set(selected["trainer_nodes"]) | set(selected["verifier_nodes"])
    assert set(selected["replication_verifier_nodes"]).isdisjoint(
        set(selected["trainer_nodes"]) | set(selected["verifier_nodes"]) | {selected["replay_node"]}
    )
    stages = {value.get("stage") for value in observed}
    assert {"independent-search", "blind-selection-verification", "replay-audit", "fresh-final-replication"} <= stages


def test_status_reports_independent_search_worker_progress(tmp_path):
    app = create_app(tmp_path)
    native = app.state.service.native10_v6
    native.initialize("compact", seed=7)
    native.queue_demo_round(category=0, operation="field_train")
    active = native.store.status()["active_round"]
    app.state.service.db.record_work_report(
        active["work_key"],
        TaskKind.DENDRITRON_MUTATION.value,
        "worker-one",
        "task-one",
        {"worker_duration_ms": 1234},
    )

    status = native.status()
    progress = status["active_round"]
    assert progress["search_reports_received"] == 1
    assert progress["search_reports_remaining"] == progress["expected_search_reports"] - 1
    assert progress["search_report_nodes"] == ["worker-one"]


def test_selection_and_replication_artifacts_must_be_distinct(tmp_path):
    app = create_app(tmp_path)
    native = app.state.service.native10_v6
    native.initialize("compact", seed=7)
    model = native.store.model()
    x, y = synthetic_raw_samples(model.config, per_class=5)
    policy = GlobalValidationPolicy(
        min_samples_per_class=5,
        max_candidate_evaluations=5,
        min_discordant=2,
        minimum_net_wins=1,
        minimum_effect_rate=0,
        max_loss_per_class=2,
        max_loss_rate_per_class=0.5,
    )
    artifact = make_global_validation_artifact(model.config, x, y, source="test", policy=policy)
    native.store.set_global_validation(artifact)
    with pytest.raises(ValueError, match="distinct"):
        native.store.set_replication_validation(artifact)


def test_coordinator_recomputes_complete_verifier_evidence(tmp_path):
    app = create_app(tmp_path)
    native = app.state.service.native10_v6
    native.initialize("compact", seed=7)
    model = native.store.model()
    train_x, train_y = synthetic_raw_samples(
        model.config, per_class=12, prototype_seed=20260723, sample_seed=20260725
    )
    bundle = model.component_bundle("field_train", 0)
    output = execute_mutation(bundle, train_x, train_y, subset_seed=7, optimizer_steps=20, learning_rate=0.035)
    validation_x, validation_y = synthetic_raw_samples(
        model.config, per_class=12, prototype_seed=20260723, sample_seed=20260724
    )
    policy = GlobalValidationPolicy(
        min_samples_per_class=5,
        max_candidate_evaluations=20,
        min_discordant=2,
        minimum_net_wins=1,
        minimum_effect_rate=0.0,
        max_loss_per_class=8,
        max_loss_rate_per_class=0.5,
    )
    artifact = make_global_validation_artifact(
        model.config, validation_x, validation_y, source="test", policy=policy
    )
    native.store.set_global_validation(artifact)
    evidence = verify_mutation_full(
        model.artifact(), bundle, output["delta"], validation_x, validation_y,
        validation_hash_value=artifact["sha256"], validation_policy=policy,
    )
    candidate = {"operation": "field_train", "category": 0}
    # The valid evidence is internally recomputed even when it does not pass the
    # configured practical gate.
    native._verification_gate(evidence, artifact, candidate)

    forged_rate = copy.deepcopy(evidence)
    forged_rate["loss_rates_by_class"] = [0.0] * model.config.classes
    if any(evidence["losses_by_class"]):
        with pytest.raises(ValueError, match="loss rates"):
            native._verification_gate(forged_rate, artifact, candidate)

    forged_operation = copy.deepcopy(evidence)
    forged_operation["operation"] = "memory_train"
    with pytest.raises(ValueError, match="operation"):
        native._verification_gate(forged_operation, artifact, candidate)

    forged_counts = copy.deepcopy(evidence)
    forged_counts["samples_by_class"][0] += 1
    with pytest.raises(ValueError, match="class counts|sample count"):
        native._verification_gate(forged_counts, artifact, candidate)


def test_external_baseline_reference_and_evaluation_are_provenance_bound(tmp_path):
    from dendriswarm.v6.benchmark import (
        compare_with_baseline,
        evaluate_checkpoint,
        make_baseline_reference,
        validate_baseline_reference,
        validate_evaluation_report,
    )

    model = _compact()
    inputs, labels = synthetic_raw_samples(
        model.config, per_class=3, prototype_seed=13, sample_seed=14
    )
    evaluation = evaluate_checkpoint(
        model, inputs, labels, dataset="fixture-12", split="test", source="pytest fixture"
    )
    validate_evaluation_report(evaluation)
    baseline = make_baseline_reference(
        dataset="fixture-12",
        split="test",
        metric="accuracy",
        value=max(0.0, float(evaluation["value"]) - 0.01),
        model="established external control",
        source="external report",
        evidence_sha256="a" * 64,
        notes="Import-only reference; no baseline trainer is packaged.",
    )
    validate_baseline_reference(baseline)
    comparison = compare_with_baseline(evaluation, baseline)
    assert comparison["baseline_training_included"] is False
    assert comparison["meets_or_exceeds_baseline"] is True

    app = create_app(tmp_path / "coordinator")
    native = app.state.service.native10_v6
    native.initialize("compact", seed=7)
    status = native.store.set_baseline_reference(baseline)
    assert status["configured"] is True
    assert status["evidence_sha256"] == "a" * 64
    assert status["training_code_included"] is False

    tampered = copy.deepcopy(baseline)
    tampered["value"] += 0.1
    with pytest.raises(ValueError, match="inconsistent"):
        validate_baseline_reference(tampered)


def test_hidden_artifacts_are_one_tournament_only(tmp_path):
    app = create_app(tmp_path)
    native = app.state.service.native10_v6
    native.initialize("compact", seed=7)
    native.queue_demo_round(category=0, operation="field_train")
    selection_status = native.store.validation_status()
    replication_status = native.store.replication_validation_status()
    assert selection_status["search_rounds"] == 1
    assert replication_status["search_rounds"] == 1
    assert selection_status["max_search_rounds"] == 1
    assert replication_status["max_search_rounds"] == 1
    native.store.close_without_promotion("pytest closes the first tournament")
    with pytest.raises(ValueError, match="tournament is exhausted|one-shot"):
        native.queue_demo_round(category=0, operation="field_train")


def test_status_polling_does_not_materialize_canonical_model_tensors(tmp_path, monkeypatch):
    app = create_app(tmp_path)
    native = app.state.service.native10_v6
    native.initialize("compact", seed=7)
    native.queue_demo_round(category=0, operation="field_train")

    def fail_model_load():
        raise AssertionError("status polling decoded the canonical model")

    monkeypatch.setattr(native.store, "model", fail_model_load)
    status = native.store.status()
    assert status["initialized"] is True
    assert status["parameter_count"] > 0
    assert native.store.validation_status()["configured"] is True
    assert native.store.replication_validation_status()["configured"] is True


def test_exact_operation_write_plan_rejects_missing_allowed_slice():
    from dendriswarm.v6.native10 import delta_hash

    model = _compact()
    _, representations, labels = _training(model, per_class=8)
    mask = labels < model.config.classes_per_category
    bundle = model.component_bundle("expert_train", 0)
    output = execute_mutation(bundle, representations[mask], labels[mask], optimizer_steps=3)
    incomplete = copy.deepcopy(output["delta"])
    remove_index = next(
        index for index, patch in enumerate(incomplete["patches"])
        if patch["tensor"] == "expert_health"
    )
    removed = incomplete["patches"].pop(remove_index)
    removed_key = f"{removed['tensor']}:" + json.dumps(removed["selector"], separators=(",", ":"))
    incomplete["write_set"].remove(removed_key)
    from dendriswarm.v6.native10 import decode_array
    incomplete["changed_parameters"] -= int(decode_array(removed["value"]).size)
    incomplete["sha256"] = delta_hash(incomplete)
    with pytest.raises(ValueError, match="exact operation write plan"):
        validate_delta(incomplete, bundle)
