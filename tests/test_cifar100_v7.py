from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from dendriswarm.coordinator.service import CoordinatorService
from dendriswarm.core.crypto import content_hash
from dendriswarm.v6.native10 import Native10Config, Native10Dendritron, encode_array
from dendriswarm.v6.validation import GlobalValidationPolicy, make_global_validation_artifact, synthetic_raw_samples
from dendriswarm.v7.cifar100 import (
    CIFAR100DatasetStore,
    DATASET_FORMAT,
    _array_digest,
    _native_label_mapping,
    _stratified_split,
)
from dendriswarm.v7.routing import ROUTING_REPORT_FORMAT, plan_next_round, routing_gap_report, search_recipes


def _prepared_small_store(path: Path) -> CIFAR100DatasetStore:
    path.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(9)
    split_counts = {"train": 1000, "selection": 200, "replication": 200, "test": 100}
    arrays = {}
    counts_by_class = {}
    for split, count in split_counts.items():
        per_class = count // 100
        labels = np.repeat(np.arange(100, dtype=np.int16), per_class)
        images = rng.integers(0, 256, size=(count, 3072), dtype=np.uint8)
        counts_by_class[split] = np.bincount(labels, minlength=100).astype(int).tolist()
        for suffix, value in (("images", images), ("labels", labels)):
            filename = f"{split}-{suffix}.npy"
            with (path / filename).open("wb") as handle:
                np.save(handle, value, allow_pickle=False)
            arrays[f"{split}_{suffix}"] = {
                "file": filename,
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "sha256": _array_digest(value),
                "file_sha256": __import__("hashlib").sha256((path / filename).read_bytes()).hexdigest(),
            }
    value = {
        "format": DATASET_FORMAT,
        "dataset": "CIFAR-100",
        "official_url": "fixture",
        "official_archive_md5": "fixture",
        "observed_archive_md5": None,
        "source_kind": "test-fixture",
        "source_sha256": "0" * 64,
        "seed": 9,
        "input_width": 3072,
        "classes": 100,
        "categories": 20,
        "classes_per_category": 5,
        "split_counts": split_counts,
        "counts_by_class": counts_by_class,
        "normalization": {
            "source_layout": "channel-major-r1024-g1024-b1024",
            "model_layout": "spatial-patches-4x2-c3-h8-w16",
            "scale": 255.0,
            "channel_mean": [0.5, 0.5, 0.5],
            "channel_std": [0.25, 0.25, 0.25],
        },
        "fine_label_names_official": [f"official-{i}" for i in range(100)],
        "coarse_label_names": [f"coarse-{i}" for i in range(20)],
        "official_to_native": list(range(100)),
        "native_to_official": list(range(100)),
        "fine_labels_by_coarse_category": [list(range(i * 5, i * 5 + 5)) for i in range(20)],
        "native_label_names": [f"official-{i}" for i in range(100)],
        "arrays": arrays,
        "test_used_for_selection": False,
    }
    value["sha256"] = content_hash(value)
    (path / "manifest.json").write_text(json.dumps(value, sort_keys=True, separators=(",", ":")))
    return CIFAR100DatasetStore(path)


def test_official_fine_labels_are_regrouped_by_coarse_category():
    train_fine = np.tile(np.arange(100), 5)
    train_coarse = (train_fine * 7 % 100) // 5
    # Build a valid but non-contiguous fine->coarse assignment.
    mapping = np.arange(100).reshape(20, 5)[:, ::-1].reshape(-1)
    coarse_for_fine = np.empty(100, dtype=np.int64)
    for coarse, group in enumerate(mapping.reshape(20, 5)):
        coarse_for_fine[group] = coarse
    train_coarse = coarse_for_fine[train_fine]
    test_fine = np.arange(100)
    test_coarse = coarse_for_fine[test_fine]
    official_to_native, native_to_official, grouped = _native_label_mapping(
        train_fine, train_coarse, test_fine, test_coarse
    )
    assert len(grouped) == 20 and all(len(group) == 5 for group in grouped)
    assert np.array_equal((official_to_native[train_fine] // 5), train_coarse)
    assert np.array_equal(native_to_official[official_to_native], np.arange(100))


def test_stratified_campaign_split_is_450_25_25_per_class():
    labels = np.repeat(np.arange(100), 500)
    train, selection, replication = _stratified_split(labels, seed=11)
    assert len(train) == 45_000 and len(selection) == len(replication) == 2_500
    assert np.all(np.bincount(labels[train], minlength=100) == 450)
    assert np.all(np.bincount(labels[selection], minlength=100) == 25)
    assert np.all(np.bincount(labels[replication], minlength=100) == 25)
    assert not (set(train) & set(selection) or set(train) & set(replication) or set(selection) & set(replication))


def test_real_cifar_store_builds_bounded_encoded_training_shards(tmp_path):
    store = _prepared_small_store(tmp_path / "dataset")
    model = Native10Dendritron.initialize(Native10Config(seed=3))
    field = store.training_shard(model, operation="field_train", target=0, sample_budget=100, seed=4)
    assert field["sample_count"] == 100
    assert field["train_inputs"]["format"] == "dendriswarm.cifar100-patch-input.v1"
    assert field["train_inputs"]["array"]["shape"] == [100, 3072]
    scout = store.training_shard(
        model, operation="scout_train", target=0, sample_budget=100, seed=5, hard_negative_fraction=0.5
    )
    labels = np.asarray(scout["train_labels"])
    assert np.any(labels // 5 == 0) and np.any(labels // 5 != 0)
    assert scout["train_representations"]["shape"] == [100, 96]
    assert len(json.dumps(scout)) < 2_000_000


def test_routing_gap_report_measures_oracle_and_actual_paths():
    model = Native10Dendritron.initialize(Native10Config(seed=6))
    rng = np.random.default_rng(7)
    x = rng.normal(0, 1, size=(100, 3072)).astype(np.float32)
    y = np.arange(100, dtype=np.int64)
    report = routing_gap_report(
        model, x, y, dataset_sha256="1" * 64, split="campaign-train-diagnostic", sample_source="unit"
    )
    assert report["sample_count"] == 100
    assert len(report["categories"]) == 20
    assert 0.0 <= report["topk_category_recall"]["4"] <= 1.0
    assert report["oracle_category_correct"] >= 0
    assert report["test_selection_forbidden"] is False


def test_planner_targets_routing_gap_with_distinct_search_recipes():
    categories = [
        {
            "category": i, "samples": 50, "top1_recall": 0.2, "top4_recall": 0.6,
            "expanded_recall": 0.7 if i else 0.2, "route_misses": 15 if i else 40,
            "actual_correct": 10, "oracle_category_correct": 30, "actual_accuracy": 0.2,
            "oracle_category_accuracy": 0.6, "routing_gap": 0.4,
            "conditional_accuracy_when_routed": 0.3,
        }
        for i in range(20)
    ]
    report = {
        "format": ROUTING_REPORT_FORMAT, "model_root": "2" * 64, "dataset_sha256": "3" * 64,
        "split": "campaign-train-diagnostic", "sample_source": "unit", "sample_count": 1000,
        "actual_correct": 200, "actual_accuracy": 0.2, "oracle_category_correct": 600,
        "oracle_category_accuracy": 0.6, "oracle_routing_gap": 0.4,
        "topk_category_recall": {"1": 0.2, "2": 0.4, "4": 0.6, "8": 0.8, "20": 1.0},
        "expanded_category_recall": 0.7, "route_miss_count": 300, "route_miss_accuracy": 0.0,
        "conditional_accuracy_when_routed": 0.3, "average_routed_categories": 4.0,
        "mean_top_category_margin": 0.1, "median_correct_category_rank": 4.0,
        "p95_correct_category_rank": 15.0, "categories": categories,
        "labels_used_for_training": True, "promotion_holdout": False, "test_selection_forbidden": False,
    }
    report["sha256"] = content_hash(report)
    plan = plan_next_round(report, round_index=0, search_candidates=8, sample_budget=640)
    assert plan["objective"] == "close-routing-gap"
    assert plan["operation"] == "scout_train" and plan["target"] == 0
    recipes = plan["recipes"]
    assert len(recipes) == 8 and len({content_hash(value) for value in recipes}) == 8


def test_v6_queue_accepts_compressed_cifar_style_shard_and_recipes(tmp_path):
    service = CoordinatorService(tmp_path)
    native = service.native10_v6
    native.initialize("compact", seed=7)
    model = native.store.model()
    selection_x, selection_y = synthetic_raw_samples(model.config, per_class=5, sample_seed=41)
    replication_x, replication_y = synthetic_raw_samples(model.config, per_class=5, sample_seed=42)
    policy = GlobalValidationPolicy(
        min_samples_per_class=5, max_candidate_evaluations=2, min_discordant=2,
        minimum_net_wins=1, minimum_effect_rate=0.0, max_loss_per_class=2,
        max_loss_rate_per_class=0.4,
    )
    native.store.set_global_validation(make_global_validation_artifact(
        model.config, selection_x, selection_y, source="selection", policy=policy
    ))
    native.store.set_replication_validation(make_global_validation_artifact(
        model.config, replication_x, replication_y, source="replication", policy=policy
    ))
    train_x, train_y = synthetic_raw_samples(model.config, per_class=4, sample_seed=43)
    shard = {
        "category": 0,
        "train_inputs": encode_array(train_x),
        "train_labels": train_y.tolist(),
    }
    recipes = search_recipes("field_train", 2)
    queued = native.queue_mutation(
        shard, operation="field_train", category=0, search_candidates=2,
        verification_quorum=2, optimizer_steps=4, learning_rate=0.01,
        search_recipes=recipes,
    )
    tasks = [service.db.task(task_id) for task_id in queued["search_tasks"]]
    payloads = [json.loads(task["payload"]) for task in tasks]
    assert all(payload["train_data"]["codec"] == "npy+zlib+base64" for payload in payloads)
    assert [payload["search_recipe"] for payload in payloads] == recipes


def test_cifar_patch_adapter_maps_each_field_block_to_one_spatial_region(tmp_path):
    store = _prepared_small_store(tmp_path / "dataset")
    # Each of the eight 4x2 spatial patches gets a distinct constant.  Channel
    # means are zeroed so block means reveal the patch ordering directly.
    manifest = store.manifest()
    manifest["normalization"]["channel_mean"] = [0.0, 0.0, 0.0]
    manifest["normalization"]["channel_std"] = [1.0, 1.0, 1.0]
    manifest["sha256"] = content_hash({k: v for k, v in manifest.items() if k != "sha256"})
    store.manifest_path.write_text(json.dumps(manifest, sort_keys=True, separators=(",", ":")))
    image = np.zeros((1, 3, 32, 32), dtype=np.uint8)
    value = 10
    expected = []
    for row in range(4):
        for column in range(2):
            image[:, :, row * 8:(row + 1) * 8, column * 16:(column + 1) * 16] = value
            expected.append(value / 255.0)
            value += 10
    adapted = store.normalize_images(image.reshape(1, 3072)).reshape(1, 8, 384)
    assert np.allclose(adapted.mean(axis=2)[0], expected, atol=1e-7)


def test_dataset_array_tampering_is_detected_before_use(tmp_path):
    store = _prepared_small_store(tmp_path / "dataset")
    path = store.path / store.manifest()["arrays"]["train_labels"]["file"]
    data = bytearray(path.read_bytes())
    data[-1] ^= 1
    path.write_bytes(data)
    with np.testing.assert_raises_regex(ValueError, "file hash mismatch"):
        store.labels("train")


def test_field_shard_uses_compact_uint8_transport_and_decodes_deterministically(tmp_path):
    from dendriswarm.v6.native10 import decode_training_tensor

    store = _prepared_small_store(tmp_path / "dataset")
    model = Native10Dendritron.initialize(Native10Config(seed=3))
    shard = store.training_shard(model, operation="field_train", target=0, sample_budget=100, seed=44)
    encoded = shard["train_inputs"]
    decoded = decode_training_tensor(encoded)
    assert decoded.shape == (100, 3072)
    assert decoded.dtype == np.float32 and np.isfinite(decoded).all()
    # Expanded float32 JSON/base64 would be materially larger than this uint8 payload.
    assert len(json.dumps(encoded)) < 500_000


def test_test_split_routing_report_cannot_drive_planner():
    categories = [
        {
            "category": i, "samples": 5, "top1_recall": 0.2, "top4_recall": 0.6,
            "expanded_recall": 0.7, "route_misses": 1, "actual_correct": 1,
            "oracle_category_correct": 2, "actual_accuracy": 0.2,
            "oracle_category_accuracy": 0.4, "routing_gap": 0.2,
            "conditional_accuracy_when_routed": 0.25,
        }
        for i in range(20)
    ]
    report = {
        "format": ROUTING_REPORT_FORMAT, "model_root": "2" * 64, "dataset_sha256": "3" * 64,
        "split": "test", "sample_source": "forbidden", "sample_count": 100,
        "actual_correct": 20, "actual_accuracy": 0.2, "oracle_category_correct": 40,
        "oracle_category_accuracy": 0.4, "oracle_routing_gap": 0.2,
        "topk_category_recall": {"1": 0.2, "2": 0.4, "4": 0.6, "8": 0.8, "20": 1.0},
        "expanded_category_recall": 0.7, "route_miss_count": 30, "route_miss_accuracy": 0.0,
        "conditional_accuracy_when_routed": 0.25, "average_routed_categories": 4.0,
        "mean_top_category_margin": 0.1, "median_correct_category_rank": 4.0,
        "p95_correct_category_rank": 15.0, "categories": categories,
        "labels_used_for_training": False, "promotion_holdout": False, "test_selection_forbidden": True,
    }
    report["sha256"] = content_hash(report)
    with np.testing.assert_raises_regex(ValueError, "test split"):
        plan_next_round(report, round_index=0)


def test_v6_mutation_schema_rejects_tampered_recipe_hash():
    from dendriswarm.core.models import DendritronV6MutationOutput
    from dendriswarm.v6.native10 import execute_mutation, synthetic_representation_shard

    model = Native10Dendritron.initialize(Native10Config.compact_demo(seed=7))
    bundle = model.component_bundle("expert_train", 0)
    shard = synthetic_representation_shard(model.config, 0, train_per_class=4, validation_per_class=2, seed=8)
    x = np.asarray(shard["train_representations"], dtype=np.float32)
    y = np.asarray(shard["train_labels"], dtype=np.int64)
    output = execute_mutation(bundle, x, y, optimizer_steps=2, search_recipe={"expert_diversity": 0.01})
    output["runtime"] = {"backend": "numpy-cpu", "machine": "x86_64", "python": "3.13", "cpu_threads": 1}
    output["search_recipe"] = {"expert_diversity": 0.99}
    with np.testing.assert_raises_regex(ValueError, "recipe hash mismatch"):
        DendritronV6MutationOutput.model_validate(output)
