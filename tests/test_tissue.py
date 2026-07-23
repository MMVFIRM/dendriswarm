import copy

import numpy as np
import pytest

from dendriswarm.tissues.reference import (
    ReferenceDendritron,
    TissueConfig,
    artifact_hash,
    dataset_split,
    make_digits_dataset,
    reference_benchmark,
    train_and_validation,
)


def trained_artifact():
    dataset = make_digits_dataset()
    x_train, y_train = train_and_validation(dataset)
    config = TissueConfig(branches=160, top_k=3, temperature=0.18, iterations=15, seed=7)
    model = ReferenceDendritron.train(x_train, y_train, config)
    return dataset, model, model.artifact(config.as_dict(), dataset["sha256"])


def test_real_digits_accuracy_sparsity_and_roundtrip():
    dataset, model, artifact = trained_artifact()
    x_test, y_test = dataset_split(dataset, "test")
    score = model.accuracy(x_test, y_test)
    assert score >= 0.98
    assert len(model.centers) == 160
    assert model.activation_fraction == pytest.approx(3 / 160)
    restored = ReferenceDendritron.from_artifact(artifact)
    assert np.array_equal(restored.predict(x_test), model.predict(x_test))


def test_tampered_artifact_is_rejected():
    _, _, artifact = trained_artifact()
    tampered = copy.deepcopy(artifact)
    tampered["centers"][0][0] += 0.1
    with pytest.raises(ValueError, match="hash"):
        ReferenceDendritron.from_artifact(tampered)


def test_nonfinite_rehashed_artifact_is_rejected():
    _, _, artifact = trained_artifact()
    artifact["centers"][0][0] = float("inf")
    with pytest.raises(ValueError):
        artifact["sha256"] = artifact_hash(artifact)


def test_reference_benchmark_has_explicit_claim_boundary():
    report = reference_benchmark()
    assert report["accuracy"]["dendriswarm_reference_tissue"] >= report["accuracy"]["logistic_regression"]
    assert report["activation_fraction"] < 0.01
    assert "no wall-clock speedup claim" in report["claim_boundary"].lower()
