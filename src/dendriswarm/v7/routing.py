from __future__ import annotations

import math
from typing import Any

import numpy as np

from dendriswarm.core.crypto import content_hash
from dendriswarm.v6.native10 import Native10Dendritron

ROUTING_REPORT_FORMAT = "dendriswarm.cifar100-routing-gap-report.v1"
PLAN_FORMAT = "dendriswarm.cifar100-round-plan.v1"


def _expanded_categories(model: Native10Dendritron, route: np.ndarray) -> list[list[int]]:
    selected = model.top_categories_from_scores(route) if hasattr(model, "top_categories_from_scores") else None
    if selected is None:
        k = model.config.top_categories
        base = np.argpartition(route, kth=model.config.categories - k, axis=1)[:, -k:]
        order = np.take_along_axis(route, base, axis=1).argsort(axis=1)[:, ::-1]
        selected = np.take_along_axis(base, order, axis=1)
    full_order = np.argsort(route, axis=1)[:, ::-1]
    result: list[list[int]] = []
    for row in range(len(route)):
        chosen = list(np.asarray(selected[row], dtype=np.int64).astype(int))
        boundary = float(route[row, chosen[-1]])
        for candidate in full_order[row]:
            category = int(candidate)
            if category in chosen:
                continue
            if len(chosen) >= model.config.max_routed_categories:
                break
            if boundary - float(route[row, category]) <= model.config.routing_expansion_margin:
                chosen.append(category)
            else:
                break
        result.append(chosen)
    return result


def routing_gap_report(
    model: Native10Dendritron,
    inputs: np.ndarray,
    labels: np.ndarray,
    *,
    dataset_sha256: str,
    split: str,
    sample_source: str,
    chunk_size: int = 128,
) -> dict[str, Any]:
    x = np.asarray(inputs, dtype=np.float32)
    y = np.asarray(labels, dtype=np.int64)
    if x.ndim != 2 or x.shape[1] != model.config.input_width or not len(x):
        raise ValueError("routing report inputs have the wrong shape")
    if y.shape != (len(x),) or np.any(y < 0) or np.any(y >= model.config.classes):
        raise ValueError("routing report labels are invalid")
    if not np.isfinite(x).all():
        raise ValueError("routing report inputs contain non-finite values")

    correct_category = y // model.config.classes_per_category
    actual_predictions: list[np.ndarray] = []
    route_scores_rows: list[np.ndarray] = []
    oracle_predictions: list[np.ndarray] = []
    routed_sets: list[list[int]] = []
    for start in range(0, len(x), max(1, int(chunk_size))):
        stop = min(len(x), start + max(1, int(chunk_size)))
        h = model.encode(x[start:stop])
        route = model.route_scores(h)
        route_scores_rows.append(route)
        actual_predictions.append(model.predict_from_representation(h))
        routed_sets.extend(_expanded_categories(model, route))
        local_predictions = np.empty(stop - start, dtype=np.int64)
        categories = correct_category[start:stop]
        for category in np.unique(categories):
            mask = categories == category
            logits = model.category_logits(h[mask], int(category))
            local_predictions[mask] = int(category) * model.config.classes_per_category + logits.argmax(axis=1)
        oracle_predictions.append(local_predictions)

    route = np.concatenate(route_scores_rows, axis=0)
    actual = np.concatenate(actual_predictions)
    oracle = np.concatenate(oracle_predictions)
    order = np.argsort(route, axis=1)[:, ::-1]
    ranks = np.empty_like(order)
    row_index = np.arange(len(order))[:, None]
    ranks[row_index, order] = np.arange(model.config.categories)[None, :]
    correct_rank = ranks[np.arange(len(y)), correct_category]
    routed_mask = np.asarray(
        [int(category) in routed_sets[index] for index, category in enumerate(correct_category)],
        dtype=bool,
    )
    actual_correct = actual == y
    oracle_correct = oracle == y

    category_rows: list[dict[str, Any]] = []
    for category in range(model.config.categories):
        mask = correct_category == category
        samples = int(mask.sum())
        if not samples:
            raise ValueError(f"routing report has no samples for category {category}")
        routed = int(routed_mask[mask].sum())
        actual_count = int(actual_correct[mask].sum())
        oracle_count = int(oracle_correct[mask].sum())
        category_rows.append({
            "category": category,
            "samples": samples,
            "top1_recall": float((correct_rank[mask] < 1).mean()),
            "top4_recall": float((correct_rank[mask] < min(4, model.config.categories)).mean()),
            "expanded_recall": routed / samples,
            "route_misses": samples - routed,
            "actual_correct": actual_count,
            "oracle_category_correct": oracle_count,
            "actual_accuracy": actual_count / samples,
            "oracle_category_accuracy": oracle_count / samples,
            "routing_gap": (oracle_count - actual_count) / samples,
            "conditional_accuracy_when_routed": (
                float(actual_correct[mask & routed_mask].mean()) if np.any(mask & routed_mask) else 0.0
            ),
        })

    topk_recall = {
        str(k): float((correct_rank < min(k, model.config.categories)).mean())
        for k in (1, 2, 4, 8, 20)
    }
    margins = np.sort(route, axis=1)[:, ::-1]
    top_margin = margins[:, 0] - margins[:, 1]
    report: dict[str, Any] = {
        "format": ROUTING_REPORT_FORMAT,
        "model_root": model.root,
        "dataset_sha256": str(dataset_sha256),
        "split": str(split),
        "sample_source": str(sample_source),
        "sample_count": int(len(y)),
        "actual_correct": int(actual_correct.sum()),
        "actual_accuracy": float(actual_correct.mean()),
        "oracle_category_correct": int(oracle_correct.sum()),
        "oracle_category_accuracy": float(oracle_correct.mean()),
        "oracle_routing_gap": float(oracle_correct.mean() - actual_correct.mean()),
        "topk_category_recall": topk_recall,
        "expanded_category_recall": float(routed_mask.mean()),
        "route_miss_count": int((~routed_mask).sum()),
        "route_miss_accuracy": float(actual_correct[~routed_mask].mean()) if np.any(~routed_mask) else 0.0,
        "conditional_accuracy_when_routed": float(actual_correct[routed_mask].mean()) if np.any(routed_mask) else 0.0,
        "average_routed_categories": float(np.mean([len(value) for value in routed_sets])),
        "mean_top_category_margin": float(top_margin.mean()),
        "median_correct_category_rank": float(np.median(correct_rank)),
        "p95_correct_category_rank": float(np.percentile(correct_rank, 95)),
        "categories": category_rows,
        "labels_used_for_training": True,
        "promotion_holdout": False,
        "test_selection_forbidden": split == "test",
    }
    report["sha256"] = content_hash(report)
    return report


def validate_routing_report(report: dict[str, Any]) -> dict[str, Any]:
    if report.get("format") != ROUTING_REPORT_FORMAT:
        raise ValueError("unsupported routing report format")
    expected = content_hash({key: value for key, value in report.items() if key != "sha256"})
    if report.get("sha256") != expected:
        raise ValueError("routing report hash mismatch")
    if int(report.get("sample_count", 0)) < 100:
        raise ValueError("routing report is too small")
    if len(report.get("categories", [])) != 20:
        raise ValueError("routing report must cover all 20 categories")
    return dict(report)


def search_recipes(operation: str, count: int) -> list[dict[str, Any]]:
    if count < 2 or count > 32:
        raise ValueError("candidate count must be between 2 and 32")
    values: list[dict[str, Any]] = []
    for slot in range(count):
        if operation == "scout_train":
            margins = (0.25, 0.5, 0.75, 1.0)
            diversity = (0.0025, 0.005, 0.01, 0.02)
            values.append({
                "name": f"routing-hard-negative-{slot}",
                "routing_margin": margins[slot % len(margins)],
                "scout_diversity": diversity[(slot // len(margins)) % len(diversity)],
                "steps_multiplier": (0.75, 1.0, 1.25, 1.5)[slot % 4],
                "learning_rate_multiplier": (1.2, 1.0, 0.8, 0.65)[slot % 4],
                "objective": "correct-category-recall-and-scout-diversity",
            })
        elif operation == "field_train":
            values.append({
                "name": f"field-routing-margin-{slot}",
                "route_margin": (0.05, 0.10, 0.20, 0.35)[slot % 4],
                "route_margin_weight": (0.1, 0.25, 0.5, 0.75)[slot % 4],
                "steps_multiplier": (0.75, 1.0, 1.25, 1.5)[slot % 4],
                "learning_rate_multiplier": (1.1, 0.9, 0.75, 0.6)[slot % 4],
                "objective": "global-category-routing-margin",
            })
        elif operation in {"expert_train", "branch_train", "repair"}:
            values.append({
                "name": f"expert-specialization-{slot}",
                "expert_diversity": (0.0025, 0.005, 0.01, 0.02)[slot % 4],
                "steps_multiplier": (0.75, 1.0, 1.25, 1.5)[slot % 4],
                "learning_rate_multiplier": (1.15, 1.0, 0.85, 0.7)[slot % 4],
                "objective": "local-class-evidence-and-expert-diversity",
            })
        else:
            values.append({
                "name": f"memory-local-search-{slot}",
                "steps_multiplier": (0.75, 1.0, 1.25, 1.5)[slot % 4],
                "learning_rate_multiplier": (1.15, 1.0, 0.85, 0.7)[slot % 4],
                "objective": "class-addressed-associative-evidence",
            })
    return values


def plan_next_round(
    report: dict[str, Any],
    *,
    round_index: int,
    search_candidates: int = 8,
    sample_budget: int = 640,
) -> dict[str, Any]:
    report = validate_routing_report(report)
    if report.get("split") == "test" or report.get("test_selection_forbidden") is True:
        raise ValueError("the CIFAR-100 test split cannot be used for training-round planning")
    categories = list(report["categories"])
    worst_miss = max(categories, key=lambda value: (value["route_misses"] / value["samples"], value["routing_gap"], -value["category"]))
    worst_local = min(categories, key=lambda value: (value["oracle_category_accuracy"], value["actual_accuracy"], value["category"]))
    routing_limited = bool(
        float(report["topk_category_recall"]["4"]) < 0.97
        or float(report["expanded_category_recall"]) < 0.985
        or float(report["oracle_routing_gap"]) > 0.02
    )
    if routing_limited:
        if round_index % 5 == 4:
            operation = "field_train"
            target = round_index % 8
            reason = "global field routing margin is the scheduled routing-gap intervention"
        else:
            operation = "scout_train"
            target = int(worst_miss["category"])
            reason = "category has the largest route-miss burden"
        objective = "close-routing-gap"
        hard_negative_fraction = 0.55
    else:
        cycle = ("expert_train", "branch_train", "memory_train")
        operation = cycle[round_index % len(cycle)]
        target = int(worst_local["category"])
        reason = "routing is adequate; target the weakest conditional colony"
        objective = "raise-conditional-class-accuracy"
        hard_negative_fraction = 0.0
    value: dict[str, Any] = {
        "format": PLAN_FORMAT,
        "routing_report_sha256": report["sha256"],
        "model_root": report["model_root"],
        "round_index": int(round_index),
        "objective": objective,
        "operation": operation,
        "target": int(target),
        "reason": reason,
        "sample_budget": int(sample_budget),
        "search_candidates": int(search_candidates),
        "hard_negative_fraction": float(hard_negative_fraction),
        "recipes": search_recipes(operation, search_candidates),
        "selection_metric": "fresh-hidden-all-class-accuracy-with-exact-mcnemar-and-class-harm-gates",
        "routing_snapshot": {
            "actual_accuracy": float(report["actual_accuracy"]),
            "oracle_category_accuracy": float(report["oracle_category_accuracy"]),
            "oracle_routing_gap": float(report["oracle_routing_gap"]),
            "top4_category_recall": float(report["topk_category_recall"]["4"]),
            "expanded_category_recall": float(report["expanded_category_recall"]),
            "conditional_accuracy_when_routed": float(report["conditional_accuracy_when_routed"]),
            "route_miss_count": int(report["route_miss_count"]),
        },
        "test_split_used_for_planning": False,
    }
    value["sha256"] = content_hash(value)
    return value
