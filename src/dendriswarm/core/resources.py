from __future__ import annotations

import math
from typing import Any

from dendriswarm.core.models import (
    NodeCapabilities,
    ResourceClass,
    SeedPolicy,
    TaskKind,
    TaskRequirements,
    normalize_machine,
)


# Native10 mutations execute several small NumPy kernels per logical unit of
# work.  Treating them like the reference tissue's large matrix operations
# overstates throughput by roughly five times on commodity CPUs and creates
# hard timeouts that are shorter than a valid CIFAR-100 search trajectory.
NATIVE10_REFERENCE_UNITS_PER_SECOND = 8_000_000


def effective_limits(capabilities: NodeCapabilities, policy: SeedPolicy) -> dict[str, int | float | bool]:
    cpu_threads = max(1, min(capabilities.cpu_count, math.floor(capabilities.cpu_count * policy.cpu_percent / 100)))
    percent_memory = math.floor(capabilities.memory_mb * policy.memory_percent / 100)
    memory_mb = max(0, percent_memory)
    if policy.memory_limit_mb is not None:
        memory_mb = min(memory_mb, policy.memory_limit_mb)
    # Zero means zero available memory, not "unknown".
    memory_mb = min(memory_mb, capabilities.memory_available_mb)
    disk_mb = min(policy.disk_limit_mb, capabilities.disk_free_mb)
    effective_cpu_percent = 100.0 * cpu_threads / max(1, capabilities.cpu_count)
    duty_cycle = min(1.0, policy.cpu_percent / max(effective_cpu_percent, 1e-9))
    return {
        "cpu_threads": cpu_threads,
        "memory_mb": max(0, memory_mb),
        "disk_mb": max(0, disk_mb),
        "duty_cycle": duty_cycle,
        "paused": policy.paused,
    }


def adjusted_runtime_seconds(requirements: TaskRequirements, capabilities: NodeCapabilities) -> int:
    score = float(capabilities.benchmark_units_per_second)
    if score <= 0:
        return requirements.estimated_runtime_seconds
    reference_score = 100_000_000.0
    factor = min(8.0, max(0.5, reference_score / score))
    return max(1, int(math.ceil(requirements.estimated_runtime_seconds * factor)))


def node_can_run(
    kind: TaskKind,
    requirements: TaskRequirements,
    capabilities: NodeCapabilities,
    policy: SeedPolicy,
) -> tuple[bool, str]:
    limits = effective_limits(capabilities, policy)
    if policy.paused:
        return False, "seed-paused"
    if kind not in policy.allowed_task_kinds:
        return False, "task-kind-disabled"
    if requirements.backend not in capabilities.supported_backends:
        return False, "backend-unsupported"
    machine = normalize_machine(capabilities.machine)
    if requirements.supported_machines and machine not in requirements.supported_machines:
        return False, "machine-unsupported"
    effective_tags = set(capabilities.tags)
    if "reference-runtime-v2" in effective_tags:
        effective_tags.add("portable-numpy-v1")
    if not set(requirements.required_tags).issubset(effective_tags):
        return False, "required-tag-missing"
    if int(limits["cpu_threads"]) < requirements.min_cpu_threads:
        return False, "cpu-budget-too-small"
    if int(limits["memory_mb"]) < requirements.min_memory_mb:
        return False, "memory-budget-too-small"
    if int(limits["disk_mb"]) < requirements.min_disk_mb:
        return False, "disk-budget-too-small"
    artifact_bytes = int(requirements.max_artifact_bytes)
    if artifact_bytes > int(limits["disk_mb"]) * 1024 * 1024:
        return False, "artifact-exceeds-disk-budget"
    # Control-plane JSON is parsed outside the isolated compute subprocess.
    # Cap its signed maximum to a conservative fraction of the owner's active
    # memory budget so a formally valid contract cannot authorize an OOM-sized
    # response before payload-derived validation runs.
    control_plane_cap = max(1024 * 1024, int(limits["memory_mb"]) * 1024 * 1024 // 4)
    if artifact_bytes > control_plane_cap:
        return False, "artifact-exceeds-control-plane-memory-budget"
    if policy.max_task_seconds < max(requirements.estimated_runtime_seconds, requirements.hard_timeout_seconds):
        return False, "task-time-budget-too-small"
    return True, "eligible"


def _resource_class(memory_mb: int, estimated_runtime_seconds: int) -> ResourceClass:
    if memory_mb <= 192 and estimated_runtime_seconds <= 20:
        return ResourceClass.TINY
    if memory_mb <= 512 and estimated_runtime_seconds <= 180:
        return ResourceClass.SMALL
    if memory_mb <= 2048 and estimated_runtime_seconds <= 1800:
        return ResourceClass.MEDIUM
    return ResourceClass.LARGE


def estimate_reference_requirements(
    kind: TaskKind,
    *,
    samples: int,
    features: int,
    branches: int,
    iterations: int,
    artifact_bytes: int | None = None,
    required_tags: list[str] | None = None,
) -> TaskRequirements:
    samples = max(1, int(samples))
    features = max(1, int(features))
    branches = max(1, int(branches))
    iterations = max(1, int(iterations))
    matrix_bytes = samples * max(features, branches) * 8
    memory_mb = max(96, int(math.ceil(matrix_bytes * 4.0 / (1024 * 1024))) + 48)
    operation_hint = samples * branches * max(features, 1) * iterations
    reference_units_per_second = (
        NATIVE10_REFERENCE_UNITS_PER_SECOND
        if kind in {TaskKind.DENDRITRON_MUTATION, TaskKind.DENDRITRON_VERIFICATION}
        else 40_000_000
    )
    estimated = max(1, min(86400, int(math.ceil(operation_hint / reference_units_per_second))))
    if kind == TaskKind.INFERENCE:
        estimated = max(1, min(30, estimated))
        memory_mb = max(32, min(memory_mb, 256))
    elif kind == TaskKind.VERIFICATION:
        estimated = max(2, min(1800, estimated))
    elif kind == TaskKind.EXPLORATION:
        estimated = max(2, min(3600, estimated))
    elif kind == TaskKind.DENDRITRON_VERIFICATION:
        estimated = max(2, min(3600, estimated))
    elif kind == TaskKind.DENDRITRON_MUTATION:
        estimated = max(5, min(86400, estimated))
    else:
        estimated = max(5, min(86400, estimated))
    timeout_multiplier = 3.0 if kind in {
        TaskKind.DENDRITRON_MUTATION, TaskKind.DENDRITRON_VERIFICATION,
    } else 2.5
    hard_timeout = min(86400, max(estimated + 10, int(math.ceil(estimated * timeout_multiplier))))
    max_bytes = max(64 * 1024, int(artifact_bytes or (samples * features * 24 + branches * features * 24)))
    preferred_threads = 1 if memory_mb <= 256 else 2 if memory_mb <= 1024 else 4
    return TaskRequirements(
        resource_class=_resource_class(memory_mb, estimated),
        min_cpu_threads=1,
        preferred_cpu_threads=preferred_threads,
        min_memory_mb=memory_mb,
        max_memory_mb=max(256, min(1_048_576, memory_mb * 2)),
        min_disk_mb=max(1, int(math.ceil(max_bytes / (1024 * 1024)))),
        estimated_runtime_seconds=estimated,
        hard_timeout_seconds=hard_timeout,
        max_artifact_bytes=max_bytes,
        backend="numpy-cpu",
        required_tags=list(required_tags or ["portable-numpy-v1", "deterministic-v2"]),
        checkpointable=False,
    )


def derive_payload_requirements(kind: TaskKind, payload: dict[str, Any]) -> TaskRequirements:
    """Independently derive a lower bound from materialized task content."""
    if kind in {TaskKind.DENDRITRON_MUTATION, TaskKind.DENDRITRON_VERIFICATION}:
        bundle = payload.get("bundle") or {}
        config = bundle.get("config") or {}
        operation = str(bundle.get("operation") or "")
        width = int(config.get("input_width") if operation == "field_train" else config.get("representation_width") or 1)
        train_value = payload.get("train_data") or payload.get("train_representations") or []
        if isinstance(train_value, dict) and train_value.get("shape"):
            train_rows = int(train_value["shape"][0])
        elif isinstance(train_value, dict) and isinstance(train_value.get("array"), dict) and train_value["array"].get("shape"):
            train_rows = int(train_value["array"]["shape"][0])
        else:
            train_rows = len(train_value)
        validation_value = payload.get("validation_representations") or []
        if isinstance(validation_value, dict) and validation_value.get("shape"):
            validation_rows = int(validation_value["shape"][0])
        else:
            validation_rows = len(validation_value)
        validation_rows = max(validation_rows, int(payload.get("validation_sample_count") or 0))
        samples = max(1, train_rows + validation_rows)
        experts = int(config.get("active_experts_per_update") or config.get("experts_per_category") or 1)
        branches = int(config.get("expert_branches") or 1)
        branch_width = int(config.get("branch_width") or 1)
        checkpoint = payload.get("_native10_checkpoint") or {}
        serialized_hint = len(str(payload).encode("utf-8"))
        # Mutation work scales with active experts. Verification loads the complete
        # canonical model, but routed evaluation only opens the configured maximum
        # category fan-out before and after applying the exact candidate delta.
        effective_branches = experts * branches * branch_width
        if kind == TaskKind.DENDRITRON_VERIFICATION:
            effective_branches = (
                int(config.get("max_routed_categories") or config.get("categories") or 1)
                * int(config.get("experts_per_category") or experts)
                * branches
                * branch_width
            )
        requirements = estimate_reference_requirements(
            kind, samples=samples, features=width, branches=max(1, effective_branches),
            iterations=2 if kind == TaskKind.DENDRITRON_VERIFICATION else max(1, int(payload.get("optimizer_steps") or 1)),
            artifact_bytes=max(serialized_hint, 64 * 1024),
            required_tags=list(payload.get("required_tags") or ["portable-numpy-v1"]),
        )
        if checkpoint:
            parameter_count = max(0, int(checkpoint.get("parameter_count") or 0))
            checkpoint_memory_mb = max(
                256, int(math.ceil(parameter_count * 12 / (1024 * 1024))) + 128
            )
            requirements = requirements.model_copy(update={
                "min_memory_mb": max(requirements.min_memory_mb, checkpoint_memory_mb),
                "max_memory_mb": max(int(requirements.max_memory_mb or 0), checkpoint_memory_mb * 2),
            })
        return requirements
    dataset = payload.get("_dataset")
    artifact = payload.get("_artifact")
    config = payload.get("config") or (artifact or {}).get("config") or {}
    if kind == TaskKind.INFERENCE:
        features = len(payload.get("features") or [])
        branches = len((artifact or {}).get("centers") or [])
        samples, iterations = 1, 1
    elif dataset is not None:
        features = int(dataset.get("feature_width") or len((dataset.get("features") or [[0]])[0]))
        if kind == TaskKind.TRAINING:
            samples = len(dataset.get("splits", {}).get("train", [])) + len(dataset.get("splits", {}).get("validation", []))
        elif kind == TaskKind.VERIFICATION:
            samples = len(dataset.get("splits", {}).get("test", []))
        else:
            samples = len(dataset.get("splits", {}).get("train", [])) + len(dataset.get("splits", {}).get("validation", []))
        branches = int(config.get("branches") or len((artifact or {}).get("centers") or []) or 1)
        iterations = int(config.get("iterations") or 1)
    else:
        raise ValueError("task payload is missing materialized data")
    serialized_hint = len(str(dataset or {}).encode("utf-8")) + len(str(artifact or {}).encode("utf-8"))
    if not dataset and not artifact:
        serialized_hint = len(str(payload).encode("utf-8"))
    return estimate_reference_requirements(
        kind,
        samples=samples,
        features=features,
        branches=branches,
        iterations=iterations,
        artifact_bytes=serialized_hint,
    )


def contract_covers(declared: TaskRequirements, derived: TaskRequirements) -> tuple[bool, str]:
    checks = {
        "min_memory_mb": declared.min_memory_mb >= derived.min_memory_mb,
        "max_memory_mb": declared.max_memory_mb >= derived.max_memory_mb,
        "min_disk_mb": declared.min_disk_mb >= derived.min_disk_mb,
        "hard_timeout_seconds": declared.hard_timeout_seconds >= derived.hard_timeout_seconds,
        "max_artifact_bytes": declared.max_artifact_bytes >= derived.max_artifact_bytes,
    }
    for name, passed in checks.items():
        if not passed:
            return False, f"understated-{name}"
    return True, "covered"


def requirements_from_value(value: TaskRequirements | dict[str, Any] | None, kind: TaskKind) -> TaskRequirements:
    if isinstance(value, TaskRequirements):
        return value
    if value:
        return TaskRequirements.model_validate(value)
    defaults = {
        TaskKind.INFERENCE: dict(min_memory_mb=64, max_memory_mb=256, estimated_runtime_seconds=5, hard_timeout_seconds=30),
        TaskKind.EXPLORATION: dict(min_memory_mb=128, max_memory_mb=384, estimated_runtime_seconds=30, hard_timeout_seconds=120),
        TaskKind.TRAINING: dict(min_memory_mb=256, max_memory_mb=768, estimated_runtime_seconds=120, hard_timeout_seconds=600),
        TaskKind.VERIFICATION: dict(min_memory_mb=128, max_memory_mb=384, estimated_runtime_seconds=30, hard_timeout_seconds=120),
        TaskKind.DENDRITRON_MUTATION: dict(min_memory_mb=128, max_memory_mb=1024, estimated_runtime_seconds=90, hard_timeout_seconds=600),
        TaskKind.DENDRITRON_VERIFICATION: dict(min_memory_mb=128, max_memory_mb=1024, estimated_runtime_seconds=45, hard_timeout_seconds=300),
    }
    return TaskRequirements(**defaults[kind])
