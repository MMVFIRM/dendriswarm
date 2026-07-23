from __future__ import annotations

from enum import Enum
from typing import Any, Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from dendriswarm.core.crypto import content_hash


def validate_bounded_json_tree(
    value: Any,
    *,
    max_depth: int = 32,
    max_nodes: int = 250_000,
    max_mapping_keys: int = 100_000,
) -> Any:
    """Reject pathological nested JSON before service-side serialization/storage."""
    stack: list[tuple[Any, int]] = [(value, 0)]
    nodes = 0
    while stack:
        current, depth = stack.pop()
        nodes += 1
        if nodes > max_nodes:
            raise ValueError("JSON structure exceeds node limit")
        if depth > max_depth:
            raise ValueError("JSON structure exceeds depth limit")
        if isinstance(current, dict):
            if len(current) > max_mapping_keys:
                raise ValueError("JSON object exceeds key limit")
            stack.extend((item, depth + 1) for item in current.values())
        elif isinstance(current, list):
            stack.extend((item, depth + 1) for item in current)
    return value


def normalize_machine(value: str) -> str:
    normalized = (value or "unknown").strip().lower().replace(" ", "_")
    aliases = {
        "amd64": "x86_64",
        "x64": "x86_64",
        "x86-64": "x86_64",
        "arm64": "aarch64",
        "armv8": "aarch64",
        "armv8l": "aarch64",
    }
    return aliases.get(normalized, normalized)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class TaskKind(str, Enum):
    INFERENCE = "inference"
    TRAINING = "training"
    EXPLORATION = "exploration"
    VERIFICATION = "verification"
    DENDRITRON_MUTATION = "dendritron-mutation"
    DENDRITRON_VERIFICATION = "dendritron-verification"


class TaskStatus(str, Enum):
    QUEUED = "queued"
    ASSIGNED = "assigned"
    COMPLETED = "completed"
    FAILED = "failed"


class ResourceClass(str, Enum):
    TINY = "tiny"
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"


class SeedPolicy(StrictModel):
    """User-controlled contribution envelope, hot-reloaded while seeding."""

    paused: bool = False
    cpu_percent: int = Field(default=25, ge=1, le=100)
    memory_percent: int = Field(default=25, ge=1, le=100)
    memory_limit_mb: int | None = Field(default=None, ge=64)
    disk_limit_mb: int = Field(default=2048, ge=64)
    max_task_seconds: int = Field(default=900, ge=10, le=86400)
    allowed_task_kinds: list[TaskKind] = Field(
        default_factory=lambda: list(TaskKind), min_length=1, max_length=len(TaskKind)
    )
    allow_on_battery: bool = False
    min_battery_percent: int = Field(default=30, ge=0, le=100)
    max_system_cpu_percent: int = Field(default=95, ge=10, le=100)


class NodeCapabilities(StrictModel):
    cpu_count: int = Field(default=1, ge=1, le=4096)
    physical_cpu_count: int | None = Field(default=None, ge=1, le=4096)
    memory_mb: int = Field(default=512, ge=128)
    memory_available_mb: int = Field(default=512, ge=0)
    disk_free_mb: int = Field(default=1024, ge=0)
    accelerator: str = Field(default="cpu", max_length=64)
    accelerators: list[str] = Field(default_factory=lambda: ["cpu"], max_length=16)
    platform: str = Field(default="unknown", max_length=128)
    machine: str = Field(default="unknown", max_length=64)
    python_version: str = Field(default="unknown", max_length=64)
    supported_backends: list[str] = Field(default_factory=lambda: ["numpy-cpu"], max_length=16)
    benchmark_units_per_second: float = Field(default=0.0, ge=0.0)
    tags: list[str] = Field(default_factory=list, max_length=32)


class TaskRequirements(StrictModel):
    """Portable signed resource contract attached to every task.

    ``hard_timeout_seconds`` and ``max_artifact_bytes`` are enforcement values,
    not advisory estimates. A worker independently derives requirements from the
    payload and refuses understated contracts.
    """

    resource_class: ResourceClass = ResourceClass.SMALL
    min_cpu_threads: int = Field(default=1, ge=1, le=4096)
    preferred_cpu_threads: int = Field(default=1, ge=1, le=4096)
    min_memory_mb: int = Field(default=128, ge=32)
    max_memory_mb: int | None = Field(default=None, ge=64, le=1_048_576)
    min_disk_mb: int = Field(default=16, ge=0)
    estimated_runtime_seconds: int = Field(default=30, ge=1, le=86400)
    hard_timeout_seconds: int = Field(default=120, ge=5, le=86400)
    max_artifact_bytes: int = Field(default=8 * 1024 * 1024, ge=1024, le=2 * 1024 * 1024 * 1024)
    backend: str = Field(default="numpy-cpu", max_length=64)
    required_tags: list[str] = Field(default_factory=list, max_length=32)
    supported_machines: list[str] = Field(default_factory=list, max_length=32)
    checkpointable: bool = False

    @field_validator("supported_machines", mode="before")
    @classmethod
    def _normalize_machines(cls, value: Any) -> list[str]:
        return [normalize_machine(str(item)) for item in (value or [])]

    @model_validator(mode="after")
    def _validate_timeout(self) -> "TaskRequirements":
        if self.hard_timeout_seconds < self.estimated_runtime_seconds:
            raise ValueError("hard timeout must not be below estimated runtime")
        if self.preferred_cpu_threads < self.min_cpu_threads:
            raise ValueError("preferred_cpu_threads must be >= min_cpu_threads")
        if self.max_memory_mb is None:
            self.max_memory_mb = max(256, min(1_048_576, self.min_memory_mb * 2))
        if self.max_memory_mb < self.min_memory_mb:
            raise ValueError("max_memory_mb must be >= min_memory_mb")
        return self


class NodeRegistration(StrictModel):
    node_id: str = Field(min_length=16, max_length=128)
    public_key: str = Field(min_length=32, max_length=256)
    capabilities: NodeCapabilities
    policy: SeedPolicy | None = None
    timestamp: int
    nonce: str = Field(min_length=16, max_length=128)
    signature: str = Field(min_length=32, max_length=256)


class SignedNodeRequest(StrictModel):
    node_id: str = Field(min_length=16, max_length=128)
    timestamp: int
    nonce: str = Field(min_length=16, max_length=128)
    signature: str = Field(min_length=32, max_length=256)


class TaskLeaseRequest(StrictModel):
    node_id: str = Field(min_length=16, max_length=128)
    task_id: str = Field(min_length=16, max_length=128)
    lease_token: str = Field(min_length=16, max_length=128)
    timestamp: int
    nonce: str = Field(min_length=16, max_length=128)
    signature: str = Field(min_length=32, max_length=256)


class TaskAbandonRequest(TaskLeaseRequest):
    reason: str = Field(default="local-policy-change", min_length=1, max_length=160)


class TaskBody(StrictModel):
    id: str
    kind: TaskKind
    payload: dict[str, Any]
    requirements: TaskRequirements = Field(default_factory=TaskRequirements)
    reward: float = Field(ge=0)
    priority: int
    created_at: float
    assigned_to: str
    lease_token: str
    lease_expires_at: float
    lease_deadline_at: float


class SignedTask(StrictModel):
    task: TaskBody
    coordinator_public_key: str
    signature: str


class RuntimeMetadata(StrictModel):
    backend: Literal["numpy-cpu"]
    machine: str = Field(max_length=64)
    python: str = Field(max_length=64)
    cpu_threads: int | None = Field(default=None, ge=1, le=4096)

    @field_validator("machine", mode="before")
    @classmethod
    def _normalize_machine(cls, value: Any) -> str:
        return normalize_machine(str(value))


class ExplorationOutput(StrictModel):
    config: dict[str, Any]
    validation_accuracy: float = Field(ge=0, le=1)
    sample_count: int = Field(ge=1, le=10_000_000)
    correct_count: int = Field(ge=0, le=10_000_000)
    runtime: RuntimeMetadata

    @model_validator(mode="after")
    def _counts_align(self) -> "ExplorationOutput":
        if self.correct_count > self.sample_count:
            raise ValueError("correct_count exceeds sample_count")
        if abs(self.validation_accuracy - self.correct_count / self.sample_count) > 1e-12:
            raise ValueError("exploration accuracy does not match integer counts")
        return self


class TrainingOutput(StrictModel):
    artifact: dict[str, Any]
    train_accuracy: float = Field(ge=0, le=1)
    sample_count: int = Field(ge=1, le=10_000_000)
    correct_count: int = Field(ge=0, le=10_000_000)
    runtime: RuntimeMetadata

    @model_validator(mode="after")
    def _counts_align(self) -> "TrainingOutput":
        if self.correct_count > self.sample_count:
            raise ValueError("correct_count exceeds sample_count")
        if abs(self.train_accuracy - self.correct_count / self.sample_count) > 1e-12:
            raise ValueError("training accuracy does not match integer counts")
        return self


class VerificationOutput(StrictModel):
    artifact_hash: str = Field(min_length=64, max_length=64)
    test_accuracy: float = Field(ge=0, le=1)
    sample_count: int = Field(ge=1, le=10_000_000)
    correct_count: int = Field(ge=0, le=10_000_000)
    runtime: RuntimeMetadata

    @model_validator(mode="after")
    def _counts_align(self) -> "VerificationOutput":
        if self.correct_count > self.sample_count:
            raise ValueError("correct_count exceeds sample_count")
        if abs(self.test_accuracy - self.correct_count / self.sample_count) > 1e-12:
            raise ValueError("verification accuracy does not match integer counts")
        return self


class DendritronMutationOutput(StrictModel):
    delta: dict[str, Any]
    sample_count: int = Field(ge=1, le=10_000_000)
    pre_correct: int = Field(ge=0, le=10_000_000)
    post_correct: int = Field(ge=0, le=10_000_000)
    net_wins: int = Field(ge=-10_000_000, le=10_000_000)
    pre_correct_by_class: list[int] = Field(min_length=1, max_length=4096)
    post_correct_by_class: list[int] = Field(min_length=1, max_length=4096)
    active_experts: list[int] = Field(default_factory=list, max_length=4096)
    rotation_phase_before: int | None = None
    rotation_phase_after: int | None = None
    metrics_scope: Literal["trainer-visible-training-diagnostic-not-promotion-evidence"] = "trainer-visible-training-diagnostic-not-promotion-evidence"
    runtime: RuntimeMetadata

    @model_validator(mode="after")
    def _metrics_align(self) -> "DendritronMutationOutput":
        if self.pre_correct > self.sample_count or self.post_correct > self.sample_count:
            raise ValueError("Dendritron mutation correct count exceeds sample count")
        if self.net_wins != self.post_correct - self.pre_correct:
            raise ValueError("Dendritron mutation net_wins mismatch")
        if len(self.pre_correct_by_class) != len(self.post_correct_by_class):
            raise ValueError("Dendritron mutation class metrics are misaligned")
        return self


class DendritronVerificationOutput(StrictModel):
    delta_hash: str = Field(min_length=64, max_length=64)
    validation_hash: str = Field(min_length=64, max_length=64)
    base_root: str = Field(min_length=64, max_length=64)
    operation: str = Field(min_length=1, max_length=64)
    category: int = Field(ge=0, le=4096)
    sample_count: int = Field(ge=1, le=10_000_000)
    pre_correct: int = Field(ge=0, le=10_000_000)
    post_correct: int = Field(ge=0, le=10_000_000)
    net_wins: int = Field(ge=-10_000_000, le=10_000_000)
    pre_correct_by_class: list[int] = Field(min_length=1, max_length=4096)
    post_correct_by_class: list[int] = Field(min_length=1, max_length=4096)
    samples_by_class: list[int] = Field(min_length=1, max_length=4096)
    losses_by_class: list[int] = Field(min_length=1, max_length=4096)
    loss_rates_by_class: list[float] = Field(min_length=1, max_length=4096)
    informative: bool
    runtime: RuntimeMetadata

    @model_validator(mode="after")
    def _verification_aligns(self) -> "DendritronVerificationOutput":
        if self.pre_correct > self.sample_count or self.post_correct > self.sample_count:
            raise ValueError("Dendritron verification correct count exceeds sample count")
        if self.net_wins != self.post_correct - self.pre_correct:
            raise ValueError("Dendritron verification net_wins mismatch")
        lengths = {
            len(self.pre_correct_by_class), len(self.post_correct_by_class),
            len(self.samples_by_class), len(self.losses_by_class), len(self.loss_rates_by_class)
        }
        if len(lengths) != 1:
            raise ValueError("Dendritron verification class metrics are misaligned")
        expected_losses = [max(0, before - after) for before, after in zip(self.pre_correct_by_class, self.post_correct_by_class, strict=True)]
        if expected_losses != self.losses_by_class:
            raise ValueError("Dendritron verification loss vector mismatch")
        if sum(self.samples_by_class) != self.sample_count:
            raise ValueError("Dendritron verification class coverage does not match sample count")
        expected_rates = [
            (loss / count if count else 0.0)
            for loss, count in zip(self.losses_by_class, self.samples_by_class, strict=True)
        ]
        if any(abs(expected - actual) > 1e-12 for expected, actual in zip(expected_rates, self.loss_rates_by_class, strict=True)):
            raise ValueError("Dendritron verification loss-rate vector mismatch")
        return self


class DendritronV6MutationOutput(StrictModel):
    delta: dict[str, Any]
    sample_count: int = Field(ge=1, le=10_000_000)
    pre_correct: int = Field(ge=0, le=10_000_000)
    post_correct: int = Field(ge=0, le=10_000_000)
    net_wins: int = Field(ge=-10_000_000, le=10_000_000)
    pre_correct_by_class: list[int] = Field(min_length=1, max_length=4096)
    post_correct_by_class: list[int] = Field(min_length=1, max_length=4096)
    active_experts: list[int] = Field(default_factory=list, max_length=4096)
    rotation_phase_before: int | None = None
    rotation_phase_after: int | None = None
    metrics_scope: Literal["trainer-visible-training-diagnostic-not-promotion-evidence"]
    changed_parameters: int = Field(ge=1, le=100_000_000)
    write_set: list[str] = Field(min_length=1, max_length=100_000)
    search_seed: int
    optimizer: Literal["local-sgd-v1"]
    optimizer_steps: int = Field(ge=1, le=100_000)
    learning_rate: float = Field(gt=0, le=10)
    search_recipe: dict[str, Any] = Field(default_factory=dict)
    search_recipe_hash: str = Field(min_length=64, max_length=64)
    initial_loss: float | None = None
    final_loss: float | None = None
    runtime: RuntimeMetadata

    @model_validator(mode="after")
    def _align(self) -> "DendritronV6MutationOutput":
        if self.pre_correct > self.sample_count or self.post_correct > self.sample_count:
            raise ValueError("Dendritron mutation correct count exceeds sample count")
        if self.net_wins != self.post_correct - self.pre_correct:
            raise ValueError("Dendritron mutation net_wins mismatch")
        if len(self.pre_correct_by_class) != len(self.post_correct_by_class):
            raise ValueError("Dendritron mutation class metrics are misaligned")
        if self.search_recipe_hash != content_hash(self.search_recipe):
            raise ValueError("Dendritron search recipe hash mismatch")
        return self


class DendritronV6VerificationOutput(StrictModel):
    delta_hash: str = Field(min_length=64, max_length=64)
    validation_hash: str = Field(min_length=64, max_length=64)
    base_root: str = Field(min_length=64, max_length=64)
    operation: str = Field(min_length=1, max_length=64)
    category: int = Field(ge=0, le=4096)
    sample_count: int = Field(ge=1, le=10_000_000)
    pre_correct: int = Field(ge=0, le=10_000_000)
    post_correct: int = Field(ge=0, le=10_000_000)
    net_wins: int = Field(ge=-10_000_000, le=10_000_000)
    wins: int = Field(ge=0, le=10_000_000)
    losses: int = Field(ge=0, le=10_000_000)
    discordant: int = Field(ge=0, le=10_000_000)
    effect_rate: float = Field(ge=-1, le=1)
    mcnemar_p_value: float = Field(ge=0, le=1)
    corrected_alpha: float = Field(gt=0, le=1)
    statistically_significant: bool
    pre_correct_by_class: list[int] = Field(min_length=1, max_length=4096)
    post_correct_by_class: list[int] = Field(min_length=1, max_length=4096)
    samples_by_class: list[int] = Field(min_length=1, max_length=4096)
    losses_by_class: list[int] = Field(min_length=1, max_length=4096)
    loss_rates_by_class: list[float] = Field(min_length=1, max_length=4096)
    informative: bool
    write_set: list[str] = Field(min_length=1, max_length=100_000)
    runtime: RuntimeMetadata

    @model_validator(mode="after")
    def _align(self) -> "DendritronV6VerificationOutput":
        if self.net_wins != self.wins - self.losses or self.discordant != self.wins + self.losses:
            raise ValueError("paired verification counts are inconsistent")
        if self.net_wins != self.post_correct - self.pre_correct:
            raise ValueError("verification net_wins mismatch")
        lengths = {len(self.pre_correct_by_class), len(self.post_correct_by_class), len(self.samples_by_class), len(self.losses_by_class), len(self.loss_rates_by_class)}
        if len(lengths) != 1 or sum(self.samples_by_class) != self.sample_count:
            raise ValueError("verification class vectors are inconsistent")
        return self


class InferenceOutput(StrictModel):
    prediction: int = Field(ge=0, le=1_000_000)
    confidence: float = Field(ge=0, le=1)
    scores: list[float] = Field(min_length=1, max_length=4096)
    active_branches: int = Field(ge=1, le=4096)
    total_branches: int = Field(ge=1, le=4096)
    activation_fraction: float = Field(gt=0, le=1)
    runtime: RuntimeMetadata


class TaskResult(StrictModel):
    node_id: str = Field(min_length=16, max_length=128)
    task_id: str = Field(min_length=16, max_length=128)
    lease_token: str = Field(min_length=16, max_length=128)
    duration_ms: int = Field(ge=0, le=86_400_000)
    output: dict[str, Any]
    signature: str = Field(min_length=32, max_length=256)

    @field_validator("output", mode="before")
    @classmethod
    def _bounded_output(cls, value: Any) -> Any:
        return validate_bounded_json_tree(value)


FeatureRow = Annotated[list[float], Field(min_length=1, max_length=4096)]


class DatasetSubmission(StrictModel):
    node_id: str = Field(min_length=16, max_length=128)
    request_id: str = Field(min_length=16, max_length=128)
    timestamp: int
    nonce: str = Field(min_length=16, max_length=128)
    signature: str = Field(min_length=32, max_length=256)
    name: str = Field(min_length=1, max_length=160)
    license: str = Field(min_length=1, max_length=80)
    source: str = Field(default="", max_length=500)
    description: str = Field(default="", max_length=2000)
    features: list[FeatureRow] = Field(min_length=1, max_length=10_000)
    labels: list[int] = Field(min_length=1, max_length=10_000)

    @model_validator(mode="after")
    def _shape(self) -> "DatasetSubmission":
        if len(self.features) != len(self.labels):
            raise ValueError("features and labels must align")
        width = len(self.features[0])
        if any(len(row) != width for row in self.features):
            raise ValueError("feature rows must have equal width")
        return self


class InferenceRequest(StrictModel):
    node_id: str = Field(min_length=16, max_length=128)
    request_id: str = Field(min_length=16, max_length=128)
    timestamp: int
    nonce: str = Field(min_length=16, max_length=128)
    features: list[float] = Field(min_length=1, max_length=4096)
    signature: str = Field(min_length=32, max_length=256)


class LeverageSubmissionRequest(StrictModel):
    node_id: str = Field(min_length=16, max_length=128)
    timestamp: int
    nonce: str = Field(min_length=16, max_length=128)
    manifest: dict[str, Any]
    signature: str = Field(min_length=32, max_length=256)

    @field_validator("manifest", mode="before")
    @classmethod
    def _bounded_manifest(cls, value: Any) -> Any:
        return validate_bounded_json_tree(value)


class LeverageCanaryRequest(StrictModel):
    node_id: str = Field(min_length=16, max_length=128)
    timestamp: int
    nonce: str = Field(min_length=16, max_length=128)
    features: list[FeatureRow] = Field(min_length=1, max_length=10_000)
    labels: list[int] = Field(min_length=1, max_length=10_000)
    subgroup_ids: list[int | str] | None = Field(default=None, max_length=10_000)
    source_id: str = Field(min_length=1, max_length=256)
    source_kind: str = Field(min_length=1, max_length=80)
    signature: str = Field(min_length=32, max_length=256)

    @model_validator(mode="after")
    def _shape(self) -> "LeverageCanaryRequest":
        if len(self.features) != len(self.labels):
            raise ValueError("features and labels must align")
        if self.subgroup_ids is not None and len(self.subgroup_ids) != len(self.labels):
            raise ValueError("subgroup_ids and labels must align")
        width = len(self.features[0])
        if any(len(row) != width for row in self.features):
            raise ValueError("feature rows must have equal width")
        return self
