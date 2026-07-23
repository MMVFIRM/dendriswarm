from __future__ import annotations

import base64
import copy
import io
import json
import math
import hashlib
import zlib
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

from dendriswarm.core.crypto import content_hash

ENGINE = "dendriswarm.native10-trainable.v6"
CHECKPOINT_FORMAT = "dendriswarm.native10-checkpoint.v6"
DELTA_FORMAT = "dendriswarm.native10-sparse-delta.v6"
BUNDLE_FORMAT = "dendriswarm.native10-work-bundle.v6"
QUANTIZED_FORMAT = "dendriswarm.native10-int8.v2"


@dataclass(frozen=True)
class Native10Config:
    """CPU-portable tensorization of the Native10 topology.

    The exact profile preserves the defining topology: eight sensory blocks,
    96-wide shared representation, 1,000 fine scouts, top-four category
    proposals, 20 category colonies, 45 experts per colony, and rotating
    15-of-45 expert updates.  The compact profile exists only for tests and
    local protocol demonstrations; it uses the identical computation graph.
    """

    input_width: int = 3072
    field_blocks: int = 8
    representation_width: int = 96
    scout_projection_width: int = 3
    categories: int = 20
    classes: int = 100
    scouts_per_category: int = 50
    experts_per_category: int = 45
    active_experts_per_update: int = 15
    expert_branches: int = 4
    branch_width: int = 12
    top_categories: int = 4
    seed: int = 7
    routing_expansion_margin: float = 0.15
    max_routed_categories: int = 8
    memory_strength_init: float = 0.20

    def __post_init__(self) -> None:
        if self.input_width < 1 or self.input_width % self.field_blocks:
            raise ValueError("input_width must be positive and divisible by field_blocks")
        if self.representation_width < self.field_blocks or self.representation_width % self.field_blocks:
            raise ValueError("representation_width must be divisible by field_blocks")
        if self.classes < self.categories or self.classes % self.categories:
            raise ValueError("classes must be divisible by categories")
        if self.experts_per_category % self.active_experts_per_update:
            raise ValueError("experts_per_category must be divisible by active_experts_per_update")
        if not 1 <= self.top_categories <= self.categories:
            raise ValueError("top_categories is invalid")
        if self.scouts_per_category < 1 or self.expert_branches < 1 or self.branch_width < 1:
            raise ValueError("scout and expert dimensions must be positive")
        if not 0.0 <= self.routing_expansion_margin <= 10.0:
            raise ValueError("routing_expansion_margin must be non-negative")
        if not self.top_categories <= self.max_routed_categories <= self.categories:
            raise ValueError("max_routed_categories must be between top_categories and categories")
        if not 0.0 <= self.memory_strength_init <= 2.0:
            raise ValueError("memory_strength_init must be in [0,2]")

    @property
    def classes_per_category(self) -> int:
        return self.classes // self.categories

    @property
    def scout_count(self) -> int:
        return self.categories * self.scouts_per_category

    @property
    def rotation_groups(self) -> int:
        return self.experts_per_category // self.active_experts_per_update

    @property
    def field_block_width(self) -> int:
        return self.input_width // self.field_blocks

    @property
    def field_block_output(self) -> int:
        return self.representation_width // self.field_blocks

    def as_dict(self) -> dict[str, Any]:
        return {
            "input_width": self.input_width,
            "field_blocks": self.field_blocks,
            "representation_width": self.representation_width,
            "scout_projection_width": self.scout_projection_width,
            "categories": self.categories,
            "classes": self.classes,
            "scouts_per_category": self.scouts_per_category,
            "experts_per_category": self.experts_per_category,
            "active_experts_per_update": self.active_experts_per_update,
            "expert_branches": self.expert_branches,
            "branch_width": self.branch_width,
            "top_categories": self.top_categories,
            "seed": self.seed,
            "routing_expansion_margin": self.routing_expansion_margin,
            "max_routed_categories": self.max_routed_categories,
            "memory_strength_init": self.memory_strength_init,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Native10Config":
        return cls(**{key: value[key] for key in cls.__dataclass_fields__ if key in value})

    @classmethod
    def compact_demo(cls, seed: int = 7) -> "Native10Config":
        return cls(
            input_width=32,
            field_blocks=4,
            representation_width=24,
            scout_projection_width=3,
            categories=4,
            classes=12,
            scouts_per_category=4,
            experts_per_category=6,
            active_experts_per_update=2,
            expert_branches=3,
            branch_width=8,
            top_categories=2,
            seed=seed,
            routing_expansion_margin=0.20,
            max_routed_categories=4,
            memory_strength_init=0.20,
        )


def _array_bytes(array: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    np.save(buffer, np.asarray(array), allow_pickle=False)
    return buffer.getvalue()


def encode_array(array: np.ndarray) -> dict[str, Any]:
    raw = _array_bytes(np.asarray(array))
    compressed = zlib.compress(raw, level=9)
    return {
        "codec": "npy+zlib+base64",
        "dtype": str(np.asarray(array).dtype),
        "shape": list(np.asarray(array).shape),
        "raw_sha256": content_hash({"bytes_b64": base64.b64encode(raw).decode("ascii")}),
        "data": base64.b64encode(compressed).decode("ascii"),
    }


def decode_array(value: dict[str, Any], *, max_raw_bytes: int = 256 * 1024 * 1024) -> np.ndarray:
    if value.get("codec") != "npy+zlib+base64":
        raise ValueError("unsupported tensor codec")
    encoded = str(value["data"])
    if len(encoded) > max_raw_bytes * 2 + 4096:
        raise ValueError("encoded tensor exceeds byte limit")
    compressed = base64.b64decode(encoded, validate=True)
    decoder = zlib.decompressobj()
    raw = decoder.decompress(compressed, max_raw_bytes + 1)
    if len(raw) > max_raw_bytes or decoder.unconsumed_tail:
        raise ValueError("decoded tensor exceeds byte limit")
    remaining = max_raw_bytes + 1 - len(raw)
    raw += decoder.flush(max(1, remaining))
    if len(raw) > max_raw_bytes or not decoder.eof:
        raise ValueError("decoded tensor exceeds byte limit or is truncated")
    expected = content_hash({"bytes_b64": base64.b64encode(raw).decode("ascii")})
    if expected != value.get("raw_sha256"):
        raise ValueError("tensor payload hash mismatch")
    array = np.load(io.BytesIO(raw), allow_pickle=False)
    if list(array.shape) != list(value.get("shape", [])) or str(array.dtype) != str(value.get("dtype")):
        raise ValueError("tensor metadata mismatch")
    if array.dtype.kind == "f" and not np.isfinite(array).all():
        raise ValueError("tensor contains non-finite values")
    return np.asarray(array)



def decode_training_tensor(value: Any, *, max_raw_bytes: int = 256 * 1024 * 1024) -> np.ndarray:
    """Decode a bounded training tensor, including the CIFAR-100 patch adapter.

    CIFAR contributors receive uint8 images instead of expanded float32 rows.
    Normalization and the 4x2 spatial patch ordering are deterministic and
    therefore independently derivable from the signed payload.
    """
    if isinstance(value, dict) and value.get("codec"):
        return np.asarray(decode_array(value, max_raw_bytes=max_raw_bytes), dtype=np.float32)
    if isinstance(value, dict) and value.get("format") == "dendriswarm.cifar100-patch-input.v1":
        encoded = value.get("array")
        if not isinstance(encoded, dict):
            raise ValueError("CIFAR-100 training tensor is missing its encoded array")
        images = np.asarray(decode_array(encoded, max_raw_bytes=max_raw_bytes), dtype=np.uint8)
        if images.ndim != 2 or images.shape[1] != 3072:
            raise ValueError("CIFAR-100 encoded images must have shape [N,3072]")
        mean = np.asarray(value.get("channel_mean"), dtype=np.float32)
        std = np.asarray(value.get("channel_std"), dtype=np.float32)
        if mean.shape != (3,) or std.shape != (3,) or np.any(std < 1e-6):
            raise ValueError("CIFAR-100 normalization metadata is invalid")
        scale = float(value.get("scale", 255.0))
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError("CIFAR-100 normalization scale is invalid")
        shaped = images.reshape(len(images), 3, 32, 32).astype(np.float32) / scale
        shaped = (shaped - mean[None, :, None, None]) / std[None, :, None, None]
        # Eight spatial tissues: a 4x2 grid of 8x16 RGB patches.  Flattening
        # this order means each consecutive 384-value field block owns one
        # spatial patch rather than an arbitrary channel-major segment.
        patched = shaped.reshape(len(images), 3, 4, 8, 2, 16)
        patched = patched.transpose(0, 2, 4, 1, 3, 5).reshape(len(images), 8, 384)
        return patched.reshape(len(images), 3072).astype(np.float32)
    return np.asarray(value, dtype=np.float32)

def _sigmoid(value: np.ndarray) -> np.ndarray:
    clipped = np.clip(value, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _softmax(value: np.ndarray) -> np.ndarray:
    shifted = value - np.max(value, axis=-1, keepdims=True)
    exp = np.exp(np.clip(shifted, -60.0, 60.0))
    return exp / np.maximum(exp.sum(axis=-1, keepdims=True), 1e-12)


def _round_tensor(value: np.ndarray, decimals: int = 7) -> np.ndarray:
    return np.round(np.asarray(value, dtype=np.float32), decimals=decimals).astype(np.float32)


class Native10Dendritron:
    """Native10-derived sparse Dendritron implemented with portable NumPy.

    It is intentionally a tensor-level runtime rather than a PyTorch baseline
    trainer.  The package imports an existing canonical checkpoint, assigns
    bounded tissue mutations, and composes verified deltas back into that model.
    """

    TENSOR_NAMES = (
        "field_weights", "field_bias", "field_mixer", "field_mixer_bias",
        "scout_weights", "scout_bias", "scout_centers", "scout_quality",
        "category_readout", "category_bias",
        "expert_branch_weights", "expert_branch_bias", "expert_branch_centers",
        "expert_branch_readout", "expert_local_bias", "expert_gate_weights",
        "expert_gate_bias", "expert_health", "expert_branch_health",
        "associative_memory", "associative_counts", "associative_strength", "rotation_phase",
    )

    def __init__(self, config: Native10Config, tensors: dict[str, np.ndarray], *, lineage: list[dict[str, Any]] | None = None):
        self.config = config
        self.tensors = {name: np.asarray(value) for name, value in tensors.items()}
        self.lineage = list(lineage or [])
        self._root_cache: str | None = None
        self._validate()

    @classmethod
    def initialize(cls, config: Native10Config, seed: int | None = None) -> "Native10Dendritron":
        """Initialize topology only; this is not baseline training."""
        rng = np.random.default_rng(config.seed if seed is None else int(seed))
        scale = 1.0 / math.sqrt(config.representation_width)
        b = config.field_blocks
        bw = config.field_block_width
        bo = config.field_block_output
        c = config.categories
        s = config.scout_count
        e = config.experts_per_category
        br = config.expert_branches
        r = config.representation_width
        d = config.branch_width
        l = config.classes_per_category
        sp = config.scout_projection_width

        category_readout = np.zeros((c, r), dtype=np.float32)
        for category in range(c):
            category_readout[category, category % r] = 2.0

        tensors: dict[str, np.ndarray] = {
            "field_weights": rng.normal(0, 1 / math.sqrt(bw), size=(b, bw, bo)).astype(np.float32),
            "field_bias": np.zeros((b, bo), dtype=np.float32),
            "field_mixer": (np.eye(r, dtype=np.float32) + rng.normal(0, 0.01, size=(r, r))).astype(np.float32),
            "field_mixer_bias": np.zeros(r, dtype=np.float32),
            "scout_weights": rng.normal(0, scale, size=(s, r, sp)).astype(np.float32),
            "scout_bias": np.zeros((s, sp), dtype=np.float32),
            "scout_centers": rng.normal(0, 0.05, size=(s, sp)).astype(np.float32),
            "scout_quality": np.ones(s, dtype=np.float32),
            "category_readout": category_readout,
            "category_bias": np.zeros(c, dtype=np.float32),
            "expert_branch_weights": rng.normal(0, scale, size=(c, e, br, r, d)).astype(np.float32),
            "expert_branch_bias": np.zeros((c, e, br, d), dtype=np.float32),
            "expert_branch_centers": rng.normal(0, 0.05, size=(c, e, br, d)).astype(np.float32),
            "expert_branch_readout": rng.normal(0, 0.02, size=(c, e, br, d, l)).astype(np.float32),
            "expert_local_bias": np.zeros((c, e, l), dtype=np.float32),
            "expert_gate_weights": rng.normal(0, scale, size=(c, e, r)).astype(np.float32),
            "expert_gate_bias": np.zeros((c, e), dtype=np.float32),
            "expert_health": np.ones((c, e), dtype=np.float32),
            "expert_branch_health": np.ones((c, e, br), dtype=np.float32),
            "associative_memory": np.zeros((config.classes, r), dtype=np.float32),
            "associative_counts": np.zeros(config.classes, dtype=np.int64),
            "associative_strength": np.full(config.classes, config.memory_strength_init, dtype=np.float32),
            "rotation_phase": np.zeros(c, dtype=np.int64),
        }
        return cls(config, tensors, lineage=[{
            "event": "topology-initialized",
            "note": "No baseline training was performed by DendriSwarm.",
            "seed": int(config.seed if seed is None else seed),
        }])

    def copy(self) -> "Native10Dendritron":
        return Native10Dendritron(self.config, {k: v.copy() for k, v in self.tensors.items()}, lineage=copy.deepcopy(self.lineage))

    def _validate(self) -> None:
        cfg = self.config
        expected = {
            "field_weights": (cfg.field_blocks, cfg.field_block_width, cfg.field_block_output),
            "field_bias": (cfg.field_blocks, cfg.field_block_output),
            "field_mixer": (cfg.representation_width, cfg.representation_width),
            "field_mixer_bias": (cfg.representation_width,),
            "scout_weights": (cfg.scout_count, cfg.representation_width, cfg.scout_projection_width),
            "scout_bias": (cfg.scout_count, cfg.scout_projection_width),
            "scout_centers": (cfg.scout_count, cfg.scout_projection_width),
            "scout_quality": (cfg.scout_count,),
            "category_readout": (cfg.categories, cfg.representation_width),
            "category_bias": (cfg.categories,),
            "expert_branch_weights": (cfg.categories, cfg.experts_per_category, cfg.expert_branches, cfg.representation_width, cfg.branch_width),
            "expert_branch_bias": (cfg.categories, cfg.experts_per_category, cfg.expert_branches, cfg.branch_width),
            "expert_branch_centers": (cfg.categories, cfg.experts_per_category, cfg.expert_branches, cfg.branch_width),
            "expert_branch_readout": (cfg.categories, cfg.experts_per_category, cfg.expert_branches, cfg.branch_width, cfg.classes_per_category),
            "expert_local_bias": (cfg.categories, cfg.experts_per_category, cfg.classes_per_category),
            "expert_gate_weights": (cfg.categories, cfg.experts_per_category, cfg.representation_width),
            "expert_gate_bias": (cfg.categories, cfg.experts_per_category),
            "expert_health": (cfg.categories, cfg.experts_per_category),
            "expert_branch_health": (cfg.categories, cfg.experts_per_category, cfg.expert_branches),
            "associative_memory": (cfg.classes, cfg.representation_width),
            "associative_counts": (cfg.classes,),
            "associative_strength": (cfg.classes,),
            "rotation_phase": (cfg.categories,),
        }
        if set(self.tensors) != set(expected):
            missing = sorted(set(expected) - set(self.tensors))
            extra = sorted(set(self.tensors) - set(expected))
            raise ValueError(f"checkpoint tensor set mismatch; missing={missing}, extra={extra}")
        for name, shape in expected.items():
            value = self.tensors[name]
            if tuple(value.shape) != shape:
                raise ValueError(f"{name} shape mismatch: expected {shape}, got {value.shape}")
            if value.dtype.kind == "f" and not np.isfinite(value).all():
                raise ValueError(f"{name} contains non-finite values")
        if np.any(self.tensors["expert_health"] < 0) or np.any(self.tensors["expert_health"] > 1):
            raise ValueError("expert health must be in [0,1]")
        if np.any(self.tensors["expert_branch_health"] < 0) or np.any(self.tensors["expert_branch_health"] > 1):
            raise ValueError("branch health must be in [0,1]")
        if np.any(self.tensors["associative_strength"] < 0) or np.any(self.tensors["associative_strength"] > 2):
            raise ValueError("associative strength must be in [0,2]")
        if np.any(self.tensors["rotation_phase"] < 0) or np.any(self.tensors["rotation_phase"] >= cfg.rotation_groups):
            raise ValueError("rotation phase is invalid")

    @property
    def parameter_count(self) -> int:
        return int(sum(value.size for value in self.tensors.values() if value.dtype.kind == "f"))

    @property
    def root(self) -> str:
        if self._root_cache is None:
            digest = hashlib.sha256()
            digest.update(b"dendriswarm.native10-model-root.v1\0")
            digest.update(json.dumps(self.config.as_dict(), sort_keys=True, separators=(",", ":")).encode("utf-8"))
            for name in self.TENSOR_NAMES:
                value = np.ascontiguousarray(self.tensors[name])
                digest.update(name.encode("utf-8") + b"\0")
                digest.update(str(value.dtype).encode("ascii") + b"\0")
                digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode("ascii") + b"\0")
                digest.update(memoryview(value).cast("B"))
            self._root_cache = digest.hexdigest()
        return self._root_cache

    def artifact(self, *, include_lineage: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "format": CHECKPOINT_FORMAT,
            "engine": ENGINE,
            "config": self.config.as_dict(),
            "parameter_count": self.parameter_count,
            "model_root": self.root,
            "tensors": {name: encode_array(self.tensors[name]) for name in self.TENSOR_NAMES},
        }
        if include_lineage:
            value["lineage"] = copy.deepcopy(self.lineage)
        value["sha256"] = checkpoint_hash(value)
        return value

    @classmethod
    def from_artifact(cls, artifact: dict[str, Any]) -> "Native10Dendritron":
        if artifact.get("format") != CHECKPOINT_FORMAT or artifact.get("engine") != ENGINE:
            raise ValueError("unsupported Native10 checkpoint format")
        if artifact.get("sha256") != checkpoint_hash(artifact):
            raise ValueError("Native10 checkpoint hash mismatch")
        config = Native10Config.from_dict(dict(artifact["config"]))
        tensors = {name: decode_array(artifact["tensors"][name]) for name in cls.TENSOR_NAMES}
        model = cls(config, tensors, lineage=list(artifact.get("lineage", [])))
        if int(artifact.get("parameter_count", -1)) != model.parameter_count:
            raise ValueError("checkpoint parameter count mismatch")
        if artifact.get("model_root") != model.root:
            raise ValueError("checkpoint model root mismatch")
        return model

    def encode(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        single = x.ndim == 1
        if single:
            x = x[None, :]
        if x.ndim != 2 or x.shape[1] != self.config.input_width:
            raise ValueError(f"expected input width {self.config.input_width}")
        blocks = x.reshape(len(x), self.config.field_blocks, self.config.field_block_width)
        projected = np.einsum("nbi,bio->nbo", blocks, self.tensors["field_weights"], optimize=True)
        projected = np.tanh(projected + self.tensors["field_bias"][None, :, :]).reshape(len(x), -1)
        representation = np.tanh(projected @ self.tensors["field_mixer"] + self.tensors["field_mixer_bias"])
        return representation[0] if single else representation

    def route_scores(self, representations: np.ndarray) -> np.ndarray:
        h = np.asarray(representations, dtype=np.float32)
        single = h.ndim == 1
        if single:
            h = h[None, :]
        if h.ndim != 2 or h.shape[1] != self.config.representation_width:
            raise ValueError("representation width mismatch")
        projection = np.einsum("nr,srd->nsd", h, self.tensors["scout_weights"], optimize=True)
        projection = np.tanh(projection + self.tensors["scout_bias"][None, :, :])
        distance = np.mean((projection - self.tensors["scout_centers"][None, :, :]) ** 2, axis=-1)
        affinity = np.exp(-distance) * self.tensors["scout_quality"][None, :]
        shaped = affinity.reshape(len(h), self.config.categories, self.config.scouts_per_category)
        top = min(3, self.config.scouts_per_category)
        scout_score = np.partition(shaped, -top, axis=2)[:, :, -top:].mean(axis=2)
        score = scout_score + h @ self.tensors["category_readout"].T + self.tensors["category_bias"]
        return score[0] if single else score

    def top_categories_from_scores(self, scores: np.ndarray) -> np.ndarray:
        values = np.asarray(scores, dtype=np.float32)
        single = values.ndim == 1
        if single:
            values = values[None, :]
        if values.ndim != 2 or values.shape[1] != self.config.categories:
            raise ValueError("category score width mismatch")
        k = self.config.top_categories
        selected = np.argpartition(values, kth=self.config.categories - k, axis=1)[:, -k:]
        order = np.take_along_axis(values, selected, axis=1).argsort(axis=1)[:, ::-1]
        selected = np.take_along_axis(selected, order, axis=1)
        return selected[0] if single else selected

    def top_categories(self, representations: np.ndarray) -> np.ndarray:
        return self.top_categories_from_scores(self.route_scores(representations))

    def _category_logits(self, h: np.ndarray, category: int) -> np.ndarray:
        cfg = self.config
        c = int(category)
        w = self.tensors["expert_branch_weights"][c]
        bias = self.tensors["expert_branch_bias"][c]
        centers = self.tensors["expert_branch_centers"][c]
        z = np.tanh(np.einsum("nr,ebrd->nebd", h, w, optimize=True) + bias[None, :, :, :])
        distance = np.mean((z - centers[None, :, :, :]) ** 2, axis=-1)
        branch_evidence = np.exp(-distance) * self.tensors["expert_branch_health"][c][None, :, :]
        branch_logits = np.einsum(
            "nebd,ebdl->nebl", z, self.tensors["expert_branch_readout"][c], optimize=True
        )
        branch_weight = branch_evidence / np.maximum(branch_evidence.sum(axis=2, keepdims=True), 1e-8)
        expert_logits = (branch_logits * branch_weight[..., None]).sum(axis=2)
        expert_logits += self.tensors["expert_local_bias"][c][None, :, :]
        gate = _sigmoid(h @ self.tensors["expert_gate_weights"][c].T + self.tensors["expert_gate_bias"][c])
        gate *= self.tensors["expert_health"][c][None, :]
        gate /= np.maximum(gate.sum(axis=1, keepdims=True), 1e-8)
        local = (expert_logits * gate[..., None]).sum(axis=1)
        start = c * cfg.classes_per_category
        memory = self.tensors["associative_memory"][start:start + cfg.classes_per_category]
        counts = self.tensors["associative_counts"][start:start + cfg.classes_per_category]
        normalized_h = h / np.maximum(np.linalg.norm(h, axis=1, keepdims=True), 1e-8)
        normalized_memory = memory / np.maximum(np.linalg.norm(memory, axis=1, keepdims=True), 1e-8)
        memory_score = normalized_h @ normalized_memory.T
        memory_score[:, counts == 0] = 0.0
        strength = self.tensors["associative_strength"][start:start + cfg.classes_per_category]
        return local + memory_score * strength[None, :]

    def category_logits(self, representations: np.ndarray, category: int) -> np.ndarray:
        """Return one colony's local class logits for routing diagnostics."""
        h = np.asarray(representations, dtype=np.float32)
        single = h.ndim == 1
        if single:
            h = h[None, :]
        if h.ndim != 2 or h.shape[1] != self.config.representation_width:
            raise ValueError("representation width mismatch")
        if not 0 <= int(category) < self.config.categories:
            raise ValueError("category is invalid")
        logits = self._category_logits(h, int(category))
        return logits[0] if single else logits

    def scores_from_representation(self, representations: np.ndarray) -> np.ndarray:
        h = np.asarray(representations, dtype=np.float32)
        single = h.ndim == 1
        if single:
            h = h[None, :]
        route = self.route_scores(h)
        selected = self.top_categories(h)
        # Every class retains a bounded recovery path through category routing and
        # associative memory. Detailed expert computation remains sparse.
        logits = np.repeat(route, self.config.classes_per_category, axis=1).astype(np.float32) - 6.0
        memory = self.tensors["associative_memory"]
        counts = self.tensors["associative_counts"]
        normalized_h = h / np.maximum(np.linalg.norm(h, axis=1, keepdims=True), 1e-8)
        normalized_memory = memory / np.maximum(np.linalg.norm(memory, axis=1, keepdims=True), 1e-8)
        memory_score = normalized_h @ normalized_memory.T
        memory_score[:, counts == 0] = 0.0
        logits += memory_score * self.tensors["associative_strength"][None, :]

        # Expand routing on ambiguous boundaries rather than making a top-k miss
        # unrecoverable. The normal path remains top_categories sparse.
        order = np.argsort(route, axis=1)[:, ::-1]
        for row in range(len(h)):
            chosen = list(selected[row].astype(int))
            boundary = route[row, chosen[-1]]
            for candidate in order[row]:
                candidate = int(candidate)
                if candidate in chosen:
                    continue
                if len(chosen) >= self.config.max_routed_categories:
                    break
                if boundary - route[row, candidate] <= self.config.routing_expansion_margin:
                    chosen.append(candidate)
                else:
                    break
            for category in chosen:
                local = self._category_logits(h[row:row + 1], category)[0]
                start = category * self.config.classes_per_category
                stop = start + self.config.classes_per_category
                logits[row, start:stop] = local + route[row, category]
        probabilities = _softmax(logits)
        return probabilities[0] if single else probabilities

    def scores(self, x: np.ndarray) -> np.ndarray:
        return self.scores_from_representation(self.encode(x))

    def predict_from_representation(self, representations: np.ndarray) -> np.ndarray:
        scores = self.scores_from_representation(representations)
        return np.asarray(int(scores.argmax())) if scores.ndim == 1 else scores.argmax(axis=1)

    def predict(self, x: np.ndarray) -> np.ndarray:
        scores = self.scores(x)
        return np.asarray(int(scores.argmax())) if scores.ndim == 1 else scores.argmax(axis=1)

    def component_bundle(self, operation: str, category: int) -> dict[str, Any]:
        """Return only the tissue required to generate one bounded v0.6 mutation."""
        operation = canonical_operation(operation)
        target = int(category)
        cfg = self.config
        tensors: dict[str, Any] = {}
        if operation in {"expert_train", "branch_train", "repair"}:
            if not 0 <= target < cfg.categories:
                raise ValueError("category is outside model range")
            names = (
                "expert_branch_weights", "expert_branch_bias", "expert_branch_centers",
                "expert_branch_readout", "expert_local_bias", "expert_gate_weights",
                "expert_gate_bias", "expert_health", "expert_branch_health", "rotation_phase",
            )
            for name in names:
                tensors[name] = encode_array(self.tensors[name][target])
        elif operation == "scout_train":
            if not 0 <= target < cfg.categories:
                raise ValueError("category is outside model range")
            start = target * cfg.scouts_per_category
            stop = start + cfg.scouts_per_category
            for name in ("scout_weights", "scout_bias", "scout_centers", "scout_quality"):
                tensors[name] = encode_array(self.tensors[name][start:stop])
            tensors["category_readout"] = encode_array(self.tensors["category_readout"])
            tensors["category_bias"] = encode_array(self.tensors["category_bias"])
        elif operation == "memory_train":
            if not 0 <= target < cfg.categories:
                raise ValueError("category is outside model range")
            start = target * cfg.classes_per_category
            stop = start + cfg.classes_per_category
            for name in ("associative_memory", "associative_counts", "associative_strength"):
                tensors[name] = encode_array(self.tensors[name][start:stop])
        elif operation == "field_train":
            if not 0 <= target < cfg.field_blocks:
                raise ValueError("field block is outside model range")
            for name in (
                "field_weights", "field_bias", "field_mixer", "field_mixer_bias",
                "category_readout", "category_bias",
            ):
                tensors[name] = encode_array(self.tensors[name])
        else:
            raise ValueError(f"unsupported Native10 mutation operation: {operation}")
        bundle = {
            "format": BUNDLE_FORMAT,
            "engine": ENGINE,
            "base_root": self.root,
            "operation": operation,
            "category": target,
            "target_kind": "field-block" if operation == "field_train" else "category",
            "config": self.config.as_dict(),
            "tensors": tensors,
            "schema_hash": operation_schema_hash(operation, self.config),
        }
        bundle["sha256"] = bundle_hash(bundle)
        return bundle

    def apply_delta(self, delta: dict[str, Any], *, contribution: dict[str, Any] | None = None) -> "Native10Dendritron":
        if delta.get("base_root") != self.root:
            raise ValueError("delta parent is not the current canonical root")
        parent_bundle = self.component_bundle(delta["operation"], int(delta["category"]))
        validate_delta(delta, parent_bundle)
        if Native10Config.from_dict(delta["config"]) != self.config:
            raise ValueError("delta model configuration mismatch")
        updated = self.copy()
        for patch in delta["patches"]:
            name = str(patch["tensor"])
            selector = normalize_selector(patch["selector"], updated.tensors[name].shape)
            value = decode_array(patch["value"])
            assign_patch(updated.tensors[name], selector, value)
        updated._root_cache = None
        event = {
            "event": "verified-trainable-tissue-promotion",
            "delta_hash": delta["sha256"],
            "base_root": delta["base_root"],
            "operation": delta["operation"],
            "category": int(delta["category"]),
            "write_set": list(delta["write_set"]),
            "changed_parameters": int(delta["changed_parameters"]),
            **(contribution or {}),
        }
        updated.lineage.append(event)
        updated._validate()
        return updated

    def export_int8(self) -> dict[str, Any]:
        tensors: dict[str, Any] = {}
        for name, value in self.tensors.items():
            if value.dtype.kind != "f":
                tensors[name] = encode_array(value)
                continue
            array = np.asarray(value, dtype=np.float32)
            if array.ndim == 0:
                scale = np.asarray(max(abs(float(array)), 1e-8) / 127.0, dtype=np.float32)
            else:
                reduce_axes = tuple(range(1, array.ndim)) if array.ndim > 1 else ()
                maximum = np.max(np.abs(array), axis=reduce_axes, keepdims=True) if reduce_axes else np.max(np.abs(array), keepdims=True)
                scale = np.maximum(maximum / 127.0, 1e-8).astype(np.float32)
            quantized = np.clip(np.rint(array / scale), -127, 127).astype(np.int8)
            tensors[name] = {"q": encode_array(quantized), "scale": encode_array(scale)}
        value = {
            "format": QUANTIZED_FORMAT,
            "source_root": self.root,
            "config": self.config.as_dict(),
            "tensors": tensors,
            "accumulation": "int32",
            "boundaries": "fp32",
        }
        value["sha256"] = content_hash(value)
        return value


def checkpoint_hash(artifact: dict[str, Any]) -> str:
    return content_hash({key: value for key, value in artifact.items() if key != "sha256"})


def bundle_hash(bundle: dict[str, Any]) -> str:
    return content_hash({key: value for key, value in bundle.items() if key != "sha256"})


def delta_hash(delta: dict[str, Any]) -> str:
    return content_hash({key: value for key, value in delta.items() if key != "sha256"})


OPERATION_ALIASES = {
    "expert_refit": "expert_train",
    "branch_lifecycle": "branch_train",
    "scout_refit": "scout_train",
    "memory_update": "memory_train",
}
PROMOTABLE_OPERATIONS = {"expert_train", "branch_train", "repair", "scout_train", "memory_train", "field_train"}


def canonical_operation(operation: str) -> str:
    value = OPERATION_ALIASES.get(str(operation), str(operation))
    if value not in PROMOTABLE_OPERATIONS:
        raise ValueError(f"unsupported Native10 v0.6 operation: {operation}")
    return value


def tensor_shapes(config: Native10Config) -> dict[str, tuple[int, ...]]:
    return {
        "field_weights": (config.field_blocks, config.field_block_width, config.field_block_output),
        "field_bias": (config.field_blocks, config.field_block_output),
        "field_mixer": (config.representation_width, config.representation_width),
        "field_mixer_bias": (config.representation_width,),
        "scout_weights": (config.scout_count, config.representation_width, config.scout_projection_width),
        "scout_bias": (config.scout_count, config.scout_projection_width),
        "scout_centers": (config.scout_count, config.scout_projection_width),
        "scout_quality": (config.scout_count,),
        "category_readout": (config.categories, config.representation_width),
        "category_bias": (config.categories,),
        "expert_branch_weights": (config.categories, config.experts_per_category, config.expert_branches, config.representation_width, config.branch_width),
        "expert_branch_bias": (config.categories, config.experts_per_category, config.expert_branches, config.branch_width),
        "expert_branch_centers": (config.categories, config.experts_per_category, config.expert_branches, config.branch_width),
        "expert_branch_readout": (config.categories, config.experts_per_category, config.expert_branches, config.branch_width, config.classes_per_category),
        "expert_local_bias": (config.categories, config.experts_per_category, config.classes_per_category),
        "expert_gate_weights": (config.categories, config.experts_per_category, config.representation_width),
        "expert_gate_bias": (config.categories, config.experts_per_category),
        "expert_health": (config.categories, config.experts_per_category),
        "expert_branch_health": (config.categories, config.experts_per_category, config.expert_branches),
        "associative_memory": (config.classes, config.representation_width),
        "associative_counts": (config.classes,),
        "associative_strength": (config.classes,),
        "rotation_phase": (config.categories,),
    }


def operation_schema(operation: str, config: Native10Config) -> dict[str, Any]:
    operation = canonical_operation(operation)
    expert_limit = config.active_experts_per_update * (
        config.expert_branches * config.representation_width * config.branch_width
        + config.expert_branches * config.branch_width
        + config.expert_branches * config.branch_width
        + config.expert_branches * config.branch_width * config.classes_per_category
        + config.classes_per_category + config.representation_width + 1 + 1 + config.expert_branches
    ) + 1
    schemas: dict[str, dict[str, Any]] = {
        "expert_train": {
            "tensors": ["expert_branch_weights", "expert_branch_bias", "expert_branch_centers", "expert_branch_readout", "expert_local_bias", "expert_gate_weights", "expert_gate_bias", "expert_health", "expert_branch_health", "rotation_phase"],
            "max_changed_parameters": expert_limit,
            "max_abs_value": 12.0,
        },
        "branch_train": {
            "tensors": ["expert_branch_weights", "expert_branch_bias", "expert_branch_centers", "expert_branch_readout", "expert_branch_health", "expert_health", "rotation_phase"],
            "max_changed_parameters": config.active_experts_per_update * (
                config.expert_branches * config.representation_width * config.branch_width
                + config.expert_branches * config.branch_width * (2 + config.classes_per_category)
                + config.expert_branches + 1
            ) + 1,
            "max_abs_value": 12.0,
        },
        "repair": {
            "tensors": ["expert_branch_weights", "expert_branch_bias", "expert_branch_centers", "expert_branch_readout", "expert_local_bias", "expert_gate_weights", "expert_gate_bias", "expert_health", "expert_branch_health", "rotation_phase"],
            "max_changed_parameters": expert_limit,
            "max_abs_value": 12.0,
        },
        "scout_train": {
            "tensors": ["scout_weights", "scout_bias", "scout_centers", "scout_quality", "category_readout", "category_bias"],
            "max_changed_parameters": config.scouts_per_category * (config.representation_width * config.scout_projection_width + 2 * config.scout_projection_width + 1) + config.representation_width + 1,
            "max_abs_value": 12.0,
        },
        "memory_train": {
            "tensors": ["associative_memory", "associative_counts", "associative_strength"],
            "max_changed_parameters": config.classes_per_category * (config.representation_width + 2),
            "max_abs_value": 12.0,
        },
        "field_train": {
            "tensors": ["field_weights", "field_bias", "field_mixer", "field_mixer_bias", "category_readout", "category_bias"],
            "max_changed_parameters": config.field_block_width * config.field_block_output + config.field_block_output + config.field_block_output * config.representation_width + config.representation_width + config.categories * config.representation_width + config.categories,
            "max_abs_value": 12.0,
        },
    }
    return schemas[operation]


def operation_schema_hash(operation: str, config: Native10Config) -> str:
    return content_hash({"version": 1, "operation": canonical_operation(operation), "config": config.as_dict(), "schema": operation_schema(operation, config)})


def normalize_selector(selector: Any, shape: tuple[int, ...]) -> tuple[int | tuple[int, int], ...]:
    if not isinstance(selector, list) or len(selector) != len(shape):
        raise ValueError("patch selector rank mismatch")
    normalized: list[int | tuple[int, int]] = []
    for item, size in zip(selector, shape, strict=True):
        if isinstance(item, int):
            if not 0 <= item < size:
                raise ValueError("patch index is outside tensor shape")
            normalized.append(int(item))
        elif isinstance(item, list) and len(item) == 2:
            start, stop = int(item[0]), int(item[1])
            if not 0 <= start < stop <= size:
                raise ValueError("patch slice is outside tensor shape")
            normalized.append((start, stop))
        else:
            raise ValueError("patch selector entries must be integers or [start, stop]")
    return tuple(normalized)


def selector_shape(selector: tuple[int | tuple[int, int], ...]) -> tuple[int, ...]:
    return tuple(stop - start for item in selector if isinstance(item, tuple) for start, stop in [item])


def selector_index(selector: tuple[int | tuple[int, int], ...]) -> tuple[Any, ...]:
    return tuple(item if isinstance(item, int) else slice(item[0], item[1]) for item in selector)


def extract_patch(array: np.ndarray, selector: list[Any]) -> np.ndarray:
    normalized = normalize_selector(selector, tuple(array.shape))
    return np.asarray(array[selector_index(normalized)]).copy()


def assign_patch(array: np.ndarray, selector: tuple[int | tuple[int, int], ...], value: np.ndarray) -> None:
    expected = selector_shape(selector)
    value_array = np.asarray(value)
    if tuple(value_array.shape) != expected:
        if expected == () and value_array.shape == ():
            pass
        else:
            raise ValueError(f"patch value shape mismatch: expected {expected}, got {value_array.shape}")
    array[selector_index(selector)] = value_array


def selector_json(*items: int | tuple[int, int]) -> list[Any]:
    return [item if isinstance(item, int) else [item[0], item[1]] for item in items]


def patch_key(tensor: str, selector: list[Any]) -> str:
    return f"{tensor}:" + json.dumps(selector, separators=(",", ":"))


def make_patch(tensor: str, selector: list[Any], value: np.ndarray) -> dict[str, Any]:
    return {"tensor": tensor, "selector": selector, "value": encode_array(_round_tensor(value)) if np.asarray(value).dtype.kind == "f" else encode_array(np.asarray(value))}


def _selector_intervals(selector: list[Any]) -> list[tuple[int, int]]:
    return [(int(item), int(item) + 1) if isinstance(item, int) else (int(item[0]), int(item[1])) for item in selector]


def patches_overlap(left: dict[str, Any], right: dict[str, Any]) -> bool:
    if left["tensor"] != right["tensor"]:
        return False
    return all(max(a0, b0) < min(a1, b1) for (a0, a1), (b0, b1) in zip(_selector_intervals(left["selector"]), _selector_intervals(right["selector"]), strict=True))


def deltas_conflict(left: dict[str, Any], right: dict[str, Any]) -> bool:
    validate_delta(left)
    validate_delta(right)
    return any(patches_overlap(a, b) for a in left["patches"] for b in right["patches"])


def parameter_reachability(config: Native10Config) -> dict[str, Any]:
    """Machine-readable proof that each persistent tensor family has an owner operation."""
    shapes = tensor_shapes(config)
    owners = {
        "field_weights": ["field_train"], "field_bias": ["field_train"], "field_mixer": ["field_train"], "field_mixer_bias": ["field_train"],
        "scout_weights": ["scout_train"], "scout_bias": ["scout_train"], "scout_centers": ["scout_train"], "scout_quality": ["scout_train"],
        "category_readout": ["field_train", "scout_train"], "category_bias": ["field_train", "scout_train"],
        "expert_branch_weights": ["expert_train", "branch_train", "repair"], "expert_branch_bias": ["expert_train", "branch_train", "repair"],
        "expert_branch_centers": ["expert_train", "branch_train", "repair"], "expert_branch_readout": ["expert_train", "branch_train", "repair"],
        "expert_local_bias": ["expert_train", "repair"], "expert_gate_weights": ["expert_train", "repair"], "expert_gate_bias": ["expert_train", "repair"],
        "expert_health": ["expert_train", "branch_train", "repair"], "expert_branch_health": ["expert_train", "branch_train", "repair"],
        "associative_memory": ["memory_train"], "associative_counts": ["memory_train"], "associative_strength": ["memory_train"],
        "rotation_phase": ["expert_train", "branch_train", "repair"],
    }
    state_names = {"associative_counts", "rotation_phase"}
    float_names = set(shapes) - state_names
    float_total = sum(math.prod(shapes[name]) for name in float_names)
    float_reachable = sum(math.prod(shapes[name]) for name in float_names if owners.get(name))
    state_total = sum(math.prod(shapes[name]) for name in state_names)
    operation_coverage = {
        operation: int(sum(math.prod(shapes[name]) for name, operations in owners.items() if operation in operations))
        for operation in sorted(PROMOTABLE_OPERATIONS)
    }
    return {
        "format": "dendriswarm.native10-parameter-reachability.v2",
        "trainable_float_parameters": int(float_total),
        "reachable_float_parameters": int(float_reachable),
        "reachable_float_fraction": float(float_reachable / max(1, float_total)),
        "persistent_state_elements": int(state_total),
        "total_persistent_tensor_elements": int(float_total + state_total),
        "operation_coverage_elements": operation_coverage,
        "tensors": {
            name: {
                "shape": list(shape),
                "kind": "state" if name in state_names else "trainable-float",
                "operations": owners.get(name, []),
                "reachable": bool(owners.get(name)),
            }
            for name, shape in shapes.items()
        },
        "all_trainable_parameter_families_reachable": all(owners.get(name) for name in float_names),
        "all_persistent_tensor_families_owned": all(owners.get(name) for name in shapes),
    }


def delta_consensus_hash(delta: dict[str, Any], decimals: int = 5) -> str:
    """Architecture-tolerant replay fingerprint; candidate search does not require equality."""
    validate_delta(delta)
    digest = hashlib.sha256()
    digest.update(b"dendriswarm.native10-delta-replay.v2\0")
    for key in ("base_root", "bundle_hash", "operation", "category"):
        digest.update(json.dumps(delta[key], sort_keys=True, separators=(",", ":")).encode() + b"\0")
    for patch in sorted(delta["patches"], key=lambda item: patch_key(item["tensor"], item["selector"])):
        value = decode_array(patch["value"])
        if value.dtype.kind == "f":
            value = np.round(value.astype(np.float64), decimals=decimals).astype(np.float32)
        digest.update(patch_key(patch["tensor"], patch["selector"]).encode() + b"\0")
        digest.update(memoryview(np.ascontiguousarray(value)).cast("B"))
    return digest.hexdigest()


def validate_bundle(bundle: dict[str, Any]) -> None:
    if bundle.get("format") != BUNDLE_FORMAT or bundle.get("engine") != ENGINE:
        raise ValueError("unsupported Native10 v0.6 work bundle")
    if bundle.get("sha256") != bundle_hash(bundle):
        raise ValueError("Native10 work bundle hash mismatch")
    config = Native10Config.from_dict(dict(bundle["config"]))
    operation = canonical_operation(bundle["operation"])
    target = int(bundle["category"])
    limit = config.field_blocks if operation == "field_train" else config.categories
    if not 0 <= target < limit:
        raise ValueError("bundle target is invalid")
    if bundle.get("schema_hash") != operation_schema_hash(operation, config):
        raise ValueError("bundle operation schema commitment mismatch")
    for encoded in bundle["tensors"].values():
        decode_array(encoded)


def _patches_allowed(delta: dict[str, Any], config: Native10Config) -> None:
    operation = canonical_operation(delta["operation"])
    schema = operation_schema(operation, config)
    target = int(delta["category"])
    shapes = tensor_shapes(config)
    active = [int(item) for item in delta.get("metrics", {}).get("active_experts", [])]
    seen: set[str] = set()
    changed = 0
    for patch in delta["patches"]:
        name = str(patch.get("tensor"))
        if name not in schema["tensors"] or name not in shapes:
            raise ValueError(f"{operation} cannot update tensor {name}")
        selector = normalize_selector(patch.get("selector"), shapes[name])
        value = decode_array(patch["value"])
        if tuple(value.shape) != selector_shape(selector):
            raise ValueError("patch value shape does not match selector")
        key = patch_key(name, patch["selector"])
        if key in seen:
            raise ValueError("duplicate patch selector")
        seen.add(key)
        changed += int(value.size)
        if value.dtype.kind == "f" and float(np.max(np.abs(value), initial=0.0)) > float(schema["max_abs_value"]):
            raise ValueError("patch exceeds committed magnitude bound")
        # Territory and operation-specific selector enforcement.
        first = selector[0]
        if operation in {"expert_train", "branch_train", "repair"}:
            if name == "rotation_phase":
                if first != target:
                    raise ValueError("rotation phase patch escapes category")
            else:
                if first != target or len(selector) < 2 or not isinstance(selector[1], int) or selector[1] not in active:
                    raise ValueError("expert patch escapes active category experts")
        elif operation == "scout_train":
            if name.startswith("scout_"):
                start = target * config.scouts_per_category
                stop = start + config.scouts_per_category
                if not isinstance(first, tuple) or first != (start, stop):
                    raise ValueError("scout patch escapes category scout range")
            elif name in {"category_readout", "category_bias"} and first != target:
                raise ValueError("scout patch escapes category readout")
        elif operation == "memory_train":
            start = target * config.classes_per_category
            stop = start + config.classes_per_category
            if not isinstance(first, tuple) or first != (start, stop):
                raise ValueError("memory patch escapes category class range")
        elif operation == "field_train":
            block_start = target * config.field_block_output
            block_stop = block_start + config.field_block_output
            if name in {"field_weights", "field_bias"} and first != target:
                raise ValueError("field patch escapes selected block")
            if name == "field_mixer" and (not isinstance(first, tuple) or first != (block_start, block_stop)):
                raise ValueError("field mixer patch escapes selected block rows")
            if name == "field_mixer_bias" and (not isinstance(first, tuple) or first != (0, config.representation_width)):
                raise ValueError("field mixer bias selector is incomplete")
            if name in {"category_readout", "category_bias"} and (not isinstance(first, tuple) or first[0] != 0):
                raise ValueError("field router patch must cover the committed global slice")
    if changed != int(delta.get("changed_parameters", -1)):
        raise ValueError("delta changed-parameter count mismatch")
    if changed > int(schema["max_changed_parameters"]):
        raise ValueError("delta exceeds operation parameter budget")
    if sorted(seen) != sorted(delta.get("write_set", [])):
        raise ValueError("delta write set is not derived from patches")
    if operation in {"expert_train", "branch_train", "repair"}:
        if len(active) != config.active_experts_per_update or len(set(active)) != len(active):
            raise ValueError("expert operation must update one complete rotation group")
        if not any(p["tensor"] == "rotation_phase" for p in delta["patches"]):
            raise ValueError("expert operation must advance rotation phase")


def expected_write_plan(bundle: dict[str, Any]) -> dict[str, Any]:
    """Derive the exact patch selectors and state transition for one bundle."""
    validate_bundle(bundle)
    config = Native10Config.from_dict(bundle["config"])
    operation = canonical_operation(bundle["operation"])
    target = int(bundle["category"])
    arrays = {name: decode_array(value) for name, value in bundle["tensors"].items()}
    selectors: list[tuple[str, list[Any]]] = []
    active: list[int] = []
    phase_before: int | None = None
    phase_after: int | None = None

    if operation in {"expert_train", "branch_train", "repair"}:
        phase_before = int(np.asarray(arrays["rotation_phase"]).item())
        phase_after = (phase_before + 1) % config.rotation_groups
        if operation == "repair":
            active = np.argsort(np.asarray(arrays["expert_health"]))[: config.active_experts_per_update].astype(int).tolist()
        else:
            start = phase_before * config.active_experts_per_update
            active = list(range(start, start + config.active_experts_per_update))
        names = (
            "expert_branch_weights", "expert_branch_bias", "expert_branch_centers",
            "expert_branch_readout", "expert_health", "expert_branch_health",
        )
        if operation in {"expert_train", "repair"}:
            names = names[:4] + ("expert_local_bias", "expert_gate_weights", "expert_gate_bias") + names[4:]
        for expert in active:
            for name in names:
                selector = selector_json(
                    target, expert, *[(0, size) for size in tensor_shapes(config)[name][2:]]
                )
                selectors.append((name, selector))
        selectors.append(("rotation_phase", selector_json(target)))
    elif operation == "scout_train":
        start = target * config.scouts_per_category
        stop = start + config.scouts_per_category
        for name in ("scout_weights", "scout_bias", "scout_centers", "scout_quality"):
            shape = tensor_shapes(config)[name]
            selectors.append((name, selector_json((start, stop), *[(0, size) for size in shape[1:]])))
        selectors.append(("category_readout", selector_json(target, (0, config.representation_width))))
        selectors.append(("category_bias", selector_json(target)))
    elif operation == "memory_train":
        start = target * config.classes_per_category
        stop = start + config.classes_per_category
        for name in ("associative_memory", "associative_counts", "associative_strength"):
            shape = tensor_shapes(config)[name]
            selectors.append((name, selector_json((start, stop), *[(0, size) for size in shape[1:]])))
    elif operation == "field_train":
        row_start = target * config.field_block_output
        row_stop = row_start + config.field_block_output
        selectors.extend([
            ("field_weights", selector_json(target, (0, config.field_block_width), (0, config.field_block_output))),
            ("field_bias", selector_json(target, (0, config.field_block_output))),
            ("field_mixer", selector_json((row_start, row_stop), (0, config.representation_width))),
            ("field_mixer_bias", selector_json((0, config.representation_width))),
            ("category_readout", selector_json((0, config.categories), (0, config.representation_width))),
            ("category_bias", selector_json((0, config.categories))),
        ])
    else:  # canonical_operation already makes this unreachable.
        raise ValueError("unsupported operation write plan")

    return {
        "operation": operation,
        "target": target,
        "active_experts": active,
        "rotation_phase_before": phase_before,
        "rotation_phase_after": phase_after,
        "write_set": sorted(patch_key(name, selector) for name, selector in selectors),
    }


def validate_delta(delta: dict[str, Any], bundle: dict[str, Any] | None = None) -> None:
    if delta.get("format") != DELTA_FORMAT or delta.get("engine") != ENGINE:
        raise ValueError("unsupported Native10 v0.6 delta")
    if delta.get("sha256") != delta_hash(delta):
        raise ValueError("Native10 delta hash mismatch")
    config = Native10Config.from_dict(dict(delta["config"]))
    if len(str(delta.get("base_root", ""))) != 64:
        raise ValueError("delta base root is invalid")
    if delta.get("schema_hash") != operation_schema_hash(delta["operation"], config):
        raise ValueError("delta operation schema commitment mismatch")
    if not isinstance(delta.get("patches"), list) or not delta["patches"]:
        raise ValueError("delta contains no sparse patches")
    _patches_allowed(delta, config)
    if bundle is not None:
        validate_bundle(bundle)
        for key in ("base_root", "sha256", "operation", "category", "schema_hash"):
            expected_key = "bundle_hash" if key == "sha256" else key
            if delta.get(expected_key) != bundle.get(key):
                raise ValueError(f"delta is not bound to bundle field {key}")
        plan = expected_write_plan(bundle)
        if sorted(delta.get("write_set", [])) != plan["write_set"]:
            raise ValueError("delta does not implement the exact operation write plan")
        metrics = delta.get("metrics", {})
        if [int(value) for value in metrics.get("active_experts", [])] != plan["active_experts"]:
            raise ValueError("delta active experts do not match the bundle-owned update group")
        if plan["rotation_phase_before"] is not None:
            if int(metrics.get("rotation_phase_before", -1)) != plan["rotation_phase_before"]:
                raise ValueError("delta rotation phase before does not match the bundle")
            if int(metrics.get("rotation_phase_after", -1)) != plan["rotation_phase_after"]:
                raise ValueError("delta rotation phase after does not match the operation transition")
            rotation_patch = next(
                patch for patch in delta["patches"] if patch["tensor"] == "rotation_phase"
            )
            if int(np.asarray(decode_array(rotation_patch["value"])).item()) != plan["rotation_phase_after"]:
                raise ValueError("delta rotation state value does not match the operation transition")


def _bundle_arrays(bundle: dict[str, Any]) -> tuple[Native10Config, dict[str, np.ndarray]]:
    validate_bundle(bundle)
    return Native10Config.from_dict(bundle["config"]), {name: decode_array(value) for name, value in bundle["tensors"].items()}


def _category_logits_from_bundle(config: Native10Config, tensors: dict[str, np.ndarray], representations: np.ndarray) -> np.ndarray:
    h = np.asarray(representations, dtype=np.float32)
    w = tensors["expert_branch_weights"]
    z = np.tanh(np.einsum("nr,ebrd->nebd", h, w, optimize=True) + tensors["expert_branch_bias"][None, :, :, :])
    distance = np.mean((z - tensors["expert_branch_centers"][None, :, :, :]) ** 2, axis=-1)
    evidence = np.exp(-distance) * tensors["expert_branch_health"][None, :, :]
    branch_logits = np.einsum("nebd,ebdl->nebl", z, tensors["expert_branch_readout"], optimize=True)
    branch_weight = evidence / np.maximum(evidence.sum(axis=2, keepdims=True), 1e-8)
    expert_logits = (branch_logits * branch_weight[..., None]).sum(axis=2) + tensors["expert_local_bias"][None, :, :]
    gate = _sigmoid(h @ tensors["expert_gate_weights"].T + tensors["expert_gate_bias"])
    gate *= tensors["expert_health"][None, :]
    gate /= np.maximum(gate.sum(axis=1, keepdims=True), 1e-8)
    return (expert_logits * gate[..., None]).sum(axis=1)


def _local_labels(labels: np.ndarray, category: int, config: Native10Config) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.int64)
    start = int(category) * config.classes_per_category
    local = labels - start
    if np.any(local < 0) or np.any(local >= config.classes_per_category):
        raise ValueError("mutation shard contains labels outside its category")
    return local


def _metrics(logits: np.ndarray, labels: np.ndarray) -> tuple[int, list[int]]:
    predictions = logits.argmax(axis=1)
    return int((predictions == labels).sum()), [int(((predictions == label) & (labels == label)).sum()) for label in range(logits.shape[1])]


def _softmax_gradient(logits: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, float]:
    probabilities = _softmax(logits)
    n = len(labels)
    loss = -float(np.log(np.maximum(probabilities[np.arange(n), labels], 1e-12)).mean())
    gradient = probabilities
    gradient[np.arange(n), labels] -= 1.0
    gradient /= max(1, n)
    return gradient, loss


def _clip_gradient(value: np.ndarray, limit: float = 5.0) -> np.ndarray:
    norm = float(np.linalg.norm(value))
    return value if norm <= limit or norm == 0 else value * (limit / norm)


def _train_experts(config: Native10Config, tensors: dict[str, np.ndarray], h: np.ndarray, y: np.ndarray, active: np.ndarray, *, seed: int, steps: int, learning_rate: float, train_gates: bool, diversity_strength: float = 0.01) -> list[float]:
    rng = np.random.default_rng(seed)
    losses: list[float] = []
    for step in range(max(1, steps)):
        order = rng.permutation(len(h))
        batch_size = min(len(h), 64)
        for offset in range(0, len(h), batch_size):
            rows = order[offset:offset + batch_size]
            x = h[rows]
            labels = y[rows]
            expert_logits: list[np.ndarray] = []
            cache: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
            for expert in active:
                e = int(expert)
                z = np.tanh(np.einsum("nr,brd->nbd", x, tensors["expert_branch_weights"][e], optimize=True) + tensors["expert_branch_bias"][e][None, :, :])
                diff = z - tensors["expert_branch_centers"][e][None, :, :]
                evidence = np.exp(-np.mean(diff * diff, axis=-1)) * tensors["expert_branch_health"][e][None, :]
                bw = evidence / np.maximum(evidence.sum(axis=1, keepdims=True), 1e-8)
                branch_logits = np.einsum("nbd,bdl->nbl", z, tensors["expert_branch_readout"][e], optimize=True)
                logits = (branch_logits * bw[..., None]).sum(axis=1) + tensors["expert_local_bias"][e]
                expert_logits.append(logits)
                cache.append((z, diff, bw, branch_logits))
            stacked = np.stack(expert_logits, axis=1)
            gate_raw = _sigmoid(x @ tensors["expert_gate_weights"][active].T + tensors["expert_gate_bias"][active])
            gate_raw *= tensors["expert_health"][active][None, :]
            gates = gate_raw / np.maximum(gate_raw.sum(axis=1, keepdims=True), 1e-8)
            combined = (stacked * gates[..., None]).sum(axis=1)
            dcombined, loss = _softmax_gradient(combined, labels)
            losses.append(loss)
            dgate = np.einsum("nl,nel->ne", dcombined, stacked)
            gate_sum = np.maximum(gate_raw.sum(axis=1, keepdims=True), 1e-8)
            draw = (dgate - (dgate * gates).sum(axis=1, keepdims=True)) / gate_sum
            dpre_gate = draw * gate_raw * (1.0 - np.clip(gate_raw, 0.0, 1.0))
            for position, expert in enumerate(active):
                e = int(expert)
                z, diff, bw, branch_logits = cache[position]
                dlogits = dcombined * gates[:, position, None]
                dbranch_logits = dlogits[:, None, :] * bw[..., None]
                grad_readout = np.einsum("nbl,nbd->bdl", dbranch_logits, z, optimize=True)
                dz = np.einsum("nbl,bdl->nbd", dbranch_logits, tensors["expert_branch_readout"][e], optimize=True)
                dbw = np.einsum("nl,nbl->nb", dlogits, branch_logits, optimize=True)
                dlog_evidence = bw * (dbw - (dbw * bw).sum(axis=1, keepdims=True))
                dz += dlog_evidence[..., None] * (-2.0 * diff / config.branch_width)
                grad_centers = np.mean(dlog_evidence[..., None] * (2.0 * diff / config.branch_width), axis=0)
                dpre = dz * (1.0 - z * z)
                grad_w = np.einsum("nr,nbd->brd", x, dpre, optimize=True) / max(1, len(x))
                grad_b = dpre.mean(axis=0)
                grad_local = dlogits.sum(axis=0)
                tensors["expert_branch_weights"][e] -= learning_rate * _clip_gradient(grad_w + 1e-4 * tensors["expert_branch_weights"][e])
                tensors["expert_branch_bias"][e] -= learning_rate * _clip_gradient(grad_b)
                tensors["expert_branch_centers"][e] -= learning_rate * _clip_gradient(grad_centers)
                tensors["expert_branch_readout"][e] -= learning_rate * _clip_gradient(grad_readout + 1e-4 * tensors["expert_branch_readout"][e])
                tensors["expert_local_bias"][e] -= learning_rate * _clip_gradient(grad_local)
                if train_gates:
                    tensors["expert_gate_weights"][e] -= learning_rate * _clip_gradient(x.T @ dpre_gate[:, position] / max(1, len(x)) + 1e-4 * tensors["expert_gate_weights"][e])
                    tensors["expert_gate_bias"][e] -= learning_rate * float(dpre_gate[:, position].mean())
                accuracy = float((stacked[:, position].argmax(axis=1) == labels).mean())
                tensors["expert_health"][e] = np.clip(0.85 * tensors["expert_health"][e] + 0.15 * accuracy, 0.05, 1.0)
                tensors["expert_branch_health"][e] = np.clip(0.9 * tensors["expert_branch_health"][e] + 0.1 * bw.mean(axis=0) * config.expert_branches, 0.05, 1.0)
        # Explicit diversity pressure prevents every expert from collapsing to one gate.
        if train_gates and len(active) > 1:
            weights = tensors["expert_gate_weights"][active]
            mean = weights.mean(axis=0, keepdims=True)
            weights += float(diversity_strength) * (weights - mean)
            weights /= np.maximum(np.linalg.norm(weights, axis=1, keepdims=True), 1e-8)
            tensors["expert_gate_weights"][active] = weights
    return losses


def _train_memory(config: Native10Config, tensors: dict[str, np.ndarray], h: np.ndarray, y: np.ndarray, *, steps: int, learning_rate: float) -> list[float]:
    memory = tensors["associative_memory"].astype(np.float32, copy=True)
    strength = tensors["associative_strength"].astype(np.float32, copy=True)
    counts = tensors["associative_counts"].astype(np.int64, copy=True)
    for label in range(config.classes_per_category):
        rows = h[y == label]
        if len(rows) and counts[label] == 0:
            memory[label] = rows.mean(axis=0)
    losses: list[float] = []
    h_norm = h / np.maximum(np.linalg.norm(h, axis=1, keepdims=True), 1e-8)
    for _ in range(max(1, steps)):
        norms = np.maximum(np.linalg.norm(memory, axis=1, keepdims=True), 1e-8)
        unit = memory / norms
        cosine = h_norm @ unit.T
        logits = cosine * strength[None, :]
        dlogits, loss = _softmax_gradient(logits, y)
        losses.append(loss)
        grad_strength = (dlogits * cosine).sum(axis=0)
        grad_unit = (dlogits * strength[None, :]).T @ h_norm
        grad_memory = (grad_unit - unit * (grad_unit * unit).sum(axis=1, keepdims=True)) / norms
        memory -= learning_rate * _clip_gradient(grad_memory)
        strength = np.clip(strength - learning_rate * _clip_gradient(grad_strength), 0.0, 2.0)
    tensors["associative_memory"] = memory
    tensors["associative_strength"] = strength
    tensors["associative_counts"] = counts + np.bincount(y, minlength=config.classes_per_category)
    return losses


def _train_scouts(config: Native10Config, tensors: dict[str, np.ndarray], h: np.ndarray, labels: np.ndarray, category: int, *, seed: int, steps: int, learning_rate: float, routing_margin: float = 0.5, diversity_strength: float = 0.005) -> list[float]:
    target = (labels // config.classes_per_category == category).astype(np.float32)
    if target.min() == target.max():
        raise ValueError("scout training requires positive and negative categories")
    rng = np.random.default_rng(seed)
    w_read = tensors["category_readout"][category]
    b_read = float(tensors["category_bias"][category])
    losses: list[float] = []
    for _ in range(max(1, steps)):
        logits = h @ w_read + b_read
        probability = _sigmoid(logits)
        error = probability - target
        loss = -float(np.mean(target * np.log(np.maximum(probability, 1e-8)) + (1-target) * np.log(np.maximum(1-probability, 1e-8))))
        losses.append(loss)
        w_read -= learning_rate * _clip_gradient(h.T @ error / len(h) + 1e-4 * w_read)
        b_read -= learning_rate * float(error.mean())
        positives = np.flatnonzero(target == 1)
        negatives = np.flatnonzero(target == 0)
        pair_count = min(len(positives), len(negatives), 64)
        for scout in range(config.scouts_per_category):
            pos = rng.choice(positives, pair_count, replace=len(positives) < pair_count)
            neg = rng.choice(negatives, pair_count, replace=len(negatives) < pair_count)
            W = tensors["scout_weights"][scout]
            bias = tensors["scout_bias"][scout]
            center = tensors["scout_centers"][scout]
            zp = np.tanh(h[pos] @ W + bias)
            zn = np.tanh(h[neg] @ W + bias)
            dp = np.mean((zp-center)**2, axis=1)
            dn = np.mean((zn-center)**2, axis=1)
            active = float(routing_margin) + dp - dn > 0
            if not np.any(active):
                tensors["scout_quality"][scout] = min(2.0, tensors["scout_quality"][scout] * 1.01)
                continue
            ap, an = zp[active], zn[active]
            hp, hn = h[pos[active]], h[neg[active]]
            dzp = 2.0 * (ap-center) / config.scout_projection_width
            dzn = -2.0 * (an-center) / config.scout_projection_width
            dprep = dzp * (1-ap*ap)
            dpren = dzn * (1-an*an)
            grad_w = (hp.T @ dprep + hn.T @ dpren) / max(1, len(ap))
            grad_b = (dprep + dpren).mean(axis=0)
            grad_c = (-2*(ap-center) + 2*(an-center)).mean(axis=0) / config.scout_projection_width
            tensors["scout_weights"][scout] -= learning_rate * _clip_gradient(grad_w + 1e-4*W)
            tensors["scout_bias"][scout] -= learning_rate * _clip_gradient(grad_b)
            tensors["scout_centers"][scout] -= learning_rate * _clip_gradient(grad_c)
            success = float((dp < dn).mean())
            tensors["scout_quality"][scout] = np.clip(0.5 + success, 0.1, 2.0)
    tensors["category_readout"][category] = w_read
    tensors["category_bias"][category] = b_read
    # Orthogonalize scouts lightly to preserve category-internal specialization.
    flat = tensors["scout_weights"].reshape(config.scouts_per_category, -1)
    for index in range(1, len(flat)):
        previous = flat[:index]
        coefficients = (previous @ flat[index]) / np.maximum((previous * previous).sum(axis=1), 1e-8)
        projection = (coefficients[:, None] * previous).sum(axis=0)
        flat[index] -= float(diversity_strength) * projection
    tensors["scout_weights"] = flat.reshape(tensors["scout_weights"].shape)
    return losses


def _train_field(config: Native10Config, tensors: dict[str, np.ndarray], x: np.ndarray, labels: np.ndarray, block: int, *, seed: int, steps: int, learning_rate: float, route_margin: float = 0.0, route_margin_weight: float = 0.0) -> list[float]:
    categories = labels // config.classes_per_category
    rng = np.random.default_rng(seed)
    start = block * config.field_block_output
    stop = start + config.field_block_output
    losses: list[float] = []
    for _ in range(max(1, steps)):
        order = rng.permutation(len(x))
        batch_size = min(64, len(x))
        for offset in range(0, len(x), batch_size):
            rows = order[offset:offset+batch_size]
            batch = x[rows]
            y = categories[rows]
            blocks = batch.reshape(len(batch), config.field_blocks, config.field_block_width)
            projected_pre = np.einsum("nbi,bio->nbo", blocks, tensors["field_weights"], optimize=True) + tensors["field_bias"][None,:,:]
            projected = np.tanh(projected_pre).reshape(len(batch), -1)
            mixed_pre = projected @ tensors["field_mixer"] + tensors["field_mixer_bias"]
            h = np.tanh(mixed_pre)
            logits = h @ tensors["category_readout"].T + tensors["category_bias"]
            dlogits, loss = _softmax_gradient(logits, y)
            if route_margin_weight > 0.0:
                masked = logits.copy()
                masked[np.arange(len(y)), y] = -np.inf
                wrong = masked.argmax(axis=1)
                violation = float(route_margin) + logits[np.arange(len(y)), wrong] - logits[np.arange(len(y)), y]
                active_margin = violation > 0
                if np.any(active_margin):
                    scale = float(route_margin_weight) / max(1, int(active_margin.sum()))
                    rows_margin = np.flatnonzero(active_margin)
                    dlogits[rows_margin, wrong[rows_margin]] += scale
                    dlogits[rows_margin, y[rows_margin]] -= scale
                    loss += float(route_margin_weight) * float(np.maximum(violation, 0.0).mean())
            losses.append(loss)
            grad_readout = dlogits.T @ h
            grad_category_bias = dlogits.sum(axis=0)
            dh = dlogits @ tensors["category_readout"]
            dmixed = dh * (1-h*h)
            grad_mixer_rows = projected[:, start:stop].T @ dmixed
            dprojected_block = dmixed @ tensors["field_mixer"][start:stop].T
            projected_block = projected.reshape(len(batch), config.field_blocks, config.field_block_output)[:, block]
            dpre_block = dprojected_block * (1-projected_block*projected_block)
            grad_field_w = np.einsum("ni,no->io", blocks[:, block], dpre_block, optimize=True)
            grad_field_b = dpre_block.sum(axis=0)
            tensors["field_weights"][block] -= learning_rate * _clip_gradient(grad_field_w / len(batch) + 1e-4*tensors["field_weights"][block])
            tensors["field_bias"][block] -= learning_rate * _clip_gradient(grad_field_b / len(batch))
            tensors["field_mixer"][start:stop] -= learning_rate * _clip_gradient(grad_mixer_rows / len(batch) + 1e-4*tensors["field_mixer"][start:stop])
            tensors["field_mixer_bias"] -= learning_rate * _clip_gradient(dmixed.mean(axis=0))
            tensors["category_readout"] -= learning_rate * _clip_gradient(grad_readout / len(batch) + 1e-4*tensors["category_readout"])
            tensors["category_bias"] -= learning_rate * _clip_gradient(grad_category_bias / len(batch))
    return losses


def _make_delta(bundle: dict[str, Any], patches: list[dict[str, Any]], metrics: dict[str, Any]) -> dict[str, Any]:
    changed = sum(int(decode_array(patch["value"]).size) for patch in patches)
    value = {
        "format": DELTA_FORMAT, "engine": ENGINE, "base_root": bundle["base_root"], "bundle_hash": bundle["sha256"],
        "operation": bundle["operation"], "category": int(bundle["category"]), "target_kind": bundle["target_kind"],
        "config": dict(bundle["config"]), "schema_hash": bundle["schema_hash"], "patches": patches,
        "write_set": sorted(patch_key(p["tensor"], p["selector"]) for p in patches),
        "changed_parameters": changed, "metrics": metrics,
    }
    value["sha256"] = delta_hash(value)
    validate_delta(value, bundle)
    return value


def execute_mutation(bundle: dict[str, Any], train_data: np.ndarray, train_labels: np.ndarray, diagnostic_data: np.ndarray | None = None, diagnostic_labels: np.ndarray | None = None, *, subset_seed: int = 7, optimizer_steps: int = 12, learning_rate: float = 0.03, search_recipe: dict[str, Any] | None = None) -> dict[str, Any]:
    config, tensors = _bundle_arrays(bundle)
    operation = canonical_operation(bundle["operation"])
    recipe = dict(search_recipe or {})
    target = int(bundle["category"])
    data = np.asarray(train_data, dtype=np.float32)
    labels = np.asarray(train_labels, dtype=np.int64)
    diagnostic = data if diagnostic_data is None else np.asarray(diagnostic_data, dtype=np.float32)
    diag_labels = labels if diagnostic_labels is None else np.asarray(diagnostic_labels, dtype=np.int64)
    expected_width = config.input_width if operation == "field_train" else config.representation_width
    if data.ndim != 2 or data.shape[1] != expected_width or len(data) != len(labels) or not len(data):
        raise ValueError("mutation training arrays are invalid")
    if diagnostic.ndim != 2 or diagnostic.shape[1] != expected_width or len(diagnostic) != len(diag_labels) or not len(diagnostic):
        raise ValueError("mutation diagnostic arrays are invalid")
    before = {name: value.copy() for name, value in tensors.items()}
    active = np.asarray([], dtype=np.int64)
    phase: int | None = None
    losses: list[float] = []

    if operation in {"expert_train", "branch_train", "repair", "memory_train"}:
        local = _local_labels(labels, target, config)
        diag_local = _local_labels(diag_labels, target, config)
        pre_logits = _category_logits_from_bundle(config, tensors, diagnostic) if operation != "memory_train" else np.zeros((len(diagnostic), config.classes_per_category), dtype=np.float32)
        if operation in {"expert_train", "branch_train", "repair"}:
            phase = int(np.asarray(tensors["rotation_phase"]).item())
            start_expert = phase * config.active_experts_per_update
            active = np.arange(start_expert, start_expert + config.active_experts_per_update, dtype=np.int64)
            if operation == "repair":
                active = np.argsort(tensors["expert_health"])[:config.active_experts_per_update]
                donor = int(np.argmax(tensors["expert_health"]))
                rng = np.random.default_rng(subset_seed)
                for expert in active:
                    for name in ("expert_branch_weights", "expert_branch_bias", "expert_branch_centers", "expert_branch_readout", "expert_local_bias", "expert_gate_weights", "expert_gate_bias", "expert_branch_health"):
                        tensors[name][expert] = tensors[name][donor]
                    tensors["expert_branch_weights"][expert] += rng.normal(0, 0.002, size=tensors["expert_branch_weights"][expert].shape).astype(np.float32)
                    tensors["expert_health"][expert] = 0.5
            losses = _train_experts(
                config, tensors, data, local, active, seed=subset_seed,
                steps=optimizer_steps, learning_rate=learning_rate,
                train_gates=operation != "branch_train",
                diversity_strength=float(recipe.get("expert_diversity", 0.01)),
            )
            tensors["rotation_phase"] = np.asarray((phase + 1) % config.rotation_groups, dtype=np.int64)
            post_logits = _category_logits_from_bundle(config, tensors, diagnostic)
        else:
            losses = _train_memory(config, tensors, data, local, steps=optimizer_steps, learning_rate=learning_rate)
            # Diagnostic uses memory-only scores because the bundle intentionally lacks expert tissue.
            normalized = diagnostic / np.maximum(np.linalg.norm(diagnostic, axis=1, keepdims=True), 1e-8)
            pre_memory = before["associative_memory"] / np.maximum(np.linalg.norm(before["associative_memory"], axis=1, keepdims=True), 1e-8)
            post_memory = tensors["associative_memory"] / np.maximum(np.linalg.norm(tensors["associative_memory"], axis=1, keepdims=True), 1e-8)
            pre_logits = normalized @ pre_memory.T * before["associative_strength"][None,:]
            post_logits = normalized @ post_memory.T * tensors["associative_strength"][None,:]
        pre_correct, pre_by = _metrics(pre_logits, diag_local)
        post_correct, post_by = _metrics(post_logits, diag_local)
    elif operation == "scout_train":
        if data.shape[1] != config.representation_width or np.any(labels < 0) or np.any(labels >= config.classes):
            raise ValueError("scout training labels or representations are invalid")
        category_labels = labels // config.classes_per_category
        pre_score = diagnostic @ before["category_readout"][target] + before["category_bias"][target]
        pre_prediction = pre_score >= 0
        losses = _train_scouts(
            config, tensors, data, labels, target, seed=subset_seed,
            steps=optimizer_steps, learning_rate=learning_rate,
            routing_margin=float(recipe.get("routing_margin", 0.5)),
            diversity_strength=float(recipe.get("scout_diversity", 0.005)),
        )
        post_score = diagnostic @ tensors["category_readout"][target] + tensors["category_bias"][target]
        post_prediction = post_score >= 0
        truth = (diag_labels // config.classes_per_category) == target
        pre_correct, post_correct = int((pre_prediction == truth).sum()), int((post_prediction == truth).sum())
        pre_by, post_by = [pre_correct], [post_correct]
    elif operation == "field_train":
        if np.any(labels < 0) or np.any(labels >= config.classes):
            raise ValueError("field training labels are invalid")
        def route_correct(state: dict[str, np.ndarray], values: np.ndarray, value_labels: np.ndarray) -> int:
            blocks = values.reshape(len(values), config.field_blocks, config.field_block_width)
            projected = np.tanh(np.einsum("nbi,bio->nbo", blocks, state["field_weights"], optimize=True) + state["field_bias"][None,:,:]).reshape(len(values), -1)
            h = np.tanh(projected @ state["field_mixer"] + state["field_mixer_bias"])
            prediction = (h @ state["category_readout"].T + state["category_bias"]).argmax(axis=1)
            return int((prediction == value_labels // config.classes_per_category).sum())
        pre_correct = route_correct(before, diagnostic, diag_labels)
        losses = _train_field(
            config, tensors, data, labels, target, seed=subset_seed,
            steps=optimizer_steps, learning_rate=learning_rate,
            route_margin=float(recipe.get("route_margin", 0.0)),
            route_margin_weight=float(recipe.get("route_margin_weight", 0.0)),
        )
        post_correct = route_correct(tensors, diagnostic, diag_labels)
        pre_by, post_by = [pre_correct], [post_correct]
    else:
        raise ValueError("unsupported mutation operation")

    patches: list[dict[str, Any]] = []
    if operation in {"expert_train", "repair"}:
        names = ("expert_branch_weights", "expert_branch_bias", "expert_branch_centers", "expert_branch_readout", "expert_local_bias", "expert_gate_weights", "expert_gate_bias", "expert_health", "expert_branch_health")
        for expert in active:
            for name in names:
                selector = selector_json(target, int(expert), *[(0, size) for size in tensor_shapes(config)[name][2:]])
                patches.append(make_patch(name, selector, tensors[name][int(expert)]))
        patches.append(make_patch("rotation_phase", selector_json(target), np.asarray(tensors["rotation_phase"])))
    elif operation == "branch_train":
        names = ("expert_branch_weights", "expert_branch_bias", "expert_branch_centers", "expert_branch_readout", "expert_health", "expert_branch_health")
        for expert in active:
            for name in names:
                selector = selector_json(target, int(expert), *[(0, size) for size in tensor_shapes(config)[name][2:]])
                patches.append(make_patch(name, selector, tensors[name][int(expert)]))
        patches.append(make_patch("rotation_phase", selector_json(target), np.asarray(tensors["rotation_phase"])))
    elif operation == "scout_train":
        start = target * config.scouts_per_category
        stop = start + config.scouts_per_category
        for name in ("scout_weights", "scout_bias", "scout_centers", "scout_quality"):
            shape = tensor_shapes(config)[name]
            selector = selector_json((start, stop), *[(0, size) for size in shape[1:]])
            patches.append(make_patch(name, selector, tensors[name]))
        patches.append(make_patch("category_readout", selector_json(target, (0, config.representation_width)), tensors["category_readout"][target]))
        patches.append(make_patch("category_bias", selector_json(target), np.asarray(tensors["category_bias"][target])))
    elif operation == "memory_train":
        start = target * config.classes_per_category
        stop = start + config.classes_per_category
        for name in ("associative_memory", "associative_counts", "associative_strength"):
            shape = tensor_shapes(config)[name]
            selector = selector_json((start, stop), *[(0, size) for size in shape[1:]])
            patches.append(make_patch(name, selector, tensors[name]))
    elif operation == "field_train":
        row_start = target * config.field_block_output
        row_stop = row_start + config.field_block_output
        patches.extend([
            make_patch("field_weights", selector_json(target, (0, config.field_block_width), (0, config.field_block_output)), tensors["field_weights"][target]),
            make_patch("field_bias", selector_json(target, (0, config.field_block_output)), tensors["field_bias"][target]),
            make_patch("field_mixer", selector_json((row_start,row_stop), (0,config.representation_width)), tensors["field_mixer"][row_start:row_stop]),
            make_patch("field_mixer_bias", selector_json((0,config.representation_width)), tensors["field_mixer_bias"]),
            make_patch("category_readout", selector_json((0,config.categories),(0,config.representation_width)), tensors["category_readout"]),
            make_patch("category_bias", selector_json((0,config.categories)), tensors["category_bias"]),
        ])
    metrics = {
        "sample_count": int(len(diag_labels)), "pre_correct": int(pre_correct), "post_correct": int(post_correct), "net_wins": int(post_correct-pre_correct),
        "pre_correct_by_class": pre_by, "post_correct_by_class": post_by, "active_experts": active.astype(int).tolist(),
        "rotation_phase_before": phase, "rotation_phase_after": int(np.asarray(tensors["rotation_phase"]).item()) if phase is not None else None,
        "metrics_scope": "trainer-visible-training-diagnostic-not-promotion-evidence", "search_seed": int(subset_seed),
        "optimizer": "local-sgd-v1", "optimizer_steps": int(optimizer_steps), "learning_rate": float(learning_rate),
        "search_recipe": recipe, "search_recipe_hash": content_hash(recipe),
        "initial_loss": float(losses[0]) if losses else None, "final_loss": float(losses[-1]) if losses else None,
    }
    delta = _make_delta(bundle, patches, metrics)
    return {"delta": delta, **metrics, "changed_parameters": delta["changed_parameters"], "write_set": delta["write_set"]}


def verify_mutation_full(checkpoint_artifact: dict[str, Any], bundle: dict[str, Any], delta: dict[str, Any], validation_inputs: np.ndarray, validation_labels: np.ndarray, *, validation_hash_value: str | None = None, validation_policy: Any | None = None) -> dict[str, Any]:
    model = Native10Dendritron.from_artifact(checkpoint_artifact)
    validate_delta(delta, bundle)
    if model.root != bundle["base_root"]:
        raise ValueError("verification checkpoint is not the candidate parent")
    x = np.asarray(validation_inputs, dtype=np.float32)
    labels = np.asarray(validation_labels, dtype=np.int64)
    if x.ndim != 2 or x.shape[1] != model.config.input_width or len(x) != len(labels) or not len(x):
        raise ValueError("verification raw inputs are invalid")
    if np.any(labels < 0) or np.any(labels >= model.config.classes):
        raise ValueError("verification labels are outside model range")
    pre_predictions = model.predict(x)
    updated = model.apply_delta(delta)
    post_predictions = updated.predict(x)
    pre_mask = pre_predictions == labels
    post_mask = post_predictions == labels
    pre_by, post_by, samples = [], [], []
    for class_id in range(model.config.classes):
        rows = labels == class_id
        samples.append(int(rows.sum()))
        pre_by.append(int(pre_mask[rows].sum()))
        post_by.append(int(post_mask[rows].sum()))
    losses_by = [max(0, a-b) for a,b in zip(pre_by,post_by,strict=True)]
    loss_rates = [loss/count if count else 0.0 for loss,count in zip(losses_by,samples,strict=True)]
    from dendriswarm.v6.validation import GlobalValidationPolicy, paired_evidence
    policy = validation_policy or GlobalValidationPolicy()
    paired = paired_evidence(pre_predictions, post_predictions, labels)
    corrected_alpha = float(policy.corrected_alpha)
    significant = bool(
        paired["discordant"] >= policy.min_discordant
        and paired["net_wins"] >= policy.minimum_net_wins
        and paired["effect_rate"] >= policy.minimum_effect_rate
        and paired["mcnemar_p_value"] <= corrected_alpha
    )
    if validation_hash_value is None:
        validation_hash_value = content_hash({"inputs": encode_array(x), "labels": encode_array(labels), "policy": policy.as_dict()})
    return {
        "delta_hash": delta["sha256"], "validation_hash": str(validation_hash_value), "base_root": model.root,
        "operation": delta["operation"], "category": int(delta["category"]), "sample_count": int(len(labels)),
        "pre_correct": int(pre_mask.sum()), "post_correct": int(post_mask.sum()), "net_wins": int(paired["net_wins"]),
        "wins": int(paired["wins"]), "losses": int(paired["losses"]), "discordant": int(paired["discordant"]),
        "effect_rate": float(paired["effect_rate"]), "mcnemar_p_value": float(paired["mcnemar_p_value"]),
        "corrected_alpha": corrected_alpha, "statistically_significant": significant,
        "pre_correct_by_class": pre_by, "post_correct_by_class": post_by, "samples_by_class": samples,
        "losses_by_class": losses_by, "loss_rates_by_class": loss_rates,
        "informative": bool(paired["discordant"]), "write_set": list(delta["write_set"]),
    }


def verify_mutation(bundle: dict[str, Any], delta: dict[str, Any], validation_representations: np.ndarray, validation_labels: np.ndarray) -> dict[str, Any]:
    """Local diagnostic verifier retained for tests; never promotion evidence."""
    config, base = _bundle_arrays(bundle)
    validate_delta(delta, bundle)
    if canonical_operation(bundle["operation"]) not in {"expert_train", "branch_train", "repair", "memory_train"}:
        return {"delta_hash": delta["sha256"], "informative": False, "metrics_scope": "local-diagnostic-only"}
    category = int(bundle["category"])
    local = _local_labels(np.asarray(validation_labels), category, config)
    before = _category_logits_from_bundle(config, base, np.asarray(validation_representations,dtype=np.float32)) if "expert_branch_weights" in base else np.zeros((len(local), config.classes_per_category),dtype=np.float32)
    # Full correctness is always established by verify_mutation_full.
    return {"delta_hash": delta["sha256"], "pre_correct": int((before.argmax(1)==local).sum()), "informative": True, "metrics_scope": "local-diagnostic-only"}


def compose_non_conflicting_deltas(model: Native10Dendritron, deltas: Iterable[dict[str, Any]], *, contribution: dict[str, Any] | None = None) -> Native10Dendritron:
    ordered = sorted(list(deltas), key=lambda item: item["sha256"])
    for delta in ordered:
        if delta.get("base_root") != model.root:
            raise ValueError("batch delta is not based on the shared parent root")
        validate_delta(
            delta, model.component_bundle(delta["operation"], int(delta["category"]))
        )
    for index, left in enumerate(ordered):
        for right in ordered[index+1:]:
            if deltas_conflict(left, right):
                raise ValueError("batch contains overlapping write sets")
    updated = model.copy()
    for delta in ordered:
        for patch in delta["patches"]:
            selector = normalize_selector(patch["selector"], updated.tensors[patch["tensor"]].shape)
            assign_patch(updated.tensors[patch["tensor"]], selector, decode_array(patch["value"]))
    updated._root_cache = None
    updated.lineage.append({"event":"verified-conflict-free-batch-promotion", "base_root":model.root, "delta_hashes":[d["sha256"] for d in ordered], **(contribution or {})})
    updated._validate()
    return updated


def synthetic_representation_shard(
    config: Native10Config,
    category: int,
    *,
    train_per_class: int = 24,
    validation_per_class: int = 20,
    seed: int = 20260723,
) -> dict[str, Any]:
    """Generate a deterministic protocol fixture, not a baseline benchmark."""
    rng = np.random.default_rng(seed + int(category) * 1009)
    start = int(category) * config.classes_per_category
    category_axis = np.zeros(config.representation_width, dtype=np.float32)
    category_axis[int(category) % config.representation_width] = 1.0
    class_axes = rng.normal(0, 1, size=(config.classes_per_category, config.representation_width)).astype(np.float32)
    class_axes[:, :config.categories] = 0.0
    class_axes /= np.maximum(np.linalg.norm(class_axes, axis=1, keepdims=True), 1e-8)

    def make(count: int) -> tuple[np.ndarray, np.ndarray]:
        rows: list[np.ndarray] = []
        labels: list[int] = []
        for local, axis in enumerate(class_axes):
            values = category_axis[None, :] * 3.0 + axis[None, :] * 2.0 + rng.normal(0, 0.25, size=(count, config.representation_width))
            rows.append(values.astype(np.float32))
            labels.extend([start + local] * count)
        order = rng.permutation(len(labels))
        return np.concatenate(rows, axis=0)[order], np.asarray(labels, dtype=np.int64)[order]

    train_x, train_y = make(train_per_class)
    val_x, val_y = make(validation_per_class)
    return {
        "format": "dendriswarm.native10-representation-shard.v1",
        "category": int(category),
        "representation_width": config.representation_width,
        "train_representations": train_x.tolist(),
        "train_labels": train_y.tolist(),
        "validation_representations": val_x.tolist(),
        "validation_labels": val_y.tolist(),
        "note": "Synthetic protocol fixture only; not a baseline performance benchmark.",
    }


def load_external_checkpoint(
    path: str | "os.PathLike[str]",
    *,
    config: Native10Config | None = None,
    key_map: dict[str, str] | None = None,
) -> Native10Dendritron:
    """Load a v6 artifact or adapt a v5/NPZ/PyTorch tensor checkpoint.

    The adapter never trains a baseline. A v5 checkpoint gains the new learned
    memory-strength tensor at the configured initial value and records that
    explicit conversion in lineage. NPZ/PyTorch archives may omit only that new
    tensor; all other persistent tensors must be present or mapped.
    """
    from pathlib import Path

    source = Path(path)
    suffix = source.suffix.lower()
    if suffix == ".json":
        value = json.loads(source.read_text())
        if value.get("format") == CHECKPOINT_FORMAT:
            return Native10Dendritron.from_artifact(value)
        if value.get("format") == "dendriswarm.native10-checkpoint.v5":
            from dendriswarm.v5.native10 import Native10Dendritron as Native10V5
            old = Native10V5.from_artifact(value)
            if config is None:
                old_config = old.config.as_dict()
                config = Native10Config(**{
                    **old_config,
                    "routing_expansion_margin": 0.20,
                    "max_routed_categories": min(old_config["categories"], max(old_config["top_categories"], 8)),
                    "memory_strength_init": 0.20,
                })
            tensors = {name: np.asarray(old.tensors[name]).copy() for name in old.TENSOR_NAMES}
            tensors["associative_strength"] = np.full(
                (config.classes,), config.memory_strength_init, dtype=np.float32
            )
            return Native10Dendritron(config, tensors, lineage=list(old.lineage) + [{
                "event": "v5-checkpoint-adapted-to-v6",
                "source_name": source.name,
                "added_tensor": "associative_strength",
                "baseline_training_included": False,
            }])
        raise ValueError("JSON checkpoint is not a supported Native10 v5 or v6 artifact")

    if config is None:
        raise ValueError("NPZ and PyTorch checkpoint imports require an explicit Native10Config")
    required = [name for name in Native10Dendritron.TENSOR_NAMES if name != "associative_strength"]
    mapping = key_map or {name: name for name in required}
    if set(mapping) not in {set(required), set(Native10Dendritron.TENSOR_NAMES)}:
        raise ValueError("checkpoint key map must cover every required Native10 tensor")

    if suffix == ".npz":
        with np.load(source, allow_pickle=False) as archive:
            tensors = {name: np.asarray(archive[mapping[name]]) for name in required}
            if "associative_strength" in mapping and mapping["associative_strength"] in archive:
                tensors["associative_strength"] = np.asarray(archive[mapping["associative_strength"]])
    elif suffix in {".pt", ".pth"}:
        try:
            import torch
        except ImportError as error:
            raise ValueError("PyTorch checkpoint import requires the optional torch package") from error
        loaded = torch.load(source, map_location="cpu", weights_only=True)
        if isinstance(loaded, dict) and "state_dict" in loaded and isinstance(loaded["state_dict"], dict):
            loaded = loaded["state_dict"]
        if not isinstance(loaded, dict):
            raise ValueError("PyTorch checkpoint must contain a state dictionary")
        tensors = {}
        for name in required:
            value = loaded[mapping[name]]
            if hasattr(value, "detach"):
                value = value.detach().cpu().numpy()
            tensors[name] = np.asarray(value)
        if "associative_strength" in mapping and mapping["associative_strength"] in loaded:
            value = loaded[mapping["associative_strength"]]
            if hasattr(value, "detach"):
                value = value.detach().cpu().numpy()
            tensors["associative_strength"] = np.asarray(value)
    else:
        raise ValueError("checkpoint must be .json, .npz, .pt, or .pth")

    tensors.setdefault(
        "associative_strength",
        np.full((config.classes,), config.memory_strength_init, dtype=np.float32),
    )
    return Native10Dendritron(config, tensors, lineage=[{
        "event": "external-checkpoint-imported",
        "source_name": source.name,
        "added_default_memory_strength": "associative_strength" not in mapping,
        "baseline_training_included": False,
    }])

