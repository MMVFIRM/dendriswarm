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

ENGINE = "dendriswarm.native10-derived.v5"
CHECKPOINT_FORMAT = "dendriswarm.native10-checkpoint.v5"
DELTA_FORMAT = "dendriswarm.native10-delta.v5"
BUNDLE_FORMAT = "dendriswarm.native10-work-bundle.v5"
QUANTIZED_FORMAT = "dendriswarm.native10-int8.v1"


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
    compressed = base64.b64decode(str(value["data"]), validate=True)
    raw = zlib.decompress(compressed)
    if len(raw) > max_raw_bytes:
        raise ValueError("decoded tensor exceeds byte limit")
    expected = content_hash({"bytes_b64": base64.b64encode(raw).decode("ascii")})
    if expected != value.get("raw_sha256"):
        raise ValueError("tensor payload hash mismatch")
    array = np.load(io.BytesIO(raw), allow_pickle=False)
    if list(array.shape) != list(value.get("shape", [])) or str(array.dtype) != str(value.get("dtype")):
        raise ValueError("tensor metadata mismatch")
    if array.dtype.kind == "f" and not np.isfinite(array).all():
        raise ValueError("tensor contains non-finite values")
    return np.asarray(array)


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
        "associative_memory", "associative_counts", "rotation_phase",
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

    def top_categories(self, representations: np.ndarray) -> np.ndarray:
        scores = self.route_scores(representations)
        single = scores.ndim == 1
        if single:
            scores = scores[None, :]
        k = self.config.top_categories
        selected = np.argpartition(scores, kth=self.config.categories - k, axis=1)[:, -k:]
        order = np.take_along_axis(scores, selected, axis=1).argsort(axis=1)[:, ::-1]
        selected = np.take_along_axis(selected, order, axis=1)
        return selected[0] if single else selected

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
        return local + 0.20 * memory_score

    def scores_from_representation(self, representations: np.ndarray) -> np.ndarray:
        h = np.asarray(representations, dtype=np.float32)
        single = h.ndim == 1
        if single:
            h = h[None, :]
        route = self.route_scores(h)
        selected = self.top_categories(h)
        logits = np.full((len(h), self.config.classes), -30.0, dtype=np.float32)
        for category in range(self.config.categories):
            rows = np.flatnonzero(np.any(selected == category, axis=1))
            if not len(rows):
                continue
            local = self._category_logits(h[rows], category)
            start = category * self.config.classes_per_category
            logits[rows, start:start + self.config.classes_per_category] = local + route[rows, category, None]
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
        category = int(category)
        if not 0 <= category < self.config.categories:
            raise ValueError("category is outside model range")
        category_names = (
            "expert_branch_weights", "expert_branch_bias", "expert_branch_centers",
            "expert_branch_readout", "expert_local_bias", "expert_gate_weights",
            "expert_gate_bias", "expert_health", "expert_branch_health", "rotation_phase",
        )
        tensors: dict[str, Any] = {}
        if operation in {"expert_refit", "repair", "branch_lifecycle"}:
            for name in category_names:
                tensors[name] = encode_array(self.tensors[name][category])
            start = category * self.config.classes_per_category
            tensors["associative_memory"] = encode_array(self.tensors["associative_memory"][start:start + self.config.classes_per_category])
            tensors["associative_counts"] = encode_array(self.tensors["associative_counts"][start:start + self.config.classes_per_category])
        elif operation == "scout_refit":
            start = category * self.config.scouts_per_category
            stop = start + self.config.scouts_per_category
            for name in ("scout_weights", "scout_bias", "scout_centers", "scout_quality"):
                tensors[name] = encode_array(self.tensors[name][start:stop])
            tensors["category_readout"] = encode_array(self.tensors["category_readout"][category])
            tensors["category_bias"] = encode_array(np.asarray(self.tensors["category_bias"][category]))
        elif operation == "memory_update":
            start = category * self.config.classes_per_category
            tensors["associative_memory"] = encode_array(self.tensors["associative_memory"][start:start + self.config.classes_per_category])
            tensors["associative_counts"] = encode_array(self.tensors["associative_counts"][start:start + self.config.classes_per_category])
        else:
            raise ValueError(f"unsupported Native10 mutation operation: {operation}")
        bundle = {
            "format": BUNDLE_FORMAT,
            "engine": ENGINE,
            "base_root": self.root,
            "operation": operation,
            "category": category,
            "config": self.config.as_dict(),
            "tensors": tensors,
        }
        bundle["sha256"] = bundle_hash(bundle)
        return bundle

    def apply_delta(self, delta: dict[str, Any], *, contribution: dict[str, Any] | None = None) -> "Native10Dendritron":
        validate_delta(delta)
        if delta["base_root"] != self.root:
            raise ValueError("delta parent is not the current canonical root")
        if Native10Config.from_dict(delta["config"]) != self.config:
            raise ValueError("delta model configuration mismatch")
        category = int(delta["category"])
        updated = self.copy()
        for name, encoded in delta["tensors"].items():
            value = decode_array(encoded)
            if name in {
                "expert_branch_weights", "expert_branch_bias", "expert_branch_centers",
                "expert_branch_readout", "expert_local_bias", "expert_gate_weights",
                "expert_gate_bias", "expert_health", "expert_branch_health", "rotation_phase",
            }:
                if name == "rotation_phase":
                    updated.tensors[name][category] = int(np.asarray(value).item())
                else:
                    updated.tensors[name][category] = value
            elif name in {"scout_weights", "scout_bias", "scout_centers", "scout_quality"}:
                start = category * self.config.scouts_per_category
                stop = start + self.config.scouts_per_category
                updated.tensors[name][start:stop] = value
            elif name in {"category_readout", "category_bias"}:
                updated.tensors[name][category] = value
            elif name in {"associative_memory", "associative_counts"}:
                start = category * self.config.classes_per_category
                stop = start + self.config.classes_per_category
                updated.tensors[name][start:stop] = value
            else:
                raise ValueError(f"delta attempts to update unsupported tensor: {name}")
        event = {
            "event": "verified-tissue-promotion",
            "delta_hash": delta["sha256"],
            "base_root": delta["base_root"],
            "operation": delta["operation"],
            "category": category,
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


def delta_consensus_hash(delta: dict[str, Any], decimals: int = 5) -> str:
    """Architecture-tolerant fingerprint; exact content remains independently verified."""
    validate_delta(delta)
    digest = hashlib.sha256()
    digest.update(b"dendriswarm.native10-delta-consensus.v1\0")
    digest.update(str(int(decimals)).encode("ascii") + b"\0")
    for key in ("base_root", "bundle_hash", "operation", "category"):
        digest.update(json.dumps(delta[key], sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\0")
    for name in sorted(delta["tensors"]):
        value = decode_array(delta["tensors"][name])
        if value.dtype.kind == "f":
            value = np.round(value.astype(np.float64), decimals=decimals).astype(np.float32)
        value = np.ascontiguousarray(value)
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(str(value.dtype).encode("ascii") + b"\0")
        digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode("ascii") + b"\0")
        digest.update(memoryview(value).cast("B"))
    return digest.hexdigest()


def validate_bundle(bundle: dict[str, Any]) -> None:
    if bundle.get("format") != BUNDLE_FORMAT or bundle.get("engine") != ENGINE:
        raise ValueError("unsupported Native10 work bundle")
    if bundle.get("sha256") != bundle_hash(bundle):
        raise ValueError("Native10 work bundle hash mismatch")
    Native10Config.from_dict(dict(bundle["config"]))
    if not 0 <= int(bundle["category"]) < int(bundle["config"]["categories"]):
        raise ValueError("bundle category is invalid")
    for encoded in bundle["tensors"].values():
        decode_array(encoded)


def validate_delta(delta: dict[str, Any]) -> None:
    if delta.get("format") != DELTA_FORMAT or delta.get("engine") != ENGINE:
        raise ValueError("unsupported Native10 delta")
    if delta.get("sha256") != delta_hash(delta):
        raise ValueError("Native10 delta hash mismatch")
    Native10Config.from_dict(dict(delta["config"]))
    if len(str(delta.get("base_root", ""))) != 64:
        raise ValueError("delta base root is invalid")
    for encoded in delta["tensors"].values():
        decode_array(encoded)


def _bundle_arrays(bundle: dict[str, Any]) -> tuple[Native10Config, dict[str, np.ndarray]]:
    validate_bundle(bundle)
    return Native10Config.from_dict(bundle["config"]), {
        name: decode_array(value) for name, value in bundle["tensors"].items()
    }


def _category_logits_from_bundle(
    config: Native10Config,
    tensors: dict[str, np.ndarray],
    representations: np.ndarray,
) -> np.ndarray:
    h = np.asarray(representations, dtype=np.float32)
    w = tensors["expert_branch_weights"]
    z = np.tanh(np.einsum("nr,ebrd->nebd", h, w, optimize=True) + tensors["expert_branch_bias"][None, :, :, :])
    distance = np.mean((z - tensors["expert_branch_centers"][None, :, :, :]) ** 2, axis=-1)
    branch_evidence = np.exp(-distance) * tensors["expert_branch_health"][None, :, :]
    branch_logits = np.einsum("nebd,ebdl->nebl", z, tensors["expert_branch_readout"], optimize=True)
    branch_weight = branch_evidence / np.maximum(branch_evidence.sum(axis=2, keepdims=True), 1e-8)
    expert_logits = (branch_logits * branch_weight[..., None]).sum(axis=2) + tensors["expert_local_bias"][None, :, :]
    gate = _sigmoid(h @ tensors["expert_gate_weights"].T + tensors["expert_gate_bias"])
    gate *= tensors["expert_health"][None, :]
    gate /= np.maximum(gate.sum(axis=1, keepdims=True), 1e-8)
    local = (expert_logits * gate[..., None]).sum(axis=1)
    memory = tensors.get("associative_memory")
    counts = tensors.get("associative_counts")
    if memory is not None and counts is not None:
        normalized_h = h / np.maximum(np.linalg.norm(h, axis=1, keepdims=True), 1e-8)
        normalized_memory = memory / np.maximum(np.linalg.norm(memory, axis=1, keepdims=True), 1e-8)
        memory_score = normalized_h @ normalized_memory.T
        memory_score[:, counts == 0] = 0.0
        local += 0.20 * memory_score
    return local


def _local_labels(labels: np.ndarray, category: int, config: Native10Config) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.int64)
    start = int(category) * config.classes_per_category
    local = labels - start
    if np.any(local < 0) or np.any(local >= config.classes_per_category):
        raise ValueError("mutation shard contains labels outside its category")
    return local


def _metrics(logits: np.ndarray, local_labels: np.ndarray) -> tuple[int, list[int]]:
    predictions = logits.argmax(axis=1)
    correct = int((predictions == local_labels).sum())
    per_class = [int(((predictions == label) & (local_labels == label)).sum()) for label in range(logits.shape[1])]
    return correct, per_class


def _make_delta(bundle: dict[str, Any], tensors: dict[str, np.ndarray], metrics: dict[str, Any]) -> dict[str, Any]:
    value = {
        "format": DELTA_FORMAT,
        "engine": ENGINE,
        "base_root": bundle["base_root"],
        "bundle_hash": bundle["sha256"],
        "operation": bundle["operation"],
        "category": int(bundle["category"]),
        "config": dict(bundle["config"]),
        "tensors": {name: encode_array(_round_tensor(array)) if np.asarray(array).dtype.kind == "f" else encode_array(array) for name, array in sorted(tensors.items())},
        "metrics": metrics,
    }
    value["sha256"] = delta_hash(value)
    return value


def execute_mutation(
    bundle: dict[str, Any],
    train_representations: np.ndarray,
    train_labels: np.ndarray,
    validation_representations: np.ndarray,
    validation_labels: np.ndarray,
    *,
    subset_seed: int = 7,
) -> dict[str, Any]:
    config, tensors = _bundle_arrays(bundle)
    operation = str(bundle["operation"])
    category = int(bundle["category"])
    train_h = np.asarray(train_representations, dtype=np.float32)
    val_h = np.asarray(validation_representations, dtype=np.float32)
    if train_h.ndim != 2 or val_h.ndim != 2 or train_h.shape[1] != config.representation_width or val_h.shape[1] != config.representation_width:
        raise ValueError("mutation representations have the wrong width")
    train_local = _local_labels(np.asarray(train_labels), category, config)
    val_local = _local_labels(np.asarray(validation_labels), category, config)
    if len(train_h) != len(train_local) or len(val_h) != len(val_local) or not len(train_h) or not len(val_h):
        raise ValueError("mutation arrays are empty or misaligned")

    pre_correct = 0
    pre_per_class: list[int] = []
    if operation in {"expert_refit", "repair", "branch_lifecycle"}:
        pre_logits = _category_logits_from_bundle(config, tensors, val_h)
        pre_correct, pre_per_class = _metrics(pre_logits, val_local)
        phase = int(np.asarray(tensors["rotation_phase"]).item())
        start_expert = phase * config.active_experts_per_update
        active = np.arange(start_expert, start_expert + config.active_experts_per_update, dtype=np.int64)
        if operation == "repair":
            unhealthy = np.argsort(tensors["expert_health"])[:config.active_experts_per_update]
            healthy = int(np.argmax(tensors["expert_health"]))
            for offset, expert in enumerate(unhealthy):
                for name in (
                    "expert_branch_weights", "expert_branch_bias", "expert_branch_centers",
                    "expert_branch_readout", "expert_local_bias", "expert_gate_weights",
                    "expert_gate_bias", "expert_branch_health",
                ):
                    tensors[name][expert] = tensors[name][healthy]
                jitter_rng = np.random.default_rng(subset_seed + int(expert) * 7919 + offset)
                tensors["expert_branch_centers"][expert] += jitter_rng.normal(
                    0, 0.01, size=tensors["expert_branch_centers"][expert].shape
                ).astype(np.float32)
                tensors["expert_health"][expert] = 0.5
            active = unhealthy

        rng = np.random.default_rng(int(subset_seed))
        for expert in active:
            expert = int(expert)
            indices = rng.integers(0, len(train_h), size=max(len(train_h), config.classes_per_category * 4))
            x = train_h[indices]
            y = train_local[indices]
            z = np.tanh(
                np.einsum("nr,brd->nbd", x, tensors["expert_branch_weights"][expert], optimize=True)
                + tensors["expert_branch_bias"][expert][None, :, :]
            )
            centers = tensors["expert_branch_centers"][expert]
            distance = np.mean((z - centers[None, :, :]) ** 2, axis=-1)
            assignment = distance.argmin(axis=1)
            predictions = _category_logits_from_bundle(config, tensors, x).argmax(axis=1)
            errors = np.flatnonzero(predictions != y)
            for branch in range(config.expert_branches):
                selected = np.flatnonzero(assignment == branch)
                if len(selected) < max(2, config.classes_per_category):
                    source = int(errors[branch % len(errors)]) if len(errors) else int((expert + branch) % len(x))
                    tensors["expert_branch_centers"][expert, branch] = z[source, branch]
                    tensors["expert_branch_health"][expert, branch] = 0.55
                    selected = np.asarray([source], dtype=np.int64)
                else:
                    tensors["expert_branch_centers"][expert, branch] = z[selected, branch].mean(axis=0)
                    tensors["expert_branch_health"][expert, branch] = min(1.0, len(selected) / max(1.0, len(x) / config.expert_branches))
                branch_z = z[selected, branch]
                branch_y = y[selected]
                overall = branch_z.mean(axis=0)
                for label in range(config.classes_per_category):
                    class_rows = branch_z[branch_y == label]
                    direction = (class_rows.mean(axis=0) - overall) if len(class_rows) else np.zeros(config.branch_width, dtype=np.float32)
                    tensors["expert_branch_readout"][expert, branch, :, label] = direction
            counts = np.bincount(y, minlength=config.classes_per_category).astype(np.float32)
            tensors["expert_local_bias"][expert] = np.log((counts + 1.0) / (counts.sum() + config.classes_per_category))
            category_mean = x.mean(axis=0)
            tensors["expert_gate_weights"][expert] = category_mean / max(float(np.linalg.norm(category_mean)), 1e-8)
            tensors["expert_gate_bias"][expert] = 0.0
            local_logits = _category_logits_from_bundle(config, tensors, x)
            tensors["expert_health"][expert] = 0.25 + 0.75 * float((local_logits.argmax(axis=1) == y).mean())

        if operation == "branch_lifecycle":
            usage = tensors["expert_branch_health"]
            prune = usage < 0.08
            usage[prune] = 0.0
            tensors["expert_branch_health"] = usage
        tensors["rotation_phase"] = np.asarray((phase + 1) % config.rotation_groups, dtype=np.int64)

    elif operation == "scout_refit":
        pre_correct = 0
        pre_per_class = [0] * config.classes_per_category
        projected = np.einsum("nr,srd->nsd", train_h, tensors["scout_weights"], optimize=True)
        projected = np.tanh(projected + tensors["scout_bias"][None, :, :])
        order = np.arange(len(train_h))
        rng = np.random.default_rng(subset_seed)
        rng.shuffle(order)
        groups = np.array_split(order, config.scouts_per_category)
        for scout, rows in enumerate(groups):
            if len(rows):
                tensors["scout_centers"][scout] = projected[rows, scout].mean(axis=0)
                variance = float(np.mean((projected[rows, scout] - tensors["scout_centers"][scout]) ** 2))
                tensors["scout_quality"][scout] = 1.0 / (1.0 + variance)
        mean = train_h.mean(axis=0)
        tensors["category_readout"] = mean / max(float(np.linalg.norm(mean)), 1e-8)
        tensors["category_bias"] = np.asarray(0.0, dtype=np.float32)

    elif operation == "memory_update":
        pre_correct = 0
        pre_per_class = [0] * config.classes_per_category
        for label in range(config.classes_per_category):
            rows = train_h[train_local == label]
            if not len(rows):
                continue
            old_count = int(tensors["associative_counts"][label])
            new_count = old_count + len(rows)
            old = tensors["associative_memory"][label]
            tensors["associative_memory"][label] = (old * old_count + rows.sum(axis=0)) / max(1, new_count)
            tensors["associative_counts"][label] = new_count
    else:
        raise ValueError(f"unsupported Native10 mutation operation: {operation}")

    post_correct = pre_correct
    post_per_class = pre_per_class
    if operation in {"expert_refit", "repair", "branch_lifecycle"}:
        post_logits = _category_logits_from_bundle(config, tensors, val_h)
        post_correct, post_per_class = _metrics(post_logits, val_local)
    metrics = {
        "sample_count": int(len(val_h)),
        "pre_correct": int(pre_correct),
        "post_correct": int(post_correct),
        "net_wins": int(post_correct - pre_correct),
        "pre_correct_by_class": pre_per_class,
        "post_correct_by_class": post_per_class,
        "active_experts": (
            active.astype(int).tolist() if operation in {"expert_refit", "repair", "branch_lifecycle"} else []
        ),
        "rotation_phase_before": (
            phase if operation in {"expert_refit", "repair", "branch_lifecycle"} else None
        ),
        "rotation_phase_after": (
            int(np.asarray(tensors["rotation_phase"]).item()) if operation in {"expert_refit", "repair", "branch_lifecycle"} else None
        ),
        "metrics_scope": "trainer-visible-training-diagnostic-not-promotion-evidence",
    }
    delta_tensors = dict(tensors)
    delta = _make_delta(bundle, delta_tensors, metrics)
    return {"delta": delta, **metrics}


def verify_mutation(
    bundle: dict[str, Any],
    delta: dict[str, Any],
    validation_representations: np.ndarray,
    validation_labels: np.ndarray,
) -> dict[str, Any]:
    config, base_tensors = _bundle_arrays(bundle)
    validate_delta(delta)
    if delta["bundle_hash"] != bundle["sha256"] or delta["base_root"] != bundle["base_root"]:
        raise ValueError("delta is not bound to the supplied work bundle")
    if int(delta["category"]) != int(bundle["category"]) or delta["operation"] != bundle["operation"]:
        raise ValueError("delta operation or territory mismatch")
    category = int(bundle["category"])
    val_h = np.asarray(validation_representations, dtype=np.float32)
    val_local = _local_labels(np.asarray(validation_labels), category, config)
    updated_tensors = {name: value.copy() for name, value in base_tensors.items()}
    for name, encoded in delta["tensors"].items():
        updated_tensors[name] = decode_array(encoded)
    if bundle["operation"] in {"expert_refit", "repair", "branch_lifecycle"}:
        pre_logits = _category_logits_from_bundle(config, base_tensors, val_h)
        post_logits = _category_logits_from_bundle(config, updated_tensors, val_h)
        pre_correct, pre_by_class = _metrics(pre_logits, val_local)
        post_correct, post_by_class = _metrics(post_logits, val_local)
    else:
        # Scout and memory changes are verified structurally here; their full
        # route/global effects are evaluated by separate model-level canaries.
        pre_correct = post_correct = 0
        pre_by_class = post_by_class = [0] * config.classes_per_category
    losses_by_class = [max(0, pre - post) for pre, post in zip(pre_by_class, post_by_class, strict=True)]
    samples_by_class = [int((val_local == label).sum()) for label in range(config.classes_per_category)]
    loss_rates_by_class = [
        (loss / count if count else 0.0)
        for loss, count in zip(losses_by_class, samples_by_class, strict=True)
    ]
    return {
        "delta_hash": delta["sha256"],
        "base_root": bundle["base_root"],
        "operation": bundle["operation"],
        "category": category,
        "sample_count": int(len(val_h)),
        "pre_correct": int(pre_correct),
        "post_correct": int(post_correct),
        "net_wins": int(post_correct - pre_correct),
        "pre_correct_by_class": pre_by_class,
        "post_correct_by_class": post_by_class,
        "samples_by_class": samples_by_class,
        "losses_by_class": losses_by_class,
        "loss_rates_by_class": loss_rates_by_class,
        "informative": bool(bundle["operation"] not in {"scout_refit", "memory_update"} or len(val_h)),
    }


def verify_mutation_full(
    checkpoint_artifact: dict[str, Any],
    bundle: dict[str, Any],
    delta: dict[str, Any],
    validation_representations: np.ndarray,
    validation_labels: np.ndarray,
    *,
    validation_hash_value: str | None = None,
) -> dict[str, Any]:
    """Verify a tissue delta against the complete canonical Dendritron.

    The component bundle proves what may change.  The checkpoint proves the
    parent model whose end-to-end predictions are being measured.  Promotion
    metrics therefore include routing, competing categories, associative
    memory, and every untouched tissue rather than an oracle-local colony.
    """
    model = Native10Dendritron.from_artifact(checkpoint_artifact)
    validate_delta(delta)
    if model.root != str(bundle.get("base_root")) or model.root != str(delta.get("base_root")):
        raise ValueError("verification checkpoint is not the candidate parent")
    if delta.get("bundle_hash") != bundle.get("sha256"):
        raise ValueError("delta is not bound to the supplied work bundle")
    if int(delta.get("category", -1)) != int(bundle.get("category", -2)):
        raise ValueError("delta territory mismatch")
    if str(delta.get("operation")) != str(bundle.get("operation")):
        raise ValueError("delta operation mismatch")

    representations = np.asarray(validation_representations, dtype=np.float32)
    labels = np.asarray(validation_labels, dtype=np.int64)
    if representations.ndim != 2 or representations.shape[1] != model.config.representation_width:
        raise ValueError("verification representations do not match the canonical model")
    if labels.ndim != 1 or len(labels) != len(representations) or not len(labels):
        raise ValueError("verification labels are empty or misaligned")
    if np.any(labels < 0) or np.any(labels >= model.config.classes):
        raise ValueError("verification labels are outside the model class range")

    pre_predictions = model.predict_from_representation(representations)
    updated = model.apply_delta(delta)
    post_predictions = updated.predict_from_representation(representations)
    pre_correct_mask = np.asarray(pre_predictions == labels)
    post_correct_mask = np.asarray(post_predictions == labels)
    pre_by_class: list[int] = []
    post_by_class: list[int] = []
    samples_by_class: list[int] = []
    for class_id in range(model.config.classes):
        rows = labels == class_id
        samples_by_class.append(int(rows.sum()))
        pre_by_class.append(int(pre_correct_mask[rows].sum()))
        post_by_class.append(int(post_correct_mask[rows].sum()))
    losses_by_class = [
        max(0, before - after)
        for before, after in zip(pre_by_class, post_by_class, strict=True)
    ]
    loss_rates_by_class = [
        (loss / count if count else 0.0)
        for loss, count in zip(losses_by_class, samples_by_class, strict=True)
    ]
    pre_correct = int(pre_correct_mask.sum())
    post_correct = int(post_correct_mask.sum())
    if validation_hash_value is None:
        validation_hash_value = content_hash({
            "representations": encode_array(representations),
            "labels": encode_array(labels),
        })
    return {
        "delta_hash": delta["sha256"],
        "validation_hash": str(validation_hash_value),
        "base_root": model.root,
        "operation": str(bundle["operation"]),
        "category": int(bundle["category"]),
        "sample_count": int(len(labels)),
        "pre_correct": pre_correct,
        "post_correct": post_correct,
        "net_wins": int(post_correct - pre_correct),
        "pre_correct_by_class": pre_by_class,
        "post_correct_by_class": post_by_class,
        "samples_by_class": samples_by_class,
        "losses_by_class": losses_by_class,
        "loss_rates_by_class": loss_rates_by_class,
        "informative": bool(pre_correct or post_correct or np.any(pre_predictions != post_predictions)),
    }


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
    """Load a v5 JSON artifact, NPZ tensor archive, or optional PyTorch state dict.

    NPZ and PyTorch imports expect the v5 tensor names by default. ``key_map``
    maps each v5 tensor name to a source key, allowing an existing Native10
    exporter to adapt without adding baseline training code to this package.
    """
    from pathlib import Path

    source = Path(path)
    suffix = source.suffix.lower()
    if suffix == ".json":
        return Native10Dendritron.from_artifact(json.loads(source.read_text()))
    if config is None:
        raise ValueError("NPZ and PyTorch checkpoint imports require an explicit Native10Config")
    mapping = key_map or {name: name for name in Native10Dendritron.TENSOR_NAMES}
    if set(mapping) != set(Native10Dendritron.TENSOR_NAMES):
        raise ValueError("checkpoint key map must cover every v5 tensor")
    if suffix == ".npz":
        with np.load(source, allow_pickle=False) as archive:
            tensors = {name: np.asarray(archive[mapping[name]]) for name in Native10Dendritron.TENSOR_NAMES}
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
        for name in Native10Dendritron.TENSOR_NAMES:
            value = loaded[mapping[name]]
            if hasattr(value, "detach"):
                value = value.detach().cpu().numpy()
            tensors[name] = np.asarray(value)
    else:
        raise ValueError("checkpoint must be .json, .npz, .pt, or .pth")
    return Native10Dendritron(config, tensors, lineage=[{
        "event": "external-checkpoint-imported",
        "source_name": source.name,
        "baseline_training_included": False,
    }])
