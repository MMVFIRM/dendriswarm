from __future__ import annotations

import hashlib
import json
import os
import pickle
import tarfile
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from dendriswarm.core.crypto import content_hash
from dendriswarm.v6.native10 import Native10Dendritron, encode_array

CIFAR100_URL = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
CIFAR100_MD5 = "eb9058c3a382ffc7106e4002c42a8d85"
DATASET_FORMAT = "dendriswarm.cifar100-campaign-dataset.v1"
SHARD_FORMAT = "dendriswarm.cifar100-training-shard.v1"


def download_official_archive(destination: str | os.PathLike[str], *, url: str = CIFAR100_URL) -> dict[str, Any]:
    """Stream the official archive to disk and verify the published MD5."""
    from urllib.parse import urlparse
    import httpx

    parsed = urlparse(url)
    if parsed.scheme != "https":
        raise ValueError("CIFAR-100 download requires HTTPS")
    target = Path(destination).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    os.close(fd)
    temporary = Path(temporary_name)
    limit = 256 * 1024 * 1024
    size = 0
    try:
        with httpx.stream("GET", url, follow_redirects=True, timeout=120.0) as response:
            response.raise_for_status()
            length = response.headers.get("content-length")
            if length is not None and int(length) > limit:
                raise ValueError("CIFAR-100 archive exceeds the download limit")
            with temporary.open("wb") as handle:
                for chunk in response.iter_bytes(1024 * 1024):
                    size += len(chunk)
                    if size > limit:
                        raise ValueError("CIFAR-100 archive exceeds the download limit")
                    handle.write(chunk)
                handle.flush()
                os.fsync(handle.fileno())
        observed_md5 = _file_digest(temporary, "md5")
        if observed_md5 != CIFAR100_MD5:
            raise ValueError(f"CIFAR-100 archive MD5 mismatch: {observed_md5}")
        os.replace(temporary, target)
        return {
            "path": str(target),
            "bytes": size,
            "md5": observed_md5,
            "sha256": _file_digest(target, "sha256"),
            "source": url,
        }
    finally:
        temporary.unlink(missing_ok=True)


def _file_digest(path: Path, algorithm: str = "sha256") -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_digest(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii") + b"\0")
    digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode("ascii") + b"\0")
    digest.update(memoryview(value).cast("B"))
    return digest.hexdigest()


def _safe_member(member: tarfile.TarInfo, root: Path) -> bool:
    if member.issym() or member.islnk() or member.isdev():
        return False
    destination = (root / member.name).resolve()
    try:
        destination.relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _extract_archive(archive: Path, target: Path) -> Path:
    with tarfile.open(archive, "r:gz") as bundle:
        members = bundle.getmembers()
        if not members or any(not _safe_member(member, target) for member in members):
            raise ValueError("CIFAR-100 archive contains an unsafe member")
        total = sum(max(0, int(member.size)) for member in members)
        if total > 512 * 1024 * 1024:
            raise ValueError("CIFAR-100 archive expands beyond the safety limit")
        bundle.extractall(target, members=members, filter="data")
    candidates = [target / "cifar-100-python", target]
    for candidate in candidates:
        if (candidate / "train").exists() and (candidate / "test").exists() and (candidate / "meta").exists():
            return candidate
    raise ValueError("CIFAR-100 Python files were not found in the archive")


class _RestrictedCIFARUnpickler(pickle.Unpickler):
    """Load the official NumPy/list CIFAR payload without arbitrary imports."""

    _ALLOWED = {
        ("numpy.core.multiarray", "_reconstruct"),
        ("numpy._core.multiarray", "_reconstruct"),
        ("numpy", "ndarray"),
        ("numpy", "dtype"),
        ("numpy.core.multiarray", "scalar"),
        ("numpy._core.multiarray", "scalar"),
        ("builtins", "set"),
        ("builtins", "slice"),
    }

    def find_class(self, module: str, name: str) -> Any:
        if (module, name) not in self._ALLOWED:
            raise pickle.UnpicklingError(f"forbidden CIFAR pickle global: {module}.{name}")
        return super().find_class(module, name)


def _unpickle(path: Path) -> dict[bytes, Any]:
    with path.open("rb") as handle:
        value = _RestrictedCIFARUnpickler(handle, encoding="bytes").load()
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} is not a CIFAR dictionary")
    return value


def _as_label_array(value: Any, name: str, rows: int) -> np.ndarray:
    labels = np.asarray(value, dtype=np.int64)
    if labels.shape != (rows,) or np.any(labels < 0):
        raise ValueError(f"CIFAR-100 {name} are invalid")
    return labels


def _native_label_mapping(
    train_fine: np.ndarray,
    train_coarse: np.ndarray,
    test_fine: np.ndarray,
    test_coarse: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[list[int]]]:
    relation: dict[int, set[int]] = {}
    for fine, coarse in zip(
        np.concatenate([train_fine, test_fine]),
        np.concatenate([train_coarse, test_coarse]),
        strict=True,
    ):
        relation.setdefault(int(coarse), set()).add(int(fine))
    if set(relation) != set(range(20)):
        raise ValueError("CIFAR-100 must contain exactly 20 coarse categories")
    grouped = [sorted(relation[coarse]) for coarse in range(20)]
    if any(len(group) != 5 for group in grouped) or len({fine for group in grouped for fine in group}) != 100:
        raise ValueError("every CIFAR-100 coarse category must own exactly five fine classes")
    native_to_official = np.asarray([fine for group in grouped for fine in group], dtype=np.int64)
    official_to_native = np.full(100, -1, dtype=np.int64)
    official_to_native[native_to_official] = np.arange(100, dtype=np.int64)
    if np.any(official_to_native < 0):
        raise ValueError("CIFAR-100 fine-label mapping is incomplete")
    return official_to_native, native_to_official, grouped


def _stratified_split(labels: np.ndarray, *, seed: int, train_per_class: int = 450) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    train_rows: list[np.ndarray] = []
    selection_rows: list[np.ndarray] = []
    replication_rows: list[np.ndarray] = []
    for class_id in range(100):
        rows = np.flatnonzero(labels == class_id)
        if len(rows) != 500:
            raise ValueError(f"CIFAR-100 class {class_id} expected 500 training examples, found {len(rows)}")
        rows = rng.permutation(rows)
        train_rows.append(rows[:train_per_class])
        remainder = rows[train_per_class:]
        midpoint = len(remainder) // 2
        selection_rows.append(remainder[:midpoint])
        replication_rows.append(remainder[midpoint:])
    return (
        np.sort(np.concatenate(train_rows)).astype(np.int64),
        np.sort(np.concatenate(selection_rows)).astype(np.int64),
        np.sort(np.concatenate(replication_rows)).astype(np.int64),
    )


def _channel_stats(images: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = images.reshape(len(images), 3, 1024).astype(np.float64) / 255.0
    mean = values.mean(axis=(0, 2))
    std = values.std(axis=(0, 2))
    if np.any(std < 1e-6):
        raise ValueError("CIFAR-100 channel standard deviation is degenerate")
    return mean.astype(np.float32), std.astype(np.float32)


def _decode_names(values: Iterable[Any]) -> list[str]:
    names: list[str] = []
    for value in values:
        names.append(value.decode("utf-8") if isinstance(value, bytes) else str(value))
    return names


@dataclass(frozen=True)
class CIFAR100Manifest:
    value: dict[str, Any]

    @property
    def sha256(self) -> str:
        return str(self.value["sha256"])


class CIFAR100DatasetStore:
    """Content-addressed CIFAR-100 data prepared for a Native10 campaign.

    Images remain uint8 on disk. Normalized float32 rows are materialized only
    for the bounded shard or verifier split being used, so the contributor path
    does not require loading the full dataset into memory.
    """

    SPLITS = ("train", "selection", "replication", "test")

    def __init__(self, path: str | os.PathLike[str]):
        self.path = Path(path)
        self.manifest_path = self.path / "manifest.json"
        self._verified_files: set[str] = set()

    def prepared(self) -> bool:
        return self.manifest_path.exists()

    def manifest(self) -> dict[str, Any]:
        if not self.manifest_path.exists():
            raise ValueError("CIFAR-100 is not prepared")
        value = json.loads(self.manifest_path.read_text())
        expected = content_hash({key: item for key, item in value.items() if key != "sha256"})
        if value.get("format") != DATASET_FORMAT or value.get("sha256") != expected:
            raise ValueError("CIFAR-100 manifest integrity check failed")
        return value

    def prepare(self, source: str | os.PathLike[str], *, seed: int = 20260723, replace: bool = False) -> dict[str, Any]:
        source_path = Path(source).expanduser().resolve()
        if not source_path.exists():
            raise FileNotFoundError(source_path)
        if self.prepared() and not replace:
            raise ValueError("CIFAR-100 is already prepared")
        self.path.mkdir(parents=True, exist_ok=True)
        if source_path.is_file():
            md5 = _file_digest(source_path, "md5")
            if md5 != CIFAR100_MD5:
                raise ValueError(f"CIFAR-100 archive MD5 mismatch: {md5}")
            with tempfile.TemporaryDirectory(prefix="dendriswarm-cifar100-") as temporary:
                root = _extract_archive(source_path, Path(temporary))
                loaded = self._load_python_directory(root)
            source_sha256 = _file_digest(source_path, "sha256")
            source_kind = "official-python-tar-gz"
        elif source_path.is_dir():
            root = source_path / "cifar-100-python" if (source_path / "cifar-100-python").exists() else source_path
            loaded = self._load_python_directory(root)
            source_sha256 = content_hash({
                name: _file_digest(root / name, "sha256") for name in ("train", "test", "meta")
            })
            md5 = None
            source_kind = "extracted-official-python-directory"
        else:
            raise ValueError("CIFAR-100 source must be an archive or directory")

        train_images, train_fine, train_coarse, test_images, test_fine, test_coarse, fine_names, coarse_names = loaded
        official_to_native, native_to_official, grouped = _native_label_mapping(
            train_fine, train_coarse, test_fine, test_coarse
        )
        native_train = official_to_native[train_fine]
        native_test = official_to_native[test_fine]
        native_coarse_train = native_train // 5
        native_coarse_test = native_test // 5
        if not np.array_equal(native_coarse_train, train_coarse) or not np.array_equal(native_coarse_test, test_coarse):
            raise ValueError("Native10 class grouping does not preserve CIFAR-100 coarse labels")
        train_idx, selection_idx, replication_idx = _stratified_split(native_train, seed=seed)
        mean, std = _channel_stats(train_images[train_idx])
        split_values = {
            "train": (train_images[train_idx], native_train[train_idx]),
            "selection": (train_images[selection_idx], native_train[selection_idx]),
            "replication": (train_images[replication_idx], native_train[replication_idx]),
            "test": (test_images, native_test),
        }
        arrays: dict[str, dict[str, Any]] = {}
        temporary_files: list[tuple[Path, Path]] = []
        try:
            for split, (images, labels) in split_values.items():
                for suffix, array in (("images", images.astype(np.uint8, copy=False)), ("labels", labels.astype(np.int16, copy=False))):
                    destination = self.path / f"{split}-{suffix}.npy"
                    fd, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}.", dir=self.path)
                    os.close(fd)
                    temporary = Path(temporary_name)
                    with temporary.open("wb") as handle:
                        np.save(handle, array, allow_pickle=False)
                        handle.flush()
                        os.fsync(handle.fileno())
                    temporary_files.append((temporary, destination))
                    arrays[f"{split}_{suffix}"] = {
                        "file": destination.name,
                        "shape": list(array.shape),
                        "dtype": str(array.dtype),
                        "sha256": _array_digest(array),
                        "file_sha256": _file_digest(temporary, "sha256"),
                    }
            manifest: dict[str, Any] = {
                "format": DATASET_FORMAT,
                "dataset": "CIFAR-100",
                "official_url": CIFAR100_URL,
                "official_archive_md5": CIFAR100_MD5,
                "observed_archive_md5": md5,
                "source_kind": source_kind,
                "source_sha256": source_sha256,
                "seed": int(seed),
                "input_width": 3072,
                "classes": 100,
                "categories": 20,
                "classes_per_category": 5,
                "split_counts": {split: int(len(labels)) for split, (_, labels) in split_values.items()},
                "counts_by_class": {
                    split: np.bincount(labels, minlength=100).astype(int).tolist()
                    for split, (_, labels) in split_values.items()
                },
                "normalization": {
                    "source_layout": "channel-major-r1024-g1024-b1024",
                    "model_layout": "spatial-patches-4x2-c3-h8-w16",
                    "scale": 255.0,
                    "channel_mean": mean.astype(float).tolist(),
                    "channel_std": std.astype(float).tolist(),
                },
                "fine_label_names_official": fine_names,
                "coarse_label_names": coarse_names,
                "official_to_native": official_to_native.astype(int).tolist(),
                "native_to_official": native_to_official.astype(int).tolist(),
                "fine_labels_by_coarse_category": grouped,
                "native_label_names": [fine_names[index] for index in native_to_official],
                "arrays": arrays,
                "test_used_for_selection": False,
            }
            manifest["sha256"] = content_hash(manifest)
            for temporary, destination in temporary_files:
                os.replace(temporary, destination)
            fd, temporary_name = tempfile.mkstemp(prefix=".manifest.", dir=self.path)
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(manifest, handle, sort_keys=True, separators=(",", ":"), allow_nan=False)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_name, self.manifest_path)
            return self.status()
        finally:
            for temporary, _ in temporary_files:
                temporary.unlink(missing_ok=True)

    @staticmethod
    def _load_python_directory(root: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str], list[str]]:
        train = _unpickle(root / "train")
        test = _unpickle(root / "test")
        meta = _unpickle(root / "meta")
        train_images = np.asarray(train.get(b"data"), dtype=np.uint8)
        test_images = np.asarray(test.get(b"data"), dtype=np.uint8)
        if train_images.shape != (50_000, 3072) or test_images.shape != (10_000, 3072):
            raise ValueError("CIFAR-100 image matrices have unexpected shapes")
        train_fine = _as_label_array(train.get(b"fine_labels"), "train fine labels", 50_000)
        train_coarse = _as_label_array(train.get(b"coarse_labels"), "train coarse labels", 50_000)
        test_fine = _as_label_array(test.get(b"fine_labels"), "test fine labels", 10_000)
        test_coarse = _as_label_array(test.get(b"coarse_labels"), "test coarse labels", 10_000)
        fine_names = _decode_names(meta.get(b"fine_label_names", []))
        coarse_names = _decode_names(meta.get(b"coarse_label_names", []))
        if len(fine_names) != 100 or len(coarse_names) != 20:
            raise ValueError("CIFAR-100 metadata is incomplete")
        return train_images, train_fine, train_coarse, test_images, test_fine, test_coarse, fine_names, coarse_names

    def _load_array(self, split: str, suffix: str, *, mmap: bool = True) -> np.ndarray:
        manifest = self.manifest()
        if split not in self.SPLITS:
            raise ValueError(f"unknown CIFAR-100 split: {split}")
        info = manifest["arrays"][f"{split}_{suffix}"]
        path = self.path / info["file"]
        verification_key = f"{path}:{info.get('file_sha256', '')}"
        if verification_key not in self._verified_files:
            expected_file = info.get("file_sha256")
            if expected_file and _file_digest(path, "sha256") != expected_file:
                raise ValueError(f"CIFAR-100 {split} {suffix} file hash mismatch")
            self._verified_files.add(verification_key)
        array = np.load(path, mmap_mode="r" if mmap else None, allow_pickle=False)
        if list(array.shape) != info["shape"] or str(array.dtype) != info["dtype"]:
            raise ValueError(f"CIFAR-100 {split} {suffix} metadata mismatch")
        return array

    def images(self, split: str) -> np.ndarray:
        return self._load_array(split, "images")

    def labels(self, split: str) -> np.ndarray:
        return self._load_array(split, "labels")

    def normalized_rows(self, split: str, rows: np.ndarray | list[int] | slice | None = None) -> tuple[np.ndarray, np.ndarray]:
        manifest = self.manifest()
        images = self.images(split)
        labels = self.labels(split)
        selected_images = np.asarray(images if rows is None else images[rows], dtype=np.uint8)
        selected_labels = np.asarray(labels if rows is None else labels[rows], dtype=np.int64)
        return self.normalize_images(selected_images), selected_labels

    def normalize_images(self, images: np.ndarray) -> np.ndarray:
        manifest = self.manifest()
        values = np.asarray(images, dtype=np.uint8)
        if values.ndim != 2 or values.shape[1] != 3072:
            raise ValueError("CIFAR-100 images must have shape [N,3072]")
        shaped = values.reshape(len(values), 3, 32, 32).astype(np.float32) / float(manifest["normalization"]["scale"])
        mean = np.asarray(manifest["normalization"]["channel_mean"], dtype=np.float32)
        std = np.asarray(manifest["normalization"]["channel_std"], dtype=np.float32)
        shaped = (shaped - mean[None, :, None, None]) / std[None, :, None, None]
        patches = shaped.reshape(len(values), 3, 4, 8, 2, 16)
        patches = patches.transpose(0, 2, 4, 1, 3, 5).reshape(len(values), 8, 384)
        return patches.reshape(len(values), 3072).astype(np.float32)

    @staticmethod
    def augment_images(images: np.ndarray, *, seed: int, crop_padding: int = 4, horizontal_flip: bool = True) -> np.ndarray:
        """Apply deterministic CIFAR crop/flip augmentation to bounded rows."""
        values = np.asarray(images, dtype=np.uint8)
        if values.ndim != 2 or values.shape[1] != 3072:
            raise ValueError("CIFAR-100 augmentation expects [N,3072] uint8 rows")
        rng = np.random.default_rng(seed)
        nchw = values.reshape(len(values), 3, 32, 32)
        if crop_padding < 0 or crop_padding > 8:
            raise ValueError("crop_padding must be between 0 and 8")
        if crop_padding:
            padded = np.pad(nchw, ((0, 0), (0, 0), (crop_padding, crop_padding), (crop_padding, crop_padding)), mode="reflect")
            augmented = np.empty_like(nchw)
            offsets = rng.integers(0, crop_padding * 2 + 1, size=(len(values), 2))
            for index, (row, column) in enumerate(offsets):
                augmented[index] = padded[index, :, row:row + 32, column:column + 32]
        else:
            augmented = nchw.copy()
        if horizontal_flip:
            mask = rng.random(len(values)) < 0.5
            augmented[mask] = augmented[mask, :, :, ::-1]
        return augmented.reshape(len(values), 3072)

    def encoded_patch_input(self, images: np.ndarray) -> dict[str, Any]:
        manifest = self.manifest()
        return {
            "format": "dendriswarm.cifar100-patch-input.v1",
            "array": encode_array(np.asarray(images, dtype=np.uint8)),
            "source_layout": "channel-major-r1024-g1024-b1024",
            "model_layout": "spatial-patches-4x2-c3-h8-w16",
            "scale": float(manifest["normalization"]["scale"]),
            "channel_mean": list(manifest["normalization"]["channel_mean"]),
            "channel_std": list(manifest["normalization"]["channel_std"]),
        }

    def iter_normalized_batches(self, split: str, *, batch_size: int = 128):
        if batch_size < 1 or batch_size > 4096:
            raise ValueError("batch_size must be between 1 and 4096")
        count = int(self.manifest()["split_counts"][split])
        for start in range(0, count, batch_size):
            stop = min(count, start + batch_size)
            yield self.normalized_rows(split, slice(start, stop))

    def balanced_indices(self, split: str, *, per_class: int, seed: int) -> np.ndarray:
        labels = np.asarray(self.labels(split), dtype=np.int64)
        rng = np.random.default_rng(seed)
        rows: list[np.ndarray] = []
        for class_id in range(100):
            choices = np.flatnonzero(labels == class_id)
            if len(choices) < per_class:
                raise ValueError(f"split {split} has fewer than {per_class} rows for class {class_id}")
            rows.append(rng.choice(choices, size=per_class, replace=False))
        result = np.concatenate(rows).astype(np.int64)
        return result[rng.permutation(len(result))]

    def budgeted_balanced_indices(self, split: str, *, class_ids: Iterable[int], total: int, seed: int) -> np.ndarray:
        classes = [int(value) for value in class_ids]
        if not classes or total < len(classes):
            raise ValueError("balanced budget must include at least one row per class")
        labels = np.asarray(self.labels(split), dtype=np.int64)
        rng = np.random.default_rng(seed)
        order = list(np.asarray(classes)[rng.permutation(len(classes))].astype(int))
        base, remainder = divmod(int(total), len(classes))
        rows: list[np.ndarray] = []
        for position, class_id in enumerate(order):
            count = base + (1 if position < remainder else 0)
            pool = np.flatnonzero(labels == class_id)
            if len(pool) < count:
                raise ValueError(f"split {split} has fewer than {count} rows for class {class_id}")
            rows.append(rng.choice(pool, size=count, replace=False))
        result = np.concatenate(rows).astype(np.int64)
        return result[rng.permutation(len(result))]

    def training_shard(
        self,
        model: Native10Dendritron,
        *,
        operation: str,
        target: int,
        sample_budget: int,
        seed: int,
        hard_negative_fraction: float = 0.5,
    ) -> dict[str, Any]:
        from dendriswarm.v6.native10 import canonical_operation

        operation = canonical_operation(operation)
        if sample_budget < 20 or sample_budget > 10_000:
            raise ValueError("sample_budget must be between 20 and 10,000")
        rng = np.random.default_rng(seed)
        labels = np.asarray(self.labels("train"), dtype=np.int64)
        if operation == "field_train":
            rows = self.budgeted_balanced_indices("train", class_ids=range(100), total=sample_budget, seed=seed)
            images = np.asarray(self.images("train")[rows], dtype=np.uint8)
            images = self.augment_images(images, seed=seed)
            data = images
            selected_labels = np.asarray(self.labels("train")[rows], dtype=np.int64)
        elif operation == "scout_train":
            if not 0 <= target < model.config.categories:
                raise ValueError("scout target is invalid")
            positive_pool = np.flatnonzero(labels // 5 == target)
            positive_count = max(10, int(round(sample_budget * (1.0 - hard_negative_fraction))))
            positive_count = min(positive_count, len(positive_pool), sample_budget - 10)
            positive_rows = rng.choice(positive_pool, size=positive_count, replace=False)
            candidate_pool = np.flatnonzero(labels // 5 != target)
            candidate_count = min(len(candidate_pool), max(sample_budget * 8, 2_000))
            candidate_rows = rng.choice(candidate_pool, size=candidate_count, replace=False)
            candidate_x, _ = self.normalized_rows("train", candidate_rows)
            candidate_h = model.encode(candidate_x)
            route = model.route_scores(candidate_h)
            negative_count = sample_budget - positive_count
            hardest = np.argsort(route[:, target])[::-1][:negative_count]
            negative_rows = candidate_rows[hardest]
            rows = np.concatenate([positive_rows, negative_rows])
            rows = rows[rng.permutation(len(rows))]
            images = np.asarray(self.images("train")[rows], dtype=np.uint8)
            images = self.augment_images(images, seed=seed)
            raw = self.normalize_images(images)
            selected_labels = np.asarray(self.labels("train")[rows], dtype=np.int64)
            data = model.encode(raw)
        else:
            if not 0 <= target < model.config.categories:
                raise ValueError("category target is invalid")
            class_ids = np.arange(target * 5, target * 5 + 5)
            rows = self.budgeted_balanced_indices("train", class_ids=class_ids, total=sample_budget, seed=seed)
            images = np.asarray(self.images("train")[rows], dtype=np.uint8)
            images = self.augment_images(images, seed=seed)
            raw = self.normalize_images(images)
            selected_labels = np.asarray(self.labels("train")[rows], dtype=np.int64)
            data = model.encode(raw)
        shard: dict[str, Any] = {
            "format": SHARD_FORMAT,
            "dataset_sha256": self.manifest()["sha256"],
            "model_root": model.root,
            "operation": operation,
            "category": int(target),
            "sample_count": int(len(selected_labels)),
            "train_labels": selected_labels.astype(int).tolist(),
            "sample_seed": int(seed),
            "hard_negative_fraction": float(hard_negative_fraction),
        }
        if operation == "field_train":
            shard["train_inputs"] = self.encoded_patch_input(data)
        else:
            shard["train_representations"] = encode_array(data)
        shard["augmentation"] = {"crop_padding": 4, "horizontal_flip": True, "seed": int(seed)}
        shard["sha256"] = content_hash(shard)
        return shard

    def status(self) -> dict[str, Any]:
        if not self.prepared():
            return {"prepared": False, "dataset": "CIFAR-100"}
        manifest = self.manifest()
        return {
            "prepared": True,
            "dataset": manifest["dataset"],
            "sha256": manifest["sha256"],
            "source_kind": manifest["source_kind"],
            "source_sha256": manifest["source_sha256"],
            "split_counts": manifest["split_counts"],
            "classes": manifest["classes"],
            "categories": manifest["categories"],
            "test_used_for_selection": manifest["test_used_for_selection"],
        }
