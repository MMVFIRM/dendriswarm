# Contributing to DendriSwarm

Contributions must preserve portability, real Native10 trainability, statistical validity, and hostile-participant safety.

## Required checks

```bash
python -m pip install -e '.[dev]'
make check
python -m build
```

## Native10 v0.6 invariants

1. Baseline training and benchmark datasets remain outside the package unless a future version explicitly changes the release boundary.
2. Every persistent floating-point tensor family must have a bounded owner operation or be removed from the trainable parameter count.
3. A mutation is a sparse operation-scoped patch, never an unrestricted full-model replacement.
4. Every delta binds parent root, bundle hash, schema hash, target, selectors, write set, search trajectory, and content hash.
5. Operation-specific schemas must reject cross-tissue writes, duplicate selectors, escaped territories, non-finite values, excessive magnitude, and excessive changed-element counts.
6. Independent search tasks must differ in trajectory; deterministic replay is a separate audit stage.
7. Trainers receive no selection or replication features/labels.
8. Selection and replication artifacts must be different and cover every model class.
9. The coordinator independently recomputes all aggregate, class-specific, paired, and statistical evidence fields.
10. Promotion requires exact McNemar significance, committed multiple-comparison correction, practical effect, bounded class harm, replay, and fresh replication.
11. Accepted deltas must produce exact lineage and public contribution records.
12. Same-parent delta composition must reject every overlapping write interval and remain order-deterministic.
13. INT8 export must bind the exact canonical root.
14. Baseline comparison may import an external result but must not imply the package reproduced that baseline.
15. Do not strengthen benchmark or leverage claims without external data and a reproducible artifact.

## Transport and worker invariants

1. One identity holds at most one active lease and one replica per logical work key.
2. Expired or invalid work cannot be reclaimed indefinitely; renewal cannot exceed an absolute deadline.
3. Payload-derived local resource requirements override optimistic coordinator estimates.
4. Downloads, JSON, requests, outputs, cache writes, child RSS, and runtime remain bounded.
5. Hot-reloaded policy can terminate active work safely.
6. Results are persisted before transmission and retained until a terminal response or signed receipt.
7. Remote coordinators require TLS unless explicitly overridden; fingerprint pinning remains available.
8. New execution backends require estimators, schemas, consensus rules, verifiers, and adversarial tests.
9. The default seed installation remains GPU-free and coordinator-dependency-free.

## Proof discipline

The synthetic fixtures are protocol/model-mechanics fixtures, not performance benchmarks. Keep claim language synchronized across `README.md`, `docs/CLAIMS.md`, proof scripts, release notes, and `RELEASE_MANIFEST.json`.

## Contributing to the CIFAR-100 campaign

Ordinary contributors do not download CIFAR-100. The coordinator sends only the bounded shard required by an assigned mutation task. Field tasks receive augmented uint8 rows plus committed normalization metadata; scout and colony tasks receive bounded 96-wide representations.

A contribution may be either model training or search:

- field, expert, branch, memory, and repair tasks optimize model tissues;
- scout and field-routing recipes search ways to close the measured routing gap;
- verification tasks independently score exact candidate deltas on trainer-invisible evidence.

Contributor output must remain deterministic under the signed recipe and seed. Do not add executable-code payloads, arbitrary plugins, or network access to worker tasks. New search algorithms must be represented as bounded, versioned recipe fields with exact validation and tests.

Real campaign changes should include tests for data isolation, official-test non-use, class mapping, shard bounds, recipe commitments, and final report provenance.
