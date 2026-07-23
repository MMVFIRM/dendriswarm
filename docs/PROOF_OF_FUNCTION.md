# Proof of function

## Complete verification

```bash
python -m pip install -e '.[dev]'
make check
```

The release retains seven proof families:

- v0.2 signed transport/model-promotion proof;
- v0.3.2 12-gate Locality Leverage proof;
- v0.4.0 9-gate heterogeneous-seeding proof;
- v0.4.1 11-gate hostile-public-participation proof;
- v0.5.1 15-gate trainer-blind all-class proof;
- v0.6.0 20-gate Trainable Native10 proof;
- v0.8.0 12-gate local dashboard and package-usability proof.
- v0.7.0 16-gate CIFAR-100 campaign-readiness proof.


## v0.7.0 proof

`docs/PROOF_RUN_V07.json` demonstrates:

1. Official CIFAR-100 fine/coarse label regrouping into 20 Native10 colonies.
2. A stratified 45,000/2,500/2,500 campaign split plus a final-report-only 10,000-example official test split.
3. Spatial conversion of channel-major CIFAR images into eight 8×16 RGB sensory fields.
4. Compact uint8 trainer transport with deterministic local normalization and decoding.
5. Bounded field, scout, expert, branch, memory, and repair training shards.
6. Complete-model routing-gap telemetry, including oracle-category accuracy and top-k category recall.
7. Planner decisions driven by the measured routing bottleneck rather than test-set feedback.
8. Distinct trainer search recipes for routing and tissue optimization.
9. Signed queue payloads bound to dataset, model root, operation, augmentation, and recipe.
10. Trainer blindness to selection, replication, and official-test rows.
11. One-shot selection and replication fold accounting.
12. Rejection of test-derived planner input.
13. Recipe-hash tamper rejection.
14. Execution of the exact 4,898,812-parameter Native10 profile on CIFAR-shaped tensors.
15. Explicit absence of packaged CIFAR data, historical weights, and baseline training code.
16. Optional verified ingestion of the official CIFAR-100 Python archive when supplied by the operator.

The packaged proof establishes campaign readiness without fabricating an accuracy result. In this build environment the official archive was unavailable, so `real_cifar100.executed` is false in the proof report. A benchmark claim requires a separately produced, content-addressed campaign report from the verified official archive.

## v0.6.0 proof

`docs/PROOF_RUN_V06.json` demonstrates:

1. Execution of the exact 4,898,812-parameter profile.
2. 100% protocol reachability across trainable tensor families.
3. Real nonlinear expert-branch-weight updates.
4. Exact operation-specific sparse schemas.
5. Multiple distinct search candidates from different trajectories.
6. Rejection of a noise-level `+1` gain.
7. Acceptance of strong paired evidence under corrected alpha.
8. Trainer blindness to both hidden artifacts.
9. Different selection and replication artifacts.
10. Significant selection evidence.
11. Significant fresh replication evidence.
12. Identity separation across search, selection, replay, and replication.
13. Canonical-root change only after replication.
14. Conflict-free deterministic composition.
15. Sparse delta size below the complete component bundle.
16. Valid audit lineage.
17. Coordinator rejection of forged/inconsistent verifier evidence.
18. Import-only baseline comparison with no packaged baseline trainer.
19. One-tournament selection and one-shot replication artifact rotation.
20. Explicit exclusion of historical trained weights and baseline training.

The exact-profile proof runs a real mutation and full 100-class inference. The network promotion proof uses a compact topology and synthetic raw-input fixtures so it remains reproducible on ordinary CPUs. Neither is an external benchmark-accuracy claim.
