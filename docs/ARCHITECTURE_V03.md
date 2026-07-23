# DendriSwarm v0.3.2 architecture

## 1. Structural integrity invariant

The v0.3.2 invariant is structural rather than behavioral:

```text
current lineage root
+ sole candidate manifest
+ rehashed replacement/addition artifacts
+ versioned deterministic composition kernel
+ parent representation-schema root
──────────────────────────────────────────────
= candidate lineage root and composition certificate
```

A submission contains exactly one `CandidateManifest`. Every branch artifact is embedded in and rehashed from that manifest. There is no second center array, branch list, or evaluator-only artifact channel.

The guarded evaluator computes touched distances and constructs the exact candidate cache in the same deterministic branch ordering used by composition. An accepted candidate is composed only after territorial and statistical gates pass. The service verifies structural facts—branch count, owners, manifest hashes, representation root, parent root, and composition certificate—without replaying a full candidate prediction pass. Rejected candidates never pay a full binding evaluation because no such pass exists.

A full state hash is still used for genesis registration, persistence validation, and explicit independent replication. It is not part of per-candidate admission binding.

## 2. Canonical tissue and route regions

A `TerritoryTissue` contains:

- branch centers and semantic owners;
- immutable route anchors inherited from genesis;
- an active-route-region mask for every branch;
- deterministic top-k radial evidence parameters.

For a replacement inside declared route region set `T`:

```text
old branch: active in old_mask \ T
new branch: active in old_mask ∩ T
```

An added branch is active only in `T`. Empty historical branches are removed deterministically. Anchors and masks are part of the representation root.

A route region is a nearest inherited anchor region. It is **not** claimed to equal a true semantic class. Semantic labels are separately used by the subgroup harm gate when labels are available.

## 3. Submission state machine

```text
signed manifest
    ├─ exact duplicate? return prior result
    ├─ authenticate contributor == manifest contributor
    ├─ active canary? defer without charge
    ├─ stale parent? reject without challenge use or bond loss
    ├─ enforce global and contributor budgets
    ├─ burn escalating fee; lock bond
    ├─ reconstruct and validate manifest artifacts
    ├─ guarded incremental evaluation against current canonical cache
    ├─ route-bound, aggregate, significance, effect, and subgroup gates
    │
    ├─ reject → burn bond under the committed economics
    └─ would accept
         ├─ deterministic structural composition
         ├─ register candidate lineage root
         ├─ install exact incremental candidate cache
         ├─ lock reward
         └─ open concrete-observation canary
```

Staleness is treated as concurrency, not misbehavior. It is checked before protected evaluation and accounting. The default stale fee is zero, the bond is never locked, and no epoch test is consumed.

Only one canary can be active at a time. Submissions arriving during that window are deferred without charges and are not persisted as final candidate verdicts, so they may be retried against the later canonical root.

## 4. Statistical admission and subgroup harm

Admission compares aligned parent and candidate correctness vectors with an exact one-sided paired McNemar test. The committed policy specifies:

- epoch-wide alpha and Bonferroni per-test alpha;
- minimum aggregate net wins;
- minimum represented subgroup size;
- maximum allowed net loss in any semantic-label subgroup;
- territory, route-share, and touched-branch limits.

A candidate can therefore be rejected even with positive aggregate wins when those wins are financed by concentrated damage to a victim label. This is a distributional-harm gate, not a general backdoor detector.

## 5. Canary observations

The removed API was:

```text
record_canary_event(candidate_id, clean: bool)
```

The v0.3.2 API accepts an observable:

```text
candidate id
labeled feature batch
source id
committed source kind
optional subgroup ids
```

The HTTP adapter additionally requires a registered observer listed in `DENDRISWARM_CANARY_AUDITORS` and forbids a candidate contributor from auditing its own candidate. The service validates dimensions and minimum sample count, hashes the complete observation data, predicts with the materialized parent and candidate roots, computes aggregate and subgroup paired outcomes, and applies the committed thresholds. Duplicate data cannot be counted twice.

Supported reference source kinds are:

- `heldout-canary-batch`: coordinator/auditor-held labeled examples not used for admission;
- `audited-inference-batch`: labeled outcomes sampled from an auditable inference stream.

The mechanism defines and computes cleanliness; it does not accept an oracle Boolean. A single-coordinator deployment still depends on data governance: a compromised coordinator or authorized observer can select unrepresentative evidence. That limitation is explicit.

A canary pass does not vest funds. It changes status to `canary-passed`, clears the active window, and leaves the bond and reward locked for final replication.

## 6. Adaptive reuse and final replication

All sequential admissions within an epoch use one committed protected challenge, so per-test alpha alone cannot eliminate adaptive-overfitting risk across composed promotions.

v0.3.2 therefore commits two protected artifacts at epoch open:

```text
H(admission challenge, final replication holdout, salt, complete policy)
```

The final holdout is never used for search, admission, or canary decisions. At epoch close, the service evaluates **genesis versus the final composed root once** on that holdout and applies aggregate significance, effect-size, and subgroup-loss thresholds.

- Pass: every `canary-passed` candidate in the surviving lineage vests its bond and reward.
- Fail: bonds are refunded, rewards are cancelled, candidate statuses become `replication-rejected`, and canonical state returns to genesis.

After closure, reveal discloses both complete artifacts, the salt, policy hash, and final replication result so third parties can reconstruct the commitment and outcome.

This final test validates the composed epoch lineage. It does not identify which individual candidate caused a failed lineage; finer attribution is future work.

## 7. Identity and anti-farming assumptions

Permissionless registration creates an account with zero units. It does not mint the former 8,000-unit grant.

The reference `fund_contributor()` operation maps one externally verified principal to at most one contributor and issues one idempotent grant. The principal identifier is hashed in the audit trail. Fee escalation and per-contributor budgets are meaningful only under this stated assumption or another scarce-cost funding mechanism.

The proof’s variance-farmer bankruptcy result is explicitly conditional on `externally-verified-one-grant-per-principal`. DendriSwarm does not claim protocol-native Sybil resistance.

## 8. Search and leverage measurement

The reference search harness receives only:

- the current public parent tissue;
- training features and labels;
- public validation features and labels;
- candidate territory blocks and search seeds.

It generates local refits, ranks them using public outcomes, and reports untrusted candidate-generation distance work. Protected admission, canary, and replication arrays are not function inputs.

The proof separately reports:

```text
untrusted search candidates and distance operations
trusted candidate admissions
trusted admission distance/copy/selection/aggregation work
trusted canary work
trusted cache rebuild work
trusted final replication work
fresh-holdout replicated net wins
replicated net wins per million total trusted distance operations
```

This is the first v0.3.x release to test a search-to-admission-to-replication path rather than inserting a hand-selected honest candidate as the leverage numerator.

## 9. Cost boundary

Candidate admission recomputes distances only for touched branches, but it still carries full-width costs for copying cached state, selecting top-k branches, and aggregating evidence. Composition adds hashing/materialization work. Canary and final replication are additional trusted costs.

The release therefore makes three distinct statements:

1. **Verified:** per-candidate new distance calculations are delta-proportional.
2. **Measured in the reference proof:** total trusted operations plus replicated utility are positive for the disclosed synthetic run.
3. **Not claimed:** generic end-to-end or wall-clock speedup over full evaluation on arbitrary hardware or workloads.

The structural-binding change removes the previous guaranteed anti-leverage of incremental evaluation plus a mandatory full behavioral replay.

## 10. Persistence, transport, and trust

With `--enable-leverage`, the coordinator persists the challenge and replication artifacts, policy, model store, lineage, ledger, candidates, canary observations, search evidence, final replication report, and audit chain in an atomically replaced state snapshot. Roots and audit history are validated on reload.

The leverage checkpoint is signed by the coordinator’s Ed25519 identity. This is durable single-coordinator evidence, not Byzantine consensus. Protected coordinator state must be secured until disclosure.

The original v0.2 task network remains available for prescribed exploration, training, verification, and inference. Inference-audit selection uses a coordinator-private persisted secret so workers cannot predict sub-100% audit choices from public task fields.
