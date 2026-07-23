# DendriSwarm v0.8.0 architecture


## Local application layer — v0.8.0

The default CLI dispatches to a loopback-only dashboard when no subcommand is supplied. The dashboard reads and atomically updates the contributor `SeedPolicy`, supervises seed/coordinator child processes, polls public coordinator telemetry, and invokes campaign mutations only through token-authenticated admin routes. Static HTML/JS is packaged inside the Python wheel; no separate frontend service is present.

```text
browser on loopback
      ↓ dashboard token
stdlib dashboard server
      ├── atomic dashboard config
      ├── hot-reloaded SeedPolicy
      ├── seed/coordinator process supervisor
      └── local coordinator admin token
                ↓
       CIFAR-100 campaign service
```

## CIFAR-100 campaign layer

The v0.7 campaign wraps the trainable Native10 protocol in a durable real-data control loop:

```text
verified official CIFAR-100 archive
  → explicit fine/coarse remapping into 20 colonies
  → 45k trainer pool + one-shot hidden selection/replication folds
  → routing-gap telemetry and bottleneck-aware planning
  → independent volunteer candidate search
  → hidden all-class selection + fresh replication
  → canonical checkpoint promotion and provenance
  → final-report-only official test evaluation
```

The official 10,000-example test set is never accepted as planner input or promotion evidence. The campaign halts when its committed hidden evidence bank is exhausted rather than silently reusing adaptive holdouts.

## Four planes

```text
Independent search plane
  bounded tissue + trainer shard + distinct search trajectory
  → competing sparse candidates

Blind selection plane
  parent checkpoint + candidate delta + hidden all-class selection artifact
  → paired class-complete evidence

Replay and replication plane
  deterministic replay by a new identity
  + fresh hidden all-class replication artifact
  → final evidence

Canonical plane
  validated sparse write set → deterministic composition → new root + provenance
```

## Model topology

The exact profile contains eight trainable field blocks, a 96-wide mixed representation, 1,000 trainable routing scouts, 20 colonies, 45 experts per colony, four nonlinear branches per expert, category evidence, and learned associative memory. Low-margin routing may expand beyond the ordinary top four to a committed maximum of eight categories.

## Parameter ownership

Every persistent tensor is classified as trainable float or persistent state and mapped to one or more operations. `parameter_reachability()` reports tensor shape, owner operations, operation coverage, and total reachable fraction. Unowned trainable tensors fail the v0.6 proof.

## Sparse delta contract

A delta commits to:

- exact parent root;
- exact work-bundle hash;
- canonical operation;
- operation-schema hash;
- target category or field block;
- sparse tensor selectors and encoded values;
- exact write set;
- changed-element count;
- search seed and diagnostic scope.

Validation enforces operation-specific tensor names, exact target slices, active expert ownership, finite values, magnitude bounds, shape bounds, duplicate-selector rejection, changed-element ceilings, and content hashes. The same validation runs before evidence generation and during composition.

## Statistical promotion

For each candidate, verifiers return paired pre/post evidence on a hidden all-class artifact. The coordinator verifies:

- artifact sample count and exact class counts;
- class-vector lengths;
- aggregate totals from class totals;
- class losses and loss rates;
- paired wins, losses, discordance, and net wins;
- effect rate;
- exact one-sided McNemar probability;
- committed Bonferroni threshold;
- informative/significant flags;
- operation, target, delta, root, validation hash, and write set.

Candidates are ranked only after all selection evaluations complete. The selection artifact is then exhausted and must be rotated. The best valid candidate is replayed independently and evaluated once on a distinct fresh replication artifact. If final replication fails, the round closes; another candidate is not adaptively tested on that same holdout.

## Independent search versus replay

Candidate generators intentionally receive different seeds, step counts, and learning rates. Their purpose is search diversity. Exact deterministic replay is a later audit of the selected trajectory; it is not presented as an independent opinion.

## Composition and concurrency

`compose_non_conflicting_deltas()` sorts same-parent deltas by hash and rejects any overlapping tensor intervals. This proves deterministic composition of disjoint writes.

The reference coordinator keeps one adaptive selection round active at a time. Live multi-round scheduling and batched final promotion remain future scalability work.

## Baseline comparison boundary

The optional baseline artifact stores only a metric value, dataset/split identity, model label, source, and SHA-256 of external evidence. It contains no trainer or weights. Canonical evaluation reports bind the model root and evaluation data commitment. Comparison is allowed only when dataset, split, and metric match.

## Retained public-network substrate

v0.6 inherits transactionally enforced leases, bounded requests and artifacts, payload-derived resource limits, killable subprocess execution, durable result delivery, TLS/fingerprint controls, identity-distinct work assignment, and hostile-participation tests from v0.4.1/v0.5.1.
