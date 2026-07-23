# v0.5.1 Native10 topology contribution architecture

## Purpose and claim boundary

v0.5.1 distributes real mutations of a persistent Native10-derived tensor topology. It demonstrates topology execution, category ownership, bounded mutation, independent all-class verification, deterministic composition, provenance, and INT8 export.

It deliberately does not bundle baseline training or benchmark datasets. The packaged proof initializes tensors and uses synthetic protocol fixtures; it does not exercise the historical trained Native10 checkpoint or claim benchmark accuracy.

## Exact topology executed

| Invariant | Exact profile |
|---|---:|
| Sensory field blocks | 8 |
| Shared representation width | 96 |
| Fine routing scouts | 1,000 |
| Admitted categories | top 4 |
| Category colonies | 20 |
| Experts per colony | 45 |
| Active experts per update | 15 |
| Rotation groups | 3 |
| Branches per expert | 4 |
| Output classes | 100 |
| Persistent parameters | 4,898,712 |

The proof initializes this exact profile, exports and reimports its tensor archive, mutates one category, composes the delta, and evaluates the complete 100-class routed model.

## Trainer data boundary

A trainer receives:

- One content-addressed category bundle.
- One category-local training representation shard.
- The global validation artifact hash.
- No global validation features or labels.

Mutation metrics are diagnostics over trainer-visible data and carry the literal scope marker:

```text
trainer-visible-training-diagnostic-not-promotion-evidence
```

They are never consumed by the promotion gate.

## Coordinator-held global validation

The operator installs a hash-bound artifact containing representations and labels for every class. Its commitment includes:

- Model configuration.
- All encoded arrays.
- Sample counts by class.
- Source and split metadata.
- Minimum samples per class.
- Minimum net wins.
- Maximum integer loss per class.
- Maximum loss rate per class.
- Maximum candidate evaluations before rotation.

Only an identity currently holding the matching `dendritron-verification` lease may retrieve it. The endpoint verifies the signed node request, task assignment, lease token, expiry, task kind, candidate payload, and validation hash.

This prevents accidental trainer exposure within the protocol. It does not defeat a Sybil actor who controls both trainer and verifier principals; principal independence remains an external deployment requirement.

## Complete-model promotion evidence

Two non-trainer verifiers fetch:

- The exact parent checkpoint by model root.
- The exact candidate delta.
- The exact coordinator-held all-class artifact.

Each independently runs the full routed model before and after deterministic composition and returns integer evidence for every class:

```text
sample_count
pre_correct
post_correct
net_wins
samples_by_class
pre_correct_by_class
post_correct_by_class
losses_by_class
loss_rates_by_class
validation_hash
```

Promotion requires identity-distinct consensus, full class coverage, positive net wins, committed class-loss bounds, an informative comparison, and a fresh canonical parent.

The regression suite includes a candidate whose trainer diagnostics are positive while its all-class effect is negative. The candidate is rejected and the root remains unchanged.

## Mutation operations

- `expert_refit`: refits the next deterministic 15-of-45 expert group.
- `repair`: repairs low-health experts from category-local donors and refits them.
- `branch_lifecycle`: prunes, regrows, and refits branch tissue.
- `scout_refit`: updates category routing scouts and evidence.
- `memory_update`: updates class-addressed associative memory.

Scout and memory changes use the same global gate because their effects can extend beyond the assigned colony.

## Delta binding and composition

A delta binds:

- Canonical parent root.
- Component-bundle hash.
- Operation and category.
- Exact changed slices and tensor values.
- Active expert indices and rotation phases.
- Content hash over the complete delta.

The store validates operation-specific tensor names, shapes, finite values, owner boundaries, and deterministic recomposition. A volunteer cannot submit a replacement full checkpoint.

## Lease serialization

The one-active-lease invariant does not depend solely on a Python lock. Claims execute under SQLite `BEGIN IMMEDIATE`, use a conditional assignment that checks for another active row, and are protected by a partial unique index on active `assigned_to` values. A test races two independent database connections against the same file and proves that exactly one claim succeeds.

## Transfer asymmetry

Measured exact-profile JSON sizes in the v0.5.1 proof:

| Item | Bytes |
|---|---:|
| Canonical checkpoint | 23,933,844 |
| Trainer bundle | 1,116,811 |
| Expert-refit delta | 1,105,903 |
| Small 100-class validation artifact | 49,385 |
| Two cold verifier checkpoint transfers | 47,867,688 |

Trainers are cheap relative to verifiers. Checkpoints are content-addressed and cached by root, so repeated verification against an unchanged root may reuse the local artifact. Every new root requires a new checkpoint representation.

## Explicit exclusions

v0.5.1 does not claim:

- Baseline or benchmark accuracy.
- Historical trained-weight equivalence.
- Confidential validation against colluding principals.
- Production Sybil resistance.
- Universal cross-platform bit identity beyond the versioned consensus quantization policy.
- Public-scale economic leverage before live measurements exist.
