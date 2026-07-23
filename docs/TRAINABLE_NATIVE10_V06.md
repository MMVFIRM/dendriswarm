# Trainable Native10 v0.6

## Why this revision exists

v0.5.1 established a careful contribution and promotion protocol but left most Native10 capacity frozen. v0.6 makes the topology's counted capacity reachable by bounded volunteer operations.

## Reachability

The exact profile has 4,898,812 trainable floating-point parameters. The generated reachability report assigns every float tensor family to an owner operation and separately accounts for 120 persistent state elements.

Large families are now operationally reachable:

- `expert_branch_weights`: `expert_train`, `branch_train`, `repair`;
- `scout_weights`: `scout_train`;
- `field_weights` and `field_mixer`: `field_train`;
- `associative_memory` and `associative_strength`: `memory_train`.

## Local optimization

The worker executes bounded local SGD over the selected tissue. Only the declared target slices are materialized into the delta. The trainer reports its local loss and correctness solely as a diagnostic; hidden evidence determines promotion.

The implementation is intentionally local and CPU-portable. It is not an unrestricted end-to-end PyTorch trainer and does not claim to reproduce the historical optimizer exactly.

## Routing recovery

Ordinary inference routes through the top four categories. When routing scores are within the configured uncertainty margin, evaluation expands to additional categories up to the committed maximum. This prevents every top-four miss from becoming mathematically unrecoverable while preserving sparse normal execution.

## Candidate tournaments

A search round produces multiple independently generated candidates. Blind verifiers evaluate each against the same committed selection artifact, with the full candidate count included in the familywise correction budget. The strongest statistically valid candidate proceeds to replay and fresh replication.

## Honest scalability boundary

Sparse deltas materially reduce trainer upload size, but complete verification still evaluates the full routed checkpoint. Content-addressed caching avoids repeated cold downloads for the same root. The core supports deterministic conflict-free merge, while the reference coordinator remains single-round. Public-scale leverage is therefore an empirical question, not a release claim.
