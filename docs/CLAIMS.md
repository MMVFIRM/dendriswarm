# Claims and evidence boundary — DendriSwarm v0.8.0

| Claim | Status | Evidence |
|---|---|---|
| Running `dendriswarm` opens a local configuration dashboard | Demonstrated | CLI default dispatch, loopback HTTP test, packaged static asset, v0.8 proof |
| The dashboard requires Node.js, Electron, or a frontend framework | False by design | standard-library server and static HTML/JS; dependency audit |
| Dashboard APIs are reachable without the launch token | False by design | unauthorized request tests and `SameSite=Strict` HTTP-only cookie |
| The dashboard can expose a remote network listener | False by design | loopback-only bind validation |
| Contributor controls update the actual hot-reloaded worker policy | Demonstrated | atomic `SeedPolicyStore` integration and dashboard regression test |
| A user can start, pause, resume, and stop contribution from the UI | Demonstrated | managed-process and policy-control tests |
| Campaign mutations are authorized by the public coordinator API | False by design | separate local admin token and unauthorized endpoint regression |
| The dashboard exposes live model, routing, campaign, worker, credit, and log telemetry | Demonstrated in implementation | aggregate status endpoint and UI bindings |
| The package implements a real CIFAR-100 campaign path | Demonstrated | official-format ingestion, split, mapping, shard, planner, campaign, CLI, and v0.7 tests |
| CIFAR-100's 20 coarse categories are preserved as 20 Native10 colonies | Demonstrated | derived mapping and adversarial non-contiguous fine-label test |
| The official archive is bundled | False by design | external verified download/prepare path |
| Every persistent floating-point Native10 tensor family is protocol-trainable | Demonstrated | v0.6 reachability report retained and executed |
| Trainer diagnostics authorize promotion | False by design | hidden all-class selection and replication remain required |
| The planner can use the official test split | False by design | explicit rejection and regression test |
| Promotion uses exact paired significance and per-class harm controls | Demonstrated | inherited v0.6 gate, tests, and proofs |
| The package establishes competitive CIFAR-100 accuracy | Not yet demonstrated | requires a completed real campaign and untouched-test report |
| Positive distributed-compute leverage is established | Not yet demonstrated | live campaign must report gain per worker-hour, verifier-hour, bandwidth, and cache hit |
| Distinct public keys prove distinct principals | Not claimed | external Sybil controls remain required |
| The coordinator cannot misbehave | Not claimed | reference deployment remains coordinator-trusted |

Historical proof families remain evidence for transport, locality, heterogeneous resource control, hostile participation, all-class hidden verification, trainable Native10 mechanics, and CIFAR-100 campaign readiness. They do not substitute for a real completed benchmark campaign.
