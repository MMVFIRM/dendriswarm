# DendriSwarm v0.8.0 release notes

## Headline

v0.8.0 packages the real CIFAR-100 swarm campaign as a one-command local application. Running `dendriswarm` opens a contributor and training dashboard while preserving the complete CLI and the lightweight CPU-only seed installation.

## Added

- Loopback-only local dashboard requiring no frontend framework or Node.js runtime.
- First-run persistent configuration for contributor, coordinator, and campaign settings.
- Live CPU, memory, disk, battery, and task-duration controls backed by `SeedPolicy`.
- Start, pause, resume, and stop controls for the contributor process.
- Optional local coordinator process supervision.
- Authenticated campaign controls for CIFAR-100 preparation, checkpoint initialization/import, planning, queueing, and final evaluation.
- Canonical-root, routing-gap, top-k recall, campaign-round, worker-hour, task, credit, and log telemetry.
- Atomic dashboard configuration and PID creation-time validation.
- Random loopback dashboard token and separate coordinator admin token.
- Windows and macOS/Linux contributor/operator launchers.
- `dendriswarm-dashboard` console entry point.
- Dedicated dashboard tests and v0.8 proof.

## Preserved

The v0.7 CIFAR-100 campaign, v0.6 trainable Native10 operations, trainer-invisible global evidence, hostile-worker controls, heterogeneous resource enforcement, and all historical proofs remain intact.

## Explicit boundaries

The dashboard is a local control surface, not a remote multi-tenant administration service. It binds only to loopback. The coordinator remains trusted, principal independence remains external, and a completed real CIFAR-100 campaign is still required for a new accuracy or distributed-leverage claim.
