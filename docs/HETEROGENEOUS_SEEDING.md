# Heterogeneous volunteer seeding

## Resource descriptions

A seed signs observed `NodeCapabilities` separately from the owner's `SeedPolicy`. Hardware availability never implies permission to consume it.

`TaskRequirements` includes resource class, minimum/preferred threads, minimum and maximum memory, disk, estimated and hard runtime, total artifact bytes, backend, tags, machine restrictions, and checkpointability.

## Three enforcement phases

1. **Scheduler admission:** compare signed capabilities/policy with the signed task contract.
2. **Pre-download local admission:** repeat matching, reject expired envelopes, and require declared artifact bytes to fit local disk and a conservative control-plane memory ceiling.
3. **Post-materialization derivation:** derive requirements from actual dataset/artifact/config dimensions and reject any understated contract.

Built-in computation then runs in a spawned subprocess. The parent polls child RSS, hard elapsed time, and the hot-reloaded policy. Incompatible changes terminate the child and abandon/requeue the lease.

## Bounded datasets and downloads

Budgeted scouting receives a new deterministic compact artifact containing only assigned rows and remapped splits. It does not download, parse, or convert the full source dataset.

Responses are streamed with declared and running byte limits. JSON structure has depth/node/key limits. Cached objects are size-checked before parsing and content-hash verified afterward.

## Live controls

`--share N` means exactly `N%` for both CPU and memory. Explicit CPU or memory flags override the combined value. A sub-one-core CPU share uses one native thread plus duty-cycle cooldown.

```bash
dendriswarm seed-config --share 15
dendriswarm seed-config --task-types exploration,verification
dendriswarm seed-pause
dendriswarm seed-resume
```

Active work is no longer allowed to run indefinitely after a policy change. The enforcement process terminates work when pause, task eligibility, thread allowance, memory allowance, or duration policy becomes incompatible.

## Architecture portability

Matching normalizes common aliases (`amd64`→`x86_64`, `arm64`→`aarch64`). Reference work is portable `numpy-cpu`; GPUs are optional metadata only. Semantic artifact consensus quantizes below a committed tolerance, and independent verification evaluates the exact selected artifact.

## Honest limits

- CPU share is an average application-level envelope, not an instantaneous kernel quota.
- Child RSS enforcement does not formally sandbox native dependencies or the parent control process.
- Calibration is a scheduling hint.
- Distinct node keys are not proof of distinct real-world principals.
- Actual multi-OS/architecture support is evidence only after CI or hardware runs execute.
