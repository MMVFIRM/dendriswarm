# DendriSwarm v0.8.0 security boundary

DendriSwarm is a public research network. Never distribute secrets, private prompts, regulated records, proprietary raw datasets, or unredacted personal data to volunteer nodes.

## Mutation boundary

Volunteer trainers receive built-in mutation code, one bounded work bundle, and a bounded trainer-visible shard. They cannot upload executable code or replace the complete checkpoint.

Every sparse delta is bound to the parent root, bundle hash, operation-schema hash, target, exact tensor selectors, write set, changed-element count, and content hash. The coordinator validates operation-specific tensor ownership, selector territory, shapes, dtypes, finite values, magnitude limits, and changed-element ceilings before evidence generation and again before composition.

This prevents one operation from silently writing another operation's tissue. It does not prevent colluding identities from proposing a malicious but schema-valid candidate.

## Hidden evidence boundary

Selection and final replication use different coordinator-held all-class artifacts. Trainers receive only their hashes. Retrieval requires a signed request from the identity holding the matching active verification lease.

The coordinator recomputes consistency of every submitted evidence field: sample/class counts, aggregate/class correctness, class losses/rates, paired wins/losses, effect rate, exact McNemar probability, corrected alpha, significance flags, operation, target, root, delta, validation hash, and write set.

Lease gating controls issuance, not retention. A verifier can copy or leak an artifact after receiving it. A malicious coordinator can also promote arbitrary state. Production use requires independently audited governance, artifact rotation, and principal-level anti-collusion controls.

## Statistical boundary

The committed policy includes a maximum candidate count for one precommitted tournament. Selection uses an exact one-sided McNemar test with Bonferroni correction, a minimum discordant count, a minimum practical effect, and class-specific harm ceilings. The selection artifact is exhausted after that tournament. The chosen candidate must reproduce once on a distinct one-shot artifact; failure closes the round.

These controls bound the packaged adaptive-selection protocol. They do not prove validity under unlimited unrecorded evaluations, coordinator data leakage, or verifier collusion.

## Data boundary

Training and validation artifacts can leak source information. v0.6 does not claim differential privacy, secure aggregation, trusted execution, or confidential computation. Use only data that may legally and ethically be processed by the assigned workers.

## Resource and transport boundary

- Remote coordinators require TLS by default.
- Seeds support out-of-band coordinator fingerprint pinning.
- One active lease per identity is enforced by SQLite transaction and partial unique index.
- Lease renewal has an absolute deadline.
- Requests, artifacts, JSON trees, outputs, cache writes, RSS, and wall time are bounded.
- Work runs in a killable subprocess.
- Results persist in a durable local outbox until a signed receipt arrives.
- Resource requirements are independently derived from materialized payloads.

These remain application-level controls rather than a formally verified kernel sandbox. Run seeds unprivileged and use containers, cgroups, Job Objects, VMs, or equivalent isolation where appropriate.

## Checkpoint and baseline import

Checkpoint conversion trusts the operator's mapping and then validates tensor names, shapes, dtypes, finite values, configuration, and resulting root. Publish conversion inputs and output roots independently.

External baseline references are informational, content-addressed records. Their `evidence_sha256` identifies an external report but does not prove that report is truthful. Operators and reviewers must inspect the referenced evidence.

## Explicit non-claims

v0.6.0 does not claim:

- production Sybil resistance;
- safety against colluding principals or a malicious coordinator;
- holdout secrecy after authorized issuance;
- historical checkpoint equivalence;
- benchmark accuracy from synthetic proof fixtures;
- live multi-round conflict-aware scheduling;
- positive public-scale economics before real-network measurement.

## CIFAR-100 campaign boundary — v0.7.0

- The official archive is accepted only from a local file whose MD5 matches the published CIFAR-100 Python archive, or from an explicitly trusted extracted directory.
- Tar members are rejected if they are links, devices, path traversals, or exceed the expansion ceiling.
- CIFAR pickle loading uses a restricted global whitelist; arbitrary pickle imports are rejected.
- Prepared `.npy` arrays are content-addressed in the manifest and their encoded files are verified before first use in each process.
- Trainers never receive selection, replication, or official-test rows.
- The planner accepts routing diagnostics only from the campaign training split and rejects test-tagged reports.
- Selection and replication rows are consumed once according to the committed fold plan. Evidence-bank exhaustion stops the campaign.
- The official test split is coordinator-readable because the coordinator produces the final report; this does not protect volunteers from a malicious coordinator.
- Dataset privacy is not provided. CIFAR-100 is public data.
- A malicious coordinator can still fabricate tasks, withhold results, leak hidden evidence, or publish an unauthorized root. The reference design protects the coordinator from hostile volunteers more strongly than it protects volunteers from the coordinator.

## v0.8 local dashboard boundary

The browser dashboard is intentionally not a remote administration service. It refuses non-loopback bind addresses. First launch creates a random token under the dashboard state directory; the launch URL exchanges it for a `SameSite=Strict`, HTTP-only cookie. Dashboard API calls without the token are rejected.

Coordinator-mutating campaign routes require a separate random admin token stored under the coordinator's local `keys/` directory. The public `/v1/meta` response never includes it. Anyone with read access to either state directory can control the corresponding local process, so normal operating-system account and filesystem protections remain required.

Managed process records bind a PID to its process creation time to reduce PID-reuse errors. Dashboard logs are bounded tails of local stdout/stderr and may contain operational filenames or errors; do not place secrets in command-line arguments or dataset paths.
