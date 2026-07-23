# v0.3.2 claims and evidence boundary

| Claim | Status | Evidence / boundary |
|---|---|---|
| Candidate artifacts are bound to the manifest used by evaluation | **Verified** | Manifest is the sole artifact source; every embedded branch is rehashed and dimension/owner checked; adversarial omission/tampering tests |
| Structural binding requires a second full candidate prediction pass | **False by design** | Versioned deterministic composition certificate; exact incremental next-root cache; proof gate 1 reports zero behavioral binding passes |
| Rejected candidates incur full behavioral binding cost | **False by design** | Composition occurs only after all admission gates; no behavioral binding pass exists; rejection-path test |
| Promotions compose from the current canonical lineage root | **Verified** | Parent-root validation, `ModelStore` lineage, deterministic composition, sequential-promotion and persistence tests |
| Revocation restores the exact parent root and behavior | **Verified** | Materialized parent reload/cache restoration; sleeper rollback test; proof gate 12 |
| Candidate distance calculations are proportional to touched branches | **Verified for the reference evaluator** | Incremental distance counter and proof gate 2 |
| Total trusted candidate verification is always faster than full evaluation | **Not claimed** | Full-width copying/selection/aggregation and wall time are reported; hardware and workload can erase distance savings |
| v0.3.2 avoids the prior incremental-plus-full mandatory anti-leverage | **Verified** | Zero full behavioral replay and exact cache carry-forward |
| Canary cleanliness is a caller-supplied Boolean | **False by design** | Boolean API raises; concrete labeled batch is hashed and evaluated by the service |
| The reference canary can detect the constructed delayed trigger | **Verified on the disclosed synthetic attack** | 120 parent-correct/candidate-wrong audited examples; proof gate 3 |
| Candidate contributors can submit their own canary verdicts through the reference HTTP adapter | **Prevented** | Authorized-auditor allowlist plus self-audit rejection; integration test |
| Canary data governance is trustless | **Not claimed** | A compromised coordinator or authorized observer can select unrepresentative evidence; source/data hashes provide auditability, not Byzantine truth |
| Positive aggregate gain can freely concentrate losses in one semantic class | **False under the committed threshold** | Admission label-subgroup gate; concentrated victim-class attack; proof gate 4 |
| The subgroup gate detects every in-territory backdoor or unknown subgroup | **Not claimed** | It covers represented supplied subgroup identifiers, semantic labels by default, and committed loss thresholds |
| Stale work burns a bond or consumes protected challenge budget | **False by design** | Staleness checked before accounting/evaluation; default stale fee zero; proof gate 5 |
| Repeated use of one admission challenge is sufficient evidence of composed gain | **Not claimed** | Fresh separately committed genesis-vs-final replication is mandatory before vesting |
| Final composed gain replicates on a never-reused holdout in the packaged proof | **Verified for the synthetic reference run** | Proof gate 6; full post-close disclosure and verifier |
| Rewards vest after canary alone | **False by design** | Bond/reward remain locked until final replication; proof gate 7 |
| Permissionless registration grants 8,000 units | **False by design** | Registration grant is zero |
| Anti-farming economics are Sybil-resistant without assumptions | **Not claimed** | Result is conditional on an externally verified one-grant-per-principal or equivalent scarce funding mechanism |
| One verified principal can claim multiple reference grants | **Prevented in reference service** | Bidirectional principal/contributor mapping and idempotent grant key; tests |
| The proof inserts a hand-scripted honest candidate as its leverage numerator | **False for the v0.3.2 search gate** | Public-only search generates/ranks candidates; proof records generated, submitted, admitted, and replicated outcomes |
| The public-only search establishes production economic leverage | **Not claimed** | It establishes one disclosed synthetic search-to-replication result, not Native8/CIFAR-100 or market economics |
| Admission, canary, and replication splits are independent and committed | **Verified** | Independent seeds/artifacts; epoch commitment; proof gate 11 |
| A compromised coordinator cannot leak protected splits | **Not claimed** | Commit-reveal detects substitution, not exfiltration |
| Declared route regions equal true semantic classes | **Not claimed** | Regions derive from inherited nearest anchors; labels are handled separately by subgroup analysis |
| Influence is zero outside declared inherited route regions | **Verified in reference evaluator** | Guard masks, outside-discordance checks, escape test, proof gate 10 |
| Duplicate submissions or canary data can be counted repeatedly | **Prevented in reference service** | Idempotent candidate/ledger keys and duplicate observation-data hash checks |
| Reference leverage state survives restart | **Verified** | v3.2 state snapshot round-trip, model/audit validation, tests |
| Sybil resistance, Byzantine consensus, private public-worker inference, arbitrary-code sandboxing, Native8/CIFAR-100 capability | **Not claimed** | Explicit scope exclusions |
