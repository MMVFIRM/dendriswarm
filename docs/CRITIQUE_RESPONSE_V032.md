# v0.3.2 falsification-response matrix

This document maps the seven thesis-level criticisms of v0.3.1 to executable v0.3.2 changes.

| Critique | Architectural correction | Falsification evidence |
|---|---|---|
| Behavioral binding negated locality by adding a full prediction pass | Root binding is structural: parent lineage root + sole manifest + versioned compose kernel + representation root. Admission returns the exact next-root cache; composition happens only on would-accept. | `test_structural_binding_uses_no_full_prediction_replay`; rejection-before-compose test; proof gates 1–2 |
| Canary was an oracle Boolean | Removed Boolean API. Canary accepts a labeled batch, source id/kind, and optional groups; hashes data; computes parent/candidate paired and subgroup outcomes. | Boolean-removal, duplicate-batch, sleeper-detection tests; proof gate 3 |
| Subgroup harm gate was absent | Added committed minimum subgroup size and maximum challenge subgroup net loss. Labels are default subgroup IDs. | Concentrated victim-class attack test; proof gate 4 |
| Promotion churn burned stale bonds | Stale root checked before challenge admission, fee, or bond. Default stale fee is zero. | Ledger/test-budget equality test; proof gate 5 |
| Fixed challenge was adaptively reused across composed promotions | Added separately committed never-used final replication artifact; compare genesis vs final root once; vest only on pass, otherwise refund/cancel/rollback. | Synthetic overfit-lineage test and successful composed-lineage proof; gates 6–7 and 11 |
| Anti-farming assumed absent Sybil resistance | Permissionless registration grants zero. Explicit `fund_contributor` maps one externally verified principal to one contributor. Claims are conditional. | Duplicate-principal funding test; proof gate 8 |
| Protocol accounting did not test search leverage | Added public-only local-refit search harness and search evidence audit. Report untrusted work, admissions, all trusted costs, final replicated utility, and replicated utility per million trusted distance ops. | Search-interface/test and proof gate 9 |

## Remaining falsifiers

v0.3.2 should still be rejected as a general leverage result if any of the following fail on broader workloads:

- public/untrusted search cannot produce candidates whose final composed gain replicates;
- total trusted cost overwhelms replicated value;
- subgroup definitions miss meaningful harmed populations;
- canary traffic is unrepresentative or controlled by the candidate;
- identity/funding scarcity is not provided operationally;
- route-region locality fails to correspond to useful modularity;
- final epoch replication passes while important unmeasured behavior regresses.

Those are research questions, not claims hidden by the reference implementation.
