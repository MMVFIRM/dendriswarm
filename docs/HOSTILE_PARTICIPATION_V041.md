# v0.4.1 hostile-participation repair map

| Review finding | v0.4.1 correction | Primary evidence |
|---|---|---|
| Route-share bypass through inherited branch removal | Compare parent and candidate route identities plus behavior for every sample | `test_route_share_counts_removed_inherited_branch_influence` |
| Queue hoarding/reclaim and unlimited renewal | One active lease, exclusion/quarantine, absolute deadline | lease adversarial tests |
| Concurrent leverage promotion race | One re-entrant rollback-capable mutation boundary and unique atomic persistence temp files | concurrent submit/persistence test |
| Coordinator repeats volunteer compute | Identity-distinct replicated exploration/training consensus and independent verification; coordinator training is not called | no-retraining integration test |
| Resource declaration not bound to payload | Local payload-derived requirements and spawned hard runtime/RSS enforcement | resource-contract and live-cancel tests |
| Artifact buffered without effective ceiling | Streaming byte counter, `Content-Length`, local memory/disk-derived ceiling, JSON complexity limits | response-limit and artifact-budget tests |
| Budgeted work materializes full dataset | Deterministic compact dataset shard containing only selected rows | compact-shard test |
| Zero available memory ignored | Zero is always applied as the current memory ceiling | zero-memory test |
| Active controls ineffective | Parent enforcement loop kills and abandons active work after incompatible hot reload | active-pause test |
| Impossible/non-informative canary labels | Model class-range validation and minimum informative/discordant predicates | canary test |
| Policy disclosure unauthenticated | Reconstruct `GatePolicy`, recompute registration hash, then verify commitment | disclosure test |
| Weak inference schema/economics/refund | Exact schema, consistency checks, secret audit, claim bond, requeue, exactly-once refund | inference security tests |
| Unbounded coordinator bodies/results | ASGI body cap before Pydantic, strict models, JSON tree and stored-result limits | body/result tests |
| Eligible task hidden after 256 | Continue paged scan until eligible work or queue exhaustion | scheduler test |
| Result delivery not durable | Atomic outbox and receipt persistence | outbox test |
| Expired signed task executes | Check lease expiry/deadline before materialization | expired-envelope test |
| First-use coordinator substitution | Remote TLS requirement and optional out-of-band fingerprint | trust-bootstrap test |
| Lower-level identity/share/architecture/composition issues | O_EXCL identity creation, exact share semantics, alias normalization, semantic consensus, independent recomposition | targeted tests |

The full proof artifact is `docs/PROOF_RUN_V041.json`. The remaining boundary is explicit: this is not production Sybil resistance or Byzantine consensus against colluding externally funded principals.
