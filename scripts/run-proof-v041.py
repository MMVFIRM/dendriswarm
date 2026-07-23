#!/usr/bin/env python3
"""Reproduce the v0.4.1 hostile-public-participation proof."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

GROUPS: list[tuple[str, list[str], str]] = [
    (
        "route-and-composition-integrity",
        [
            "tests/test_adversarial_v041.py::test_route_share_counts_removed_inherited_branch_influence",
            "tests/test_adversarial_v041.py::test_model_store_rejects_caller_supplied_non_composition",
        ],
        "Any inherited-route removal/masking effect counts toward route share; stored roots are independently recomposed.",
    ),
    (
        "queue-hoarding-and-lease-defense",
        [
            "tests/test_adversarial_v041.py::test_one_node_cannot_hoard_or_reclaim_expired_task",
            "tests/test_adversarial_v041.py::test_lease_renewal_never_exceeds_absolute_deadline",
            "tests/test_adversarial_v041.py::test_one_identity_cannot_supply_multiple_replicas_of_same_work",
        ],
        "One active lease per identity, expiry exclusion/quarantine, absolute deadlines, and identity-distinct replicas.",
    ),
    (
        "atomic-leverage-state-machine",
        [
            "tests/test_adversarial_v041.py::test_leverage_submit_is_serialized_and_persistence_remains_valid",
        ],
        "Submit, canary, rollback, close, and persistence mutations share one rollback-capable lock boundary.",
    ),
    (
        "volunteer-compute-not-coordinator-replay",
        [
            "tests/test_integration.py::test_coordinator_promotes_without_retraining_worker_contributions",
        ],
        "Exploration/training promotion succeeds while coordinator training is replaced by an exception; independent replicas and verifiers carry the evidence.",
    ),
    (
        "payload-bound-resource-enforcement",
        [
            "tests/test_adversarial_v041.py::test_resource_contract_is_derived_from_materialized_payload",
            "tests/test_adversarial_v041.py::test_declared_artifact_ceiling_must_fit_local_control_plane_memory",
            "tests/test_adversarial_v041.py::test_budgeted_dataset_artifact_contains_only_assigned_rows",
            "tests/test_adversarial_v041.py::test_zero_available_memory_is_never_treated_as_unknown",
            "tests/test_adversarial_v041.py::test_active_task_stops_after_hot_reloaded_pause",
        ],
        "Workers derive requirements locally, bound downloads before parsing, use compact shards, enforce zero-memory truthfully, and kill active work after policy changes.",
    ),
    (
        "bounded-wire-and-malformed-result-defense",
        [
            "tests/test_adversarial_v041.py::test_request_body_limit_runs_before_endpoint_parsing",
            "tests/test_adversarial_v041.py::test_worker_rejects_invalid_or_oversized_declared_response_length",
            "tests/test_adversarial_v041.py::test_malformed_exploration_result_is_committed_as_worker_failure",
        ],
        "Request/response bytes are capped before endpoint work; malformed stage evidence is excluded, quarantined, and auditable.",
    ),
    (
        "canary-and-disclosure-authenticity",
        [
            "tests/test_adversarial_v041.py::test_canary_rejects_invalid_and_noninformative_labels",
            "tests/test_adversarial_v041.py::test_disclosure_policy_body_is_cryptographically_bound",
            "tests/test_integration.py::test_canary_http_requires_an_authorized_non_contributor_auditor",
        ],
        "Canary labels follow the committed class schema, non-informative batches fail, policy disclosure is hash-bound, and self-audit is forbidden.",
    ),
    (
        "inference-proof-economics-and-refunds",
        [
            "tests/test_security.py::test_audited_wrong_inference_is_rejected_and_worker_penalized",
            "tests/test_adversarial_v041.py::test_terminal_inference_failure_refunds_requester_once",
        ],
        "Strict proof-carrying outputs, unpredictable audits, a 4,000-unit claim bond, requeue-on-fraud, and exactly-once terminal requester refunds.",
    ),
    (
        "heterogeneous-scheduler-liveness",
        [
            "tests/test_adversarial_v041.py::test_scheduler_scans_past_256_incompatible_tasks",
            "tests/test_adversarial_v041.py::test_cross_architecture_artifact_consensus_tolerates_harmless_rounding",
            "tests/test_adversarial_v041.py::test_share_flag_has_exact_cpu_and_memory_semantics",
        ],
        "Compatibility scanning cannot hide eligible work; architecture aliases and bounded numeric consensus are portable; share percentages are exact.",
    ),
    (
        "durable-delivery-expiry-and-trust-bootstrap",
        [
            "tests/test_adversarial_v041.py::test_result_outbox_survives_network_failure",
            "tests/test_adversarial_v041.py::test_expired_signed_task_is_rejected_before_materialization",
            "tests/test_adversarial_v041.py::test_remote_http_requires_opt_in_and_fingerprint_mismatch_fails",
        ],
        "Results survive network failure, expired envelopes do no work, remote coordinators require TLS or explicit override, and out-of-band fingerprints are supported.",
    ),
    (
        "atomic-identity-creation",
        ["tests/test_adversarial_v041.py::test_identity_creation_race_converges_on_one_key"],
        "Concurrent first-run identity creation converges on one exclusively created Ed25519 key.",
    ),
]


def main() -> None:
    started = time.perf_counter()
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(ROOT / "src")
    selected_tests = list(dict.fromkeys(test for _, tests, _ in GROUPS for test in tests))
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", *selected_tests],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=240,
    )
    suite_passed = result.returncode == 0
    gates: list[dict[str, object]] = [
        {
            "name": name,
            "pass": suite_passed,
            "tests": tests,
            "claim": claim,
        }
        for name, tests, claim in GROUPS
    ]
    report = {
        "format": "dendriswarm.hostile-participation-proof.v1",
        "version": "0.4.1",
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "gates": gates,
        "all_gates_pass": suite_passed,
        "test_nodes_executed": len(selected_tests),
        "pytest_stdout": result.stdout.strip()[-4000:],
        "pytest_stderr": result.stderr.strip()[-4000:],
        "verification_modes": {
            "exploration": "identity-distinct replicated consensus",
            "training": "identity-distinct semantic-artifact consensus",
            "verification": "independent identity-distinct replicated consensus",
            "inference": "strict proof-carrying output plus secret bonded spot audit",
        },
        "claim_boundary": (
            "Demonstrates the reference implementation's defenses against the enumerated hostile-public-participation failures. "
            "It does not establish production Sybil resistance, Byzantine consensus across colluding principals, confidential public-worker inference, "
            "or formal OS/kernel isolation beyond the built-in subprocess resource boundary."
        ),
    }
    output = ROOT / "docs" / "PROOF_RUN_V041.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not suite_passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
