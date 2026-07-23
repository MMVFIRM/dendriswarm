#!/usr/bin/env python3
"""Run the v0.3.2 falsification proof and write a machine-readable report."""
from __future__ import annotations

import json
import os
import secrets
import time
from pathlib import Path

import numpy as np

from dendriswarm.leverage.epoch import ChallengeEpoch
from dendriswarm.leverage.manifest import build_manifest
from dendriswarm.leverage.search import search_local_refits
from dendriswarm.leverage.service import LeverageService
from dendriswarm.leverage.stats import GatePolicy
from dendriswarm.leverage.tissue import Territory, build_eval_cache, evaluate_candidate
from dendriswarm.leverage.workload import (
    escape_delta,
    honest_delta,
    make_surrogate_workload,
    sleeper_delta,
    train_parent,
    variance_farm_delta,
)


def submit(service, contributor, replaced, added, territory):
    parent = service.canonical_tissue
    manifest = build_manifest(
        service.canonical_root,
        parent.representation_root(),
        contributor,
        replaced,
        added,
        territory,
    )
    candidate_id, verdict = service.submit(manifest, contributor)
    return manifest, candidate_id, verdict


def fund(service, contributor):
    service.register_contributor(contributor)
    service.fund_contributor(contributor, f"proof-verified-principal:{contributor}")


def stratified_batches(workload, count):
    buckets = [[] for _ in range(count)]
    for label in range(100):
        indices = np.flatnonzero(workload.y_canary == label).tolist()
        for offset, index in enumerate(indices):
            buckets[offset % count].append(index)
    return [(workload.x_canary[idx], workload.y_canary[idx]) for idx in buckets]


def pass_canary(service, candidate_id, workload, prefix):
    results = []
    for index, (x, y) in enumerate(stratified_batches(workload, service.policy.canary_window_events)):
        results.append(service.record_canary_batch(
            candidate_id,
            x,
            y,
            source_id=f"{prefix}-heldout-{index}",
            source_kind="heldout-canary-batch",
        ))
    return results


def build_subgroup_attack(service, workload, categories):
    parent = service.canonical_tissue
    canonical = parent.canonical_categories(service.epoch.x)
    predictions = service.cache.predictions
    eligible = (
        (predictions == service.epoch.y)
        & np.isin(canonical, categories)
        & ~np.isin(service.epoch.y, categories)
    )
    labels, counts = np.unique(service.epoch.y[eligible], return_counts=True)
    victim = int(labels[counts.argmax()])
    points = service.epoch.x[eligible & (service.epoch.y == victim)]
    replaced, added = honest_delta(parent, workload, categories, seed=73)
    for point in points[np.linspace(0, len(points) - 1, min(4, len(points))).astype(int)]:
        for _ in range(parent.top_k):
            added.append((point.copy(), categories[0]))
    return victim, replaced, added


def main() -> None:
    started = time.perf_counter()
    seed_override = os.getenv("DENDRISWARM_PROOF_SEED")
    master = int(seed_override) if seed_override else secrets.randbits(128)
    private_seed = master ^ 0xA11CE
    canary_seed = master ^ 0xCA4A2
    replication_seed = master ^ 0x5EED5
    workload = make_surrogate_workload(
        private_seed=private_seed,
        canary_seed=canary_seed,
        replication_seed=replication_seed,
        private_per_class=120,
        canary_per_class=40,
        replication_per_class=120,
    )
    parent = train_parent(workload)
    policy = GatePolicy()
    epoch = ChallengeEpoch(
        workload.x_private,
        workload.y_private,
        workload.x_replication,
        workload.y_replication,
        policy,
    )
    commitment = epoch.commitment
    service = LeverageService(parent, epoch, policy)
    genesis_root = service.genesis_root
    genesis_predictions = service.cache.predictions.copy()

    for contributor in (
        "sleeper", "searcher-a", "searcher-b", "stale", "subgroup-attacker", "farmer"
    ):
        fund(service, contributor)

    # Concrete sleeper: passes protected admission, fails a labeled canary batch.
    sleeper_categories = tuple(range(16, 24))
    sleeper_replaced, sleeper_added, trigger_x, trigger_y = sleeper_delta(
        parent, workload, sleeper_categories, target_label=16, wrong_owner=17, seed=3
    )
    _, sleeper_id, sleeper_verdict = submit(
        service,
        "sleeper",
        sleeper_replaced,
        sleeper_added,
        Territory(sleeper_categories, 0.25),
    )
    sleeper_root = service.canonical_root
    sleeper_status = service.record_canary_batch(
        sleeper_id,
        trigger_x,
        trigger_y,
        source_id="audited-trigger-traffic-001",
        source_kind="audited-inference-batch",
    )
    sleeper_observation = service.candidates[sleeper_id].canary_observations[-1]
    rollback_root_ok = service.canonical_root == genesis_root
    rollback_behavior_ok = np.array_equal(service.cache.predictions, genesis_predictions)

    # Build stale work before another contributor promotes.
    stale_delta = honest_delta(parent, workload, tuple(range(40, 48)), seed=61)
    stale_manifest = build_manifest(
        genesis_root,
        parent.representation_root(),
        "stale",
        *stale_delta,
        Territory(tuple(range(40, 48)), 0.25),
    )

    # Actual public-data-only search, then trusted admission and concrete canaries.
    search_rounds = (
        ("searcher-a", (tuple(range(8)), tuple(range(8, 16)))),
        ("searcher-b", (tuple(range(16, 24)), tuple(range(24, 32)))),
    )
    search_reports = []
    search_candidates = []
    accepted_search = []
    for round_index, (contributor, blocks) in enumerate(search_rounds):
        report = search_local_refits(
            service.canonical_tissue,
            workload.x_train,
            workload.y_train,
            workload.x_public,
            workload.y_public,
            blocks,
            seeds=(11, 29, 47),
            iterations=20,
        )
        selected = report.candidates[0]
        manifest, candidate_id, verdict = submit(
            service,
            contributor,
            selected.replaced,
            selected.added,
            Territory(selected.categories, 0.25),
        )
        statuses = []
        if verdict == "accepted:promotion-candidate":
            statuses = pass_canary(service, candidate_id, workload, contributor)
            accepted_search.append(candidate_id)
        search_reports.append(report.as_dict())
        search_candidates.append({
            "round": round_index,
            "candidate": candidate_id,
            "manifest": manifest.commitment,
            "public_net_wins": selected.public_net_wins,
            "verdict": verdict,
            "canary_statuses": statuses,
            "challenge_paired": service.candidates[candidate_id].detail.get("paired"),
        })


    if not accepted_search:
        raise RuntimeError("proof search did not produce a promoted root for stale-parent testing")
    stale_before = service.ledger.snapshot()
    stale_tests_before = service.epoch.tests_spent
    _, stale_verdict = service.submit(stale_manifest, "stale")
    stale_after = service.ledger.snapshot()
    stale_tests_after = service.epoch.tests_spent

    # Semantic subgroup attack: positive aggregate result, concentrated victim loss.
    harm_categories = tuple(range(32, 40))
    victim, harm_replaced, harm_added = build_subgroup_attack(service, workload, harm_categories)
    _, harm_id, harm_verdict = submit(
        service,
        "subgroup-attacker",
        harm_replaced,
        harm_added,
        Territory(harm_categories, 0.25),
    )
    harm_detail = service.candidates[harm_id].detail

    # Anti-farming evidence is explicitly conditional on one verified grant.
    farm_categories = tuple(range(80, 88))
    farm_attempts = 0
    farm_accepted = 0
    farmer_out_of_funds = False
    for seed in range(20):
        current = service.canonical_tissue
        replaced, added = variance_farm_delta(current, farm_categories, seed=1000 + seed)
        try:
            _, candidate_id, verdict = submit(
                service,
                "farmer",
                replaced,
                added,
                Territory(farm_categories, 0.25),
            )
        except ValueError as error:
            farmer_out_of_funds = "insufficient available balance" in str(error)
            break
        farm_attempts += 1
        if verdict == "accepted:promotion-candidate":
            farm_accepted += 1
            # Do not inject an oracle verdict. Evaluate the independent heldout data.
            statuses = pass_canary(service, candidate_id, workload, f"farmer-{seed}")
            if statuses[-1] == "canary-passed":
                # An accepted farmer becomes part of the lineage and must survive
                # final replication like every other candidate.
                pass

    # Territory escape remains inert outside declared frozen route regions.
    canonical = service.canonical_tissue
    cache = build_eval_cache(canonical, workload.x_private)
    _, intruders = escape_delta(canonical, workload, (0,), target=50)
    guarded = evaluate_candidate(canonical, cache, {}, intruders, Territory((0,), 0.05))
    unguarded = evaluate_candidate(canonical, cache, {}, intruders, Territory(tuple(range(100)), 1.0))

    search_evidence = {
        "format": "dendriswarm.untrusted-search-evidence.v3.2",
        "protected_split_accessed": False,
        "candidates_generated": sum(item["candidates_generated"] for item in search_reports),
        "candidates_submitted": len(search_candidates),
        "candidates_admitted": len(accepted_search),
        "total_untrusted_distance_ops": sum(item["total_untrusted_distance_ops"] for item in search_reports),
        "rounds": search_candidates,
    }
    service.record_search_evidence(search_evidence)

    replication = service.close_epoch()
    disclosure = service.reveal_epoch()
    disclosure_verified = ChallengeEpoch.verify_disclosure(disclosure)
    metrics = service.metrics()
    locality = metrics["verification_locality"]
    leverage = metrics["replicated_leverage"]

    structural_records = [
        service.candidates[candidate_id]
        for candidate_id in accepted_search
    ]
    rewards_vested = all(record.status == "vested" for record in structural_records)
    search_yield = len(accepted_search) / max(1, search_evidence["candidates_generated"])

    gates = {
        "1_structural_binding_has_zero_full_behavioral_replays": {
            "pass": bool(structural_records)
            and all(record.detail.get("artifact_binding_verified") for record in structural_records)
            and locality["behavioral_binding_full_passes"] == 0,
            "evidence": {
                "accepted_search_candidates": len(structural_records),
                "behavioral_binding_full_passes": locality["behavioral_binding_full_passes"],
                "binding_method": structural_records[0].detail.get("binding_method") if structural_records else None,
            },
        },
        "2_candidate_admission_cost_is_below_full_distance_recompute": {
            "pass": locality["mean_candidate_distance_ratio"] < 1.0,
            "evidence": locality,
        },
        "3_canary_uses_observable_labeled_batch_and_detects_sleeper": {
            "pass": sleeper_verdict == "accepted:promotion-candidate"
            and sleeper_status == "revoked"
            and sleeper_observation["paired"]["losses"] == len(trigger_y)
            and bool(sleeper_observation["data_hash"]),
            "evidence": sleeper_observation,
        },
        "4_semantic_subgroup_harm_gate_rejects_concentrated_loss": {
            "pass": harm_verdict == "rejected:subgroup-harm"
            and harm_detail["paired"]["net_wins"] > 0
            and harm_detail["worst_subgroup"]["net_wins"] < -policy.max_challenge_subgroup_net_loss,
            "evidence": {
                "victim_label": victim,
                "verdict": harm_verdict,
                "aggregate": harm_detail.get("paired"),
                "worst_subgroup": harm_detail.get("worst_subgroup"),
            },
        },
        "5_stale_work_never_loses_bond_or_challenge_budget": {
            "pass": stale_verdict == "rejected:stale-parent"
            and stale_tests_after == stale_tests_before
            and stale_after == stale_before,
            "evidence": {
                "verdict": stale_verdict,
                "tests_before": stale_tests_before,
                "tests_after": stale_tests_after,
                "ledger_unchanged": stale_after == stale_before,
            },
        },
        "6_fresh_final_holdout_validates_composed_lineage": {
            "pass": replication["passed"]
            and replication["paired"]["net_wins"] >= policy.replication_min_net_wins
            and replication["worst_subgroup"]["net_wins"] >= -policy.replication_max_subgroup_net_loss,
            "evidence": replication,
        },
        "7_rewards_remain_locked_until_final_replication": {
            "pass": rewards_vested
            and all(service.ledger.account(record.manifest.contributor).locked == 0 for record in structural_records),
            "evidence": {
                "settled_candidates": replication["settled_candidates"],
                "candidate_statuses": {record.manifest.contributor: record.status for record in structural_records},
            },
        },
        "8_anti_farming_is_conditional_on_verified_identity_funding": {
            "pass": farmer_out_of_funds
            and metrics["identity_economics"]["permissionless_registration_grant_units"] == 0
            and metrics["identity_economics"]["anti_farming_claim_is_conditional"],
            "evidence": {
                "identity_economics": metrics["identity_economics"],
                "farmer_attempts": farm_attempts,
                "farmer_accepted": farm_accepted,
                "farmer_out_of_funds": farmer_out_of_funds,
            },
        },
        "9_public_untrusted_search_produces_replicating_candidates": {
            "pass": not search_evidence["protected_split_accessed"]
            and search_evidence["candidates_generated"] > 0
            and search_evidence["candidates_admitted"] > 0
            and replication["passed"]
            and leverage["replicated_net_wins"] > 0,
            "evidence": {
                **search_evidence,
                "admission_yield": search_yield,
                "replicated_leverage": leverage,
            },
        },
        "10_runtime_territory_guard_is_inert_outside_declared_regions": {
            "pass": guarded.outside_discordant == 0
            and int((unguarded.predictions != cache.predictions).sum()) > 0,
            "evidence": {
                "guarded_outside_discordant": guarded.outside_discordant,
                "unguarded_changed_predictions": int((unguarded.predictions != cache.predictions).sum()),
            },
        },
        "11_commit_reveal_discloses_both_admission_and_replication_splits": {
            "pass": disclosure_verified
            and disclosure["commitment"] == commitment
            and disclosure["replication_result"]["passed"],
            "evidence": {
                "commitment": commitment,
                "verified": disclosure_verified,
                "challenge_hash": disclosure["challenge_hash"],
                "replication_hash": disclosure["replication_hash"],
            },
        },
        "12_revocation_restores_parent_root_and_behavior": {
            "pass": rollback_root_ok and rollback_behavior_ok,
            "evidence": {
                "sleeper_root": sleeper_root,
                "restored_root": genesis_root,
                "root_restored": rollback_root_ok,
                "behavior_restored": rollback_behavior_ok,
            },
        },
    }

    report = {
        "format": "dendriswarm.proof-run.v3.2",
        "workload_disclosure": (
            "synthetic 100-class multimodal Gaussian protocol workload; NOT CIFAR-100, "
            "NOT Native8; search sees training/public data only"
        ),
        "seed_disclosure_after_run": {
            "master": master,
            "private_seed": private_seed,
            "canary_seed": canary_seed,
            "replication_seed": replication_seed,
        },
        "epoch": {
            "commitment": commitment,
            "disclosure": disclosure,
            "submissions_charged": service.epoch.tests_spent,
            "stale_submissions": service.epoch.stale_submissions,
        },
        "candidates": {
            "sleeper": {
                "verdict": sleeper_verdict,
                "final": sleeper_status,
                "observation": sleeper_observation,
            },
            "search": search_candidates,
            "subgroup_attack": {
                "verdict": harm_verdict,
                "victim": victim,
                "paired": harm_detail.get("paired"),
                "worst_subgroup": harm_detail.get("worst_subgroup"),
            },
            "variance_farmer": {
                "identity_condition": policy.identity_assumption,
                "attempts": farm_attempts,
                "accepted": farm_accepted,
                "out_of_funds": farmer_out_of_funds,
            },
        },
        "gates": gates,
        "all_gates_pass": all(gate["pass"] for gate in gates.values()),
        "metrics": metrics,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }

    output = Path(__file__).resolve().parents[1] / "docs" / "PROOF_RUN_V03.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True))

    print(f"untrusted search generated   {search_evidence['candidates_generated']} candidates")
    print(f"trusted admissions           {len(accepted_search)}")
    print(f"candidate distance ratio     {locality['mean_candidate_distance_ratio']:.4f}")
    print(f"behavioral binding passes    {locality['behavioral_binding_full_passes']}")
    print(f"sleeper canary               {sleeper_status}, losses={sleeper_observation['paired']['losses']}")
    print(f"subgroup attack              {harm_verdict}, worst={harm_detail['worst_subgroup']['net_wins']}")
    print(f"final replication            pass={replication['passed']} net_wins={replication['paired']['net_wins']}")
    print(f"replicated utility/1M ops    {leverage['replicated_net_wins_per_million_total_trusted_distance_ops']:.6f}")
    print(f"ALL TWELVE GATES PASS        {report['all_gates_pass']}")
    print(f"report written               {output}")
    if not report["all_gates_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
