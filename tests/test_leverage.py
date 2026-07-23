import copy

import numpy as np
import pytest

from dendriswarm.leverage.epoch import ChallengeEpoch
from dendriswarm.leverage.manifest import CandidateManifest, build_manifest
from dendriswarm.leverage.search import search_local_refits
from dendriswarm.leverage.service import LeverageService
from dendriswarm.leverage.stats import GatePolicy, mcnemar_exact_one_sided
from dendriswarm.leverage.tissue import Territory, TerritoryTissue, build_eval_cache, evaluate_candidate
from dendriswarm.leverage.workload import (
    escape_delta,
    honest_delta,
    make_surrogate_workload,
    sleeper_delta,
    train_parent,
    variance_farm_delta,
)


@pytest.fixture(scope="module")
def world():
    workload = make_surrogate_workload(
        private_seed=42,
        canary_seed=43,
        replication_seed=44,
    )
    return workload, train_parent(workload)


def make_service(workload, parent, policy=None, *, state_path=None):
    policy = policy or GatePolicy()
    epoch = ChallengeEpoch(
        workload.x_private,
        workload.y_private,
        workload.x_replication,
        workload.y_replication,
        policy,
        salt="fixture-salt",
    )
    return LeverageService(parent, epoch, policy, state_path=state_path)


def fund(service, contributor, principal=None, units=None):
    service.register_contributor(contributor)
    service.fund_contributor(contributor, principal or f"principal:{contributor}", units)


def make_manifest(service, contributor, replaced, added, territory):
    parent = service.canonical_tissue
    return build_manifest(
        service.canonical_root,
        parent.representation_root(),
        contributor,
        replaced,
        added,
        territory,
    )


def submit_delta(service, contributor, replaced, added, territory):
    manifest = make_manifest(service, contributor, replaced, added, territory)
    return manifest, service.submit(manifest, contributor)


def stratified_canary_batches(workload, count=2):
    batches = [[] for _ in range(count)]
    for label in range(100):
        indices = np.flatnonzero(workload.y_canary == label).tolist()
        for offset, index in enumerate(indices):
            batches[offset % count].append(index)
    return [
        (workload.x_canary[index], workload.y_canary[index])
        for index in batches
    ]


def pass_canary(service, candidate_id, workload):
    result = None
    for index, (x, y) in enumerate(stratified_canary_batches(workload, service.policy.canary_window_events)):
        result = service.record_canary_batch(
            candidate_id,
            x,
            y,
            source_id=f"heldout-{candidate_id[:8]}-{index}",
            source_kind="heldout-canary-batch",
        )
    return result


# ---------------------------------------------------------------- statistics and policy

def test_mcnemar_exact_values():
    assert mcnemar_exact_one_sided(0, 0) == 1.0
    assert mcnemar_exact_one_sided(1, 0) == 0.5
    assert abs(mcnemar_exact_one_sided(9, 1) - 11 / 1024) < 1e-12
    with pytest.raises(ValueError):
        mcnemar_exact_one_sided(-1, 0)


def test_policy_hash_binds_subgroup_canary_replication_and_identity_fields():
    base = GatePolicy()
    variants = (
        GatePolicy(canary_window_events=3),
        GatePolicy(max_challenge_subgroup_net_loss=9),
        GatePolicy(replication_min_net_wins=26),
        GatePolicy(identity_assumption="different-assumption"),
        GatePolicy(verdict_bands=base.verdict_bands + ("rejected:new-band",)),
    )
    assert all(item.registration_hash != base.registration_hash for item in variants)


def test_service_rejects_policy_different_from_committed_epoch(world):
    workload, parent = world
    epoch = ChallengeEpoch(
        workload.x_private,
        workload.y_private,
        workload.x_replication,
        workload.y_replication,
        GatePolicy(),
        salt="x",
    )
    with pytest.raises(ValueError, match="policy"):
        LeverageService(parent, epoch, GatePolicy(min_net_wins=99))


# ---------------------------------------------------------------- identity economics

def test_permissionless_registration_has_no_grant_and_principal_is_one_use(world):
    workload, parent = world
    service = make_service(workload, parent)
    service.register_contributor("alice")
    assert service.ledger.account("alice").available == 0
    service.fund_contributor("alice", "verified-human-1")
    assert service.ledger.account("alice").available == service.policy.grant_units
    service.register_contributor("sybil")
    with pytest.raises(ValueError, match="principal"):
        service.fund_contributor("sybil", "verified-human-1")
    metrics = service.metrics()["identity_economics"]
    assert metrics["permissionless_registration_grant_units"] == 0
    assert metrics["anti_farming_claim_is_conditional"] is True


# ---------------------------------------------------------------- structural binding and exact cache composition

def test_honest_delta_uses_structural_binding_and_final_replication(world):
    workload, parent = world
    service = make_service(workload, parent)
    fund(service, "honest")
    categories = tuple(range(8))
    replaced, added = honest_delta(parent, workload, categories)
    manifest, (candidate_id, verdict) = submit_delta(
        service, "honest", replaced, added, Territory(categories, 0.25)
    )
    assert verdict == "accepted:promotion-candidate"
    record = service.candidates[candidate_id]
    assert record.detail["artifact_binding_verified"] is True
    assert record.detail["behavioral_binding_full_passes"] == 0
    assert record.detail["composition_certificate"]["candidate_manifest"] == manifest.commitment
    assert service.promotion_costs[-1]["challenge_cache_rebuild_distance_ops"] == 0
    # Independent full evaluation is a test oracle, not part of trusted admission.
    assert np.array_equal(service.canonical_tissue.predict(service.epoch.x), service.cache.predictions)
    assert pass_canary(service, candidate_id, workload) == "canary-passed"
    assert record.status == "canary-passed"
    assert service.ledger.account("honest").vested == 0
    replication = service.close_epoch()
    assert replication["passed"] is True
    assert record.status == "vested"
    assert service.ledger.account("honest").vested == service.policy.bond_units + service.policy.reward_units


def test_rejected_candidate_never_runs_full_binding_or_composition(world):
    workload, parent = world
    service = make_service(workload, parent)
    fund(service, "reject")
    roots_before = service.store.roots()
    replaced, added = variance_farm_delta(parent, tuple(range(8)), seed=9001)
    _, (_, verdict) = submit_delta(
        service, "reject", replaced, added, Territory(tuple(range(8)), 0.25)
    )
    assert verdict != "accepted:promotion-candidate"
    assert service.store.roots() == roots_before
    assert service.evaluation_costs[-1]["behavioral_binding_full_passes"] == 0
    assert not any(item.get("candidate") for item in service.promotion_costs)


def test_manifest_is_only_artifact_channel_and_tampering_burns_bond(world):
    workload, parent = world
    service = make_service(workload, parent)
    fund(service, "tamper")
    replaced, added = honest_delta(parent, workload, tuple(range(8)))
    manifest = make_manifest(service, "tamper", replaced, added, Territory(tuple(range(8)), 0.25))
    manifest.added[0]["center"][0] += 0.5
    _, verdict = service.submit(manifest, "tamper")
    assert verdict == "rejected:territorial-violation"
    assert service.ledger.burned_units == service.policy.submission_fee_units + service.policy.bond_units


def test_every_committed_touched_artifact_is_evaluated(world):
    workload, parent = world
    service = make_service(workload, parent)
    fund(service, "complete")
    replaced, added = honest_delta(parent, workload, tuple(range(4)))
    manifest, (candidate_id, _) = submit_delta(
        service, "complete", replaced, added, Territory(tuple(range(4)), 0.25)
    )
    expected = len(service.epoch.x) * manifest.touched_count()
    assert service.candidates[candidate_id].detail["distance_ops"] == expected


# ---------------------------------------------------------------- subgroup harm

def test_concentrated_victim_class_harm_is_rejected(world):
    workload, parent = world
    categories = tuple(range(8))
    policy = GatePolicy(max_touched_branches=128, max_challenge_subgroup_net_loss=5)
    service = make_service(workload, parent, policy)
    fund(service, "victimizer")

    canonical = parent.canonical_categories(workload.x_private)
    parent_predictions = parent.predict(workload.x_private)
    eligible = (
        (parent_predictions == workload.y_private)
        & np.isin(canonical, categories)
        & ~np.isin(workload.y_private, categories)
    )
    labels, counts = np.unique(workload.y_private[eligible], return_counts=True)
    victim = int(labels[counts.argmax()])
    victim_points = workload.x_private[eligible & (workload.y_private == victim)]

    replaced, added = honest_delta(parent, workload, categories)
    for point in victim_points[np.linspace(0, len(victim_points) - 1, 3).astype(int)]:
        for _ in range(parent.top_k):
            added.append((point.copy(), categories[0]))
    _, (candidate_id, verdict) = submit_delta(
        service, "victimizer", replaced, added, Territory(categories, 0.25)
    )
    detail = service.candidates[candidate_id].detail
    assert detail["paired"]["net_wins"] > 0
    assert detail["worst_subgroup"]["net_wins"] < -policy.max_challenge_subgroup_net_loss
    assert verdict == "rejected:subgroup-harm"


# ---------------------------------------------------------------- staleness and lineage

def test_stale_parent_costs_no_bond_and_no_challenge_budget(world):
    workload, parent = world
    service = make_service(workload, parent)
    fund(service, "first")
    fund(service, "stale")

    stale_delta = honest_delta(parent, workload, tuple(range(16, 24)))
    stale_manifest = build_manifest(
        service.genesis_root,
        parent.representation_root(),
        "stale",
        *stale_delta,
        Territory(tuple(range(16, 24)), 0.25),
    )

    first_delta = honest_delta(parent, workload, tuple(range(8)))
    first_manifest, (first_id, first_verdict) = submit_delta(
        service, "first", *first_delta, Territory(tuple(range(8)), 0.25)
    )
    assert first_verdict == "accepted:promotion-candidate"
    assert pass_canary(service, first_id, workload) == "canary-passed"

    account_before = copy.deepcopy(service.ledger.account("stale"))
    tests_before = service.epoch.tests_spent
    burned_before = service.ledger.burned_units
    _, verdict = service.submit(stale_manifest, "stale")
    assert verdict == "rejected:stale-parent"
    assert service.epoch.tests_spent == tests_before
    assert service.ledger.burned_units == burned_before
    assert service.ledger.account("stale") == account_before
    assert service.candidates[stale_manifest.commitment].bond_locked == 0
    assert first_manifest.parent_root == service.genesis_root


def test_promotions_compose_and_final_holdout_validates_lineage(world):
    workload, parent = world
    service = make_service(workload, parent)
    prior_root = service.canonical_root
    accepted = []
    for index, categories in enumerate((tuple(range(8)), tuple(range(8, 16)))):
        contributor = f"builder-{index}"
        fund(service, contributor)
        delta = honest_delta(service.canonical_tissue, workload, categories, seed=11 + 18 * index)
        manifest, (candidate_id, verdict) = submit_delta(
            service, contributor, *delta, Territory(categories, 0.25)
        )
        assert manifest.parent_root == prior_root
        assert verdict == "accepted:promotion-candidate"
        assert pass_canary(service, candidate_id, workload) == "canary-passed"
        prior_root = service.canonical_root
        accepted.append(candidate_id)
    result = service.close_epoch()
    assert result["passed"] is True
    assert result["paired"]["net_wins"] >= service.policy.replication_min_net_wins
    assert all(service.candidates[candidate_id].status == "vested" for candidate_id in accepted)


def test_final_replication_rejects_challenge_selected_overfit_and_refunds_bond():
    parent = TerritoryTissue(
        np.asarray([[0.0], [10.0]]),
        np.asarray([0, 1]),
        top_k=1,
        temperature=1.0,
    )
    challenge_x = np.full((20, 1), 4.0)
    challenge_y = np.ones(20, dtype=np.int64)
    replication_x = np.full((20, 1), 4.0)
    replication_y = np.zeros(20, dtype=np.int64)
    policy = GatePolicy(
        alpha_epoch=0.05,
        max_submissions_per_epoch=1,
        max_submissions_per_contributor=1,
        min_net_wins=5,
        max_challenge_subgroup_net_loss=100,
        min_subgroup_samples=5,
        max_territory_categories=2,
        max_route_share=1.0,
        canary_window_events=1,
        canary_min_batch_samples=10,
        canary_max_subgroup_net_loss=100,
        replication_min_net_wins=5,
        replication_max_subgroup_net_loss=100,
    )
    epoch = ChallengeEpoch(challenge_x, challenge_y, replication_x, replication_y, policy, salt="adaptive")
    service = LeverageService(parent, epoch, policy)
    fund(service, "overfit")
    manifest = build_manifest(
        service.canonical_root,
        parent.representation_root(),
        "overfit",
        {},
        [(np.asarray([4.0]), 1)],
        Territory((0, 1), 1.0),
    )
    candidate_id, verdict = service.submit(manifest, "overfit")
    assert verdict == "accepted:promotion-candidate"
    assert service.record_canary_batch(
        candidate_id,
        challenge_x,
        challenge_y,
        source_id="independent-shaped-canary",
        source_kind="heldout-canary-batch",
    ) == "canary-passed"
    result = service.close_epoch()
    assert result["passed"] is False
    assert service.canonical_root == service.genesis_root
    assert service.candidates[candidate_id].status == "replication-rejected"
    account = service.ledger.account("overfit")
    assert account.locked == 0
    assert account.available == policy.grant_units - policy.submission_fee_units
    assert account.vested == 0
    assert service.ledger.cancelled_reward_units == policy.reward_units


# ---------------------------------------------------------------- canary observables

def test_boolean_canary_oracle_is_removed_and_labeled_batch_detects_sleeper(world):
    workload, parent = world
    service = make_service(workload, parent)
    fund(service, "sleeper")
    replaced, added, trigger_x, trigger_y = sleeper_delta(parent, workload, tuple(range(8)))
    _, (candidate_id, verdict) = submit_delta(
        service, "sleeper", replaced, added, Territory(tuple(range(8)), 0.25)
    )
    assert verdict == "accepted:promotion-candidate"
    with pytest.raises(TypeError, match="Boolean canary"):
        service.record_canary_event(candidate_id, clean=False)
    status = service.record_canary_batch(
        candidate_id,
        trigger_x,
        trigger_y,
        source_id="audited-trigger-batch-001",
        source_kind="audited-inference-batch",
    )
    observation = service.candidates[candidate_id].canary_observations[-1]
    assert status == "revoked"
    assert observation["paired"]["losses"] == len(trigger_y)
    assert observation["clean"] is False
    assert service.canonical_root == service.genesis_root
    assert service.ledger.slashed_units == service.policy.bond_units + service.policy.reward_units


def test_duplicate_canary_batch_is_rejected(world):
    workload, parent = world
    policy = GatePolicy(canary_window_events=2)
    service = make_service(workload, parent, policy)
    fund(service, "canary")
    delta = honest_delta(parent, workload, tuple(range(8)))
    _, (candidate_id, verdict) = submit_delta(
        service, "canary", *delta, Territory(tuple(range(8)), 0.25)
    )
    assert verdict == "accepted:promotion-candidate"
    x, y = stratified_canary_batches(workload, 2)[0]
    assert service.record_canary_batch(
        candidate_id, x, y, source_id="one", source_kind="heldout-canary-batch"
    ) == "pending"
    with pytest.raises(ValueError, match="duplicate"):
        service.record_canary_batch(
            candidate_id, x, y, source_id="one", source_kind="heldout-canary-batch"
        )


# ---------------------------------------------------------------- budgets, idempotency, territory, and cost

def test_duplicate_submission_is_idempotent_and_not_recharged(world):
    workload, parent = world
    service = make_service(workload, parent)
    fund(service, "retry")
    replaced, added = variance_farm_delta(parent, tuple(range(8)), seed=3000)
    manifest = make_manifest(service, "retry", replaced, added, Territory(tuple(range(8)), 0.25))
    first = service.submit(manifest, "retry")
    snapshot = service.ledger.snapshot()
    tests = service.epoch.tests_spent
    second = service.submit(manifest, "retry")
    assert second == first
    assert service.ledger.snapshot() == snapshot
    assert service.epoch.tests_spent == tests


def test_global_and_per_contributor_budgets_are_enforced(world):
    workload, parent = world
    policy = GatePolicy(max_submissions_per_epoch=5, max_submissions_per_contributor=2, grant_units=100_000)
    service = make_service(workload, parent, policy)
    fund(service, "miner")
    verdicts = []
    for seed in range(3):
        replaced, added = variance_farm_delta(parent, tuple(range(8)), seed=4000 + seed)
        _, (_, verdict) = submit_delta(
            service, "miner", replaced, added, Territory(tuple(range(8)), 0.25)
        )
        verdicts.append(verdict)
    assert verdicts[-1] == "rejected:contributor-budget-exhausted"
    assert service.epoch.tests_spent == 2


def test_active_canary_blocks_overlapping_submission_without_charge(world):
    workload, parent = world
    service = make_service(workload, parent)
    fund(service, "a")
    fund(service, "b")
    delta = honest_delta(parent, workload, tuple(range(8)))
    _, (candidate_id, verdict) = submit_delta(service, "a", *delta, Territory(tuple(range(8)), 0.25))
    assert verdict == "accepted:promotion-candidate"
    tests_before = service.epoch.tests_spent
    account_before = copy.deepcopy(service.ledger.account("b"))
    other = honest_delta(service.canonical_tissue, workload, tuple(range(8, 16)))
    _, (_, blocked) = submit_delta(service, "b", *other, Territory(tuple(range(8, 16)), 0.25))
    assert blocked == "rejected:canary-window-active"
    assert service.epoch.tests_spent == tests_before
    assert service.ledger.account("b") == account_before
    assert pass_canary(service, candidate_id, workload) == "canary-passed"


def test_guard_makes_escape_attempt_inert(world):
    workload, parent = world
    cache = build_eval_cache(parent, workload.x_private)
    _, intruders = escape_delta(parent, workload, (0,), target=50)
    guarded = evaluate_candidate(parent, cache, {}, intruders, Territory((0,), 0.05))
    unguarded = evaluate_candidate(parent, cache, {}, intruders, Territory(tuple(range(100)), 1.0))
    assert guarded.outside_discordant == 0
    assert (unguarded.predictions != cache.predictions).sum() > 0


def test_replacement_owner_change_is_rejected(world):
    workload, parent = world
    service = make_service(workload, parent)
    fund(service, "owner-change")
    manifest = make_manifest(
        service,
        "owner-change",
        {0: (parent.centers[0].copy(), 1)},
        [],
        Territory((0, 1), 0.25),
    )
    _, verdict = service.submit(manifest, "owner-change")
    assert verdict == "rejected:territorial-violation"


def test_cost_accounting_includes_zero_behavioral_binding_passes(world):
    workload, parent = world
    service = make_service(workload, parent)
    fund(service, "metrics")
    delta = honest_delta(parent, workload, tuple(range(8)))
    submit_delta(service, "metrics", *delta, Territory(tuple(range(8)), 0.25))
    metrics = service.metrics()["verification_locality"]
    assert metrics["mean_candidate_distance_ratio"] < 0.5
    assert metrics["behavioral_binding_full_passes"] == 0
    assert "zero full inference replays" in metrics["claim_boundary"].lower()


# ---------------------------------------------------------------- search evidence and disclosure

def test_untrusted_public_search_measures_actual_candidate_discovery(world):
    workload, parent = world
    report = search_local_refits(
        parent,
        workload.x_train,
        workload.y_train,
        workload.x_public,
        workload.y_public,
        (tuple(range(8)), tuple(range(8, 16))),
        seeds=(11, 29),
        iterations=10,
    )
    assert report.protected_split_accessed is False
    assert report.candidates_generated == 4
    assert report.candidates_with_positive_public_net_wins > 0
    assert report.best_public_net_wins > 0


def test_challenge_and_replication_are_independently_fresh_and_disclosed(world):
    first = make_surrogate_workload()
    second = make_surrogate_workload()
    assert first.private_seed != second.private_seed
    assert first.replication_seed != second.replication_seed
    assert not np.array_equal(first.x_private, second.x_private)
    assert not np.array_equal(first.x_replication, second.x_replication)

    workload, parent = world
    policy = GatePolicy(replication_min_net_wins=0, replication_alpha=0.99)
    service = make_service(workload, parent, policy)
    commitment = service.epoch.commitment
    result = service.close_epoch()
    assert result["passed"] is False
    disclosure = service.reveal_epoch()
    assert disclosure["commitment"] == commitment
    assert ChallengeEpoch.verify_disclosure(disclosure)
    assert disclosure["challenge"]["features"]
    assert disclosure["replication"]["features"]


def test_leverage_state_roundtrip_preserves_lineage_funding_and_audit(tmp_path, world):
    workload, parent = world
    path = tmp_path / "leverage-state.json"
    service = make_service(workload, parent, state_path=path)
    fund(service, "persistent", "verified:persistent")
    delta = honest_delta(parent, workload, tuple(range(8)))
    _, (candidate_id, verdict) = submit_delta(
        service, "persistent", *delta, Territory(tuple(range(8)), 0.25)
    )
    assert verdict == "accepted:promotion-candidate"
    root = service.canonical_root
    restored = LeverageService.load(path)
    assert restored.canonical_root == root
    assert restored.active_canary_id == candidate_id
    assert restored.candidates[candidate_id].status == "promoted"
    assert restored.funded_principals == service.funded_principals
    assert restored.ledger.snapshot() == service.ledger.snapshot()
    assert restored.validate_audit_chain()
    assert np.array_equal(restored.cache.predictions, service.cache.predictions)
