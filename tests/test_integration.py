import time

from fastapi.testclient import TestClient

from dendriswarm.coordinator.app import create_app
from dendriswarm.core.crypto import Identity
from tests.helpers import claim_and_execute, register, signed_action, signed_inference


def build_canonical(client, seeds):
    last = None
    for step in range(240):
        result = claim_and_execute(client, seeds[step % len(seeds)])
        if result:
            task, body, response = result
            assert response.status_code == 200, response.text
            last = (task, body)
        stats = client.get("/v1/stats").json()
        if stats["canonical"] and stats["queued_tasks"] == 0 and stats["assigned_tasks"] == 0:
            return stats, last
    raise AssertionError("swarm did not converge to a canonical tissue")


def test_four_seeds_independent_quorum_promote_paid_inference_and_duplicate_result(tmp_path):
    app = create_app(tmp_path / "coordinator", bootstrap=True, inference_audit_rate=1.0)
    with TestClient(app) as client:
        seeds = [Identity.generate() for _ in range(4)]
        for seed in seeds:
            register(client, seed)
        stats, last = build_canonical(client, seeds)
        canonical = stats["canonical"]
        assert canonical["test_accuracy"] >= stats["benchmark"]["accuracy"]["logistic_regression"]
        assert canonical["hidden_accuracy"] >= 0.94
        assert canonical["verifications"] >= 2
        assert stats["audit"]["valid"] is True
        candidate = app.state.service.db.candidate_by_hash(canonical["artifact_hash"])
        verifiers = {r["verifier_node"] for r in app.state.service.db.candidate_verifications(candidate["id"])}
        assert len(verifiers) >= 2
        assert candidate["trainer_node"] not in verifiers

        before_units = stats["credit_supply_units"]
        duplicate = client.post("/v1/tasks/result", json=last[1])
        assert duplicate.status_code == 200
        assert duplicate.json()["duplicate"] is True
        assert client.get("/v1/stats").json()["credit_supply_units"] == before_units

        sample = client.get("/v1/samples/digit/0").json()
        request = signed_inference(seeds[0], sample["features"])
        queued = client.post("/v1/inference", json=request)
        assert queued.status_code == 200, queued.text
        task_id = queued.json()["task_id"]
        task, _, response = claim_and_execute(client, seeds[1])
        assert task["id"] == task_id and response.status_code == 200
        job = client.get(f"/v1/jobs/{task_id}").json()
        assert job["output"]["prediction"] == sample["label"]

        # A retried identical signed request is idempotent and cannot double-spend.
        balance_before_retry = app.state.service.db.node(seeds[0].node_id)["credit_units"]
        retried = client.post("/v1/inference", json=request)
        assert retried.status_code == 200 and retried.json()["created"] is False
        assert retried.json()["task_id"] == task_id
        assert app.state.service.db.node(seeds[0].node_id)["credit_units"] == balance_before_retry


def test_expired_lease_requeues_with_new_token(tmp_path):
    app = create_app(tmp_path, bootstrap=True, lease_seconds=0.01)
    with TestClient(app) as client:
        a, b = Identity.generate(), Identity.generate()
        register(client, a)
        register(client, b)
        first = client.post("/v1/tasks/claim", json=signed_action(a, "claim")).json()["task"]
        time.sleep(0.02)
        second = client.post("/v1/tasks/claim", json=signed_action(b, "claim")).json()["task"]
        assert second["id"] == first["id"]
        assert second["lease_token"] != first["lease_token"]

def test_api_routes_are_unique(tmp_path):
    from dendriswarm.coordinator.app import create_app

    app = create_app(tmp_path / "route-state")
    seen = set()
    duplicates = []
    for route in app.routes:
        path = getattr(route, "path", None)
        methods = tuple(sorted(getattr(route, "methods", set()) or set()))
        if not path or not methods:
            continue
        key = (path, methods)
        if key in seen:
            duplicates.append(key)
        seen.add(key)
    assert duplicates == []



def test_concurrent_claims_serialize_single_sqlite_connection(tmp_path):
    from concurrent.futures import ThreadPoolExecutor

    from dendriswarm.coordinator.db import Database
    from dendriswarm.core.models import TaskKind

    db = Database(tmp_path / "concurrent.sqlite3")
    node_ids = [f"node-{index}" for index in range(12)]
    for node_id in node_ids:
        db.register_node(node_id, f"key-{node_id}", {"tags": ["deterministic-v2"]})
    for index in range(12):
        db.add_task(
            TaskKind.EXPLORATION,
            {"config": {"index": index}, "required_tags": ["deterministic-v2"]},
            1000,
            1,
        )

    with ThreadPoolExecutor(max_workers=12) as pool:
        rows = list(pool.map(lambda node_id: db.claim_task(node_id, 30.0), node_ids))

    assert all(row is not None for row in rows)
    assert len({row["id"] for row in rows if row is not None}) == 12
    assert all(row["assigned_to"] in node_ids for row in rows if row is not None)


def test_leverage_http_submission_is_authenticated_and_banded(tmp_path):
    from dendriswarm.leverage.manifest import build_manifest
    from dendriswarm.leverage.tissue import Territory
    from dendriswarm.leverage.workload import honest_delta, make_surrogate_workload

    app = create_app(tmp_path / "leverage-http", enable_leverage=True)
    with TestClient(app) as client:
        identity = Identity.generate()
        register(client, identity)
        meta = client.get("/v1/leverage/meta")
        assert meta.status_code == 200
        assert "challenge" not in meta.json()

        leverage = app.state.leverage
        leverage.register_contributor(identity.node_id)
        leverage.fund_contributor(identity.node_id, f"verified-test-node:{identity.public_key_b64}")
        public_workload = make_surrogate_workload(private_seed=42)
        parent = leverage.canonical_tissue
        categories = tuple(range(8))
        replaced, added = honest_delta(parent, public_workload, categories)
        manifest = build_manifest(
            leverage.canonical_root,
            parent.representation_root(),
            identity.node_id,
            replaced,
            added,
            Territory(categories, 0.25),
        )
        body = {
            "node_id": identity.node_id,
            "timestamp": int(time.time()),
            "nonce": __import__("dendriswarm.core.crypto", fromlist=["nonce"]).nonce(),
            "manifest": manifest.as_dict(include_artifacts=True),
        }
        body["signature"] = identity.sign({"action": "leverage-submit", **body})
        response = client.post("/v1/leverage/candidates", json=body)
        assert response.status_code == 200, response.text
        assert response.json()["verdict"] in leverage.policy.verdict_bands
        candidate = client.get(f"/v1/leverage/candidates/{response.json()['candidate_id']}")
        assert candidate.status_code == 200
        assert "paired" not in candidate.json()


def test_coordinator_promotes_without_retraining_worker_contributions(tmp_path, monkeypatch):
    import dendriswarm.coordinator.service as coordinator_service

    real_reference = coordinator_service.ReferenceDendritron

    class CoordinatorValidationOnly:
        @staticmethod
        def from_artifact(artifact):
            return real_reference.from_artifact(artifact)

        @staticmethod
        def train(*args, **kwargs):
            raise AssertionError("the coordinator must not retrain volunteer exploration or training work")

    monkeypatch.setattr(coordinator_service, "ReferenceDendritron", CoordinatorValidationOnly)
    app = create_app(tmp_path / "no-replay", bootstrap=True, lease_seconds=5.0)
    with TestClient(app) as client:
        seeds = [Identity.generate() for _ in range(4)]
        for seed in seeds:
            register(client, seed)
        stats, _ = build_canonical(client, seeds)
        assert stats["canonical"] is not None
        assert stats["verification_modes"]["exploration"] == "replicated-consensus"
        assert stats["verification_modes"]["training"] == "replicated-artifact-consensus"


def test_result_side_effects_roll_back_if_completion_fails(tmp_path, monkeypatch):
    from dendriswarm.core.models import TaskKind
    from dendriswarm.worker.executor import execute_task
    from tests.helpers import materialize

    app = create_app(tmp_path / "atomic-result", bootstrap=True, lease_seconds=5.0)
    with TestClient(app) as client:
        identity = Identity.generate()
        register(client, identity)
        task = client.post("/v1/tasks/claim", json=signed_action(identity, "claim")).json()["task"]
        output = execute_task(TaskKind(task["kind"]), materialize(client, task["payload"]))
        body = {
            "node_id": identity.node_id,
            "task_id": task["id"],
            "lease_token": task["lease_token"],
            "duration_ms": 1,
            "output": output,
        }
        body["signature"] = identity.sign(body)
        monkeypatch.setattr(app.state.service.db, "complete_task", lambda *args, **kwargs: False)
        response = client.post("/v1/tasks/result", json=body)
        assert response.status_code == 400
        with app.state.service.db.lock:
            reward = app.state.service.db.conn.execute(
                "SELECT 1 FROM ledger WHERE entry_key=?", (f"reward:{task['id']}",)
            ).fetchone()
            row = app.state.service.db.conn.execute(
                "SELECT status,output FROM tasks WHERE id=?", (task["id"],)
            ).fetchone()
        assert reward is None
        assert row["status"] == "assigned" and row["output"] is None


def test_canary_http_requires_an_authorized_non_contributor_auditor(tmp_path):
    app = create_app(tmp_path / "canary-auth", enable_leverage=True)
    with TestClient(app) as client:
        identity = Identity.generate()
        register(client, identity)
        body = {
            "node_id": identity.node_id,
            "timestamp": int(time.time()),
            "nonce": __import__("dendriswarm.core.crypto", fromlist=["nonce"]).nonce(),
            "features": [[0.0] * 8 for _ in range(50)],
            "labels": [0] * 50,
            "source_id": "candidate-selected-data",
            "source_kind": "heldout-canary-batch",
        }
        body["signature"] = identity.sign({"action": "leverage-canary-batch", **body})
        response = client.post("/v1/leverage/candidates/not-active/canary", json=body)
        assert response.status_code == 403
        assert "authorized canary auditor" in response.text
