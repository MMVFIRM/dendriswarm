import time

from fastapi.testclient import TestClient

from dendriswarm.coordinator.app import create_app
from dendriswarm.core.crypto import Identity, nonce
from tests.helpers import claim_and_execute, register, signed_action, signed_inference


def registration(identity, node_id=None):
    value = {
        "node_id": node_id or identity.node_id,
        "public_key": identity.public_key_b64,
        "capabilities": {
            "cpu_count": 2,
            "memory_mb": 1024,
            "accelerator": "cpu",
            "platform": "pytest",
            "tags": ["deterministic-v2"],
        },
        "timestamp": int(time.time()),
        "nonce": nonce(),
    }
    value["signature"] = identity.sign(value)
    return value


def test_spoofed_registration_is_rejected(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        honest, attacker = Identity.generate(), Identity.generate()
        assert client.post("/v1/nodes/register", json=registration(honest, attacker.node_id)).status_code == 401


def test_replayed_claim_is_rejected(tmp_path):
    with TestClient(create_app(tmp_path, bootstrap=True)) as client:
        identity = Identity.generate()
        register(client, identity)
        request = signed_action(identity, "claim", nonce())
        assert client.post("/v1/tasks/claim", json=request).status_code == 200
        assert client.post("/v1/tasks/claim", json=request).status_code == 401


def test_attacker_cannot_spend_victim_credits(tmp_path):
    app = create_app(tmp_path)
    with TestClient(app) as client:
        victim, attacker = Identity.generate(), Identity.generate()
        register(client, victim)
        register(client, attacker)
        app.state.service.db.credit("test-credit", victim.node_id, 1000, "test")
        request = signed_inference(victim, [0.0] * 64)
        request["signature"] = attacker.sign({"action": "inference", **{k: v for k, v in request.items() if k != "signature"}})
        assert client.post("/v1/inference", json=request).status_code == 401
        assert app.state.service.db.node(victim.node_id)["credit_units"] == 1000


def test_audited_wrong_inference_is_rejected_and_worker_penalized(tmp_path):
    app = create_app(tmp_path, bootstrap=True, inference_audit_rate=1.0)
    with TestClient(app) as client:
        seeds = [Identity.generate() for _ in range(4)]
        for seed in seeds:
            register(client, seed)
        for step in range(240):
            result = claim_and_execute(client, seeds[step % 4])
            if result:
                assert result[2].status_code == 200
            stats = client.get("/v1/stats").json()
            if stats["canonical"]:
                break
        # Ensure both the adversary and a recovery worker can post the bonded claim.
        app.state.service.db.credit("test-bond-adversary", seeds[1].node_id, 5_000, "test")
        app.state.service.db.credit("test-bond-recovery", seeds[2].node_id, 5_000, "test")
        sample = client.get("/v1/samples/digit/1").json()
        queued = client.post("/v1/inference", json=signed_inference(seeds[0], sample["features"]))
        assert queued.status_code == 200

        def lie(_, output):
            output["prediction"] = (output["prediction"] + 1) % 10
            return output

        before_bad = app.state.service.db.node(seeds[1].node_id)["credit_units"]
        task, _, bad = claim_and_execute(client, seeds[1], mutate=lie)
        assert task["id"] == queued.json()["task_id"]
        assert bad.status_code == 400
        assert client.get(f"/v1/jobs/{task['id']}").json()["status"] == "queued"
        assert app.state.service.db.node(seeds[1].node_id)["failed"] >= 1
        assert app.state.service.db.node(seeds[1].node_id)["credit_units"] == before_bad - 4_000

        recovered_task, _, recovered = claim_and_execute(client, seeds[2])
        assert recovered_task["id"] == task["id"] and recovered.status_code == 200
        assert client.get(f"/v1/jobs/{task['id']}").json()["status"] == "completed"


def test_integer_credit_ledger_has_no_float_drift(tmp_path):
    app = create_app(tmp_path)
    identity = Identity.generate()
    app.state.service.db.register_node(identity.node_id, identity.public_key_b64, {})
    for index in range(1_000):
        app.state.service.db.credit(f"micro-{index}", identity.node_id, 1, "micro")
    assert app.state.service.db.node(identity.node_id)["credit_units"] == 1_000


def test_inference_audit_sampling_uses_persisted_coordinator_secret(tmp_path):
    first = create_app(tmp_path / "secret-state")
    secret = first.state.service._inference_audit_secret
    assert secret and secret != first.state.service.identity.public_key_b64
    second = create_app(tmp_path / "secret-state")
    assert second.state.service._inference_audit_secret == secret
    secret_path = tmp_path / "secret-state" / "keys" / "inference-audit-secret"
    assert secret_path.exists()
