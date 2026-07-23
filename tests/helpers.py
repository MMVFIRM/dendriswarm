import time

from dendriswarm.core.crypto import nonce, verify
from dendriswarm.core.models import TaskKind
from dendriswarm.tissues.reference import artifact_hash, dataset_hash
from dendriswarm.worker.executor import execute_task


def register(client, identity):
    value = {
        "node_id": identity.node_id,
        "public_key": identity.public_key_b64,
        "capabilities": {
            "cpu_count": 2,
            "memory_mb": 1024,
            "accelerator": "cpu",
            "platform": "pytest",
            "tags": ["reference-runtime-v2", "deterministic-v2"],
        },
        "timestamp": int(time.time()),
        "nonce": nonce(),
    }
    value["signature"] = identity.sign(value)
    response = client.post("/v1/nodes/register", json=value)
    assert response.status_code == 200, response.text


def signed_action(identity, action, fixed_nonce=None):
    body = {
        "action": action,
        "node_id": identity.node_id,
        "timestamp": int(time.time()),
        "nonce": fixed_nonce or nonce(),
    }
    return {k: v for k, v in body.items() if k != "action"} | {"signature": identity.sign(body)}


def materialize(client, payload, identity=None, task=None):
    out = dict(payload)
    if "dataset_hash" in payload:
        dataset = client.get(f"/v1/datasets/{payload['dataset_hash']}").json()
        assert dataset_hash(dataset) == payload["dataset_hash"]
        out["_dataset"] = dataset
    if "artifact_hash" in payload:
        artifact = client.get(f"/v1/artifacts/{payload['artifact_hash']}").json()
        assert artifact_hash(artifact) == payload["artifact_hash"]
        out["_artifact"] = artifact
    if "native10_checkpoint_root" in payload:
        from dendriswarm.v5.native10 import Native10Dendritron
        checkpoint = client.get(f"/v1/native10/checkpoints/{payload['native10_checkpoint_root']}").json()
        assert Native10Dendritron.from_artifact(checkpoint).root == payload["native10_checkpoint_root"]
        out["_native10_checkpoint"] = checkpoint
    if "global_validation_hash" in payload and task is not None and task.get("kind") == TaskKind.DENDRITRON_VERIFICATION.value:
        assert identity is not None and task is not None
        body = {
            "action": "fetch-native10-validation",
            "node_id": identity.node_id,
            "task_id": task["id"],
            "lease_token": task["lease_token"],
            "timestamp": int(time.time()),
            "nonce": __import__("dendriswarm.core.crypto", fromlist=["nonce"]).nonce(),
        }
        request = {key: value for key, value in body.items() if key != "action"}
        request["signature"] = identity.sign(body)
        response = client.post(
            f"/v1/native10/validation/{payload['global_validation_hash']}", json=request
        )
        assert response.status_code == 200, response.text
        validation = response.json()
        assert validation["sha256"] == payload["global_validation_hash"]
        out["_native10_validation"] = validation
    return out


def claim_and_execute(client, identity, mutate=None):
    response = client.post("/v1/tasks/claim", json=signed_action(identity, "claim"))
    if response.status_code == 204:
        return None
    assert response.status_code == 200, response.text
    envelope = response.json()
    assert verify(envelope["coordinator_public_key"], envelope["task"], envelope["signature"])
    task = envelope["task"]
    output = execute_task(
        TaskKind(task["kind"]), materialize(client, task["payload"], identity, task)
    )
    if mutate:
        output = mutate(task, output)
    body = {
        "node_id": identity.node_id,
        "task_id": task["id"],
        "lease_token": task["lease_token"],
        "duration_ms": 1,
        "output": output,
    }
    body["signature"] = identity.sign(body)
    return task, body, client.post("/v1/tasks/result", json=body)


def signed_inference(identity, features, request_id=None, request_nonce=None):
    request = {
        "node_id": identity.node_id,
        "request_id": request_id or nonce(),
        "timestamp": int(time.time()),
        "nonce": request_nonce or nonce(),
        "features": features,
    }
    request["signature"] = identity.sign({"action": "inference", **request})
    return request
