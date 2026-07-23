#!/usr/bin/env python3
from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path

from fastapi.testclient import TestClient

from dendriswarm.coordinator.app import create_app
from dendriswarm.core.crypto import Identity, nonce, verify
from dendriswarm.core.models import TaskKind
from dendriswarm.tissues.reference import artifact_hash, dataset_hash
from dendriswarm.worker.executor import execute_task


def register(client: TestClient, identity: Identity) -> None:
    value = {
        "node_id": identity.node_id,
        "public_key": identity.public_key_b64,
        "capabilities": {
            "cpu_count": 2,
            "memory_mb": 1024,
            "accelerator": "cpu",
            "platform": "proof-runner",
            "tags": ["reference-runtime-v2", "deterministic-v2"],
        },
        "timestamp": int(time.time()),
        "nonce": nonce(),
    }
    value["signature"] = identity.sign(value)
    response = client.post("/v1/nodes/register", json=value)
    response.raise_for_status()


def signed_action(identity: Identity, action: str) -> dict[str, object]:
    body: dict[str, object] = {
        "action": action,
        "node_id": identity.node_id,
        "timestamp": int(time.time()),
        "nonce": nonce(),
    }
    return {k: v for k, v in body.items() if k != "action"} | {"signature": identity.sign(body)}


def materialize(client: TestClient, payload: dict) -> dict:
    out = dict(payload)
    if "dataset_hash" in payload:
        dataset = client.get(f"/v1/datasets/{payload['dataset_hash']}").json()
        assert dataset_hash(dataset) == payload["dataset_hash"]
        out["_dataset"] = dataset
    if "artifact_hash" in payload:
        artifact = client.get(f"/v1/artifacts/{payload['artifact_hash']}").json()
        assert artifact_hash(artifact) == payload["artifact_hash"]
        out["_artifact"] = artifact
    return out


def execute_one(client: TestClient, identity: Identity):
    response = client.post("/v1/tasks/claim", json=signed_action(identity, "claim"))
    if response.status_code == 204:
        return None
    response.raise_for_status()
    envelope = response.json()
    assert verify(envelope["coordinator_public_key"], envelope["task"], envelope["signature"])
    task = envelope["task"]
    output = execute_task(TaskKind(task["kind"]), materialize(client, task["payload"]))
    result = {
        "node_id": identity.node_id,
        "task_id": task["id"],
        "lease_token": task["lease_token"],
        "duration_ms": 1,
        "output": output,
    }
    result["signature"] = identity.sign(result)
    accepted = client.post("/v1/tasks/result", json=result)
    accepted.raise_for_status()
    return task, accepted.json()


def signed_inference(identity: Identity, features: list[float]) -> dict:
    value = {
        "node_id": identity.node_id,
        "request_id": nonce(),
        "timestamp": int(time.time()),
        "nonce": nonce(),
        "features": features,
    }
    value["signature"] = identity.sign({"action": "inference", **value})
    return value


def main() -> None:
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="dendriswarm-proof-") as temp:
        app = create_app(Path(temp), bootstrap=True, inference_audit_rate=1.0)
        with TestClient(app) as client:
            seeds = [Identity.generate() for _ in range(4)]
            for seed in seeds:
                register(client, seed)
            task_kinds: dict[str, int] = {}
            for step in range(240):
                completed = execute_one(client, seeds[step % len(seeds)])
                if completed:
                    kind = completed[0]["kind"]
                    task_kinds[kind] = task_kinds.get(kind, 0) + 1
                stats = client.get("/v1/stats").json()
                if stats["canonical"] and stats["queued_tasks"] == 0 and stats["assigned_tasks"] == 0:
                    break
            sample = client.get("/v1/samples/digit/0").json()
            inference_request = client.post(
                "/v1/inference", json=signed_inference(seeds[0], sample["features"])
            )
            inference_request.raise_for_status()
            task_id = inference_request.json()["task_id"]
            inference_execution = execute_one(client, seeds[1])
            assert inference_execution and inference_execution[0]["id"] == task_id
            job = client.get(f"/v1/jobs/{task_id}").json()
            stats = client.get("/v1/stats").json()
            checkpoint = client.get("/v1/audit/checkpoint").json()
            report = {
                "format": "dendriswarm.proof-run.v2",
                "elapsed_seconds": round(time.perf_counter() - started, 3),
                "seed_nodes": len(seeds),
                "task_counts_before_inference": task_kinds,
                "stats": stats,
                "inference": {
                    "sample_index": sample["index"],
                    "expected_label": sample["label"],
                    "task_id": task_id,
                    "output": job["output"],
                    "correct": job["output"]["prediction"] == sample["label"],
                },
                "audit_checkpoint": checkpoint,
            }
            print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
