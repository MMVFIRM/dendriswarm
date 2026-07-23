#!/usr/bin/env python3
"""Reproduce the v0.4.0 heterogeneous volunteer-compute proof."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from fastapi.testclient import TestClient

from dendriswarm.coordinator.app import create_app
from dendriswarm.coordinator.db import Database
from dendriswarm.core.crypto import Identity, nonce, verify
from dendriswarm.core.models import NodeCapabilities, SeedPolicy, TaskKind, TaskRequirements
from dendriswarm.core.resources import effective_limits, node_can_run
from dendriswarm.tissues.reference import artifact_hash, dataset_hash
from dendriswarm.worker.config import SeedPolicyStore
from dendriswarm.worker.executor import execute_task


def signed_registration(identity: Identity, capabilities: dict, policy: dict) -> dict:
    body = {
        "node_id": identity.node_id,
        "public_key": identity.public_key_b64,
        "capabilities": capabilities,
        "policy": policy,
        "timestamp": int(time.time()),
        "nonce": nonce(),
    }
    body["signature"] = identity.sign(body)
    return body


def signed_action(identity: Identity, action: str, **extra: object) -> dict:
    body = {
        "action": action,
        "node_id": identity.node_id,
        **extra,
        "timestamp": int(time.time()),
        "nonce": nonce(),
    }
    return {key: value for key, value in body.items() if key != "action"} | {
        "signature": identity.sign(body)
    }


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


def main() -> None:
    started = time.perf_counter()
    gates: list[dict] = []
    evidence: dict = {}

    with tempfile.TemporaryDirectory(prefix="dendriswarm-v04-proof-") as temp:
        root = Path(temp)
        app = create_app(root / "coordinator", bootstrap=True, lease_seconds=1.0)
        with TestClient(app) as client:
            meta = client.get("/v1/meta").json()
            gates.append({
                "name": "no-accelerator-required",
                "pass": meta["required_accelerator"] is None and meta["portable_backend"] == "numpy-cpu",
            })

            low = Identity.generate()
            low_peer = Identity.generate()
            capabilities = {
                "cpu_count": 1,
                "physical_cpu_count": 1,
                "memory_mb": 768,
                "memory_available_mb": 640,
                "disk_free_mb": 1024,
                "accelerator": "cpu",
                "accelerators": ["cpu"],
                "platform": "proof-low-resource",
                "machine": "arm64",
                "python_version": sys.version.split()[0],
                "supported_backends": ["numpy-cpu"],
                "benchmark_units_per_second": 100_000_000.0,
                "tags": ["reference-runtime-v2", "deterministic-v2", "portable-numpy-v1"],
            }
            policy = SeedPolicy(
                cpu_percent=100,
                memory_percent=75,
                disk_limit_mb=512,
                max_task_seconds=900,
                allowed_task_kinds=[TaskKind.EXPLORATION],
                allow_on_battery=True,
            ).model_dump(mode="json")
            for identity in (low, low_peer):
                registration = client.post(
                    "/v1/nodes/register", json=signed_registration(identity, capabilities, policy)
                )
                registration.raise_for_status()
            claim = client.post("/v1/tasks/claim", json=signed_action(low, "claim"))
            claim.raise_for_status()
            envelope = claim.json()
            task = envelope["task"]
            signed_ok = verify(envelope["coordinator_public_key"], task, envelope["signature"])
            gates.append({
                "name": "one-core-arm-model-scouting",
                "pass": signed_ok and task["kind"] == "exploration" and task["requirements"]["backend"] == "numpy-cpu",
            })

            renewal = client.post(
                "/v1/tasks/renew",
                json=signed_action(low, "renew-lease", task_id=task["id"], lease_token=task["lease_token"]),
            )
            gates.append({
                "name": "slow-node-signed-lease-renewal",
                "pass": renewal.status_code == 200 and renewal.json()["lease_expires_at"] > task["lease_expires_at"],
            })

            output = execute_task(TaskKind(task["kind"]), materialize(client, task["payload"]), cpu_threads=1)
            result = {
                "node_id": low.node_id,
                "task_id": task["id"],
                "lease_token": task["lease_token"],
                "duration_ms": 1,
                "output": output,
            }
            result["signature"] = low.sign(result)
            accepted = client.post("/v1/tasks/result", json=result)
            accepted.raise_for_status()

            # Replicated consensus replaces coordinator replay. A second
            # independent one-core volunteer confirms the logical experiment
            # before either result is rewarded.
            peer_claim = client.post("/v1/tasks/claim", json=signed_action(low_peer, "claim"))
            peer_claim.raise_for_status()
            peer_task = peer_claim.json()["task"]
            peer_output = execute_task(
                TaskKind(peer_task["kind"]), materialize(client, peer_task["payload"]), cpu_threads=1
            )
            peer_result = {
                "node_id": low_peer.node_id,
                "task_id": peer_task["id"],
                "lease_token": peer_task["lease_token"],
                "duration_ms": 1,
                "output": peer_output,
            }
            peer_result["signature"] = low_peer.sign(peer_result)
            peer_accepted = client.post("/v1/tasks/result", json=peer_result)
            peer_accepted.raise_for_status()
            account = client.get(f"/v1/nodes/{low.node_id}").json()
            gates.append({
                "name": "low-resource-replicated-work-accepted-and-credited",
                "pass": (
                    accepted.status_code == 200
                    and peer_accepted.status_code == 200
                    and account["completed"] == 1
                    and account["credit_units"] > 0
                ),
            })
            evidence["low_resource_task"] = {
                "machine": capabilities["machine"],
                "cpu_count": capabilities["cpu_count"],
                "memory_mb": capabilities["memory_mb"],
                "kind": task["kind"],
                "requirements": task["requirements"],
                "credit_units": account["credit_units"],
            }

        portable_requirements = TaskRequirements(
            backend="numpy-cpu", min_memory_mb=128, min_disk_mb=8, estimated_runtime_seconds=10
        )
        portable_policy = SeedPolicy(cpu_percent=50, memory_percent=50, disk_limit_mb=256)
        machine_results = {}
        for machine in ("x86_64", "amd64", "aarch64", "arm64"):
            eligible, reason = node_can_run(
                TaskKind.EXPLORATION,
                portable_requirements,
                NodeCapabilities(
                    cpu_count=4,
                    memory_mb=4096,
                    memory_available_mb=3000,
                    disk_free_mb=5000,
                    machine=machine,
                    supported_backends=["numpy-cpu"],
                ),
                portable_policy,
            )
            machine_results[machine] = {"eligible": eligible, "reason": reason}
        gates.append({
            "name": "mainstream-cpu-architecture-portability",
            "pass": all(value["eligible"] for value in machine_results.values()),
        })
        evidence["machine_eligibility"] = machine_results

        store = SeedPolicyStore(root / "seed" / "seed-config.json")
        initial = store.load()
        changed = store.update(
            cpu_percent=60,
            memory_percent=40,
            allowed_task_kinds=["exploration", "verification"],
            paused=True,
        )
        resumed = store.update(paused=False)
        gates.append({
            "name": "atomic-hot-reload-pause-resume",
            "pass": initial.cpu_percent == 25 and changed.paused and not resumed.paused and resumed.cpu_percent == 60,
        })
        evidence["hot_reload_policy"] = resumed.model_dump(mode="json")

        db = Database(root / "best-fit.sqlite3")
        capable = NodeCapabilities(
            cpu_count=8,
            memory_mb=16000,
            memory_available_mb=12000,
            disk_free_mb=20000,
            tags=["portable-numpy-v1"],
        ).model_dump(mode="json")
        db.register_node(
            "large-node",
            "key",
            capable,
            SeedPolicy(cpu_percent=100, memory_percent=90, disk_limit_mb=10000).model_dump(mode="json"),
        )
        small_id = db.add_task(
            TaskKind.EXPLORATION,
            {},
            1,
            10,
            requirements=TaskRequirements(min_memory_mb=128, preferred_cpu_threads=1),
        )
        large_id = db.add_task(
            TaskKind.EXPLORATION,
            {},
            1,
            10,
            requirements=TaskRequirements(min_memory_mb=4000, preferred_cpu_threads=4),
        )
        best_fit = db.claim_task("large-node", 30)
        gates.append({
            "name": "best-fit-preserves-small-work",
            "pass": best_fit is not None and best_fit["id"] == large_id and best_fit["id"] != small_id,
        })

        limits = effective_limits(
            NodeCapabilities(cpu_count=8, memory_mb=8000, memory_available_mb=6000, disk_free_mb=10000),
            SeedPolicy(cpu_percent=5, memory_percent=25, disk_limit_mb=1000),
        )
        gates.append({
            "name": "sub-one-core-share-uses-duty-cycle",
            "pass": limits["cpu_threads"] == 1 and 0 < limits["duty_cycle"] < 1,
        })
        evidence["five_percent_limits"] = limits

        environment = dict(os.environ)
        environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
        import_probe = subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys; import dendriswarm.cli; "
                "assert 'fastapi' not in sys.modules; assert 'sklearn' not in sys.modules",
            ],
            env=environment,
            capture_output=True,
            text=True,
        )
        gates.append({
            "name": "seed-runtime-does-not-import-coordinator-stack",
            "pass": import_probe.returncode == 0,
            "stderr": import_probe.stderr,
        })

    report = {
        "format": "dendriswarm.heterogeneous-seeding-proof.v1",
        "version": "0.4.0",
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "gates": gates,
        "all_gates_pass": all(gate["pass"] for gate in gates),
        "evidence": evidence,
        "claim_boundary": (
            "Demonstrates portable signed resource matching, low-resource CPU scouting, hot-reloaded "
            "owner policy, best-fit scheduling, lease renewal, and seed-only dependency separation. "
            "Resource limits remain application-level rather than OS-enforced."
        ),
    }
    output_path = Path(__file__).resolve().parents[1] / "docs" / "PROOF_RUN_V04.json"
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["all_gates_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
