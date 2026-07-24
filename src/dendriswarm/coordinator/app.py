from __future__ import annotations

import json
import math
import os
import secrets
import time
from pathlib import Path

from fastapi import FastAPI, Header, HTTPException, Response
from fastapi.responses import HTMLResponse

from dendriswarm.coordinator.service import CoordinatorService
from dendriswarm.core.limits import MAX_HTTP_BODY_BYTES
from dendriswarm.core.crypto import content_hash, public_key_fingerprint
from dendriswarm.core.models import (
    DatasetSubmission, InferenceRequest, LeverageCanaryRequest, LeverageSubmissionRequest,
    NodeRegistration, SeedPolicy, SignedNodeRequest, StrictModel, TaskAbandonRequest, TaskKind, TaskLeaseRequest, TaskResult,
)
from pydantic import Field


class CIFARPlanRequest(StrictModel):
    search_candidates: int = Field(default=8, ge=2, le=64)
    sample_budget: int = Field(default=640, ge=100, le=5000)


class CIFARQueueRequest(CIFARPlanRequest):
    optimizer_steps: int = Field(default=36, ge=1, le=500)
    learning_rate: float = Field(default=0.03, gt=0.0, le=1.0)
    verification_quorum: int = Field(default=2, ge=2, le=8)


class CIFARPrepareRequest(StrictModel):
    source: str = Field(min_length=1, max_length=4096)
    seed: int = 20260723
    holdout_per_class: int = Field(default=5, ge=1, le=100)
    replace: bool = False


class CIFARInitRequest(StrictModel):
    checkpoint_path: str | None = Field(default=None, max_length=4096)
    seed: int = 7
    replace: bool = False


class CIFAREvaluateRequest(StrictModel):
    output: str = Field(min_length=1, max_length=4096)
    source: str = Field(default="official-cifar100-test", min_length=1, max_length=256)


class BodyLimitMiddleware:
    """Reject oversized request bodies before Pydantic or endpoint parsing."""

    def __init__(self, app, max_bytes: int = MAX_HTTP_BODY_BYTES):
        self.app = app
        self.max_bytes = int(max_bytes)

    async def __call__(self, scope, receive, send):
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return
        headers = {key.lower(): value for key, value in scope.get("headers", [])}
        length = headers.get(b"content-length")
        try:
            declared_length = int(length) if length is not None else None
        except (TypeError, ValueError):
            await send({"type": "http.response.start", "status": 400, "headers": [(b"content-type", b"application/json")]})
            await send({"type": "http.response.body", "body": b'{"detail":"invalid content-length"}'})
            return
        if declared_length is not None and (declared_length < 0 or declared_length > self.max_bytes):
            await send({"type": "http.response.start", "status": 413, "headers": [(b"content-type", b"application/json")]})
            await send({"type": "http.response.body", "body": b'{"detail":"request body too large"}'})
            return
        messages = []
        total = 0
        while True:
            message = await receive()
            if message.get("type") == "http.disconnect":
                return
            body = message.get("body", b"")
            total += len(body)
            if total > self.max_bytes:
                await send({"type": "http.response.start", "status": 413, "headers": [(b"content-type", b"application/json")]})
                await send({"type": "http.response.body", "body": b'{"detail":"request body too large"}'})
                return
            messages.append(message)
            if not message.get("more_body", False):
                break
        iterator = iter(messages)
        async def replay_receive():
            try:
                return next(iterator)
            except StopIteration:
                return {"type": "http.request", "body": b"", "more_body": False}
        await self.app(scope, replay_receive, send)


def create_app(
    state_dir: str | Path | None = None,
    bootstrap: bool = False,
    lease_seconds: float = 60.0,
    inference_audit_rate: float = 0.20,
    enable_leverage: bool = False,
    leverage_canary_auditors: set[str] | None = None,
) -> FastAPI:
    state = Path(state_dir or os.getenv("DENDRISWARM_STATE", "./state")).resolve()
    service = CoordinatorService(state, lease_seconds=lease_seconds, inference_audit_rate=inference_audit_rate)
    app = FastAPI(title="DendriSwarm Coordinator", version="0.8.0")
    app.add_middleware(BodyLimitMiddleware, max_bytes=MAX_HTTP_BODY_BYTES)
    app.state.service = service
    admin_token_path = state / "keys" / "dashboard-admin-token"
    admin_token_path.parent.mkdir(parents=True, exist_ok=True)
    if admin_token_path.exists():
        app.state.dashboard_admin_token = admin_token_path.read_text(encoding="utf-8").strip()
    else:
        app.state.dashboard_admin_token = secrets.token_urlsafe(32)
        admin_token_path.write_text(app.state.dashboard_admin_token, encoding="utf-8")
        try:
            admin_token_path.chmod(0o600)
        except OSError:
            pass

    def require_dashboard_admin(x_dendriswarm_admin: str | None = Header(default=None)) -> None:
        supplied = x_dendriswarm_admin or ""
        if not secrets.compare_digest(supplied, app.state.dashboard_admin_token):
            raise HTTPException(401, "invalid dashboard admin token")

    configured_auditors = {
        item.strip()
        for item in os.getenv("DENDRISWARM_CANARY_AUDITORS", "").split(",")
        if item.strip()
    }
    if leverage_canary_auditors is not None:
        configured_auditors = set(leverage_canary_auditors)
    app.state.leverage_canary_auditors = configured_auditors

    leverage = None
    if enable_leverage:
        from dendriswarm.leverage.epoch import ChallengeEpoch
        from dendriswarm.leverage.service import LeverageService
        from dendriswarm.leverage.stats import GatePolicy
        from dendriswarm.leverage.workload import make_surrogate_workload, train_parent

        leverage_state = state / "leverage-state.json"
        if leverage_state.exists():
            leverage = LeverageService.load(leverage_state)
        else:
            leverage_workload = make_surrogate_workload(private_per_class=120)
            leverage_policy = GatePolicy()
            leverage = LeverageService(
                parent=train_parent(leverage_workload),
                epoch=ChallengeEpoch(
                    leverage_workload.x_private, leverage_workload.y_private,
                    leverage_workload.x_replication, leverage_workload.y_replication,
                    leverage_policy,
                ),
                policy=leverage_policy,
                state_path=leverage_state,
            )
        app.state.leverage = leverage

    if bootstrap and service.db.stats()["queued_tasks"] == 0 and service.db.stats()["candidates"] == 0:
        service.bootstrap()

    @app.get("/v1/meta")
    def meta():
        return {
            "name": "DendriSwarm",
            "version": "0.8.0",
            "coordinator_public_key": service.identity.public_key_b64,
            "coordinator_fingerprint": public_key_fingerprint(service.identity.public_key_b64),
            "policy": "approved-compute-only; evidence-gated promotion",
            "lease_seconds": service.lease_seconds,
            "verification_quorum": 2,
            "exploration_verification": "independent-replicated-consensus",
            "training_verification": "independent-artifact-consensus",
            "inference_verification": "strict-proof-carrying-schema-plus-secret-bonded-spot-audit",
            "locality_leverage_enabled": bool(enable_leverage),
            "worker_contract": "heterogeneous-resource-v1",
            "required_accelerator": None,
            "portable_backend": "numpy-cpu",
            "live_seed_policy": True,
            "native10_contribution_engine": "dendriswarm.native10-trainable.v6",
            "native10_legacy_engine": "dendriswarm.native10-derived.v5",
            "baseline_training_included": False,
        }

    @app.post("/v1/nodes/register")
    def register(registration: NodeRegistration):
        try:
            service.register_node(registration)
        except ValueError as exc:
            raise HTTPException(401, str(exc)) from exc
        return {
            "registered": True,
            "node_id": registration.node_id,
            "policy": (registration.policy or SeedPolicy(
                cpu_percent=100, memory_percent=100, disk_limit_mb=1_000_000,
                max_task_seconds=86400, allow_on_battery=True, max_system_cpu_percent=100,
            )).model_dump(mode="json"),
        }

    @app.post("/v1/nodes/heartbeat")
    def heartbeat(request: SignedNodeRequest):
        if not service.verify_node_request(request.node_id, request.timestamp, request.nonce, request.signature, "heartbeat"):
            raise HTTPException(401, "invalid, stale, or replayed node signature")
        service.db.heartbeat(request.node_id)
        return {"ok": True}

    @app.post("/v1/tasks/claim")
    def claim(request: SignedNodeRequest, response: Response):
        if not service.verify_node_request(request.node_id, request.timestamp, request.nonce, request.signature, "claim"):
            raise HTTPException(401, "invalid, stale, or replayed node signature")
        service.db.heartbeat(request.node_id)
        row = service.db.claim_task(request.node_id, service.lease_seconds)
        if not row:
            response.status_code = 204
            return None
        return service.signed_task(row)

    @app.post("/v1/tasks/renew")
    def renew_lease(request: TaskLeaseRequest):
        try:
            return service.renew_lease(request.model_dump(mode="json"))
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.post("/v1/tasks/abandon")
    def abandon_task(request: TaskAbandonRequest):
        try:
            return service.abandon_lease(request.model_dump(mode="json", exclude_unset=True))
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.post("/v1/tasks/result")
    def result(task_result: TaskResult):
        try:
            return service.process_result(task_result.model_dump(mode="json"))
        except (ValueError, KeyError, TypeError, IndexError) as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.post("/v1/datasets/submit")
    def submit_dataset(submission: DatasetSubmission):
        value = submission.model_dump(mode="json", exclude_unset=True)
        if not service.verify_signed_object(value, "dataset-submit"):
            raise HTTPException(401, "invalid, stale, or replayed dataset signature")
        if len(submission.features) != len(submission.labels) or not submission.features:
            raise HTTPException(400, "features and labels must be non-empty and aligned")
        if len(submission.features) > 10000:
            raise HTTPException(413, "testnet submissions are limited to 10,000 samples")
        width = len(submission.features[0])
        if width < 1 or width > 4096 or any(len(row) != width for row in submission.features):
            raise HTTPException(400, "feature rows must be rectangular and between 1 and 4096 values")
        if any(not math.isfinite(value) for row in submission.features for value in row):
            raise HTTPException(400, "features must contain only finite values")
        artifact = {
            "format": "dendriswarm.dataset-submission.v1",
            "name": submission.name,
            "source": submission.source,
            "license": submission.license,
            "description": submission.description,
            "features": submission.features,
            "labels": submission.labels,
            "feature_width": width,
        }
        artifact["sha256"] = content_hash(artifact)
        dataset_id = service.db.add_dataset(artifact, submission.node_id, status="pending")
        return {"dataset_id": dataset_id, "content_hash": artifact["sha256"], "status": "pending"}

    @app.get("/v1/datasets/{value_hash}")
    def dataset_artifact(value_hash: str):
        row = service.db.dataset_by_hash(value_hash, approved_only=True)
        if not row:
            raise HTTPException(404, "unknown approved dataset")
        return json.loads(row["artifact"])

    @app.get("/v1/artifacts/{value_hash}")
    def tissue_artifact(value_hash: str):
        row = service.db.candidate_by_hash(value_hash)
        if not row:
            raise HTTPException(404, "unknown tissue artifact")
        return json.loads(row["artifact"])

    @app.get("/v1/samples/digit/{index}")
    def digit_sample(index: int):
        row = service.db.approved_dataset()
        if not row:
            raise HTTPException(404, "reference dataset is not available")
        dataset = json.loads(row["artifact"])
        test_indices = dataset["splits"]["test"]
        if index < 0 or index >= len(test_indices):
            raise HTTPException(404, "sample index is outside the test split")
        source_index = int(test_indices[index])
        return {"index": index, "source_index": source_index,
                "features": dataset["features"][source_index],
                "label": dataset["labels"][source_index]}

    @app.post("/v1/inference")
    def enqueue_inference(request: InferenceRequest):
        try:
            return service.enqueue_inference(request.model_dump(mode="json"))
        except ValueError as exc:
            message = str(exc)
            status = 402 if "insufficient credits" in message else 409 if "no canonical" in message else 401 if "signature" in message else 400
            raise HTTPException(status, message) from exc

    @app.get("/v1/jobs/{task_id}")
    def job(task_id: str):
        row = service.db.task(task_id)
        if not row:
            raise HTTPException(404, "unknown task")
        return {
            "id": row["id"],
            "kind": row["kind"],
            "status": row["status"],
            "assigned_to": row["assigned_to"],
            "attempts": row["attempts"],
            "requirements": json.loads(row["requirements"] or "{}"),
            "lease_expires_at": row["lease_expires_at"],
            "output": None if not row["output"] else json.loads(row["output"]),
        }

    @app.get("/v1/stats")
    def stats():
        value = service.db.stats()
        value["native10"] = service.native10.store.status()
        native10_v6_status = service.native10_v6.store.status()
        value["native10_v6"] = native10_v6_status
        value["cifar100_campaign"] = service.cifar100.status(
            native_status=native10_v6_status
        )
        return value

    @app.get("/v1/native10/status")
    def native10_status():
        return service.native10.store.status()

    @app.get("/v1/native10/checkpoints/{model_root}")
    def native10_checkpoint(model_root: str):
        if len(model_root) != 64:
            raise HTTPException(404, "unknown Native10 checkpoint")
        try:
            artifact = json.loads(service.native10.store.checkpoint_path.read_text())
            model = service.native10.store.model()
        except (FileNotFoundError, ValueError, json.JSONDecodeError):
            raise HTTPException(404, "Native10 checkpoint is not initialized")
        if model.root != model_root or artifact.get("model_root") != model_root:
            raise HTTPException(404, "unknown Native10 checkpoint")
        return artifact

    @app.post("/v1/native10/validation/{validation_hash}")
    def native10_validation(validation_hash: str, request: TaskLeaseRequest):
        value = request.model_dump(mode="json")
        if not service.verify_signed_object(value, "fetch-native10-validation"):
            raise HTTPException(401, "invalid, stale, or replayed validation fetch signature")
        task = service.db.task(request.task_id)
        now = time.time()
        if (
            not task
            or task["kind"] != TaskKind.DENDRITRON_VERIFICATION.value
            or task["status"] != "assigned"
            or task["assigned_to"] != request.node_id
            or task["lease_token"] != request.lease_token
            or float(task["lease_expires_at"] or 0) < now
            or float(task["lease_deadline_at"] or 0) < now
        ):
            raise HTTPException(403, "validation is available only to the active assigned verifier")
        payload = json.loads(task["payload"])
        if payload.get("global_validation_hash") != validation_hash:
            raise HTTPException(403, "verification task is not authorized for this validation artifact")
        try:
            artifact = service.native10.store.global_validation()
        except (FileNotFoundError, ValueError, json.JSONDecodeError) as error:
            raise HTTPException(404, "global Native10 validation is not configured") from error
        if artifact.get("sha256") != validation_hash:
            raise HTTPException(404, "unknown global Native10 validation artifact")
        service.db.append_audit("native10_global_validation_issued", {
            "validation_hash": validation_hash,
            "task_id": request.task_id,
            "verifier_node": request.node_id,
        })
        return artifact

    @app.get("/v1/native10/contributions")
    def native10_contributions():
        state_value = service.native10.store.state()
        return {
            "canonical_root": state_value.get("canonical_root"),
            "contributions": state_value.get("contributions", []),
        }

    @app.get("/v1/cifar100/status")
    def cifar100_status():
        return service.cifar100.status()

    @app.post("/v1/admin/cifar100/plan")
    def cifar100_admin_plan(request: CIFARPlanRequest, admin_token: str | None = Header(default=None, alias="X-DendriSwarm-Admin")):
        require_dashboard_admin(admin_token)
        try:
            return service.cifar100.plan_next(
                search_candidates=request.search_candidates, sample_budget=request.sample_budget
            )
        except (ValueError, OSError) as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.post("/v1/admin/cifar100/queue-next")
    def cifar100_admin_queue(request: CIFARQueueRequest, admin_token: str | None = Header(default=None, alias="X-DendriSwarm-Admin")):
        require_dashboard_admin(admin_token)
        try:
            return service.cifar100.queue_next(
                search_candidates=request.search_candidates, sample_budget=request.sample_budget,
                optimizer_steps=request.optimizer_steps, learning_rate=request.learning_rate,
                verification_quorum=request.verification_quorum,
            )
        except (ValueError, OSError) as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.post("/v1/admin/cifar100/prepare")
    def cifar100_admin_prepare(request: CIFARPrepareRequest, admin_token: str | None = Header(default=None, alias="X-DendriSwarm-Admin")):
        require_dashboard_admin(admin_token)
        source = Path(request.source).expanduser().resolve()
        if not source.exists():
            raise HTTPException(400, "CIFAR-100 source path does not exist")
        try:
            return service.cifar100.prepare_dataset(
                source, seed=request.seed, holdout_per_class=request.holdout_per_class, replace=request.replace
            )
        except (ValueError, OSError) as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.post("/v1/admin/cifar100/init")
    def cifar100_admin_init(request: CIFARInitRequest, admin_token: str | None = Header(default=None, alias="X-DendriSwarm-Admin")):
        require_dashboard_admin(admin_token)
        checkpoint = None
        if request.checkpoint_path:
            path = Path(request.checkpoint_path).expanduser().resolve()
            if not path.is_file():
                raise HTTPException(400, "checkpoint path does not exist")
            try:
                checkpoint = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                raise HTTPException(400, f"invalid checkpoint: {exc}") from exc
        try:
            return service.cifar100.initialize_model(seed=request.seed, checkpoint=checkpoint, replace=request.replace)
        except (ValueError, OSError) as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.post("/v1/admin/cifar100/evaluate-test")
    def cifar100_admin_evaluate(request: CIFAREvaluateRequest, admin_token: str | None = Header(default=None, alias="X-DendriSwarm-Admin")):
        require_dashboard_admin(admin_token)
        try:
            report = service.cifar100.evaluate_test(source=request.source)
            output = Path(request.output).expanduser().resolve()
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(report, sort_keys=True), encoding="utf-8")
            return {"written": str(output), **report}
        except (ValueError, OSError) as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.get("/v1/native10-v6/status")
    def native10_v6_status():
        return service.native10_v6.store.status()

    @app.get("/v1/native10-v6/checkpoints/{model_root}")
    def native10_v6_checkpoint(model_root: str):
        if len(model_root) != 64:
            raise HTTPException(404, "unknown Native10 v0.6 checkpoint")
        try:
            artifact = json.loads(service.native10_v6.store.checkpoint_path.read_text())
            model = service.native10_v6.store.model()
        except (FileNotFoundError, ValueError, json.JSONDecodeError):
            raise HTTPException(404, "Native10 v0.6 checkpoint is not initialized")
        if model.root != model_root or artifact.get("model_root") != model_root:
            raise HTTPException(404, "unknown Native10 v0.6 checkpoint")
        return artifact

    @app.post("/v1/native10-v6/validation/{validation_hash}")
    def native10_v6_validation(validation_hash: str, request: TaskLeaseRequest):
        value = request.model_dump(mode="json")
        if not service.verify_signed_object(value, "fetch-native10-validation"):
            raise HTTPException(401, "invalid, stale, or replayed validation fetch signature")
        task = service.db.task(request.task_id)
        now = time.time()
        if (
            not task or task["kind"] != TaskKind.DENDRITRON_VERIFICATION.value
            or task["status"] != "assigned" or task["assigned_to"] != request.node_id
            or task["lease_token"] != request.lease_token
            or float(task["lease_expires_at"] or 0) < now or float(task["lease_deadline_at"] or 0) < now
        ):
            raise HTTPException(403, "validation is available only to the active assigned verifier")
        payload = json.loads(task["payload"])
        if payload.get("engine") != "dendriswarm.native10-trainable.v6" or payload.get("global_validation_hash") != validation_hash:
            raise HTTPException(403, "verification task is not authorized for this validation artifact")
        try:
            artifact = service.native10_v6.store.validation_by_hash(validation_hash)
        except (FileNotFoundError, ValueError, json.JSONDecodeError) as error:
            raise HTTPException(404, "unknown Native10 v0.6 validation artifact") from error
        service.db.append_audit("native10_v6_global_validation_issued", {
            "validation_hash": validation_hash, "task_id": request.task_id, "verifier_node": request.node_id,
        })
        return artifact

    @app.get("/v1/native10-v6/contributions")
    def native10_v6_contributions():
        state_value = service.native10_v6.store.state()
        return {"canonical_root": state_value.get("canonical_root"), "contributions": state_value.get("contributions", [])}

    @app.get("/v1/audit/checkpoint")
    def audit_checkpoint():
        return service.audit_checkpoint()

    @app.get("/v1/nodes/{node_id}")
    def node_account(node_id: str):
        row = service.db.node(node_id)
        if not row:
            raise HTTPException(404, "unknown node")
        ledger = [dict(item) for item in service.db.ledger_entries(node_id, 50)]
        return {
            "node_id": node_id,
            "credit_units": int(row["credit_units"]),
            "credits": int(row["credit_units"]) / 1000.0,
            "completed": row["completed"],
            "failed": row["failed"],
            "contributed_ms": int(row["contributed_ms"]),
            "contributed_seconds": int(row["contributed_ms"]) / 1000.0,
            "contributed_hours": int(row["contributed_ms"]) / 3_600_000.0,
            "capabilities": json.loads(row["capabilities"]),
            "policy": json.loads(row["policy"] or "{}"),
            "last_seen": row["last_seen"],
            "ledger": ledger,
        }

    @app.get("/v1/leverage/meta")
    def leverage_meta():
        if leverage is None:
            raise HTTPException(404, "locality leverage engine is disabled")
        return {
            "format": "dendriswarm.leverage-meta.v3.2",
            "epoch_commitment": leverage.epoch.commitment,
            "policy": leverage.policy.as_dict(),
            "policy_hash": leverage.policy.registration_hash,
            "canonical_root": leverage.canonical_root,
            "active_canary": leverage.active_canary_id,
            "registration_grant_units": 0,
            "funding_assumption": leverage.policy.identity_assumption,
            "final_replication_required": True,
            "canary_observer_mode": "authorized-registered-node; candidate self-audit forbidden",
            "authorized_canary_auditors": len(app.state.leverage_canary_auditors),
        }

    @app.post("/v1/leverage/candidates")
    def leverage_submit(request: LeverageSubmissionRequest):
        value = request.model_dump(mode="json", exclude_unset=True)
        if leverage is None:
            raise HTTPException(404, "locality leverage engine is disabled")
        if not service.verify_signed_object(value, "leverage-submit"):
            raise HTTPException(401, "invalid, stale, or replayed leverage submission signature")
        try:
            from dendriswarm.leverage.manifest import CandidateManifest

            manifest = CandidateManifest.from_dict(value["manifest"])
            leverage.register_contributor(value["node_id"])
            candidate_id, verdict = leverage.submit(manifest, value["node_id"])
        except (ValueError, KeyError, TypeError) as exc:
            raise HTTPException(400, str(exc)) from exc
        return {"candidate_id": candidate_id, "verdict": verdict}

    @app.get("/v1/leverage/candidates/{candidate_id}")
    def leverage_candidate(candidate_id: str):
        if leverage is None:
            raise HTTPException(404, "locality leverage engine is disabled")
        record = leverage.candidates.get(candidate_id)
        if not record:
            raise HTTPException(404, "unknown leverage candidate")
        # Exact challenge statistics remain private until epoch disclosure.
        return {
            "candidate_id": candidate_id,
            "verdict": record.verdict,
            "status": record.status,
            "parent_root": record.parent_root,
            "candidate_root": record.candidate_root,
            "canary_events": record.canary_events,
        }


    @app.post("/v1/leverage/candidates/{candidate_id}/canary")
    def leverage_canary(candidate_id: str, request: LeverageCanaryRequest):
        value = request.model_dump(mode="json", exclude_unset=True)
        if leverage is None:
            raise HTTPException(404, "locality leverage engine is disabled")
        if not service.verify_signed_object(value, "leverage-canary-batch"):
            raise HTTPException(401, "invalid, stale, or replayed canary batch signature")
        observer_id = str(value.get("node_id", ""))
        if observer_id not in app.state.leverage_canary_auditors:
            raise HTTPException(403, "node is not an authorized canary auditor")
        record = leverage.candidates.get(candidate_id)
        if record is not None and record.manifest.contributor == observer_id:
            raise HTTPException(403, "candidate contributor cannot audit its own canary")
        try:
            status = leverage.record_canary_batch(
                candidate_id,
                value["features"],
                value["labels"],
                source_id=str(value["source_id"]),
                source_kind=str(value["source_kind"]),
                subgroup_ids=value.get("subgroup_ids"),
            )
        except (ValueError, KeyError, TypeError) as exc:
            raise HTTPException(400, str(exc)) from exc
        record = leverage.candidates[candidate_id]
        return {
            "candidate_id": candidate_id,
            "status": status,
            "observation": record.canary_observations[-1],
        }

    @app.get("/v1/leverage/audit/checkpoint")
    def leverage_audit_checkpoint():
        if leverage is None:
            raise HTTPException(404, "locality leverage engine is disabled")
        checkpoint = leverage.audit_checkpoint()
        return {
            "checkpoint": checkpoint,
            "coordinator_public_key": service.identity.public_key_b64,
            "signature": service.identity.sign(checkpoint),
        }

    @app.get("/", response_class=HTMLResponse)
    def dashboard():
        return """
<!doctype html><html><head><meta charset='utf-8'><title>DendriSwarm v0.8.0</title>
<style>body{font-family:system-ui;background:#0b1020;color:#e8ecff;max-width:1080px;margin:40px auto;padding:0 20px}pre{background:#151c33;padding:20px;border-radius:14px;overflow:auto}.card{background:#11182c;padding:24px;border-radius:18px;border:1px solid #26304d}h1{font-size:42px;margin-bottom:4px}.muted{color:#9da8ca}.warn{color:#ffd58a}</style></head>
<body><h1>DendriSwarm v0.8.0</h1><p class='muted'>Portable volunteer workers receive only tasks that fit their architecture and user-selected resource budget.</p><p class='warn'>Reference testnet: public data is non-sensitive; workers cannot submit executable code; verified tissue deltas are the only canonical mutation path.</p><div class='card'><h2>Live evidence</h2><pre id='stats'>Loading…</pre></div>
<script>async function tick(){let r=await fetch('/v1/stats');document.getElementById('stats').textContent=JSON.stringify(await r.json(),null,2)}tick();setInterval(tick,2000)</script></body></html>
"""

    return app
