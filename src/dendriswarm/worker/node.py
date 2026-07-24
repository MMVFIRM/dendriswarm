from __future__ import annotations

import json
import os
import tempfile
import threading
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx

from dendriswarm.core.crypto import Identity, content_hash, nonce, public_key_fingerprint, verify
from dendriswarm.core.limits import MAX_CONTROL_REQUEST_BYTES
from dendriswarm.core.models import NodeCapabilities, SeedPolicy, TaskKind, TaskRequirements
from dendriswarm.core.resources import (
    contract_covers,
    derive_payload_requirements,
    effective_limits,
    node_can_run,
)
from dendriswarm.tissues.reference import artifact_hash, dataset_hash
from dendriswarm.worker.config import SeedPolicyStore
from dendriswarm.worker.isolation import TaskExecutionCancelled, execute_task_isolated
from dendriswarm.worker.resources import cooldown_seconds, detect_capabilities, local_run_condition, refresh_dynamic_resources

MAX_CONTROL_RESPONSE_BYTES = 8 * 1024 * 1024
MAX_JSON_DEPTH = 32
MAX_JSON_NODES = 2_000_000


class SeedNode:
    """Portable hostile-network volunteer seed with locally enforced budgets."""

    def __init__(
        self,
        coordinator: str,
        state_dir: Path,
        poll_seconds: float = 1.0,
        *,
        expected_coordinator_fingerprint: str | None = None,
        allow_insecure_http: bool = False,
    ):
        self.coordinator = coordinator.rstrip("/")
        parsed = urlparse(self.coordinator)
        host = (parsed.hostname or "").lower()
        local = host in {"localhost", "127.0.0.1", "::1"}
        if parsed.scheme not in {"http", "https"}:
            raise ValueError("coordinator URL must use HTTP or HTTPS")
        if parsed.scheme != "https" and not local and not allow_insecure_http:
            raise ValueError("remote coordinators require HTTPS; pass --allow-insecure-http only for an explicitly trusted test network")
        self.expected_coordinator_fingerprint = (
            expected_coordinator_fingerprint.lower().replace(":", "")
            if expected_coordinator_fingerprint else None
        )
        self.state_dir = state_dir
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.identity = Identity.load_or_create(state_dir / "keys")
        self.poll_seconds = poll_seconds
        self.client = httpx.Client(timeout=httpx.Timeout(30.0, read=120.0), follow_redirects=False)
        self.coordinator_public_key = ""
        self.cache_dir = state_dir / "cache"
        self.cache_dir.mkdir(exist_ok=True)
        self.outbox_dir = state_dir / "outbox"
        self.outbox_dir.mkdir(exist_ok=True)
        self.receipts_dir = state_dir / "receipts"
        self.receipts_dir.mkdir(exist_ok=True)
        self.policy_store = SeedPolicyStore(state_dir / "seed-config.json")
        self.status_path = state_dir / "seed-status.json"
        self.policy = self.policy_store.load()
        self._capabilities = detect_capabilities(self.state_dir, self.policy)
        self._registered_policy_hash = ""
        self._last_heartbeat = 0.0
        self._last_registration = 0.0
        self._last_status_write = 0.0
        self._last_status_key = ""
        self._last_receipt: dict[str, Any] | None = None

    def capabilities(self) -> NodeCapabilities:
        return self._capabilities

    def _write_json_atomic(self, path: Path, value: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass

    @staticmethod
    def _validate_json_shape(value: Any, max_depth: int = MAX_JSON_DEPTH, max_nodes: int = MAX_JSON_NODES) -> None:
        stack: list[tuple[Any, int]] = [(value, 0)]
        nodes = 0
        while stack:
            current, depth = stack.pop()
            nodes += 1
            if nodes > max_nodes:
                raise RuntimeError("JSON structure exceeds node limit")
            if depth > max_depth:
                raise RuntimeError("JSON structure exceeds depth limit")
            if isinstance(current, dict):
                if len(current) > 100_000:
                    raise RuntimeError("JSON object exceeds key limit")
                stack.extend((item, depth + 1) for item in current.values())
            elif isinstance(current, list):
                stack.extend((item, depth + 1) for item in current)

    def _request_json(
        self,
        method: str,
        url: str,
        *,
        max_bytes: int = MAX_CONTROL_RESPONSE_BYTES,
        body: dict[str, Any] | None = None,
        max_request_bytes: int = MAX_CONTROL_REQUEST_BYTES,
    ) -> tuple[int, dict[str, Any] | None]:
        if body is not None:
            encoded_request = json.dumps(body, separators=(",", ":"), allow_nan=False).encode("utf-8")
            if len(encoded_request) > max_request_bytes:
                raise RuntimeError("outgoing request exceeds local byte limit")
        with self.client.stream(method, url, json=body) as response:
            content_length = response.headers.get("content-length")
            if content_length is not None:
                try:
                    declared_length = int(content_length)
                except (TypeError, ValueError) as error:
                    raise RuntimeError("coordinator returned an invalid Content-Length") from error
                if declared_length < 0 or declared_length > max_bytes:
                    raise RuntimeError("coordinator response exceeds declared byte limit")
            raw = bytearray()
            for chunk in response.iter_bytes():
                raw.extend(chunk)
                if len(raw) > max_bytes:
                    raise RuntimeError("coordinator response exceeded byte limit while streaming")
            if response.status_code == 204:
                return response.status_code, None
            if response.status_code >= 400:
                message = raw.decode("utf-8", errors="replace")[:1000]
                raise httpx.HTTPStatusError(message, request=response.request, response=response)
            value = json.loads(raw or b"{}")
            self._validate_json_shape(value)
            if not isinstance(value, dict):
                raise RuntimeError("coordinator JSON response must be an object")
            return response.status_code, value

    def _status(self, **updates: Any) -> None:
        if updates.get("state") not in {"error", "offline"} and "last_error" not in updates:
            updates["last_error"] = None
        status_key = json.dumps({
            "state": updates.get("state"),
            "reason": updates.get("reason"),
            "current_task": updates.get("current_task"),
        }, sort_keys=True, default=str)
        if status_key == self._last_status_key and time.time() - self._last_status_write < 5.0:
            return
        current: dict[str, Any] = {}
        if self.status_path.exists():
            try:
                current = json.loads(self.status_path.read_text())
            except Exception:
                current = {}
        current.update({
            "format": "dendriswarm.seed-status.v5",
            "pid": os.getpid(),
            "node_id": self.identity.node_id,
            "coordinator": self.coordinator,
            "updated_at": time.time(),
            "policy": self.policy.model_dump(mode="json"),
            "capabilities": self._capabilities.model_dump(mode="json"),
            "effective_limits": effective_limits(self._capabilities, self.policy),
            "outbox_pending": len(list(self.outbox_dir.glob("*.json"))),
            **updates,
        })
        self._write_json_atomic(self.status_path, current)
        self._last_status_key = status_key
        self._last_status_write = time.time()

    def _pin_coordinator(self, public_key: str) -> None:
        fingerprint = public_key_fingerprint(public_key)
        if self.expected_coordinator_fingerprint and fingerprint != self.expected_coordinator_fingerprint:
            raise RuntimeError("coordinator fingerprint does not match the out-of-band expected fingerprint")
        path = self.state_dir / "coordinator.json"
        if path.exists():
            trusted = json.loads(path.read_text())
            if trusted.get("public_key") != public_key or trusted.get("fingerprint", fingerprint) != fingerprint:
                raise RuntimeError("coordinator identity changed; verify the new fingerprint before replacing the pin")
        else:
            self._write_json_atomic(path, {
                "public_key": public_key,
                "fingerprint": fingerprint,
                "coordinator": self.coordinator,
                "trust_mode": "out-of-band-pin" if self.expected_coordinator_fingerprint else "tls-tofu",
            })
        self.coordinator_public_key = public_key

    def _reload_policy(self) -> bool:
        updated = self.policy_store.load()
        if updated == self.policy:
            return False
        self.policy = updated
        self._capabilities = detect_capabilities(self.state_dir, self.policy)
        self._status(state="policy-updated", reason="hot-reload")
        return True

    def register(self, force: bool = False) -> None:
        _, meta = self._request_json("GET", f"{self.coordinator}/v1/meta")
        assert meta is not None
        self._pin_coordinator(str(meta["coordinator_public_key"]))
        advertised = str(meta.get("coordinator_fingerprint", public_key_fingerprint(self.coordinator_public_key)))
        if advertised != public_key_fingerprint(self.coordinator_public_key):
            raise RuntimeError("coordinator meta fingerprint is inconsistent with its public key")
        policy_hash = content_hash(self.policy.model_dump(mode="json"))
        if not force and policy_hash == self._registered_policy_hash:
            return
        value: dict[str, Any] = {
            "node_id": self.identity.node_id,
            "public_key": self.identity.public_key_b64,
            "capabilities": self.capabilities().model_dump(mode="json"),
            "policy": self.policy.model_dump(mode="json"),
            "timestamp": int(time.time()),
            "nonce": nonce(),
        }
        value["signature"] = self.identity.sign(value)
        self._request_json("POST", f"{self.coordinator}/v1/nodes/register", body=value)
        self._registered_policy_hash = policy_hash
        self._last_registration = time.time()
        self._write_json_atomic(self.state_dir / "node.json", {"node_id": self.identity.node_id})
        self._status(state="registered", reason="ready")

    def signed_request(self, action: str, **extra: object) -> dict[str, object]:
        value: dict[str, object] = {
            "action": action,
            "node_id": self.identity.node_id,
            **extra,
            "timestamp": int(time.time()),
            "nonce": nonce(),
        }
        return {k: v for k, v in value.items() if k != "action"} | {"signature": self.identity.sign(value)}

    def _heartbeat(self) -> None:
        if time.time() - self._last_heartbeat < 20:
            return
        self._request_json("POST", f"{self.coordinator}/v1/nodes/heartbeat", body=self.signed_request("heartbeat"))
        self._last_heartbeat = time.time()

    def _make_cache_room(self, incoming_bytes: int, target: Path) -> None:
        disk_budget = int(effective_limits(self._capabilities, self.policy)["disk_mb"]) * 1024 * 1024
        if incoming_bytes > disk_budget:
            raise RuntimeError("artifact exceeds configured seed cache budget")
        files = [item for item in self.cache_dir.glob("*.json") if item != target]
        usage = sum(item.stat().st_size for item in files)
        for item in sorted(files, key=lambda value: value.stat().st_mtime):
            if usage + incoming_bytes <= disk_budget:
                break
            size = item.stat().st_size
            item.unlink(missing_ok=True)
            usage -= size
        if usage + incoming_bytes > disk_budget:
            raise RuntimeError("seed cache budget cannot accommodate artifact")

    def _fetch_cached(self, kind: str, value_hash: str, max_bytes: int) -> dict[str, Any]:
        path = self.cache_dir / f"{kind}-{value_hash}.json"
        if path.exists():
            if path.stat().st_size > max_bytes:
                path.unlink(missing_ok=True)
                raise RuntimeError("cached artifact exceeds signed resource contract")
            value = json.loads(path.read_text())
            self._validate_json_shape(value)
        else:
            endpoint = {
                "dataset": "datasets",
                "artifact": "artifacts",
                "native10-checkpoint": "native10/checkpoints",
                "native10-v6-checkpoint": "native10-v6/checkpoints",
            }.get(kind)
            if endpoint is None:
                raise RuntimeError(f"unsupported cached artifact kind: {kind}")
            _, value = self._request_json(
                "GET", f"{self.coordinator}/v1/{endpoint}/{value_hash}", max_bytes=max_bytes
            )
            assert value is not None
            encoded = json.dumps(value, separators=(",", ":"), allow_nan=False).encode("utf-8")
            self._make_cache_room(len(encoded), path)
            self._write_json_atomic(path, value)
        if kind == "dataset":
            actual = dataset_hash(value)
            valid = actual == value_hash and value.get("sha256") == value_hash
        elif kind == "artifact":
            actual = artifact_hash(value)
            valid = actual == value_hash and value.get("sha256") == value_hash
        else:
            if kind == "native10-v6-checkpoint":
                from dendriswarm.v6.native10 import Native10Dendritron
            else:
                from dendriswarm.v5.native10 import Native10Dendritron
            actual = Native10Dendritron.from_artifact(value).root
            valid = actual == value_hash and value.get("model_root") == value_hash
        if not valid:
            path.unlink(missing_ok=True)
            raise RuntimeError(f"{kind} content hash mismatch")
        return value

    def _fetch_private_native10_validation(
        self, validation_hash: str, task: dict[str, Any], max_bytes: int, *, engine: str
    ) -> dict[str, Any]:
        if task.get("kind") != TaskKind.DENDRITRON_VERIFICATION.value:
            raise RuntimeError("only a verification task may request global validation")
        _, value = self._request_json(
            "POST",
            f"{self.coordinator}/v1/{'native10-v6' if engine == 'dendriswarm.native10-trainable.v6' else 'native10'}/validation/{validation_hash}",
            max_bytes=max_bytes,
            body=self.signed_request(
                "fetch-native10-validation",
                task_id=task["id"],
                lease_token=task["lease_token"],
            ),
        )
        assert value is not None
        if engine == "dendriswarm.native10-trainable.v6":
            from dendriswarm.v6.validation import decode_global_validation_artifact
        else:
            from dendriswarm.v5.validation import decode_global_validation_artifact
        decode_global_validation_artifact(value)
        if value.get("sha256") != validation_hash:
            raise RuntimeError("global validation content hash mismatch")
        return value

    def materialize_payload(
        self, payload: dict[str, Any], max_bytes: int, *, task: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        materialized = dict(payload)
        remaining = int(max_bytes)
        if "dataset_hash" in payload:
            dataset = self._fetch_cached("dataset", str(payload["dataset_hash"]), remaining)
            dataset_bytes = len(json.dumps(dataset, separators=(",", ":"), allow_nan=False).encode("utf-8"))
            remaining -= dataset_bytes
            if remaining < 0:
                raise RuntimeError("materialized task artifacts exceed the signed total byte budget")
            materialized["_dataset"] = dataset
        if "artifact_hash" in payload:
            if remaining < 1024:
                raise RuntimeError("signed total artifact budget leaves no room for the model artifact")
            artifact = self._fetch_cached("artifact", str(payload["artifact_hash"]), remaining)
            artifact_bytes = len(json.dumps(artifact, separators=(",", ":"), allow_nan=False).encode("utf-8"))
            remaining -= artifact_bytes
            if remaining < 0:
                raise RuntimeError("materialized task artifacts exceed the signed total byte budget")
            materialized["_artifact"] = artifact
        if "native10_checkpoint_root" in payload:
            if remaining < 1024:
                raise RuntimeError("signed artifact budget leaves no room for the Native10 checkpoint")
            checkpoint_kind = "native10-v6-checkpoint" if payload.get("engine") == "dendriswarm.native10-trainable.v6" else "native10-checkpoint"
            checkpoint = self._fetch_cached(
                checkpoint_kind, str(payload["native10_checkpoint_root"]), remaining
            )
            checkpoint_bytes = len(json.dumps(checkpoint, separators=(",", ":"), allow_nan=False).encode("utf-8"))
            remaining -= checkpoint_bytes
            if remaining < 0:
                raise RuntimeError("materialized task artifacts exceed the signed total byte budget")
            materialized["_native10_checkpoint"] = checkpoint
        if "global_validation_hash" in payload and task is not None and task.get("kind") == TaskKind.DENDRITRON_VERIFICATION.value:
            if task is None:
                raise RuntimeError("private global validation requires the signed task envelope")
            if remaining < 1024:
                raise RuntimeError("signed artifact budget leaves no room for global validation")
            validation = self._fetch_private_native10_validation(
                str(payload["global_validation_hash"]), task, remaining, engine=str(payload.get("engine", ""))
            )
            validation_bytes = len(json.dumps(validation, separators=(",", ":"), allow_nan=False).encode("utf-8"))
            remaining -= validation_bytes
            if remaining < 0:
                raise RuntimeError("materialized task artifacts exceed the signed total byte budget")
            materialized["_native10_validation"] = validation
        return materialized

    def _lease_renewal_loop(self, task: dict[str, Any], stop: threading.Event) -> None:
        while not stop.wait(max(5.0, min(30.0, (task["lease_expires_at"] - time.time()) / 3.0))):
            if time.time() >= float(task["lease_deadline_at"]):
                return
            try:
                _, result = self._request_json(
                    "POST", f"{self.coordinator}/v1/tasks/renew",
                    body=self.signed_request(
                        "renew-lease", task_id=task["id"], lease_token=task["lease_token"]
                    ),
                )
                assert result is not None
                task["lease_expires_at"] = float(result["lease_expires_at"])
            except Exception:
                return

    def _execute_signed_task(self, task: dict[str, Any]) -> tuple[dict[str, Any], int, float]:
        if time.time() >= float(task["lease_expires_at"]) or time.time() >= float(task["lease_deadline_at"]):
            raise TaskExecutionCancelled("signed task envelope is already expired")
        requirements = TaskRequirements.model_validate(task.get("requirements") or {})
        eligible, reason = node_can_run(TaskKind(task["kind"]), requirements, self._capabilities, self.policy)
        if not eligible:
            raise TaskExecutionCancelled(f"coordinator assigned task outside local policy: {reason}")
        limits = effective_limits(self._capabilities, self.policy)
        materialized = self.materialize_payload(
            task["payload"], requirements.max_artifact_bytes, task=task
        )
        derived = derive_payload_requirements(TaskKind(task["kind"]), materialized)
        covered, coverage_reason = contract_covers(requirements, derived)
        if not covered:
            raise TaskExecutionCancelled(f"signed resource contract is understated: {coverage_reason}")
        threads = min(int(limits["cpu_threads"]), requirements.preferred_cpu_threads)
        memory_limit = min(int(limits["memory_mb"]), requirements.max_memory_mb)
        timeout = min(self.policy.max_task_seconds, requirements.hard_timeout_seconds)
        self._status(
            state="working", reason="task-active",
            current_task={
                "id": task["id"], "kind": task["kind"],
                "resource_class": requirements.resource_class.value,
                "cpu_threads": threads, "memory_limit_mb": memory_limit,
                "hard_timeout_seconds": timeout, "started_at": time.time(),
            },
        )
        stop = threading.Event()
        renewer = threading.Thread(target=self._lease_renewal_loop, args=(task, stop), daemon=True)
        renewer.start()
        started = time.perf_counter()
        try:
            output = execute_task_isolated(
                TaskKind(task["kind"]), materialized,
                cpu_threads=threads,
                memory_limit_mb=max(32, memory_limit),
                timeout_seconds=max(5, timeout),
                requirements=requirements,
                capabilities=self._capabilities,
                policy_store=self.policy_store,
            )
        finally:
            stop.set()
            renewer.join(timeout=2.0)
        duration = max(time.perf_counter() - started, 0.001)
        return output, max(1, int(duration * 1000)), duration

    def _queue_result(self, result_body: dict[str, Any]) -> Path:
        path = self.outbox_dir / f"{result_body['task_id']}.json"
        self._write_json_atomic(path, result_body)
        return path

    def _flush_outbox(self) -> int:
        delivered = 0
        for path in sorted(self.outbox_dir.glob("*.json")):
            body = json.loads(path.read_text())
            try:
                _, response = self._request_json(
                    "POST", f"{self.coordinator}/v1/tasks/result",
                    body=body, max_bytes=MAX_CONTROL_RESPONSE_BYTES,
                )
            except httpx.HTTPStatusError as error:
                status = error.response.status_code
                if 400 <= status < 500 and status not in {408, 409, 425, 429}:
                    self._write_json_atomic(
                        self.receipts_dir / f"{body['task_id']}-rejected.json",
                        {"accepted": False, "status_code": status, "detail": str(error)},
                    )
                    path.unlink(missing_ok=True)
                    continue
                raise
            assert response is not None
            receipt = response.get("receipt")
            if receipt is not None:
                if response.get("coordinator_public_key") != self.coordinator_public_key:
                    raise RuntimeError("result receipt was signed by another coordinator")
                if not verify(self.coordinator_public_key, receipt, str(response.get("signature", ""))):
                    raise RuntimeError("invalid coordinator work receipt")
            self._write_json_atomic(self.receipts_dir / f"{body['task_id']}.json", response)
            self._last_receipt = response
            path.unlink(missing_ok=True)
            delivered += 1
        return delivered

    def _abandon(self, task: dict[str, Any], reason: str) -> None:
        try:
            self._request_json(
                "POST", f"{self.coordinator}/v1/tasks/abandon",
                body=self.signed_request(
                    "abandon-task", task_id=task["id"], lease_token=task["lease_token"], reason=reason[:160]
                ),
            )
        except Exception:
            pass

    def _interruptible_cooldown(self, duration: float) -> None:
        deadline = time.time() + duration
        while time.time() < deadline:
            before = self.policy
            time.sleep(min(1.0, max(0.0, deadline - time.time())))
            if self.policy_store.load() != before:
                return

    def run(self, max_tasks: int = 0) -> None:
        while True:
            try:
                self.register(force=True)
                break
            except (httpx.HTTPError, OSError, RuntimeError, ValueError) as exc:
                self._status(state="offline", reason=str(exc), last_error=str(exc))
                print(f"seed registration error: {exc}", flush=True)
                time.sleep(max(2.0, self.poll_seconds))
        completed = 0
        while max_tasks <= 0 or completed < max_tasks:
            try:
                if self._reload_policy():
                    self.register(force=True)
                self._capabilities = refresh_dynamic_resources(self._capabilities, self.state_dir)
                if time.time() - self._last_registration >= 60.0:
                    self.register(force=True)
                self._flush_outbox()
                ready, reason = local_run_condition(self.policy)
                if not ready:
                    self._heartbeat()
                    self._status(state="idle", reason=reason, current_task=None)
                    time.sleep(max(1.0, self.poll_seconds))
                    continue
                status, envelope = self._request_json(
                    "POST", f"{self.coordinator}/v1/tasks/claim",
                    body=self.signed_request("claim"),
                )
                if status == 204:
                    self._status(state="idle", reason="no-compatible-work", current_task=None)
                    time.sleep(self.poll_seconds)
                    continue
                assert envelope is not None
                if envelope["coordinator_public_key"] != self.coordinator_public_key:
                    raise RuntimeError("coordinator identity changed")
                if not verify(self.coordinator_public_key, envelope["task"], envelope["signature"]):
                    raise RuntimeError("invalid coordinator task signature")
                task = envelope["task"]
                if task["assigned_to"] != self.identity.node_id:
                    raise RuntimeError("signed task targets another node")
                if time.time() >= float(task["lease_expires_at"]):
                    self._abandon(task, "expired-before-execution")
                    continue
                try:
                    output, duration_ms, duration = self._execute_signed_task(task)
                except TaskExecutionCancelled as error:
                    self._abandon(task, str(error))
                    self._status(state="idle", reason=str(error), current_task=None)
                    continue
                result_body = {
                    "node_id": self.identity.node_id,
                    "task_id": task["id"],
                    "lease_token": task["lease_token"],
                    "duration_ms": duration_ms,
                    "output": output,
                }
                result_body["signature"] = self.identity.sign(result_body)
                self._queue_result(result_body)
                self._flush_outbox()
                completed += 1
                self._status(
                    state="cooldown", reason="resource-share-enforcement", current_task=None,
                    last_result={
                        "task_id": task["id"], "kind": task["kind"], "accepted": True,
                        "impact": (self._last_receipt or {}).get("contribution"),
                        "promoted": (self._last_receipt or {}).get("promoted"),
                        "candidate_id": (self._last_receipt or {}).get("candidate_id"),
                    },
                )
                receipt = self._last_receipt or {}
                contribution = receipt.get("contribution")
                if contribution:
                    print(
                        f"[{self.identity.node_id}] {task['kind']} promoted "
                        f"root {contribution['root_before'][:8]} -> {contribution['root_after'][:8]} "
                        f"net_wins={contribution['net_wins']}",
                        flush=True,
                    )
                else:
                    print(f"[{self.identity.node_id}] {task['kind']} {task['id'][:8]} delivered", flush=True)
                cooldown = cooldown_seconds(duration, self._capabilities, self.policy)
                if cooldown:
                    self._interruptible_cooldown(cooldown)
            except (httpx.HTTPError, OSError, RuntimeError, ValueError, KeyError, json.JSONDecodeError) as exc:
                self._status(state="error", reason=str(exc), current_task=None, last_error=str(exc))
                print(f"seed error: {exc}", flush=True)
                time.sleep(max(2.0, self.poll_seconds))
        self._status(state="stopped", reason="max-tasks-complete", current_task=None)
