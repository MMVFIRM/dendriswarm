from __future__ import annotations

import json
import os
import secrets
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import httpx
import psutil

from dendriswarm.dashboard.config import CampaignDefaults, DashboardConfig, DashboardConfigStore
from dendriswarm.worker.config import SeedPolicyStore
from dendriswarm.worker.resources import detect_capabilities
from dendriswarm.core.resources import effective_limits
from dendriswarm.core.models import SeedPolicy


class ProcessManager:
    """Small persistent supervisor for dashboard-owned seed/coordinator processes."""

    def __init__(self, state_dir: Path):
        self.state_dir = state_dir
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.state_dir / "processes.json"
        self.logs_dir = self.state_dir / "logs"
        self.logs_dir.mkdir(exist_ok=True)

    def _read(self) -> dict[str, Any]:
        try:
            value = json.loads(self.path.read_text(encoding="utf-8"))
            return value if isinstance(value, dict) else {}
        except (FileNotFoundError, OSError, ValueError, json.JSONDecodeError):
            return {}

    def _write(self, value: dict[str, Any]) -> None:
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, self.path)

    @staticmethod
    def pid_alive(pid: int, expected_create_time: float | None = None) -> bool:
        if pid <= 0 or not psutil.pid_exists(pid):
            return False
        try:
            process = psutil.Process(pid)
            if expected_create_time is not None and abs(process.create_time() - expected_create_time) > 1.0:
                return False
            if process.status() == psutil.STATUS_ZOMBIE:
                try:
                    process.wait(timeout=0)
                except (psutil.TimeoutExpired, psutil.Error):
                    pass
                return False
            return process.is_running()
        except psutil.Error:
            return False

    def status(self) -> dict[str, Any]:
        value = self._read()
        changed = False
        result: dict[str, Any] = {}
        for name, record in value.items():
            pid = int(record.get("pid", 0))
            running = self.pid_alive(pid, record.get("create_time"))
            item = {**record, "running": running}
            result[name] = item
            if record.get("running") != running:
                record["running"] = running
                changed = True
        if changed:
            self._write(value)
        return result

    def start(self, name: str, command: list[str]) -> dict[str, Any]:
        current = self.status().get(name)
        if current and current.get("running"):
            return current
        log_path = self.logs_dir / f"{name}.log"
        log_handle = log_path.open("a", encoding="utf-8", buffering=1)
        started_at = time.time()
        log_handle.write(
            f"\n--- DendriSwarm {name} session started "
            f"{time.strftime('%Y-%m-%d %H:%M:%S')} ---\n"
        )
        kwargs: dict[str, Any] = {
            "stdout": log_handle,
            "stderr": subprocess.STDOUT,
            "stdin": subprocess.DEVNULL,
            "cwd": str(Path.cwd()),
        }
        if os.name == "nt":
            kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]
        else:
            kwargs["start_new_session"] = True
        try:
            process = subprocess.Popen(command, **kwargs)
        finally:
            log_handle.close()
        record = {
            "pid": process.pid,
            "create_time": psutil.Process(process.pid).create_time(),
            "command": command,
            "log_path": str(log_path),
            "started_at": started_at,
            "running": True,
        }
        value = self._read()
        value[name] = record
        self._write(value)
        return record

    def stop(self, name: str, *, timeout: float = 8.0) -> dict[str, Any]:
        value = self._read()
        record = value.get(name)
        if not record:
            return {"running": False, "stopped": False}
        pid = int(record.get("pid", 0))
        if self.pid_alive(pid, record.get("create_time")):
            try:
                parent = psutil.Process(pid)
                targets = parent.children(recursive=True) + [parent]
                for process in targets:
                    try:
                        process.terminate()
                    except psutil.Error:
                        pass
                _, alive = psutil.wait_procs(targets, timeout=timeout)
                for process in alive:
                    try:
                        process.kill()
                    except psutil.Error:
                        pass
                psutil.wait_procs(alive, timeout=2.0)
            except psutil.Error:
                pass
        record["running"] = False
        record["stopped_at"] = time.time()
        value[name] = record
        self._write(value)
        return {**record, "stopped": True}

    def tail(self, name: str, *, max_bytes: int = 48_000) -> str:
        record = self._read().get(name, {})
        path = Path(record.get("log_path", self.logs_dir / f"{name}.log"))
        try:
            with path.open("rb") as handle:
                handle.seek(0, os.SEEK_END)
                size = handle.tell()
                handle.seek(max(0, size - max_bytes))
                return handle.read(max_bytes).decode("utf-8", errors="replace")
        except OSError:
            return ""


class DashboardRuntime:
    def __init__(
        self,
        dashboard_state: Path,
        *,
        seed_state: Path | None = None,
        operator_state: Path | None = None,
    ):
        self.dashboard_state = dashboard_state.expanduser()
        self.dashboard_state.mkdir(parents=True, exist_ok=True)
        defaults: dict[str, Any] = {}
        if seed_state is not None:
            defaults["seed_state"] = str(seed_state.expanduser())
        if operator_state is not None:
            defaults["operator_state"] = str(operator_state.expanduser())
        self.config_store = DashboardConfigStore(self.dashboard_state / "dashboard-config.json")
        self.config = self.config_store.load(defaults=defaults)
        self.processes = ProcessManager(self.dashboard_state)
        token_path = self.dashboard_state / "dashboard-token"
        if token_path.exists():
            self.token = token_path.read_text(encoding="utf-8").strip()
        else:
            self.token = secrets.token_urlsafe(32)
            token_path.write_text(self.token, encoding="utf-8")
            try:
                token_path.chmod(0o600)
            except OSError:
                pass

    @property
    def seed_state(self) -> Path:
        return Path(self.config.seed_state).expanduser()

    @property
    def operator_state(self) -> Path:
        return Path(self.config.operator_state).expanduser()

    def reload(self) -> DashboardConfig:
        self.config = self.config_store.load()
        return self.config

    @staticmethod
    def _read_json(path: Path, default: Any) -> Any:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (FileNotFoundError, OSError, ValueError, json.JSONDecodeError):
            return default

    def _admin_token(self) -> str | None:
        path = self.operator_state / "keys" / "dashboard-admin-token"
        try:
            token = path.read_text(encoding="utf-8").strip()
            return token or None
        except OSError:
            return None

    def _coordinator_get(self, path: str) -> dict[str, Any]:
        with httpx.Client(timeout=2.5, follow_redirects=False) as client:
            response = client.get(f"{self.config.coordinator_url}{path}")
            response.raise_for_status()
            value = response.json()
            if not isinstance(value, dict):
                raise RuntimeError("coordinator response must be an object")
            return value

    def _coordinator_admin_post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        token = self._admin_token()
        if not token:
            raise RuntimeError("local coordinator admin token is unavailable; start the coordinator from this dashboard")
        with httpx.Client(timeout=3600.0, follow_redirects=False) as client:
            response = client.post(
                f"{self.config.coordinator_url}{path}",
                json=payload,
                headers={"X-DendriSwarm-Admin": token},
            )
            if response.status_code >= 400:
                try:
                    detail = response.json().get("detail", response.text)
                except Exception:
                    detail = response.text
                raise RuntimeError(str(detail))
            value = response.json()
            if not isinstance(value, dict):
                raise RuntimeError("coordinator response must be an object")
            return value

    def aggregate_status(self) -> dict[str, Any]:
        self.reload()
        policy_store = SeedPolicyStore(self.seed_state / "seed-config.json")
        policy = policy_store.load()
        seed_status = self._read_json(
            self.seed_state / "seed-status.json",
            {"state": "not-running", "policy": policy.model_dump(mode="json")},
        )
        try:
            capabilities = detect_capabilities(self.seed_state, policy)
            doctor = {
                "capabilities": capabilities.model_dump(mode="json"),
                "effective_limits": effective_limits(capabilities, policy),
            }
        except Exception as exc:
            doctor = {"error": str(exc)}
        coordinator: dict[str, Any]
        try:
            coordinator_meta = self._coordinator_get("/v1/meta")
        except Exception as exc:
            coordinator = {"online": False, "error": str(exc)}
        else:
            try:
                coordinator = {
                    "online": True,
                    "meta": coordinator_meta,
                    "stats": self._coordinator_get("/v1/stats"),
                }
            except Exception as exc:
                # A responsive control plane is online even when expensive or
                # temporarily contended telemetry misses its deadline.
                coordinator = {
                    "online": True,
                    "degraded": True,
                    "meta": coordinator_meta,
                    "stats": {},
                    "error": f"telemetry delayed: {exc}",
                }
            node_id = str(seed_status.get("node_id", ""))
            if node_id and not coordinator.get("degraded"):
                try:
                    coordinator["node_account"] = self._coordinator_get(f"/v1/nodes/{node_id}")
                except Exception:
                    pass
        process_status = self.processes.status()
        return {
            "version": "0.8.0",
            "updated_at": time.time(),
            "config": self.config.model_dump(mode="json"),
            "seed": seed_status,
            "doctor": doctor,
            "processes": process_status,
            "coordinator": coordinator,
            "logs": {
                "seed": self.processes.tail("seed"),
                "coordinator": self.processes.tail("coordinator"),
            },
        }

    def update_settings(self, payload: dict[str, Any]) -> dict[str, Any]:
        current = self.config.model_dump(mode="python")
        connection = payload.get("connection") or {}
        for key in (
            "coordinator_url", "coordinator_fingerprint", "allow_insecure_http",
            "seed_state", "operator_state", "auto_start_seed", "auto_start_coordinator",
            "refresh_seconds",
        ):
            if key in connection:
                current[key] = connection[key]
        if "campaign" in payload:
            current["campaign"] = CampaignDefaults.model_validate(payload["campaign"]).model_dump(mode="python")
        config = DashboardConfig.model_validate(current)
        self.config_store.save(config)
        self.config = config
        if "policy" in payload:
            policy = SeedPolicy.model_validate(payload["policy"])
            SeedPolicyStore(self.seed_state / "seed-config.json").save(policy)
        return self.aggregate_status()

    def seed_start(self) -> dict[str, Any]:
        command = [
            sys.executable, "-m", "dendriswarm", "seed",
            "--coordinator", self.config.coordinator_url,
            "--state", str(self.seed_state),
        ]
        if self.config.coordinator_fingerprint:
            command += ["--coordinator-fingerprint", self.config.coordinator_fingerprint]
        if self.config.allow_insecure_http:
            command.append("--allow-insecure-http")
        return self.processes.start("seed", command)

    def seed_stop(self) -> dict[str, Any]:
        return self.processes.stop("seed")

    def seed_pause(self, paused: bool) -> dict[str, Any]:
        policy = SeedPolicyStore(self.seed_state / "seed-config.json").update(paused=paused)
        return policy.model_dump(mode="json")

    def coordinator_start(self) -> dict[str, Any]:
        try:
            import fastapi  # noqa: F401
            import uvicorn  # noqa: F401
        except ImportError as exc:
            raise RuntimeError("operator dependencies are missing; install dendriswarm[coordinator]") from exc
        from urllib.parse import urlparse
        parsed = urlparse(self.config.coordinator_url)
        if (parsed.hostname or "").lower() not in {"127.0.0.1", "localhost", "::1"}:
            raise RuntimeError("the dashboard can start only a local coordinator")
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        if parsed.scheme != "http":
            raise RuntimeError("the built-in local coordinator uses HTTP on loopback")
        command = [
            sys.executable, "-m", "dendriswarm", "coordinator",
            "--host", "127.0.0.1", "--port", str(port),
            "--state", str(self.operator_state),
        ]
        return self.processes.start("coordinator", command)

    def coordinator_stop(self) -> dict[str, Any]:
        return self.processes.stop("coordinator")

    def campaign_action(self, action: str, payload: dict[str, Any]) -> dict[str, Any]:
        routes = {
            "plan": "/v1/admin/cifar100/plan",
            "queue": "/v1/admin/cifar100/queue-next",
            "prepare": "/v1/admin/cifar100/prepare",
            "init": "/v1/admin/cifar100/init",
            "evaluate": "/v1/admin/cifar100/evaluate-test",
        }
        if action not in routes:
            raise ValueError("unsupported campaign action")
        if action in {"plan", "queue"}:
            defaults = self.config.campaign.model_dump(mode="json")
            defaults.update(payload)
            payload = defaults
        return self._coordinator_admin_post(routes[action], payload)

    def autostart(self) -> None:
        if self.config.auto_start_coordinator:
            try:
                self.coordinator_start()
            except Exception:
                pass
        if self.config.auto_start_seed:
            try:
                self.seed_start()
            except Exception:
                pass
