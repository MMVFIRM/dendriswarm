from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path

import httpx
from fastapi.testclient import TestClient

from dendriswarm.coordinator.app import create_app
from dendriswarm.dashboard.config import DashboardConfigStore
from dendriswarm.dashboard.runtime import DashboardRuntime, ProcessManager
from dendriswarm.dashboard.server import DashboardHTTPServer


def test_dashboard_config_is_atomic_and_validated(tmp_path: Path):
    store = DashboardConfigStore(tmp_path / "dashboard.json")
    config = store.load(defaults={"seed_state": str(tmp_path / "seed")})
    assert config.campaign.search_candidates == 8
    updated = store.update(coordinator_url="https://example.test", refresh_seconds=3)
    assert updated.coordinator_url == "https://example.test"
    assert json.loads(store.path.read_text())["refresh_seconds"] == 3


def test_process_manager_tracks_real_process(tmp_path: Path):
    manager = ProcessManager(tmp_path)
    record = manager.start("probe", [sys.executable, "-c", "import time; time.sleep(30)"])
    assert record["running"] is True
    assert manager.status()["probe"]["running"] is True
    assert "DendriSwarm probe session started" in manager.tail("probe")
    stopped = manager.stop("probe")
    assert stopped["stopped"] is True
    assert manager.status()["probe"]["running"] is False


def test_loopback_dashboard_requires_token_and_serves_status(tmp_path: Path):
    runtime = DashboardRuntime(
        tmp_path / "dashboard",
        seed_state=tmp_path / "seed",
        operator_state=tmp_path / "operator",
    )
    server = DashboardHTTPServer(("127.0.0.1", 0), runtime)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        with httpx.Client(follow_redirects=True, timeout=5) as client:
            assert client.get(base + "/health").status_code == 200
            assert client.get(base + "/api/status").status_code == 401
            page = client.get(base + f"/?token={runtime.token}")
            assert page.status_code == 200
            assert "DendriSwarm" in page.text
            assert 'id="coordinator-error"' in page.text
            assert "let refreshPromise=null" in page.text
            status = client.get(base + "/api/status")
            assert status.status_code == 200
            assert status.json()["data"]["version"] == "0.8.0"
            response = client.post(base + "/api/settings", json={
                "connection": {"coordinator_url": "http://127.0.0.1:9999"},
                "policy": {
                    "paused": False,
                    "cpu_percent": 17,
                    "memory_percent": 19,
                    "memory_limit_mb": None,
                    "disk_limit_mb": 512,
                    "max_task_seconds": 600,
                    "allowed_task_kinds": [
                        "exploration", "training", "verification", "inference",
                        "dendritron-mutation", "dendritron-verification",
                    ],
                    "allow_on_battery": False,
                    "min_battery_percent": 25,
                    "max_system_cpu_percent": 90,
                },
                "campaign": {
                    "search_candidates": 6,
                    "sample_budget": 500,
                    "optimizer_steps": 30,
                    "learning_rate": 0.02,
                    "verification_quorum": 2,
                },
            })
            assert response.status_code == 200
            assert response.json()["data"]["seed"]["policy"]["cpu_percent"] == 17
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_dashboard_reports_responsive_coordinator_when_telemetry_is_delayed(tmp_path, monkeypatch):
    runtime = DashboardRuntime(
        tmp_path / "dashboard",
        seed_state=tmp_path / "seed",
        operator_state=tmp_path / "operator",
    )

    def coordinator_get(path):
        if path == "/v1/meta":
            return {"name": "DendriSwarm", "version": "0.8.0"}
        raise TimeoutError("stats are busy")

    monkeypatch.setattr(runtime, "_coordinator_get", coordinator_get)
    status = runtime.aggregate_status()["coordinator"]
    assert status["online"] is True
    assert status["degraded"] is True
    assert status["stats"] == {}
    assert status["error"] == "telemetry delayed: stats are busy"


def test_stats_endpoint_builds_native10_v6_status_once(tmp_path, monkeypatch):
    app = create_app(tmp_path / "operator")
    original = app.state.service.native10_v6.store.status
    calls = 0

    def counted_status():
        nonlocal calls
        calls += 1
        return original()

    monkeypatch.setattr(app.state.service.native10_v6.store, "status", counted_status)
    response = TestClient(app).get("/v1/stats")
    assert response.status_code == 200
    assert calls == 1


def test_coordinator_dashboard_admin_routes_require_local_token(tmp_path: Path):
    app = create_app(tmp_path / "operator")
    client = TestClient(app)
    unauthorized = client.post("/v1/admin/cifar100/plan", json={})
    assert unauthorized.status_code == 401
    token = (tmp_path / "operator" / "keys" / "dashboard-admin-token").read_text().strip()
    authorized = client.post(
        "/v1/admin/cifar100/plan",
        json={},
        headers={"X-DendriSwarm-Admin": token},
    )
    assert authorized.status_code == 400
    assert "initialized" in authorized.json()["detail"]
