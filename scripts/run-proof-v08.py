#!/usr/bin/env python3
"""Reproduce the v0.8.0 local dashboard and package-usability proof."""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

import httpx
from fastapi.testclient import TestClient

from dendriswarm import __version__
from dendriswarm.coordinator.app import create_app
from dendriswarm.dashboard.config import DashboardConfigStore
from dendriswarm.dashboard.runtime import DashboardRuntime, ProcessManager
from dendriswarm.dashboard.server import DashboardHTTPServer, run_dashboard
from dendriswarm.worker.config import SeedPolicyStore

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "docs" / "PROOF_RUN_V08.json"


def gate(name: str, passed: bool, evidence: dict | str) -> dict:
    return {"name": name, "pass": bool(passed), "evidence": evidence}


def main() -> None:
    gates: list[dict] = []
    with tempfile.TemporaryDirectory(prefix="dendriswarm-v08-proof-") as temporary:
        base = Path(temporary)
        runtime = DashboardRuntime(base / "dashboard", seed_state=base / "seed", operator_state=base / "operator")
        gates.append(gate("package_version", __version__ == "0.8.0", {"version": __version__}))

        store = DashboardConfigStore(base / "dashboard" / "config-proof.json")
        initial = store.load(defaults={"seed_state": str(base / "seed")})
        updated = store.update(coordinator_url="https://coordinator.example", refresh_seconds=3)
        gates.append(gate(
            "atomic_configuration",
            initial.campaign.search_candidates == 8 and updated.refresh_seconds == 3 and json.loads(store.path.read_text())["coordinator_url"] == "https://coordinator.example",
            {"path": str(store.path), "refresh_seconds": updated.refresh_seconds},
        ))

        runtime.update_settings({
            "policy": {
                **SeedPolicyStore(runtime.seed_state / "seed-config.json").load().model_dump(mode="json"),
                "cpu_percent": 13,
                "memory_percent": 17,
                "disk_limit_mb": 512,
            }
        })
        policy = SeedPolicyStore(runtime.seed_state / "seed-config.json").load()
        gates.append(gate(
            "worker_policy_binding",
            policy.cpu_percent == 13 and policy.memory_percent == 17 and policy.disk_limit_mb == 512,
            policy.model_dump(mode="json"),
        ))

        manager = ProcessManager(base / "processes")
        record = manager.start("probe", [sys.executable, "-c", "import time; time.sleep(30)"])
        running = manager.status()["probe"]["running"]
        stopped = manager.stop("probe")
        gates.append(gate(
            "managed_process_lifecycle",
            record["running"] and running and stopped["stopped"] and not manager.status()["probe"]["running"],
            {"pid": record["pid"], "creation_time_bound": "create_time" in record},
        ))

        loopback_rejected = False
        try:
            run_dashboard(dashboard_state=base / "bad", host="0.0.0.0", port=0, open_browser=False)
        except ValueError:
            loopback_rejected = True
        gates.append(gate("loopback_only", loopback_rejected, {"rejected_host": "0.0.0.0"}))

        server = DashboardHTTPServer(("127.0.0.1", 0), runtime)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        url = f"http://127.0.0.1:{server.server_address[1]}"
        try:
            with httpx.Client(timeout=5, follow_redirects=True) as client:
                unauthorized = client.get(url + "/api/status")
                page = client.get(url + f"/?token={runtime.token}")
                authorized = client.get(url + "/api/status")
                gates.append(gate(
                    "dashboard_token_authentication",
                    unauthorized.status_code == 401 and page.status_code == 200 and authorized.status_code == 200,
                    {"unauthorized": unauthorized.status_code, "authorized": authorized.status_code},
                ))
                gates.append(gate(
                    "dashboard_control_surface",
                    all(label in page.text for label in ("Contribution allocation", "Next training tournament", "Routing progress", "Campaign setup")),
                    {"html_bytes": len(page.content)},
                ))
        finally:
            server.shutdown(); server.server_close(); thread.join(timeout=5)

        app = create_app(base / "operator")
        client = TestClient(app)
        missing = client.post("/v1/admin/cifar100/plan", json={})
        token = (base / "operator" / "keys" / "dashboard-admin-token").read_text().strip()
        authorized = client.post("/v1/admin/cifar100/plan", json={}, headers={"X-DendriSwarm-Admin": token})
        meta = client.get("/v1/meta").json()
        gates.append(gate(
            "separate_operator_admin_token",
            missing.status_code == 401 and authorized.status_code == 400 and token not in json.dumps(meta),
            {"missing_status": missing.status_code, "authorized_status": authorized.status_code, "token_published": token in json.dumps(meta)},
        ))
        stats = client.get("/v1/stats").json()
        gates.append(gate(
            "campaign_telemetry_available",
            "cifar100_campaign" in stats and "native10_v6" in stats,
            {"stats_keys": sorted(stats)},
        ))

        import_probe = subprocess.run(
            [sys.executable, "-c", "import sys; from dendriswarm.dashboard.runtime import DashboardRuntime; print('fastapi' in sys.modules, 'sklearn' in sys.modules)"],
            cwd=ROOT,
            env={**__import__("os").environ, "PYTHONPATH": str(ROOT / "src")},
            text=True,
            capture_output=True,
            check=True,
        )
        gates.append(gate(
            "lightweight_contributor_import",
            import_probe.stdout.strip() == "False False",
            {"loaded": import_probe.stdout.strip()},
        ))

        pyproject = (ROOT / "pyproject.toml").read_text()
        gates.append(gate(
            "no_frontend_runtime_dependency",
            all(name not in pyproject.lower() for name in ("electron", "react", "nodejs", "streamlit", "gradio")),
            {"static_asset": "src/dendriswarm/dashboard/static/index.html"},
        ))
        launchers = [
            ROOT / "launch-dashboard.sh", ROOT / "launch-dashboard.bat",
            ROOT / "launch-operator-dashboard.sh", ROOT / "launch-operator-dashboard.bat",
        ]
        gates.append(gate(
            "cross_platform_launchers",
            all(path.is_file() and path.stat().st_size > 100 for path in launchers),
            {"launchers": [path.name for path in launchers]},
        ))

    report = {
        "proof": "dendriswarm-v0.8.0-local-dashboard",
        "gate_count": len(gates),
        "all_pass": all(item["pass"] for item in gates),
        "gates": gates,
        "claim_boundary": {
            "local_control_surface": True,
            "remote_multi_tenant_admin": False,
            "cifar100_accuracy_claim": False,
        },
    }
    REPORT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
