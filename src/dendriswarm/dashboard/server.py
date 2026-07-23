from __future__ import annotations

import argparse
import json
import threading
import time
import webbrowser
from http import HTTPStatus
from http.cookies import SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from dendriswarm.dashboard.runtime import DashboardRuntime

MAX_DASHBOARD_REQUEST_BYTES = 256 * 1024


class DashboardHTTPServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address: tuple[str, int], runtime: DashboardRuntime):
        super().__init__(address, DashboardHandler)
        self.runtime = runtime


class DashboardHandler(BaseHTTPRequestHandler):
    server: DashboardHTTPServer

    def log_message(self, format: str, *args: Any) -> None:
        return

    def _authorized(self) -> bool:
        token = self.headers.get("X-DendriSwarm-Token", "")
        if token == self.server.runtime.token:
            return True
        cookie = SimpleCookie(self.headers.get("Cookie", ""))
        morsel = cookie.get("dendriswarm_dashboard")
        return bool(morsel and morsel.value == self.server.runtime.token)

    def _json(self, value: Any, status: int = 200) -> None:
        encoded = json.dumps(value, allow_nan=False, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def _error(self, message: str, status: int = 400) -> None:
        self._json({"ok": False, "error": message}, status)

    def _body(self) -> dict[str, Any]:
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError as exc:
            raise ValueError("invalid Content-Length") from exc
        if length < 0 or length > MAX_DASHBOARD_REQUEST_BYTES:
            raise ValueError("dashboard request is too large")
        raw = self.rfile.read(length)
        value = json.loads(raw or b"{}")
        if not isinstance(value, dict):
            raise ValueError("request body must be a JSON object")
        return value

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/health":
            self._json({"ok": True, "version": "0.8.0"})
            return
        if parsed.path == "/":
            query = parse_qs(parsed.query)
            token = query.get("token", [""])[0]
            if token:
                if token != self.server.runtime.token:
                    self._error("invalid dashboard token", HTTPStatus.UNAUTHORIZED)
                    return
                self.send_response(HTTPStatus.FOUND)
                self.send_header("Set-Cookie", f"dendriswarm_dashboard={token}; HttpOnly; SameSite=Strict; Path=/")
                self.send_header("Location", "/")
                self.end_headers()
                return
            if not self._authorized():
                self._error("open the dashboard using the launch URL printed by DendriSwarm", HTTPStatus.UNAUTHORIZED)
                return
            html = files("dendriswarm.dashboard.static").joinpath("index.html").read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(html)))
            self.end_headers()
            self.wfile.write(html)
            return
        if parsed.path == "/api/status":
            if not self._authorized():
                self._error("unauthorized", HTTPStatus.UNAUTHORIZED)
                return
            try:
                self._json({"ok": True, "data": self.server.runtime.aggregate_status()})
            except Exception as exc:
                self._error(str(exc), 500)
            return
        self._error("not found", HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        if not self._authorized():
            self._error("unauthorized", HTTPStatus.UNAUTHORIZED)
            return
        try:
            body = self._body()
            route = urlparse(self.path).path
            runtime = self.server.runtime
            if route == "/api/settings":
                result = runtime.update_settings(body)
            elif route == "/api/seed/start":
                result = runtime.seed_start()
            elif route == "/api/seed/stop":
                result = runtime.seed_stop()
            elif route == "/api/seed/pause":
                result = runtime.seed_pause(True)
            elif route == "/api/seed/resume":
                result = runtime.seed_pause(False)
            elif route == "/api/coordinator/start":
                result = runtime.coordinator_start()
            elif route == "/api/coordinator/stop":
                result = runtime.coordinator_stop()
            elif route.startswith("/api/campaign/"):
                result = runtime.campaign_action(route.rsplit("/", 1)[-1], body)
            else:
                self._error("not found", HTTPStatus.NOT_FOUND)
                return
            self._json({"ok": True, "data": result})
        except (ValueError, RuntimeError, OSError, json.JSONDecodeError) as exc:
            self._error(str(exc), 400)
        except Exception as exc:
            self._error(str(exc), 500)


def run_dashboard(
    *,
    dashboard_state: Path,
    seed_state: Path | None = None,
    operator_state: Path | None = None,
    host: str = "127.0.0.1",
    port: int = 8788,
    open_browser: bool = True,
) -> None:
    if host not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("the dashboard must bind to a loopback address")
    runtime = DashboardRuntime(dashboard_state, seed_state=seed_state, operator_state=operator_state)
    runtime.autostart()
    server = DashboardHTTPServer((host, port), runtime)
    url = f"http://{host}:{port}/?token={runtime.token}"
    print(f"DendriSwarm dashboard: {url}", flush=True)
    print("Press Ctrl+C to stop the dashboard. Managed seed/coordinator processes continue until stopped in the UI.", flush=True)
    if open_browser:
        threading.Timer(0.35, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def main() -> None:
    parser = argparse.ArgumentParser(prog="dendriswarm-dashboard")
    parser.add_argument("--state", default=str(Path.home() / ".dendriswarm" / "dashboard"))
    parser.add_argument("--seed-state")
    parser.add_argument("--operator-state")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8788)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    run_dashboard(
        dashboard_state=Path(args.state).expanduser(),
        seed_state=Path(args.seed_state).expanduser() if args.seed_state else None,
        operator_state=Path(args.operator_state).expanduser() if args.operator_state else None,
        host=args.host,
        port=args.port,
        open_browser=not args.no_browser,
    )


if __name__ == "__main__":
    main()
