# DendriSwarm v0.8.0 local dashboard

The v0.8 dashboard is a loopback-only browser application included in the default contributor package. It uses the Python standard library for HTTP serving and the dependencies already required by a seed. No Node.js toolchain, Electron runtime, database server, or frontend framework is required.

## Start

After installing DendriSwarm, either command opens the dashboard:

```bash
dendriswarm
dendriswarm app
```

A repository checkout also includes launch scripts:

- `launch-dashboard.sh` and `launch-dashboard.bat` install the lightweight contributor package in `.venv`.
- `launch-operator-dashboard.sh` and `launch-operator-dashboard.bat` install `dendriswarm[coordinator]` for local CIFAR-100 campaign operation.

The dashboard binds only to `127.0.0.1`, `localhost`, or `::1`. A random local token is generated on first run. The browser receives it once through the launch URL and stores it in a `SameSite=Strict`, HTTP-only cookie. API calls without that token are rejected.

## Contributor controls

The Contribute page manages the same hot-reloaded `SeedPolicy` enforced by the worker:

- CPU percentage.
- Memory percentage and optional absolute memory ceiling.
- Disk/cache allowance.
- Maximum task duration.
- Battery permission and minimum battery charge.
- System-wide CPU pause threshold.
- Allowed task kinds.
- Pause, resume, start, and stop.
- Coordinator URL and optional out-of-band fingerprint.

The dashboard starts the seed as a separate managed process. Closing the browser does not terminate it. Use **Stop process** when the contribution process should exit.

## Training controls

When `dendriswarm[coordinator]` is installed, the Training page can:

- Start and stop the local coordinator.
- Prepare the verified official CIFAR-100 archive.
- Initialize Native10 topology or import an established checkpoint.
- Preview the next routing/model tournament.
- Set independent candidate count, sample budget, optimizer steps, learning rate, and verifier quorum.
- Queue the next tournament.
- Produce the final official-test report.

Mutating campaign endpoints require a coordinator-generated 256-bit admin token stored under the local operator state directory. The token is not exposed by the public coordinator metadata endpoint.

## Live telemetry

The Overview page displays:

- Canonical model root.
- Campaign round and remaining one-shot evidence rounds.
- Latest routed accuracy and oracle routing gap.
- Top-4 and expanded category recall.
- Conditional accuracy when the correct category is routed.
- Active worker task and outbox state.
- Contributor credits and completed/failed tasks.
- Donated worker hours and the latest promoted round.

The System and Logs pages expose detected machine limits, effective resource budgets, and bounded tails of seed/coordinator output.

## Persistence

Dashboard settings are stored atomically in `~/.dendriswarm/dashboard/dashboard-config.json` by default. Seed identity and policy remain in `~/.dendriswarm/seed`. Operator state defaults to `~/.dendriswarm/operator`. These locations are independently configurable in the dashboard or with command-line flags.
