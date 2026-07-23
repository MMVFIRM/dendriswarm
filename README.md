# DendriSwarm v0.8.0 — CIFAR-100 Training Dashboard

DendriSwarm coordinates heterogeneous CPU machines to train and search a persistent **Native10-derived Dendritron on real CIFAR-100 data**. v0.8.0 keeps the full v0.7 campaign and adds a one-command local application for contributors and campaign operators.

> **New to DendriSwarm?** Follow the [first-time user guide](docs/FIRST_TIME_USER.md) for prerequisite installation, the correct coordinator/seed startup order, CIFAR-100 preparation, campaign launch, health checks, and common Windows fixes.

## Open the app

Install the lightweight contributor package and run it with no subcommand:

```bash
python -m pip install "git+https://github.com/MMVFIRM/dendriswarm.git"
dendriswarm
```

`dendriswarm app` is equivalent. A browser opens to a token-protected dashboard on `127.0.0.1:8788`.

A downloaded repository can be launched directly:

- Windows contributor: `launch-dashboard.bat`
- Windows campaign operator: `launch-operator-dashboard.bat`
- macOS/Linux contributor: `./launch-dashboard.sh`
- macOS/Linux campaign operator: `./launch-operator-dashboard.sh`

The launchers create `.venv`, install the appropriate package, and open the dashboard. Operator launchers install `dendriswarm[coordinator]`; ordinary contributors retain the smaller package.

## What the dashboard controls

### Contributor view

- Coordinator URL and optional out-of-band fingerprint.
- CPU and memory contribution percentages.
- Disk/cache budget and maximum task duration.
- Battery policy and system-load pause threshold.
- Start, pause, resume, and stop controls.
- Live task, outbox, credit, completion, capability, and log telemetry.

The worker continues to enforce the signed resource contract in an isolated subprocess. Dashboard settings update the same atomic, hot-reloaded `SeedPolicy` used by the CLI.

### Campaign operator view

- Start and stop a local coordinator.
- Prepare and verify the official CIFAR-100 archive.
- Initialize Native10 topology or import an established checkpoint.
- Preview the next routing/model tournament.
- Configure independent search count, sample budget, optimizer steps, learning rate, and verifier quorum.
- Queue the next training round.
- Track canonical roots, routed accuracy, oracle routing gap, top-k category recall, promotions, worker-hours, and campaign history.
- Produce the final official-test report.

Mutating operator actions require a local coordinator-generated admin token. The token is never published through coordinator metadata.

## Campaign topology

```text
official CIFAR-100 images
        ↓
channel normalization + eight spatial 8×16 RGB patches
        ↓
8 trainable sensory field blocks
        ↓
96-wide shared representation
        ↓
1,000 trainable routing scouts
        ↓
top-4 routing with bounded low-margin expansion to top-8
        ↓
20 colonies aligned to CIFAR-100's 20 coarse categories
        ↓
5 fine classes per colony
        ↓
45 experts per colony; rotating 15-of-45 local updates
        ↓
4 nonlinear branches per expert + associative memory
        ↓
100 fine-class predictions
```

The exact model contains **4,898,812 trainable floating-point parameters**, all reachable through bounded protocol operations.

## Data and evidence boundary

The dataset is not redistributed. The operator supplies the official Python archive. Preparation verifies its published MD5, safely extracts it, preserves the official fine/coarse mapping, and creates:

| Split | Rows | Use |
|---|---:|---|
| Train | 45,000 | contributor training and public routing diagnostics |
| Selection bank | 2,500 | trainer-invisible candidate selection; one-shot folds |
| Replication bank | 2,500 | separate final replication; one-shot folds |
| Official test | 10,000 | final reporting only; rejected by the planner |

Candidates promote only after independent search, hidden all-class selection, exact one-sided McNemar testing, familywise correction, effect and per-class harm gates, deterministic replay, and a separate one-shot replication quorum.

## CLI remains available

Every dashboard action has a command-line equivalent:

```bash
dendriswarm doctor
dendriswarm seed --coordinator https://HOST --coordinator-fingerprint SHA256 --share 25
dendriswarm cifar100-prepare ./cifar-100-python.tar.gz --state ./state
dendriswarm cifar100-init --state ./state --checkpoint ./native10-checkpoint.json
dendriswarm cifar100-plan --state ./state
dendriswarm cifar100-queue-next --state ./state
dendriswarm cifar100-evaluate-test ./cifar100-test-report.json --state ./state
```

## Dashboard security and package size

The dashboard uses the Python standard library HTTP server and the dependencies already needed by a contributor. It does not require Node.js, Electron, React, or a separate web server.

- Loopback binding only.
- Random 256-bit launch token.
- `SameSite=Strict`, HTTP-only browser cookie.
- Request-size ceiling.
- Local process supervision with PID creation-time checks.
- Separate coordinator admin token for mutating campaign actions.
- Bounded log tails; no packaged identities, databases, datasets, or secrets.

See `docs/DASHBOARD_V08.md` for the complete interface and persistence model.

## Verification and claims

```bash
python -m pip install -e '.[dev]'
make test
make proof-v08
```

v0.8.0 adds dashboard/configuration tests and a dedicated usability/security proof while retaining every v0.2–v0.7 proof family. The official archive was not available in the packaging environment, so the package still does **not** fabricate a CIFAR-100 accuracy result. Competitive accuracy and distributed-compute leverage remain outputs of the real campaign.

See:

- `docs/FIRST_TIME_USER.md`
- `docs/DASHBOARD_V08.md`
- `docs/CIFAR100_SWARM_V07.md`
- `docs/CLAIMS.md`
- `SECURITY.md`
- `DATA_LICENSES.md`
