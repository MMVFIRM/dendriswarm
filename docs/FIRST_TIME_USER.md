# First-time user guide

This guide takes a new user from an empty machine to a working local DendriSwarm dashboard. It covers both roles:

- **Contributor:** donates bounded CPU time to an existing coordinator.
- **Campaign operator:** runs a coordinator, prepares CIFAR-100, initializes the model, and queues training rounds.

If you are evaluating DendriSwarm on one computer, use the **campaign operator** path. The same machine can run the dashboard, coordinator, and one seed.

## 1. Install prerequisites

DendriSwarm requires Python 3.10 or newer. Git is required when installing the current version directly from GitHub.

### Windows

Open PowerShell and install Python and Git:

```powershell
winget install -e --id Python.Python.3.12
winget install -e --id Git.Git
```

Close PowerShell, open a new window, and verify the installation:

```powershell
python --version
git --version
```

If `python` still opens the Microsoft Store, try `py -3.12` in place of `python`, or disable the Python App Installer aliases under **Settings > Apps > Advanced app settings > App execution aliases**.

### macOS or Linux

Install Python 3.10 or newer and Git with your system package manager, then verify:

```bash
python3 --version
git --version
```

Use `python3` instead of `python` in the commands below if that is how Python is installed on your system.

## 2. Install DendriSwarm

Choose the package for your role.

### Contributor

```bash
python -m pip install "git+https://github.com/MMVFIRM/dendriswarm.git"
```

### Campaign operator or single-machine evaluation

The operator extras include the local coordinator and campaign dependencies:

```bash
python -m pip install "dendriswarm[coordinator] @ git+https://github.com/MMVFIRM/dendriswarm.git"
```

Confirm the command is available:

```bash
dendriswarm --help
dendriswarm doctor
```

If the `dendriswarm` command is not found, close and reopen the terminal so the Python scripts directory is added to the shell. You can also run the equivalent module command:

```bash
python -m dendriswarm --help
```

## 3. Open the dashboard

Run:

```bash
dendriswarm
```

The dashboard opens at `http://127.0.0.1:8788`. Keep the terminal window open. The launch URL contains a one-time local dashboard token; do not share that URL.

The local services use these ports:

| Service | Default address |
|---|---|
| Coordinator API | `http://127.0.0.1:8787` |
| Dashboard | `http://127.0.0.1:8788` |

## 4. Start the services in the correct order

For a single-machine campaign, start the coordinator before the seed:

1. Open the dashboard's **Training** page.
2. Confirm the coordinator URL is `http://127.0.0.1:8787`.
3. Click **Start coordinator** and wait for **Coordinator online**.
4. Open the **Contribute** page.
5. Choose conservative CPU, memory, disk, duration, and battery limits.
6. Save the settings, then click **Start seed**.

The Overview page should show one active node. You can also verify the coordinator from PowerShell:

```powershell
Invoke-RestMethod "http://127.0.0.1:8787/v1/stats"
```

On macOS or Linux:

```bash
curl http://127.0.0.1:8787/v1/stats
```

### Contributor connecting to another machine

Obtain the coordinator URL and SHA-256 coordinator fingerprint from the campaign operator. Use HTTPS for an Internet-accessible coordinator. Do not enable insecure remote HTTP on an untrusted network.

Enter the supplied values on the **Contribute** page, save the settings, and start the seed. The coordinator operator must make the service reachable through the host firewall and network configuration.

## 5. Download and prepare CIFAR-100

Only campaign operators need the complete dataset. DendriSwarm downloads the official Python archive over HTTPS and verifies its published MD5 before using it.

Recommended Windows paths:

```text
C:\Users\YOUR_NAME\Documents\Datasets\cifar-100-python.tar.gz
C:\Users\YOUR_NAME\Documents\DendriSwarm\state
```

Download the archive:

```powershell
dendriswarm cifar100-download "$HOME\Documents\Datasets\cifar-100-python.tar.gz"
```

You can then use the dashboard:

1. Open **Training > Campaign setup**.
2. Enter the full path to `cifar-100-python.tar.gz`.
3. Click **Prepare dataset**.
4. Wait until the dataset status changes to **Prepared**.

The CLI equivalent is:

```powershell
dendriswarm cifar100-prepare "$HOME\Documents\Datasets\cifar-100-python.tar.gz" --state "$HOME\Documents\DendriSwarm\state"
```

On macOS or Linux, a typical archive path is `$HOME/datasets/cifar-100-python.tar.gz`.

## 6. Initialize and start the campaign

After the dataset is prepared:

1. Click **Initialize / import model**. Leave the checkpoint field empty to initialize a new Native10 topology, or supply an established checkpoint.
2. Review the campaign parameters. The defaults are intentionally bounded for local operation.
3. Click **Preview next round** and inspect the proposed tournament.
4. Click **Queue next round**.
5. Watch the Overview and Logs pages for worker activity, completed tasks, promotions, and campaign telemetry.

The seed must remain active for queued work to run. Closing the browser does not necessarily stop dashboard-managed coordinator or seed processes; use the dashboard's stop controls when you want them to exit.

## 7. Resume later

Run `dendriswarm` again and use the same state directories. By default, DendriSwarm stores:

- Dashboard configuration: `~/.dendriswarm/dashboard/dashboard-config.json`
- Seed identity and policy: `~/.dendriswarm/seed`
- Operator state: `~/.dendriswarm/operator`

Do not delete the state directories if you want to preserve node identity, campaign history, model state, and coordinator administration data.

## Troubleshooting

### `Python was not found`

Install Python, open a new terminal, and verify `python --version`. On Windows, try `py -3.12` and check the App Installer aliases if the Microsoft Store opens unexpectedly.

### `dendriswarm` is not recognized

Reopen the terminal after installation. If the scripts directory is not on `PATH`, use:

```bash
python -m dendriswarm
```

### `seed registration error: [WinError 10061]`

The seed cannot connect because no coordinator is listening at the configured address. Stop the seed retry loop, start the coordinator, confirm that `http://127.0.0.1:8787/v1/stats` responds, and then start the seed again.

### `[WinError 10048] only one usage of each socket address`

Another process is already listening on that port. On Windows, identify it with:

```powershell
$conn = Get-NetTCPConnection -LocalPort 8787 -State Listen
Get-Process -Id $conn.OwningProcess
Invoke-RestMethod "http://127.0.0.1:8787/v1/stats"
```

If the stats request succeeds, the coordinator is already healthy; do not launch a second copy. If the process is unrelated, confirm its identity before stopping it or choose another port and update the dashboard's coordinator URL to match.

### Dashboard says coordinator dependencies are missing

Install the operator package and restart the dashboard:

```bash
python -m pip install "dendriswarm[coordinator] @ git+https://github.com/MMVFIRM/dendriswarm.git"
```

### Seed is registered but inactive

Confirm the coordinator is online, then start or resume the seed from the **Contribute** page. Check its CPU, battery, and system-load policies; a seed may pause itself when a configured safety limit is reached.

### Check current coordinator state

```powershell
$stats = Invoke-RestMethod "http://127.0.0.1:8787/v1/stats"
$stats.active_nodes
$stats.cifar100_campaign
```

## Security reminders

- Keep the dashboard bound to loopback (`127.0.0.1`).
- Use HTTPS and verify the coordinator fingerprint for remote contribution.
- Do not publish dashboard launch tokens, coordinator admin tokens, state directories, or private checkpoints.
- Use the dashboard controls to limit CPU, memory, disk, task duration, system load, and battery use.
- Keep the official CIFAR-100 test split reserved for final reporting; the campaign planner does not use it for selection.

For architecture and campaign details, continue with [DASHBOARD_V08.md](DASHBOARD_V08.md), [CIFAR100_SWARM_V07.md](CIFAR100_SWARM_V07.md), and the project [security policy](../SECURITY.md).
