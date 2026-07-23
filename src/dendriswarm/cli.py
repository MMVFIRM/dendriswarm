from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import httpx

from dendriswarm.core.crypto import Identity, nonce
from dendriswarm.core.models import TaskKind
from dendriswarm.worker.config import SeedPolicyStore
from dendriswarm.worker.node import SeedNode


DEFAULT_SEED_STATE = str(Path.home() / ".dendriswarm" / "seed")


def submit_and_wait(base: str, state: Path, features: list[float], wait_seconds: float) -> None:
    identity = Identity.load_or_create(state / "keys")
    request = {
        "node_id": identity.node_id,
        "request_id": nonce(),
        "timestamp": int(time.time()),
        "nonce": nonce(),
        "features": features,
    }
    request["signature"] = identity.sign({"action": "inference", **request})
    response = httpx.post(f"{base}/v1/inference", json=request, timeout=30.0)
    response.raise_for_status()
    task_id = response.json()["task_id"]
    deadline = time.time() + wait_seconds
    while time.time() < deadline:
        job_response = httpx.get(f"{base}/v1/jobs/{task_id}", timeout=30.0)
        job_response.raise_for_status()
        job = job_response.json()
        if job["status"] == "completed":
            print(json.dumps(job, indent=2))
            return
        if job["status"] == "failed":
            raise SystemExit(f"inference task {task_id} failed")
        time.sleep(0.5)
    raise SystemExit(f"inference task {task_id} did not finish before timeout")


def _task_kinds(value: str | None) -> list[str] | None:
    if value is None:
        return None
    names = [item.strip().lower() for item in value.split(",") if item.strip()]
    if not names:
        raise argparse.ArgumentTypeError("task type list cannot be empty")
    try:
        return [TaskKind(item).value for item in names]
    except ValueError as exc:
        allowed = ", ".join(kind.value for kind in TaskKind)
        raise argparse.ArgumentTypeError(f"task types must be drawn from: {allowed}") from exc


def _policy_changes(args: argparse.Namespace) -> dict[str, object]:
    changes: dict[str, object] = {}
    share = getattr(args, "share", None)
    if share is not None:
        changes["cpu_percent"] = share
        changes["memory_percent"] = share
    mapping = {
        "cpu_percent": "cpu_percent",
        "memory_percent": "memory_percent",
        "memory_mb": "memory_limit_mb",
        "disk_mb": "disk_limit_mb",
        "max_task_minutes": "max_task_seconds",
        "task_types": "allowed_task_kinds",
        "min_battery_percent": "min_battery_percent",
        "max_system_cpu_percent": "max_system_cpu_percent",
    }
    for argument, field in mapping.items():
        value = getattr(args, argument, None)
        if value is not None:
            changes[field] = int(value * 60) if argument == "max_task_minutes" else value
    battery = getattr(args, "battery", None)
    if battery is not None:
        changes["allow_on_battery"] = battery == "allow"
    paused = getattr(args, "paused", None)
    if paused is not None:
        changes["paused"] = paused
    return changes


def _configure_seed(state: Path, args: argparse.Namespace) -> dict[str, object]:
    store = SeedPolicyStore(state / "seed-config.json")
    changes = _policy_changes(args)
    policy = store.update(**changes) if changes else store.load()
    return policy.model_dump(mode="json")


def _add_resource_flags(parser: argparse.ArgumentParser, include_pause: bool = False) -> None:
    parser.add_argument("--share", type=int, choices=range(1, 101), metavar="1-100", help="Set CPU and memory contribution percentages together")
    parser.add_argument("--cpu-percent", type=int, choices=range(1, 101), metavar="1-100")
    parser.add_argument("--memory-percent", type=int, choices=range(1, 101), metavar="1-100")
    parser.add_argument("--memory-mb", type=int, help="Absolute memory ceiling in MB")
    parser.add_argument("--disk-mb", type=int, help="Maximum local cache/work disk budget in MB")
    parser.add_argument("--max-task-minutes", type=int, help="Reject tasks estimated to exceed this duration")
    parser.add_argument("--task-types", type=_task_kinds, help="Comma-separated task kinds, including dendritron-mutation and dendritron-verification")
    parser.add_argument("--battery", choices=("allow", "disallow"), help="Whether seeding may run while on battery")
    parser.add_argument("--min-battery-percent", type=int, choices=range(0, 101))
    parser.add_argument("--max-system-cpu-percent", type=int, choices=range(10, 101), help="Pause claims when total system CPU is above this level")
    if include_pause:
        group = parser.add_mutually_exclusive_group()
        group.add_argument("--pause", dest="paused", action="store_true")
        group.add_argument("--resume", dest="paused", action="store_false")
        parser.set_defaults(paused=None)


def main() -> None:
    parser = argparse.ArgumentParser(prog="dendriswarm")
    sub = parser.add_subparsers(dest="command")

    app = sub.add_parser("app", help="Open the local configuration and training dashboard")
    app.add_argument("--state", default=str(Path.home() / ".dendriswarm" / "dashboard"))
    app.add_argument("--seed-state")
    app.add_argument("--operator-state")
    app.add_argument("--host", default="127.0.0.1")
    app.add_argument("--port", type=int, default=8788)
    app.add_argument("--no-browser", action="store_true")

    coordinator = sub.add_parser("coordinator", help="Run the coordinator API")
    coordinator.add_argument("--host", default="0.0.0.0")
    coordinator.add_argument("--port", type=int, default=8787)
    coordinator.add_argument("--state", default="./state")
    coordinator.add_argument("--bootstrap", action="store_true")
    coordinator.add_argument("--lease-seconds", type=float, default=60.0)
    coordinator.add_argument("--inference-audit-rate", type=float, default=0.2)
    coordinator.add_argument("--enable-leverage", action="store_true")

    seed = sub.add_parser("seed", help="Contribute portable CPU work within a local resource budget")
    seed.add_argument("--coordinator", default="http://127.0.0.1:8787")
    seed.add_argument("--state", default=DEFAULT_SEED_STATE)
    seed.add_argument("--poll", type=float, default=0.5)
    seed.add_argument("--max-tasks", type=int, default=0)
    seed.add_argument("--coordinator-fingerprint", help="Expected SHA-256 coordinator key fingerprint supplied out of band")
    seed.add_argument("--allow-insecure-http", action="store_true", help="Allow a non-local HTTP coordinator only on an explicitly trusted test network")
    _add_resource_flags(seed)

    seed_config = sub.add_parser("seed-config", help="Change a running seed's hot-reloaded resource budget")
    seed_config.add_argument("--state", default=DEFAULT_SEED_STATE)
    _add_resource_flags(seed_config, include_pause=True)

    seed_pause = sub.add_parser("seed-pause", help="Pause a running seed and cancel active work at the next enforcement poll")
    seed_pause.add_argument("--state", default=DEFAULT_SEED_STATE)

    seed_resume = sub.add_parser("seed-resume", help="Resume a paused seed")
    seed_resume.add_argument("--state", default=DEFAULT_SEED_STATE)

    seed_local_status = sub.add_parser("seed-local-status", help="Show the local seed policy and runtime state")
    seed_local_status.add_argument("--state", default=DEFAULT_SEED_STATE)

    doctor = sub.add_parser("doctor", help="Inspect portable seeding support and effective local limits")
    doctor.add_argument("--state", default=DEFAULT_SEED_STATE)

    status = sub.add_parser("status", help="Show swarm status")
    status.add_argument("--coordinator", default="http://127.0.0.1:8787")

    audit = sub.add_parser("audit", help="Show the signed audit checkpoint")
    audit.add_argument("--coordinator", default="http://127.0.0.1:8787")

    infer = sub.add_parser("infer", help="Spend seed credits on a 64-feature digit request")
    infer.add_argument("--coordinator", default="http://127.0.0.1:8787")
    infer.add_argument("--state", default=DEFAULT_SEED_STATE)
    infer.add_argument("features", nargs=64, type=float, help="64 normalized pixel values for an 8x8 digit")
    infer.add_argument("--wait", type=float, default=30.0)

    sample = sub.add_parser("infer-sample", help="Classify a built-in held-out digit sample")
    sample.add_argument("index", type=int, nargs="?", default=0)
    sample.add_argument("--coordinator", default="http://127.0.0.1:8787")
    sample.add_argument("--state", default=DEFAULT_SEED_STATE)
    sample.add_argument("--wait", type=float, default=30.0)

    cifar_download = sub.add_parser("cifar100-download", help="Download and verify the official CIFAR-100 Python archive")
    cifar_download.add_argument("output", nargs="?", default="./cifar-100-python.tar.gz")

    cifar_prepare = sub.add_parser("cifar100-prepare", help="Prepare the official CIFAR-100 Python archive for a real swarm campaign")
    cifar_prepare.add_argument("source", help="Official cifar-100-python.tar.gz or extracted cifar-100-python directory")
    cifar_prepare.add_argument("--state", default="./state")
    cifar_prepare.add_argument("--seed", type=int, default=20260723)
    cifar_prepare.add_argument("--holdout-per-class", type=int, default=5)
    cifar_prepare.add_argument("--replace", action="store_true")

    cifar_init = sub.add_parser("cifar100-init", help="Initialize the exact Native10 CIFAR-100 topology or import a checkpoint")
    cifar_init.add_argument("--state", default="./state")
    cifar_init.add_argument("--checkpoint")
    cifar_init.add_argument("--seed", type=int, default=7)
    cifar_init.add_argument("--replace", action="store_true")

    cifar_status = sub.add_parser("cifar100-status", help="Show CIFAR-100 campaign, model, routing, and holdout status")
    cifar_status.add_argument("--state", default="./state")

    cifar_plan = sub.add_parser("cifar100-plan", help="Plan the next real CIFAR-100 tissue or routing-search round")
    cifar_plan.add_argument("--state", default="./state")
    cifar_plan.add_argument("--search-candidates", type=int, default=8)
    cifar_plan.add_argument("--sample-budget", type=int, default=640)

    cifar_queue = sub.add_parser("cifar100-queue-next", help="Queue the next CIFAR-100 swarm training/search tournament")
    cifar_queue.add_argument("--state", default="./state")
    cifar_queue.add_argument("--search-candidates", type=int, default=8)
    cifar_queue.add_argument("--sample-budget", type=int, default=640)
    cifar_queue.add_argument("--optimizer-steps", type=int, default=36)
    cifar_queue.add_argument("--learning-rate", type=float, default=0.03)
    cifar_queue.add_argument("--verification-quorum", type=int, default=2)

    cifar_test = sub.add_parser("cifar100-evaluate-test", help="Evaluate a canonical snapshot on the untouched official CIFAR-100 test split")
    cifar_test.add_argument("output")
    cifar_test.add_argument("--state", default="./state")
    cifar_test.add_argument("--source", default="official-cifar100-test")

    native_init = sub.add_parser("native10-init", help="Initialize a Native10-derived topology without baseline training")
    native_init.add_argument("--state", default="./state")
    native_init.add_argument("--profile", choices=("compact", "native10"), default="native10")
    native_init.add_argument("--input-width", type=int)
    native_init.add_argument("--seed", type=int, default=7)
    native_init.add_argument("--replace", action="store_true")

    native_import = sub.add_parser("native10-import", help="Import a converted Native10-derived checkpoint")
    native_import.add_argument("checkpoint")
    native_import.add_argument("--state", default="./state")
    native_import.add_argument("--profile", choices=("compact", "native10"), default="native10")
    native_import.add_argument("--input-width", type=int)
    native_import.add_argument("--seed", type=int, default=7)
    native_import.add_argument("--key-map", help="Optional JSON mapping from v5 tensor names to source checkpoint keys")
    native_import.add_argument("--replace", action="store_true")

    native_validation_create = sub.add_parser(
        "native10-validation-create",
        help="Build a hash-bound all-class coordinator validation artifact from an NPZ archive",
    )
    native_validation_create.add_argument("input", help="NPZ containing representations and labels arrays")
    native_validation_create.add_argument("output")
    native_validation_create.add_argument("--state", default="./state")
    native_validation_create.add_argument("--source", required=True)
    native_validation_create.add_argument("--split", default="validation")
    native_validation_create.add_argument("--min-samples-per-class", type=int, default=5)
    native_validation_create.add_argument("--minimum-net-wins", type=int, default=1)
    native_validation_create.add_argument("--max-loss-per-class", type=int, default=1)
    native_validation_create.add_argument("--max-loss-rate-per-class", type=float, default=0.20)
    native_validation_create.add_argument("--max-candidate-evaluations", type=int, default=40)
    native_validation_create.add_argument("--install", action="store_true")
    native_validation_create.add_argument("--replace", action="store_true")

    native_validation_import = sub.add_parser(
        "native10-validation-import",
        help="Install a prebuilt all-class coordinator validation artifact",
    )
    native_validation_import.add_argument("artifact")
    native_validation_import.add_argument("--state", default="./state")
    native_validation_import.add_argument("--replace", action="store_true")

    native_status = sub.add_parser("native10-status", help="Show the canonical Dendritron root and contribution lineage")
    native_status.add_argument("--state", default="./state")

    native_queue = sub.add_parser("native10-queue", help="Queue Native10-derived tissue work from a representation shard")
    native_queue.add_argument("--state", default="./state")
    native_queue.add_argument("--shard", help="JSON representation shard; omit with --demo")
    native_queue.add_argument("--demo", action="store_true", help="Use a synthetic protocol fixture, not a baseline benchmark")
    native_queue.add_argument("--operation", choices=("expert_refit", "repair", "branch_lifecycle", "scout_refit", "memory_update"), default="expert_refit")
    native_queue.add_argument("--category", type=int, default=0)
    native_queue.add_argument("--subset-seed", type=int, default=7)

    native_export = sub.add_parser("native10-export-int8", help="Export the current canonical Dendritron as a signed INT8 bundle")
    native_export.add_argument("output")
    native_export.add_argument("--state", default="./state")

    native_checkpoint = sub.add_parser("native10-export-checkpoint", help="Export the current canonical checkpoint JSON")
    native_checkpoint.add_argument("output")
    native_checkpoint.add_argument("--state", default="./state")

    v6_init = sub.add_parser("native10-v6-init", help="Initialize the trainable Native10 v0.6 topology")
    v6_init.add_argument("--state", default="./state")
    v6_init.add_argument("--profile", choices=("compact", "native10"), default="native10")
    v6_init.add_argument("--input-width", type=int)
    v6_init.add_argument("--seed", type=int, default=7)
    v6_init.add_argument("--replace", action="store_true")

    v6_import = sub.add_parser("native10-v6-import", help="Import a trainable Native10 v0.6 checkpoint")
    v6_import.add_argument("checkpoint")
    v6_import.add_argument("--state", default="./state")
    v6_import.add_argument("--profile", choices=("compact", "native10"), default="native10")
    v6_import.add_argument("--input-width", type=int)
    v6_import.add_argument("--seed", type=int, default=7)
    v6_import.add_argument("--key-map")
    v6_import.add_argument("--replace", action="store_true")

    v6_validation_create = sub.add_parser("native10-v6-validation-create", help="Build a trainer-invisible raw-input all-class validation artifact")
    v6_validation_create.add_argument("input", help="NPZ containing inputs and labels arrays")
    v6_validation_create.add_argument("output")
    v6_validation_create.add_argument("--state", default="./state")
    v6_validation_create.add_argument("--source", required=True)
    v6_validation_create.add_argument("--split", default="validation")
    v6_validation_create.add_argument("--min-samples-per-class", type=int, default=10)
    v6_validation_create.add_argument("--familywise-alpha", type=float, default=0.05)
    v6_validation_create.add_argument("--max-candidate-evaluations", type=int)
    v6_validation_create.add_argument("--min-discordant", type=int, default=20)
    v6_validation_create.add_argument("--minimum-net-wins", type=int, default=2)
    v6_validation_create.add_argument("--minimum-effect-rate", type=float, default=0.002)
    v6_validation_create.add_argument("--max-loss-per-class", type=int, default=1)
    v6_validation_create.add_argument("--max-loss-rate-per-class", type=float, default=0.10)
    v6_validation_create.add_argument("--role", choices=("selection", "replication"), default="selection")
    v6_validation_create.add_argument("--install", action="store_true")
    v6_validation_create.add_argument("--replace", action="store_true")

    v6_validation_import = sub.add_parser("native10-v6-validation-import", help="Install a prebuilt v0.6 validation artifact")
    v6_validation_import.add_argument("artifact")
    v6_validation_import.add_argument("--state", default="./state")
    v6_validation_import.add_argument("--role", choices=("selection", "replication"), default="selection")
    v6_validation_import.add_argument("--replace", action="store_true")

    v6_status = sub.add_parser("native10-v6-status", help="Show trainable Native10 status and parameter reachability")
    v6_status.add_argument("--state", default="./state")

    v6_queue = sub.add_parser("native10-v6-queue", help="Queue independent trainable Native10 candidate search")
    v6_queue.add_argument("--state", default="./state")
    v6_queue.add_argument("--shard")
    v6_queue.add_argument("--demo", action="store_true")
    v6_queue.add_argument("--operation", choices=("expert_train", "branch_train", "repair", "scout_train", "memory_train", "field_train"), default="expert_train")
    v6_queue.add_argument("--category", type=int, default=0, help="Category, or field block for field_train")
    v6_queue.add_argument("--subset-seed", type=int, default=7)
    v6_queue.add_argument("--search-candidates", type=int, default=4)
    v6_queue.add_argument("--verification-quorum", type=int, default=2)
    v6_queue.add_argument("--optimizer-steps", type=int, default=24)
    v6_queue.add_argument("--learning-rate", type=float, default=0.04)

    v6_export = sub.add_parser("native10-v6-export-int8", help="Export the v0.6 canonical model as INT8")
    v6_export.add_argument("output")
    v6_export.add_argument("--state", default="./state")

    v6_checkpoint = sub.add_parser("native10-v6-export-checkpoint", help="Export the v0.6 canonical checkpoint")
    v6_checkpoint.add_argument("output")
    v6_checkpoint.add_argument("--state", default="./state")

    v6_baseline = sub.add_parser("native10-v6-baseline-import", help="Install a provenance-bound external baseline result without training it")
    v6_baseline.add_argument("artifact")
    v6_baseline.add_argument("--state", default="./state")
    v6_baseline.add_argument("--replace", action="store_true")

    v6_evaluate = sub.add_parser("native10-v6-evaluate", help="Evaluate the canonical checkpoint on an NPZ without training")
    v6_evaluate.add_argument("input", help="NPZ containing inputs and labels arrays")
    v6_evaluate.add_argument("output")
    v6_evaluate.add_argument("--state", default="./state")
    v6_evaluate.add_argument("--dataset", required=True)
    v6_evaluate.add_argument("--split", default="test")
    v6_evaluate.add_argument("--source", required=True)

    v6_compare = sub.add_parser("native10-v6-compare-baseline", help="Compare a v0.6 evaluation report with the installed external baseline")
    v6_compare.add_argument("evaluation")
    v6_compare.add_argument("output")
    v6_compare.add_argument("--state", default="./state")

    args = parser.parse_args()
    if args.command in {None, "app"}:
        from dendriswarm.dashboard.server import run_dashboard
        run_dashboard(
            dashboard_state=Path(getattr(args, "state", str(Path.home() / ".dendriswarm" / "dashboard"))).expanduser(),
            seed_state=Path(args.seed_state).expanduser() if getattr(args, "seed_state", None) else None,
            operator_state=Path(args.operator_state).expanduser() if getattr(args, "operator_state", None) else None,
            host=getattr(args, "host", "127.0.0.1"),
            port=getattr(args, "port", 8788),
            open_browser=not getattr(args, "no_browser", False),
        )
    elif args.command == "coordinator":
        try:
            import uvicorn
            from dendriswarm.coordinator.app import create_app
        except ImportError as exc:
            raise SystemExit(
                "Coordinator dependencies are not installed. Run: pip install 'dendriswarm[coordinator]'"
            ) from exc
        uvicorn.run(
            create_app(
                args.state,
                args.bootstrap,
                args.lease_seconds,
                args.inference_audit_rate,
                args.enable_leverage,
            ),
            host=args.host,
            port=args.port,
        )
    elif args.command == "seed":
        state = Path(args.state).expanduser()
        _configure_seed(state, args)
        SeedNode(
            args.coordinator, state, args.poll,
            expected_coordinator_fingerprint=args.coordinator_fingerprint,
            allow_insecure_http=args.allow_insecure_http,
        ).run(args.max_tasks)
    elif args.command == "seed-config":
        print(json.dumps(_configure_seed(Path(args.state).expanduser(), args), indent=2))
    elif args.command in {"seed-pause", "seed-resume"}:
        state = Path(args.state).expanduser()
        policy = SeedPolicyStore(state / "seed-config.json").update(paused=args.command == "seed-pause")
        print(json.dumps(policy.model_dump(mode="json"), indent=2))
    elif args.command == "seed-local-status":
        state = Path(args.state).expanduser()
        status_path = state / "seed-status.json"
        if status_path.exists():
            print(status_path.read_text(), end="")
        else:
            policy = SeedPolicyStore(state / "seed-config.json").load()
            print(json.dumps({"state": "not-running", "policy": policy.model_dump(mode="json")}, indent=2))
    elif args.command == "doctor":
        from dendriswarm import __version__
        from dendriswarm.core.resources import effective_limits
        from dendriswarm.worker.resources import detect_capabilities

        state = Path(args.state).expanduser()
        policy = SeedPolicyStore(state / "seed-config.json").load()
        capabilities = detect_capabilities(state, policy)
        print(json.dumps({
            "dendriswarm_version": __version__,
            "seed_supported": True,
            "gpu_required": False,
            "backend": "numpy-cpu",
            "policy": policy.model_dump(mode="json"),
            "capabilities": capabilities.model_dump(mode="json"),
            "effective_limits": effective_limits(capabilities, policy),
        }, indent=2))
    elif args.command == "cifar100-download":
        from dendriswarm.v7.cifar100 import download_official_archive
        print(json.dumps(download_official_archive(args.output), indent=2))
    elif args.command in {"cifar100-prepare", "cifar100-init", "cifar100-status", "cifar100-plan", "cifar100-queue-next", "cifar100-evaluate-test"}:
        from dendriswarm.coordinator.service import CoordinatorService
        service = CoordinatorService(Path(args.state).expanduser())
        campaign = service.cifar100
        if args.command == "cifar100-prepare":
            result = campaign.prepare_dataset(
                args.source, seed=args.seed, holdout_per_class=args.holdout_per_class, replace=args.replace
            )
        elif args.command == "cifar100-init":
            checkpoint = json.loads(Path(args.checkpoint).read_text()) if args.checkpoint else None
            result = campaign.initialize_model(seed=args.seed, checkpoint=checkpoint, replace=args.replace)
        elif args.command == "cifar100-status":
            result = campaign.status()
        elif args.command == "cifar100-plan":
            result = campaign.plan_next(search_candidates=args.search_candidates, sample_budget=args.sample_budget)
        elif args.command == "cifar100-queue-next":
            result = campaign.queue_next(
                search_candidates=args.search_candidates, sample_budget=args.sample_budget,
                optimizer_steps=args.optimizer_steps, learning_rate=args.learning_rate,
                verification_quorum=args.verification_quorum,
            )
        else:
            result = campaign.evaluate_test(source=args.source)
            output_path = Path(args.output)
            output_path.write_text(json.dumps(result, sort_keys=True))
            result = {"written": str(output_path), **result}
        print(json.dumps(result, indent=2))
    elif args.command in {
        "native10-init", "native10-import", "native10-validation-create",
        "native10-validation-import", "native10-status", "native10-queue",
        "native10-export-int8", "native10-export-checkpoint",
    }:
        from dendriswarm.coordinator.service import CoordinatorService

        service = CoordinatorService(Path(args.state).expanduser())
        if args.command == "native10-init":
            result = service.native10.initialize(
                args.profile, input_width=args.input_width, seed=args.seed, replace=args.replace
            )
            print(json.dumps(result, indent=2))
        elif args.command == "native10-import":
            from dendriswarm.v5.native10 import Native10Config, load_external_checkpoint
            if args.profile == "compact":
                config = Native10Config.compact_demo(seed=args.seed)
                if args.input_width is not None and args.input_width != config.input_width:
                    config = Native10Config(**{**config.as_dict(), "input_width": int(args.input_width)})
            else:
                config = Native10Config(input_width=int(args.input_width or 3072), seed=args.seed)
            key_map = json.loads(Path(args.key_map).read_text()) if args.key_map else None
            model = load_external_checkpoint(args.checkpoint, config=config, key_map=key_map)
            result = service.native10.store.import_checkpoint(model.artifact(), replace=args.replace)
            print(json.dumps(result, indent=2))
        elif args.command == "native10-validation-create":
            import numpy as np
            from dendriswarm.v5.validation import GlobalValidationPolicy, make_global_validation_artifact
            with np.load(args.input, allow_pickle=False) as archive:
                representations = np.asarray(archive["representations"], dtype=np.float32)
                labels = np.asarray(archive["labels"], dtype=np.int64)
            policy = GlobalValidationPolicy(
                min_samples_per_class=args.min_samples_per_class,
                minimum_net_wins=args.minimum_net_wins,
                max_loss_per_class=args.max_loss_per_class,
                max_loss_rate_per_class=args.max_loss_rate_per_class,
                max_candidate_evaluations=args.max_candidate_evaluations,
            )
            artifact = make_global_validation_artifact(
                service.native10.store.model().config, representations, labels,
                source=args.source, split=args.split, policy=policy,
            )
            output = Path(args.output)
            output.write_text(json.dumps(artifact, sort_keys=True))
            response = {"written": str(output), "sha256": artifact["sha256"], "sample_count": artifact["sample_count"]}
            if args.install:
                response["installed"] = service.native10.store.set_global_validation(artifact, replace=args.replace)
            print(json.dumps(response, indent=2))
        elif args.command == "native10-validation-import":
            artifact = json.loads(Path(args.artifact).read_text())
            print(json.dumps(service.native10.store.set_global_validation(artifact, replace=args.replace), indent=2))
        elif args.command == "native10-status":
            print(json.dumps(service.native10.store.status(), indent=2))
        elif args.command == "native10-queue":
            if args.demo:
                result = service.native10.queue_demo_round(category=args.category, operation=args.operation)
            else:
                if not args.shard:
                    raise SystemExit("native10-queue requires --shard or --demo")
                shard = json.loads(Path(args.shard).read_text())
                result = service.native10.queue_mutation(
                    shard, operation=args.operation, category=args.category, subset_seed=args.subset_seed
                )
            print(json.dumps(result, indent=2))
        elif args.command == "native10-export-int8":
            output = Path(args.output)
            output.write_text(json.dumps(service.native10.store.model().export_int8(), sort_keys=True))
            print(json.dumps({"written": str(output), "source_root": service.native10.store.model().root}, indent=2))
        else:
            output = Path(args.output)
            output.write_text(service.native10.store.checkpoint_path.read_text())
            print(json.dumps({"written": str(output), "root": service.native10.store.model().root}, indent=2))
    elif args.command in {
        "native10-v6-init", "native10-v6-import", "native10-v6-validation-create",
        "native10-v6-validation-import", "native10-v6-status", "native10-v6-queue",
        "native10-v6-export-int8", "native10-v6-export-checkpoint",
        "native10-v6-baseline-import", "native10-v6-evaluate", "native10-v6-compare-baseline",
    }:
        from dendriswarm.coordinator.service import CoordinatorService
        service = CoordinatorService(Path(args.state).expanduser())
        native = service.native10_v6
        if args.command == "native10-v6-init":
            print(json.dumps(native.initialize(args.profile, input_width=args.input_width, seed=args.seed, replace=args.replace), indent=2))
        elif args.command == "native10-v6-import":
            from dendriswarm.v6.native10 import Native10Config, load_external_checkpoint
            config = Native10Config.compact_demo(seed=args.seed) if args.profile == "compact" else Native10Config(input_width=int(args.input_width or 3072), seed=args.seed)
            if args.profile == "compact" and args.input_width is not None and args.input_width != config.input_width:
                config = Native10Config(**{**config.as_dict(), "input_width": int(args.input_width)})
            key_map = json.loads(Path(args.key_map).read_text()) if args.key_map else None
            model = load_external_checkpoint(args.checkpoint, config=config, key_map=key_map)
            print(json.dumps(native.store.import_checkpoint(model.artifact(), replace=args.replace), indent=2))
        elif args.command == "native10-v6-validation-create":
            import numpy as np
            from dendriswarm.v6.validation import GlobalValidationPolicy, make_global_validation_artifact
            with np.load(args.input, allow_pickle=False) as archive:
                inputs = np.asarray(archive["inputs"], dtype=np.float32)
                labels = np.asarray(archive["labels"], dtype=np.int64)
            policy = GlobalValidationPolicy(
                min_samples_per_class=args.min_samples_per_class, familywise_alpha=args.familywise_alpha,
                max_candidate_evaluations=(args.max_candidate_evaluations if args.max_candidate_evaluations is not None else (20 if args.role == "selection" else 1)), min_discordant=args.min_discordant,
                minimum_net_wins=args.minimum_net_wins, minimum_effect_rate=args.minimum_effect_rate,
                max_loss_per_class=args.max_loss_per_class, max_loss_rate_per_class=args.max_loss_rate_per_class,
            )
            artifact = make_global_validation_artifact(native.store.model().config, inputs, labels, source=args.source, split=args.split, policy=policy)
            output_path = Path(args.output); output_path.write_text(json.dumps(artifact, sort_keys=True))
            response = {"written": str(output_path), "sha256": artifact["sha256"], "sample_count": artifact["sample_count"]}
            if args.install:
                installer = native.store.set_global_validation if args.role == "selection" else native.store.set_replication_validation
                response["installed"] = installer(artifact, replace=args.replace)
            response["role"] = args.role
            print(json.dumps(response, indent=2))
        elif args.command == "native10-v6-validation-import":
            artifact = json.loads(Path(args.artifact).read_text())
            installer = native.store.set_global_validation if args.role == "selection" else native.store.set_replication_validation
            print(json.dumps(installer(artifact, replace=args.replace), indent=2))
        elif args.command == "native10-v6-status":
            value = native.store.status()
            from dendriswarm.v6.native10 import parameter_reachability
            if value.get("initialized"):
                value["parameter_reachability"] = parameter_reachability(native.store.model().config)
            print(json.dumps(value, indent=2))
        elif args.command == "native10-v6-queue":
            if args.demo:
                result = native.queue_demo_round(category=args.category, operation=args.operation)
            else:
                if not args.shard:
                    raise SystemExit("native10-v6-queue requires --shard or --demo")
                shard = json.loads(Path(args.shard).read_text())
                result = native.queue_mutation(
                    shard, operation=args.operation, category=args.category, subset_seed=args.subset_seed,
                    search_candidates=args.search_candidates, verification_quorum=args.verification_quorum,
                    optimizer_steps=args.optimizer_steps, learning_rate=args.learning_rate,
                )
            print(json.dumps(result, indent=2))
        elif args.command == "native10-v6-export-int8":
            output_path = Path(args.output); output_path.write_text(json.dumps(native.store.model().export_int8(), sort_keys=True))
            print(json.dumps({"written": str(output_path), "source_root": native.store.model().root}, indent=2))
        elif args.command == "native10-v6-export-checkpoint":
            output_path = Path(args.output); output_path.write_text(native.store.checkpoint_path.read_text())
            print(json.dumps({"written": str(output_path), "root": native.store.model().root}, indent=2))
        elif args.command == "native10-v6-baseline-import":
            artifact = json.loads(Path(args.artifact).read_text())
            print(json.dumps(native.store.set_baseline_reference(artifact, replace=args.replace), indent=2))
        elif args.command == "native10-v6-evaluate":
            import numpy as np
            from dendriswarm.v6.benchmark import evaluate_checkpoint
            with np.load(args.input, allow_pickle=False) as archive:
                inputs = np.asarray(archive["inputs"], dtype=np.float32)
                labels = np.asarray(archive["labels"], dtype=np.int64)
            report = evaluate_checkpoint(
                native.store.model(), inputs, labels, dataset=args.dataset, split=args.split, source=args.source
            )
            output_path = Path(args.output); output_path.write_text(json.dumps(report, sort_keys=True))
            print(json.dumps({"written": str(output_path), "sha256": report["sha256"], "value": report["value"]}, indent=2))
        else:
            from dendriswarm.v6.benchmark import compare_with_baseline
            evaluation = json.loads(Path(args.evaluation).read_text())
            comparison = compare_with_baseline(evaluation, native.store.baseline_reference())
            output_path = Path(args.output); output_path.write_text(json.dumps(comparison, sort_keys=True))
            print(json.dumps({"written": str(output_path), **comparison}, indent=2))
    elif args.command == "status":
        print(json.dumps(httpx.get(f"{args.coordinator.rstrip('/')}/v1/stats").json(), indent=2))
    elif args.command == "audit":
        print(json.dumps(httpx.get(f"{args.coordinator.rstrip('/')}/v1/audit/checkpoint").json(), indent=2))
    elif args.command == "infer":
        submit_and_wait(args.coordinator.rstrip("/"), Path(args.state).expanduser(), args.features, args.wait)
    elif args.command == "infer-sample":
        base = args.coordinator.rstrip("/")
        sample_response = httpx.get(f"{base}/v1/samples/digit/{args.index}", timeout=30.0)
        sample_response.raise_for_status()
        sample_value = sample_response.json()
        print(f"Expected label: {sample_value['label']}")
        submit_and_wait(base, Path(args.state).expanduser(), sample_value["features"], args.wait)
