from __future__ import annotations

import multiprocessing as mp
import queue
import time
from dataclasses import dataclass
from typing import Any

import psutil

from dendriswarm.core.models import NodeCapabilities, SeedPolicy, TaskKind, TaskRequirements
from dendriswarm.core.resources import effective_limits, node_can_run
from dendriswarm.worker.config import SeedPolicyStore
from dendriswarm.worker.executor import execute_task


class TaskExecutionCancelled(RuntimeError):
    pass


class TaskExecutionFailed(RuntimeError):
    pass


def _child(kind_value: str, payload: dict[str, Any], threads: int, result_queue: Any) -> None:
    try:
        result_queue.put(("ok", execute_task(TaskKind(kind_value), payload, cpu_threads=threads)))
    except BaseException as error:  # child boundary: serialize only a bounded message
        result_queue.put(("error", f"{type(error).__name__}: {error}"))


def execute_task_isolated(
    kind: TaskKind,
    payload: dict[str, Any],
    *,
    cpu_threads: int,
    memory_limit_mb: int,
    timeout_seconds: int,
    requirements: TaskRequirements,
    capabilities: NodeCapabilities,
    policy_store: SeedPolicyStore,
    poll_seconds: float = 0.25,
) -> dict[str, Any]:
    """Execute built-in work in a killable subprocess with active policy enforcement."""
    # ``spawn`` provides a clean interpreter boundary on every supported
    # platform and avoids inheriting coordinator/client threads or file
    # descriptors into volunteer work.  Fall back only on exotic runtimes
    # where spawn is unavailable.
    methods = mp.get_all_start_methods()
    context = mp.get_context("spawn" if "spawn" in methods else methods[0])
    result_queue = context.Queue(maxsize=1)
    process = context.Process(
        target=_child,
        args=(kind.value, payload, max(1, int(cpu_threads)), result_queue),
        daemon=True,
    )
    process.start()
    child = psutil.Process(process.pid)
    started = time.monotonic()
    reason: str | None = None
    result: tuple[str, Any] | None = None
    try:
        while process.is_alive():
            elapsed = time.monotonic() - started
            if elapsed > timeout_seconds:
                reason = "hard-timeout"
                break
            try:
                rss_mb = child.memory_info().rss / (1024 * 1024)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                rss_mb = 0.0
            if rss_mb > memory_limit_mb:
                reason = "hard-rss-limit"
                break
            policy: SeedPolicy = policy_store.load()
            eligible, eligibility_reason = node_can_run(kind, requirements, capabilities, policy)
            limits = effective_limits(capabilities, policy)
            if not eligible:
                reason = f"live-policy-change:{eligibility_reason}"
                break
            if int(limits["cpu_threads"]) < cpu_threads:
                reason = "live-policy-change:cpu-budget-reduced"
                break
            try:
                # Drain the queue before waiting for the child to exit. Large
                # artifacts can fill the multiprocessing pipe and otherwise
                # keep the child's queue feeder alive indefinitely.
                result = result_queue.get(timeout=max(0.05, poll_seconds))
                break
            except queue.Empty:
                pass
        if reason is not None:
            process.terminate()
            process.join(timeout=3.0)
            if process.is_alive():
                process.kill()
                process.join(timeout=1.0)
            raise TaskExecutionCancelled(reason)
        if result is None:
            try:
                result = result_queue.get(timeout=1.0)
            except queue.Empty as error:
                raise TaskExecutionFailed(f"worker subprocess exited with code {process.exitcode} without a result") from error
        process.join(timeout=3.0)
        status, value = result
        if status != "ok":
            raise TaskExecutionFailed(str(value))
        return value
    finally:
        if process.is_alive():
            process.terminate()
            process.join(timeout=1.0)
        result_queue.close()
