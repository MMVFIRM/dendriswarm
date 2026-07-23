from __future__ import annotations

import os
import platform
import shutil
import sys
import time
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from dendriswarm.core.models import NodeCapabilities, SeedPolicy
from dendriswarm.core.resources import effective_limits


def _memory() -> tuple[int, int]:
    try:
        import psutil

        memory = psutil.virtual_memory()
        return int(memory.total / 1024 / 1024), int(memory.available / 1024 / 1024)
    except Exception:
        return 1024, 768


def _physical_cpus() -> int | None:
    try:
        import psutil

        return psutil.cpu_count(logical=False) or None
    except Exception:
        return None


def _accelerators() -> list[str]:
    # The v0.4 reference executor is deliberately CPU-portable. Record
    # user-declared accelerator presence for future backends without importing
    # heavyweight optional frameworks or claiming those devices are used.
    declared = [item.strip() for item in os.getenv("DENDRISWARM_ACCELERATORS", "").split(",") if item.strip()]
    return ["cpu", *[item for item in declared if item != "cpu"]]


def benchmark_numpy(cpu_threads: int) -> float:
    rng = np.random.default_rng(404)
    x = rng.normal(size=(256, 64))
    c = rng.normal(size=(64, 64))
    operations = x.shape[0] * c.shape[0] * x.shape[1]
    rounds = 3
    started = time.perf_counter()
    with threadpool_limits(limits=max(1, cpu_threads)):
        for _ in range(rounds):
            ((x[:, None, :] - c[None, :, :]) ** 2).sum(axis=2)
    elapsed = max(time.perf_counter() - started, 1e-9)
    return float(operations * rounds / elapsed)


def detect_capabilities(state_dir: Path, policy: SeedPolicy) -> NodeCapabilities:
    logical = os.cpu_count() or 1
    memory_total, memory_available = _memory()
    disk_free = int(shutil.disk_usage(state_dir).free / 1024 / 1024)
    provisional = NodeCapabilities(
        cpu_count=logical,
        physical_cpu_count=_physical_cpus(),
        memory_mb=memory_total,
        memory_available_mb=memory_available,
        disk_free_mb=disk_free,
        accelerator="cpu",
        accelerators=_accelerators(),
        platform=platform.platform(),
        machine=platform.machine().lower() or "unknown",
        python_version=platform.python_version(),
        supported_backends=["numpy-cpu"],
        tags=["reference-runtime-v2", "deterministic-v2", "portable-numpy-v1", "numeric-f64-r12", "independent-search-v1", "blind-global-verification-v2"],
    )
    threads = int(effective_limits(provisional, policy)["cpu_threads"])
    return provisional.model_copy(update={"benchmark_units_per_second": benchmark_numpy(threads)})


def refresh_dynamic_resources(
    capabilities: NodeCapabilities, state_dir: Path
) -> NodeCapabilities:
    memory_total, memory_available = _memory()
    disk_free = int(shutil.disk_usage(state_dir).free / 1024 / 1024)
    return capabilities.model_copy(update={
        "memory_mb": memory_total,
        "memory_available_mb": memory_available,
        "disk_free_mb": disk_free,
    })


def local_run_condition(policy: SeedPolicy) -> tuple[bool, str]:
    if policy.paused:
        return False, "paused-by-user"
    try:
        import psutil

        battery = psutil.sensors_battery()
        if battery is not None and not battery.power_plugged:
            if not policy.allow_on_battery:
                return False, "on-battery"
            if battery.percent is not None and battery.percent < policy.min_battery_percent:
                return False, "battery-below-threshold"
        if psutil.cpu_percent(interval=0.05) > policy.max_system_cpu_percent:
            return False, "system-busy"
    except Exception:
        pass
    return True, "ready"


def cooldown_seconds(duration_seconds: float, capabilities: NodeCapabilities, policy: SeedPolicy) -> float:
    duty_cycle = float(effective_limits(capabilities, policy)["duty_cycle"])
    if duty_cycle >= 1.0:
        return 0.0
    return max(0.0, duration_seconds * (1.0 / max(duty_cycle, 1e-6) - 1.0))
