from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from dendriswarm.core.models import SeedPolicy, TaskKind


class SeedPolicyStore:
    """Atomic, hot-reloadable seeding policy persisted beside node identity."""

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def load(self) -> SeedPolicy:
        if not self.path.exists():
            policy = SeedPolicy()
            self.save(policy)
            return policy
        try:
            return SeedPolicy.model_validate(json.loads(self.path.read_text()))
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid seed policy at {self.path}: {exc}") from exc

    def save(self, policy: SeedPolicy) -> None:
        value = json.dumps(policy.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"
        fd, temporary = tempfile.mkstemp(prefix=f".{self.path.name}.", dir=self.path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(value)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.path)
        finally:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass

    def update(self, **changes: Any) -> SeedPolicy:
        current = self.load().model_dump(mode="python")
        if "allowed_task_kinds" in changes and changes["allowed_task_kinds"] is not None:
            changes["allowed_task_kinds"] = [TaskKind(value) for value in changes["allowed_task_kinds"]]
        current.update({key: value for key, value in changes.items() if value is not None})
        policy = SeedPolicy.model_validate(current)
        self.save(policy)
        return policy
