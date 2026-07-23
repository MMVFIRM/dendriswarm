from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from pydantic import Field, field_validator

from dendriswarm.core.models import StrictModel


class CampaignDefaults(StrictModel):
    search_candidates: int = Field(default=8, ge=2, le=64)
    sample_budget: int = Field(default=640, ge=100, le=5000)
    optimizer_steps: int = Field(default=36, ge=1, le=500)
    learning_rate: float = Field(default=0.03, gt=0.0, le=1.0)
    verification_quorum: int = Field(default=2, ge=2, le=8)


class DashboardConfig(StrictModel):
    coordinator_url: str = "http://127.0.0.1:8787"
    coordinator_fingerprint: str | None = Field(default=None, max_length=256)
    allow_insecure_http: bool = False
    seed_state: str = str(Path.home() / ".dendriswarm" / "seed")
    operator_state: str = str(Path.home() / ".dendriswarm" / "operator")
    auto_start_seed: bool = False
    auto_start_coordinator: bool = False
    campaign: CampaignDefaults = Field(default_factory=CampaignDefaults)
    refresh_seconds: float = Field(default=2.0, ge=0.5, le=30.0)

    @field_validator("coordinator_url")
    @classmethod
    def _coordinator_url(cls, value: str) -> str:
        normalized = value.strip().rstrip("/")
        if not normalized.startswith(("http://", "https://")):
            raise ValueError("coordinator URL must start with http:// or https://")
        return normalized

    @field_validator("seed_state", "operator_state")
    @classmethod
    def _state_path(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("state path cannot be empty")
        return str(Path(value).expanduser())


class DashboardConfigStore:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def load(self, *, defaults: dict[str, Any] | None = None) -> DashboardConfig:
        if self.path.exists():
            try:
                return DashboardConfig.model_validate(json.loads(self.path.read_text(encoding="utf-8")))
            except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
                raise ValueError(f"invalid dashboard config at {self.path}: {exc}") from exc
        config = DashboardConfig.model_validate(defaults or {})
        self.save(config)
        return config

    def save(self, config: DashboardConfig) -> None:
        encoded = json.dumps(config.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"
        fd, temporary = tempfile.mkstemp(prefix=f".{self.path.name}.", dir=self.path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.path)
        finally:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass

    def update(self, **changes: Any) -> DashboardConfig:
        current = self.load().model_dump(mode="python")
        for key, value in changes.items():
            if value is not None:
                current[key] = value
        config = DashboardConfig.model_validate(current)
        self.save(config)
        return config
