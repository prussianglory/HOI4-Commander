"""Настройки без хранения ключей API."""
import json
from pathlib import Path
from typing import Literal

from pydantic import Field, StrictBool, model_validator
from .models import StrictModel


class ProviderConfig(StrictModel):
    kind: Literal["local", "openai_compatible"] = "local"
    base_url: str = "https://api.openai.com/v1"
    model: str = "gpt-4.1-mini"
    api_key_env: str = "HOI4_MODEL_API_KEY"
    timeout_seconds: float = Field(default=30, gt=0, le=120)


class Settings(StrictModel):
    mode: Literal["dry_run", "live"] = "dry_run"
    allow_external_screenshots: StrictBool = False
    template_manifest: Path | None = None
    journal_dir: Path = Path("runtime/journal")
    consent_file: Path = Path("runtime/live-consent.json")
    frame_max_age_seconds: float = Field(default=2, gt=0, le=5)
    verification_timeout_seconds: float = Field(default=5, gt=0, le=30)
    provider: ProviderConfig = ProviderConfig()

    @model_validator(mode="after")
    def validate_live(self):
        if self.mode == "live" and self.template_manifest is None:
            raise ValueError("Для реальной игры необходим собственный набор шаблонов")
        return self

    @classmethod
    def load(cls, path: Path):
        return cls.model_validate(json.loads(path.read_text(encoding="utf-8")))
