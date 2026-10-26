"""Локальный журнал без ключей, ответов API и снимков экрана."""
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4


class Journal:
    def __init__(self, directory: Path, secret_env_names=()):
        self._secrets = tuple(os.environ[name] for name in secret_env_names if os.environ.get(name))
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / f"run-{uuid4().hex}.jsonl"

    def redact(self, text: str):
        for secret in self._secrets:
            text = text.replace(secret, "[СКРЫТО]")
        return text

    def write(self, event: str, **data):
        def clean(value):
            if isinstance(value, str):
                return self.redact(value)
            if isinstance(value, dict):
                return {self.redact(str(k)): clean(v) for k, v in value.items()}
            if isinstance(value, (list, tuple)):
                return [clean(v) for v in value]
            return value
        record = clean({"time": datetime.now(timezone.utc).isoformat(), "event": event, **data})
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
