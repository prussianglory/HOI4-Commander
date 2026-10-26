"""Команды диагностики используют выбранную модель без реального ввода."""
import json
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from hoi4_commander import cli
from hoi4_commander.config import ProviderConfig, Settings
from hoi4_commander.model_provider import LocalProvider, ProviderError, create_provider
from hoi4_commander.ollama_provider import OllamaModelInfo, OllamaProvider


@pytest.fixture
def config(tmp_path):
    path = tmp_path / "settings.json"
    settings = Settings(provider=ProviderConfig(kind="ollama"), allow_local_screenshots=True,
                        journal_dir=tmp_path / "journal")
    path.write_text(settings.model_dump_json(), encoding="utf-8")
    return path


def test_ollama_defaults_and_factory():
    settings = Settings(provider={"kind": "ollama"})
    assert settings.provider.base_url == "http://127.0.0.1:11434"
    assert settings.provider.model == "qwen2.5vl:7b"
    assert settings.provider.timeout_seconds == 180
    provider = create_provider(settings)
    assert isinstance(provider, OllamaProvider)
    assert not provider.allow_local_screenshots
    assert not provider.allow_external_screenshots
    assert ProviderConfig().base_url == "https://api.openai.com/v1"


@pytest.mark.parametrize("value", ["true", 1, "yes"])
def test_local_permission_requires_boolean(value):
    with pytest.raises(ValidationError):
        Settings(allow_local_screenshots=value)


def test_provider_check_does_not_call_inference(config, monkeypatch, capsys):
    class Provider:
        def probe(self):
            return OllamaModelInfo("0.12.0", "qwen2.5vl:7b", ("completion", "vision"))

        def choose(self, *_):
            raise AssertionError("Проверка доступности не должна передавать кадр")

    monkeypatch.setattr(cli, "create_provider", lambda _: Provider())
    assert cli.main(["provider-check", "--config", str(config)]) == 0
    text = capsys.readouterr().out
    assert "vision" in text and "Скриншоты не отправлялись" in text


def test_provider_check_failure_is_actionable(config, monkeypatch, capsys):
    class Provider:
        def probe(self):
            raise ProviderError("Ollama недоступна")

    monkeypatch.setattr(cli, "create_provider", lambda _: Provider())
    assert cli.main(["provider-check", "--config", str(config)]) == 2
    assert "Ollama недоступна" in capsys.readouterr().out


def test_model_test_uses_chosen_provider_and_never_constructs_executor(config, monkeypatch, capsys):
    called = []

    class Provider(LocalProvider):
        def choose(self, observation, step):
            called.append((observation, step))
            return super().choose(observation, step)

    monkeypatch.setattr(cli, "create_provider", lambda _: Provider())

    def forbidden(*_args, **_kwargs):
        raise AssertionError("Диагностика модели не должна создавать контроллер ввода")

    monkeypatch.setattr(cli, "build_controller", forbidden)
    assert cli.main(["model-test", "--config", str(config)]) == 0
    assert len(called) == 1 and called[0][1].id == "open_focus"
    assert "Ввод не выполнялся" in capsys.readouterr().out
    settings = Settings.load(config)
    logs = list(settings.journal_dir.glob("*.jsonl"))
    assert len(logs) == 1
    record = json.loads(logs[0].read_text(encoding="utf-8"))
    assert record["event"] == "model_test" and record["source"] == "synthetic"


def test_saved_model_test_requires_image_and_manifest_together(config, monkeypatch, capsys):
    monkeypatch.setattr(cli, "create_provider", lambda _: LocalProvider())
    assert cli.main(["model-test", "--config", str(config), "--image", "nothing.png"]) == 2
    assert "одновременно" in capsys.readouterr().out


def test_demo_config_cannot_enable_real_input_or_real_capture(config, monkeypatch, capsys):
    settings = Settings.load(config)
    data = {**settings.model_dump(mode="json"), "mode": "live", "template_manifest": "no-real-capture.json"}
    config.write_text(json.dumps(data), encoding="utf-8")
    captured = []

    def factory(current, callback, *, mock_execute):
        captured.append((current, mock_execute))
        return SimpleNamespace(run=lambda: SimpleNamespace(status="completed", completed_steps=10),
                               journal=SimpleNamespace(path="mock-journal.jsonl"))

    monkeypatch.setattr(cli, "build_controller", factory)
    assert cli.main(["demo", "--config", str(config), "--mock-execute"]) == 0
    current, mock = captured[0]
    assert current.mode == "dry_run" and current.template_manifest is None
    assert mock and current.provider.kind == "ollama"
    assert current.allow_local_screenshots and not current.allow_external_screenshots
