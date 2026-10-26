"""Конфигурация требует буквального разрешения; команды по умолчанию локальны."""
import json

import pytest
from pydantic import ValidationError

from hoi4_commander.cli import main
from hoi4_commander.config import Settings
from hoi4_commander.fixtures import create_demo
from hoi4_commander.models import ClickAction


@pytest.mark.parametrize("value", ["true", "yes", 1, "false"])
def test_external_opt_in_does_not_coerce_strings_or_numbers(value):
    with pytest.raises(ValidationError):
        Settings(allow_external_screenshots=value)


def test_live_requires_calibration():
    with pytest.raises(ValidationError):
        Settings(mode="live")


def test_explanation_rejects_terminal_escape_sequences():
    with pytest.raises(ValidationError):
        ClickAction(target_id="focus_menu", reason="Текст\x1b]52;clipboard")


def test_cli_inspects_saved_image_without_input_or_network(tmp_path, capsys):
    manifest = create_demo(tmp_path / "fixtures")
    assert main(["inspect", "--manifest", str(manifest), "--image", str(manifest.parent / "frame_00.png")]) == 0
    result = json.loads(capsys.readouterr().out)
    assert "focus_menu" in result["targets"]
    assert result["ambiguous"] == []


def test_cli_mock_input_diagnostic(capsys):
    assert main(["input-test"]) == 0
    assert "Ввод в ОС не выполнялся" in capsys.readouterr().out
