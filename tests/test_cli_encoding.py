"""Регрессия русскоязычного вывода в перенаправленном терминале Windows."""
import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize("arguments,expected", [
    (["input-test"], "Ввод в ОС не выполнялся"),
    (["--help"], "безопасный прототип"),
])
def test_cli_outputs_utf8_despite_legacy_pipe_encoding(arguments, expected):
    environment = {**os.environ, "PYTHONIOENCODING": "cp1252", "PYTHONUTF8": "0"}
    result = subprocess.run([sys.executable, "-m", "hoi4_commander.cli", *arguments],
                            cwd=Path(__file__).resolve().parents[1], env=environment,
                            capture_output=True, timeout=20, check=False)
    assert result.returncode == 0, result.stderr.decode("utf-8", errors="replace")
    assert expected in result.stdout.decode("utf-8")
    assert "UnicodeEncodeError" not in result.stderr.decode("utf-8")
