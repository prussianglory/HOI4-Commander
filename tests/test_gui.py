"""Проверки панели в виртуальном Qt без доступа к экрану и реальному вводу."""
import os
import threading
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication, QMessageBox

from hoi4_commander.config import Settings
from hoi4_commander.gui import CommanderWindow
from hoi4_commander.models import Observation, Rect, Target, WindowInfo
from hoi4_commander.safety import ConsentStore, SafetyError


@pytest.fixture(scope="module")
def app():
    instance = QApplication.instance() or QApplication([])
    yield instance


def pump(app, predicate, timeout=3):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        app.processEvents()
        if predicate():
            return
        time.sleep(0.005)
    pytest.fail("Панель не завершила ожидаемый переход состояния")


@pytest.fixture
def window(app, tmp_path):
    panel = CommanderWindow(Settings(consent_file=tmp_path / "consent.json"), lambda *_: None)
    panel.show()
    app.processEvents()
    yield panel
    panel._stop()
    pump(app, lambda: not panel._busy)
    panel.close()
    app.processEvents()


def test_config_cannot_enable_live_or_external_without_ui(app, tmp_path):
    settings = Settings(
        mode="live", allow_external_screenshots=True, allow_local_screenshots=True,
        template_manifest=tmp_path / "manifest.json",
        consent_file=tmp_path / "consent.json",
    )
    received = []

    class Controller:
        def run(self):
            pass

        def stop(self):
            pass

    def factory(current, callback):
        received.append(current)
        return Controller()

    panel = CommanderWindow(settings, factory)
    assert not panel.live.isChecked()
    assert not panel.external.isChecked()
    assert not panel.local_screenshots.isChecked()
    assert "dry-run" in panel.start_button.text()
    panel.start_button.click()
    pump(app, lambda: not panel._busy)
    assert len(received) == 1
    assert received[0].mode == "dry_run"
    assert received[0].allow_external_screenshots is False
    assert received[0].allow_local_screenshots is False
    assert not settings.consent_file.exists()
    assert panel.start_button.isEnabled()
    assert not panel.stop_button.isEnabled()
    panel.close()


def test_plan_draws_target_and_keeps_source_frame_unchanged(window):
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    target = Target(id="focus_menu", rect=Rect(x=70, y=30, width=20, height=20), confidence=1, frame_id="test")
    observation = Observation(
        frame_id="test", captured_at=time.monotonic(),
        window=WindowInfo(handle=1, title="Тест", process_name="hoi4.exe", client=Rect(x=0, y=0, width=200, height=100)),
        image=image, targets={target.id: target},
    )
    window._on_event({"type": "plan", "observation": observation, "target": target, "reason": "Проверить фокус"})
    pixmap = window.preview.pixmap()
    assert pixmap is not None and not pixmap.isNull()
    displayed = pixmap.toImage()
    red_pixels = sum(
        displayed.pixelColor(x, y).red() > 180
        and displayed.pixelColor(x, y).green() < 100
        for y in range(displayed.height()) for x in range(displayed.width())
    )
    assert red_pixels > 30
    assert not image.any()
    assert "Проверить фокус" in window.log.toPlainText()


def test_stop_reaches_background_controller(window, app):
    started = threading.Event()
    stopped = threading.Event()

    class Controller:
        def run(self):
            started.set()
            assert stopped.wait(3)

        def stop(self):
            stopped.set()

    window._factory = lambda *_: Controller()
    window.start_button.click()
    pump(app, started.is_set)
    assert window._busy and window.stop_button.isEnabled()
    assert not window.start_button.isEnabled()
    assert not window.local_screenshots.isEnabled()
    assert not window.external.isEnabled()
    window.stop_button.click()
    pump(app, lambda: not window._busy)
    assert stopped.is_set()
    assert window.start_button.isEnabled()
    assert window.local_screenshots.isEnabled()
    assert window.external.isEnabled()


def test_consent_requires_explicit_yes(window, monkeypatch):
    path = window._settings.consent_file
    with pytest.raises(SafetyError):
        ConsentStore(path).require()
    monkeypatch.setattr(QMessageBox, "question", lambda *_: QMessageBox.StandardButton.No)
    window.confirm_button.click()
    assert not path.exists()
    monkeypatch.setattr(QMessageBox, "question", lambda *_: QMessageBox.StandardButton.Yes)
    window.confirm_button.click()
    ConsentStore(path).require()
    assert not window.live.isChecked()
    assert not window.external.isChecked()


def test_worker_error_does_not_display_raw_exception(window, app):
    def failing_factory(*_):
        raise RuntimeError("секретный-токен-не-должен-появиться")

    window._factory = failing_factory
    window.start_button.click()
    pump(app, lambda: not window._busy)
    assert "Ошибка" in window.log.toPlainText()
    assert "секретный-токен" not in window.log.toPlainText()
    assert "секретный-токен" not in window.status.text()


def test_model_explanation_is_plain_text(window):
    window._on_event({"type": "status", "message": "<b>Объяснение</b>"})
    assert "<b>Объяснение</b>" in window.log.toPlainText()


@pytest.mark.parametrize("local, external", [(True, False), (False, True)])
def test_screenshot_permissions_are_explicit_and_independent(window, app, local, external):
    received = []

    class Controller:
        def run(self):
            pass

        def stop(self):
            pass

    def factory(current, callback):
        received.append(current)
        return Controller()

    window._factory = factory
    window.local_screenshots.setChecked(local)
    window.external.setChecked(external)
    window.start_button.click()
    pump(app, lambda: not window._busy)
    assert len(received) == 1
    assert received[0].allow_local_screenshots is local
    assert received[0].allow_external_screenshots is external
    assert received[0].mode == "dry_run"
