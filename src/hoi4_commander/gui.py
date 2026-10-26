"""Панель управления: фоновый цикл, остановка и визуализация сухого прогона."""
from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any

from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtGui import QColor, QCloseEvent, QImage, QPainter, QPen, QPixmap, QTextCursor
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from .config import Settings
from .models import Observation, Target


class _Worker(QObject):
    """Передаёт события в главный поток; виджеты из рабочего потока не меняет."""

    event = Signal(object)
    finished = Signal()
    failed = Signal()

    def __init__(self, settings: Settings, factory: Callable):
        super().__init__()
        self._settings = settings
        self._factory = factory
        self._controller: Any = None
        self._lock = threading.Lock()
        self._stop_requested = threading.Event()

    def stop(self) -> None:
        self._stop_requested.set()
        with self._lock:
            controller = self._controller
        if controller is not None:
            controller.stop()

    def run(self) -> None:
        try:
            controller = self._factory(self._settings, self.event.emit)
            with self._lock:
                self._controller = controller
            if self._stop_requested.is_set():
                controller.stop()
            else:
                controller.run()
        except Exception:
            # Текст исключений API может содержать чувствительные данные.
            self.failed.emit()
        finally:
            self.finished.emit()


class CommanderWindow(QMainWindow):
    """Все реальные действия требуют отдельного выбора и подтверждения."""

    def __init__(self, settings: Settings, controller_factory: Callable):
        super().__init__()
        self._settings = settings
        self._factory = controller_factory
        self._worker: _Worker | None = None
        self._thread: threading.Thread | None = None
        self._close_requested = False
        self._busy = False
        self.setWindowTitle("HOI4 AI Commander — мирное управление")
        self.resize(1020, 820)

        self.live = QCheckBox("Реальное управление HOI4 (Windows)")
        self.live.setChecked(False)
        self.local_screenshots = QCheckBox("Явно разрешаю передавать скриншоты локальной Ollama на этом компьютере")
        self.local_screenshots.setChecked(False)
        self.external = QCheckBox("Явно разрешаю отправку скриншотов во внешний API")
        self.external.setChecked(False)
        self.confirm_button = QPushButton("Подтвердить доступ к реальной игре")
        self.confirm_button.clicked.connect(self._confirm_live)
        self.start_button = QPushButton("Начать dry-run")
        self.start_button.clicked.connect(self._start)
        self.stop_button = QPushButton("СТОП")
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self._stop)
        self.live.toggled.connect(self._update_start_label)

        self.status = QLabel("По умолчанию: dry-run, отправка изображений выключена. F10 — экстренная остановка в Windows.")
        self.status.setWordWrap(True)
        self.status.setTextFormat(Qt.TextFormat.PlainText)
        self.preview = QLabel("Здесь появятся свежий кадр и проверенная целевая точка")
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setMinimumSize(640, 360)
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumHeight(190)
        controls = QHBoxLayout()
        for widget in (self.start_button, self.stop_button, self.confirm_button):
            controls.addWidget(widget)
        layout = QVBoxLayout()
        layout.addWidget(self.live)
        layout.addWidget(self.local_screenshots)
        layout.addWidget(self.external)
        layout.addLayout(controls)
        layout.addWidget(self.status)
        layout.addWidget(self.preview, 1)
        layout.addWidget(self.log)
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def _update_start_label(self) -> None:
        self.start_button.setText("Начать управление игрой" if self.live.isChecked() else "Начать dry-run")

    def _confirm_live(self) -> None:
        result = QMessageBox.question(
            self,
            "Разрешение реального ввода",
            "Агент сможет нажимать мышь и клавиши только в активном окне HOI4. "
            "Перед запуском нужны проверенные шаблоны вашего интерфейса и тесты остановки. "
            "F10 останавливает агента. Подтверждаете доступ к реальной игре? "
            "Это не разрешает передачу скриншотов локальной Ollama или внешнему API.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if result != QMessageBox.StandardButton.Yes:
            return
        try:
            from .safety import ConsentStore

            ConsentStore(self._settings.consent_file).grant()
        except Exception:
            self.status.setText("Не удалось сохранить подтверждение. Реальный ввод не разрешён.")
            return
        self.status.setText("Подтверждение сохранено. Для запуска отдельно включите реальное управление.")

    def _start(self) -> None:
        if self._busy:
            return
        try:
            current = Settings.model_validate({
                **self._settings.model_dump(),
                "mode": "live" if self.live.isChecked() else "dry_run",
                "allow_external_screenshots": self.external.isChecked(),
                "allow_local_screenshots": self.local_screenshots.isChecked(),
            })
        except Exception:
            self.status.setText("Некорректная конфигурация. Для игры укажите собственный манифест шаблонов.")
            return
        self._busy = True
        self._set_controls_running(True)
        self.status.setText("Работа началась. Можно остановить кнопкой СТОП или F10.")
        worker = _Worker(current, self._factory)
        worker.event.connect(self._on_event, Qt.ConnectionType.QueuedConnection)
        worker.failed.connect(self._on_failure, Qt.ConnectionType.QueuedConnection)
        worker.finished.connect(self._finished, Qt.ConnectionType.QueuedConnection)
        self._worker = worker
        self._thread = threading.Thread(target=worker.run, name="hoi4-controller", daemon=True)
        self._thread.start()

    def _set_controls_running(self, running: bool) -> None:
        for widget in (self.start_button, self.live, self.local_screenshots, self.external, self.confirm_button):
            widget.setEnabled(not running)
        self.stop_button.setEnabled(running)

    def _stop(self) -> None:
        if self._worker is not None:
            self._worker.stop()
            self.status.setText("Запрошена остановка. Новые действия запрещены; ожидаем завершения текущего запроса.")

    def _append_log(self, message: str) -> None:
        self.log.moveCursor(QTextCursor.MoveOperation.End)
        self.log.insertPlainText(message + "\n")

    def _on_failure(self) -> None:
        self._append_log("Ошибка: цикл остановлен. Проверьте журнал и конфигурацию; повторного ввода нет.")
        self.status.setText("Работа остановлена из-за ошибки.")

    def _finished(self) -> None:
        self._busy = False
        self._set_controls_running(False)
        self._append_log("Цикл завершён. Подробности результата находятся в журнале.")
        if self._close_requested:
            self.close()

    def _on_event(self, event: dict) -> None:
        if not isinstance(event, dict):
            return
        kind = event.get("type")
        if kind == "status":
            message = str(event.get("message", ""))
            self.status.setText(message)
            self._append_log(message)
        elif kind == "observation":
            observation = event.get("observation")
            image = observation.image if isinstance(observation, Observation) else event.get("image")
            if image is not None:
                self._show_frame(image)
        elif kind == "plan":
            observation = event.get("observation")
            target = event.get("target")
            reason = str(event.get("reason", ""))
            self._append_log(reason)
            if isinstance(observation, Observation) and isinstance(target, Target):
                self._show_frame(observation.image, target)

    def _show_frame(self, image, target: Target | None = None) -> None:
        # Копия отделяет изображение Qt от буфера наблюдателя в другом потоке.
        rgb = image[:, :, :3][:, :, ::-1].copy()
        height, width = rgb.shape[:2]
        frame = QImage(rgb.data, width, height, rgb.strides[0], QImage.Format.Format_RGB888).copy()
        if target is not None:
            painter = QPainter(frame)
            painter.setPen(QPen(QColor("red"), 3))
            x, y = target.center
            painter.drawEllipse(x - 9, y - 9, 18, 18)
            painter.drawText(x + 14, y, target.id)
            painter.end()
        self.preview.setPixmap(QPixmap.fromImage(frame).scaled(
            self.preview.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
        ))

    def closeEvent(self, event: QCloseEvent) -> None:
        if self._busy:
            self._close_requested = True
            self._stop()
            event.ignore()
        else:
            event.accept()


def run_gui(settings: Settings, controller_factory: Callable) -> int:
    """Запускает панель; фабрика получает настройки и потокобезопасный обработчик событий."""
    app = QApplication.instance() or QApplication([])
    window = CommanderWindow(settings, controller_factory)
    window.show()
    return app.exec()
