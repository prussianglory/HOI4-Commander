"""Независимые запреты, согласие пользователя и глобальная остановка F10."""
import ctypes
import json
import sys
import threading
import time
from pathlib import Path

from .models import Observation, Target, WindowInfo


class SafetyError(RuntimeError):
    pass


class StopToken:
    def __init__(self):
        self._event = threading.Event()

    def stop(self):
        self._event.set()

    @property
    def stopped(self):
        return self._event.is_set()

    def check(self):
        if self.stopped:
            raise SafetyError("Выполнение остановлено пользователем")

    def wait(self, seconds):
        self._event.wait(seconds)
        self.check()


class ConsentStore:
    def __init__(self, path: Path):
        self.path = path

    def grant(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps({"version": 1, "live_game_input": True}), encoding="utf-8")

    def require(self):
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            data = None
        if data != {"version": 1, "live_game_input": True}:
            raise SafetyError("Сначала явно подтвердите управление реальной игрой")


REQUIRED_ANCHORS = frozenset({"nation_germany", "date_1936_01_01", "game_paused"})


def validate_observation(observation: Observation, max_age: float):
    age = time.monotonic() - observation.captured_at
    if not 0 <= age <= max_age:
        raise SafetyError("Скриншот устарел")
    if REQUIRED_ANCHORS - observation.targets.keys() or REQUIRED_ANCHORS & observation.ambiguous:
        raise SafetyError("Не подтверждены Германия, дата 01.01.1936 и пауза игры")


def guard_target(backend, observation: Observation, target: Target, stop: StopToken):
    stop.check()
    current: WindowInfo = backend.current_window()
    if current != observation.window or current.process_name.lower() != "hoi4.exe":
        raise SafetyError("Активное окно HOI4 изменилось")
    r = target.rect
    if target.frame_id != observation.frame_id:
        raise SafetyError("Цель принадлежит другому скриншоту")
    if not (r.x >= 0 and r.y >= 0 and r.x + r.width <= current.client.width
            and r.y + r.height <= current.client.height):
        raise SafetyError("Цель выходит за границы клиентской области HOI4")
    return (current.client.x + target.center[0], current.client.y + target.center[1])


class EmergencyHotkey:
    """F10 регистрируется в отдельном потоке; сбой регистрации запрещает ввод."""
    def __init__(self, stop: StopToken):
        self.stop = stop
        self._quit = threading.Event()
        self._ready = threading.Event()
        self._error = None
        self._tested = threading.Event()
        self._armed = threading.Event()
        self._thread = None

    def start(self):
        if sys.platform != "win32":
            raise SafetyError("Глобальная остановка реального ввода доступна только в Windows")
        self._thread = threading.Thread(target=self._listen, daemon=True)
        self._thread.start()
        if not self._ready.wait(3) or self._error:
            self.close()
            raise SafetyError("Не удалось зарегистрировать F10; реальный ввод запрещён")

    def verify_before_input(self, on_status, timeout=20):
        on_status("Проверка остановки: выведите HOI4 на передний план и нажмите F10 в течение 20 секунд. Ввод пока запрещён.")
        deadline = time.monotonic() + timeout
        while not self._tested.is_set():
            self.stop.check()
            if time.monotonic() >= deadline:
                raise SafetyError("F10 не проверена: ввод запрещён")
            self._tested.wait(0.02)
        self.stop.check()
        self._armed.set()
        on_status("F10 проверена. Следующее нажатие F10 немедленно запрещает новые действия.")

    def _handle_hotkey(self):
        if self._armed.is_set():
            self.stop.stop()
        else:
            self._tested.set()

    def _listen(self):
        user32 = None
        registered = False
        try:
            from ctypes import wintypes
            user32 = ctypes.WinDLL("user32", use_last_error=True)
            user32.RegisterHotKey.argtypes = [wintypes.HWND, ctypes.c_int, wintypes.UINT, wintypes.UINT]
            user32.RegisterHotKey.restype = wintypes.BOOL
            user32.PeekMessageW.argtypes = [ctypes.POINTER(wintypes.MSG), wintypes.HWND, wintypes.UINT, wintypes.UINT, wintypes.UINT]
            user32.PeekMessageW.restype = wintypes.BOOL
            user32.UnregisterHotKey.argtypes = [wintypes.HWND, ctypes.c_int]
            if not user32.RegisterHotKey(None, 1, 0x4000, 0x79):
                self._error = True
                return
            registered = True
            self._ready.set()
            msg = wintypes.MSG()
            while not self._quit.wait(0.01):
                while user32.PeekMessageW(ctypes.byref(msg), None, 0, 0, 1):
                    if msg.message == 0x0312:
                        self._handle_hotkey()
        except Exception:
            # Потеря наблюдателя остановки всегда запрещает последующий ввод.
            self._error = True
            self.stop.stop()
        finally:
            if registered:
                try:
                    if not user32.UnregisterHotKey(None, 1):
                        self._error = True
                except Exception:
                    self._error = True
            if self._error or not self._quit.is_set():
                self.stop.stop()
            # Разблокируем start даже при отказе загрузки Win32 или регистрации.
            self._ready.set()

    def close(self):
        self._quit.set()
        if self._thread:
            self._thread.join(timeout=2)
