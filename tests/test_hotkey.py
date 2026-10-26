"""Логика проверки F10 тестируется без регистрации клавиш операционной системы."""
import pytest

from hoi4_commander.safety import EmergencyHotkey, SafetyError, StopToken


def test_f10_requires_proof_then_latches_stop():
    token = StopToken()
    hotkey = EmergencyHotkey(token)
    hotkey._handle_hotkey()
    assert not token.stopped
    messages = []
    hotkey.verify_before_input(messages.append, timeout=0.01)
    assert hotkey._armed.is_set()
    hotkey._handle_hotkey()
    assert token.stopped
    with pytest.raises(SafetyError):
        token.check()
    assert len(messages) == 2


def test_no_f10_no_arming():
    hotkey = EmergencyHotkey(StopToken())
    with pytest.raises(SafetyError, match="F10"):
        hotkey.verify_before_input(lambda _: None, timeout=0.001)
    assert not hotkey._armed.is_set()


def test_stop_during_preflight_blocks_arming():
    token = StopToken()
    token.stop()
    hotkey = EmergencyHotkey(token)
    with pytest.raises(SafetyError):
        hotkey.verify_before_input(lambda _: None, timeout=0.001)
    assert not hotkey._armed.is_set()


class _WinFunction:
    """Подмена функции ctypes допускает настройку argtypes и restype."""
    def __init__(self, callback):
        self.callback = callback

    def __call__(self, *args):
        return self.callback(*args)


class _FailingMessages:
    def __init__(self):
        self.unregistered = False
        self.RegisterHotKey = _WinFunction(lambda *_: True)
        self.PeekMessageW = _WinFunction(self._fail)
        self.UnregisterHotKey = _WinFunction(self._unregister)

    def _fail(self, *_):
        raise RuntimeError("Тестовый отказ Win32")

    def _unregister(self, *_):
        self.unregistered = True
        return True


def test_listener_failure_after_arming_latches_stop(monkeypatch):
    import threading
    import hoi4_commander.safety as safety

    token = StopToken()
    hotkey = EmergencyHotkey(token)
    hotkey._armed.set()
    user32 = _FailingMessages()
    monkeypatch.setattr(safety.ctypes, "WinDLL", lambda *_args, **_kwargs: user32, raising=False)
    listener = threading.Thread(target=hotkey._listen)
    listener.start()
    listener.join(timeout=1)
    assert not listener.is_alive()
    assert token.stopped
    assert hotkey._error
    assert user32.unregistered


def test_listener_startup_failure_unblocks_start_and_stops(monkeypatch):
    import hoi4_commander.safety as safety

    def fail(*_args, **_kwargs):
        raise OSError("Тестовый отказ загрузки библиотеки")

    hotkey = EmergencyHotkey(StopToken())
    monkeypatch.setattr(safety.ctypes, "WinDLL", fail, raising=False)
    hotkey._listen()
    assert hotkey._ready.is_set()
    assert hotkey._error
    assert hotkey.stop.stopped


def test_listener_normal_close_unregisters_without_failure(monkeypatch):
    import hoi4_commander.safety as safety

    hotkey = EmergencyHotkey(StopToken())
    user32 = _FailingMessages()
    monkeypatch.setattr(safety.ctypes, "WinDLL", lambda *_args, **_kwargs: user32, raising=False)
    hotkey._quit.set()
    hotkey._listen()
    assert user32.unregistered
    assert not hotkey._error
    assert not hotkey.stop.stopped
