"""Ограниченный ввод с повторным наблюдением прямо перед действием."""
from .models import ClickAction, KeyAction, Observation, Plan
from .safety import SafetyError, StopToken, guard_target, validate_observation


class MockInput:
    def __init__(self, on_click=None):
        self.events = []
        self.on_click = on_click
        self.cursor = (0, 0)

    def assert_neutral(self):
        pass

    def move_to(self, x, y):
        self.cursor = (x, y)
        self.events.append(("move", x, y))

    def position(self):
        return self.cursor

    def click(self):
        self.events.append(("click", *self.cursor))
        if self.on_click:
            self.on_click()

    def press(self, key):
        self.events.append(("key", key))


class WindowsInput:
    def __init__(self):
        import sys
        if sys.platform != "win32":
            raise SafetyError("PyDirectInput доступен только в Windows")
        import pydirectinput
        self.api = pydirectinput
        self.api.PAUSE = 0
        self.api.FAILSAFE = True

    def assert_neutral(self):
        import ctypes
        from ctypes import wintypes
        user32 = ctypes.WinDLL("user32", use_last_error=True)
        user32.GetAsyncKeyState.argtypes = [ctypes.c_int]
        user32.GetAsyncKeyState.restype = wintypes.SHORT
        # Удерживаемые модификаторы и кнопки меняют смысл клика или создают перетаскивание.
        if any(user32.GetAsyncKeyState(key) & 0x8000 for key in (0x10, 0x11, 0x12, 0x5B, 0x5C, 1, 2, 4, 5, 6)):
            raise SafetyError("Отпустите модификаторы и кнопки мыши перед действием")

    def move_to(self, x, y):
        self.api.moveTo(x, y)

    def position(self):
        return tuple(self.api.position())

    def click(self):
        self.api.click()

    def press(self, key):
        self.api.press(key)


class Executor:
    def __init__(self, observer, backend, input_device, stop: StopToken, *, dry_run=True, max_age=2):
        self.observer = observer
        self.backend = backend
        self.input = input_device
        self.stop = stop
        self.dry_run = dry_run
        self.max_age = max_age

    def execute(self, plan: Plan, observation: Observation):
        self.stop.check()
        validate_observation(observation, self.max_age)
        action = plan.action
        # Клавиатура не нужна экономическому сценарию: ESC доступен только диагностике мок-ввода.
        if isinstance(action, KeyAction):
            raise SafetyError("Клавиатурные действия не разрешены в сценарии germany_1936")
        if not isinstance(action, ClickAction):
            raise SafetyError("Неизвестное действие")
        if action.target_id in observation.ambiguous or action.target_id not in observation.targets:
            raise SafetyError("Нет однозначной цели")
        original = observation.targets[action.target_id]
        guard_target(self.backend, observation, original, self.stop)
        fresh = self.observer.observe()
        validate_observation(fresh, self.max_age)
        if fresh.window != observation.window or fresh.ambiguous:
            raise SafetyError("Окно или цель изменились после планирования")
        target = fresh.targets.get(action.target_id)
        if target is None or target.rect != original.rect:
            raise SafetyError("Цель исчезла или переместилась; требуется новый план")
        point = guard_target(self.backend, fresh, target, self.stop)
        if self.dry_run:
            return target
        self.input.assert_neutral()
        self.input.move_to(*point)
        # Проверяем фокус ещё раз после движения, непосредственно перед нажатием.
        self.stop.check()
        guard_target(self.backend, fresh, target, self.stop)
        if self.input.position() != point:
            raise SafetyError("Указатель переместился до клика")
        self.input.assert_neutral()
        self.stop.check()
        self.input.click()
        return target
