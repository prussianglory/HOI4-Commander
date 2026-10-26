"""Удерживаемые модификаторы не должны менять смысл экономического клика."""
import pytest
from test_safety_executor import rig
from hoi4_commander.safety import SafetyError


def test_modifier_before_movement_prevents_all_input():
    before, after, backend, device, stop, observer, executor, plan = rig()

    def held():
        raise SafetyError("Модификатор удерживается")

    device.assert_neutral = held
    with pytest.raises(SafetyError):
        executor.execute(plan, before)
    assert device.events == []


def test_modifier_after_movement_prevents_click():
    before, after, backend, device, stop, observer, executor, plan = rig()

    def check():
        if device.events:
            raise SafetyError("Модификатор удерживается")

    device.assert_neutral = check
    with pytest.raises(SafetyError):
        executor.execute(plan, before)
    assert len(device.events) == 1 and device.events[0][0] == "move"
