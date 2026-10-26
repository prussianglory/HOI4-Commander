"""Проверки запретов ввода на подменяемом окне и мок-устройстве."""
from dataclasses import replace
import time

import numpy as np
import pytest
from pydantic import ValidationError

from hoi4_commander.executor import Executor, MockInput
from hoi4_commander.models import ClickAction, KeyAction, Observation, Plan, Rect, Target, WindowInfo
from hoi4_commander.safety import ConsentStore, REQUIRED_ANCHORS, SafetyError, StopToken, guard_target


def observation(frame="frame", **changes):
    window = WindowInfo(handle=1, title="Hearts of Iron IV", process_name="hoi4.exe",
                        client=Rect(x=100, y=50, width=400, height=300))
    targets = {name: Target(id=name, rect=Rect(x=20, y=30, width=20, height=20),
                            confidence=1, frame_id=frame)
               for name in REQUIRED_ANCHORS | {"focus_menu"}}
    return replace(Observation(frame, time.monotonic(), window, np.zeros((300, 400, 3), dtype=np.uint8), targets), **changes)


class Backend:
    def __init__(self, window):
        self.window = window

    def current_window(self):
        return self.window


class FreshObserver:
    def __init__(self, obs):
        self.obs = obs

    def observe(self):
        return self.obs


def rig(*, dry_run=False):
    before = observation()
    after = observation("fresh")
    backend, device, stop = Backend(before.window), MockInput(), StopToken()
    observer = FreshObserver(after)
    executor = Executor(observer, backend, device, stop, dry_run=dry_run)
    plan = Plan(action=ClickAction(target_id="focus_menu", reason="Открыть фокусы"))
    return before, after, backend, device, stop, observer, executor, plan


def test_mock_click_uses_client_relative_verified_target():
    before, after, backend, device, stop, observer, executor, plan = rig()
    executor.execute(plan, before)
    assert device.events == [("move", 130, 90), ("click", 130, 90)]


def test_dry_run_has_no_mouse_or_keyboard_events():
    before, after, backend, device, stop, observer, executor, plan = rig(dry_run=True)
    assert executor.execute(plan, before).id == "focus_menu"
    assert not device.events


@pytest.mark.parametrize("mutation", ["old", "future", "missing_anchor", "ambiguous", "foreign_frame", "out_of_bounds"])
def test_bad_observation_never_emits_input(mutation):
    before, after, backend, device, stop, observer, executor, plan = rig()
    if mutation == "old":
        before = replace(before, captured_at=time.monotonic() - 10)
    elif mutation == "future":
        before = replace(before, captured_at=time.monotonic() + 10)
    elif mutation == "missing_anchor":
        before.targets.pop("game_paused")
    elif mutation == "ambiguous":
        before = replace(before, ambiguous=frozenset({"focus_menu"}))
    elif mutation == "foreign_frame":
        before.targets["focus_menu"] = before.targets["focus_menu"].model_copy(update={"frame_id": "чужой"})
    else:
        before.targets["focus_menu"] = before.targets["focus_menu"].model_copy(update={"rect": Rect(x=390, y=10, width=20, height=20)})
    with pytest.raises(SafetyError):
        executor.execute(plan, before)
    assert device.events == []


def test_changed_target_on_fresh_image_never_moves_mouse():
    before, after, backend, device, stop, observer, executor, plan = rig()
    after.targets["focus_menu"] = after.targets["focus_menu"].model_copy(update={"rect": Rect(x=21, y=30, width=20, height=20)})
    with pytest.raises(SafetyError, match="переместилась"):
        executor.execute(plan, before)
    assert device.events == []


def test_window_focus_change_after_mouse_move_prevents_click():
    before, after, backend, device, stop, observer, executor, plan = rig()
    original_move = device.move_to

    def move_and_lose_focus(x, y):
        original_move(x, y)
        backend.window = backend.window.model_copy(update={"handle": 2, "process_name": "explorer.exe"})

    device.move_to = move_and_lose_focus
    with pytest.raises(SafetyError, match="окно"):
        executor.execute(plan, before)
    assert [event[0] for event in device.events] == ["move"]


def test_stop_after_mouse_move_prevents_click():
    before, after, backend, device, stop, observer, executor, plan = rig()
    original_move = device.move_to

    def move_and_stop(x, y):
        original_move(x, y)
        stop.stop()

    device.move_to = move_and_stop
    with pytest.raises(SafetyError):
        executor.execute(plan, before)
    assert [event[0] for event in device.events] == ["move"]


def test_stop_before_action_never_emits_input():
    before, after, backend, device, stop, observer, executor, plan = rig()
    stop.stop()
    with pytest.raises(SafetyError):
        executor.execute(plan, before)
    assert not device.events


def test_external_cursor_movement_prevents_click():
    before, after, backend, device, stop, observer, executor, plan = rig()
    device.position = lambda: (0, 0)
    with pytest.raises(SafetyError, match="Указатель"):
        executor.execute(plan, before)
    assert [event[0] for event in device.events] == ["move"]


def test_keyboard_action_disallowed_in_economic_scenario():
    before, after, backend, device, stop, observer, executor, plan = rig()
    with pytest.raises(SafetyError):
        executor.execute(Plan(action=KeyAction(key="esc", reason="Закрыть")), before)
    assert device.events == []


@pytest.mark.parametrize("action", [
    {"kind": "exec", "code": "print(1)"},
    {"kind": "click", "target_id": "focus_menu", "reason": "Нажать", "x": 3, "y": 4},
    {"kind": "key", "key": "win", "reason": "Открыть ОС"},
])
def test_plan_schema_rejects_code_coordinates_and_os_keys(action):
    with pytest.raises(ValidationError):
        Plan.model_validate({"action": action})


def test_consent_missing_corrupt_or_unrelated_does_not_grant_live(tmp_path):
    store = ConsentStore(tmp_path / "consent.json")
    for value in (None, "не JSON", '{"live_game_input": true}', '{"version": 1, "live_game_input": false}'):
        if value is not None:
            store.path.write_text(value, encoding="utf-8")
        with pytest.raises(SafetyError):
            store.require()
    store.grant()
    store.require()
