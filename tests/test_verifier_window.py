"""Другой игровой HWND или геометрия не подтверждают результат действия."""
from dataclasses import replace
import time

import numpy as np
import pytest

from hoi4_commander.models import Observation, Rect, Step, Target, WindowInfo
from hoi4_commander.safety import REQUIRED_ANCHORS, StopToken
from hoi4_commander.verifier import Verifier


@pytest.mark.parametrize("change", [
    {"handle": 2},
    {"process_name": "другая-игра.exe"},
    {"client": Rect(x=101, y=50, width=400, height=300)},
])
def test_changed_window_rejects_even_matching_new_result(change):
    window = WindowInfo(handle=1, title="Hearts of Iron IV", process_name="hoi4.exe",
                        client=Rect(x=100, y=50, width=400, height=300))
    before = Observation("before", time.monotonic(), window,
                         np.zeros((300, 400, 3), dtype=np.uint8), {})
    targets = {name: Target(id=name, frame_id="after", confidence=1,
                            rect=Rect(x=10, y=10, width=10, height=10))
               for name in REQUIRED_ANCHORS | {"focus_available"}}
    after = Observation("after", time.monotonic(), window.model_copy(update=change),
                        np.ones((300, 400, 3), dtype=np.uint8), targets)

    class NewWindowObserver:
        def observe(self):
            return after

    step = Step(id="open_focus", target_id="focus_menu", expected_present=("focus_available",),
                explanation="Открываю национальные фокусы")
    result, actual = Verifier(NewWindowObserver(), StopToken(), timeout=0.01).verify(before, step)
    assert not result.ok
    assert "Окно" in result.detail
    assert actual is after
