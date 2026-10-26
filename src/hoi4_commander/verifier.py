"""Проверка нового кадра и фактического изменения интерфейса без повторных кликов."""
import time
import numpy as np
from .models import Step, Verification
from .safety import validate_observation


class Verifier:
    def __init__(self, observer, stop, timeout=5, interval=0.2, max_age=2):
        self.observer, self.stop = observer, stop
        self.timeout, self.interval, self.max_age = timeout, interval, max_age

    def verify(self, before, step: Step):
        # Уже существующий маркер не доказывает причинный результат клика.
        if all(t in before.targets for t in step.expected_present) and all(t not in before.targets for t in step.expected_absent):
            return Verification(ok=False, detail="Ожидаемый результат присутствовал до действия"), before
        deadline = time.monotonic() + self.timeout
        while True:
            self.stop.check()
            after = self.observer.observe()
            if after.window != before.window:
                return Verification(ok=False, detail="Окно изменилось во время проверки результата"), after
            validate_observation(after, self.max_age)
            if after.ambiguous:
                return Verification(ok=False, detail="Результат действия неоднозначен"), after
            changed = after.frame_id != before.frame_id and not np.array_equal(before.image, after.image)
            ok = changed and all(t in after.targets for t in step.expected_present) and all(t not in after.targets for t in step.expected_absent)
            if ok:
                return Verification(ok=True, detail="Изменение подтверждено новым скриншотом"), after
            if time.monotonic() >= deadline:
                return Verification(ok=False, detail="Ожидаемое изменение интерфейса не подтверждено"), after
            self.stop.wait(self.interval)
