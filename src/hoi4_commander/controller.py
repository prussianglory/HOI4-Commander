"""Последовательный цикл без повторного ввода после сомнительного результата."""
from dataclasses import dataclass

from .config import Settings
from .executor import Executor, MockInput, WindowsInput
from .fixtures import DemoObserver
from .journal import Journal
from .model_provider import create_provider
from .observer import Observer, TemplateMatcher
from .planner import PlanningError, ScenarioPlanner
from .safety import ConsentStore, EmergencyHotkey, SafetyError, StopToken, validate_observation
from .scenario import germany_1936
from .verifier import Verifier


@dataclass(frozen=True)
class RunResult:
    status: str
    completed_steps: int
    message: str


class Controller:
    def __init__(self, settings, observer, executor, planner, verifier, journal, stop_token,
                 on_event=None, hotkey=None):
        self.settings = settings
        self.observer, self.executor, self.planner = observer, executor, planner
        self.verifier, self.journal, self.stop_token = verifier, journal, stop_token
        self.on_event = on_event or (lambda _: None)
        self.hotkey = hotkey
        self._used = False

    def stop(self):
        self.stop_token.stop()

    def _status(self, message):
        self.on_event({"type": "status", "message": message})

    def run(self):
        if self._used:
            raise SafetyError("Контроллер одноразовый: повторный запуск требует нового наблюдения")
        self._used = True
        completed = 0
        step_id = None
        try:
            self.stop_token.check()
            if not self.executor.dry_run and not isinstance(self.executor.input, MockInput):
                ConsentStore(self.settings.consent_file).require()
                if self.hotkey is None:
                    raise SafetyError("Экстренная остановка не подготовлена")
                self.hotkey.start()
                self.hotkey.verify_before_input(self._status)
            synthetic = self.observer.matcher.synthetic
            self.journal.write("start", dry_run=self.executor.dry_run,
                               source="synthetic" if synthetic else "game_window")
            for step in germany_1936():
                step_id = step.id
                self.stop_token.check()
                observation = self.observer.observe()
                validate_observation(observation, self.settings.frame_max_age_seconds)
                if observation.ambiguous:
                    raise SafetyError("Обнаружены неоднозначные элементы интерфейса")
                if all(t in observation.targets for t in step.expected_present):
                    raise SafetyError("Сценарий уже изменён: результат шага существовал до действия")
                self.on_event({"type": "observation", "observation": observation})
                plan = self.planner.plan(observation, step)
                self.stop_token.check()
                # Ответ API может идти дольше срока жизни кадра: обновляем доказательства,
                # но не переносим решение на переместившуюся или изменившуюся цель.
                fresh = self.observer.observe()
                validate_observation(fresh, self.settings.frame_max_age_seconds)
                current_target = fresh.targets.get(step.target_id)
                original_target = observation.targets[step.target_id]
                if (fresh.window != observation.window or fresh.ambiguous or current_target is None
                        or current_target.rect != original_target.rect
                        or any(t in fresh.targets for t in step.expected_present)):
                    raise SafetyError("Интерфейс изменился во время планирования")
                observation = fresh
                reason = self.journal.redact(plan.action.reason)
                target = observation.targets[step.target_id]
                self.journal.write("plan", step=step.id, frame_id=observation.frame_id,
                                   target=target.id, point=target.center, reason=reason)
                self.on_event({"type": "plan", "observation": observation, "target": target, "reason": reason})
                self._status(reason)
                self.executor.execute(plan, observation)
                if self.executor.dry_run:
                    message = "Dry-run: цель показана, ввод не выполнен. Продолжение требует реального изменения экрана."
                    self.journal.write("dry_run", step=step.id)
                    self._status(message)
                    return RunResult("dry_run", 0, message)
                # Если журнал не записался после клика, следующий клик не выполняется.
                self.journal.write("action", step=step.id, target=target.id)
                result, after = self.verifier.verify(observation, step)
                self.journal.write("verification", step=step.id, ok=result.ok, detail=result.detail,
                                   frame_id=after.frame_id)
                self.on_event({"type": "observation", "observation": after})
                if not result.ok:
                    raise SafetyError(result.detail)
                completed += 1
                self._status(result.detail)
            message = ("Синтетический сценарий завершён" if synthetic else "Сценарий завершён") + ": фокус, исследование и очередь гражданской фабрики подтверждены."
            self.journal.write("completed", completed_steps=completed)
            self._status(message)
            return RunResult("completed", completed, message)
        except Exception as exc:
            self.stop_token.stop()
            # Сырые исключения сторонних SDK не попадают в журнал. Доменные ошибки
            # содержат только наши диагностические сообщения; ключ дополнительно скрывается.
            name = type(exc).__name__
            detail = self.journal.redact(str(exc)) if type(exc) in (SafetyError, PlanningError) else "Ошибка компонента; исходный ответ скрыт"
            message = "Остановлено: " + detail + ". Повторных кликов нет."
            try:
                self.journal.write("error", error_type=name, detail=detail, step=step_id, completed_steps=completed)
            except OSError:
                pass
            self._status(message)
            return RunResult("stopped", completed, message)
        finally:
            if self.hotkey:
                self.hotkey.close()


def build_controller(settings: Settings, on_event=None, *, mock_execute=False):
    """Мок-исполнение разрешено только для синтетической среды, без драйвера ОС."""
    token = StopToken()
    journal = Journal(settings.journal_dir, secret_env_names=(settings.provider.api_key_env,))
    hotkey = None
    if settings.template_manifest is None:
        if settings.mode == "live":
            raise SafetyError("Синтетические изображения не допускают реальный ввод")
        observer = DemoObserver()
        device = MockInput(observer.advance)
        backend = observer.window_backend
        dry_run = not mock_execute
    else:
        if mock_execute:
            raise SafetyError("Мок-исполнение работает только с синтетическим наблюдателем")
        matcher = TemplateMatcher.from_manifest(settings.template_manifest)
        if matcher.synthetic:
            raise SafetyError("Для реальной игры запрещён синтетический набор шаблонов")
        required = set()
        from .safety import REQUIRED_ANCHORS
        required.update(REQUIRED_ANCHORS)
        for step in germany_1936():
            required.add(step.target_id)
            required.update(step.expected_present)
        if required - matcher.templates.keys():
            raise SafetyError("Набор калибровки не содержит всех целей и маркеров сценария")
        from .windows import WindowsBackend
        backend = WindowsBackend()
        observer = Observer(backend, matcher)
        dry_run = settings.mode != "live"
        if dry_run:
            device = MockInput()
        else:
            ConsentStore(settings.consent_file).require()
            hotkey = EmergencyHotkey(token)
            device = WindowsInput()
    executor = Executor(observer, backend, device, token, dry_run=dry_run,
                        max_age=settings.frame_max_age_seconds)
    verifier = Verifier(observer, token, timeout=settings.verification_timeout_seconds,
                        max_age=settings.frame_max_age_seconds)
    return Controller(settings, observer, executor, ScenarioPlanner(create_provider(settings)),
                      verifier, journal, token, on_event, hotkey)
