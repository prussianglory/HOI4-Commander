"""Командная строка: сначала безопасная демонстрация, затем калибровка."""
import argparse
import json
from pathlib import Path
import sys

from .config import Settings
from .controller import build_controller
from .executor import MockInput
from .fixtures import DemoObserver, create_demo
from .journal import Journal
from .model_provider import ProviderError, create_provider
from .models import Rect, WindowInfo
from .planner import PlanningError, ScenarioPlanner
from .scenario import germany_1936
from .safety import ConsentStore, SafetyError, StopToken


def console_event(event):
    if event["type"] == "status":
        print(event["message"])
    elif event["type"] == "plan":
        print(f"Цель: {event['target'].id}; точка в кадре: {event['target'].center}")


def main(argv=None):
    parser = argparse.ArgumentParser(description="HOI4 AI Commander — безопасный прототип")
    commands = parser.add_subparsers(dest="command", required=True)
    demo = commands.add_parser("demo", help="Синтетические кадры, без ввода в ОС")
    demo.add_argument("--mock-execute", action="store_true", help="Проверить все переходы мок-вводом")
    demo.add_argument("--journal-dir", type=Path)
    demo.add_argument("--config", type=Path, help="Выбранный провайдер; среда и ввод всегда синтетические")
    check = commands.add_parser("provider-check", help="Проверить сервер Ollama и vision-модель без отправки изображений")
    check.add_argument("--config", type=Path, required=True)
    model_test = commands.add_parser("model-test", help="Получить один план модели по кадру без реального ввода")
    model_test.add_argument("--config", type=Path, required=True)
    model_test.add_argument("--image", type=Path)
    model_test.add_argument("--manifest", type=Path)
    model_test.add_argument("--step", choices=[step.id for step in germany_1936()], default="open_focus")
    fixtures = commands.add_parser("fixtures", help="Создать синтетические PNG для распознавания")
    fixtures.add_argument("--output", type=Path, default=Path("runtime/demo"))
    for name in ("run", "gui", "consent"):
        sub = commands.add_parser(name)
        sub.add_argument("--config", type=Path, required=name in ("run", "consent"))
    inspect = commands.add_parser("inspect", help="Распознать сохранённый кадр локально, без ввода и API")
    inspect.add_argument("--manifest", type=Path, required=True)
    inspect.add_argument("--image", type=Path, required=True)
    commands.add_parser("input-test", help="Тест мыши, клавиатуры и остановки только на MockInput")
    args = parser.parse_args(argv)
    try:
        if args.command == "fixtures":
            print(create_demo(args.output))
            return 0
        if args.command == "inspect":
            from .observer import TemplateMatcher, read_bgr
            matcher = TemplateMatcher.from_manifest(args.manifest)
            targets, ambiguous = matcher.detect(read_bgr(args.image), "saved-image")
            print(json.dumps({"targets": {key: value.model_dump() for key, value in targets.items()},
                              "ambiguous": sorted(ambiguous)}, ensure_ascii=False, indent=2))
            return 1 if ambiguous else 0
        if args.command == "input-test":
            token, device = StopToken(), MockInput()
            token.check()
            device.move_to(100, 100)
            device.click()
            device.press("esc")
            token.stop()
            try:
                token.check()
            except SafetyError:
                pass
            else:
                raise AssertionError("Остановка не блокирует следующее действие")
            assert token.stopped and len(device.events) == 3
            print("Мок-ввод: движение, клик, ESC и сигнал остановки проверены. Ввод в ОС не выполнялся.")
            return 0
        if args.command in ("provider-check", "model-test"):
            settings = Settings.load(args.config)
            provider = create_provider(settings)
            if args.command == "provider-check":
                if settings.provider.kind != "ollama":
                    print("Команда provider-check предназначена для провайдера ollama.")
                    return 2
                info = provider.probe()
                print(json.dumps({"version": info.version, "model": info.model,
                                  "capabilities": info.capabilities}, ensure_ascii=False, indent=2))
                print("Сервер и vision-возможности проверены. Скриншоты не отправлялись; генерация не проверялась.")
                return 0
            if bool(args.image) != bool(args.manifest):
                print("Для сохранённого кадра нужны одновременно --image и --manifest.")
                return 2
            if args.image:
                from .observer import SavedImageObserver, TemplateMatcher, read_bgr
                image = read_bgr(args.image)
                window = WindowInfo(handle=0, title="Сохранённый кадр — без ввода", process_name="hoi4.exe",
                                    client=Rect(x=0, y=0, width=image.shape[1], height=image.shape[0]))
                observer = SavedImageObserver(args.image, window, TemplateMatcher.from_manifest(args.manifest))
            else:
                if args.step != "open_focus":
                    print("Для другого шага укажите его сохранённый кадр и манифест.")
                    return 2
                observer = DemoObserver()
            observation = observer.observe()
            from .safety import validate_observation
            validate_observation(observation, settings.frame_max_age_seconds)
            if observation.ambiguous:
                raise SafetyError("Неоднозначный кадр не отправляется модели")
            step = next(step for step in germany_1936() if step.id == args.step)
            journal = Journal(settings.journal_dir, secret_env_names=(settings.provider.api_key_env,))
            try:
                plan = ScenarioPlanner(provider).plan(observation, step)
            except Exception as exc:
                journal.write("model_test_error", error_type=type(exc).__name__, step=step.id)
                raise
            reason = journal.redact(plan.action.reason)
            journal.write("model_test", step=step.id, target=plan.action.target_id,
                          reason=reason, source="saved_image" if args.image else "synthetic")
            print(json.dumps({"action": {"kind": plan.action.kind, "target_id": plan.action.target_id,
                                         "reason": reason}}, ensure_ascii=False, indent=2))
            print("План проверен по схеме и обнаруженной цели. Ввод не выполнялся; качество распознавания игры не подтверждено.")
            print(f"Журнал: {journal.path}")
            return 0
        if args.command == "demo":
            original = Settings.load(args.config) if args.config else Settings()
            updates = {"mode": "dry_run", "template_manifest": None}
            if args.journal_dir is not None:
                updates["journal_dir"] = args.journal_dir
            settings = Settings.model_validate({**original.model_dump(), **updates})
            controller = build_controller(settings, console_event, mock_execute=args.mock_execute)
        else:
            settings = Settings.load(args.config) if args.config else Settings()
            if args.command == "consent":
                if not sys.stdin.isatty():
                    print("Согласие требует интерактивного ввода пользователя.")
                    return 2
                print("Разрешение на реальные клики в HOI4. Сначала выполните тесты и dry-run. F10 — остановка.")
                if input("Введите РАЗРЕШАЮ для подтверждения: ") != "РАЗРЕШАЮ":
                    print("Разрешение не предоставлено.")
                    return 2
                ConsentStore(settings.consent_file).grant()
                print("Согласие сохранено. Отправка скриншотов в API остаётся отдельной настройкой.")
                return 0
            if args.command == "gui":
                from .gui import run_gui
                return run_gui(settings, build_controller)
            if settings.template_manifest is None:
                print("Для run укажите собственный манифест; для тестовых кадров используйте demo.")
                return 2
            controller = build_controller(settings, console_event)
        result = controller.run()
        print(f"Результат: {result.status}; подтверждено шагов: {result.completed_steps}.")
        print(f"Журнал: {controller.journal.path}")
        return 0 if result.status in ("dry_run", "completed") else 1
    except (ProviderError, PlanningError, SafetyError) as exc:
        print(f"Проверка не пройдена: {exc}")
        return 2
    except KeyboardInterrupt:
        if "controller" in locals():
            controller.stop()
        print("Остановлено пользователем.")
        return 130
    except Exception as exc:
        # Не выводим исходные исключения сторонних библиотек и валидации конфигурации.
        print(f"Запуск запрещён: {type(exc).__name__}. Проверьте конфигурацию и README.")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
