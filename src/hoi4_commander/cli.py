"""Командная строка: сначала безопасная демонстрация, затем калибровка."""
import argparse
import json
from pathlib import Path
import sys

from .config import Settings
from .controller import build_controller
from .executor import MockInput
from .fixtures import create_demo
from .safety import ConsentStore, StopToken


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
    demo.add_argument("--journal-dir", type=Path, default=Path("runtime/journal"))
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
            from .safety import SafetyError
            try:
                token.check()
            except SafetyError:
                pass
            else:
                raise AssertionError("Остановка не блокирует следующее действие")
            assert token.stopped and len(device.events) == 3
            print("Мок-ввод: движение, клик, ESC и сигнал остановки проверены. Ввод в ОС не выполнялся.")
            return 0
        if args.command == "demo":
            settings = Settings(journal_dir=args.journal_dir)
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
