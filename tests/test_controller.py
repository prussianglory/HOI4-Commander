"""Конечный сценарий, проверка переходов и запрет повторных кликов."""
from dataclasses import replace
import json
from pathlib import Path

import pytest

from hoi4_commander.config import Settings
from hoi4_commander.controller import build_controller
from hoi4_commander.journal import Journal
from hoi4_commander.model_provider import LocalProvider
from hoi4_commander.models import ClickAction, Plan
from hoi4_commander.planner import ScenarioPlanner
from hoi4_commander.safety import SafetyError
from hoi4_commander.scenario import germany_1936
from hoi4_commander.verifier import Verifier


def records(controller):
    return [json.loads(line) for line in controller.journal.path.read_text(encoding="utf-8").splitlines()]


def make_controller(tmp_path, *, execute=True, callback=None):
    return build_controller(Settings(journal_dir=tmp_path / "journal"), callback, mock_execute=execute)


def test_full_mock_scenario_verifies_all_ten_transitions(tmp_path):
    controller = make_controller(tmp_path)
    result = controller.run()
    assert result.status == "completed"
    assert result.completed_steps == 10
    assert controller.observer.state == 10
    events = controller.executor.input.events
    assert len([event for event in events if event[0] == "click"]) == 10
    journal = records(controller)
    verifications = [record for record in journal if record["event"] == "verification"]
    assert len(verifications) == 10 and all(record["ok"] for record in verifications)
    assert len({record["frame_id"] for record in verifications}) == 10
    assert {"focus_active", "research_active", "construction_queued"} <= controller.observer.observe().targets.keys()


def test_dry_run_only_shows_first_observable_plan_without_faking_progress(tmp_path):
    events = []
    controller = make_controller(tmp_path, execute=False, callback=events.append)
    result = controller.run()
    assert result.status == "dry_run" and result.completed_steps == 0
    assert controller.observer.state == 0
    assert controller.executor.input.events == []
    assert any(event["type"] == "plan" and event["target"].id == "focus_menu" for event in events)
    assert not any(record["event"] in {"action", "verification", "completed"} for record in records(controller))


def test_controller_stop_before_run_has_no_input(tmp_path):
    controller = make_controller(tmp_path)
    controller.stop()
    assert controller.run().status == "stopped"
    assert controller.executor.input.events == []


def test_stop_during_provider_decision_never_reaches_executor(tmp_path):
    controller = make_controller(tmp_path)

    class StoppingProvider:
        def choose(self, observation, step):
            plan = LocalProvider().choose(observation, step)
            controller.stop()
            return plan

    controller.planner = ScenarioPlanner(StoppingProvider())
    result = controller.run()
    assert result.status == "stopped"
    assert controller.executor.input.events == []


def test_unchanged_result_stops_after_one_click_without_retry(tmp_path):
    controller = make_controller(tmp_path)
    controller.executor.input.on_click = None
    controller.verifier.timeout = 0.001
    controller.verifier.interval = 0
    result = controller.run()
    assert result.status == "stopped" and result.completed_steps == 0
    assert len([event for event in controller.executor.input.events if event[0] == "click"]) == 1
    verification = [record for record in records(controller) if record["event"] == "verification"]
    assert len(verification) == 1 and verification[0]["ok"] is False


def test_preexisting_result_is_not_treated_as_action_success(tmp_path):
    controller = make_controller(tmp_path)
    controller.observer.state = 1
    assert controller.run().status == "stopped"
    assert controller.executor.input.events == []


def test_any_ambiguous_element_stops_controller_before_input(tmp_path):
    controller = make_controller(tmp_path)
    original = controller.observer.observe
    controller.observer.observe = lambda: replace(original(), ambiguous=frozenset({"unrelated_button"}))
    assert controller.run().status == "stopped"
    assert controller.executor.input.events == []


def test_verifier_rejects_preexisting_expected_marker(tmp_path):
    controller = make_controller(tmp_path)
    controller.observer.state = 1
    before = controller.observer.observe()
    result, after = controller.verifier.verify(before, germany_1936()[0])
    assert result.ok is False
    assert after is before


def test_controller_is_single_use(tmp_path):
    controller = make_controller(tmp_path, execute=False)
    controller.run()
    with pytest.raises(SafetyError, match="одноразовый"):
        controller.run()


def test_provider_failure_secret_never_reaches_journal_or_events(tmp_path, monkeypatch):
    secret = "секретный-ключ-проверки"
    monkeypatch.setenv("HOI4_MODEL_API_KEY", secret)
    emitted = []
    controller = make_controller(tmp_path, callback=emitted.append)

    class BrokenProvider:
        def choose(self, observation, step):
            raise RuntimeError("Ответ сервера содержит " + secret)

    controller.planner = ScenarioPlanner(BrokenProvider())
    assert controller.run().status == "stopped"
    text = controller.journal.path.read_text(encoding="utf-8")
    assert secret not in text and "Ответ сервера" not in text
    assert secret not in str([event for event in emitted if event["type"] == "status"])
    assert controller.executor.input.events == []


def test_secret_in_reason_is_redacted_from_journal_and_ui(tmp_path, monkeypatch):
    secret = "test-private-api-key"
    monkeypatch.setenv("HOI4_MODEL_API_KEY", secret)
    emitted = []
    controller = make_controller(tmp_path, execute=False, callback=emitted.append)

    class EchoProvider:
        def choose(self, observation, step):
            return Plan(action=ClickAction(target_id=step.target_id, reason="Объяснение " + secret))

    controller.planner = ScenarioPlanner(EchoProvider())
    assert controller.run().status == "dry_run"
    assert secret not in controller.journal.path.read_text(encoding="utf-8")
    assert secret not in str([event for event in emitted if event["type"] in {"plan", "status"}])


def test_live_executor_without_consent_cannot_start(tmp_path):
    controller = make_controller(tmp_path)

    class RealInputSentinel:
        def move_to(self, *args):
            raise AssertionError("Недопустимый реальный ввод")

        def click(self):
            raise AssertionError("Недопустимый реальный ввод")

    controller.executor.input = RealInputSentinel()
    controller.settings = controller.settings.model_copy(update={"consent_file": tmp_path / "absent-consent.json"})
    assert controller.run().status == "stopped"
    assert controller.observer.state == 0


def test_journal_redacts_secrets_before_json_escaping(tmp_path, monkeypatch):
    secret = 'часть"ключа\\с\\символами\nи переносом'
    monkeypatch.setenv("HOI4_MODEL_API_KEY", secret)
    journal = Journal(tmp_path, secret_env_names=("HOI4_MODEL_API_KEY",))
    journal.write("probe", nested={"reason": "Пример " + secret}, items=[secret])
    record = json.loads(journal.path.read_text(encoding="utf-8"))
    assert secret not in str(record)
    assert record["nested"]["reason"] == "Пример [СКРЫТО]"
    assert record["items"] == ["[СКРЫТО]"]


def test_expected_target_without_pixel_change_does_not_verify_action(tmp_path):
    controller = make_controller(tmp_path)
    before = controller.observer.observe()
    controller.observer.state = 1
    after = controller.observer.observe()
    controller.observer.observe = lambda: replace(after, image=before.image)
    controller.verifier.timeout = 0.001
    controller.verifier.interval = 0
    result, observed = controller.verifier.verify(before, germany_1936()[0])
    assert result.ok is False


def test_hotkey_preflight_failure_prevents_first_real_input(tmp_path):
    from hoi4_commander.safety import ConsentStore
    controller = make_controller(tmp_path)
    consent = tmp_path / "consent.json"
    ConsentStore(consent).grant()
    controller.settings = controller.settings.model_copy(update={"consent_file": consent})

    class RealInputSentinel:
        def move_to(self, *args):
            raise AssertionError("До проверки F10 ввод запрещён")

        def click(self):
            raise AssertionError("До проверки F10 ввод запрещён")

    class FailingHotkey:
        def __init__(self):
            self.calls = []

        def start(self):
            self.calls.append("start")

        def verify_before_input(self, status):
            self.calls.append("verify")
            raise SafetyError("F10 не подтверждён")

        def close(self):
            self.calls.append("close")

    hotkey = FailingHotkey()
    controller.hotkey = hotkey
    controller.executor.input = RealInputSentinel()
    assert controller.run().status == "stopped"
    assert hotkey.calls == ["start", "verify", "close"]
    assert controller.observer.state == 0


def test_consent_without_hotkey_is_insufficient_for_real_input(tmp_path):
    from hoi4_commander.safety import ConsentStore
    controller = make_controller(tmp_path)
    consent = tmp_path / "consent.json"
    ConsentStore(consent).grant()
    controller.settings = controller.settings.model_copy(update={"consent_file": consent})
    controller.executor.input = object()
    assert controller.run().status == "stopped"
    assert controller.observer.state == 0
