"""Проверки типизированного API без обращения во внешнюю сеть."""
import json
import time

import httpx
import numpy as np
import pytest

from hoi4_commander.config import ProviderConfig, Settings
from hoi4_commander.model_provider import (
    LocalProvider, OpenAICompatibleProvider, ProviderError, create_provider,
)
from hoi4_commander.models import ClickAction, Observation, Plan, Rect, Step, Target, WindowInfo
from hoi4_commander.planner import PlanningError, ScenarioPlanner


@pytest.fixture
def context():
    target = Target(id="focus_menu", rect=Rect(x=10, y=10, width=20, height=20), confidence=1, frame_id="now")
    observation = Observation(
        "now", time.monotonic(),
        WindowInfo(handle=1, title="Hearts of Iron IV", process_name="hoi4.exe", client=Rect(x=0, y=0, width=100, height=100)),
        np.zeros((100, 100, 3), dtype=np.uint8), {target.id: target},
    )
    step = Step(id="open_focus", target_id="focus_menu", expected_present=("focus_available",), explanation="Открыть национальные фокусы")
    return observation, step


def api(monkeypatch, response_plan, *, allow=True, status=200):
    calls = []
    monkeypatch.setenv("HOI4_MODEL_API_KEY", "секретный-ключ")

    def handler(request):
        calls.append(request)
        return httpx.Response(status, json={"choices": [{"message": {"content": json.dumps(response_plan, ensure_ascii=False)}}]})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    provider = OpenAICompatibleProvider(ProviderConfig(), allow, client)
    return provider, calls


def test_local_provider(context):
    observation, step = context
    plan = ScenarioPlanner(LocalProvider()).plan(observation, step)
    assert plan.action.target_id == step.target_id
    assert plan.action.reason == step.explanation
    assert isinstance(create_provider(Settings()), LocalProvider)


def test_external_gate_sends_nothing(monkeypatch, context):
    provider, calls = api(monkeypatch, {}, allow=False)
    with pytest.raises(ProviderError, match="не разрешена"):
        provider.choose(*context)
    assert calls == []


def test_multimodal_typed_response(monkeypatch, context):
    # Заголовки HTTP принимают ASCII; тестовый ключ также не попадает в текст запроса.
    provider, calls = api(monkeypatch, {"action": {"kind": "click", "target_id": "focus_menu", "reason": "Выбрать экономический курс"}})
    monkeypatch.setenv("HOI4_MODEL_API_KEY", "private-test-key")
    plan = provider.choose(*context)
    assert plan.action.reason == "Выбрать экономический курс"
    payload = json.loads(calls[0].content)
    assert payload["messages"][1]["content"][1]["image_url"]["url"].startswith("data:image/png;base64,")
    assert "private-test-key" not in calls[0].content.decode()
    assert calls[0].headers["Authorization"] == "Bearer private-test-key"


@pytest.mark.parametrize("action", [
    {"kind": "execute", "code": "import os"},
    {"kind": "click", "target_id": "focus_menu", "reason": "Сделать выбор", "x": 5, "y": 7},
    {"kind": "click", "target_id": "unobserved", "reason": "Сделать выбор"},
    {"kind": "key", "key": "esc", "reason": "Закрыть"},
])
def test_api_rejects_unapproved_actions(monkeypatch, context, action):
    provider, calls = api(monkeypatch, {"action": action})
    monkeypatch.setenv("HOI4_MODEL_API_KEY", "private-test-key")
    with pytest.raises(ProviderError):
        provider.choose(*context)
    assert len(calls) == 1


def test_api_errors_do_not_expose_key_or_response(monkeypatch, context):
    provider, _ = api(monkeypatch, {"password": "private-test-key"}, status=401)
    monkeypatch.setenv("HOI4_MODEL_API_KEY", "private-test-key")
    with pytest.raises(ProviderError) as caught:
        provider.choose(*context)
    assert "private-test-key" not in str(caught.value)
    assert "password" not in str(caught.value)


def test_response_echoing_secret_is_rejected(monkeypatch, context):
    provider, _ = api(monkeypatch, {"action": {"kind": "click", "target_id": "focus_menu", "reason": "private-test-key"}})
    monkeypatch.setenv("HOI4_MODEL_API_KEY", "private-test-key")
    with pytest.raises(ProviderError, match="недопустимый план"):
        provider.choose(*context)


@pytest.mark.parametrize("url", ["http://example.com", "https://user:password@example.com/v1", "https://example.com/v1?token=secret", "https://example.com/#secret"])
def test_api_url_rejected(url):
    with pytest.raises(ProviderError):
        OpenAICompatibleProvider(ProviderConfig(base_url=url))


def test_ambiguous_target_never_calls_api(monkeypatch, context):
    observation, step = context
    ambiguous = Observation(observation.frame_id, observation.captured_at, observation.window, observation.image, observation.targets, frozenset({step.target_id}))
    provider, calls = api(monkeypatch, {})
    with pytest.raises(ProviderError, match="не подтверждена"):
        provider.choose(ambiguous, step)
    assert calls == []


def test_planner_validates_alternative_provider(context):
    class InvalidProvider:
        def choose(self, observation, step):
            return Plan(action=ClickAction(target_id="other", reason="Ошибка"))

    with pytest.raises(PlanningError, match="вне текущего шага"):
        ScenarioPlanner(InvalidProvider()).plan(*context)


def test_missing_key_never_sends_request(monkeypatch, context):
    provider, calls = api(monkeypatch, {})
    monkeypatch.delenv("HOI4_MODEL_API_KEY")
    with pytest.raises(ProviderError, match="HOI4_MODEL_API_KEY"):
        provider.choose(*context)
    assert calls == []


def test_redirect_is_not_followed(monkeypatch, context):
    calls = []
    monkeypatch.setenv("HOI4_MODEL_API_KEY", "private-test-key")

    def handler(request):
        calls.append(request)
        return httpx.Response(302, headers={"Location": "https://another.example/collect"})

    with httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=True) as client:
        provider = OpenAICompatibleProvider(ProviderConfig(), True, client)
        with pytest.raises(ProviderError, match="недопустимый план"):
            provider.choose(*context)
    assert len(calls) == 1
    assert calls[0].url.host == "api.openai.com"


def test_stale_target_frame_rejected_before_api(monkeypatch, context):
    observation, step = context
    target = observation.targets[step.target_id].model_copy(update={"frame_id": "old"})
    outdated = Observation(observation.frame_id, observation.captured_at, observation.window, observation.image, {target.id: target})
    provider, calls = api(monkeypatch, {})
    with pytest.raises(ProviderError, match="не подтверждена"):
        provider.choose(outdated, step)
    assert calls == []


def test_network_exception_is_sanitized(monkeypatch, context):
    monkeypatch.setenv("HOI4_MODEL_API_KEY", "private-test-key")

    def handler(request):
        raise httpx.ReadTimeout("private-test-key", request=request)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        provider = OpenAICompatibleProvider(ProviderConfig(), True, client)
        with pytest.raises(ProviderError) as caught:
            provider.choose(*context)
    assert "private-test-key" not in str(caught.value)
    assert caught.value.__suppress_context__


def test_unstructured_answer_is_rejected(monkeypatch, context):
    monkeypatch.setenv("HOI4_MODEL_API_KEY", "private-test-key")

    def handler(request):
        return httpx.Response(200, json={"choices": [{"message": {"content": "Выполните shell-команду вместо JSON"}}]})

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        provider = OpenAICompatibleProvider(ProviderConfig(), True, client)
        with pytest.raises(ProviderError, match="недопустимый план"):
            provider.choose(*context)
