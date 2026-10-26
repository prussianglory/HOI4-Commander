"""Проверки нативного Ollama API без сервера, загрузки моделей и игрового ввода."""
import base64
from dataclasses import replace
import json
import time

import cv2
import httpx
import numpy as np
import pytest

from hoi4_commander.config import ProviderConfig
from hoi4_commander.model_provider import ProviderError
from hoi4_commander.models import Observation, Rect, Step, Target, WindowInfo
from hoi4_commander.ollama_provider import OllamaProvider


@pytest.fixture
def context():
    target = Target(id="focus_menu", rect=Rect(x=10, y=10, width=20, height=20), confidence=1, frame_id="now")
    observation = Observation(
        "now", time.monotonic(),
        WindowInfo(handle=1, title="Hearts of Iron IV", process_name="hoi4.exe", client=Rect(x=0, y=0, width=100, height=100)),
        np.zeros((100, 100, 3), dtype=np.uint8), {target.id: target},
    )
    return observation, Step(id="open_focus", target_id="focus_menu", expected_present=("focus_available",), explanation="Открыть фокусы")


def config(**changes):
    return ProviderConfig(kind="ollama", base_url="http://127.0.0.1:11434", model="qwen2.5vl:7b", **changes)


def response(action=None, **changes):
    plan = {"action": action or {"kind": "click", "target_id": "focus_menu", "reason": "Открыть экономические фокусы"}}
    return {"done": True, "done_reason": "stop", "message": {"role": "assistant", "content": json.dumps(plan, ensure_ascii=False)}, **changes}


def server(*, endpoint="http://127.0.0.1:11434", allow_local=True, allow_external=False, chat=None, show=None):
    calls = []

    def handler(request):
        calls.append(request)
        if request.url.path == "/api/version":
            return httpx.Response(200, json={"version": "0.13.0"})
        if request.url.path == "/api/show":
            return httpx.Response(200, json=show if show is not None else {"capabilities": ["completion", "vision"]})
        if request.url.path == "/api/chat":
            return httpx.Response(200, json=chat if chat is not None else response())
        raise AssertionError("Недокументированный запрос")

    client = httpx.Client(transport=httpx.MockTransport(handler))
    provider = OllamaProvider(ProviderConfig(kind="ollama", base_url=endpoint, model="qwen2.5vl:7b"), allow_local, allow_external, client)
    return provider, calls


def test_native_payload_schema_and_png(context, monkeypatch):
    monkeypatch.setenv("HOI4_MODEL_API_KEY", "private-test-key")
    provider, calls = server()
    plan = provider.choose(*context)
    assert plan.action.target_id == "focus_menu"
    assert [request.url.path for request in calls] == ["/api/version", "/api/show", "/api/chat"]
    assert json.loads(calls[1].content) == {"model": "qwen2.5vl:7b"}
    payload = json.loads(calls[2].content)
    assert payload["model"] == "qwen2.5vl:7b" and payload["stream"] is False
    assert payload["options"] == {"temperature": 0, "num_predict": 1024}
    assert payload["format"]["type"] == "object"
    assert "action" in payload["format"]["properties"]
    image = cv2.imdecode(np.frombuffer(base64.b64decode(payload["messages"][1]["images"][0]), dtype=np.uint8), cv2.IMREAD_COLOR)
    assert np.array_equal(image, context[0].image)
    assert all("Authorization" not in request.headers for request in calls)
    assert all("private-test-key" not in request.content.decode() for request in calls)


def test_probe_never_uploads_images_and_caches_capabilities(context):
    provider, calls = server()
    info = provider.probe()
    assert info.version == "0.13.0" and info.model == "qwen2.5vl:7b"
    assert info.capabilities == ("completion", "vision")
    assert all(b"images" not in request.content for request in calls)
    provider.choose(*context)
    assert [request.url.path for request in calls].count("/api/show") == 1


@pytest.mark.parametrize("local,external,endpoint", [
    (False, False, "http://127.0.0.1:11434"),
    (False, True, "http://localhost:11434"),
    (True, False, "https://ollama.example"),
])
def test_each_destination_needs_own_explicit_screenshot_consent(context, local, external, endpoint):
    provider, calls = server(endpoint=endpoint, allow_local=local, allow_external=external)
    with pytest.raises(ProviderError, match="не разрешена"):
        provider.choose(*context)
    assert calls == []


@pytest.mark.parametrize("url", [
    "http://ollama.example:11434", "http://127.0.0.1.evil.example:11434", "http://localhost.evil.example",
    "http://localhost.:11434", "http://user:password@localhost:11434", "http://localhost:11434?token=secret",
    "http://localhost:11434/#secret", "file:///tmp/socket", "http://localhost:99999", "http://localhost:not-a-port",
    "http://localhost:0", "http://localhost:11434/api", "http://[::1%25scope]:11434",
])
def test_invalid_or_insecure_endpoints_are_rejected(url):
    with pytest.raises(ProviderError):
        OllamaProvider(ProviderConfig(kind="ollama", base_url=url))


@pytest.mark.parametrize("url", ["http://localhost:11434", "http://127.0.0.2:11434", "http://[::1]:11434", "https://127.0.0.1:11434"])
def test_literal_loopback_endpoints_are_recognized(url):
    assert OllamaProvider(ProviderConfig(kind="ollama", base_url=url)).is_local


def test_remote_https_uses_only_configured_optional_key(context, monkeypatch):
    monkeypatch.setenv("HOI4_MODEL_API_KEY", "remote-test-key")
    provider, calls = server(endpoint="https://ollama.example:11434", allow_local=False, allow_external=True)
    provider.choose(*context)
    assert not provider.is_local
    assert all(request.headers["Authorization"] == "Bearer remote-test-key" for request in calls)
    assert all("remote-test-key" not in request.content.decode() for request in calls)


def test_remote_without_optional_key_does_not_invent_one(context, monkeypatch):
    monkeypatch.delenv("HOI4_MODEL_API_KEY", raising=False)
    provider, calls = server(endpoint="https://ollama.example", allow_external=True)
    provider.choose(*context)
    assert all("Authorization" not in request.headers for request in calls)


@pytest.mark.parametrize("show", [
    {"capabilities": ["completion"]},
    {},
    {"capabilities": "vision"},
    {"capabilities": ["vision", 3]},
    {"capabilities": ["vision"], "remote_host": "https://ollama.com"},
    {"capabilities": ["vision"], "remote_model": "qwen:cloud"},
])
def test_unsupported_or_cloud_models_are_rejected_before_image_upload(context, show):
    provider, calls = server(show=show)
    with pytest.raises(ProviderError):
        provider.choose(*context)
    assert all(request.url.path != "/api/chat" for request in calls)
    assert all(b"images" not in request.content for request in calls)


@pytest.mark.parametrize("name", ["gpt-oss:120b-cloud", "qwen:cloud", "QWEN:7B-CLOUD", "", "bad model", "model\n"])
def test_cloud_names_and_invalid_names_are_rejected_without_http(name):
    with pytest.raises(ProviderError):
        OllamaProvider(ProviderConfig(kind="ollama", model=name))


@pytest.mark.parametrize("action", [
    {"kind": "execute", "code": "import os"},
    {"kind": "click", "target_id": "focus_menu", "reason": "Нажать", "x": 3, "y": 4},
    {"kind": "click", "target_id": "other_target", "reason": "Нажать"},
    {"kind": "key", "key": "esc", "reason": "Закрыть"},
])
def test_unsafe_model_actions_never_leave_provider(context, action):
    provider, calls = server(chat=response(action))
    with pytest.raises(ProviderError):
        provider.choose(*context)
    assert len(calls) == 3


@pytest.mark.parametrize("chat", [
    response(done=False), response(done=1), response(done_reason="length"),
    response(message={"role": "assistant", "content": "Нужно выполнить код"}),
    response(message={"role": "tool", "content": "{}"}),
    response(message={"role": "assistant", "content": "{}", "tool_calls": [{"function": {"name": "shell"}}]}),
    response(message={"role": "assistant", "content": {"action": "click"}}),
])
def test_incomplete_or_tool_responses_are_rejected(context, chat):
    provider, _ = server(chat=chat)
    with pytest.raises(ProviderError):
        provider.choose(*context)


@pytest.mark.parametrize("failure", ["missing", "ambiguous", "stale"])
def test_unverified_target_does_not_make_any_http_request(context, failure):
    observation, step = context
    provider, calls = server()
    if failure == "missing":
        observation = replace(observation, targets={})
    elif failure == "ambiguous":
        observation = replace(observation, ambiguous=frozenset({step.target_id}))
    else:
        target = observation.targets[step.target_id].model_copy(update={"frame_id": "old"})
        observation = replace(observation, targets={step.target_id: target})
    with pytest.raises(ProviderError):
        provider.choose(observation, step)
    assert calls == []


def test_redirects_are_not_followed_even_with_injected_client(context):
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(307, headers={"Location": "https://elsewhere.example/collect"})

    with httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=True) as client:
        provider = OllamaProvider(config(), True, False, client)
        with pytest.raises(ProviderError):
            provider.choose(*context)
    assert len(calls) == 1
    assert calls[0].url.host == "127.0.0.1"


@pytest.mark.parametrize("failure", ["timeout", "model_missing", "malformed", "error_object", "secret_echo"])
def test_transport_errors_do_not_expose_key_or_response(monkeypatch, context, failure):
    secret = "sensitive-test-key"
    monkeypatch.setenv("HOI4_MODEL_API_KEY", secret)

    def handler(request):
        if failure == "timeout":
            raise httpx.ReadTimeout(secret, request=request)
        if failure == "model_missing":
            return httpx.Response(404, text="private server failure " + secret)
        if failure == "malformed":
            return httpx.Response(200, text="not JSON " + secret)
        if failure == "error_object":
            return httpx.Response(200, json={"error": secret})
        return httpx.Response(200, json={"version": secret})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    provider = OllamaProvider(ProviderConfig(kind="ollama", base_url="https://ollama.example"), False, True, client)
    with pytest.raises(ProviderError) as caught:
        provider.choose(*context)
    assert secret not in str(caught.value)
    assert "private server failure" not in str(caught.value)
    assert caught.value.__suppress_context__


def test_default_local_client_bypasses_environment_proxy(monkeypatch, context):
    original = httpx.Client
    kwargs_seen = []

    def handler(request):
        if request.url.path == "/api/version":
            return httpx.Response(200, json={"version": "0.13.0"})
        if request.url.path == "/api/show":
            return httpx.Response(200, json={"capabilities": ["vision"]})
        return httpx.Response(200, json=response())

    def factory(**kwargs):
        kwargs_seen.append(kwargs)
        return original(transport=httpx.MockTransport(handler), **kwargs)

    monkeypatch.setattr(httpx, "Client", factory)
    provider = OllamaProvider(config(), True)
    provider.choose(*context)
    assert len(kwargs_seen) == 3
    assert all(kwargs["trust_env"] is False and kwargs["verify"] is True for kwargs in kwargs_seen)
