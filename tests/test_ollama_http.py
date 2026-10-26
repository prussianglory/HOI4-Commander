"""Настоящий loopback HTTP и фиктивный Ollama API: без модели и игрового ввода."""
import base64
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading
import time

import cv2
import numpy as np
import pytest

from hoi4_commander.config import ProviderConfig
from hoi4_commander.model_provider import ProviderError
from hoi4_commander.models import ClickAction, Observation, Rect, Step, Target, WindowInfo
from hoi4_commander.ollama_provider import OllamaProvider


@pytest.fixture
def fake_ollama_http():
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            # Тестовый HTTP-сервер не выводит заголовки и тела запросов.
            pass

        def reply(self, body):
            encoded = json.dumps(body, ensure_ascii=False).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def do_GET(self):
            requests.append((self.path, dict(self.headers), b""))
            assert self.path == "/api/version"
            self.reply({"version": "0.13.0"})

        def do_POST(self):
            body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            requests.append((self.path, dict(self.headers), body))
            payload = json.loads(body)
            if self.path == "/api/show":
                self.reply({"capabilities": ["completion", "vision"]})
            else:
                assert self.path == "/api/chat"
                context = json.loads(payload["messages"][1]["content"])
                plan = {"action": {"kind": "click", "target_id": context["allowed_target_id"],
                                   "reason": "Открыть экономические фокусы"}}
                self.reply({"done": True, "done_reason": "stop",
                            "message": {"role": "assistant", "content": json.dumps(plan, ensure_ascii=False)}})

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    thread = threading.Thread(target=lambda: server.serve_forever(poll_interval=0.02), daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", requests
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def context():
    image = np.random.default_rng(1936).integers(0, 255, (100, 100, 3), dtype=np.uint8)
    target = Target(id="focus_menu", rect=Rect(x=10, y=20, width=30, height=20), confidence=1, frame_id="current")
    observation = Observation("current", time.monotonic(),
                              WindowInfo(handle=1, title="Тест", process_name="hoi4.exe", client=Rect(x=0, y=0, width=100, height=100)),
                              image, {target.id: target})
    step = Step(id="open_focus", target_id="focus_menu", expected_present=("focus_available",), explanation="Открыть фокусы")
    return observation, step


def test_real_http_bypasses_proxy_and_never_sends_local_api_key(fake_ollama_http, monkeypatch):
    endpoint, requests = fake_ollama_http
    secret = "local-private-key-never-transmitted"
    monkeypatch.setenv("HOI4_MODEL_API_KEY", secret)
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        monkeypatch.setenv(name, "http://127.0.0.1:1")
    monkeypatch.setenv("NO_PROXY", "")
    monkeypatch.setenv("no_proxy", "")
    provider = OllamaProvider(ProviderConfig(kind="ollama", base_url=endpoint, timeout_seconds=2),
                              allow_local_screenshots=True)
    observation, step = context()
    plan = provider.choose(observation, step)
    assert isinstance(plan.action, ClickAction)
    assert plan.action.target_id == step.target_id
    assert [path for path, _, _ in requests] == ["/api/version", "/api/show", "/api/chat"]
    assert all("authorization" not in {key.lower() for key in headers} for _, headers, _ in requests)
    assert all(secret not in str(headers) and secret.encode() not in body for _, headers, body in requests)
    payload = json.loads(requests[-1][2])
    png = base64.b64decode(payload["messages"][1]["images"][0])
    decoded = cv2.imdecode(np.frombuffer(png, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert np.array_equal(decoded, observation.image)
    assert payload["stream"] is False
    assert payload["format"]["type"] == "object"


def test_real_http_server_receives_nothing_without_local_consent(fake_ollama_http):
    endpoint, requests = fake_ollama_http
    provider = OllamaProvider(ProviderConfig(kind="ollama", base_url=endpoint, timeout_seconds=2))
    with pytest.raises(ProviderError, match="не разрешена"):
        provider.choose(*context())
    assert requests == []
