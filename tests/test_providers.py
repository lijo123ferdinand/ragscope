import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from types import SimpleNamespace

import pytest

from ragscope.llm import AnthropicProvider, MockProvider, OpenAICompatProvider, ProviderError, make_provider


class _Handler(BaseHTTPRequestHandler):
    status = 200
    seen: list = []

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        type(self).seen.append((self.path, dict(self.headers), body))
        if type(self).status != 200:
            self.send_response(type(self).status)
            self.end_headers()
            self.wfile.write(b'{"error": "nope"}')
            return
        payload = json.dumps({"choices": [{"message": {"content": "pong [1]."}}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args):
        pass


@pytest.fixture
def server():
    _Handler.status, _Handler.seen = 200, []
    httpd = HTTPServer(("127.0.0.1", 0), _Handler)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{httpd.server_port}/v1"
    httpd.shutdown()


def test_openai_compat_request_shape(server):
    p = OpenAICompatProvider(model="m1", base_url=server, api_key="test-key")
    assert p.complete("sys", "hello", max_tokens=7) == "pong [1]."
    path, headers, body = _Handler.seen[0]
    assert path == "/v1/chat/completions"
    assert headers["Authorization"] == "Bearer test-key"
    assert body["model"] == "m1" and body["max_tokens"] == 7 and body["temperature"] == 0
    assert body["messages"] == [{"role": "system", "content": "sys"}, {"role": "user", "content": "hello"}]


def test_openai_compat_no_key_sends_no_auth_header(server, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    OpenAICompatProvider(base_url=server).complete("s", "p")
    assert "Authorization" not in _Handler.seen[0][1]


def test_openai_compat_client_error_is_not_retried(server):
    _Handler.status = 401
    with pytest.raises(ProviderError, match="HTTP 401"):
        OpenAICompatProvider(base_url=server, max_retries=3).complete("s", "p")
    assert len(_Handler.seen) == 1


def test_openai_compat_unreachable(monkeypatch):
    p = OpenAICompatProvider(base_url="http://127.0.0.1:9/v1", max_retries=0, timeout=1)
    with pytest.raises(ProviderError, match="cannot reach"):
        p.complete("s", "p")


def test_anthropic_provider_with_injected_client():
    calls = {}

    class Messages:
        def create(self, **kw):
            calls.update(kw)
            return SimpleNamespace(content=[SimpleNamespace(type="text", text="hi "), SimpleNamespace(type="text", text="there")])

    p = AnthropicProvider(client=SimpleNamespace(messages=Messages()))
    assert p.complete("sys", "q", max_tokens=9) == "hi there"
    assert calls["model"] == "claude-sonnet-5-5"
    assert calls["system"] == "sys" and calls["max_tokens"] == 9
    assert calls["messages"] == [{"role": "user", "content": "q"}]


def test_mock_rejects_unknown_task():
    with pytest.raises(ProviderError):
        MockProvider().complete("no task header", "x")


def test_make_provider():
    assert make_provider("mock").name == "mock"
    assert make_provider("ollama", "qwen").model == "qwen"
    with pytest.raises(ValueError):
        make_provider("nope")
