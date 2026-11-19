import json
import sys
import textwrap

import pytest

from ftserve.server import Client, serve

FAKE = textwrap.dedent("""
    import json, sys
    from http.server import BaseHTTPRequestHandler, HTTPServer
    port = int(sys.argv[sys.argv.index("--port") + 1])
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a): pass
        def do_GET(self):
            self.send_response(200); self.end_headers(); self.wfile.write(b"{}")
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            assert body["response_format"]["type"] == "json_schema" and body["temperature"] == 0.0
            out = {"choices": [{"message": {"content": '{"facts": []}'}}],
                   "usage": {"prompt_tokens": 50, "completion_tokens": 6},
                   "timings": {"cache_n": 40, "prompt_ms": 12.0, "predicted_ms": 30.0}}
            data = json.dumps(out).encode()
            self.send_response(200); self.send_header("Content-Length", str(len(data))); self.end_headers()
            self.wfile.write(data)
    HTTPServer(("127.0.0.1", port), H).serve_forever()
""")


@pytest.fixture
def fake_binary(tmp_path):
    script = tmp_path / "fake_server.py"
    script.write_text(FAKE)
    binary = tmp_path / "llama-server"
    binary.write_text(f'#!/bin/sh\nexec {sys.executable} {script} "$@"\n')
    binary.chmod(0o755)
    return binary


def test_serve_and_chat(fake_binary, tmp_path):
    with serve(fake_binary, tmp_path / "m.gguf", port=18431, startup_s=20) as client:
        c = client.chat([{"role": "user", "content": "x"}], {"type": "object"})
    assert json.loads(c.text) == {"facts": []}
    assert (c.prompt_tokens, c.completion_tokens, c.cached_tokens) == (50, 6, 40)
    assert not Client("http://127.0.0.1:18431").healthy()  # stopped


def test_serve_reports_a_crash(tmp_path):
    binary = tmp_path / "llama-server"
    binary.write_text("#!/bin/sh\nexit 3\n")
    binary.chmod(0o755)
    with (
        pytest.raises(RuntimeError, match="exited with 3"),
        serve(binary, tmp_path / "m.gguf", port=18432, startup_s=10),
    ):
        pass
