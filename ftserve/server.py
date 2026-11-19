"""Run a GGUF model with llama.cpp's llama-server and call its OpenAI-compatible API.

Every model in the comparison is served the same way (same binary, same threads, same
constrained decoding), so differences in quality and speed come from the model, not the stack.
"""

from __future__ import annotations

import json
import subprocess
import time
import urllib.error
import urllib.request
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Completion:
    text: str
    prompt_tokens: int
    completion_tokens: int
    cached_tokens: int
    seconds: float
    prompt_ms: float
    predicted_ms: float


class Client:
    def __init__(self, base_url: str, timeout_s: float = 300.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self.last_raw: dict | None = None

    def _post(self, path: str, body: dict) -> dict:
        req = urllib.request.Request(
            self.base_url + path,
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=self.timeout_s) as r:
            return json.loads(r.read())

    def healthy(self) -> bool:
        try:
            with urllib.request.urlopen(self.base_url + "/health", timeout=5) as r:
                return r.status == 200
        except (urllib.error.URLError, OSError):
            return False

    def chat(self, messages: list[dict], schema: dict | None, max_tokens: int = 256) -> Completion:
        body: dict = {
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": max_tokens,
            "cache_prompt": True,
        }
        if schema is not None:
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "facts", "schema": schema},
            }
        t0 = time.perf_counter()
        r = self._post("/v1/chat/completions", body)
        seconds = time.perf_counter() - t0
        self.last_raw = r
        usage = r.get("usage", {})
        timings = r.get("timings", {})
        return Completion(
            text=r["choices"][0]["message"]["content"],
            prompt_tokens=int(usage.get("prompt_tokens", 0)),
            completion_tokens=int(usage.get("completion_tokens", 0)),
            cached_tokens=int(timings.get("cache_n", 0)),
            seconds=seconds,
            prompt_ms=float(timings.get("prompt_ms", 0.0)),
            predicted_ms=float(timings.get("predicted_ms", 0.0)),
        )


@contextmanager
def serve(
    binary: Path,
    model: Path,
    port: int = 8080,
    threads: int = 4,
    parallel: int = 1,
    ctx: int = 4096,
    log_file: Path | None = None,
    startup_s: float = 300.0,
):
    """Start llama-server, wait until it answers /health, yield a Client, stop it."""
    cmd = [
        str(binary),
        "-m", str(model),
        "--port", str(port),
        "--host", "127.0.0.1",
        "-t", str(threads),
        "-tb", str(threads),
        "-np", str(parallel),
        "-c", str(ctx * parallel),
    ]  # fmt: skip
    log = open(log_file, "w") if log_file else subprocess.DEVNULL  # noqa: SIM115
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
    client = Client(f"http://127.0.0.1:{port}")
    try:
        deadline = time.time() + startup_s
        while not client.healthy():
            if proc.poll() is not None:
                raise RuntimeError(f"llama-server exited with {proc.returncode}: {' '.join(cmd)}")
            if time.time() > deadline:
                raise TimeoutError("llama-server did not become healthy")
            time.sleep(0.5)
        yield client
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
        if log_file:
            log.close()
