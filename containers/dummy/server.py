"""Dummy Triton-compatible server for exercising the reconciler lifecycle by hand.

Speaks enough of the KServe v2 API that traefik routing and the /status projection behave like the
real tritonserver, while boot time, health and exit behaviour are driven entirely by environment
variables -- so every ObservedState the reconciler can produce is reachable on demand, without
pulling the 19GB triton image or waiting for a real model load.

The platform spawns services with `--load-model=<name>` arguments; those are parsed here so the
model endpoints report exactly the models the service was created with.

Environment:
    DUMMY_BOOT_DELAY  seconds to wait before binding the port at all (default 0). Models a slow
                      model load: the container is `running` but nothing is listening yet, which
                      is precisely the window where an uptime-only readiness check lies.
    DUMMY_LIFETIME    seconds to stay up before exiting (default 0 = forever).
    DUMMY_EXIT_CODE   exit code used when DUMMY_LIFETIME elapses (default 0). 0 -> EXITED_OK,
                      non-zero -> CRASHED.
    DUMMY_HEALTHY     when false, /v2/health/ready serves 503 forever while the process stays
                      alive -- a wedged server, which only becomes visible to the reconciler when
                      the service carries a healthcheck.
    DUMMY_PORT        listen port (default 8000, matching triton and the traefik loadBalancer url).
"""

import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

SERVER_NAME = "triton-dummy"
SERVER_VERSION = "2.36.0"
EXTENSIONS = ["classification", "sequence", "model_repository", "schedule_policy", "statistics"]


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    return int(raw) if raw else default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    return raw not in ("0", "false", "no") if raw else default


BOOT_DELAY = _env_int("DUMMY_BOOT_DELAY", 0)
LIFETIME = _env_int("DUMMY_LIFETIME", 0)
EXIT_CODE = _env_int("DUMMY_EXIT_CODE", 0)
HEALTHY = _env_bool("DUMMY_HEALTHY", True)
PORT = _env_int("DUMMY_PORT", 8000)

MODELS = [arg.split("=", 1)[1] for arg in sys.argv[1:] if arg.startswith("--load-model=")]


def _model_metadata(name: str) -> dict:
    return {
        "name": name,
        "versions": ["1"],
        "platform": "dummy",
        "inputs": [{"name": "INPUT__0", "datatype": "FP32", "shape": [-1, 3]}],
        "outputs": [{"name": "OUTPUT__0", "datatype": "FP32", "shape": [-1, 3]}],
    }


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args) -> None:
        sys.stdout.write("%s - %s\n" % (self.address_string(), fmt % args))

    def _send(self, code: int, payload: dict | None = None) -> None:
        body = json.dumps(payload).encode() if payload is not None else b""
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if body:
            self.wfile.write(body)

    def _route(self, path: str, method: str) -> tuple[int, dict | None]:
        parts = [p for p in path.split("?")[0].strip("/").split("/") if p]
        if not parts or parts[0] != "v2":
            return 404, {"error": "not found"}

        # /v2 -> server metadata
        if len(parts) == 1:
            return 200, {"name": SERVER_NAME, "version": SERVER_VERSION, "extensions": EXTENSIONS}

        # /v2/health/{live,ready}
        if parts[1] == "health" and len(parts) == 3:
            if parts[2] == "live":
                return 200, None  # the process is up; liveness never depends on DUMMY_HEALTHY
            if parts[2] == "ready":
                return (200, None) if HEALTHY else (503, {"error": "server not ready"})
            return 404, {"error": "not found"}

        # /v2/repository/index -> the models this service was told to load
        if parts[1] == "repository" and parts[2:] == ["index"]:
            return 200, [{"name": m, "version": "1", "state": "READY"} for m in MODELS]  # type: ignore[return-value]

        if parts[1] == "models" and len(parts) >= 3:
            name = parts[2]
            if name not in MODELS:
                return 404, {"error": f"Request for unknown model: '{name}' is not found"}
            tail = parts[3:]
            if not tail:
                return 200, _model_metadata(name)
            if tail == ["ready"]:
                return (200, None) if HEALTHY else (503, {"error": "model not ready"})
            if tail == ["config"]:
                return 200, {"name": name, "platform": "dummy", "max_batch_size": 0}
            if tail == ["stats"]:
                return 200, {"model_stats": [{"name": name, "version": "1"}]}
            if tail == ["infer"] and method == "POST":
                return 200, {
                    "model_name": name,
                    "model_version": "1",
                    "outputs": [{"name": "OUTPUT__0", "datatype": "FP32", "shape": [1, 3], "data": [0.0, 0.0, 0.0]}],
                }
        return 404, {"error": "not found"}

    def do_GET(self) -> None:
        code, payload = self._route(self.path, "GET")
        self._send(code, payload)

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        if length:
            self.rfile.read(length)
        code, payload = self._route(self.path, "POST")
        self._send(code, payload)


def _exit_after(seconds: int, code: int) -> None:
    time.sleep(seconds)
    print(f"[dummy] lifetime {seconds}s elapsed, exiting with {code}", flush=True)
    os._exit(code)


def main() -> None:
    print(f"[dummy] models={MODELS or '<none>'} boot_delay={BOOT_DELAY}s healthy={HEALTHY}", flush=True)
    if BOOT_DELAY:
        # deliberately before bind(): nothing is listening while the container is already `running`
        print(f"[dummy] simulating model load for {BOOT_DELAY}s (port closed)", flush=True)
        time.sleep(BOOT_DELAY)

    if LIFETIME:
        threading.Thread(target=_exit_after, args=(LIFETIME, EXIT_CODE), daemon=True).start()

    server = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    print(f"[dummy] listening on 0.0.0.0:{PORT}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
