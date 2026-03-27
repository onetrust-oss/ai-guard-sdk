from __future__ import annotations

import json
import ssl
import threading
import time
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, HTTPServer

_ALLOWED_METER_NAMES = {"ai_guard.agent", "ai_guard.user", "ai_guard.redact"}
_HISTOGRAM_METERS = {"ai_guard.agent"}


class TestClassificationService:
    __test__ = False

    def __init__(self, expected_token: str):
        self._expected_token = expected_token

        self._response_lock = threading.Lock()
        self._classification_handler: Callable[[dict], list[dict]] | None = None
        self._classification_matches: list[dict] | None = None
        self._classification_error: tuple[int, str] | None = None

        self._metrics: list[dict] = []
        self._metrics_lock = threading.Lock()

        self._server: HTTPServer | None = None
        self._thread: threading.Thread | None = None
        self._port: int | None = None

    @property
    def port(self) -> int:
        assert self._port is not None, "classification service not started"
        return self._port

    def set_classification_response(self, matches: list[dict]) -> None:
        with self._response_lock:
            self._classification_matches = matches
            self._classification_error = None
            self._classification_handler = None

    def set_classification_error(self, status: int, message: str) -> None:
        with self._response_lock:
            self._classification_error = (status, message)
            self._classification_matches = None
            self._classification_handler = None

    def set_classification_handler(
        self,
        handler: Callable[[dict], list[dict]],
    ) -> None:
        with self._response_lock:
            self._classification_handler = handler
            self._classification_matches = None
            self._classification_error = None

    def poll_metrics(
        self,
        min_count: int = 1,
        timeout: float = 15.0,
    ) -> list[dict]:
        deadline = time.monotonic() + timeout
        delay = 0.1
        while True:
            with self._metrics_lock:
                if len(self._metrics) >= min_count:
                    return list(self._metrics)
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"timed out waiting for {min_count} metric events after {timeout}s"
                )
            time.sleep(delay)
            delay = min(delay * 2, 2.0)

    def start(self, port: int, ssl_ctx: ssl.SSLContext | None = None) -> None:
        self._port = port
        expected_token = self._expected_token
        service_ref = self

        class _Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path.split("?")[0] == "/health":
                    self.send_response(200)
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                self.send_response(404)
                self.send_header("Content-Length", "0")
                self.end_headers()

            def do_POST(self):
                path = self.path.split("?")[0]
                if path == "/classifications/v1":
                    self._handle_classification()
                elif path == "/metric":
                    self._handle_metric()
                else:
                    self.send_response(404)
                    self.send_header("Content-Length", "0")
                    self.end_headers()

            def _check_auth(self) -> bool:
                auth_header = self.headers.get("Authorization", "")
                if not auth_header.startswith("Bearer "):
                    return False
                return auth_header[7:] == expected_token

            def _send_json(self, status: int, body: dict) -> None:
                data = json.dumps(body).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def _send_error(self, status: int, message: str) -> None:
                self._send_json(status, {"code": status, "message": message})

            def _handle_classification(self):
                if not self._check_auth():
                    self._send_error(401, "Unauthorized")
                    return

                length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(length)
                try:
                    request = json.loads(body)
                except json.JSONDecodeError:
                    self._send_error(400, "Invalid JSON")
                    return

                context = request.get("context", {})

                with service_ref._response_lock:
                    handler = service_ref._classification_handler
                    error = service_ref._classification_error
                    static_matches = service_ref._classification_matches

                if handler is not None:
                    matches = handler(request)
                elif error is not None:
                    self._send_error(error[0], error[1])
                    return
                else:
                    matches = static_matches if static_matches is not None else []

                self._send_json(200, {"context": context, "matches": matches})

            def _handle_metric(self):
                if not self._check_auth():
                    self._send_error(401, "Unauthorized")
                    return

                length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(length)
                try:
                    event = json.loads(body)
                except json.JSONDecodeError:
                    self._send_error(400, "Invalid JSON")
                    return

                meter = event.get("meter", {})
                meter_name = meter.get("name", "")

                if meter_name not in _ALLOWED_METER_NAMES:
                    self._send_error(400, f"Unknown meter: {meter_name}")
                    return

                with service_ref._metrics_lock:
                    service_ref._metrics.append(event)

                self.send_response(200)
                self.send_header("Content-Length", "0")
                self.end_headers()

            def log_message(self, format, *args):
                pass

        httpd = HTTPServer(("127.0.0.1", self._port), _Handler)
        if ssl_ctx is not None:
            httpd.socket = ssl_ctx.wrap_socket(httpd.socket, server_side=True)
        self._server = httpd
        self._thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None
