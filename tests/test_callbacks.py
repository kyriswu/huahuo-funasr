import hashlib
import hmac
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import threading
import time

import pytest

from app.callbacks import parse_callback_target, send_callback


@pytest.mark.parametrize("url", [
    "file:///tmp/audio", "https://user:password@receiver.example/hook", "https://receiver.example/hook#fragment",
    "https://receiver.example:0/hook", "https://receiver.example:99999/hook", "https://receiver.example/\r\nx",
    "https://other.example/hook", "http://169.254.169.254/latest/meta-data/",
])
def test_callbacks_require_explicit_origin_and_valid_url(url):
    with pytest.raises(ValueError):
        parse_callback_target(url, ("https://receiver.example",))


def test_callback_http_signature_full_body_and_no_redirects():
    received = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = self.rfile.read(int(self.headers["Content-Length"]))
            received.append((self.path, body, dict(self.headers)))
            self.send_response(302)
            self.send_header("Location", "/must-not-follow")
            self.send_header("Content-Length", "0")
            self.end_headers()

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        origin = f"http://127.0.0.1:{server.server_port}"
        target = parse_callback_target(origin + "/hook?source=asr", (origin,))
        body, event_id, secret = b'{"result":"test"}', "evt_1234", "independent-secret"
        assert send_callback(target, body, event_id, secret, 2) == 302
        assert len(received) == 1
        path, actual, headers = received[0]
        assert path == "/hook?source=asr" and actual == body
        timestamp = headers["X-FunASR-Timestamp"]
        assert abs(int(timestamp) - time.time()) < 5
        expected = "v1=" + hmac.new(secret.encode(), timestamp.encode() + b"." + event_id.encode() + b"." + actual, hashlib.sha256).hexdigest()
        assert headers["X-FunASR-Signature"] == expected
        assert headers["X-FunASR-Event-Id"] == event_id
    finally:
        server.shutdown()
        server.server_close()
        thread.join(5)
