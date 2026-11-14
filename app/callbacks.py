"""Authenticated callbacks to explicitly configured destinations; no redirects/proxies."""

from dataclasses import dataclass
import hashlib
import hmac
import http.client
import ssl
import time
from urllib.parse import urlsplit


@dataclass(frozen=True)
class CallbackTarget:
    scheme: str
    host: str
    port: int
    path: str


def _parse(url: str) -> tuple[CallbackTarget, str]:
    if len(url) > 2048 or not url.isascii() or any(ord(char) <= 32 or ord(char) == 127 for char in url):
        raise ValueError("Callback URL must be an ASCII URL of at most 2048 characters, without whitespace")
    parsed = urlsplit(url)
    if (
        parsed.scheme not in {"http", "https"} or not parsed.hostname
        or parsed.username is not None or parsed.password is not None or parsed.fragment
    ):
        raise ValueError("Callback URL must use HTTP(S), without credentials or fragment")
    port = parsed.port if parsed.port is not None else (443 if parsed.scheme == "https" else 80)
    if not 1 <= port <= 65535:
        raise ValueError("Invalid callback port")
    host = parsed.hostname.lower()
    authority = f"[{host}]" if ":" in host else host
    origin = f"{parsed.scheme}://{authority}:{port}"
    path = parsed.path or "/"
    if parsed.query:
        path += "?" + parsed.query
    return CallbackTarget(parsed.scheme, host, port, path), origin


def parse_callback_target(url: str, allowed_origins: tuple[str, ...]) -> CallbackTarget:
    target, origin = _parse(url)
    allowed = set()
    for item in allowed_origins:
        parsed, canonical = _parse(item)
        if parsed.path != "/" or urlsplit(item).query:
            raise ValueError("CALLBACK_ALLOWED_ORIGINS entries must contain only scheme, host and port")
        allowed.add(canonical)
    if origin not in allowed:
        raise ValueError("Callback origin is not in CALLBACK_ALLOWED_ORIGINS")
    return target


def signature(secret: str, timestamp: str, event_id: str, body: bytes) -> str:
    signed = timestamp.encode("ascii") + b"." + event_id.encode("ascii") + b"." + body
    return "v1=" + hmac.new(secret.encode(), signed, hashlib.sha256).hexdigest()


def send_callback(target: CallbackTarget, body: bytes, event_id: str, secret: str, timeout: float) -> int:
    timestamp = str(int(time.time()))
    headers = {
        "Content-Type": "application/json; charset=utf-8",
        "User-Agent": "FunASR-Callback/1.1",
        "X-FunASR-Event-Id": event_id,
        "X-FunASR-Timestamp": timestamp,
        "X-FunASR-Signature": signature(secret, timestamp, event_id, body),
    }
    if target.scheme == "https":
        connection = http.client.HTTPSConnection(
            target.host, target.port, timeout=timeout, context=ssl.create_default_context(),
        )
    else:
        connection = http.client.HTTPConnection(target.host, target.port, timeout=timeout)
    try:
        connection.request("POST", target.path, body=body, headers=headers)
        # Acknowledgment is the response status. Never buffer an unbounded response body.
        return connection.getresponse().status
    finally:
        connection.close()
