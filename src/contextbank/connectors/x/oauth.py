from __future__ import annotations

import base64
import hashlib
import json
import secrets
import time
import webbrowser
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlencode, urlparse

import httpx

from contextbank.paths import ContextBankPaths, secure_write_text

AUTHORIZE_URL = "https://x.com/i/oauth2/authorize"
TOKEN_URL = "https://api.x.com/2/oauth2/token"
DEFAULT_REDIRECT_URI = "http://127.0.0.1:8765/oauth/x/callback"
DEFAULT_SCOPES = ("tweet.read", "users.read", "bookmark.read")
OFFLINE_SCOPE = "offline.access"


@dataclass(frozen=True)
class StoredXToken:
    access_token: str
    token_type: str = "bearer"
    expires_at: datetime | None = None
    refresh_token: str | None = None
    scope: str | None = None
    client_id: str | None = None
    created_at: datetime | None = None

    @property
    def is_expired(self) -> bool:
        if self.expires_at is None:
            return False
        return datetime.now(UTC) >= self.expires_at - timedelta(seconds=60)

    @classmethod
    def from_token_response(
        cls,
        payload: dict[str, Any],
        *,
        client_id: str,
        fallback_refresh_token: str | None = None,
    ) -> StoredXToken:
        expires_in = payload.get("expires_in")
        expires_at = None
        if isinstance(expires_in, int):
            expires_at = datetime.now(UTC) + timedelta(seconds=expires_in)
        return cls(
            access_token=str(payload["access_token"]),
            token_type=str(payload.get("token_type") or "bearer"),
            expires_at=expires_at,
            refresh_token=str(payload.get("refresh_token") or fallback_refresh_token or "")
            or None,
            scope=str(payload.get("scope") or "") or None,
            client_id=client_id,
            created_at=datetime.now(UTC),
        )

    @classmethod
    def model_validate(cls, data: dict[str, Any]) -> StoredXToken:
        return cls(
            access_token=str(data["access_token"]),
            token_type=str(data.get("token_type") or "bearer"),
            expires_at=_parse_datetime(data.get("expires_at")),
            refresh_token=data.get("refresh_token") or None,
            scope=data.get("scope") or None,
            client_id=data.get("client_id") or None,
            created_at=_parse_datetime(data.get("created_at")),
        )

    def model_dump(self) -> dict[str, Any]:
        return {
            "access_token": self.access_token,
            "token_type": self.token_type,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "refresh_token": self.refresh_token,
            "scope": self.scope,
            "client_id": self.client_id,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


def token_store_path(paths: ContextBankPaths) -> Path:
    return paths.home / "auth" / "x_token.json"


def load_stored_token(paths: ContextBankPaths) -> StoredXToken | None:
    path = token_store_path(paths)
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as input_file:
        return StoredXToken.model_validate(json.load(input_file))


def save_stored_token(paths: ContextBankPaths, token: StoredXToken) -> Path:
    path = token_store_path(paths)
    secure_write_text(path, json.dumps(token.model_dump(), indent=2, sort_keys=True))
    return path


def delete_stored_token(paths: ContextBankPaths) -> bool:
    path = token_store_path(paths)
    if not path.exists():
        return False
    path.unlink()
    return True


def build_scopes(*, include_offline: bool = False) -> tuple[str, ...]:
    scopes = list(DEFAULT_SCOPES)
    if include_offline:
        scopes.append(OFFLINE_SCOPE)
    return tuple(scopes)


def generate_code_verifier() -> str:
    verifier = secrets.token_urlsafe(64)
    return verifier[:128]


def code_challenge(verifier: str) -> str:
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    return base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")


def build_authorization_url(
    *,
    client_id: str,
    redirect_uri: str,
    scopes: tuple[str, ...],
    state: str,
    verifier: str,
) -> str:
    query = urlencode(
        {
            "response_type": "code",
            "client_id": client_id,
            "redirect_uri": redirect_uri,
            "scope": " ".join(scopes),
            "state": state,
            "code_challenge": code_challenge(verifier),
            "code_challenge_method": "S256",
        }
    )
    return f"{AUTHORIZE_URL}?{query}"


def exchange_authorization_code(
    *,
    client_id: str,
    code: str,
    redirect_uri: str,
    verifier: str,
    client: httpx.Client | None = None,
) -> StoredXToken:
    payload = _post_token(
        {
            "grant_type": "authorization_code",
            "client_id": client_id,
            "code": code,
            "redirect_uri": redirect_uri,
            "code_verifier": verifier,
        },
        client=client,
    )
    return StoredXToken.from_token_response(payload, client_id=client_id)


def refresh_stored_token(
    token: StoredXToken,
    *,
    client_id: str | None = None,
    client: httpx.Client | None = None,
) -> StoredXToken:
    resolved_client_id = client_id or token.client_id
    if not resolved_client_id:
        raise ValueError("X OAuth refresh requires a client id.")
    if not token.refresh_token:
        raise ValueError("Stored X token does not include a refresh token.")
    payload = _post_token(
        {
            "grant_type": "refresh_token",
            "client_id": resolved_client_id,
            "refresh_token": token.refresh_token,
        },
        client=client,
    )
    return StoredXToken.from_token_response(
        payload,
        client_id=resolved_client_id,
        fallback_refresh_token=token.refresh_token,
    )


def run_local_authorization_flow(
    *,
    client_id: str,
    redirect_uri: str = DEFAULT_REDIRECT_URI,
    scopes: tuple[str, ...] = DEFAULT_SCOPES,
    open_browser: bool = True,
    timeout_seconds: int = 180,
    client: httpx.Client | None = None,
    authorization_url_callback: Callable[[str], None] | None = None,
) -> StoredXToken:
    verifier = generate_code_verifier()
    state = secrets.token_urlsafe(32)
    authorization_url = build_authorization_url(
        client_id=client_id,
        redirect_uri=redirect_uri,
        scopes=scopes,
        state=state,
        verifier=verifier,
    )
    if authorization_url_callback:
        authorization_url_callback(authorization_url)
    callback = wait_for_oauth_callback(
        redirect_uri=redirect_uri,
        authorization_url=authorization_url,
        expected_state=state,
        open_browser=open_browser,
        timeout_seconds=timeout_seconds,
    )
    return exchange_authorization_code(
        client_id=client_id,
        code=callback.code,
        redirect_uri=redirect_uri,
        verifier=verifier,
        client=client,
    )


@dataclass(frozen=True)
class OAuthCallback:
    code: str
    state: str


def wait_for_oauth_callback(
    *,
    redirect_uri: str,
    authorization_url: str,
    expected_state: str,
    open_browser: bool,
    timeout_seconds: int,
) -> OAuthCallback:
    parsed = urlparse(redirect_uri)
    if parsed.scheme != "http" or parsed.hostname != "127.0.0.1":
        raise ValueError("Local OAuth callback must use http://127.0.0.1.")
    port = parsed.port
    if port is None:
        raise ValueError("Local OAuth callback must include a port.")
    callback_path = parsed.path or "/"
    result: dict[str, str] = {}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            request_url = urlparse(self.path)
            if request_url.path != callback_path:
                self.send_error(404)
                return
            query = parse_qs(request_url.query)
            if "error" in query:
                result["error"] = query["error"][0]
            if "code" in query:
                result["code"] = query["code"][0]
            if "state" in query:
                result["state"] = query["state"][0]
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(
                b"<html><body><h1>ContextBank X authorization received.</h1>"
                b"<p>You can close this browser tab.</p></body></html>"
            )

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            return

    server = HTTPServer(("127.0.0.1", port), Handler)
    server.timeout = 0.25
    try:
        if open_browser:
            webbrowser.open(authorization_url)
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline and "code" not in result and "error" not in result:
            server.handle_request()
    finally:
        server.server_close()

    if result.get("error"):
        raise RuntimeError(f"X authorization failed: {result['error']}")
    if result.get("state") != expected_state:
        raise RuntimeError("X authorization failed state validation.")
    if not result.get("code"):
        raise TimeoutError("Timed out waiting for X OAuth callback.")
    return OAuthCallback(code=result["code"], state=result["state"])


def _post_token(data: dict[str, str], *, client: httpx.Client | None = None) -> dict[str, Any]:
    created_client = client is None
    client = client or httpx.Client(timeout=20.0, trust_env=False)
    try:
        response = client.post(
            TOKEN_URL,
            data=data,
            headers={
                "Accept": "application/json",
                "Content-Type": "application/x-www-form-urlencoded",
                "User-Agent": "ContextBank/0.1",
            },
        )
    finally:
        if created_client:
            client.close()
    payload = response.json() if response.content else {}
    if response.status_code >= 400:
        raise RuntimeError(_token_error_message(payload, response.status_code))
    if not isinstance(payload, dict) or not payload.get("access_token"):
        raise RuntimeError("X token endpoint response did not include an access token.")
    return payload


def _token_error_message(payload: Any, status_code: int) -> str:
    if isinstance(payload, dict):
        detail = payload.get("error_description") or payload.get("error") or payload
        return f"X token endpoint failed with HTTP {status_code}: {detail}"
    return f"X token endpoint failed with HTTP {status_code}."


def _parse_datetime(value: Any) -> datetime | None:
    if not value:
        return None
    parsed = datetime.fromisoformat(str(value))
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)
