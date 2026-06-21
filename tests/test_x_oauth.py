from __future__ import annotations

from datetime import UTC, datetime
from urllib.parse import parse_qs, urlparse

import httpx

from contextbank.connectors.x.oauth import (
    DEFAULT_SCOPES,
    StoredXToken,
    build_authorization_url,
    code_challenge,
    exchange_authorization_code,
    generate_code_verifier,
    load_stored_token,
    refresh_stored_token,
    save_stored_token,
    token_store_path,
)
from contextbank.paths import ContextBankPaths


def test_x_oauth_authorization_url_uses_pkce_s256() -> None:
    verifier = generate_code_verifier()
    url = build_authorization_url(
        client_id="client-123",
        redirect_uri="http://127.0.0.1:8765/oauth/x/callback",
        scopes=DEFAULT_SCOPES,
        state="state-123",
        verifier=verifier,
    )

    parsed = urlparse(url)
    query = parse_qs(parsed.query)

    assert parsed.netloc == "x.com"
    assert query["response_type"] == ["code"]
    assert query["client_id"] == ["client-123"]
    assert query["scope"] == ["tweet.read users.read bookmark.read"]
    assert query["code_challenge_method"] == ["S256"]
    assert query["code_challenge"] == [code_challenge(verifier)]


def test_x_oauth_token_store_is_private_and_round_trips(tmp_path) -> None:
    paths = ContextBankPaths.from_home(tmp_path / "bank")
    paths.ensure()
    token = StoredXToken(
        access_token="access-token",
        refresh_token="refresh-token",
        client_id="client-123",
        scope="tweet.read users.read bookmark.read offline.access",
        expires_at=datetime(2026, 6, 20, tzinfo=UTC),
        created_at=datetime(2026, 6, 20, tzinfo=UTC),
    )

    saved = save_stored_token(paths, token)
    loaded = load_stored_token(paths)

    assert saved == token_store_path(paths)
    assert loaded == token
    assert (saved.stat().st_mode & 0o777) == 0o600


def test_x_oauth_exchange_and_refresh_parse_token_responses() -> None:
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        body = request.content.decode()
        if "authorization_code" in body:
            return httpx.Response(
                200,
                json={
                    "access_token": "first-access",
                    "refresh_token": "first-refresh",
                    "expires_in": 7200,
                    "token_type": "bearer",
                    "scope": "tweet.read users.read bookmark.read offline.access",
                },
            )
        return httpx.Response(
            200,
            json={
                "access_token": "refreshed-access",
                "expires_in": 7200,
                "token_type": "bearer",
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))

    token = exchange_authorization_code(
        client_id="client-123",
        code="code-123",
        redirect_uri="http://127.0.0.1:8765/oauth/x/callback",
        verifier="verifier",
        client=client,
    )
    refreshed = refresh_stored_token(token, client=client)

    assert token.access_token == "first-access"
    assert token.refresh_token == "first-refresh"
    assert refreshed.access_token == "refreshed-access"
    assert refreshed.refresh_token == "first-refresh"
    assert len(calls) == 2


def test_x_oauth_exchange_disables_env_proxy_by_default(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeClient:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def post(self, *args, **kwargs):
            return httpx.Response(
                200,
                json={
                    "access_token": "first-access",
                    "expires_in": 7200,
                    "token_type": "bearer",
                    "scope": "tweet.read users.read bookmark.read",
                },
            )

        def close(self):
            return None

    monkeypatch.setattr("contextbank.connectors.x.oauth.httpx.Client", FakeClient)

    token = exchange_authorization_code(
        client_id="client-123",
        code="code-123",
        redirect_uri="http://127.0.0.1:8765/oauth/x/callback",
        verifier="verifier",
    )

    assert token.access_token == "first-access"
    assert captured["trust_env"] is False
