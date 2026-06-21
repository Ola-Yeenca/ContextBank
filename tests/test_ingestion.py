from __future__ import annotations

from pathlib import Path

import httpcore
import httpx
import pytest

from contextbank.connectors.adapters import (
    LocalFileSourceAdapter,
    ManualTextSourceAdapter,
    WebSourceAdapter,
    XBookmarkSourceAdapter,
    create_source_adapter,
    source_adapter_manifest,
)
from contextbank.connectors.base import SourceAdapter
from contextbank.connectors.files.importer import import_file
from contextbank.connectors.text.importer import import_text
from contextbank.connectors.web.extractor import _GuardedNetworkBackend, ingest_url
from contextbank.connectors.x.client import (
    OWNED_READ_UNIT_COST_USD,
    XBookmarkClient,
    XRateLimitError,
    estimate_bookmark_sync,
)
from contextbank.models import AvailabilityStatus, ExtractionStatus
from contextbank.paths import ContextBankPaths


def _public_resolver(hostname: str) -> list[str]:
    assert hostname == "example.com"
    return ["93.184.216.34"]


def test_import_markdown_preserves_raw_and_extracted_text(tmp_path: Path) -> None:
    paths = ContextBankPaths.from_home(tmp_path / "bank")
    source = tmp_path / "note.md"
    source.write_text("# Useful Note\n\nLocal-first memory.", encoding="utf-8")

    result = import_file(source, paths)

    assert result.source_item.source_type == "file"
    assert result.source_item.availability_status == AvailabilityStatus.AVAILABLE
    assert result.document.extraction_status == ExtractionStatus.COMPLETE
    assert Path(result.source_item.raw_path).read_text(encoding="utf-8") == source.read_text(
        encoding="utf-8"
    )
    assert Path(result.document.extracted_text_path).read_text(encoding="utf-8").startswith(
        "# Useful Note"
    )
    assert Path(result.source_item.raw_path).is_relative_to(paths.raw / "files")
    assert Path(result.document.extracted_text_path).is_relative_to(paths.documents)


def test_import_json_and_html_report_honest_status(tmp_path: Path) -> None:
    paths = ContextBankPaths.from_home(tmp_path / "bank")
    json_source = tmp_path / "data.json"
    json_source.write_text('{"b": 2, "a": 1}', encoding="utf-8")
    html_source = tmp_path / "page.html"
    html_source.write_text(
        "<html><head><title>Hello</title><meta name='author' content='Ada Example'>"
        "<meta property='article:published_time' content='2026-06-20T10:00:00Z'></head>"
        "<body><main><h1>Hi</h1><h2>Details</h2><script>no</script>"
        "<p>Visible text.</p><pre><code class='language-python'>print('ok')</code></pre>"
        "</main></body></html>",
        encoding="utf-8",
    )

    json_result = import_file(json_source, paths)
    html_result = import_file(html_source, paths)

    assert json_result.document.extraction_status == ExtractionStatus.COMPLETE
    assert '"a": 1' in json_result.document.text
    assert html_result.document.extraction_status == ExtractionStatus.COMPLETE
    assert html_result.document.title == "Hello"
    assert html_result.document.author == "Ada Example"
    assert html_result.document.published_at.isoformat() == "2026-06-20T10:00:00+00:00"
    assert "Visible text." in html_result.document.text
    assert "no" not in html_result.document.text
    assert html_result.document.metadata["headings"] == [
        {"level": 1, "text": "Hi"},
        {"level": 2, "text": "Details"},
    ]
    assert html_result.document.metadata["code_blocks"] == [
        {"language": "python", "text": "print('ok')"}
    ]
    assert html_result.document.metadata["extracted_at"]


def test_manual_text_import_preserves_raw_and_extracted_text(tmp_path: Path) -> None:
    paths = ContextBankPaths.from_home(tmp_path / "bank")

    result = import_text(
        "Pasted architecture note\n\nUse bounded context packs.",
        paths,
        source_url="https://example.com/original",
        title="Architecture note",
    )

    assert result.source_item.source_type == "text"
    assert result.source_item.source_url == "https://example.com/original"
    assert result.document.document_type == "manual_text"
    assert result.document.extraction_status == ExtractionStatus.COMPLETE
    assert result.document.title == "Architecture note"
    assert Path(result.source_item.raw_path).read_text(encoding="utf-8").startswith(
        "Pasted architecture"
    )
    assert Path(result.document.extracted_text_path).is_relative_to(paths.documents)


def test_local_source_adapters_fetch_and_report_unsupported_listing(tmp_path: Path) -> None:
    paths = ContextBankPaths.from_home(tmp_path / "bank")
    source = tmp_path / "note.md"
    source.write_text("# Adapter note\n\nShared connector contract.", encoding="utf-8")

    file_adapter = LocalFileSourceAdapter(paths)
    file_item = file_adapter.fetch_one(str(source))
    file_listing = file_adapter.list_new_or_changed()

    assert file_adapter.capabilities.fetch_one is True
    assert file_item.source_item.source_type == "file"
    assert file_item.document.title == "note.md"
    assert file_listing.errors[0].code == "unsupported"

    text_adapter = ManualTextSourceAdapter(paths)
    text_item = text_adapter.fetch_one(
        "Pasted adapter note",
        source_url="https://example.com/adapter",
        title="Adapter note",
    )

    assert text_item.source_item.source_type == "text"
    assert text_item.document.document_type == "manual_text"


def test_source_adapter_registry_exposes_common_contract(tmp_path: Path) -> None:
    paths = ContextBankPaths.from_home(tmp_path / "bank")

    manifest = source_adapter_manifest()
    names = {item["name"] for item in manifest}

    assert names == {"file", "text", "web", "x.bookmarks"}
    assert all(
        item["capabilities"]["normalized_documents"]
        for item in manifest
        if item["name"] != "x.bookmarks"
    )
    assert next(item for item in manifest if item["name"] == "x.bookmarks")[
        "requires_credentials"
    ] is True
    assert next(item for item in manifest if item["name"] == "x.bookmarks")[
        "capabilities"
    ]["list_items"] is True

    for name in ("web", "file", "text"):
        adapter = create_source_adapter(name, paths=paths)
        assert isinstance(adapter, SourceAdapter)
        assert adapter.capabilities.fetch_one is True
        listing = adapter.list_new_or_changed()
        assert listing.errors[0].code == "unsupported"

    alias_adapter = create_source_adapter("manual-text", paths=paths)
    assert isinstance(alias_adapter, ManualTextSourceAdapter)

    with pytest.raises(ValueError, match="requires an authenticated XBookmarkClient"):
        create_source_adapter("x", paths=paths)


def test_web_source_adapter_fetches_one_url_with_structured_errors(tmp_path: Path) -> None:
    paths = ContextBankPaths.from_home(tmp_path / "bank")

    def fake_fetch(url, paths_arg):
        assert paths_arg == paths
        return import_text(f"Fetched {url}", paths_arg, source_url=url, title="Fetched page")

    adapter = WebSourceAdapter(paths, fetch=fake_fetch)
    item = adapter.fetch_one("https://example.com/page")

    assert item.source_item.source_url == "https://example.com/page"
    assert item.errors == []

    def failing_fetch(url, paths_arg):
        raise ValueError("bad url")

    failing_adapter = WebSourceAdapter(paths, fetch=failing_fetch)
    failed_item = failing_adapter.fetch_one("bad")
    assert failed_item.errors[0].message == "bad url"


def test_web_ingest_writes_raw_html_and_extracted_text(tmp_path: Path) -> None:
    paths = ContextBankPaths.from_home(tmp_path / "bank")

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url == "https://example.com/article?x=1"
        return httpx.Response(
            200,
            headers={"content-type": "text/html; charset=utf-8"},
            text=(
                "<html><head><title>Article</title></head>"
                "<body><main>Body text</main></body></html>"
            ),
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    result = ingest_url(
        "Example.COM/article?x=1#ignored",
        paths,
        client=client,
        host_resolver=_public_resolver,
    )

    assert result.source_item.source_url == "https://example.com/article?x=1"
    assert result.source_item.availability_status == AvailabilityStatus.AVAILABLE
    assert result.document.extraction_status == ExtractionStatus.COMPLETE
    assert result.document.title == "Article"
    assert Path(result.source_item.raw_path).read_text(encoding="utf-8").startswith("<html>")
    assert Path(result.document.extracted_text_path).read_text(encoding="utf-8") == "Body text"


def test_web_ingest_extracts_page_metadata_headings_and_code(tmp_path: Path) -> None:
    paths = ContextBankPaths.from_home(tmp_path / "bank")

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url == "https://example.com/article"
        return httpx.Response(
            200,
            headers={"content-type": "text/html; charset=utf-8"},
            text=(
                "<html><head>"
                "<title>Fallback Title</title>"
                "<meta property='og:title' content='Structured Article'>"
                "<meta name='author' content='Grace Hopper'>"
                "<meta property='article:published_time' content='2026-06-21T09:30:00Z'>"
                "<link rel='canonical' href='/canonical-article'>"
                "</head><body><article>"
                "<h1>Structured Article</h1><h2>Implementation</h2>"
                "<p>Readable article text.</p>"
                "<pre><code class='language-python'>print('ship')</code></pre>"
                "</article></body></html>"
            ),
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    result = ingest_url(
        "https://example.com/article",
        paths,
        client=client,
        host_resolver=_public_resolver,
    )

    assert result.document.extraction_status == ExtractionStatus.COMPLETE
    assert result.document.canonical_url == "https://example.com/canonical-article"
    assert result.document.title == "Structured Article"
    assert result.document.author == "Grace Hopper"
    assert result.document.published_at.isoformat() == "2026-06-21T09:30:00+00:00"
    assert result.document.metadata["headings"] == [
        {"level": 1, "text": "Structured Article"},
        {"level": 2, "text": "Implementation"},
    ]
    assert result.document.metadata["code_blocks"] == [
        {"language": "python", "text": "print('ship')"}
    ]
    assert result.document.metadata["extracted_at"]


def test_web_ingest_blocks_private_network_targets_by_default(tmp_path: Path) -> None:
    paths = ContextBankPaths.from_home(tmp_path / "bank")

    with pytest.raises(ValueError, match="private or local"):
        ingest_url("http://127.0.0.1:8000/private", paths)


@pytest.mark.parametrize(
    "url",
    [
        "http://2130706433/",
        "http://0x7f000001/",
        "http://017700000001/",
        "http://0177.0.0.1/",
    ],
)
def test_web_ingest_blocks_encoded_private_network_targets_by_default(
    tmp_path: Path,
    url: str,
) -> None:
    paths = ContextBankPaths.from_home(tmp_path / "bank")

    with pytest.raises(ValueError, match="private or local"):
        ingest_url(url, paths)


def test_web_ingest_blocks_hostnames_resolving_to_private_networks(tmp_path: Path) -> None:
    paths = ContextBankPaths.from_home(tmp_path / "bank")

    with pytest.raises(ValueError, match="private or local"):
        ingest_url(
            "https://metadata.example/latest",
            paths,
            host_resolver=lambda hostname: ["169.254.169.254"],
        )


def test_web_ingest_blocks_if_any_resolved_hostname_address_is_private(
    tmp_path: Path,
) -> None:
    paths = ContextBankPaths.from_home(tmp_path / "bank")

    with pytest.raises(ValueError, match="private or local"):
        ingest_url(
            "https://rebind.example/article",
            paths,
            host_resolver=lambda hostname: ["93.184.216.34", "127.0.0.1"],
        )


def test_guarded_network_backend_connects_to_validated_ip() -> None:
    backend = _GuardedNetworkBackend(
        allow_private_networks=False,
        host_resolver=lambda hostname: ["93.184.216.34"],
    )
    fake_backend = _FakeNetworkBackend()
    backend._backend = fake_backend  # noqa: SLF001

    with pytest.raises(httpcore.ConnectError, match="stop before real connect"):
        backend.connect_tcp("rebind.example", 443)

    assert fake_backend.calls == [
        {
            "host": "93.184.216.34",
            "port": 443,
            "timeout": None,
            "local_address": None,
            "socket_options": None,
        }
    ]


def test_guarded_network_backend_blocks_private_rebinding_before_connect() -> None:
    backend = _GuardedNetworkBackend(
        allow_private_networks=False,
        host_resolver=lambda hostname: ["127.0.0.1"],
    )
    fake_backend = _FakeNetworkBackend()
    backend._backend = fake_backend  # noqa: SLF001

    with pytest.raises(httpcore.ConnectError, match="private or local"):
        backend.connect_tcp("rebind.example", 80)

    assert fake_backend.calls == []


class _FakeNetworkBackend:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def connect_tcp(
        self,
        host: str,
        port: int,
        timeout: float | None = None,
        local_address: str | None = None,
        socket_options: object | None = None,
    ) -> None:
        self.calls.append(
            {
                "host": host,
                "port": port,
                "timeout": timeout,
                "local_address": local_address,
                "socket_options": socket_options,
            }
        )
        raise httpcore.ConnectError("stop before real connect")


def test_web_ingest_allows_unresolvable_host_to_fail_at_fetch_layer(tmp_path: Path) -> None:
    paths = ContextBankPaths.from_home(tmp_path / "bank")

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url == "https://unresolved.example/article"
        return httpx.Response(
            200,
            headers={"content-type": "text/plain"},
            text="Fetched through mock transport.",
        )

    def unresolved(hostname: str) -> list[str]:
        raise OSError(f"unresolvable host in test: {hostname}")

    result = ingest_url(
        "https://unresolved.example/article",
        paths,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        host_resolver=unresolved,
    )

    assert result.document.extraction_status == ExtractionStatus.COMPLETE
    assert result.document.text == "Fetched through mock transport."


def test_web_ingest_blocks_redirects_to_private_network_targets(tmp_path: Path) -> None:
    paths = ContextBankPaths.from_home(tmp_path / "bank")

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url == "https://example.com/redirect"
        return httpx.Response(302, headers={"location": "http://127.0.0.1/private"})

    client = httpx.Client(transport=httpx.MockTransport(handler))

    with pytest.raises(ValueError, match="private or local"):
        ingest_url(
            "https://example.com/redirect",
            paths,
            client=client,
            host_resolver=_public_resolver,
        )


def test_web_ingest_truncates_large_bodies_before_storage(tmp_path: Path) -> None:
    paths = ContextBankPaths.from_home(tmp_path / "bank")
    client = httpx.Client(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                headers={"content-type": "text/plain"},
                content=b"abcdefghij",
            )
        )
    )

    result = ingest_url(
        "https://example.com/large",
        paths,
        client=client,
        max_response_bytes=4,
        host_resolver=_public_resolver,
    )

    assert result.document.metadata["body_truncated"] is True
    assert result.document.metadata["stored_byte_size"] == 4
    assert Path(result.source_item.raw_path).read_bytes() == b"abcd"
    assert result.document.text == "abcd"
    assert "truncated" in result.warnings[0]


def test_web_ingest_preserves_raw_for_unsupported_content_as_partial(tmp_path: Path) -> None:
    paths = ContextBankPaths.from_home(tmp_path / "bank")
    client = httpx.Client(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                headers={"content-type": "application/octet-stream"},
                content=b"\x00\x01",
            )
        )
    )

    result = ingest_url(
        "https://example.com/download",
        paths,
        client=client,
        host_resolver=_public_resolver,
    )

    assert result.source_item.availability_status == AvailabilityStatus.PARTIAL
    assert result.document.extraction_status == ExtractionStatus.PARTIAL
    assert result.document.extracted_text_path is None
    assert Path(result.source_item.raw_path).read_bytes() == b"\x00\x01"
    assert "preserved" in result.warnings[0]


def test_web_ingest_reports_http_failures_without_fake_extraction(tmp_path: Path) -> None:
    paths = ContextBankPaths.from_home(tmp_path / "bank")
    client = httpx.Client(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                404,
                text="not found",
                headers={"content-type": "text/plain"},
            )
        )
    )

    result = ingest_url(
        "https://example.com/missing",
        paths,
        client=client,
        host_resolver=_public_resolver,
    )

    assert result.source_item.availability_status == AvailabilityStatus.UNAVAILABLE
    assert result.document.extraction_status == ExtractionStatus.FAILED
    assert result.document.text is None
    assert result.document.extraction_error.startswith("HTTP 404")


def test_x_bookmark_client_requests_bookmarks_with_pagination_and_rate_headers() -> None:
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        assert request.headers["authorization"] == "Bearer test-token"
        if "pagination_token=next" in str(request.url):
            return httpx.Response(
                200,
                headers={
                    "x-rate-limit-limit": "180",
                    "x-rate-limit-remaining": "178",
                    "x-rate-limit-reset": "1893456000",
                },
                json={"data": [{"id": "2", "text": "second"}], "meta": {"result_count": 1}},
            )
        return httpx.Response(
            200,
            headers={
                "x-rate-limit-limit": "180",
                "x-rate-limit-remaining": "179",
                "x-rate-limit-reset": "1893456000",
            },
            json={
                "data": [{"id": "1", "text": "first"}],
                "meta": {"result_count": 1, "next_token": "next"},
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(handler), base_url="https://api.x.test")
    x_client = XBookmarkClient("test-token", base_url="https://api.x.test", client=client)

    bookmarks = list(x_client.iter_bookmarks("123", limit=2, page_size=1))

    assert [bookmark["id"] for bookmark in bookmarks] == ["1", "2"]
    assert len(calls) == 2
    first_url = str(calls[0].url)
    assert "tweet.fields=" in first_url
    assert "conversation_id" in first_url
    assert "referenced_tweets" in first_url
    assert "expansions=" in first_url
    assert "attachments.media_keys" in first_url
    assert "referenced_tweets.id" in first_url
    assert "media.fields=" in first_url
    assert "alt_text" in first_url
    assert x_client.auth_assumptions["required_scopes"] == [
        "tweet.read",
        "users.read",
        "bookmark.read",
    ]


def test_x_bookmark_client_fetches_one_post_with_default_metadata_fields() -> None:
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        assert request.headers["authorization"] == "Bearer test-token"
        return httpx.Response(
            200,
            headers={
                "x-rate-limit-limit": "180",
                "x-rate-limit-remaining": "179",
                "x-rate-limit-reset": "1893456000",
            },
            json={
                "data": {
                    "id": "123",
                    "text": "single post",
                    "author_id": "42",
                    "conversation_id": "123",
                    "created_at": "2026-06-20T10:00:00Z",
                    "lang": "en",
                },
                "includes": {
                    "users": [{"id": "42", "name": "Example", "username": "example_user"}]
                },
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(handler), base_url="https://api.x.test")
    x_client = XBookmarkClient("test-token", base_url="https://api.x.test", client=client)

    post = x_client.get_post("123")

    assert post.id == "123"
    assert post.data["text"] == "single post"
    assert post.rate_limit.remaining == 179
    assert len(calls) == 1
    request_url = str(calls[0].url)
    assert "/2/tweets/123" in request_url
    assert "tweet.fields=" in request_url
    assert "conversation_id" in request_url
    assert "expansions=" in request_url
    assert "author_id" in request_url


def test_x_bookmark_client_disables_env_proxy_by_default(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeClient:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def get(self, *args, **kwargs):
            return httpx.Response(
                200,
                json={"data": {"id": "123", "name": "Example", "username": "example_user"}},
            )

        def close(self):
            return None

    monkeypatch.setattr("contextbank.connectors.x.client.httpx.Client", FakeClient)

    user = XBookmarkClient("test-token").get_authenticated_user()

    assert user["id"] == "123"
    assert captured["trust_env"] is False


def test_x_bookmark_source_adapter_lists_fetches_and_normalizes_metadata() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if url.endswith("/2/users/me?user.fields=id%2Cname%2Cusername"):
            return httpx.Response(
                200,
                json={"data": {"id": "123", "name": "Example", "username": "example_user"}},
            )
        if "/2/users/123/bookmarks" in url:
            return httpx.Response(
                200,
                json={
                    "data": [
                        {
                            "id": "111",
                            "text": "bookmark",
                            "author_id": "42",
                            "conversation_id": "111",
                            "created_at": "2026-06-20T10:00:00Z",
                            "lang": "en",
                        }
                    ],
                    "meta": {"result_count": 1, "next_token": "next"},
                    "includes": {
                        "users": [{"id": "42", "name": "Author", "username": "author"}]
                    },
                },
            )
        if "/2/tweets/111" in url:
            return httpx.Response(
                200,
                json={
                    "data": {
                        "id": "111",
                        "text": "lookup",
                        "author_id": "42",
                        "conversation_id": "111",
                        "created_at": "2026-06-20T10:00:00Z",
                        "lang": "en",
                    },
                    "includes": {
                        "users": [{"id": "42", "name": "Author", "username": "author"}]
                    },
                },
            )
        return httpx.Response(404, json={"errors": [{"title": "unexpected"}]})

    client = httpx.Client(transport=httpx.MockTransport(handler), base_url="https://api.x.test")
    adapter = XBookmarkSourceAdapter(
        XBookmarkClient("test-token", base_url="https://api.x.test", client=client)
    )

    auth = adapter.authenticate()
    page = adapter.list_new_or_changed(limit=1)
    fetched = adapter.fetch_one("111")

    assert auth.errors == []
    assert adapter.user_id == "123"
    assert page.continuation_state == "next"
    assert page.items[0].external_id == "111"
    assert page.items[0].metadata["author_handle"] == "author"
    assert fetched.metadata["source_url"] == "https://x.com/i/web/status/111"
    assert fetched.metadata["author_name"] == "Author"


def test_x_bookmark_client_raises_typed_rate_limit_error() -> None:
    client = httpx.Client(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                429,
                headers={"x-rate-limit-reset": "1893456000"},
                json={"errors": [{"title": "Rate limit exceeded"}]},
            )
        )
    )
    x_client = XBookmarkClient("test-token", client=client, max_retries=0)

    with pytest.raises(XRateLimitError) as exc:
        x_client.get_bookmarks("123")

    assert exc.value.status_code == 429
    assert exc.value.rate_limit.reset_epoch == 1893456000


def test_x_bookmark_client_retries_rate_limit_with_bounded_backoff() -> None:
    calls = 0
    slept: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(
                429,
                headers={"x-rate-limit-reset": "1893456000"},
                json={"errors": [{"title": "Rate limit exceeded"}]},
            )
        return httpx.Response(
            200,
            json={"data": [{"id": "1", "text": "after retry"}], "meta": {"result_count": 1}},
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    x_client = XBookmarkClient(
        "test-token",
        client=client,
        max_retries=1,
        backoff_seconds=0.25,
        sleep=slept.append,
    )

    page = x_client.get_bookmarks("123")

    assert calls == 2
    assert slept == [0.25]
    assert page.data[0]["text"] == "after retry"


def test_x_dry_run_estimate_uses_labeled_current_docs_estimate() -> None:
    estimate = estimate_bookmark_sync(250)

    assert estimate.dry_run is True
    assert estimate.requested_limit == 250
    assert estimate.estimated_resources == 250
    assert estimate.estimated_cost_usd == 250 * OWNED_READ_UNIT_COST_USD
    assert "Dry-run estimate only" in estimate.pricing_note
    assert "2026-06-21" in estimate.pricing_note
