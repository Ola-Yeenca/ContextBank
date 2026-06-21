from __future__ import annotations

import ipaddress
import mimetypes
import socket
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlsplit

import httpcore
import httpx

from contextbank.extraction.web import EXTRACTOR_VERSION, extract_html, parse_extracted_datetime
from contextbank.ids import content_hash, document_id, normalize_url, slugify, source_item_id
from contextbank.models import (
    AvailabilityStatus,
    Document,
    ExtractionStatus,
    SourceItem,
    SourceType,
)
from contextbank.paths import ContextBankPaths, secure_write_bytes, secure_write_text

DEFAULT_USER_AGENT = "ContextBank/0.1 (+https://github.com/contextbank/contextbank)"
MAX_RESPONSE_BYTES = 5 * 1024 * 1024
MAX_REDIRECTS = 5
HostResolver = Callable[[str], Iterable[str]]


@dataclass(frozen=True)
class UrlIngestResult:
    source_item: SourceItem
    document: Document
    warnings: list[str] = field(default_factory=list)
    response_headers: dict[str, str] = field(default_factory=dict)


class WebExtractor:
    def __init__(
        self,
        *,
        paths: ContextBankPaths | None = None,
        client: httpx.Client | None = None,
        timeout: float = 20.0,
        user_agent: str = DEFAULT_USER_AGENT,
        allow_private_networks: bool = False,
        max_response_bytes: int = MAX_RESPONSE_BYTES,
        host_resolver: HostResolver | None = None,
    ) -> None:
        self.paths = paths or ContextBankPaths.from_home()
        self._client = client
        self.timeout = timeout
        self.user_agent = user_agent
        self.allow_private_networks = allow_private_networks
        self.max_response_bytes = max_response_bytes
        self.host_resolver = host_resolver

    def ingest_url(self, url: str) -> UrlIngestResult:
        normalized_url = _normalize_manual_url(
            url,
            allow_private_networks=self.allow_private_networks,
            host_resolver=self.host_resolver,
        )
        item_id = source_item_id(SourceType.WEB.value, value=normalized_url)
        paths = self.paths
        paths.ensure()

        created_client = self._client is None
        client = self._client or _guarded_http_client(
            timeout=self.timeout,
            user_agent=self.user_agent,
            allow_private_networks=self.allow_private_networks,
            host_resolver=self.host_resolver,
        )

        try:
            response = _get_with_redirect_guard(
                client,
                normalized_url,
                allow_private_networks=self.allow_private_networks,
                host_resolver=self.host_resolver,
            )
        except httpx.HTTPError as exc:
            return _failed_url_result(
                paths=paths,
                item_id=item_id,
                normalized_url=normalized_url,
                error=f"HTTP request failed: {exc}",
                availability_status=AvailabilityStatus.UNAVAILABLE,
            )
        finally:
            if created_client:
                client.close()

        raw_bytes = response.content
        response_warnings: list[str] = []
        body_truncated = False
        if len(raw_bytes) > self.max_response_bytes:
            raw_bytes = raw_bytes[: self.max_response_bytes]
            body_truncated = True
            response_warnings.append(
                f"URL response body exceeded {self.max_response_bytes} bytes and was truncated."
            )
        raw_digest = content_hash(raw_bytes) if raw_bytes else None
        doc_id = document_id(item_id, "linked_page", raw_digest or normalized_url)
        content_type = response.headers.get("content-type", "")
        extension = _extension_for_content_type(content_type)
        raw_path: Path | None = None
        if raw_bytes:
            raw_path = _raw_web_path(
                paths,
                normalized_url,
                item_id,
                raw_digest or "empty",
                extension,
            )
            _write_bytes(raw_path, raw_bytes)

        if response.status_code >= 400:
            status = ExtractionStatus.FAILED
            document = _document(
                doc_id=doc_id,
                item_id=item_id,
                normalized_url=normalized_url,
                status=status,
                error=f"HTTP {response.status_code} while fetching URL.",
                raw_path=raw_path,
                extracted_text_path=None,
                content_hash_value=None,
                title=None,
                text=None,
                metadata={
                    "http_status": response.status_code,
                    "content_type": content_type,
                    "final_url": str(response.url),
                    "body_truncated": body_truncated,
                    "stored_byte_size": len(raw_bytes),
                },
            )
            source_item = _source_item(
                item_id=item_id,
                normalized_url=normalized_url,
                raw_path=raw_path,
                raw_digest=raw_digest,
                availability_status=AvailabilityStatus.UNAVAILABLE,
            )
            return UrlIngestResult(
                source_item=source_item,
                document=document,
                warnings=[
                    *response_warnings,
                    document.extraction_error or "URL extraction failed.",
                ],
                response_headers=dict(response.headers),
            )

        extraction = _extract_response_body(
            raw_bytes,
            content_type=content_type,
            normalized_url=normalized_url,
        )
        extracted_text_path: Path | None = None
        if extraction["text"]:
            extracted_text_path = _document_text_path(paths, doc_id)
            _write_text(extracted_text_path, extraction["text"])

        document = _document(
            doc_id=doc_id,
            item_id=item_id,
            normalized_url=normalized_url,
            status=extraction["status"],
            error=extraction["error"],
            raw_path=raw_path,
            extracted_text_path=extracted_text_path,
            content_hash_value=content_hash(extraction["text"]) if extraction["text"] else None,
            title=extraction["title"],
            text=extraction["text"] or None,
            metadata={
                "http_status": response.status_code,
                "content_type": content_type,
                "final_url": str(response.url),
                "body_truncated": body_truncated,
                "stored_byte_size": len(raw_bytes),
                **extraction["metadata"],
            },
        )
        source_item = _source_item(
            item_id=item_id,
            normalized_url=normalized_url,
            raw_path=raw_path,
            raw_digest=raw_digest,
            availability_status=_availability_for_extraction(extraction["status"]),
        )
        return UrlIngestResult(
            source_item=source_item,
            document=document,
            warnings=[*response_warnings, *extraction["warnings"]],
            response_headers=dict(response.headers),
        )


def ingest_url(
    url: str,
    paths: ContextBankPaths | None = None,
    *,
    client: httpx.Client | None = None,
    timeout: float = 20.0,
    allow_private_networks: bool = False,
    max_response_bytes: int = MAX_RESPONSE_BYTES,
    host_resolver: HostResolver | None = None,
) -> UrlIngestResult:
    return WebExtractor(
        paths=paths,
        client=client,
        timeout=timeout,
        allow_private_networks=allow_private_networks,
        max_response_bytes=max_response_bytes,
        host_resolver=host_resolver,
    ).ingest_url(url)


def fetch_and_extract_url(
    url: str,
    paths: ContextBankPaths | None = None,
    *,
    client: httpx.Client | None = None,
    timeout: float = 20.0,
    allow_private_networks: bool = False,
    max_response_bytes: int = MAX_RESPONSE_BYTES,
    host_resolver: HostResolver | None = None,
) -> UrlIngestResult:
    return ingest_url(
        url,
        paths,
        client=client,
        timeout=timeout,
        allow_private_networks=allow_private_networks,
        max_response_bytes=max_response_bytes,
        host_resolver=host_resolver,
    )


extract_url = ingest_url


class _GuardedNetworkBackend(httpcore.NetworkBackend):
    def __init__(
        self,
        *,
        allow_private_networks: bool,
        host_resolver: HostResolver | None,
    ) -> None:
        self.allow_private_networks = allow_private_networks
        self.host_resolver = host_resolver
        self._backend = httpcore.SyncBackend()

    def connect_tcp(
        self,
        host: str,
        port: int,
        timeout: float | None = None,
        local_address: str | None = None,
        socket_options: Iterable[httpcore.SOCKET_OPTION] | None = None,
    ) -> httpcore.NetworkStream:
        connect_host = host
        if not self.allow_private_networks:
            lowered_host = host.lower().strip("[]").rstrip(".")
            addresses = _hostname_ip_addresses(
                lowered_host,
                host_resolver=self.host_resolver,
            )
            if not addresses:
                raise httpcore.ConnectError(
                    f"Could not validate resolved URL address for hostname: {host}."
                )
            if any(_is_private_network_address(address) for address in addresses):
                raise httpcore.ConnectError(
                    "Refusing to ingest private or local network URLs by default."
                )
            connect_host = str(addresses[0])
        return self._backend.connect_tcp(
            connect_host,
            port,
            timeout=timeout,
            local_address=local_address,
            socket_options=socket_options,
        )

    def connect_unix_socket(
        self,
        path: str,
        timeout: float | None = None,
        socket_options: Iterable[httpcore.SOCKET_OPTION] | None = None,
    ) -> httpcore.NetworkStream:
        return self._backend.connect_unix_socket(
            path,
            timeout=timeout,
            socket_options=socket_options,
        )

    def sleep(self, seconds: float) -> None:
        self._backend.sleep(seconds)


class _GuardedHTTPTransport(httpx.HTTPTransport):
    def __init__(
        self,
        *,
        allow_private_networks: bool,
        host_resolver: HostResolver | None,
    ) -> None:
        ssl_context = httpx.create_ssl_context(verify=True, trust_env=False)
        self._pool = httpcore.ConnectionPool(
            ssl_context=ssl_context,
            max_connections=10,
            max_keepalive_connections=10,
            keepalive_expiry=5.0,
            http1=True,
            http2=False,
            network_backend=_GuardedNetworkBackend(
                allow_private_networks=allow_private_networks,
                host_resolver=host_resolver,
            ),
        )


def _guarded_http_client(
    *,
    timeout: float,
    user_agent: str,
    allow_private_networks: bool,
    host_resolver: HostResolver | None,
) -> httpx.Client:
    return httpx.Client(
        follow_redirects=False,
        timeout=timeout,
        headers={"User-Agent": user_agent},
        transport=_GuardedHTTPTransport(
            allow_private_networks=allow_private_networks,
            host_resolver=host_resolver,
        ),
        trust_env=False,
    )


def _normalize_manual_url(
    url: str,
    *,
    allow_private_networks: bool = False,
    host_resolver: HostResolver | None = None,
) -> str:
    candidate = url.strip()
    if "://" not in candidate:
        candidate = f"https://{candidate}"
    normalized = normalize_url(candidate)
    _validate_fetchable_url(
        normalized,
        allow_private_networks=allow_private_networks,
        host_resolver=host_resolver,
    )
    return normalized


def _get_with_redirect_guard(
    client: httpx.Client,
    url: str,
    *,
    allow_private_networks: bool,
    host_resolver: HostResolver | None,
) -> httpx.Response:
    current_url = url
    for _ in range(MAX_REDIRECTS + 1):
        _validate_fetchable_url(
            current_url,
            allow_private_networks=allow_private_networks,
            host_resolver=host_resolver,
        )
        response = client.get(current_url, follow_redirects=False)
        if not response.is_redirect:
            return response
        location = response.headers.get("location")
        if not location:
            return response
        current_url = normalize_url(urljoin(current_url, location))
        _validate_fetchable_url(
            current_url,
            allow_private_networks=allow_private_networks,
            host_resolver=host_resolver,
        )
    raise httpx.TooManyRedirects(f"URL redirected more than {MAX_REDIRECTS} times.")


def _validate_fetchable_url(
    url: str,
    *,
    allow_private_networks: bool,
    host_resolver: HostResolver | None = None,
) -> None:
    parsed = urlsplit(url)
    if parsed.scheme.lower() not in {"http", "https"}:
        raise ValueError("Only http(s) URLs can be ingested.")
    hostname = parsed.hostname
    if not hostname:
        raise ValueError("URL must include a hostname.")
    if allow_private_networks:
        return
    lowered_host = hostname.lower().strip("[]").rstrip(".")
    if lowered_host in {"localhost", "localhost.localdomain"} or lowered_host.endswith(
        ".localhost"
    ):
        raise ValueError("Refusing to ingest private or local network URLs by default.")
    addresses = _hostname_ip_addresses(lowered_host, host_resolver=host_resolver)
    if any(_is_private_network_address(address) for address in addresses):
        raise ValueError("Refusing to ingest private or local network URLs by default.")


def _hostname_ip_addresses(
    hostname: str,
    *,
    host_resolver: HostResolver | None = None,
) -> list[ipaddress.IPv4Address | ipaddress.IPv6Address]:
    literal = _literal_ip_address(hostname)
    if literal is not None:
        return [literal]
    resolver = host_resolver or _resolve_hostname_addresses
    try:
        raw_addresses = list(resolver(hostname))
    except OSError:
        return []
    addresses: list[ipaddress.IPv4Address | ipaddress.IPv6Address] = []
    for raw_address in raw_addresses:
        try:
            addresses.append(ipaddress.ip_address(str(raw_address)))
        except ValueError as exc:
            raise ValueError(
                f"Could not validate resolved URL address for hostname: {hostname}."
            ) from exc
    return addresses


def _literal_ip_address(hostname: str) -> ipaddress.IPv4Address | ipaddress.IPv6Address | None:
    try:
        return ipaddress.ip_address(hostname)
    except ValueError:
        pass
    if ":" in hostname:
        return None
    try:
        packed = socket.inet_aton(hostname)
    except OSError:
        return None
    return ipaddress.ip_address(socket.inet_ntoa(packed))


def _resolve_hostname_addresses(hostname: str) -> list[str]:
    try:
        infos = socket.getaddrinfo(hostname, None, type=socket.SOCK_STREAM)
    except socket.gaierror:
        return []
    return sorted({str(info[4][0]) for info in infos})


def _is_private_network_address(
    address: ipaddress.IPv4Address | ipaddress.IPv6Address,
) -> bool:
    return not address.is_global or address.is_multicast


def _extract_response_body(
    raw_bytes: bytes,
    *,
    content_type: str,
    normalized_url: str,
) -> dict[str, Any]:
    if not raw_bytes:
        return {
            "title": None,
            "text": "",
            "status": ExtractionStatus.FAILED,
            "error": "URL response body was empty.",
            "metadata": {},
            "warnings": ["Fetched URL but response body was empty."],
        }

    media_type = content_type.split(";", 1)[0].strip().lower()
    if media_type in {"", "text/html", "application/xhtml+xml"} or media_type.endswith("+html"):
        extraction = extract_html(raw_bytes, source_url=normalized_url)
        return {
            "title": extraction.title,
            "text": extraction.text,
            "status": extraction.status,
            "error": extraction.error,
            "metadata": extraction.metadata,
            "warnings": extraction.warnings,
        }

    if media_type.startswith("text/"):
        text = raw_bytes.decode("utf-8", errors="replace")
        return {
            "title": None,
            "text": text,
            "status": ExtractionStatus.COMPLETE if text.strip() else ExtractionStatus.FAILED,
            "error": None if text.strip() else "Text URL response contained no text.",
            "metadata": {"non_html_content": True},
            "warnings": ["URL response was not HTML; stored decoded text without HTML extraction."],
        }

    return {
        "title": None,
        "text": "",
        "status": ExtractionStatus.PARTIAL,
        "error": f"Unsupported URL content type for text extraction: {content_type or 'unknown'}",
        "metadata": {"non_html_content": True},
        "warnings": ["Raw URL response was preserved, but no text extractor is available."],
    }


def _failed_url_result(
    *,
    paths: ContextBankPaths,
    item_id: str,
    normalized_url: str,
    error: str,
    availability_status: AvailabilityStatus,
) -> UrlIngestResult:
    doc_id = document_id(item_id, "linked_page", normalized_url)
    source_item = _source_item(
        item_id=item_id,
        normalized_url=normalized_url,
        raw_path=None,
        raw_digest=None,
        availability_status=availability_status,
    )
    document = _document(
        doc_id=doc_id,
        item_id=item_id,
        normalized_url=normalized_url,
        status=ExtractionStatus.FAILED,
        error=error,
        raw_path=None,
        extracted_text_path=None,
        content_hash_value=None,
        title=None,
        text=None,
        metadata={},
    )
    paths.ensure()
    return UrlIngestResult(source_item=source_item, document=document, warnings=[error])


def _source_item(
    *,
    item_id: str,
    normalized_url: str,
    raw_path: Path | None,
    raw_digest: str | None,
    availability_status: AvailabilityStatus,
) -> SourceItem:
    return SourceItem(
        id=item_id,
        source_type=SourceType.WEB,
        source_url=normalized_url,
        raw_path=str(raw_path) if raw_path else None,
        content_hash=raw_digest,
        availability_status=availability_status,
    )


def _document(
    *,
    doc_id: str,
    item_id: str,
    normalized_url: str,
    status: ExtractionStatus,
    error: str | None,
    raw_path: Path | None,
    extracted_text_path: Path | None,
    content_hash_value: str | None,
    title: str | None,
    text: str | None,
    metadata: dict[str, Any],
) -> Document:
    canonical_url = metadata.get("canonical_url") or normalized_url
    return Document(
        id=doc_id,
        source_item_id=item_id,
        document_type="linked_page",
        canonical_url=canonical_url,
        title=title,
        author=metadata.get("author"),
        published_at=parse_extracted_datetime(metadata.get("published_at")),
        extracted_text_path=str(extracted_text_path) if extracted_text_path else None,
        html_snapshot_path=str(raw_path) if raw_path else None,
        extraction_status=status,
        extraction_error=error,
        extractor_version=EXTRACTOR_VERSION,
        content_hash=content_hash_value,
        text=text,
        metadata=metadata,
    )


def _availability_for_extraction(status: ExtractionStatus) -> AvailabilityStatus:
    if status == ExtractionStatus.COMPLETE:
        return AvailabilityStatus.AVAILABLE
    if status == ExtractionStatus.PARTIAL:
        return AvailabilityStatus.PARTIAL
    return AvailabilityStatus.UNAVAILABLE


def _raw_web_path(
    paths: ContextBankPaths,
    url: str,
    item_id: str,
    digest: str,
    extension: str,
) -> Path:
    slug = slugify(url.split("://", 1)[-1])[:80]
    return _safe_child(paths.raw / "web", f"{item_id}-{digest[:12]}-{slug}{extension}")


def _document_text_path(paths: ContextBankPaths, doc_id: str) -> Path:
    return _safe_child(paths.documents, f"{doc_id}.txt")


def _extension_for_content_type(content_type: str) -> str:
    media_type = content_type.split(";", 1)[0].strip().lower()
    if media_type in {"text/html", "application/xhtml+xml"} or media_type.endswith("+html"):
        return ".html"
    if media_type.startswith("text/"):
        return ".txt"
    return mimetypes.guess_extension(media_type) or ".bin"


def _write_bytes(path: Path, data: bytes) -> None:
    secure_write_bytes(path, data)


def _write_text(path: Path, text: str) -> None:
    secure_write_text(path, text)


def _safe_child(base: Path, filename: str) -> Path:
    resolved_base = base.resolve()
    candidate = (resolved_base / filename).resolve()
    if not candidate.is_relative_to(resolved_base):
        raise ValueError(f"Refusing to write outside ContextBank path: {candidate}")
    return candidate
