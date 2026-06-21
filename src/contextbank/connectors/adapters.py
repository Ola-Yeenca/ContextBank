from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from contextbank.connectors.base import (
    SourceAdapter,
    SourceAdapterCapabilities,
    SourceAdapterError,
    SourceAdapterItem,
    SourceAdapterPage,
)
from contextbank.connectors.files.importer import import_file
from contextbank.connectors.text.importer import import_text
from contextbank.connectors.web.extractor import ingest_url
from contextbank.connectors.x.client import XApiError, XBookmarkClient
from contextbank.models import SourceType
from contextbank.paths import ContextBankPaths


@dataclass(frozen=True)
class SourceAdapterRegistration:
    name: str
    source_type: str
    capabilities: SourceAdapterCapabilities
    description: str
    aliases: tuple[str, ...] = ()
    requires_credentials: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "source_type": self.source_type,
            "description": self.description,
            "aliases": list(self.aliases),
            "requires_credentials": self.requires_credentials,
            "capabilities": {
                "authenticate": self.capabilities.authenticate,
                "list_items": self.capabilities.list_items,
                "fetch_one": self.capabilities.fetch_one,
                "continuation_state": self.capabilities.continuation_state,
                "normalized_documents": self.capabilities.normalized_documents,
            },
        }


class WebSourceAdapter:
    name = "web"
    source_type = SourceType.WEB.value
    capabilities = SourceAdapterCapabilities(fetch_one=True, normalized_documents=True)

    def __init__(
        self,
        paths: ContextBankPaths | None = None,
        *,
        fetch: Callable[..., Any] = ingest_url,
    ) -> None:
        self.paths = paths
        self._fetch = fetch

    def authenticate(self) -> SourceAdapterPage:
        return _unsupported_page(self.name, "Web sources do not require authentication.")

    def list_new_or_changed(
        self,
        *,
        limit: int | None = None,
        continuation_state: str | None = None,
    ) -> SourceAdapterPage:
        return _unsupported_page(self.name, "Web URL imports are caller-provided, not listed.")

    def fetch_one(self, item_id: str, **kwargs: Any) -> SourceAdapterItem:
        try:
            result = self._fetch(item_id, self.paths, **kwargs)
        except Exception as exc:
            return _error_item(self.name, str(exc), item_id=item_id)
        return _ingest_result_item(item_id, result)

    def normalize_metadata(self, raw: dict[str, Any]) -> dict[str, Any]:
        return {
            "source_type": self.source_type,
            "source_url": raw.get("url") or raw.get("source_url"),
            "title": raw.get("title"),
        }


class LocalFileSourceAdapter:
    name = "file"
    source_type = SourceType.FILE.value
    capabilities = SourceAdapterCapabilities(fetch_one=True, normalized_documents=True)

    def __init__(
        self,
        paths: ContextBankPaths | None = None,
        *,
        fetch: Callable[..., Any] = import_file,
    ) -> None:
        self.paths = paths
        self._fetch = fetch

    def authenticate(self) -> SourceAdapterPage:
        return _unsupported_page(self.name, "Local file sources do not require authentication.")

    def list_new_or_changed(
        self,
        *,
        limit: int | None = None,
        continuation_state: str | None = None,
    ) -> SourceAdapterPage:
        return _unsupported_page(self.name, "Local file imports are caller-provided, not listed.")

    def fetch_one(self, item_id: str, **kwargs: Any) -> SourceAdapterItem:
        try:
            result = self._fetch(Path(item_id).expanduser(), self.paths, **kwargs)
        except Exception as exc:
            return _error_item(self.name, str(exc), item_id=item_id)
        return _ingest_result_item(item_id, result)

    def normalize_metadata(self, raw: dict[str, Any]) -> dict[str, Any]:
        return {
            "source_type": self.source_type,
            "path": raw.get("path") or raw.get("source_external_id"),
            "title": raw.get("title"),
        }


class ManualTextSourceAdapter:
    name = "text"
    source_type = SourceType.TEXT.value
    capabilities = SourceAdapterCapabilities(fetch_one=True, normalized_documents=True)

    def __init__(
        self,
        paths: ContextBankPaths | None = None,
        *,
        fetch: Callable[..., Any] = import_text,
    ) -> None:
        self.paths = paths
        self._fetch = fetch

    def authenticate(self) -> SourceAdapterPage:
        return _unsupported_page(self.name, "Manual text sources do not require authentication.")

    def list_new_or_changed(
        self,
        *,
        limit: int | None = None,
        continuation_state: str | None = None,
    ) -> SourceAdapterPage:
        return _unsupported_page(self.name, "Manual text imports are caller-provided, not listed.")

    def fetch_one(self, item_id: str, **kwargs: Any) -> SourceAdapterItem:
        try:
            result = self._fetch(item_id, self.paths, **kwargs)
        except Exception as exc:
            return _error_item(self.name, str(exc), item_id=item_id)
        return _ingest_result_item(item_id, result)

    def normalize_metadata(self, raw: dict[str, Any]) -> dict[str, Any]:
        return {
            "source_type": self.source_type,
            "source_url": raw.get("source_url"),
            "title": raw.get("title"),
        }


class XBookmarkSourceAdapter:
    name = "x.bookmarks"
    source_type = SourceType.X.value
    capabilities = SourceAdapterCapabilities(
        authenticate=True,
        list_items=True,
        fetch_one=True,
        continuation_state=True,
    )

    def __init__(
        self,
        client: XBookmarkClient,
        *,
        user_id: str | None = None,
    ) -> None:
        self.client = client
        self.user_id = user_id

    def authenticate(self) -> SourceAdapterPage:
        try:
            user = self.client.get_authenticated_user()
        except XApiError as exc:
            return SourceAdapterPage(errors=[_adapter_error(self.name, str(exc))])
        if self.user_id and str(user.get("id") or "") != str(self.user_id):
            return SourceAdapterPage(
                raw={"authenticated_user": user},
                errors=[
                    _adapter_error(
                        self.name,
                        "Authenticated X user did not match configured bookmark owner.",
                        item_id=self.user_id,
                    )
                ],
            )
        self.user_id = str(user.get("id") or self.user_id or "")
        return SourceAdapterPage(raw={"authenticated_user": user})

    def list_new_or_changed(
        self,
        *,
        limit: int | None = None,
        continuation_state: str | None = None,
    ) -> SourceAdapterPage:
        if not self.user_id:
            auth = self.authenticate()
            if auth.errors:
                return auth
        try:
            page = self.client.get_bookmarks(
                str(self.user_id),
                max_results=min(max(limit or 100, 1), 100),
                pagination_token=continuation_state,
            )
        except XApiError as exc:
            return SourceAdapterPage(errors=[_adapter_error(self.name, str(exc))])
        return SourceAdapterPage(
            items=[
                SourceAdapterItem(
                    external_id=str(bookmark.get("id") or ""),
                    raw={"bookmark": bookmark, "includes": page.includes},
                    metadata=self.normalize_metadata(
                        {"bookmark": bookmark, "includes": page.includes}
                    ),
                )
                for bookmark in page.data
            ],
            continuation_state=page.next_token,
            raw=page.raw,
            errors=[
                _adapter_error(
                    self.name,
                    str(error.get("detail") or error.get("title") or error),
                    code=str(error.get("type") or "") or None,
                )
                for error in page.errors
                if isinstance(error, dict)
            ],
        )

    def fetch_one(self, item_id: str, **kwargs: Any) -> SourceAdapterItem:
        try:
            post = self.client.get_post(item_id, **kwargs)
        except XApiError as exc:
            return _error_item(self.name, str(exc), item_id=item_id)
        raw = {"post": post.data, "includes": post.includes, "errors": post.errors}
        return SourceAdapterItem(
            external_id=post.id,
            raw=raw,
            metadata=self.normalize_metadata(raw),
            errors=[
                _adapter_error(
                    self.name,
                    str(error.get("detail") or error.get("title") or error),
                    item_id=post.id,
                    code=str(error.get("type") or "") or None,
                )
                for error in post.errors
                if isinstance(error, dict)
            ],
        )

    def normalize_metadata(self, raw: dict[str, Any]) -> dict[str, Any]:
        post = raw.get("post") or raw.get("bookmark") or raw.get("data") or {}
        includes = raw.get("includes") or {}
        author = _included_author(post, includes)
        post_id = str(post.get("id") or "")
        return {
            "source_type": self.source_type,
            "source_external_id": post_id,
            "source_url": f"https://x.com/i/web/status/{post_id}" if post_id else None,
            "author_id": post.get("author_id"),
            "author_name": author.get("name"),
            "author_handle": author.get("username"),
            "source_created_at": post.get("created_at"),
            "language": post.get("lang"),
            "conversation_id": post.get("conversation_id"),
        }


ADAPTER_REGISTRATIONS: dict[str, SourceAdapterRegistration] = {
    WebSourceAdapter.name: SourceAdapterRegistration(
        name=WebSourceAdapter.name,
        source_type=WebSourceAdapter.source_type,
        capabilities=WebSourceAdapter.capabilities,
        description="Fetch and normalize one caller-supplied http(s) web page.",
        aliases=("url", "web-page"),
    ),
    LocalFileSourceAdapter.name: SourceAdapterRegistration(
        name=LocalFileSourceAdapter.name,
        source_type=LocalFileSourceAdapter.source_type,
        capabilities=LocalFileSourceAdapter.capabilities,
        description="Import one local Markdown, text, JSON, or HTML file.",
        aliases=("local-file",),
    ),
    ManualTextSourceAdapter.name: SourceAdapterRegistration(
        name=ManualTextSourceAdapter.name,
        source_type=ManualTextSourceAdapter.source_type,
        capabilities=ManualTextSourceAdapter.capabilities,
        description="Normalize pasted or piped user-provided text.",
        aliases=("manual-text", "pasted-text"),
    ),
    XBookmarkSourceAdapter.name: SourceAdapterRegistration(
        name=XBookmarkSourceAdapter.name,
        source_type=XBookmarkSourceAdapter.source_type,
        capabilities=XBookmarkSourceAdapter.capabilities,
        description="Authenticate, paginate, and fetch posts from X bookmarks.",
        aliases=("x", "x-bookmarks", "twitter-bookmarks"),
        requires_credentials=True,
    ),
}
ADAPTER_ALIASES = {
    alias: registration.name
    for registration in ADAPTER_REGISTRATIONS.values()
    for alias in registration.aliases
}


def list_source_adapter_registrations() -> list[SourceAdapterRegistration]:
    return [ADAPTER_REGISTRATIONS[name] for name in sorted(ADAPTER_REGISTRATIONS)]


def source_adapter_manifest() -> list[dict[str, Any]]:
    return [registration.as_dict() for registration in list_source_adapter_registrations()]


def create_source_adapter(
    name: str,
    *,
    paths: ContextBankPaths | None = None,
    x_client: XBookmarkClient | None = None,
    x_user_id: str | None = None,
) -> SourceAdapter:
    normalized = _canonical_adapter_name(name)
    if normalized == WebSourceAdapter.name:
        return WebSourceAdapter(paths)
    if normalized == LocalFileSourceAdapter.name:
        return LocalFileSourceAdapter(paths)
    if normalized == ManualTextSourceAdapter.name:
        return ManualTextSourceAdapter(paths)
    if normalized == XBookmarkSourceAdapter.name:
        if x_client is None:
            raise ValueError("X bookmark adapter requires an authenticated XBookmarkClient.")
        return XBookmarkSourceAdapter(x_client, user_id=x_user_id)
    allowed = ", ".join(sorted(ADAPTER_REGISTRATIONS | ADAPTER_ALIASES))
    raise ValueError(f"Unknown source adapter '{name}'. Expected one of: {allowed}.")


def _canonical_adapter_name(name: str) -> str:
    normalized = name.strip().lower()
    return ADAPTER_ALIASES.get(normalized, normalized)


def _ingest_result_item(item_id: str, result: Any) -> SourceAdapterItem:
    return SourceAdapterItem(
        external_id=item_id,
        source_item=result.source_item,
        document=result.document,
        warnings=list(getattr(result, "warnings", []) or []),
    )


def _unsupported_page(connector: str, message: str) -> SourceAdapterPage:
    return SourceAdapterPage(
        errors=[_adapter_error(connector, message, code="unsupported")]
    )


def _error_item(connector: str, message: str, *, item_id: str | None = None) -> SourceAdapterItem:
    return SourceAdapterItem(
        external_id=item_id,
        errors=[_adapter_error(connector, message, item_id=item_id)],
    )


def _adapter_error(
    connector: str,
    message: str,
    *,
    item_id: str | None = None,
    code: str | None = None,
    retryable: bool = False,
) -> SourceAdapterError:
    return SourceAdapterError(
        connector=connector,
        item_id=item_id,
        message=message,
        code=code,
        retryable=retryable,
    )


def _included_author(post: dict[str, Any], includes: dict[str, Any]) -> dict[str, Any]:
    author_id = str(post.get("author_id") or "")
    users = includes.get("users") if isinstance(includes, dict) else None
    if not author_id or not isinstance(users, list):
        return {}
    for user in users:
        if isinstance(user, dict) and str(user.get("id") or "") == author_id:
            return user
    return {}
