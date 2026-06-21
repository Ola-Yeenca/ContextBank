from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from contextbank.models import Document, SourceItem


@dataclass(frozen=True)
class SourceAdapterCapabilities:
    authenticate: bool = False
    list_items: bool = False
    fetch_one: bool = False
    continuation_state: bool = False
    normalized_documents: bool = False


@dataclass(frozen=True)
class SourceAdapterError:
    connector: str
    message: str
    item_id: str | None = None
    code: str | None = None
    retryable: bool = False


@dataclass(frozen=True)
class SourceAdapterItem:
    external_id: str | None = None
    source_item: SourceItem | None = None
    document: Document | None = None
    raw: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    errors: list[SourceAdapterError] = field(default_factory=list)


@dataclass(frozen=True)
class SourceAdapterPage:
    items: list[SourceAdapterItem] = field(default_factory=list)
    continuation_state: str | None = None
    warnings: list[str] = field(default_factory=list)
    errors: list[SourceAdapterError] = field(default_factory=list)
    raw: dict[str, Any] | None = None


@runtime_checkable
class SourceAdapter(Protocol):
    name: str
    source_type: str
    capabilities: SourceAdapterCapabilities

    def authenticate(self) -> SourceAdapterPage:
        """Validate connector credentials or return a structured unsupported/error page."""

    def list_new_or_changed(
        self,
        *,
        limit: int | None = None,
        continuation_state: str | None = None,
    ) -> SourceAdapterPage:
        """Return a page of raw or normalized items plus continuation state."""

    def fetch_one(self, item_id: str, **kwargs: Any) -> SourceAdapterItem:
        """Fetch one source item by connector-specific id or location."""

    def normalize_metadata(self, raw: dict[str, Any]) -> dict[str, Any]:
        """Return normalized metadata usable by the ingestion pipeline."""
