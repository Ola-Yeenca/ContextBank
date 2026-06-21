from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from contextbank.ids import content_hash, document_id, slugify, source_item_id
from contextbank.models import (
    AvailabilityStatus,
    Document,
    ExtractionStatus,
    SourceItem,
    SourceType,
)
from contextbank.paths import ContextBankPaths, secure_write_text

TEXT_IMPORTER_VERSION = "builtin-text-0.1"


@dataclass(frozen=True)
class TextImportResult:
    source_item: SourceItem
    document: Document
    warnings: list[str] = field(default_factory=list)


def import_text(
    text: str,
    paths: ContextBankPaths | None = None,
    *,
    source_url: str | None = None,
    title: str | None = None,
) -> TextImportResult:
    normalized_text = text.strip()
    if not normalized_text:
        raise ValueError("Manual text import requires non-empty text.")

    paths = paths or ContextBankPaths.from_home()
    paths.ensure()

    digest = content_hash(normalized_text)
    source_identity = digest
    item_id = source_item_id(SourceType.TEXT.value, value=source_identity)
    doc_id = document_id(item_id, "manual_text", digest)

    raw_path = _raw_text_path(paths, title or source_identity, item_id, digest)
    extracted_text_path = _document_text_path(paths, doc_id)
    _write_text(raw_path, normalized_text)
    _write_text(extracted_text_path, normalized_text)

    source_item = SourceItem(
        id=item_id,
        source_type=SourceType.TEXT,
        source_external_id=digest,
        source_url=source_url,
        raw_path=str(raw_path),
        content_hash=digest,
        availability_status=AvailabilityStatus.AVAILABLE,
    )
    document = Document(
        id=doc_id,
        source_item_id=item_id,
        document_type="manual_text",
        canonical_url=source_url,
        title=title,
        extracted_text_path=str(extracted_text_path),
        extraction_status=ExtractionStatus.COMPLETE,
        extractor_version=TEXT_IMPORTER_VERSION,
        content_hash=digest,
        text=normalized_text,
        metadata={
            "byte_size": len(normalized_text.encode("utf-8")),
            "source_kind": "manual_text",
        },
    )
    return TextImportResult(source_item=source_item, document=document)


def _raw_text_path(paths: ContextBankPaths, label: str, item_id: str, digest: str) -> Path:
    slug = slugify(label)[:80] or "manual-text"
    return _safe_child(paths.raw / "text", f"{item_id}-{digest[:12]}-{slug}.txt")


def _document_text_path(paths: ContextBankPaths, doc_id: str) -> Path:
    return _safe_child(paths.documents, f"{doc_id}.txt")


def _write_text(path: Path, text: str) -> None:
    secure_write_text(path, text)


def _safe_child(base: Path, filename: str) -> Path:
    resolved_base = base.resolve()
    candidate = (resolved_base / filename).resolve()
    if not candidate.is_relative_to(resolved_base):
        raise ValueError(f"Refusing to write outside ContextBank path: {candidate}")
    return candidate
