from __future__ import annotations

import json
import mimetypes
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import AnyUrl

from contextbank.extraction.web import EXTRACTOR_VERSION, extract_html, parse_extracted_datetime
from contextbank.ids import content_hash, document_id, slugify, source_item_id
from contextbank.models import (
    AvailabilityStatus,
    Document,
    ExtractionStatus,
    SourceItem,
    SourceType,
)
from contextbank.paths import ContextBankPaths, secure_write_bytes, secure_write_text

SUPPORTED_FILE_EXTENSIONS = {
    ".md": "markdown",
    ".markdown": "markdown",
    ".txt": "text",
    ".text": "text",
    ".json": "json",
    ".htm": "html",
    ".html": "html",
}


@dataclass(frozen=True)
class FileImportResult:
    source_item: SourceItem
    document: Document
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ExtractedFileText:
    text: str
    status: ExtractionStatus
    error: str | None = None
    title: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


def import_file(file_path: str | Path, paths: ContextBankPaths | None = None) -> FileImportResult:
    source_path = Path(file_path).expanduser()
    if not source_path.exists():
        raise FileNotFoundError(source_path)
    if not source_path.is_file():
        raise ValueError(f"Expected a file path, got: {source_path}")

    extension = source_path.suffix.lower()
    file_kind = SUPPORTED_FILE_EXTENSIONS.get(extension)
    if file_kind is None:
        supported = ", ".join(sorted(SUPPORTED_FILE_EXTENSIONS))
        raise ValueError(f"Unsupported file type '{extension}'. Supported extensions: {supported}")

    paths = paths or ContextBankPaths.from_home()
    paths.ensure()

    raw_bytes = source_path.read_bytes()
    raw_digest = content_hash(raw_bytes)
    resolved_source = source_path.resolve()
    item_id = source_item_id(SourceType.FILE.value, value=str(resolved_source))
    doc_id = document_id(item_id, "file", raw_digest)

    raw_path = _raw_file_path(paths, source_path.name, item_id, raw_digest)
    _write_bytes(raw_path, raw_bytes)

    extraction = extract_file_text(raw_bytes, file_kind=file_kind, filename=source_path.name)
    extracted_text_path: Path | None = None
    if extraction.text:
        extracted_text_path = _document_text_path(paths, doc_id)
        _write_text(extracted_text_path, extraction.text)

    source_item = SourceItem(
        id=item_id,
        source_type=SourceType.FILE,
        source_external_id=str(resolved_source),
        source_url=_path_uri(resolved_source),
        raw_path=str(raw_path),
        content_hash=raw_digest,
        availability_status=AvailabilityStatus.AVAILABLE,
    )
    document = Document(
        id=doc_id,
        source_item_id=item_id,
        document_type="file",
        canonical_url=_path_uri(resolved_source),
        title=extraction.title or source_path.name,
        author=extraction.metadata.get("author"),
        published_at=parse_extracted_datetime(extraction.metadata.get("published_at")),
        extracted_text_path=str(extracted_text_path) if extracted_text_path else None,
        html_snapshot_path=str(raw_path) if file_kind == "html" else None,
        extraction_status=extraction.status,
        extraction_error=extraction.error,
        extractor_version=EXTRACTOR_VERSION if file_kind == "html" else "builtin-file-0.1",
        content_hash=content_hash(extraction.text) if extraction.text else None,
        text=extraction.text or None,
        metadata={
            "filename": source_path.name,
            "file_kind": file_kind,
            "mime_type": mimetypes.guess_type(source_path.name)[0],
            "byte_size": len(raw_bytes),
            "source_mtime": source_path.stat().st_mtime,
            **extraction.metadata,
        },
    )
    return FileImportResult(
        source_item=source_item,
        document=document,
        warnings=extraction.warnings,
    )


def extract_file_text(
    raw_bytes: bytes,
    *,
    file_kind: str,
    filename: str | None = None,
) -> ExtractedFileText:
    if file_kind in {"markdown", "text"}:
        text = raw_bytes.decode("utf-8", errors="replace")
        status = ExtractionStatus.COMPLETE if text.strip() else ExtractionStatus.FAILED
        error = None if text.strip() else "File contained no text."
        return ExtractedFileText(text=text, status=status, error=error)

    if file_kind == "json":
        decoded = raw_bytes.decode("utf-8", errors="replace")
        try:
            parsed = json.loads(decoded)
        except json.JSONDecodeError as exc:
            return ExtractedFileText(
                text=decoded,
                status=ExtractionStatus.PARTIAL,
                error=f"Invalid JSON preserved as raw text: {exc.msg}",
                metadata={"json_valid": False},
                warnings=["JSON parsing failed; stored decoded raw text instead."],
            )

        return ExtractedFileText(
            text=json.dumps(parsed, ensure_ascii=False, indent=2, sort_keys=True),
            status=ExtractionStatus.COMPLETE,
            metadata={"json_valid": True, "json_top_level_type": type(parsed).__name__},
        )

    if file_kind == "html":
        extraction = extract_html(raw_bytes, source_url=filename)
        return ExtractedFileText(
            text=extraction.text,
            status=extraction.status,
            error=extraction.error,
            title=extraction.title,
            metadata=extraction.metadata,
            warnings=extraction.warnings,
        )

    raise ValueError(f"Unsupported file kind: {file_kind}")


def _raw_file_path(paths: ContextBankPaths, filename: str, item_id: str, digest: str) -> Path:
    original = Path(filename)
    slug = slugify(original.stem)
    suffix = original.suffix.lower() or ".bin"
    return _safe_child(paths.raw / "files", f"{item_id}-{digest[:12]}-{slug}{suffix}")


def _document_text_path(paths: ContextBankPaths, doc_id: str) -> Path:
    return _safe_child(paths.documents, f"{doc_id}.txt")


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


def _path_uri(path: Path) -> str | None:
    try:
        return str(AnyUrl(path.as_uri()))
    except ValueError:
        return None
