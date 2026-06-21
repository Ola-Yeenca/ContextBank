from __future__ import annotations

import hashlib
import re
from urllib.parse import urlsplit, urlunsplit


def stable_hash(value: str, length: int = 16) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:length]


def content_hash(value: str | bytes) -> str:
    if isinstance(value, str):
        value = value.encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def normalize_url(url: str) -> str:
    parsed = urlsplit(url.strip())
    scheme = parsed.scheme.lower() or "https"
    netloc = parsed.netloc.lower()
    path = re.sub(r"/+$", "", parsed.path) or "/"
    return urlunsplit((scheme, netloc, path, parsed.query, ""))


def source_item_id(
    source_type: str,
    external_id: str | None = None,
    value: str | None = None,
) -> str:
    basis = external_id or value or source_type
    return f"cb_{source_type}_{stable_hash(basis)}"


def document_id(source_item_id_value: str, document_type: str, value: str) -> str:
    return f"doc_{stable_hash(source_item_id_value + ':' + document_type + ':' + value)}"


def card_id(source_item_id_value: str) -> str:
    return f"card_{stable_hash(source_item_id_value)}"


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug or "uncategorized"
