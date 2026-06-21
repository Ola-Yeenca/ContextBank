from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any
from urllib.parse import urljoin

from bs4 import BeautifulSoup, Comment

from contextbank.models import ExtractionStatus

EXTRACTOR_VERSION = "builtin-web-0.2"


@dataclass(frozen=True)
class HtmlExtraction:
    title: str | None
    text: str
    status: ExtractionStatus
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


def extract_html(html: str | bytes, *, source_url: str | None = None) -> HtmlExtraction:
    if isinstance(html, bytes):
        html_text = html.decode("utf-8", errors="replace")
    else:
        html_text = html

    if not html_text.strip():
        return HtmlExtraction(
            title=None,
            text="",
            status=ExtractionStatus.FAILED,
            error="HTML payload was empty.",
            metadata={"source_url": source_url, "extractor_version": EXTRACTOR_VERSION},
        )

    soup = BeautifulSoup(html_text, "html.parser")
    _remove_non_content_nodes(soup)

    title = _extract_title(soup)
    metadata = _extract_metadata(soup, source_url=source_url)
    if source_url:
        metadata["source_url"] = source_url
    metadata["extractor_version"] = EXTRACTOR_VERSION
    metadata["extracted_at"] = datetime.now(UTC).isoformat()
    metadata["headings"] = _extract_headings(soup)
    metadata["code_blocks"] = _extract_code_blocks(soup)
    author = _extract_author(soup, metadata)
    published_at = _extract_published_at(soup, metadata)
    if author:
        metadata["author"] = author
    if published_at:
        metadata["published_at"] = published_at

    text = html_to_text(soup)
    if not text:
        return HtmlExtraction(
            title=title,
            text="",
            status=ExtractionStatus.FAILED,
            error="No visible text could be extracted from HTML.",
            metadata=metadata,
        )

    warnings: list[str] = []
    status = ExtractionStatus.COMPLETE
    if soup.find(["main", "article"]) is None:
        status = ExtractionStatus.PARTIAL
        warnings.append("No main/article element found; extracted full visible page text.")

    return HtmlExtraction(
        title=title,
        text=text,
        status=status,
        metadata=metadata,
        warnings=warnings,
    )


def html_to_text(soup_or_html: BeautifulSoup | str | bytes) -> str:
    if isinstance(soup_or_html, BeautifulSoup):
        soup = soup_or_html
    else:
        html_text = (
            soup_or_html.decode("utf-8", errors="replace")
            if isinstance(soup_or_html, bytes)
            else soup_or_html
        )
        soup = BeautifulSoup(html_text, "html.parser")
        _remove_non_content_nodes(soup)

    content_root = soup.find("main") or soup.find("article") or soup.body or soup
    lines = [line.strip() for line in content_root.get_text("\n").splitlines()]
    return "\n".join(line for line in lines if line)


def parse_extracted_datetime(value: Any) -> datetime | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed


def _remove_non_content_nodes(soup: BeautifulSoup) -> None:
    for node in soup.find_all(string=lambda value: isinstance(value, Comment)):
        node.extract()
    for tag in soup(["script", "style", "noscript", "svg", "canvas", "template"]):
        tag.decompose()


def _extract_title(soup: BeautifulSoup) -> str | None:
    for selector in (
        ("meta", {"property": "og:title"}),
        ("meta", {"name": "twitter:title"}),
        ("meta", {"name": "title"}),
    ):
        tag = soup.find(*selector)
        content = tag.get("content", "").strip() if tag else ""
        if content:
            return content

    if soup.title and soup.title.string and soup.title.string.strip():
        return soup.title.string.strip()

    heading = soup.find("h1")
    if heading:
        text = heading.get_text(" ", strip=True)
        if text:
            return text

    return None


def _extract_metadata(soup: BeautifulSoup, *, source_url: str | None = None) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    for tag in soup.find_all("meta"):
        key = tag.get("name") or tag.get("property")
        value = tag.get("content")
        if key and value:
            metadata[key] = value.strip()

    canonical = soup.find("link", {"rel": lambda value: value and "canonical" in value})
    if canonical and canonical.get("href"):
        href = canonical["href"].strip()
        metadata["canonical_url"] = urljoin(source_url, href) if source_url else href

    return metadata


def _extract_headings(soup: BeautifulSoup, *, limit: int = 32) -> list[dict[str, Any]]:
    headings: list[dict[str, Any]] = []
    for tag in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6"]):
        text = tag.get_text(" ", strip=True)
        if not text:
            continue
        headings.append({"level": int(tag.name[1]), "text": text})
        if len(headings) >= limit:
            break
    return headings


def _extract_code_blocks(soup: BeautifulSoup, *, limit: int = 16) -> list[dict[str, str]]:
    blocks: list[dict[str, str]] = []
    for tag in soup.find_all(["pre", "code"]):
        if tag.name == "code" and tag.find_parent("pre") is not None:
            continue
        text = tag.get_text("\n", strip=True)
        if not text:
            continue
        language = _code_language(tag)
        blocks.append({"text": text, "language": language})
        if len(blocks) >= limit:
            break
    return blocks


def _code_language(tag: Any) -> str:
    candidates = [tag]
    nested_code = tag.find("code") if getattr(tag, "name", None) == "pre" else None
    if nested_code is not None:
        candidates.append(nested_code)
    for candidate in candidates:
        classes = candidate.get("class") or []
        for class_name in classes:
            if isinstance(class_name, str) and class_name.startswith("language-"):
                return class_name.removeprefix("language-")
    return ""


def _extract_author(soup: BeautifulSoup, metadata: dict[str, Any]) -> str | None:
    for key in (
        "author",
        "article:author",
        "byl",
        "parsely-author",
        "twitter:creator",
    ):
        value = str(metadata.get(key) or "").strip()
        if value:
            return value
    rel_author = soup.find("a", {"rel": lambda value: value and "author" in value})
    if rel_author:
        text = rel_author.get_text(" ", strip=True)
        if text:
            return text
    return None


def _extract_published_at(soup: BeautifulSoup, metadata: dict[str, Any]) -> str | None:
    for key in (
        "article:published_time",
        "datePublished",
        "date",
        "publish_date",
        "pubdate",
        "dc.date",
        "dc.date.issued",
    ):
        value = str(metadata.get(key) or "").strip()
        if value:
            return value
    time_tag = soup.find("time")
    if time_tag:
        value = str(time_tag.get("datetime") or time_tag.get_text(" ", strip=True)).strip()
        if value:
            return value
    return None
