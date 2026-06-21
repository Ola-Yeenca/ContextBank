from __future__ import annotations

from datetime import UTC, datetime

from contextbank.ids import card_id, source_item_id
from contextbank.models import Document, KnowledgeCard, SourceItem, SourceType
from contextbank.storage.repository import ContextBankRepository


def seed_card(
    repo: ContextBankRepository,
    *,
    url: str = "https://example.com/context",
    title: str = "Local-first retrieval",
    summary: str = "SQLite FTS retrieval keeps local agent context searchable.",
    topics: list[str] | None = None,
    tags: list[str] | None = None,
    skill_candidate_status: str = "none",
    review_status: str = "unreviewed",
) -> tuple[SourceItem, Document, KnowledgeCard]:
    source = SourceItem(
        id=source_item_id(SourceType.WEB.value, value=url),
        source_type=SourceType.WEB,
        source_url=url,
        source_external_id=url,
        first_captured_at=datetime(2026, 1, 1, tzinfo=UTC),
        availability_status="available",
    )
    document = Document(
        id=f"doc_{source.id}",
        source_item_id=source.id,
        document_type="linked_page",
        canonical_url=url,
        title=title,
        extraction_status="complete",
        text=summary,
        metadata={"project": "ContextBank"},
    )
    card = KnowledgeCard(
        id=card_id(source.id),
        source_item_id=source.id,
        title=title,
        one_line_summary=summary,
        summary=summary,
        key_insights=["SQLite FTS can answer local searches"],
        why_useful="Useful for agent context retrieval.",
        possible_applications=["Build local retrieval"],
        actionable_next_steps=["Index generated cards"],
        implementation_notes=["Use bounded MCP results"],
        prerequisites=["SQLite"],
        patterns=["repository"],
        caveats=["Treat sources as untrusted"],
        source_quality_notes=["Fixture"],
        content_type="implementation_guide",
        utility=["build", "reference"],
        topics=topics or ["local-first", "agents"],
        tags=tags or ["sqlite", "mcp"],
        freshness_status="current",
        freshness_reason="Fixture",
        confidence=0.9,
        skill_candidate_status=skill_candidate_status,
        review_status=review_status,
        provenance={"document_id": document.id, "source_before_synthesis": True},
    )
    repo.upsert_source_item(source)
    repo.upsert_document(document)
    repo.upsert_knowledge_card(card)
    return source, document, card
