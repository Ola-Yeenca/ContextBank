from __future__ import annotations

from datetime import UTC, datetime

from contextbank.config.settings import initialize_settings
from contextbank.ids import card_id, content_hash, source_item_id
from contextbank.models import Document, KnowledgeCard, SearchFilters, SourceItem, SourceType
from contextbank.storage.db import (
    SCHEMA_VERSION,
    current_schema_version,
    initialize_database,
    migrate,
)
from contextbank.storage.repository import ContextBankRepository


def test_initialize_settings_and_migrate_creates_local_store(tmp_path):
    home = tmp_path / "bank"

    settings = initialize_settings(home)
    connection = initialize_database(settings.database_path)

    assert settings.database_path == home / "contextbank.db"
    assert settings.paths.raw.exists()
    assert settings.paths.documents.exists()
    assert settings.paths.cards.exists()
    assert settings.paths.config_file.exists()
    assert (settings.paths.home.stat().st_mode & 0o777) == 0o700
    assert (settings.paths.config_file.stat().st_mode & 0o777) == 0o600
    assert current_schema_version(connection) == SCHEMA_VERSION
    assert (settings.database_path.stat().st_mode & 0o777) == 0o600
    assert connection.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'source_items'"
    ).fetchone()


def test_migration_ledger_insert_tolerates_concurrent_idempotent_migration(tmp_path):
    database = tmp_path / "race.db"
    connection = initialize_database(database)

    migrate(
        connection,
        migrations=(
            (
                SCHEMA_VERSION + 1,
                f"""
                CREATE TABLE IF NOT EXISTS migration_race_marker (
                    id INTEGER PRIMARY KEY
                );
                INSERT OR IGNORE INTO schema_migrations(version, applied_at)
                VALUES ({SCHEMA_VERSION + 1}, 'external-migrator');
                """,
            ),
        ),
    )

    assert current_schema_version(connection) == SCHEMA_VERSION + 1


def test_source_upsert_is_idempotent(tmp_path):
    repo = ContextBankRepository.from_home(tmp_path)
    source = _source_item("https://example.com/a")

    assert repo.upsert_source_item(source) is True
    assert repo.upsert_source_item(source) is False

    fetched = repo.get_source_item(source.id)
    assert fetched == source
    assert repo.count_rows("source_items") == 1


def test_card_and_document_upserts_persist_json_and_links(tmp_path):
    repo = ContextBankRepository.from_home(tmp_path)
    source = _source_item("https://example.com/context")
    repo.upsert_source_item(source)
    document = Document(
        id="doc_example",
        source_item_id=source.id,
        document_type="source",
        title="Context Storage",
        text="SQLite remains canonical while markdown mirrors stay regenerable.",
        metadata={"depth": 2, "flags": ["local", "typed"]},
    )
    card = _card(source.id)

    assert repo.upsert_document(document) is True
    assert repo.upsert_document(document) is False
    assert repo.upsert_knowledge_card(card) is True
    assert repo.upsert_knowledge_card(card) is False

    fetched_document = repo.get_document(document.id)
    fetched_card = repo.get_knowledge_card(card.id)
    assert fetched_document == document
    assert fetched_card == card
    assert repo.count_rows("card_topics") == 2
    assert repo.count_rows("card_tags") == 2

    updated = card.model_copy(update={"topics": ["SQLite"], "tags": ["storage"]})
    repo.upsert_knowledge_card(updated)

    assert repo.get_knowledge_card(card.id) == updated
    assert repo.count_rows("card_topics") == 1
    assert repo.count_rows("card_tags") == 1


def test_repository_finds_duplicate_cards_by_hash_and_canonical_url(tmp_path):
    repo = ContextBankRepository.from_home(tmp_path)
    text = "Same saved source body."
    digest = content_hash(text)
    first = _source_item("https://example.com/first").model_copy(
        update={"content_hash": digest}
    )
    second = _source_item("https://example.com/second").model_copy(
        update={"content_hash": digest}
    )
    first_document = Document(
        id="doc_first",
        source_item_id=first.id,
        document_type="linked_page",
        canonical_url="https://example.com/canonical",
        extraction_status="complete",
        content_hash=digest,
        text=text,
    )
    second_document = Document(
        id="doc_second",
        source_item_id=second.id,
        document_type="linked_page",
        canonical_url="https://example.com/canonical",
        extraction_status="complete",
        content_hash=digest,
        text=text,
    )
    repo.upsert_source_item(first)
    repo.upsert_document(first_document)
    repo.upsert_knowledge_card(_card(first.id))
    repo.upsert_source_item(second)
    repo.upsert_document(second_document)
    repo.upsert_knowledge_card(_card(second.id))

    duplicates = repo.find_duplicate_cards(source_item=second, document=second_document)

    assert duplicates[0]["card"].source_item_id == first.id
    assert set(duplicates[0]["match_reasons"]) == {
        "source-content-hash",
        "document-content-hash",
        "canonical-url",
    }

    relationship = repo.upsert_duplicate_relationship(
        candidate_card_id=card_id(second.id),
        match_card_id=card_id(first.id),
        status="duplicate",
        reason="Same canonical URL and body.",
    )

    assert relationship["status"] == "duplicate"
    assert relationship["reason"] == "Same canonical URL and body."
    assert repo.count_rows("duplicate_relationships") == 1

    updated = repo.upsert_duplicate_relationship(
        candidate_card_id=card_id(second.id),
        match_card_id=card_id(first.id),
        status="not-duplicate",
        reason="Different usage despite same source.",
    )

    assert updated["status"] == "not-duplicate"
    assert updated["created_at"] == relationship["created_at"]
    assert repo.list_duplicate_relationships(card_id=card_id(second.id))[0]["status"] == (
        "not-duplicate"
    )
    assert repo.count_rows("duplicate_relationships") == 1


def test_card_upsert_maintains_single_fts_row_and_searches_it(tmp_path):
    repo = ContextBankRepository.from_home(tmp_path)
    source = _source_item("https://example.com/search").model_copy(
        update={
            "author_name": "Source Author",
            "author_handle": "source_author_handle",
        }
    )
    repo.upsert_source_item(source)
    repo.upsert_document(
        Document(
            id="doc_search",
            source_item_id=source.id,
            document_type="linked_page",
            text="Document-only archive phrase for full text search.",
            author="Document Writer",
        )
    )
    card = _card(source.id)

    repo.upsert_knowledge_card(card)
    repo.upsert_knowledge_card(card.model_copy(update={"summary": "Durable FTS retrieval index."}))
    repo.upsert_document(
        Document(
            id="doc_search",
            source_item_id=source.id,
            document_type="linked_page",
            text="Updated document-only refresh phrase for full text search.",
            author="Document Writer",
        )
    )

    assert repo.count_fts_rows(card.id) == 1
    results = repo.search_cards("retrieval")
    assert [result.card.id for result in results] == [card.id]
    assert [result.card.id for result in repo.search_cards("refresh phrase")] == [card.id]
    assert [result.card.id for result in repo.search_cards("source_author_handle")] == [
        card.id
    ]
    assert results[0].source_url == source.source_url


def test_sync_state_persists_resume_cursor(tmp_path):
    repo = ContextBankRepository.from_home(tmp_path)

    repo.upsert_sync_state(
        connector="x.bookmarks",
        account_id="123",
        cursor="next-page",
        status="running",
        last_started_at="2026-06-20T12:00:00+00:00",
        items_synced=10,
    )

    state = repo.get_sync_state("x.bookmarks", "123")
    assert state["cursor"] == "next-page"
    assert state["status"] == "running"
    assert state["items_synced"] == 10
    assert state["full_sync_completed"] is False
    assert repo.count_rows("sync_state") == 1

    repo.upsert_sync_state(
        connector="x.bookmarks",
        account_id="123",
        cursor=None,
        status="complete",
        last_completed_at="2026-06-20T12:05:00+00:00",
        items_synced=12,
        full_sync_completed=True,
    )

    completed = repo.get_sync_state("x.bookmarks", "123")
    assert completed["cursor"] is None
    assert completed["status"] == "complete"
    assert completed["full_sync_completed"] is True
    assert repo.delete_sync_state("x.bookmarks", "123") is True
    assert repo.get_sync_state("x.bookmarks", "123") is None


def test_project_links_persist_and_project_filter_uses_link_table(tmp_path):
    repo = ContextBankRepository.from_home(tmp_path)
    source = _source_item("https://example.com/project")
    card = _card(source.id)
    repo.upsert_source_item(source)
    repo.upsert_knowledge_card(card)

    project_root = tmp_path / "project"
    project_root.mkdir()
    link = repo.link_card_to_project(
        card_id=card.id,
        root_path=project_root,
        name="ContextBank",
        profile={"name": "ContextBank", "stack": ["Python"]},
        relevance=0.85,
        reason="Use this storage pattern in project-aware retrieval.",
        intended_use="implementation",
        status="used",
        outcome_note="Applied in project linking workflow.",
        created_by="test",
    )

    assert link["project_name"] == "ContextBank"
    assert link["status"] == "used"
    assert repo.count_rows("projects") == 1
    assert repo.count_rows("project_item_links") == 1
    assert repo.list_project_links_for_card(card.id)[0]["reason"].startswith("Use this")

    filtered = repo.list_knowledge_cards(filters=SearchFilters(project="ContextBank"))
    assert [item.id for item in filtered] == [card.id]


def test_review_queue_surfaces_library_health_states(tmp_path):
    repo = ContextBankRepository.from_home(tmp_path)

    unprocessed = _source_item("https://example.com/unprocessed")
    repo.upsert_source_item(unprocessed)

    partial_source = _source_item("https://example.com/partial").model_copy(
        update={"availability_status": "partial"}
    )
    repo.upsert_source_item(partial_source)
    repo.upsert_document(
        Document(
            id="doc_partial",
            source_item_id=partial_source.id,
            document_type="linked_page",
            title="Partial",
            extraction_status="partial",
            text="Partial text",
        )
    )

    failed_source = _source_item("https://example.com/failed").model_copy(
        update={"availability_status": "unavailable"}
    )
    repo.upsert_source_item(failed_source)
    repo.upsert_document(
        Document(
            id="doc_failed",
            source_item_id=failed_source.id,
            document_type="linked_page",
            title="Failed",
            extraction_status="failed",
            extraction_error="network",
        )
    )

    reviewed_source = _source_item("https://example.com/review")
    repo.upsert_source_item(reviewed_source)
    low_card = _card(reviewed_source.id).model_copy(
        update={
            "confidence": 0.2,
            "freshness_status": "stale-risk",
            "review_status": "hidden",
            "skill_candidate_status": "candidate",
        }
    )
    repo.upsert_knowledge_card(low_card)

    all_items = repo.list_review_queue(queue="all", low_confidence_threshold=0.5)
    reasons_by_source = {item["source_item_id"]: item["reasons"] for item in all_items}

    assert "unprocessed" in reasons_by_source[unprocessed.id]
    assert "partial" in reasons_by_source[partial_source.id]
    assert "failed" in reasons_by_source[failed_source.id]
    assert {"low-confidence", "stale", "hidden", "skill-review"}.issubset(
        set(reasons_by_source[reviewed_source.id])
    )

    skill_items = repo.list_review_queue(queue="skill")
    assert [item["source_item_id"] for item in skill_items] == [reviewed_source.id]


def test_mark_missing_source_items_deleted_is_soft_and_type_scoped(tmp_path):
    repo = ContextBankRepository.from_home(tmp_path)
    kept = SourceItem(
        id=source_item_id(SourceType.X.value, external_id="kept"),
        source_type=SourceType.X,
        source_external_id="kept",
        availability_status="available",
    )
    missing = SourceItem(
        id=source_item_id(SourceType.X.value, external_id="missing"),
        source_type=SourceType.X,
        source_external_id="missing",
        availability_status="available",
    )
    web = _source_item("https://example.com/not-x")
    repo.upsert_source_item(kept)
    repo.upsert_source_item(missing)
    repo.upsert_source_item(web)

    deleted = repo.mark_missing_source_items_deleted(
        source_type=SourceType.X.value,
        active_source_item_ids={kept.id},
        deleted_at="2026-06-20T12:00:00+00:00",
    )

    assert deleted == 1
    assert repo.get_source_item(kept.id).deleted_locally_at is None
    deleted_source = repo.get_source_item(missing.id)
    assert deleted_source.deleted_locally_at.isoformat() == "2026-06-20T12:00:00+00:00"
    assert deleted_source.availability_status == "unavailable"
    assert repo.get_source_item(web.id).deleted_locally_at is None


def test_source_lifecycle_marks_raw_deletes_and_detaches_notes_explicitly(tmp_path):
    repo = ContextBankRepository.from_home(tmp_path)
    source = _source_item("https://example.com/lifecycle").model_copy(
        update={"raw_path": str(tmp_path / "raw" / "source.json")}
    )
    document = Document(
        id="doc_lifecycle",
        source_item_id=source.id,
        document_type="linked_page",
        title="Lifecycle",
        extracted_text_path=str(tmp_path / "documents" / "doc_lifecycle.txt"),
        html_snapshot_path=str(tmp_path / "raw" / "snapshot.html"),
        extraction_status="complete",
        text="Lifecycle text",
        metadata={"raw_retained": True},
    )
    card = _card(source.id).model_copy(
        update={
            "id": card_id(source.id),
            "provenance": {
                "user_signals": {"usefulness_rating": 5},
                "correction_history": [{"fields": {"summary": {"corrected": "Human"}}}],
                "project_links": [{"project_name": "Demo", "reason": "Useful"}],
            },
        }
    )
    repo.upsert_source_item(source)
    repo.upsert_document(document)
    repo.upsert_knowledge_card(card)
    repo.upsert_card_embedding(
        card_id=card.id,
        provider="local-hash",
        model_id="hashing-64-v1",
        vector=[1.0, 0.0],
        source_hash="abc",
    )

    unavailable = repo.mark_source_unavailable(source.id, reason="Gone upstream")
    assert unavailable.availability_status == "unavailable"
    assert repo.list_source_lifecycle_events(source.id)[0]["event_type"] == "marked-unavailable"

    revalidation = repo.request_source_revalidation(source.id, reason="Check freshness")
    assert revalidation["updated_card_ids"] == [card.id]
    assert repo.get_source_item(source.id).availability_status == "unknown"
    rechecked = repo.get_knowledge_card(card.id)
    assert rechecked.freshness_status == "recheck-soon"
    assert rechecked.provenance["source_lifecycle_events"][0]["reason"] == "Check freshness"

    raw = repo.clear_source_raw_references(source.id, reason="Minimize retention")
    assert set(raw["paths"]) == {source.raw_path, document.html_snapshot_path}
    assert repo.get_source_item(source.id).raw_path is None
    cleared_document = repo.get_document(document.id)
    assert cleared_document.html_snapshot_path is None
    assert cleared_document.extracted_text_path == document.extracted_text_path
    assert cleared_document.metadata["raw_retained"] is False

    deleted = repo.delete_complete_source_item(source.id)
    assert deleted["deleted"] is True
    assert deleted["detached_note_id"] is None
    assert repo.get_source_item(source.id) is None
    assert repo.get_document(document.id) is None
    assert repo.get_knowledge_card(card.id) is None
    assert repo.count_rows("card_embeddings") == 0
    assert repo.count_rows("detached_notes") == 0


def test_source_delete_preserves_user_notes_only_with_detach_flag(tmp_path):
    repo = ContextBankRepository.from_home(tmp_path)
    source = _source_item("https://example.com/detach")
    document = Document(
        id="doc_detach",
        source_item_id=source.id,
        document_type="linked_page",
        title="Detach",
        extraction_status="complete",
        text="Detach text",
    )
    card = _card(source.id).model_copy(
        update={
            "id": card_id(source.id),
            "review_status": "approved",
            "provenance": {
                "user_signals": {"personal_priority": "high"},
                "correction_history": [{"fields": {"title": {"corrected": "Human title"}}}],
                "project_links": [{"project_name": "ContextBank", "status": "used"}],
            },
        }
    )
    repo.upsert_source_item(source)
    repo.upsert_document(document)
    repo.upsert_knowledge_card(card)

    deleted = repo.delete_complete_source_item(
        source.id,
        detach_user_notes=True,
        reason="User requested source deletion.",
    )

    assert deleted["deleted"] is True
    assert deleted["detached_note_id"] is not None
    detached = repo.get_detached_note(deleted["detached_note_id"])
    assert detached["original_source_item_id"] == source.id
    assert detached["user_notes"]["cards"][0]["user_signals"]["personal_priority"] == "high"
    assert detached["user_notes"]["cards"][0]["project_links"][0]["project_name"] == "ContextBank"
    assert repo.count_rows("detached_notes") == 1


def _source_item(url: str) -> SourceItem:
    return SourceItem(
        id=source_item_id(SourceType.WEB.value, value=url),
        source_type=SourceType.WEB,
        source_url=url,
        source_external_id=url,
        first_captured_at=datetime(2026, 1, 1, tzinfo=UTC),
        availability_status="available",
    )


def _card(source_id: str) -> KnowledgeCard:
    return KnowledgeCard(
        id=card_id(source_id),
        source_item_id=source_id,
        title="Local-first SQLite retrieval",
        one_line_summary="Canonical SQLite records keep agent context inspectable.",
        summary="A robust repository stores cards, documents, and sources locally.",
        key_insights=["FTS5 can index regenerated summaries", "Typed models guard boundaries"],
        why_useful="Agents can read bounded context without gaining write access by default.",
        possible_applications=["CLI import", "MCP retrieval"],
        actionable_next_steps=["Run migrations before writes"],
        implementation_notes=["Topic links are normalized"],
        prerequisites=["SQLite with FTS5"],
        patterns=["repository"],
        caveats=["Markdown mirrors should be treated as regenerable"],
        source_quality_notes=["Synthetic test fixture"],
        content_type="engineering",
        utility=["retrieval", "storage"],
        topics=["SQLite", "Local-first"],
        tags=["storage", "fts"],
        personal_priority="high",
        freshness_status="current",
        freshness_reason="Local architecture decision",
        confidence=0.91,
        provenance={"document_id": "doc_example", "project": "ContextBank"},
    )
