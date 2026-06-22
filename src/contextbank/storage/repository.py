from __future__ import annotations

import json
import re
import sqlite3
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Self

from contextbank.config.settings import initialize_settings
from contextbank.ids import slugify, stable_hash
from contextbank.models import (
    Document,
    KnowledgeCard,
    SearchFilters,
    SearchResult,
    SourceItem,
    utc_now,
)
from contextbank.storage.db import connect_database, migrate

SOURCE_FIELDS = (
    "id",
    "source_type",
    "source_external_id",
    "source_url",
    "author_id",
    "author_name",
    "author_handle",
    "source_created_at",
    "first_captured_at",
    "last_synced_at",
    "raw_path",
    "content_hash",
    "language",
    "availability_status",
    "deleted_locally_at",
)

DOCUMENT_FIELDS = (
    "id",
    "source_item_id",
    "document_type",
    "canonical_url",
    "title",
    "author",
    "published_at",
    "extracted_text_path",
    "html_snapshot_path",
    "extraction_status",
    "extraction_error",
    "extractor_version",
    "content_hash",
    "text",
)

CARD_SCALAR_FIELDS = (
    "id",
    "source_item_id",
    "schema_version",
    "title",
    "one_line_summary",
    "summary",
    "why_useful",
    "content_type",
    "personal_priority",
    "freshness_status",
    "freshness_reason",
    "confidence",
    "skill_candidate_status",
    "review_status",
    "generated_at",
    "model_provider",
    "model_id",
    "prompt_version",
)

CARD_JSON_FIELDS = (
    "key_insights",
    "possible_applications",
    "actionable_next_steps",
    "implementation_notes",
    "prerequisites",
    "patterns",
    "caveats",
    "source_quality_notes",
    "utility",
)

SYNC_STATE_FIELDS = (
    "connector",
    "account_id",
    "cursor",
    "status",
    "last_started_at",
    "last_completed_at",
    "last_error",
    "items_synced",
    "full_sync_completed",
    "updated_at",
)

PROJECT_FIELDS = (
    "id",
    "name",
    "root_path",
    "profile_json",
    "created_at",
    "last_seen_at",
)

PROJECT_LINK_FIELDS = (
    "project_id",
    "knowledge_card_id",
    "relevance",
    "reason",
    "intended_use",
    "status",
    "outcome_note",
    "created_by",
    "created_at",
    "updated_at",
)

COUNTABLE_TABLES = frozenset(
    {
        "source_items",
        "documents",
        "knowledge_cards",
        "topics",
        "tags",
        "card_topics",
        "card_tags",
        "card_fts",
        "sync_state",
        "projects",
        "project_item_links",
        "card_embeddings",
        "source_lifecycle_events",
        "detached_notes",
        "duplicate_relationships",
    }
)


class ContextBankRepository:
    def __init__(self, connection: sqlite3.Connection):
        self.connection = connection
        self.connection.row_factory = sqlite3.Row

    @classmethod
    def open(cls, database: Path | str, *, run_migrations: bool = True) -> Self:
        connection = connect_database(database)
        if run_migrations:
            migrate(connection)
        return cls(connection)

    @classmethod
    def from_home(cls, home: Path | str | None = None) -> Self:
        settings = initialize_settings(home)
        return cls.open(settings.database_path)

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()

    def upsert_source_item(self, source_item: SourceItem) -> bool:
        created = self.get_source_item(source_item.id) is None
        data = source_item.model_dump(mode="json")
        now = utc_now().isoformat()
        params = {field: data[field] for field in SOURCE_FIELDS}
        params["created_at"] = now
        params["updated_at"] = now

        columns = (*SOURCE_FIELDS, "created_at", "updated_at")
        placeholders = ", ".join(f":{column}" for column in columns)
        update_columns = [
            field for field in SOURCE_FIELDS if field not in {"id", "first_captured_at"}
        ]
        update_set = ", ".join(f"{field} = excluded.{field}" for field in update_columns)
        sql = f"""
            INSERT INTO source_items ({", ".join(columns)})
            VALUES ({placeholders})
            ON CONFLICT(id) DO UPDATE SET
                {update_set},
                first_captured_at = source_items.first_captured_at,
                updated_at = excluded.updated_at
        """
        with self.connection:
            self.connection.execute(sql, params)
            self._refresh_source_card_fts(source_item.id)
        return created

    def get_source_item(self, source_item_id: str) -> SourceItem | None:
        row = self.connection.execute(
            "SELECT * FROM source_items WHERE id = ?",
            (source_item_id,),
        ).fetchone()
        if row is None:
            return None
        return SourceItem.model_validate({field: row[field] for field in SOURCE_FIELDS})

    def get_source_item_by_external_id(
        self,
        source_type: str,
        source_external_id: str,
    ) -> SourceItem | None:
        row = self.connection.execute(
            """
            SELECT * FROM source_items
            WHERE source_type = ? AND source_external_id = ?
            """,
            (source_type, source_external_id),
        ).fetchone()
        if row is None:
            return None
        return SourceItem.model_validate({field: row[field] for field in SOURCE_FIELDS})

    def list_source_items(self, *, limit: int = 100, offset: int = 0) -> list[SourceItem]:
        rows = self.connection.execute(
            """
            SELECT * FROM source_items
            ORDER BY first_captured_at DESC, id
            LIMIT ? OFFSET ?
            """,
            (limit, offset),
        ).fetchall()
        return [
            SourceItem.model_validate({field: row[field] for field in SOURCE_FIELDS})
            for row in rows
        ]

    def list_source_items_filtered(
        self,
        *,
        source_type: str | None = None,
        availability_status: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[SourceItem]:
        where: list[str] = []
        params: list[str | int] = []
        if source_type:
            where.append("source_type = ?")
            params.append(source_type)
        if availability_status:
            where.append("availability_status = ?")
            params.append(availability_status)
        query = "SELECT * FROM source_items"
        if where:
            query += " WHERE " + " AND ".join(where)
        query += " ORDER BY first_captured_at DESC, id LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        rows = self.connection.execute(query, params).fetchall()
        return [
            SourceItem.model_validate({field: row[field] for field in SOURCE_FIELDS})
            for row in rows
        ]

    def list_source_items_by_type(self, source_type: str) -> list[SourceItem]:
        rows = self.connection.execute(
            """
            SELECT * FROM source_items
            WHERE source_type = ?
            ORDER BY first_captured_at DESC, id
            """,
            (source_type,),
        ).fetchall()
        return [
            SourceItem.model_validate({field: row[field] for field in SOURCE_FIELDS})
            for row in rows
        ]

    def count_source_items_by_field(self, field: str) -> dict[str, int]:
        if field not in {"source_type", "availability_status"}:
            raise ValueError("Unsupported source-item count field.")
        rows = self.connection.execute(
            f"""
            SELECT {field} AS value, COUNT(*) AS count
            FROM source_items
            GROUP BY {field}
            ORDER BY {field}
            """
        ).fetchall()
        return {str(row["value"]): int(row["count"]) for row in rows}

    def mark_missing_source_items_deleted(
        self,
        *,
        source_type: str,
        active_source_item_ids: set[str],
        deleted_at: str | None = None,
    ) -> int:
        deleted_at = deleted_at or utc_now().isoformat()
        sources = self.list_source_items_by_type(source_type)
        missing_ids = [
            source.id
            for source in sources
            if source.id not in active_source_item_ids and source.deleted_locally_at is None
        ]
        if not missing_ids:
            return 0
        with self.connection:
            for source_id in missing_ids:
                self.connection.execute(
                    """
                    UPDATE source_items
                    SET deleted_locally_at = ?,
                        availability_status = 'unavailable',
                        updated_at = ?
                    WHERE id = ?
                    """,
                    (deleted_at, deleted_at, source_id),
                )
        return len(missing_ids)

    def delete_source_item(self, source_item_id: str) -> bool:
        return self.delete_complete_source_item(source_item_id)["deleted"]

    def mark_source_unavailable(
        self,
        source_item_id: str,
        *,
        reason: str = "",
    ) -> SourceItem | None:
        source = self.get_source_item(source_item_id)
        if source is None:
            return None
        now = utc_now().isoformat()
        with self.connection:
            self.connection.execute(
                """
                UPDATE source_items
                SET availability_status = 'unavailable',
                    last_synced_at = ?,
                    updated_at = ?
                WHERE id = ?
                """,
                (now, now, source_item_id),
            )
            self._insert_source_lifecycle_event(
                source_item_id=source_item_id,
                event_type="marked-unavailable",
                reason=reason,
                payload={},
                created_at=now,
            )
        return self.get_source_item(source_item_id)

    def request_source_revalidation(
        self,
        source_item_id: str,
        *,
        reason: str = "",
    ) -> dict[str, Any] | None:
        source = self.get_source_item(source_item_id)
        if source is None:
            return None
        now = utc_now().isoformat()
        with self.connection:
            self.connection.execute(
                """
                UPDATE source_items
                SET availability_status = 'unknown',
                    last_synced_at = ?,
                    updated_at = ?
                WHERE id = ?
                """,
                (now, now, source_item_id),
            )
            self._insert_source_lifecycle_event(
                source_item_id=source_item_id,
                event_type="revalidation-requested",
                reason=reason,
                payload={},
                created_at=now,
            )

        updated_cards: list[str] = []
        for card in self.list_knowledge_cards_for_source(source_item_id):
            provenance = json.loads(json.dumps(card.provenance, ensure_ascii=False))
            events = provenance.get("source_lifecycle_events")
            lifecycle_events = events if isinstance(events, list) else []
            lifecycle_events.append(
                {
                    "event_type": "revalidation-requested",
                    "reason": reason.strip(),
                    "created_at": now,
                }
            )
            provenance["source_lifecycle_events"] = lifecycle_events
            updated = card.model_copy(
                update={
                    "freshness_status": "recheck-soon",
                    "freshness_reason": "Source marked for revalidation.",
                    "provenance": provenance,
                }
            )
            self.upsert_knowledge_card(updated)
            updated_cards.append(updated.id)

        return {
            "source_item": self.get_source_item(source_item_id),
            "updated_card_ids": updated_cards,
        }

    def clear_source_raw_references(
        self,
        source_item_id: str,
        *,
        reason: str = "",
    ) -> dict[str, Any]:
        source = self.get_source_item(source_item_id)
        if source is None:
            return {"updated": False, "paths": []}
        documents = self.list_documents_for_source(source_item_id)
        paths = [path for path in [source.raw_path] if path]
        for document in documents:
            paths.extend(path for path in [document.html_snapshot_path] if path)
        now = utc_now().isoformat()
        with self.connection:
            self.connection.execute(
                """
                UPDATE source_items
                SET raw_path = NULL,
                    updated_at = ?
                WHERE id = ?
                """,
                (now, source_item_id),
            )
            for document in documents:
                metadata = dict(document.metadata)
                metadata["raw_retained"] = False
                metadata["raw_deleted_at"] = now
                self.connection.execute(
                    """
                    UPDATE documents
                    SET html_snapshot_path = NULL,
                        metadata_json = ?,
                        updated_at = ?
                    WHERE id = ?
                    """,
                    (_json_dump(metadata), now, document.id),
                )
            self._insert_source_lifecycle_event(
                source_item_id=source_item_id,
                event_type="raw-data-deleted",
                reason=reason,
                payload={"path_count": len(paths)},
                created_at=now,
            )
        return {"updated": True, "paths": paths}

    def delete_complete_source_item(
        self,
        source_item_id: str,
        *,
        detach_user_notes: bool = False,
        reason: str = "",
    ) -> dict[str, Any]:
        source = self.get_source_item(source_item_id)
        if source is None:
            return {
                "deleted": False,
                "source_item_id": source_item_id,
                "deleted_card_ids": [],
                "deleted_document_ids": [],
                "detached_note_id": None,
            }
        documents = self.list_documents_for_source(source_item_id)
        cards = self.list_knowledge_cards_for_source(source_item_id)
        detached_note_id = None
        now = utc_now().isoformat()
        with self.connection:
            if detach_user_notes:
                detached_note_id = self._insert_detached_note(
                    source=source,
                    cards=cards,
                    detached_at=now,
                    reason=reason,
                )
            for card in cards:
                self.connection.execute("DELETE FROM card_fts WHERE card_id = ?", (card.id,))
            cursor = self.connection.execute(
                "DELETE FROM source_items WHERE id = ?",
                (source_item_id,),
            )
        return {
            "deleted": cursor.rowcount > 0,
            "source_item_id": source_item_id,
            "deleted_card_ids": [card.id for card in cards],
            "deleted_document_ids": [document.id for document in documents],
            "detached_note_id": detached_note_id,
        }

    def get_detached_note(self, detached_note_id: str) -> dict[str, Any] | None:
        row = self.connection.execute(
            "SELECT * FROM detached_notes WHERE id = ?",
            (detached_note_id,),
        ).fetchone()
        if row is None:
            return None
        return {
            "id": row["id"],
            "original_source_item_id": row["original_source_item_id"],
            "source_type": row["source_type"],
            "source_url": row["source_url"],
            "title": row["title"],
            "user_notes": _json_load(row["user_notes_json"], {}),
            "detached_at": row["detached_at"],
            "reason": row["reason"],
        }

    def list_source_lifecycle_events(self, source_item_id: str) -> list[dict[str, Any]]:
        rows = self.connection.execute(
            """
            SELECT * FROM source_lifecycle_events
            WHERE source_item_id = ?
            ORDER BY created_at DESC, id
            """,
            (source_item_id,),
        ).fetchall()
        return [
            {
                "id": row["id"],
                "source_item_id": row["source_item_id"],
                "event_type": row["event_type"],
                "reason": row["reason"],
                "payload": _json_load(row["payload_json"], {}),
                "created_at": row["created_at"],
            }
            for row in rows
        ]

    def _insert_source_lifecycle_event(
        self,
        *,
        source_item_id: str,
        event_type: str,
        reason: str,
        payload: dict[str, Any],
        created_at: str,
    ) -> str:
        event_id = f"source_event_{stable_hash(f'{source_item_id}:{event_type}:{created_at}')}"
        self.connection.execute(
            """
            INSERT INTO source_lifecycle_events (
                id,
                source_item_id,
                event_type,
                reason,
                payload_json,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                event_id,
                source_item_id,
                event_type,
                reason.strip(),
                _json_dump(payload),
                created_at,
            ),
        )
        return event_id

    def _insert_detached_note(
        self,
        *,
        source: SourceItem,
        cards: list[KnowledgeCard],
        detached_at: str,
        reason: str,
    ) -> str:
        note_id = f"detached_{stable_hash(f'{source.id}:{detached_at}')}"
        payload = _detached_user_notes(cards)
        title = cards[0].title if cards else source.source_url or source.source_external_id
        self.connection.execute(
            """
            INSERT INTO detached_notes (
                id,
                original_source_item_id,
                source_type,
                source_url,
                title,
                user_notes_json,
                detached_at,
                reason
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                note_id,
                source.id,
                str(source.source_type),
                source.source_url,
                title,
                _json_dump(payload),
                detached_at,
                reason.strip(),
            ),
        )
        return note_id

    def upsert_document(self, document: Document) -> bool:
        created = self.get_document(document.id) is None
        data = document.model_dump(mode="json")
        now = utc_now().isoformat()
        params = {field: data[field] for field in DOCUMENT_FIELDS}
        params["metadata_json"] = _json_dump(data["metadata"])
        params["created_at"] = now
        params["updated_at"] = now

        columns = (*DOCUMENT_FIELDS, "metadata_json", "created_at", "updated_at")
        placeholders = ", ".join(f":{column}" for column in columns)
        update_columns = [field for field in columns if field not in {"id", "created_at"}]
        update_set = ", ".join(f"{field} = excluded.{field}" for field in update_columns)
        sql = f"""
            INSERT INTO documents ({", ".join(columns)})
            VALUES ({placeholders})
            ON CONFLICT(id) DO UPDATE SET
                {update_set}
        """
        with self.connection:
            self.connection.execute(sql, params)
            self._refresh_source_card_fts(document.source_item_id)
        return created

    def get_document(self, document_id: str) -> Document | None:
        row = self.connection.execute(
            "SELECT * FROM documents WHERE id = ?",
            (document_id,),
        ).fetchone()
        if row is None:
            return None
        return self._row_to_document(row)

    def list_documents_for_source(self, source_item_id: str) -> list[Document]:
        rows = self.connection.execute(
            """
            SELECT * FROM documents
            WHERE source_item_id = ?
            ORDER BY published_at DESC, id
            """,
            (source_item_id,),
        ).fetchall()
        return [self._row_to_document(row) for row in rows]

    def delete_document(self, document_id: str) -> bool:
        with self.connection:
            cursor = self.connection.execute("DELETE FROM documents WHERE id = ?", (document_id,))
        return cursor.rowcount > 0

    def upsert_knowledge_card(self, card: KnowledgeCard) -> bool:
        created = self.get_knowledge_card(card.id) is None
        params = self._card_to_params(card)
        columns = (
            *CARD_SCALAR_FIELDS,
            *(f"{field}_json" for field in CARD_JSON_FIELDS),
            "provenance_json",
            "created_at",
            "updated_at",
        )
        placeholders = ", ".join(f":{column}" for column in columns)
        update_columns = [field for field in columns if field not in {"id", "created_at"}]
        update_set = ", ".join(f"{field} = excluded.{field}" for field in update_columns)
        sql = f"""
            INSERT INTO knowledge_cards ({", ".join(columns)})
            VALUES ({placeholders})
            ON CONFLICT(id) DO UPDATE SET
                {update_set}
        """

        with self.connection:
            self.connection.execute(sql, params)
            self._replace_card_terms(card.id, "topics", card.topics)
            self._replace_card_terms(card.id, "tags", card.tags)
            self._refresh_card_fts(card)
        return created

    def get_knowledge_card(self, card_id: str) -> KnowledgeCard | None:
        row = self.connection.execute(
            "SELECT * FROM knowledge_cards WHERE id = ?",
            (card_id,),
        ).fetchone()
        if row is None:
            return None
        return self._row_to_card(row)

    def list_knowledge_cards_for_source(self, source_item_id: str) -> list[KnowledgeCard]:
        rows = self.connection.execute(
            """
            SELECT * FROM knowledge_cards
            WHERE source_item_id = ?
            ORDER BY generated_at DESC, id
            """,
            (source_item_id,),
        ).fetchall()
        return [self._row_to_card(row) for row in rows]

    def find_duplicate_cards(
        self,
        *,
        source_item: SourceItem,
        document: Document,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        source_hash = source_item.content_hash or ""
        document_hash = document.content_hash or ""
        canonical_url = document.canonical_url or ""
        if not any((source_hash, document_hash, canonical_url)):
            return []
        rows = self.connection.execute(
            """
            SELECT DISTINCT
                knowledge_cards.*,
                source_items.id AS duplicate_source_item_id,
                source_items.source_url AS duplicate_source_url,
                source_items.content_hash AS duplicate_source_hash,
                documents.id AS duplicate_document_id,
                documents.canonical_url AS duplicate_canonical_url,
                documents.content_hash AS duplicate_document_hash
            FROM knowledge_cards
            JOIN source_items ON source_items.id = knowledge_cards.source_item_id
            LEFT JOIN documents ON documents.source_item_id = source_items.id
            WHERE knowledge_cards.source_item_id != :source_item_id
                AND (
                    (:source_hash != '' AND source_items.content_hash = :source_hash)
                    OR (:document_hash != '' AND documents.content_hash = :document_hash)
                    OR (:canonical_url != '' AND documents.canonical_url = :canonical_url)
                )
            ORDER BY knowledge_cards.generated_at DESC, knowledge_cards.id
            LIMIT :limit
            """,
            {
                "source_item_id": source_item.id,
                "source_hash": source_hash,
                "document_hash": document_hash,
                "canonical_url": canonical_url,
                "limit": limit,
            },
        ).fetchall()
        duplicates: list[dict[str, Any]] = []
        for row in rows:
            reasons: list[str] = []
            if source_hash and row["duplicate_source_hash"] == source_hash:
                reasons.append("source-content-hash")
            if document_hash and row["duplicate_document_hash"] == document_hash:
                reasons.append("document-content-hash")
            if canonical_url and row["duplicate_canonical_url"] == canonical_url:
                reasons.append("canonical-url")
            duplicates.append(
                {
                    "card": self._row_to_card(row),
                    "source_item_id": row["duplicate_source_item_id"],
                    "source_url": row["duplicate_source_url"],
                    "document_id": row["duplicate_document_id"],
                    "canonical_url": row["duplicate_canonical_url"],
                    "match_reasons": reasons,
                }
            )
        return duplicates

    def upsert_duplicate_relationship(
        self,
        *,
        candidate_card_id: str,
        match_card_id: str,
        status: str,
        reason: str = "",
        created_by: str = "user",
    ) -> dict[str, Any]:
        normalized_status = status.strip().lower()
        if normalized_status not in {"duplicate", "not-duplicate"}:
            raise ValueError("Duplicate relationship status must be duplicate or not-duplicate.")
        if candidate_card_id == match_card_id:
            raise ValueError("A card cannot be marked as a duplicate of itself.")
        if self.get_knowledge_card(candidate_card_id) is None:
            raise ValueError(f"No ContextBank card found for {candidate_card_id}")
        if self.get_knowledge_card(match_card_id) is None:
            raise ValueError(f"No ContextBank card found for {match_card_id}")

        now = utc_now().isoformat()
        with self.connection:
            self.connection.execute(
                """
                INSERT INTO duplicate_relationships (
                    candidate_card_id,
                    match_card_id,
                    status,
                    reason,
                    created_by,
                    created_at,
                    updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(candidate_card_id, match_card_id) DO UPDATE SET
                    status = excluded.status,
                    reason = excluded.reason,
                    created_by = excluded.created_by,
                    updated_at = excluded.updated_at
                """,
                (
                    candidate_card_id,
                    match_card_id,
                    normalized_status,
                    reason.strip(),
                    created_by.strip() or "user",
                    now,
                    now,
                ),
            )
        relationship = self.get_duplicate_relationship(candidate_card_id, match_card_id)
        if relationship is None:  # pragma: no cover - defensive only
            raise RuntimeError("Duplicate relationship was not persisted.")
        return relationship

    def get_duplicate_relationship(
        self,
        candidate_card_id: str,
        match_card_id: str,
    ) -> dict[str, Any] | None:
        row = self.connection.execute(
            """
            SELECT * FROM duplicate_relationships
            WHERE candidate_card_id = ? AND match_card_id = ?
            """,
            (candidate_card_id, match_card_id),
        ).fetchone()
        return _duplicate_relationship_from_row(row) if row else None

    def list_duplicate_relationships(
        self,
        *,
        card_id: str | None = None,
        status: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        clauses: list[str] = []
        params: dict[str, Any] = {"limit": limit}
        if card_id:
            clauses.append("(candidate_card_id = :card_id OR match_card_id = :card_id)")
            params["card_id"] = card_id
        if status:
            clauses.append("status = :status")
            params["status"] = status.strip().lower()
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        rows = self.connection.execute(
            f"""
            SELECT * FROM duplicate_relationships
            {where}
            ORDER BY updated_at DESC, candidate_card_id, match_card_id
            LIMIT :limit
            """,
            params,
        ).fetchall()
        return [_duplicate_relationship_from_row(row) for row in rows]

    def list_knowledge_cards(
        self,
        *,
        filters: SearchFilters | None = None,
        limit: int = 100,
        offset: int = 0,
        default_visible: bool = False,
    ) -> list[KnowledgeCard]:
        where_sql, params = self._card_filter_sql(
            filters,
            default_visible=default_visible,
        )
        params["limit"] = limit
        params["offset"] = offset
        rows = self.connection.execute(
            f"""
            SELECT knowledge_cards.* FROM knowledge_cards
            LEFT JOIN source_items ON source_items.id = knowledge_cards.source_item_id
            {where_sql}
            ORDER BY generated_at DESC, id
            LIMIT :limit OFFSET :offset
            """,
            params,
        ).fetchall()
        return [self._row_to_card(row) for row in rows]

    def search_cards(
        self,
        query: str,
        *,
        filters: SearchFilters | None = None,
        limit: int = 10,
        default_visible: bool = False,
    ) -> list[SearchResult]:
        match_query = _fts_query(query)
        if match_query is None:
            cards = self.list_knowledge_cards(
                filters=filters,
                limit=limit,
                default_visible=default_visible,
            )
            return [
                SearchResult(
                    card=card,
                    score=0.0,
                    relevance_reason="Listed without a full-text query",
                    **self._source_result_fields(card.source_item_id),
                )
                for card in cards
            ]

        where_sql, params = self._card_filter_sql(
            filters,
            prefix="AND",
            default_visible=default_visible,
        )
        params["query"] = match_query
        params["limit"] = limit
        rows = self.connection.execute(
            f"""
            SELECT
                knowledge_cards.*,
                bm25(card_fts) AS fts_score,
                snippet(card_fts, 4, '', '', '...', 18) AS fts_excerpt,
                source_items.source_url AS result_source_url,
                source_items.source_type AS result_source_type,
                source_items.source_created_at AS result_source_created_at,
                source_items.first_captured_at AS result_captured_at
            FROM card_fts
            JOIN knowledge_cards ON knowledge_cards.id = card_fts.card_id
            LEFT JOIN source_items ON source_items.id = knowledge_cards.source_item_id
            WHERE card_fts MATCH :query
            {where_sql}
            ORDER BY fts_score, knowledge_cards.generated_at DESC
            LIMIT :limit
            """,
            params,
        ).fetchall()

        results: list[SearchResult] = []
        for row in rows:
            raw_score = float(row["fts_score"] or 0.0)
            results.append(
                SearchResult(
                    card=self._row_to_card(row),
                    score=max(0.0, -raw_score),
                    relevance_reason="Matched the local FTS5 card index",
                    source_url=row["result_source_url"],
                    source_type=row["result_source_type"],
                    source_created_at=row["result_source_created_at"],
                    captured_at=row["result_captured_at"],
                    excerpt=row["fts_excerpt"],
                )
            )
        return results

    def delete_knowledge_card(self, card_id: str) -> bool:
        with self.connection:
            self.connection.execute("DELETE FROM card_fts WHERE card_id = ?", (card_id,))
            cursor = self.connection.execute("DELETE FROM knowledge_cards WHERE id = ?", (card_id,))
        return cursor.rowcount > 0

    def upsert_card_embedding(
        self,
        *,
        card_id: str,
        provider: str,
        model_id: str,
        vector: list[float],
        source_hash: str,
    ) -> None:
        now = utc_now().isoformat()
        with self.connection:
            self.connection.execute(
                """
                INSERT INTO card_embeddings (
                    card_id,
                    provider,
                    model_id,
                    vector_json,
                    source_hash,
                    indexed_at
                )
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(card_id, provider, model_id) DO UPDATE SET
                    vector_json = excluded.vector_json,
                    source_hash = excluded.source_hash,
                    indexed_at = excluded.indexed_at
                """,
                (card_id, provider, model_id, _json_dump(vector), source_hash, now),
            )

    def list_card_embeddings(
        self,
        *,
        provider: str | None = None,
        model_id: str | None = None,
        limit: int = 1000,
    ) -> list[dict[str, Any]]:
        clauses: list[str] = []
        params: dict[str, Any] = {"limit": limit}
        if provider:
            clauses.append("provider = :provider")
            params["provider"] = provider
        if model_id:
            clauses.append("model_id = :model_id")
            params["model_id"] = model_id
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        rows = self.connection.execute(
            f"""
            SELECT * FROM card_embeddings
            {where}
            ORDER BY indexed_at DESC, card_id
            LIMIT :limit
            """,
            params,
        ).fetchall()
        return [
            {
                "card_id": row["card_id"],
                "provider": row["provider"],
                "model_id": row["model_id"],
                "vector": _json_load(row["vector_json"], []),
                "source_hash": row["source_hash"],
                "indexed_at": row["indexed_at"],
            }
            for row in rows
        ]

    def count_rows(self, table: str) -> int:
        if table not in COUNTABLE_TABLES:
            raise ValueError(f"Unsupported table for row count: {table}")
        row = self.connection.execute(f"SELECT COUNT(*) AS count FROM {table}").fetchone()
        return int(row["count"])

    def count_fts_rows(self, card_id: str | None = None) -> int:
        if card_id is None:
            return self.count_rows("card_fts")
        row = self.connection.execute(
            "SELECT COUNT(*) AS count FROM card_fts WHERE card_id = ?",
            (card_id,),
        ).fetchone()
        return int(row["count"])

    def get_sync_state(self, connector: str, account_id: str) -> dict[str, Any] | None:
        row = self.connection.execute(
            """
            SELECT * FROM sync_state
            WHERE connector = ? AND account_id = ?
            """,
            (connector, account_id),
        ).fetchone()
        if row is None:
            return None
        return _sync_state_from_row(row)

    def list_sync_states(self, connector: str | None = None) -> list[dict[str, Any]]:
        if connector:
            rows = self.connection.execute(
                """
                SELECT * FROM sync_state
                WHERE connector = ?
                ORDER BY updated_at DESC, account_id
                """,
                (connector,),
            ).fetchall()
        else:
            rows = self.connection.execute(
                """
                SELECT * FROM sync_state
                ORDER BY updated_at DESC, connector, account_id
                """
            ).fetchall()
        return [_sync_state_from_row(row) for row in rows]

    def upsert_sync_state(
        self,
        *,
        connector: str,
        account_id: str,
        cursor: str | None,
        status: str,
        last_started_at: str | None = None,
        last_completed_at: str | None = None,
        last_error: str | None = None,
        items_synced: int = 0,
        full_sync_completed: bool = False,
    ) -> None:
        now = utc_now().isoformat()
        params = {
            "connector": connector,
            "account_id": account_id,
            "cursor": cursor,
            "status": status,
            "last_started_at": last_started_at,
            "last_completed_at": last_completed_at,
            "last_error": last_error,
            "items_synced": max(items_synced, 0),
            "full_sync_completed": 1 if full_sync_completed else 0,
            "updated_at": now,
        }
        columns = ", ".join(SYNC_STATE_FIELDS)
        placeholders = ", ".join(f":{field}" for field in SYNC_STATE_FIELDS)
        update_set = ", ".join(
            f"{field} = excluded.{field}"
            for field in SYNC_STATE_FIELDS
            if field not in {"connector", "account_id"}
        )
        with self.connection:
            self.connection.execute(
                f"""
                INSERT INTO sync_state ({columns})
                VALUES ({placeholders})
                ON CONFLICT(connector, account_id) DO UPDATE SET
                    {update_set}
                """,
                params,
            )

    def delete_sync_state(self, connector: str, account_id: str) -> bool:
        with self.connection:
            cursor = self.connection.execute(
                """
                DELETE FROM sync_state
                WHERE connector = ? AND account_id = ?
                """,
                (connector, account_id),
            )
        return cursor.rowcount > 0

    def upsert_project(
        self,
        *,
        root_path: Path | str,
        name: str | None = None,
        profile: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        root = Path(root_path).expanduser().resolve()
        project_id = _project_id(root)
        project_name = (name or (profile or {}).get("name") or root.name).strip() or root.name
        now = utc_now().isoformat()
        params = {
            "id": project_id,
            "name": project_name,
            "root_path": str(root),
            "profile_json": _json_dump(profile or {}),
            "created_at": now,
            "last_seen_at": now,
        }
        with self.connection:
            self.connection.execute(
                """
                INSERT INTO projects (id, name, root_path, profile_json, created_at, last_seen_at)
                VALUES (:id, :name, :root_path, :profile_json, :created_at, :last_seen_at)
                ON CONFLICT(id) DO UPDATE SET
                    name = excluded.name,
                    root_path = excluded.root_path,
                    profile_json = excluded.profile_json,
                    last_seen_at = excluded.last_seen_at
                """,
                params,
            )
        return self.get_project(project_id) or params

    def get_project(self, project_id: str) -> dict[str, Any] | None:
        row = self.connection.execute(
            "SELECT * FROM projects WHERE id = ?",
            (project_id,),
        ).fetchone()
        if row is None:
            return None
        return _project_from_row(row)

    def get_project_by_root(self, root_path: Path | str) -> dict[str, Any] | None:
        root = str(Path(root_path).expanduser().resolve())
        row = self.connection.execute(
            "SELECT * FROM projects WHERE root_path = ?",
            (root,),
        ).fetchone()
        if row is None:
            return None
        return _project_from_row(row)

    def link_card_to_project(
        self,
        *,
        card_id: str,
        root_path: Path | str,
        name: str | None = None,
        profile: dict[str, Any] | None = None,
        relevance: float = 0.0,
        reason: str = "",
        intended_use: str = "",
        status: str = "candidate",
        outcome_note: str = "",
        created_by: str = "user",
    ) -> dict[str, Any]:
        project = self.upsert_project(root_path=root_path, name=name, profile=profile)
        now = utc_now().isoformat()
        params = {
            "project_id": project["id"],
            "knowledge_card_id": card_id,
            "relevance": max(0.0, min(float(relevance), 1.0)),
            "reason": reason.strip(),
            "intended_use": intended_use.strip(),
            "status": status.strip(),
            "outcome_note": outcome_note.strip(),
            "created_by": created_by.strip() or "user",
            "created_at": now,
            "updated_at": now,
        }
        with self.connection:
            self.connection.execute(
                """
                INSERT INTO project_item_links (
                    project_id,
                    knowledge_card_id,
                    relevance,
                    reason,
                    intended_use,
                    status,
                    outcome_note,
                    created_by,
                    created_at,
                    updated_at
                )
                VALUES (
                    :project_id,
                    :knowledge_card_id,
                    :relevance,
                    :reason,
                    :intended_use,
                    :status,
                    :outcome_note,
                    :created_by,
                    :created_at,
                    :updated_at
                )
                ON CONFLICT(project_id, knowledge_card_id) DO UPDATE SET
                    relevance = excluded.relevance,
                    reason = excluded.reason,
                    intended_use = excluded.intended_use,
                    status = excluded.status,
                    outcome_note = excluded.outcome_note,
                    created_by = excluded.created_by,
                    updated_at = excluded.updated_at
                """,
                params,
            )
        return self.get_project_link(project["id"], card_id) or {**params, "project": project}

    def get_project_link(self, project_id: str, card_id: str) -> dict[str, Any] | None:
        row = self.connection.execute(
            """
            SELECT
                project_item_links.*,
                projects.name AS project_name,
                projects.root_path AS project_root_path
            FROM project_item_links
            JOIN projects ON projects.id = project_item_links.project_id
            WHERE project_item_links.project_id = ?
                AND project_item_links.knowledge_card_id = ?
            """,
            (project_id, card_id),
        ).fetchone()
        if row is None:
            return None
        return _project_link_from_row(row)

    def list_project_links_for_card(self, card_id: str) -> list[dict[str, Any]]:
        rows = self.connection.execute(
            """
            SELECT
                project_item_links.*,
                projects.name AS project_name,
                projects.root_path AS project_root_path
            FROM project_item_links
            JOIN projects ON projects.id = project_item_links.project_id
            WHERE project_item_links.knowledge_card_id = ?
            ORDER BY project_item_links.updated_at DESC, projects.name
            """,
            (card_id,),
        ).fetchall()
        return [_project_link_from_row(row) for row in rows]

    def list_project_linked_cards(
        self,
        *,
        root_path: Path | str | None = None,
        name: str | None = None,
        match_any: bool = True,
        limit: int = 20,
    ) -> list[tuple[KnowledgeCard, dict[str, Any]]]:
        clauses: list[str] = []
        params: dict[str, Any] = {"limit": limit}
        if root_path is not None:
            clauses.append("projects.root_path = :root_path")
            params["root_path"] = str(Path(root_path).expanduser().resolve())
        if name:
            clauses.append("(projects.name = :name OR projects.id = :name)")
            params["name"] = name
        if not clauses:
            return []
        joiner = " OR " if match_any else " AND "
        rows = self.connection.execute(
            f"""
            SELECT
                knowledge_cards.*,
                project_item_links.project_id AS project_id,
                project_item_links.knowledge_card_id AS knowledge_card_id,
                project_item_links.relevance AS relevance,
                project_item_links.reason AS reason,
                project_item_links.intended_use AS intended_use,
                project_item_links.status AS status,
                project_item_links.outcome_note AS outcome_note,
                project_item_links.created_by AS created_by,
                project_item_links.created_at AS link_created_at,
                project_item_links.updated_at AS link_updated_at,
                projects.name AS project_name,
                projects.root_path AS project_root_path,
                CASE project_item_links.status
                    WHEN 'used' THEN 0
                    WHEN 'candidate' THEN 1
                    ELSE 2
                END AS status_rank
            FROM project_item_links
            JOIN projects ON projects.id = project_item_links.project_id
            JOIN knowledge_cards ON knowledge_cards.id = project_item_links.knowledge_card_id
            WHERE {joiner.join(clauses)}
            ORDER BY status_rank, project_item_links.relevance DESC,
                project_item_links.updated_at DESC, knowledge_cards.generated_at DESC
            LIMIT :limit
            """,
            params,
        ).fetchall()
        return [
            (
                self._row_to_card(row),
                {
                    "project_id": row["project_id"],
                    "project_name": row["project_name"],
                    "project_root_path": row["project_root_path"],
                    "knowledge_card_id": row["knowledge_card_id"],
                    "relevance": float(row["relevance"]),
                    "reason": row["reason"],
                    "intended_use": row["intended_use"],
                    "status": row["status"],
                    "outcome_note": row["outcome_note"],
                    "created_by": row["created_by"],
                    "created_at": row["link_created_at"],
                    "updated_at": row["link_updated_at"],
                },
            )
            for row in rows
        ]

    def list_review_queue(
        self,
        *,
        queue: str = "all",
        low_confidence_threshold: float = 0.5,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        normalized_queue = _review_queue_name(queue)
        cards_by_source = {
            card.source_item_id: card
            for card in self.list_knowledge_cards(limit=max(limit * 3, limit))
        }
        items: list[dict[str, Any]] = []
        for source in self.list_source_items(limit=max(limit * 3, limit)):
            card = cards_by_source.get(source.id)
            documents = self.list_documents_for_source(source.id)
            reasons = _review_reasons(
                source,
                documents,
                card,
                low_confidence_threshold=low_confidence_threshold,
            )
            if not reasons or (normalized_queue != "all" and normalized_queue not in reasons):
                continue
            items.append(
                {
                    "source_item_id": source.id,
                    "card_id": card.id if card else None,
                    "title": _review_title(source, documents, card),
                    "source_type": source.source_type,
                    "source_url": source.source_url,
                    "reasons": reasons,
                    "availability_status": source.availability_status,
                    "document_statuses": sorted(
                        {str(document.extraction_status) for document in documents}
                    ),
                    "confidence": card.confidence if card else None,
                    "freshness_status": card.freshness_status if card else None,
                    "review_status": card.review_status if card else None,
                    "skill_candidate_status": card.skill_candidate_status if card else None,
                    "generated_at": card.generated_at.isoformat() if card else None,
                    "document_count": len(documents),
                }
            )
            if len(items) >= limit:
                break
        return items

    def _row_to_document(self, row: sqlite3.Row) -> Document:
        data = {field: row[field] for field in DOCUMENT_FIELDS}
        data["metadata"] = _json_load(row["metadata_json"], {})
        return Document.model_validate(data)

    def _card_to_params(self, card: KnowledgeCard) -> dict[str, Any]:
        data = card.model_dump(mode="json")
        params = {field: data[field] for field in CARD_SCALAR_FIELDS}
        for field in CARD_JSON_FIELDS:
            params[f"{field}_json"] = _json_dump(data[field])
        params["provenance_json"] = _json_dump(data["provenance"])
        now = utc_now().isoformat()
        params["created_at"] = now
        params["updated_at"] = now
        return params

    def _row_to_card(self, row: sqlite3.Row) -> KnowledgeCard:
        data = {field: row[field] for field in CARD_SCALAR_FIELDS}
        for field in CARD_JSON_FIELDS:
            data[field] = _json_load(row[f"{field}_json"], [])
        data["provenance"] = _json_load(row["provenance_json"], {})
        data["topics"] = self._card_terms(row["id"], "topics")
        data["tags"] = self._card_terms(row["id"], "tags")
        return KnowledgeCard.model_validate(data)

    def _card_terms(self, card_id: str, term_type: str) -> list[str]:
        table, link_table, slug_column = _term_tables(term_type)
        rows = self.connection.execute(
            f"""
            SELECT {table}.name
            FROM {link_table}
            JOIN {table} ON {table}.slug = {link_table}.{slug_column}
            WHERE {link_table}.card_id = ?
            ORDER BY {link_table}.rowid
            """,
            (card_id,),
        ).fetchall()
        return [row["name"] for row in rows]

    def _replace_card_terms(self, card_id: str, term_type: str, terms: Iterable[str]) -> None:
        table, link_table, slug_column = _term_tables(term_type)
        normalized_terms = _normalize_terms(terms)
        self.connection.execute(f"DELETE FROM {link_table} WHERE card_id = ?", (card_id,))
        for slug, name in normalized_terms:
            self.connection.execute(
                f"""
                INSERT INTO {table}(slug, name)
                VALUES (?, ?)
                ON CONFLICT(slug) DO UPDATE SET name = excluded.name
                """,
                (slug, name),
            )
            self.connection.execute(
                f"INSERT OR IGNORE INTO {link_table}(card_id, {slug_column}) VALUES (?, ?)",
                (card_id, slug),
            )

    def _refresh_card_fts(self, card: KnowledgeCard) -> None:
        source = self.get_source_item(card.source_item_id)
        documents = self.list_documents_for_source(card.source_item_id)
        body_parts = [
            card.why_useful,
            *card.key_insights,
            *card.possible_applications,
            *card.actionable_next_steps,
            *card.implementation_notes,
            *card.prerequisites,
            *card.patterns,
            *card.caveats,
            *card.source_quality_notes,
            card.content_type,
            *card.utility,
            card.freshness_reason,
            *_project_link_index_terms(card),
        ]
        document_text = "\n".join(document.text or "" for document in documents if document.text)
        source_authors = " ".join(
            part
            for part in [
                source.author_name if source else None,
                source.author_handle if source else None,
                source.author_id if source else None,
                *(document.author for document in documents),
            ]
            if part
        )
        self.connection.execute("DELETE FROM card_fts WHERE card_id = ?", (card.id,))
        self.connection.execute(
            """
            INSERT INTO card_fts(
                card_id,
                title,
                one_line_summary,
                summary,
                body,
                topics,
                tags,
                document_text,
                source_authors
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                card.id,
                card.title,
                card.one_line_summary,
                card.summary,
                "\n".join(part for part in body_parts if part),
                " ".join(card.topics),
                " ".join(card.tags),
                document_text,
                source_authors,
            ),
        )

    def _refresh_source_card_fts(self, source_item_id: str) -> None:
        for card in self.list_knowledge_cards_for_source(source_item_id):
            self._refresh_card_fts(card)

    def _card_filter_sql(
        self,
        filters: SearchFilters | None,
        *,
        prefix: str = "WHERE",
        default_visible: bool = False,
    ) -> tuple[str, dict[str, Any]]:
        clauses: list[str] = []
        params: dict[str, Any] = {}
        if default_visible and not (filters and filters.review_status):
            clauses.append(
                "knowledge_cards.review_status NOT IN "
                "(:default_hidden_status, :default_rejected_status)"
            )
            params["default_hidden_status"] = "hidden"
            params["default_rejected_status"] = "rejected"
        if filters is None:
            if not clauses:
                return "", params
            return f"{prefix} " + " AND ".join(clauses), params

        if filters.source:
            clauses.append(
                "(source_items.source_type = :source OR source_items.source_url = :source)"
            )
            params["source"] = filters.source
        if filters.author:
            clauses.append(
                """
                (
                    source_items.author_name = :author
                    OR source_items.author_handle = :author
                    OR source_items.author_id = :author
                )
                """
            )
            params["author"] = filters.author
        if filters.topic:
            clauses.append(
                """
                EXISTS (
                    SELECT 1 FROM card_topics
                    WHERE card_topics.card_id = knowledge_cards.id
                    AND card_topics.topic_slug = :topic_slug
                )
                """
            )
            params["topic_slug"] = slugify(filters.topic)
        if filters.content_type:
            clauses.append("knowledge_cards.content_type = :content_type")
            params["content_type"] = filters.content_type
        if filters.utility:
            clauses.append("knowledge_cards.utility_json LIKE :utility")
            params["utility"] = f'%"{filters.utility}"%'
        if filters.freshness_status:
            clauses.append("knowledge_cards.freshness_status = :freshness_status")
            params["freshness_status"] = filters.freshness_status
        if filters.skill_candidate_status:
            clauses.append("knowledge_cards.skill_candidate_status = :skill_candidate_status")
            params["skill_candidate_status"] = filters.skill_candidate_status
        if filters.review_status:
            clauses.append("knowledge_cards.review_status = :review_status")
            params["review_status"] = filters.review_status
        if filters.project:
            clauses.append(
                """
                (
                    knowledge_cards.provenance_json LIKE :project_like
                    OR EXISTS (
                        SELECT 1
                        FROM project_item_links
                        JOIN projects ON projects.id = project_item_links.project_id
                        WHERE project_item_links.knowledge_card_id = knowledge_cards.id
                            AND (
                                projects.id = :project_exact
                                OR projects.root_path = :project_exact
                                OR projects.name = :project_exact
                                OR projects.name LIKE :project_text
                            )
                    )
                )
                """
            )
            params["project_like"] = f'%"{filters.project}"%'
            params["project_exact"] = filters.project
            params["project_text"] = f"%{filters.project}%"
        if filters.min_confidence is not None:
            clauses.append("knowledge_cards.confidence >= :min_confidence")
            params["min_confidence"] = filters.min_confidence
        if filters.date_from is not None:
            clauses.append("source_items.source_created_at >= :date_from")
            params["date_from"] = filters.date_from.isoformat()
        if filters.date_to is not None:
            clauses.append("source_items.source_created_at <= :date_to")
            params["date_to"] = filters.date_to.isoformat()
        if filters.used is not None:
            used_exists = (
                "EXISTS (SELECT 1 FROM project_item_links "
                "WHERE project_item_links.knowledge_card_id = knowledge_cards.id "
                "AND project_item_links.status = :used_status)"
            )
            clauses.append(used_exists if filters.used else f"NOT {used_exists}")
            params["used_status"] = "used"

        if not clauses:
            return "", params
        return f"{prefix} " + " AND ".join(clauses), params

    def _source_result_fields(self, source_item_id: str) -> dict[str, Any]:
        row = self.connection.execute(
            """
            SELECT source_url, source_type, source_created_at, first_captured_at
            FROM source_items
            WHERE id = ?
            """,
            (source_item_id,),
        ).fetchone()
        if row is None:
            return {
                "source_url": None,
                "source_type": None,
                "source_created_at": None,
                "captured_at": None,
            }
        return {
            "source_url": row["source_url"],
            "source_type": row["source_type"],
            "source_created_at": row["source_created_at"],
            "captured_at": row["first_captured_at"],
        }


def _json_dump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _json_load(value: str | None, default: Any) -> Any:
    if value in (None, ""):
        return default
    return json.loads(value)


def _sync_state_from_row(row: sqlite3.Row) -> dict[str, Any]:
    data = {field: row[field] for field in SYNC_STATE_FIELDS}
    data["full_sync_completed"] = bool(data["full_sync_completed"])
    return data


def _project_from_row(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "id": row["id"],
        "name": row["name"],
        "root_path": row["root_path"],
        "profile": _json_load(row["profile_json"], {}),
        "created_at": row["created_at"],
        "last_seen_at": row["last_seen_at"],
    }


def _project_link_from_row(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "project_id": row["project_id"],
        "project_name": row["project_name"],
        "project_root_path": row["project_root_path"],
        "knowledge_card_id": row["knowledge_card_id"],
        "relevance": float(row["relevance"]),
        "reason": row["reason"],
        "intended_use": row["intended_use"],
        "status": row["status"],
        "outcome_note": row["outcome_note"],
        "created_by": row["created_by"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


def _duplicate_relationship_from_row(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "candidate_card_id": row["candidate_card_id"],
        "match_card_id": row["match_card_id"],
        "status": row["status"],
        "reason": row["reason"],
        "created_by": row["created_by"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


def _project_id(root_path: Path) -> str:
    return f"project_{stable_hash(str(root_path))}"


def _project_link_index_terms(card: KnowledgeCard) -> list[str]:
    links = card.provenance.get("project_links")
    if not isinstance(links, list):
        return []
    terms: list[str] = []
    for link in links:
        if not isinstance(link, dict):
            continue
        for key in ("project_name", "reason", "intended_use", "status", "outcome_note"):
            value = link.get(key)
            if isinstance(value, str) and value.strip():
                terms.append(value.strip())
    return terms


def _detached_user_notes(cards: list[KnowledgeCard]) -> dict[str, Any]:
    detached_cards: list[dict[str, Any]] = []
    for card in cards:
        provenance = card.provenance if isinstance(card.provenance, dict) else {}
        user_signals = provenance.get("user_signals")
        correction_history = provenance.get("correction_history")
        project_links = provenance.get("project_links")
        skill_candidate = provenance.get("skill_candidate")
        detached_cards.append(
            {
                "card_id": card.id,
                "title": card.title,
                "personal_priority": card.personal_priority,
                "review_status": card.review_status,
                "skill_candidate_status": card.skill_candidate_status,
                "user_signals": user_signals if isinstance(user_signals, dict) else {},
                "correction_history": (
                    correction_history if isinstance(correction_history, list) else []
                ),
                "project_links": project_links if isinstance(project_links, list) else [],
                "skill_candidate": skill_candidate if isinstance(skill_candidate, dict) else {},
            }
        )
    return {"cards": detached_cards}


def _review_queue_name(queue: str) -> str:
    normalized = queue.strip().lower()
    aliases = {
        "all": "all",
        "skill": "skill-review",
        "skill-candidate": "skill-review",
        "skill-review": "skill-review",
        "low-confidence": "low-confidence",
        "stale": "stale",
        "partial": "partial",
        "failed": "failed",
        "missing-source": "missing-source",
        "missing-source-content": "missing-source",
        "duplicate": "duplicates",
        "duplicates": "duplicates",
        "possible-duplicate": "duplicates",
        "hidden": "hidden",
        "unprocessed": "unprocessed",
    }
    if normalized not in aliases:
        allowed = ", ".join(sorted(aliases))
        raise ValueError(f"Unsupported review queue '{queue}'. Expected one of: {allowed}.")
    return aliases[normalized]


def _review_reasons(
    source: SourceItem,
    documents: list[Document],
    card: KnowledgeCard | None,
    *,
    low_confidence_threshold: float,
) -> list[str]:
    reasons: list[str] = []
    if card is None:
        reasons.append("unprocessed")
    if str(source.availability_status) == "unavailable" or any(
        str(document.extraction_status) == "failed" for document in documents
    ):
        reasons.append("failed")
    if str(source.availability_status) == "partial" or any(
        str(document.extraction_status) in {"partial", "pending", "skipped"}
        for document in documents
    ):
        reasons.append("partial")
    if not documents or all(
        not (document.text or document.extracted_text_path) for document in documents
    ):
        reasons.append("missing-source")
    if card is not None:
        if float(card.confidence) <= low_confidence_threshold:
            reasons.append("low-confidence")
        if str(card.freshness_status) in {"recheck-soon", "stale-risk", "stale"}:
            reasons.append("stale")
        if str(card.skill_candidate_status) == "candidate":
            reasons.append("skill-review")
        duplicate_detection = card.provenance.get("duplicate_detection")
        if (
            isinstance(duplicate_detection, dict)
            and duplicate_detection.get("status") == "possible-duplicate"
        ):
            reasons.append("duplicates")
        if str(card.review_status) == "hidden":
            reasons.append("hidden")
    return _dedupe_strings(reasons)


def _review_title(
    source: SourceItem,
    documents: list[Document],
    card: KnowledgeCard | None,
) -> str:
    if card is not None:
        return card.title
    for document in documents:
        if document.title:
            return document.title
    return source.source_url or source.source_external_id or source.id


def _dedupe_strings(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return deduped


def _normalize_terms(terms: Iterable[str]) -> list[tuple[str, str]]:
    normalized: list[tuple[str, str]] = []
    seen: set[str] = set()
    for term in terms:
        name = term.strip()
        if not name:
            continue
        slug = slugify(name)
        if slug in seen:
            continue
        seen.add(slug)
        normalized.append((slug, name))
    return normalized


def _term_tables(term_type: str) -> tuple[str, str, str]:
    if term_type == "topics":
        return "topics", "card_topics", "topic_slug"
    if term_type == "tags":
        return "tags", "card_tags", "tag_slug"
    raise ValueError(f"Unsupported term type: {term_type}")


def _fts_query(query: str) -> str | None:
    terms = re.findall(r"\w+", query, flags=re.UNICODE)
    if not terms:
        return None
    return " OR ".join(f'"{term}"' for term in terms)
