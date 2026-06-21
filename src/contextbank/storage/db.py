from __future__ import annotations

import sqlite3
from collections.abc import Iterable
from datetime import UTC
from pathlib import Path

from contextbank.models import utc_now
from contextbank.paths import secure_file, secure_mkdir

SCHEMA_VERSION = 8


class DatabaseInitializationError(RuntimeError):
    """Raised when SQLite cannot initialize the required local schema."""


MIGRATIONS: tuple[tuple[int, str], ...] = (
    (
        1,
        """
        CREATE TABLE IF NOT EXISTS source_items (
            id TEXT PRIMARY KEY,
            source_type TEXT NOT NULL,
            source_external_id TEXT,
            source_url TEXT,
            author_id TEXT,
            author_name TEXT,
            author_handle TEXT,
            source_created_at TEXT,
            first_captured_at TEXT NOT NULL,
            last_synced_at TEXT,
            raw_path TEXT,
            content_hash TEXT,
            language TEXT,
            availability_status TEXT NOT NULL,
            deleted_locally_at TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );

        CREATE UNIQUE INDEX IF NOT EXISTS idx_source_items_external
            ON source_items(source_type, source_external_id)
            WHERE source_external_id IS NOT NULL;
        CREATE INDEX IF NOT EXISTS idx_source_items_url ON source_items(source_url);
        CREATE INDEX IF NOT EXISTS idx_source_items_type ON source_items(source_type);

        CREATE TABLE IF NOT EXISTS documents (
            id TEXT PRIMARY KEY,
            source_item_id TEXT NOT NULL REFERENCES source_items(id) ON DELETE CASCADE,
            document_type TEXT NOT NULL,
            canonical_url TEXT,
            title TEXT,
            author TEXT,
            published_at TEXT,
            extracted_text_path TEXT,
            html_snapshot_path TEXT,
            extraction_status TEXT NOT NULL,
            extraction_error TEXT,
            extractor_version TEXT NOT NULL,
            content_hash TEXT,
            text TEXT,
            metadata_json TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_documents_source_item_id
            ON documents(source_item_id);
        CREATE INDEX IF NOT EXISTS idx_documents_type ON documents(document_type);

        CREATE TABLE IF NOT EXISTS knowledge_cards (
            id TEXT PRIMARY KEY,
            source_item_id TEXT NOT NULL REFERENCES source_items(id) ON DELETE CASCADE,
            schema_version TEXT NOT NULL,
            title TEXT NOT NULL,
            one_line_summary TEXT NOT NULL,
            summary TEXT NOT NULL,
            key_insights_json TEXT NOT NULL,
            why_useful TEXT NOT NULL,
            possible_applications_json TEXT NOT NULL,
            actionable_next_steps_json TEXT NOT NULL,
            implementation_notes_json TEXT NOT NULL,
            prerequisites_json TEXT NOT NULL,
            patterns_json TEXT NOT NULL,
            caveats_json TEXT NOT NULL,
            source_quality_notes_json TEXT NOT NULL,
            content_type TEXT NOT NULL,
            utility_json TEXT NOT NULL,
            personal_priority TEXT NOT NULL,
            freshness_status TEXT NOT NULL,
            freshness_reason TEXT NOT NULL,
            confidence REAL NOT NULL,
            skill_candidate_status TEXT NOT NULL,
            review_status TEXT NOT NULL,
            generated_at TEXT NOT NULL,
            model_provider TEXT NOT NULL,
            model_id TEXT NOT NULL,
            prompt_version TEXT NOT NULL,
            provenance_json TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_knowledge_cards_source_item_id
            ON knowledge_cards(source_item_id);
        CREATE INDEX IF NOT EXISTS idx_knowledge_cards_content_type
            ON knowledge_cards(content_type);
        CREATE INDEX IF NOT EXISTS idx_knowledge_cards_freshness
            ON knowledge_cards(freshness_status);
        CREATE INDEX IF NOT EXISTS idx_knowledge_cards_review
            ON knowledge_cards(review_status);
        CREATE INDEX IF NOT EXISTS idx_knowledge_cards_confidence
            ON knowledge_cards(confidence);

        CREATE TABLE IF NOT EXISTS topics (
            slug TEXT PRIMARY KEY,
            name TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS tags (
            slug TEXT PRIMARY KEY,
            name TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS card_topics (
            card_id TEXT NOT NULL REFERENCES knowledge_cards(id) ON DELETE CASCADE,
            topic_slug TEXT NOT NULL REFERENCES topics(slug) ON DELETE CASCADE,
            PRIMARY KEY (card_id, topic_slug)
        );

        CREATE TABLE IF NOT EXISTS card_tags (
            card_id TEXT NOT NULL REFERENCES knowledge_cards(id) ON DELETE CASCADE,
            tag_slug TEXT NOT NULL REFERENCES tags(slug) ON DELETE CASCADE,
            PRIMARY KEY (card_id, tag_slug)
        );

        CREATE INDEX IF NOT EXISTS idx_card_topics_topic_slug
            ON card_topics(topic_slug);
        CREATE INDEX IF NOT EXISTS idx_card_tags_tag_slug
            ON card_tags(tag_slug);

        CREATE VIRTUAL TABLE IF NOT EXISTS card_fts USING fts5(
            card_id UNINDEXED,
            title,
            one_line_summary,
            summary,
            body,
            topics,
            tags,
            tokenize = 'porter unicode61'
        );
        """,
    ),
    (
        2,
        """
        CREATE TABLE IF NOT EXISTS sync_state (
            connector TEXT NOT NULL,
            account_id TEXT NOT NULL,
            cursor TEXT,
            status TEXT NOT NULL,
            last_started_at TEXT,
            last_completed_at TEXT,
            last_error TEXT,
            items_synced INTEGER NOT NULL DEFAULT 0,
            full_sync_completed INTEGER NOT NULL DEFAULT 0,
            updated_at TEXT NOT NULL,
            PRIMARY KEY (connector, account_id)
        );

        CREATE INDEX IF NOT EXISTS idx_sync_state_connector
            ON sync_state(connector);
        """,
    ),
    (
        3,
        """
        CREATE TABLE IF NOT EXISTS projects (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            root_path TEXT NOT NULL UNIQUE,
            profile_json TEXT NOT NULL,
            created_at TEXT NOT NULL,
            last_seen_at TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_projects_name ON projects(name);

        CREATE TABLE IF NOT EXISTS project_item_links (
            project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
            knowledge_card_id TEXT NOT NULL REFERENCES knowledge_cards(id) ON DELETE CASCADE,
            relevance REAL NOT NULL DEFAULT 0,
            reason TEXT NOT NULL,
            intended_use TEXT NOT NULL,
            status TEXT NOT NULL,
            outcome_note TEXT NOT NULL,
            created_by TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            PRIMARY KEY (project_id, knowledge_card_id)
        );

        CREATE INDEX IF NOT EXISTS idx_project_item_links_card
            ON project_item_links(knowledge_card_id);
        CREATE INDEX IF NOT EXISTS idx_project_item_links_status
            ON project_item_links(status);
        """,
    ),
    (
        4,
        """
        CREATE TABLE IF NOT EXISTS card_embeddings (
            card_id TEXT NOT NULL REFERENCES knowledge_cards(id) ON DELETE CASCADE,
            provider TEXT NOT NULL,
            model_id TEXT NOT NULL,
            vector_json TEXT NOT NULL,
            source_hash TEXT NOT NULL,
            indexed_at TEXT NOT NULL,
            PRIMARY KEY (card_id, provider, model_id)
        );

        CREATE INDEX IF NOT EXISTS idx_card_embeddings_provider
            ON card_embeddings(provider, model_id);
        """,
    ),
    (
        5,
        """
        CREATE TABLE IF NOT EXISTS source_lifecycle_events (
            id TEXT PRIMARY KEY,
            source_item_id TEXT NOT NULL REFERENCES source_items(id) ON DELETE CASCADE,
            event_type TEXT NOT NULL,
            reason TEXT NOT NULL,
            payload_json TEXT NOT NULL,
            created_at TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_source_lifecycle_events_source
            ON source_lifecycle_events(source_item_id, created_at);
        CREATE INDEX IF NOT EXISTS idx_source_lifecycle_events_type
            ON source_lifecycle_events(event_type);

        CREATE TABLE IF NOT EXISTS detached_notes (
            id TEXT PRIMARY KEY,
            original_source_item_id TEXT NOT NULL,
            source_type TEXT NOT NULL,
            source_url TEXT,
            title TEXT,
            user_notes_json TEXT NOT NULL,
            detached_at TEXT NOT NULL,
            reason TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_detached_notes_original_source
            ON detached_notes(original_source_item_id);
        """,
    ),
    (
        6,
        """
        CREATE INDEX IF NOT EXISTS idx_source_items_content_hash
            ON source_items(content_hash)
            WHERE content_hash IS NOT NULL;
        CREATE INDEX IF NOT EXISTS idx_documents_content_hash
            ON documents(content_hash)
            WHERE content_hash IS NOT NULL;
        CREATE INDEX IF NOT EXISTS idx_documents_canonical_url
            ON documents(canonical_url)
            WHERE canonical_url IS NOT NULL;
        """,
    ),
    (
        7,
        """
        CREATE TABLE IF NOT EXISTS duplicate_relationships (
            candidate_card_id TEXT NOT NULL REFERENCES knowledge_cards(id) ON DELETE CASCADE,
            match_card_id TEXT NOT NULL REFERENCES knowledge_cards(id) ON DELETE CASCADE,
            status TEXT NOT NULL,
            reason TEXT NOT NULL,
            created_by TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            PRIMARY KEY (candidate_card_id, match_card_id),
            CHECK (candidate_card_id != match_card_id)
        );

        CREATE INDEX IF NOT EXISTS idx_duplicate_relationships_match
            ON duplicate_relationships(match_card_id);
        CREATE INDEX IF NOT EXISTS idx_duplicate_relationships_status
            ON duplicate_relationships(status);
        """,
    ),
    (
        8,
        """
        DROP TABLE IF EXISTS card_fts;

        CREATE VIRTUAL TABLE IF NOT EXISTS card_fts USING fts5(
            card_id UNINDEXED,
            title,
            one_line_summary,
            summary,
            body,
            topics,
            tags,
            document_text,
            source_authors,
            tokenize = 'porter unicode61'
        );

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
        SELECT
            knowledge_cards.id,
            knowledge_cards.title,
            knowledge_cards.one_line_summary,
            knowledge_cards.summary,
            COALESCE(knowledge_cards.why_useful, '') || char(10) ||
                COALESCE(knowledge_cards.key_insights_json, '') || char(10) ||
                COALESCE(knowledge_cards.possible_applications_json, '') || char(10) ||
                COALESCE(knowledge_cards.actionable_next_steps_json, '') || char(10) ||
                COALESCE(knowledge_cards.implementation_notes_json, '') || char(10) ||
                COALESCE(knowledge_cards.prerequisites_json, '') || char(10) ||
                COALESCE(knowledge_cards.patterns_json, '') || char(10) ||
                COALESCE(knowledge_cards.caveats_json, '') || char(10) ||
                COALESCE(knowledge_cards.source_quality_notes_json, '') || char(10) ||
                COALESCE(knowledge_cards.content_type, '') || char(10) ||
                COALESCE(knowledge_cards.utility_json, '') || char(10) ||
                COALESCE(knowledge_cards.freshness_reason, ''),
            COALESCE((
                SELECT group_concat(topics.name, ' ')
                FROM card_topics
                JOIN topics ON topics.slug = card_topics.topic_slug
                WHERE card_topics.card_id = knowledge_cards.id
            ), ''),
            COALESCE((
                SELECT group_concat(tags.name, ' ')
                FROM card_tags
                JOIN tags ON tags.slug = card_tags.tag_slug
                WHERE card_tags.card_id = knowledge_cards.id
            ), ''),
            COALESCE((
                SELECT group_concat(documents.text, char(10))
                FROM documents
                WHERE documents.source_item_id = knowledge_cards.source_item_id
                    AND documents.text IS NOT NULL
            ), ''),
            COALESCE(source_items.author_name, '') || ' ' ||
                COALESCE(source_items.author_handle, '') || ' ' ||
                COALESCE(source_items.author_id, '') || ' ' ||
                COALESCE((
                    SELECT group_concat(documents.author, ' ')
                    FROM documents
                    WHERE documents.source_item_id = knowledge_cards.source_item_id
                        AND documents.author IS NOT NULL
                ), '')
        FROM knowledge_cards
        LEFT JOIN source_items ON source_items.id = knowledge_cards.source_item_id;
        """,
    ),
)


def connect_database(database: Path | str) -> sqlite3.Connection:
    database_path = Path(database).expanduser() if str(database) != ":memory:" else Path(":memory:")
    if str(database_path) != ":memory:":
        secure_mkdir(database_path.parent)

    connection = sqlite3.connect(database_path)
    connection.row_factory = sqlite3.Row
    configure_connection(connection, database_path)
    if str(database_path) != ":memory:":
        secure_file(database_path)
    return connection


def configure_connection(
    connection: sqlite3.Connection,
    database: Path | str | None = None,
) -> None:
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA busy_timeout = 5000")
    if database is not None and str(database) != ":memory:":
        connection.execute("PRAGMA journal_mode = WAL")


def initialize_database(database: Path | str) -> sqlite3.Connection:
    connection = connect_database(database)
    migrate(connection)
    return connection


def migrate(
    connection: sqlite3.Connection,
    migrations: Iterable[tuple[int, str]] = MIGRATIONS,
) -> None:
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS schema_migrations (
            version INTEGER PRIMARY KEY,
            applied_at TEXT NOT NULL
        )
        """
    )
    applied = {
        int(row["version"])
        for row in connection.execute("SELECT version FROM schema_migrations").fetchall()
    }

    for version, sql in sorted(migrations, key=lambda item: item[0]):
        if version in applied:
            continue
        try:
            with connection:
                connection.executescript(sql)
                connection.execute(
                    "INSERT OR IGNORE INTO schema_migrations(version, applied_at) VALUES (?, ?)",
                    (version, utc_now().astimezone(UTC).isoformat()),
                )
        except sqlite3.Error as exc:
            raise DatabaseInitializationError(
                f"Failed to apply ContextBank database migration {version}"
            ) from exc


def current_schema_version(connection: sqlite3.Connection) -> int:
    row = connection.execute("SELECT MAX(version) AS version FROM schema_migrations").fetchone()
    if row is None or row["version"] is None:
        return 0
    return int(row["version"])
