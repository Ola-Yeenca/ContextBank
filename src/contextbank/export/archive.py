from __future__ import annotations

import json
import re
import sqlite3
import tomllib
from pathlib import Path
from typing import Any

from contextbank.compilation.cards import render_card_markdown
from contextbank.config.settings import initialize_settings
from contextbank.ids import slugify
from contextbank.models import Document, KnowledgeCard, SourceItem
from contextbank.paths import secure_file, secure_mkdir, secure_write_text
from contextbank.storage.db import current_schema_version
from contextbank.storage.repository import ContextBankRepository

SECRET_KEY_PARTS = (
    "api_key",
    "apikey",
    "access_token",
    "refresh_token",
    "bearer",
    "client_secret",
    "consumer_secret",
    "password",
    "secret",
    "token",
)
SECRET_VALUE_PATTERNS = (
    re.compile(
        r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----",
        re.DOTALL,
    ),
    re.compile(r"(?<![A-Za-z0-9])sk-proj-[A-Za-z0-9_-]{16,}"),
    re.compile(r"(?<![A-Za-z0-9])sk-[A-Za-z0-9_-]{16,}"),
    re.compile(r"(?<![A-Za-z0-9])xox[baprs]-[A-Za-z0-9-]{16,}"),
    re.compile(r"(?<![A-Za-z0-9])gh[pousr]_[A-Za-z0-9_]{16,}"),
    re.compile(r"\bBearer\s+[A-Za-z0-9._~+/=-]{16,}\b", re.IGNORECASE),
    re.compile(r"\beyJ[A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]{10,}\b"),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
)
RAW_METADATA_KEYS = {"raw", "api_response", "payload"}
REDACTED = "[REDACTED]"
BACKUP_NULL_COLUMNS = {"raw_path", "extracted_text_path", "html_snapshot_path"}
BACKUP_ID_COLUMNS = {
    "id",
    "source_item_id",
    "document_id",
    "card_id",
    "candidate_card_id",
    "match_card_id",
    "project_id",
    "knowledge_card_id",
    "original_source_item_id",
}


def export_jsonl(
    destination: str | Path | None = None,
    *,
    repo: ContextBankRepository | None = None,
    home: Path | str | None = None,
) -> Path:
    owned_repo = repo is None
    settings = initialize_settings(home)
    repo = repo or ContextBankRepository.open(settings.database_path)
    destination_path = (
        Path(destination) if destination else settings.paths.exports / "contextbank.jsonl"
    )
    secure_mkdir(destination_path.parent)
    try:
        with destination_path.open("w", encoding="utf-8") as output:
            for card in repo.list_knowledge_cards(limit=100000):
                source = repo.get_source_item(card.source_item_id)
                documents = repo.list_documents_for_source(card.source_item_id)
                row: dict[str, Any] = {
                    "card": _export_card(card),
                    "source_item": _export_source_item(source),
                    "documents": [_export_document(document) for document in documents],
                    "duplicate_relationships": _strip_secret_keys(
                        repo.list_duplicate_relationships(card_id=card.id)
                    ),
                }
                output.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        secure_file(destination_path)
        return destination_path
    finally:
        if owned_repo:
            repo.close()


def export_markdown(
    destination: str | Path | None = None,
    *,
    home: Path | str | None = None,
) -> Path:
    settings = initialize_settings(home)
    destination_path = Path(destination) if destination else settings.paths.exports / "markdown"
    _ensure_empty_destination(destination_path)
    secure_mkdir(destination_path.parent)
    _write_sanitized_markdown_cards(
        destination_path,
        repo=ContextBankRepository.open(settings.database_path),
        source_cards_path=settings.paths.cards,
    )
    return destination_path


def create_backup(
    destination: str | Path | None = None,
    *,
    home: Path | str | None = None,
) -> Path:
    settings = initialize_settings(home)
    destination_path = Path(destination) if destination else settings.paths.exports / "backup"
    secure_mkdir(destination_path)
    if settings.database_path.exists():
        database_destination = destination_path / settings.database_path.name
        _backup_sqlite_database(
            settings.database_path,
            database_destination,
        )
        _sanitize_backup_database(database_destination)
    if settings.paths.config_file.exists():
        _write_sanitized_config_backup(settings.paths.config_file, destination_path / "config.toml")
    cards_destination = destination_path / "cards"
    _ensure_empty_destination(cards_destination)
    _write_sanitized_markdown_cards(
        cards_destination,
        repo=ContextBankRepository.open(settings.database_path),
        source_cards_path=settings.paths.cards,
    )
    return destination_path


def verify_backup(
    backup_path: str | Path,
    *,
    expected_schema_version: int | None = None,
) -> dict[str, Any]:
    path = Path(backup_path).expanduser().resolve()
    checks: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    database_path = path / "contextbank.db"
    config_path = path / "config.toml"
    cards_path = path / "cards"

    checks.append(_backup_check("backup_directory", path.is_dir(), f"Backup directory: {path}"))
    checks.append(
        _backup_check("database_present", database_path.is_file(), "contextbank.db is present.")
    )
    if database_path.is_file():
        db_checks, counts = _verify_backup_database(
            database_path,
            expected_schema_version=expected_schema_version,
        )
        checks.extend(db_checks)
    checks.append(_backup_check("config_present", config_path.is_file(), "config.toml is present."))
    if config_path.is_file():
        checks.extend(_verify_backup_config(config_path))
    checks.append(
        _backup_check("cards_present", cards_path.is_dir(), "cards/ directory is present.")
    )
    if cards_path.is_dir():
        markdown_check, markdown_count = _verify_markdown_cards(cards_path)
        counts["markdown_cards"] = markdown_count
        checks.append(markdown_check)
    return {
        "backup_path": str(path),
        "valid": all(check["passed"] for check in checks),
        "checks": checks,
        "counts": counts,
    }


def _ensure_empty_destination(destination: Path) -> None:
    if not destination.exists():
        return
    if not destination.is_dir():
        raise FileExistsError(f"Destination exists and is not a directory: {destination}")
    if any(destination.iterdir()):
        raise FileExistsError(
            f"Destination directory is not empty; refusing to delete existing files: {destination}"
        )


def _backup_sqlite_database(source: Path, destination: Path) -> None:
    secure_mkdir(destination.parent)
    source_connection = sqlite3.connect(f"file:{source}?mode=ro", uri=True)
    destination_connection = sqlite3.connect(destination)
    try:
        source_connection.backup(destination_connection)
    finally:
        destination_connection.close()
        source_connection.close()
    secure_file(destination)


def _sanitize_backup_database(database_path: Path) -> None:
    connection = sqlite3.connect(database_path)
    connection.row_factory = sqlite3.Row
    try:
        connection.execute("PRAGMA secure_delete = ON")
        tables = [
            row["name"]
            for row in connection.execute(
                """
                SELECT name FROM sqlite_schema
                WHERE type = 'table'
                AND name NOT LIKE 'sqlite_%'
                AND name NOT LIKE 'card_fts_%'
                ORDER BY name
                """
            ).fetchall()
        ]
        for table in tables:
            if table == "card_fts":
                continue
            _sanitize_backup_table(connection, table)
        _rebuild_backup_fts(connection)
        connection.commit()
        connection.execute("VACUUM")
    finally:
        connection.close()
    secure_file(database_path)


def _sanitize_backup_table(connection: sqlite3.Connection, table: str) -> None:
    columns = [
        row
        for row in connection.execute(f"PRAGMA table_info({_quote_identifier(table)})").fetchall()
        if str(row["type"]).upper() == "TEXT"
    ]
    if not columns:
        return
    column_names = [str(row["name"]) for row in columns]
    select_columns = ", ".join(_quote_identifier(column) for column in column_names)
    rows = connection.execute(
        f"SELECT rowid AS _rowid, {select_columns} FROM {_quote_identifier(table)}"
    ).fetchall()
    for row in rows:
        updates: dict[str, Any] = {}
        for column in column_names:
            if column in BACKUP_ID_COLUMNS:
                continue
            value = row[column]
            if column in BACKUP_NULL_COLUMNS:
                sanitized = None
            elif column.endswith("_json"):
                sanitized = _sanitize_json_text(value)
            elif _looks_like_secret_key(column):
                sanitized = None
            else:
                sanitized = _strip_secret_keys(value)
            if sanitized != value:
                updates[column] = sanitized
        if not updates:
            continue
        set_clause = ", ".join(f"{_quote_identifier(column)} = ?" for column in updates)
        connection.execute(
            f"UPDATE {_quote_identifier(table)} SET {set_clause} WHERE rowid = ?",
            (*updates.values(), row["_rowid"]),
        )


def _sanitize_json_text(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return _redact_secret_text(value)
    sanitized = _strip_secret_keys(parsed)
    return json.dumps(sanitized, ensure_ascii=False, sort_keys=True)


def _rebuild_backup_fts(connection: sqlite3.Connection) -> None:
    exists = connection.execute(
        "SELECT 1 FROM sqlite_schema WHERE type = 'table' AND name = 'card_fts'"
    ).fetchone()
    if exists is None:
        return
    connection.execute("DELETE FROM card_fts")
    rows = connection.execute(
        """
        SELECT
            id,
            title,
            one_line_summary,
            summary,
            why_useful,
            key_insights_json,
            possible_applications_json,
            actionable_next_steps_json,
            implementation_notes_json,
            prerequisites_json,
            patterns_json,
            caveats_json,
            source_quality_notes_json,
            content_type,
            utility_json,
            freshness_reason
        FROM knowledge_cards
        """
    ).fetchall()
    for row in rows:
        body_parts = [
            row["why_useful"],
            *_json_string_list(row["key_insights_json"]),
            *_json_string_list(row["possible_applications_json"]),
            *_json_string_list(row["actionable_next_steps_json"]),
            *_json_string_list(row["implementation_notes_json"]),
            *_json_string_list(row["prerequisites_json"]),
            *_json_string_list(row["patterns_json"]),
            *_json_string_list(row["caveats_json"]),
            *_json_string_list(row["source_quality_notes_json"]),
            row["content_type"],
            *_json_string_list(row["utility_json"]),
            row["freshness_reason"],
        ]
        topics = _backup_terms(connection, "topics", "card_topics", "topic_slug", row["id"])
        tags = _backup_terms(connection, "tags", "card_tags", "tag_slug", row["id"])
        connection.execute(
            """
            INSERT INTO card_fts(card_id, title, one_line_summary, summary, body, topics, tags)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                row["id"],
                row["title"],
                row["one_line_summary"],
                row["summary"],
                "\n".join(part for part in body_parts if part),
                " ".join(topics),
                " ".join(tags),
            ),
        )


def _backup_terms(
    connection: sqlite3.Connection,
    table: str,
    link_table: str,
    slug_column: str,
    card_id: str,
) -> list[str]:
    rows = connection.execute(
        f"""
        SELECT {_quote_identifier(table)}.name
        FROM {_quote_identifier(link_table)}
        JOIN {_quote_identifier(table)}
            ON {_quote_identifier(table)}.slug = {_quote_identifier(link_table)}.{slug_column}
        WHERE {_quote_identifier(link_table)}.card_id = ?
        ORDER BY {_quote_identifier(link_table)}.rowid
        """,
        (card_id,),
    ).fetchall()
    return [str(row["name"]) for row in rows if row["name"]]


def _json_string_list(value: str) -> list[str]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return []
    if not isinstance(parsed, list):
        return []
    return [str(item) for item in parsed if item]


def _quote_identifier(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def _verify_backup_database(
    database_path: Path,
    *,
    expected_schema_version: int | None,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    checks: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    try:
        connection = sqlite3.connect(f"file:{database_path}?mode=ro", uri=True)
        connection.row_factory = sqlite3.Row
    except sqlite3.Error as exc:
        return [
            _backup_check("database_open", False, f"Could not open backup database: {exc}.")
        ], counts
    try:
        integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
        checks.append(
            _backup_check(
                "database_integrity",
                integrity == "ok",
                f"SQLite integrity_check returned: {integrity}.",
            )
        )
        schema_version = current_schema_version(connection)
        checks.append(
            _backup_check(
                "schema_version",
                expected_schema_version is None or schema_version >= expected_schema_version,
                f"SQLite schema version {schema_version}.",
            )
        )
        for table in ("source_items", "documents", "knowledge_cards"):
            counts[table] = int(connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
    except sqlite3.Error as exc:
        checks.append(_backup_check("database_query", False, f"Could not query backup: {exc}."))
    finally:
        connection.close()
    return checks, counts


def _verify_backup_config(config_path: Path) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    try:
        with config_path.open("rb") as input_file:
            config = tomllib.load(input_file)
    except tomllib.TOMLDecodeError as exc:
        return [_backup_check("config_parse", False, f"config.toml is invalid TOML: {exc}.")]
    secret_keys = _secret_key_paths(config)
    secret_values = _secret_value_paths(config)
    checks.append(_backup_check("config_parse", True, "config.toml parses as TOML."))
    checks.append(
        _backup_check(
            "config_no_secret_keys",
            not secret_keys,
            (
                "config.toml contains no secret-like keys."
                if not secret_keys
                else f"Secret-like config keys found: {', '.join(secret_keys)}."
            ),
        )
    )
    checks.append(
        _backup_check(
            "config_no_secret_values",
            not secret_values,
            (
                "config.toml contains no secret-looking scalar values."
                if not secret_values
                else f"Secret-looking config values found: {', '.join(secret_values)}."
            ),
        )
    )
    return checks


def _verify_markdown_cards(cards_path: Path) -> tuple[dict[str, Any], int]:
    markdown_files = list(cards_path.rglob("*.md"))
    unreadable: list[str] = []
    for markdown_file in markdown_files:
        try:
            markdown_file.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            unreadable.append(str(markdown_file))
    return (
        _backup_check(
            "markdown_cards_readable",
            not unreadable,
            (
                f"{len(markdown_files)} Markdown card files found and readable."
                if not unreadable
                else f"Unreadable Markdown card files: {', '.join(unreadable)}."
            ),
        ),
        len(markdown_files),
    )


def _backup_check(name: str, passed: bool, message: str) -> dict[str, Any]:
    return {"name": name, "passed": passed, "message": message}


def _secure_tree(root: Path) -> None:
    secure_mkdir(root)
    for path in root.rglob("*"):
        if path.is_dir():
            secure_mkdir(path)
        elif path.is_file():
            secure_file(path)


def _write_sanitized_config_backup(source: Path, destination: Path) -> None:
    with source.open("rb") as input_file:
        config = tomllib.load(input_file)
    sanitized = _strip_secret_keys(config)
    secure_write_text(destination, _toml_dumps(sanitized))


def _export_card(card: KnowledgeCard) -> dict[str, Any]:
    return _strip_secret_keys(card.model_dump(mode="json"))


def _export_source_item(source: SourceItem | None) -> dict[str, Any] | None:
    if source is None:
        return None
    data = source.model_dump(mode="json")
    data["raw_path"] = None
    data["raw_available"] = bool(source.raw_path)
    return _strip_secret_keys(data)


def _export_document(document: Document) -> dict[str, Any]:
    data = document.model_dump(mode="json")
    data["extracted_text_path"] = None
    data["html_snapshot_path"] = None
    data["local_paths_available"] = bool(
        document.extracted_text_path or document.html_snapshot_path
    )
    return _strip_secret_keys(data, omit_raw_metadata=True)


def _write_sanitized_markdown_cards(
    destination: Path,
    *,
    repo: ContextBankRepository,
    source_cards_path: Path,
) -> None:
    try:
        secure_mkdir(destination)
        cards = repo.list_knowledge_cards(limit=100000)
        if not cards and source_cards_path.exists():
            _copy_sanitized_markdown_tree(source_cards_path, destination)
            return
        for card in cards:
            source = repo.get_source_item(card.source_item_id)
            documents = repo.list_documents_for_source(card.source_item_id)
            document = documents[0] if documents else None
            sanitized_card = KnowledgeCard.model_validate(_export_card(card))
            sanitized_source = (
                SourceItem.model_validate(_export_source_item(source) or {}) if source else None
            )
            sanitized_document = (
                Document.model_validate(_export_document(document)) if document else None
            )
            topic = slugify(sanitized_card.topics[0] if sanitized_card.topics else "uncategorized")
            secure_write_text(
                destination / topic / f"{sanitized_card.id}.md",
                render_card_markdown(
                    sanitized_card,
                    source_item=sanitized_source,
                    document=sanitized_document,
                ),
            )
    finally:
        repo.close()
    _secure_tree(destination)


def _copy_sanitized_markdown_tree(source: Path, destination: Path) -> None:
    for path in source.rglob("*"):
        relative = path.relative_to(source)
        target = destination / relative
        if path.is_dir():
            secure_mkdir(target)
        elif path.is_file():
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            secure_write_text(target, _redact_secret_text(text))


def _strip_secret_keys(value: Any, *, omit_raw_metadata: bool = False) -> Any:
    if isinstance(value, dict):
        return {
            key: _strip_secret_keys(child, omit_raw_metadata=omit_raw_metadata)
            for key, child in value.items()
            if not _looks_like_secret_key(str(key))
            and not (omit_raw_metadata and str(key).lower().replace("-", "_") in RAW_METADATA_KEYS)
        }
    if isinstance(value, list):
        return [_strip_secret_keys(item, omit_raw_metadata=omit_raw_metadata) for item in value]
    if isinstance(value, str):
        return _redact_secret_text(value)
    return value


def _secret_key_paths(value: Any, *, prefix: str = "") -> list[str]:
    if not isinstance(value, dict):
        return []
    matches: list[str] = []
    for key, child in value.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        if _looks_like_secret_key(str(key)):
            matches.append(path)
            continue
        matches.extend(_secret_key_paths(child, prefix=path))
    return matches


def _secret_value_paths(value: Any, *, prefix: str = "") -> list[str]:
    if isinstance(value, dict):
        matches: list[str] = []
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            matches.extend(_secret_value_paths(child, prefix=path))
        return matches
    if isinstance(value, list):
        matches: list[str] = []
        for index, child in enumerate(value):
            path = f"{prefix}[{index}]"
            matches.extend(_secret_value_paths(child, prefix=path))
        return matches
    if _looks_like_secret_value(value):
        return [prefix or "<root>"]
    return []


def _looks_like_secret_key(key: str) -> bool:
    lowered = key.lower().replace("-", "_")
    return any(part in lowered for part in SECRET_KEY_PARTS)


def _looks_like_secret_value(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    stripped = value.strip()
    if not stripped:
        return False
    return any(pattern.search(stripped) for pattern in SECRET_VALUE_PATTERNS)


def _redact_secret_text(value: str) -> str:
    redacted = value
    for pattern in SECRET_VALUE_PATTERNS:
        redacted = pattern.sub(REDACTED, redacted)
    return redacted


def _toml_dumps(data: dict[str, Any], *, _path: tuple[str, ...] = ()) -> str:
    scalars = {key: value for key, value in data.items() if not isinstance(value, dict)}
    tables = {key: value for key, value in data.items() if isinstance(value, dict)}
    lines: list[str] = []
    for key, value in scalars.items():
        lines.append(f"{_toml_key(key)} = {_toml_value(value)}")
    for table, values in tables.items():
        if lines:
            lines.append("")
        next_path = (*_path, str(table))
        lines.append(f"[{'.'.join(_toml_key(part) for part in next_path)}]")
        lines.append(_toml_dumps(values, _path=next_path).rstrip())
    return "\n".join(lines) + "\n"


def _toml_key(key: Any) -> str:
    text = str(key)
    if re.fullmatch(r"[A-Za-z0-9_-]+", text):
        return text
    return json.dumps(text)


def _toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int | float):
        return str(value)
    if isinstance(value, list):
        return "[" + ", ".join(_toml_value(item) for item in value) + "]"
    return json.dumps(str(value))
