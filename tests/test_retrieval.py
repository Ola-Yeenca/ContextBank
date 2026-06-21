from __future__ import annotations

import json
import sqlite3

import pytest

import contextbank.retrieval.search as retrieval_search
from contextbank.config.settings import initialize_settings
from contextbank.export.archive import create_backup, export_jsonl, export_markdown, verify_backup
from contextbank.projects.profile import (
    configure_project_autoload,
    init_project_profile,
    inspect_project,
    load_project_profile,
)
from contextbank.retrieval.search import (
    build_context_pack,
    build_project_autoload_context,
    find_related_items,
    find_skill_candidates,
    get_knowledge_item,
    get_source_document,
    rebuild_semantic_index,
    search_knowledge,
)
from contextbank.storage.repository import ContextBankRepository
from tests.helpers import seed_card


def test_search_get_context_and_source_document(tmp_path):
    repo = ContextBankRepository.from_home(tmp_path)
    source, document, card = seed_card(repo)

    results = search_knowledge("SQLite retrieval", repo=repo, include_source_excerpts=True)
    assert results[0].card.id == card.id
    assert results[0].source_url == source.source_url

    item = get_knowledge_item(card.id, repo=repo)
    assert item is not None
    assert item["card"]["id"] == card.id
    assert item["documents"][0]["id"] == document.id

    source_doc = get_source_document(document.id, repo=repo, max_chars=20)
    assert source_doc is not None
    assert source_doc["truncated"] is True

    pack = build_context_pack("Use SQLite retrieval for MCP", repo=repo, token_budget=300)
    assert pack.items
    assert pack.source_references == [source.source_url]
    assert pack.source_claims[0]["card_id"] == card.id
    assert pack.source_claims[0]["source_url"] == source.source_url
    assert pack.actionable_suggestions[0]["card_id"] == card.id
    assert pack.actionable_suggestions[0]["source_url"] == source.source_url


def test_skill_candidates_related_export_backup_and_project_profile(tmp_path):
    home = tmp_path / "home"
    repo = ContextBankRepository.from_home(home)
    source, document, first = seed_card(repo, url="https://example.com/a", tags=["sqlite", "mcp"])
    export_secret = "sk-proj-" + "exportsecret0123456789"
    source = source.model_copy(
        update={
            "source_external_id": f"external-{export_secret}",
            "source_url": f"https://example.com/a?token={export_secret}",
            "author_name": f"Leaky Author {export_secret}",
            "author_handle": f"leaky_{export_secret}",
        }
    )
    repo.upsert_source_item(source)
    repo.upsert_document(
        document.model_copy(
            update={
                "extracted_text_path": str(home / "documents" / "secret.txt"),
                "text": f"Captured page text includes {export_secret}.",
                "metadata": {
                    "safe_label": "keep me",
                    "api_key": "secret-key",
                    "raw": {"access_token": "secret-token"},
                },
            }
        )
    )
    first = first.model_copy(
        update={
            "title": f"Local-first retrieval {export_secret}",
            "summary": (
                "SQLite FTS retrieval keeps local agent context searchable: "
                f"{export_secret}."
            ),
            "key_insights": [f"SQLite FTS can answer local searches with {export_secret}"],
            "provenance": {
                **first.provenance,
                "model_generation": {"api_key": export_secret},
                "note": f"Generated from source containing {export_secret}",
            },
        }
    )
    repo.upsert_knowledge_card(first)
    seed_card(
        repo,
        url="https://example.com/b",
        title="MCP skill checklist",
        tags=["sqlite", "mcp", "skill"],
        skill_candidate_status="candidate",
    )

    candidates = find_skill_candidates(repo=repo)
    assert len(candidates) == 1
    related = find_related_items(first.id, repo=repo)
    assert related

    jsonl_path = export_jsonl(tmp_path / "export.jsonl", repo=repo, home=home)
    jsonl_text = jsonl_path.read_text(encoding="utf-8")
    assert jsonl_text.count("\n") == 2
    exported_rows = [json.loads(line) for line in jsonl_text.splitlines()]
    exported_first = next(
        row for row in exported_rows if row["card"]["source_item_id"] == source.id
    )
    assert exported_first["source_item"]["raw_path"] is None
    assert exported_first["documents"][0]["extracted_text_path"] is None
    assert exported_first["documents"][0]["metadata"] == {"safe_label": "keep me"}
    assert "[REDACTED]" in jsonl_text
    assert export_secret not in jsonl_text
    assert "secret-key" not in jsonl_text
    assert "secret-token" not in jsonl_text

    markdown_path = export_markdown(tmp_path / "markdown-export", home=home)
    assert markdown_path.exists()
    markdown_text = "\n".join(
        path.read_text(encoding="utf-8") for path in markdown_path.rglob("*.md")
    )
    assert "[REDACTED]" in markdown_text
    assert export_secret not in markdown_text

    non_empty_destination = tmp_path / "existing"
    non_empty_destination.mkdir()
    (non_empty_destination / "keep.txt").write_text("do not delete", encoding="utf-8")
    with pytest.raises(FileExistsError):
        export_markdown(non_empty_destination, home=home)
    assert (non_empty_destination / "keep.txt").read_text(encoding="utf-8") == "do not delete"

    config_file = home / "config.toml"
    config_file.write_text(
        config_file.read_text(encoding="utf-8").replace(
            'user_id = ""',
            'user_id = "123"\naccess_token = "secret-token"\napi_key = "secret-key"',
        ),
        encoding="utf-8",
    )
    backup_path = create_backup(tmp_path / "backup", home=home)
    assert (backup_path / "contextbank.db").exists()
    backup_database_text = (backup_path / "contextbank.db").read_bytes().decode(
        "utf-8",
        errors="ignore",
    )
    assert export_secret not in backup_database_text
    with sqlite3.connect(backup_path / "contextbank.db") as backup_connection:
        document_text = backup_connection.execute(
            "SELECT text FROM documents WHERE id = ?",
            (document.id,),
        ).fetchone()[0]
        source_url = backup_connection.execute(
            "SELECT source_url FROM source_items WHERE id = ?",
            (source.id,),
        ).fetchone()[0]
        card_summary = backup_connection.execute(
            "SELECT summary FROM knowledge_cards WHERE id = ?",
            (first.id,),
        ).fetchone()[0]
    assert "[REDACTED]" in document_text
    assert "[REDACTED]" in source_url
    assert "[REDACTED]" in card_summary
    backup_markdown_text = "\n".join(
        path.read_text(encoding="utf-8") for path in (backup_path / "cards").rglob("*.md")
    )
    assert "[REDACTED]" in backup_markdown_text
    assert export_secret not in backup_markdown_text
    backed_up_config = (backup_path / "config.toml").read_text(encoding="utf-8")
    assert 'user_id = "123"' in backed_up_config
    assert "secret-token" not in backed_up_config
    assert "secret-key" not in backed_up_config
    assert "access_token" not in backed_up_config
    assert "api_key" not in backed_up_config
    verification = verify_backup(backup_path)
    assert verification["valid"] is True
    assert verification["counts"]["knowledge_cards"] == 2

    corrupt_markdown = backup_path / "cards" / "corrupt.md"
    corrupt_markdown.write_bytes(b"\xff")
    unreadable_verification = verify_backup(backup_path)
    assert unreadable_verification["valid"] is False
    assert any(
        check["name"] == "markdown_cards_readable" and check["passed"] is False
        for check in unreadable_verification["checks"]
    )
    corrupt_markdown.unlink()

    (backup_path / "config.toml").write_text(
        backed_up_config + '\naccess_token = "manual-secret"\n',
        encoding="utf-8",
    )
    failed_verification = verify_backup(backup_path)
    assert failed_verification["valid"] is False
    assert any(
        check["name"] == "config_no_secret_keys" and check["passed"] is False
        for check in failed_verification["checks"]
    )

    project_root = tmp_path / "project"
    project_root.mkdir()
    profile_file = init_project_profile(project_root, name="ContextBank", topics=["retrieval"])
    assert profile_file.exists()
    assert load_project_profile(project_root).name == "ContextBank"
    inspected = inspect_project(project_root)
    assert inspected["exists"] is True
    assert inspected["autoload"]["enabled"] is True

    configure_project_autoload(
        project_root,
        enabled=False,
        token_budget=900,
        output_type="decision",
    )
    updated_profile = load_project_profile(project_root)
    assert updated_profile.autoload.enabled is False
    assert updated_profile.autoload.token_budget == 900
    assert updated_profile.autoload.output_type == "decision"


def test_project_autoload_context_uses_profile_and_project_signals(tmp_path):
    home = tmp_path / "home"
    repo = ContextBankRepository.from_home(home)
    try:
        project_root = tmp_path / "project"
        project_root.mkdir()
        (project_root / "pyproject.toml").write_text(
            "[project]\nname = 'demo'\ndependencies = ['typer>=0.12']\n",
            encoding="utf-8",
        )
        (project_root / "README.md").write_text("# Context loading\n\n## Agent startup\n")
        init_project_profile(
            project_root,
            name="ContextBank",
            goals=["Build local-first retrieval"],
            stack=["Python", "SQLite"],
            topics=["retrieval"],
        )
        _, _, card = seed_card(
            repo,
            title="SQLite retrieval startup context",
            summary="Use SQLite FTS to retrieve relevant saved context for Python projects.",
            tags=["sqlite", "python", "retrieval"],
        )

        pack = build_project_autoload_context(
            project_root,
            task="Improve retrieval",
            repo=repo,
            token_budget=300,
        )

        assert pack.items[0].card.id == card.id
        assert pack.output_type == "implementation"
        assert "Project context included" in pack.synthesis
        assert pack.project_context["detected_signals"]["dependencies"] == ["typer"]
        assert "Context loading" in pack.project_context["detected_signals"]["readme_headings"]
        assert pack.retrieval_metadata["network_calls"] == 0
        assert pack.retrieval_metadata["cloud_calls"] == 0
    finally:
        repo.close()


def test_project_autoload_can_be_disabled_and_forced(tmp_path):
    home = tmp_path / "home"
    repo = ContextBankRepository.from_home(home)
    try:
        project_root = tmp_path / "project"
        project_root.mkdir()
        init_project_profile(
            project_root,
            name="ContextBank",
            stack=["Python", "SQLite"],
            autoload_enabled=False,
            autoload_token_budget=700,
            autoload_output_type="decision",
        )
        _, _, card = seed_card(
            repo,
            title="SQLite retrieval startup context",
            summary="Use SQLite retrieval for disabled autoload force checks.",
            tags=["sqlite", "retrieval"],
        )

        disabled = build_project_autoload_context(
            project_root,
            task="SQLite retrieval",
            repo=repo,
        )
        forced = build_project_autoload_context(
            project_root,
            task="SQLite retrieval",
            repo=repo,
            force=True,
        )

        assert disabled.items == []
        assert disabled.output_type == "decision"
        assert disabled.retrieval_metadata["autoload_disabled"] is True
        assert forced.items[0].card.id == card.id
    finally:
        repo.close()


def test_project_autoload_prioritizes_explicit_project_links(tmp_path):
    home = tmp_path / "home"
    repo = ContextBankRepository.from_home(home)
    try:
        project_root = tmp_path / "project"
        project_root.mkdir()
        init_project_profile(
            project_root,
            name="ContextBank",
            goals=["Build local-first retrieval"],
            stack=["Python", "SQLite"],
        )
        _, _, linked = seed_card(
            repo,
            title="Color palette archive",
            summary="Keep visual inspiration notes for later review.",
            tags=["design"],
        )
        _, _, lexical = seed_card(
            repo,
            url="https://example.com/lexical",
            title="Python retrieval implementation",
            summary="Use Python and SQLite retrieval for project startup context.",
            tags=["python", "sqlite", "retrieval"],
        )
        _, _, rejected = seed_card(
            repo,
            url="https://example.com/rejected",
            title="Rejected project note",
            summary="A user rejected this project note for startup retrieval.",
            tags=["project-note"],
        )
        _, _, obsolete = seed_card(
            repo,
            url="https://example.com/obsolete",
            title="Obsolete project note",
            summary="A user marked this project note obsolete for startup retrieval.",
            tags=["project-note"],
        )
        repo.link_card_to_project(
            card_id=linked.id,
            root_path=project_root,
            name="ContextBank",
            profile={"name": "ContextBank"},
            relevance=0.95,
            reason="Pinned as useful context for this exact repository.",
            intended_use="implementation",
            status="used",
        )
        for excluded_card, status in ((rejected, "rejected"), (obsolete, "obsolete")):
            repo.link_card_to_project(
                card_id=excluded_card.id,
                root_path=project_root,
                name="ContextBank",
                profile={"name": "ContextBank"},
                relevance=1.0,
                reason=f"Do not use this {status} context.",
                intended_use="implementation",
                status=status,
            )

        pack = build_project_autoload_context(
            project_root,
            task="Improve Python retrieval",
            repo=repo,
            token_budget=800,
        )

        assert pack.items[0].card.id == linked.id
        assert "Explicitly linked to this project" in pack.items[0].relevance_reason
        assert lexical.id in {item.card.id for item in pack.items}
        assert rejected.id not in {item.card.id for item in pack.items}
        assert obsolete.id not in {item.card.id for item in pack.items}
    finally:
        repo.close()


def test_project_autoload_keeps_explicit_links_after_profile_rename(tmp_path):
    home = tmp_path / "home"
    repo = ContextBankRepository.from_home(home)
    try:
        project_root = tmp_path / "project"
        project_root.mkdir()
        init_project_profile(project_root, name="Old Name", stack=["Python"])
        _, _, linked = seed_card(
            repo,
            title="Persistent project link",
            summary="Use root-path project links even after the project profile is renamed.",
            tags=["project-link"],
        )
        repo.link_card_to_project(
            card_id=linked.id,
            root_path=project_root,
            name="Old Name",
            profile={"name": "Old Name"},
            relevance=1.0,
            reason="Still useful after the profile display name changes.",
            intended_use="implementation",
            status="used",
        )
        init_project_profile(project_root, name="New Name", stack=["Python"], overwrite=True)

        pack = build_project_autoload_context(
            project_root,
            task="Use persistent project links",
            repo=repo,
            token_budget=800,
        )

        assert linked.id in {item.card.id for item in pack.items}
    finally:
        repo.close()


def test_project_autoload_does_not_cross_load_same_named_project_links(tmp_path):
    home = tmp_path / "home"
    repo = ContextBankRepository.from_home(home)
    try:
        first_project = tmp_path / "first"
        second_project = tmp_path / "second"
        first_project.mkdir()
        second_project.mkdir()
        init_project_profile(first_project, name="Demo", stack=["Python"])
        init_project_profile(second_project, name="Demo", stack=["Python"])
        _, _, first_only = seed_card(
            repo,
            title="First repository visual archive",
            summary="Only explicitly linked to the first repository.",
            tags=["visual-archive"],
        )
        repo.link_card_to_project(
            card_id=first_only.id,
            root_path=first_project,
            name="Demo",
            profile={"name": "Demo"},
            relevance=1.0,
            reason="Only useful for the first repository.",
            intended_use="reference",
            status="used",
        )

        pack = build_project_autoload_context(
            second_project,
            task="Improve Python packaging",
            repo=repo,
            token_budget=800,
        )

        assert first_only.id not in {item.card.id for item in pack.items}
    finally:
        repo.close()


def test_agent_retrieval_excludes_rejected_cards_by_default(tmp_path):
    repo = ContextBankRepository.from_home(tmp_path)
    try:
        _, _, accepted = seed_card(
            repo,
            url="https://example.com/accepted",
            title="Accepted retrieval pattern",
            summary="Use accepted SQLite retrieval guidance.",
            tags=["sqlite", "retrieval"],
        )
        _, _, rejected = seed_card(
            repo,
            url="https://example.com/rejected-review",
            title="Rejected retrieval pattern",
            summary="Use rejected SQLite retrieval guidance.",
            tags=["sqlite", "retrieval"],
            review_status="rejected",
        )

        default_results = search_knowledge("SQLite retrieval", repo=repo)
        explicit_results = search_knowledge(
            "SQLite retrieval",
            repo=repo,
            filters={"review_status": "rejected"},
        )

        assert accepted.id in {result.card.id for result in default_results}
        assert rejected.id not in {result.card.id for result in default_results}
        assert explicit_results[0].card.id == rejected.id
    finally:
        repo.close()


def test_agent_retrieval_sql_limit_counts_default_visible_cards(tmp_path):
    repo = ContextBankRepository.from_home(tmp_path)
    try:
        _, _, visible = seed_card(
            repo,
            url="https://example.com/visible-limit",
            title="Visible limit-safe retrieval",
            summary="This older visible card must survive a small SQL limit.",
        )
        seed_card(
            repo,
            url="https://example.com/hidden-limit",
            title="Hidden limit pressure",
            summary="This hidden card is newer but should not consume the default limit.",
            review_status="hidden",
        )
        seed_card(
            repo,
            url="https://example.com/rejected-limit",
            title="Rejected limit pressure",
            summary="This rejected card is newer but should not consume the default limit.",
            review_status="rejected",
        )

        default_results = search_knowledge("", repo=repo, max_results=1)
        explicit_hidden = search_knowledge(
            "",
            repo=repo,
            filters={"review_status": "hidden"},
            max_results=1,
        )

        assert [result.card.id for result in default_results] == [visible.id]
        assert explicit_hidden[0].card.review_status == "hidden"
    finally:
        repo.close()


def test_project_autoload_respects_excluded_topics(tmp_path):
    home = tmp_path / "home"
    repo = ContextBankRepository.from_home(home)
    try:
        project_root = tmp_path / "project"
        project_root.mkdir()
        init_project_profile(
            project_root,
            name="Demo",
            stack=["Python"],
            exclude_topics=["entertainment"],
        )
        _, _, useful = seed_card(
            repo,
            url="https://example.com/useful",
            title="Python retrieval guide",
            summary="Use Python retrieval for local agent startup.",
            topics=["software-development"],
            tags=["python", "retrieval"],
        )
        _, _, excluded = seed_card(
            repo,
            url="https://example.com/excluded-topic",
            title="Entertainment retrieval tangent",
            summary="Entertainment retrieval note that should not autoload.",
            topics=["entertainment"],
            tags=["python", "retrieval"],
        )

        pack = build_project_autoload_context(
            project_root,
            task="Python retrieval",
            repo=repo,
        )

        assert useful.id in {item.card.id for item in pack.items}
        assert excluded.id not in {item.card.id for item in pack.items}
    finally:
        repo.close()


def test_context_pack_surfaces_shared_caveat_conflicts(tmp_path):
    repo = ContextBankRepository.from_home(tmp_path)
    try:
        _, _, first = seed_card(
            repo,
            url="https://example.com/first-conflict",
            title="First migration guidance",
            summary="Use SQLite migration guidance for local context packs.",
            tags=["sqlite", "migration"],
        )
        _, _, second = seed_card(
            repo,
            url="https://example.com/second-conflict",
            title="Second migration guidance",
            summary="Use SQLite migration guidance with version checks.",
            tags=["sqlite", "migration"],
        )
        repo.upsert_knowledge_card(
            first.model_copy(update={"caveats": ["Risk: migration requires backup first."]})
        )
        repo.upsert_knowledge_card(
            second.model_copy(update={"caveats": ["Risk: migration needs rollback testing."]})
        )

        pack = build_context_pack("SQLite migration guidance", repo=repo, token_budget=800)

        assert pack.conflicts
        assert pack.conflicts[0]["topic"] == "risk"
        assert set(pack.conflicts[0]["card_ids"]) == {first.id, second.id}
    finally:
        repo.close()


def test_context_pack_reports_upstream_search_budget_truncation(tmp_path):
    repo = ContextBankRepository.from_home(tmp_path)
    try:
        large_summary = "budget truncation signal " + ("context " * 260)
        seed_card(
            repo,
            url="https://example.com/budget-first",
            title="Budget truncation first",
            summary=large_summary,
            tags=["budget"],
        )
        seed_card(
            repo,
            url="https://example.com/budget-second",
            title="Budget truncation second",
            summary=large_summary,
            tags=["budget"],
        )

        pack = build_context_pack("budget truncation", repo=repo, token_budget=250)

        assert len(pack.items) == 1
        assert pack.truncated is True
    finally:
        repo.close()


def test_semantic_index_rebuild_supports_local_search_without_cloud(tmp_path):
    repo = ContextBankRepository.from_home(tmp_path)
    try:
        _, _, card = seed_card(
            repo,
            title="Durable vector retrieval",
            summary="Semantic lookup can use local hash embeddings for context packs.",
            tags=["semantic", "embeddings"],
        )

        result = rebuild_semantic_index(repo=repo)
        semantic_results = search_knowledge(
            "vector embeddings",
            repo=repo,
            use_semantic=True,
        )

        assert result["provider"] == "local-hash"
        assert result["cloud_calls"] == 0
        assert repo.count_rows("card_embeddings") == 1
        assert semantic_results[0].card.id == card.id
        assert "semantic" in semantic_results[0].relevance_reason
    finally:
        repo.close()


def test_semantic_search_uses_configured_embedding_provider_for_query(
    tmp_path,
    monkeypatch,
):
    home = tmp_path / "home"
    settings = initialize_settings(home)
    settings.paths.config_file.write_text(
        settings.paths.config_file.read_text(encoding="utf-8").replace(
            'embedding_provider = "none"\nembedding_model = ""',
            'embedding_provider = "openai"\nembedding_model = "fake-embedding-model"',
        ),
        encoding="utf-8",
    )
    repo = ContextBankRepository.from_home(home)
    seen: dict[str, object] = {}

    class FakeConfiguredProvider:
        name = "openai"
        model_id = "fake-embedding-model"

        def embed(self, texts: list[str]) -> list[list[float]]:
            seen["texts"] = texts
            return [[1.0, 0.0]]

        def close(self) -> None:
            seen["closed"] = True

    def fake_embedding_provider(provider_name: str, *, settings=None):
        seen["provider_name"] = provider_name
        seen["settings_home"] = settings.home if settings else None
        return FakeConfiguredProvider()

    monkeypatch.setattr(retrieval_search, "_embedding_provider", fake_embedding_provider)

    try:
        _, _, card = seed_card(
            repo,
            title="Configured vector result",
            summary="This card should be found only through the configured vector index.",
            tags=["configured-embedding"],
        )
        repo.upsert_card_embedding(
            card_id=card.id,
            provider="openai",
            model_id="fake-embedding-model",
            vector=[1.0, 0.0],
            source_hash="test-hash",
        )

        semantic_results = search_knowledge(
            "unrelated query text",
            repo=repo,
            home=home,
            use_semantic=True,
        )

        assert seen["provider_name"] == "configured"
        assert seen["settings_home"] == home
        assert seen["texts"] == ["unrelated query text"]
        assert seen["closed"] is True
        assert semantic_results[0].card.id == card.id
    finally:
        repo.close()
