from __future__ import annotations

import json
import tomllib
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from typer.testing import CliRunner

import contextbank.cli.app as cli_app
import contextbank.retrieval.search as retrieval_search
from contextbank.cli.app import app
from contextbank.config.settings import initialize_settings
from contextbank.connectors.text.importer import import_text
from contextbank.connectors.x.oauth import StoredXToken, save_stored_token
from contextbank.ids import content_hash, document_id, source_item_id
from contextbank.mcp.server import tool_registry
from contextbank.models import (
    AvailabilityStatus,
    Document,
    ExtractionStatus,
    SourceItem,
    SourceType,
)
from contextbank.providers import GenerationResponse
from contextbank.storage.db import SCHEMA_VERSION
from contextbank.storage.repository import ContextBankRepository
from tests.helpers import seed_card

runner = CliRunner()


def test_cli_init_import_search_context_and_exports(tmp_path, monkeypatch):
    monkeypatch.setenv("CONTEXTBANK_HOME", str(tmp_path / "home"))
    sample = tmp_path / "sample.md"
    sample.write_text(
        "# Local-first MCP retrieval\n\nUse SQLite FTS to build bounded context packs.",
        encoding="utf-8",
    )

    init_result = runner.invoke(app, ["init", "--json"])
    assert init_result.exit_code == 0, init_result.output
    assert json.loads(init_result.output)["schema_version"] == SCHEMA_VERSION

    import_result = runner.invoke(app, ["import", str(sample), "--json"])
    assert import_result.exit_code == 0, import_result.output
    imported = json.loads(import_result.output)
    assert imported["card_path"].endswith(".md")

    search_result = runner.invoke(app, ["search", "SQLite retrieval", "--json"])
    assert search_result.exit_code == 0, search_result.output
    assert "Local-first MCP retrieval" in search_result.output

    context_result = runner.invoke(app, ["context", "build retrieval", "--json"])
    assert context_result.exit_code == 0, context_result.output
    assert json.loads(context_result.output)["items"]

    jsonl_result = runner.invoke(app, ["export", "jsonl"])
    assert jsonl_result.exit_code == 0, jsonl_result.output

    backup_result = runner.invoke(app, ["backup", "create"])
    assert backup_result.exit_code == 0, backup_result.output
    verify_result = runner.invoke(app, ["backup", "verify", backup_result.output.strip(), "--json"])
    assert verify_result.exit_code == 0, verify_result.output
    assert json.loads(verify_result.output)["valid"] is True

    doctor_result = runner.invoke(app, ["doctor", "--json"])
    assert doctor_result.exit_code == 0, doctor_result.output
    assert json.loads(doctor_result.output)["offline_search"] is True


def test_cli_config_project_and_x_dry_run(tmp_path, monkeypatch):
    monkeypatch.setenv("CONTEXTBANK_HOME", str(tmp_path / "home"))
    runner.invoke(app, ["init"])

    set_result = runner.invoke(app, ["config", "set", "x.user_id", "123"])
    assert set_result.exit_code == 0, set_result.output

    status_result = runner.invoke(app, ["auth", "status", "--json"])
    assert status_result.exit_code == 0, status_result.output
    assert json.loads(status_result.output)["x"]["user_id"] == "123"

    dry_run = runner.invoke(app, ["sync", "x", "--dry-run", "--limit", "25", "--json"])
    assert dry_run.exit_code == 0, dry_run.output
    assert json.loads(dry_run.output)["estimated_resources"] == 25

    standard_dry_run = runner.invoke(
        app,
        ["sync", "x", "--dry-run", "--limit", "1", "--cost-basis", "standard", "--json"],
    )
    assert standard_dry_run.exit_code == 0, standard_dry_run.output
    assert json.loads(standard_dry_run.output)["estimated_cost_usd"] == 0.005

    project_root = tmp_path / "project"
    project_root.mkdir()
    project_result = runner.invoke(app, ["project", "init", str(project_root), "--name", "Demo"])
    assert project_result.exit_code == 0, project_result.output

    inspect_result = runner.invoke(app, ["project", "inspect", str(project_root), "--json"])
    assert inspect_result.exit_code == 0, inspect_result.output
    assert json.loads(inspect_result.output)["profile"]["name"] == "Demo"


def test_cli_source_list_shows_inventory_counts_and_filtered_items(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setenv("CONTEXTBANK_HOME", str(home))
    runner.invoke(app, ["init"])
    source_file = tmp_path / "note.md"
    source_file.write_text("# Local note\n\nUseful local source.", encoding="utf-8")

    imported = runner.invoke(app, ["import", str(source_file), "--json"])
    added_x = runner.invoke(app, ["add", "https://x.com/example_user/status/1234567890", "--json"])
    listed = runner.invoke(app, ["source", "list", "--json"])
    listed_x = runner.invoke(app, ["source", "list", "--type", "x", "--json"])
    counts_only = runner.invoke(app, ["source", "list", "--counts-only", "--json"])

    assert imported.exit_code == 0, imported.output
    assert added_x.exit_code == 0, added_x.output
    assert listed.exit_code == 0, listed.output
    payload = json.loads(listed.output)
    assert payload["total"] == 2
    assert payload["by_type"] == {"file": 1, "x": 1}
    assert any(item["raw_available"] for item in payload["items"])
    assert all("raw_path" not in item for item in payload["items"])

    filtered = json.loads(listed_x.output)
    assert filtered["filters"]["source_type"] == "x"
    assert [item["source_type"] for item in filtered["items"]] == ["x"]
    assert filtered["items"][0]["author_handle"] == "example_user"

    assert json.loads(counts_only.output)["items"] == []


def test_x_payment_required_error_message_points_to_credits():
    message = cli_app._x_error_message(
        cli_app.XApiError("X API request failed with HTTP 402.", status_code=402)
    )

    assert "HTTP 402" in message
    assert "payment required" in message
    assert "X API credits" in message


def test_cli_x_sync_without_token_fails_before_writing_sync_state(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setenv("CONTEXTBANK_HOME", str(home))
    monkeypatch.delenv("CONTEXTBANK_X_BEARER_TOKEN", raising=False)
    runner.invoke(app, ["init"])

    result = runner.invoke(app, ["sync", "x", "--limit", "1", "--json"])

    assert result.exit_code != 0
    assert "X sync requires a stored OAuth token" in result.output
    repo = ContextBankRepository.from_home(home)
    try:
        assert repo.list_sync_states() == []
    finally:
        repo.close()


def test_cli_project_autoload_builds_startup_context_pack(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setenv("CONTEXTBANK_HOME", str(home))
    runner.invoke(app, ["init"])
    project_root = tmp_path / "project"
    project_root.mkdir()
    (project_root / "pyproject.toml").write_text("[project]\nname = 'demo'\n", encoding="utf-8")
    runner.invoke(app, ["project", "init", str(project_root), "--name", "Demo"])

    repo = ContextBankRepository.from_home(home)
    try:
        _, _, card = seed_card(
            repo,
            title="SQLite retrieval pattern",
            summary="Use SQLite FTS for local project retrieval in Python tools.",
            topics=["software-development"],
            tags=["python", "sqlite", "retrieval"],
        )
    finally:
        repo.close()

    autoload = runner.invoke(
        app,
        [
            "project",
            "autoload",
            str(project_root),
            "--task",
            "Improve Python retrieval",
            "--json",
        ],
    )

    assert autoload.exit_code == 0, autoload.output
    payload = json.loads(autoload.output)
    assert payload["output_type"] == "implementation"
    assert payload["items"][0]["card"]["id"] == card.id
    assert payload["items"][0]["card"]["title"] == (
        "[source-derived text is available in source_packet]"
    )
    assert "SQLite retrieval pattern" in payload["items"][0]["source_packet"]
    assert "Synthesis:" in payload["synthesis"]
    assert payload["project_context"]["autoload"]["enabled"] is True
    assert payload["retrieval_metadata"]["network_calls"] == 0
    assert payload["retrieval_metadata"]["cloud_calls"] == 0

    raw_autoload = runner.invoke(
        app,
        [
            "project",
            "autoload",
            str(project_root),
            "--task",
            "Improve Python retrieval",
            "--json",
            "--raw",
        ],
    )

    assert raw_autoload.exit_code == 0, raw_autoload.output
    raw_payload = json.loads(raw_autoload.output)
    assert raw_payload["items"][0]["card"]["title"] == "SQLite retrieval pattern"


def test_cli_project_autoload_config_toggles_startup_loading(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setenv("CONTEXTBANK_HOME", str(home))
    runner.invoke(app, ["init"])
    project_root = tmp_path / "project"
    project_root.mkdir()
    init_result = runner.invoke(
        app,
        [
            "project",
            "init",
            str(project_root),
            "--name",
            "Demo",
            "--autoload-token-budget",
            "900",
            "--autoload-output-type",
            "decision",
        ],
    )
    assert init_result.exit_code == 0, init_result.output

    disabled = runner.invoke(
        app,
        [
            "project",
            "autoload-config",
            str(project_root),
            "--disabled",
            "--token-budget",
            "800",
            "--output-type",
            "research",
            "--json",
        ],
    )
    assert disabled.exit_code == 0, disabled.output
    disabled_payload = json.loads(disabled.output)
    assert disabled_payload["autoload"]["enabled"] is False
    assert disabled_payload["autoload"]["token_budget"] == 800
    assert disabled_payload["autoload"]["output_type"] == "research"

    repo = ContextBankRepository.from_home(home)
    try:
        seed_card(
            repo,
            title="Research autoload card",
            summary="Use this research context for local project startup.",
            tags=["research", "autoload"],
        )
    finally:
        repo.close()

    autoload = runner.invoke(
        app,
        [
            "project",
            "autoload",
            str(project_root),
            "--task",
            "research autoload",
            "--json",
        ],
    )
    assert autoload.exit_code == 0, autoload.output
    autoload_payload = json.loads(autoload.output)
    assert autoload_payload["items"] == []
    assert autoload_payload["output_type"] == "research"
    assert autoload_payload["retrieval_metadata"]["autoload_disabled"] is True


def test_cli_and_mcp_project_autoload_do_not_touch_network_or_cloud(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setenv("CONTEXTBANK_HOME", str(home))
    monkeypatch.setenv("CONTEXTBANK_X_BEARER_TOKEN", "secret-x-token")
    monkeypatch.setenv("OPENAI_API_KEY", "secret-openai-key")
    runner.invoke(app, ["init"])
    project_root = tmp_path / "project"
    project_root.mkdir()
    runner.invoke(app, ["project", "init", str(project_root), "--name", "Demo"])

    repo = ContextBankRepository.from_home(home)
    try:
        _, _, card = seed_card(
            repo,
            title="Local autoload boundary",
            summary="Autoload should read local SQLite context without network or cloud calls.",
            tags=["autoload", "local-first"],
        )
    finally:
        repo.close()

    def fail_external(*args, **kwargs):
        raise AssertionError("autoload must not call network connectors or cloud providers")

    monkeypatch.setattr(cli_app, "ingest_url", fail_external)
    monkeypatch.setattr(cli_app, "import_local_file", fail_external)
    monkeypatch.setattr(cli_app, "XBookmarkClient", fail_external)
    monkeypatch.setattr(retrieval_search, "_embedding_provider", fail_external)

    cli_result = runner.invoke(
        app,
        [
            "project",
            "autoload",
            str(project_root),
            "--task",
            "local autoload boundary",
            "--json",
        ],
    )
    assert cli_result.exit_code == 0, cli_result.output
    assert "secret-x-token" not in cli_result.output
    assert "secret-openai-key" not in cli_result.output
    cli_payload = json.loads(cli_result.output)
    assert cli_payload["items"][0]["card"]["id"] == card.id
    assert cli_payload["retrieval_metadata"]["network_calls"] == 0
    assert cli_payload["retrieval_metadata"]["cloud_calls"] == 0

    mcp_payload = tool_registry(home)["autoload_project_context"](
        str(project_root),
        "local autoload boundary",
        500,
        "implementation",
        None,
    )
    assert mcp_payload["items"][0]["card"]["id"] == card.id
    assert mcp_payload["retrieval_metadata"]["network_calls"] == 0
    assert mcp_payload["retrieval_metadata"]["cloud_calls"] == 0


def test_cli_readiness_reports_agent_and_x_gates_without_secrets(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setenv("CONTEXTBANK_HOME", str(home))
    monkeypatch.setenv("CONTEXTBANK_X_CLIENT_ID", "client-ready")
    monkeypatch.setenv("CONTEXTBANK_X_BEARER_TOKEN", "secret-x-ready-token")
    monkeypatch.setenv("OPENAI_API_KEY", "secret-openai-ready-key")
    runner.invoke(app, ["init"])
    project_root = tmp_path / "project"
    project_root.mkdir()
    project_init = runner.invoke(app, ["project", "init", str(project_root), "--name", "Demo"])
    assert project_init.exit_code == 0, project_init.output

    repo = ContextBankRepository.from_home(home)
    try:
        _, _, card = seed_card(
            repo,
            title="Readiness local agent context",
            summary="Readiness checks should prove MCP autoload is local and source-linked.",
            tags=["readiness", "autoload"],
        )
    finally:
        repo.close()

    ready = runner.invoke(
        app,
        [
            "readiness",
            "--project",
            str(project_root),
            "--task",
            "readiness autoload",
            "--json",
        ],
    )
    assert ready.exit_code == 0, ready.output
    assert "secret-x-ready-token" not in ready.output
    assert "secret-openai-ready-key" not in ready.output
    payload = json.loads(ready.output)
    assert payload["ready_for_local_agent"] is True
    assert payload["ready_for_x_sync"] is False
    assert payload["mcp"]["env"] == {"CONTEXTBANK_HOME": str(home)}
    assert payload["mcp"]["secrets_included"] is False
    assert payload["mcp"]["runtime_available"] is True
    assert payload["mcp"]["runtime_error"] is None
    assert payload["autoload"]["items"] >= 1
    assert payload["autoload"]["network_calls"] == 0
    assert payload["autoload"]["cloud_calls"] == 0
    assert payload["autoload"]["startup_instruction_available"] is True
    instruction_status = {
        status["client"]: status for status in payload["autoload"]["instruction_files"]
    }
    assert instruction_status["codex"]["managed_block_installed"] is False
    assert instruction_status["generic"]["managed_block_installed"] is False
    assert payload["x"]["client"]["ready_for_auth"] is True
    assert payload["x"]["token"]["env_bearer_token_configured"] is True
    assert payload["x"]["token"]["ready_for_sync"] is False
    assert payload["x"]["token"]["sync_status"] == "env-token-present-scopes-unverified"
    actions = {action["id"]: action for action in payload["next_actions"]}
    assert actions["validate_env_x_token"]["command"] == "contextbank sync x --dry-run --limit 25"
    assert actions["validate_env_x_token"]["required_for"] == ["x_sync"]
    assert card.id in ready.output

    write_instructions = runner.invoke(
        app,
        ["mcp", "instructions", "--client", "codex", "--project", str(project_root), "--write"],
    )
    ready_with_instructions = runner.invoke(
        app,
        [
            "readiness",
            "--project",
            str(project_root),
            "--task",
            "readiness autoload",
            "--json",
        ],
    )
    assert write_instructions.exit_code == 0, write_instructions.output
    assert ready_with_instructions.exit_code == 0, ready_with_instructions.output
    installed_status = {
        status["client"]: status
        for status in json.loads(ready_with_instructions.output)["autoload"][
            "instruction_files"
        ]
    }
    assert installed_status["codex"]["managed_block_installed"] is True

    disabled = runner.invoke(
        app,
        ["project", "autoload-config", str(project_root), "--disabled", "--json"],
    )
    assert disabled.exit_code == 0, disabled.output
    not_ready = runner.invoke(
        app,
        ["readiness", "--project", str(project_root), "--json"],
    )
    assert not_ready.exit_code == 0, not_ready.output
    not_ready_payload = json.loads(not_ready.output)
    assert not_ready_payload["ready_for_local_agent"] is False
    assert not_ready_payload["autoload"]["disabled"] is True
    not_ready_actions = {action["id"]: action for action in not_ready_payload["next_actions"]}
    assert "enable_project_autoload" in not_ready_actions
    assert str(project_root) in not_ready_actions["enable_project_autoload"]["command"]
    autoload_check = next(
        check for check in not_ready_payload["checks"] if check["name"] == "autoload_enabled"
    )
    assert autoload_check["passed"] is False


def test_cli_readiness_reports_stale_agent_instruction_defaults(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setenv("CONTEXTBANK_HOME", str(home))
    runner.invoke(app, ["init"])
    project_root = tmp_path / "project"
    project_root.mkdir()
    runner.invoke(app, ["project", "init", str(project_root), "--name", "Demo"])

    configure = runner.invoke(
        app,
        [
            "project",
            "autoload-config",
            str(project_root),
            "--token-budget",
            "900",
            "--output-type",
            "implementation",
            "--json",
        ],
    )
    write_instructions = runner.invoke(
        app,
        ["mcp", "instructions", "--client", "all", "--project", str(project_root), "--write"],
    )
    current = runner.invoke(app, ["readiness", "--project", str(project_root), "--json"])

    assert configure.exit_code == 0, configure.output
    assert write_instructions.exit_code == 0, write_instructions.output
    assert current.exit_code == 0, current.output
    current_status = {
        status["client"]: status
        for status in json.loads(current.output)["autoload"]["instruction_files"]
    }
    assert all(status["managed_block_installed"] for status in current_status.values())
    assert all(status["matches_project_defaults"] for status in current_status.values())
    assert {status["expected_token_budget"] for status in current_status.values()} == {900}

    changed = runner.invoke(
        app,
        [
            "project",
            "autoload-config",
            str(project_root),
            "--token-budget",
            "1200",
            "--json",
        ],
    )
    stale = runner.invoke(app, ["readiness", "--project", str(project_root), "--json"])

    assert changed.exit_code == 0, changed.output
    assert stale.exit_code == 0, stale.output
    stale_payload = json.loads(stale.output)
    stale_status = {
        status["client"]: status
        for status in stale_payload["autoload"]["instruction_files"]
    }
    assert all(status["managed_block_installed"] for status in stale_status.values())
    assert all(status["stale"] for status in stale_status.values())
    assert not any(status["matches_project_defaults"] for status in stale_status.values())
    assert {status["expected_token_budget"] for status in stale_status.values()} == {1200}
    actions = {action["id"]: action for action in stale_payload["next_actions"]}
    assert actions["install_agent_autoload_instructions"]["clients"] == [
        "codex",
        "claude-code",
        "cursor",
        "generic",
    ]


def test_cli_readiness_requires_mcp_runtime(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setenv("CONTEXTBANK_HOME", str(home))
    runner.invoke(app, ["init"])

    def missing_mcp_runtime(home):
        raise RuntimeError("Install ContextBank with the 'mcp' extra to run the MCP server.")

    monkeypatch.setattr(cli_app, "create_server", missing_mcp_runtime)

    result = runner.invoke(app, ["readiness", "--project", str(tmp_path), "--json"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["ready_for_local_agent"] is False
    assert payload["mcp"]["runtime_available"] is False
    assert "mcp" in payload["mcp"]["runtime_error"]
    actions = {action["id"]: action for action in payload["next_actions"]}
    assert actions["configure_x_oauth_client"]["required_for"] == ["x_sync"]
    assert "x.client_id" in actions["configure_x_oauth_client"]["command"]
    runtime_check = next(check for check in payload["checks"] if check["name"] == "mcp_runtime")
    assert runtime_check["passed"] is False


def test_cli_project_link_persists_link_and_refreshes_markdown(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setenv("CONTEXTBANK_HOME", str(home))
    runner.invoke(app, ["init"])
    project_root = tmp_path / "project"
    project_root.mkdir()
    runner.invoke(app, ["project", "init", str(project_root), "--name", "Demo"])

    repo = ContextBankRepository.from_home(home)
    try:
        _, _, card = seed_card(
            repo,
            title="Project-aware retrieval",
            summary="Use project links to retrieve relevant context for active work.",
            tags=["retrieval", "project"],
        )
    finally:
        repo.close()

    linked = runner.invoke(
        app,
        [
            "project",
            "link",
            card.id,
            "--project",
            str(project_root),
            "--relevance",
            "0.9",
            "--reason",
            "Use this for startup autoload.",
            "--intended-use",
            "implementation",
            "--status",
            "used",
            "--outcome-note",
            "Applied to the retrieval workflow.",
            "--json",
        ],
    )

    assert linked.exit_code == 0, linked.output
    payload = json.loads(linked.output)
    assert payload["project_link"]["project_name"] == "Demo"
    assert payload["project_link"]["status"] == "used"
    assert payload["user_signals"]["last_used_in_project_at"] == (
        payload["project_link"]["updated_at"]
    )
    assert datetime.fromisoformat(payload["project_link"]["updated_at"]).tzinfo is not None
    assert payload["user_signals"]["used_in_projects"] == ["Demo"]
    assert payload["user_signals"]["last_project_outcome_note"] == (
        "Applied to the retrieval workflow."
    )

    repo = ContextBankRepository.from_home(home)
    try:
        links = repo.list_project_links_for_card(card.id)
        updated = repo.get_knowledge_card(card.id)
        assert updated is not None
        assert links[0]["reason"] == "Use this for startup autoload."
        assert updated.provenance["project_links"][0]["project_name"] == "Demo"
        assert updated.provenance["user_signals"]["used_in_projects"] == ["Demo"]
        assert updated.provenance["user_signals"]["last_used_in_project_at"] == (
            payload["project_link"]["updated_at"]
        )
        assert updated.provenance["user_signals"]["last_project_outcome_note"] == (
            "Applied to the retrieval workflow."
        )
    finally:
        repo.close()

    markdown = (home / "cards").glob("*/*.md")
    markdown_text = next(markdown).read_text(encoding="utf-8")
    assert "project_links:" in markdown_text
    assert "Project link: Demo (used) - Use this for startup autoload." in markdown_text

    filtered_search = runner.invoke(
        app,
        ["search", "autoload", "--project", "Demo", "--json"],
    )
    assert filtered_search.exit_code == 0, filtered_search.output
    assert json.loads(filtered_search.output)[0]["card"]["id"] == card.id


def test_cli_project_use_records_usage_outcome_and_refreshes_markdown(
    tmp_path,
    monkeypatch,
):
    home = tmp_path / "home"
    monkeypatch.setenv("CONTEXTBANK_HOME", str(home))
    runner.invoke(app, ["init"])
    project_root = tmp_path / "project"
    project_root.mkdir()
    runner.invoke(app, ["project", "init", str(project_root), "--name", "Demo"])

    repo = ContextBankRepository.from_home(home)
    try:
        _, _, card = seed_card(
            repo,
            title="Autoload usage feedback",
            summary="Used context should be remembered as a positive project signal.",
            tags=["autoload", "feedback"],
        )
    finally:
        repo.close()

    candidate = runner.invoke(
        app,
        [
            "project",
            "link",
            card.id,
            "--project",
            str(project_root),
            "--relevance",
            "0.4",
            "--reason",
            "Candidate context before it is used.",
            "--intended-use",
            "reference",
            "--status",
            "candidate",
            "--json",
        ],
    )
    assert candidate.exit_code == 0, candidate.output

    used = runner.invoke(
        app,
        [
            "project",
            "use",
            card.id,
            "--project",
            str(project_root),
            "--outcome-note",
            "Applied in the autoload flow.",
            "--json",
        ],
    )

    assert used.exit_code == 0, used.output
    payload = json.loads(used.output)
    assert payload["project_link"]["project_name"] == "Demo"
    assert payload["project_link"]["status"] == "used"
    assert payload["project_link"]["relevance"] == 1.0
    assert payload["project_link"]["reason"] == "Used during project work."
    assert payload["project_link"]["intended_use"] == "applied context"
    assert payload["user_signals"]["used_in_projects"] == ["Demo"]
    assert payload["user_signals"]["last_used_in_project_at"] == (
        payload["project_link"]["updated_at"]
    )
    assert datetime.fromisoformat(payload["project_link"]["updated_at"]).tzinfo is not None
    assert payload["user_signals"]["last_project_outcome_note"] == (
        "Applied in the autoload flow."
    )

    repo = ContextBankRepository.from_home(home)
    try:
        links = repo.list_project_links_for_card(card.id)
        updated = repo.get_knowledge_card(card.id)
        assert updated is not None
        assert len(links) == 1
        assert links[0]["status"] == "used"
        assert links[0]["reason"] == "Used during project work."
        assert links[0]["intended_use"] == "applied context"
        assert len(updated.provenance["project_links"]) == 1
        assert updated.provenance["project_links"][0]["status"] == "used"
        assert updated.provenance["project_links"][0]["outcome_note"] == (
            "Applied in the autoload flow."
        )
        assert updated.provenance["user_signals"]["used_in_projects"] == ["Demo"]
        assert updated.provenance["user_signals"]["last_used_in_project_at"] == (
            payload["project_link"]["updated_at"]
        )
    finally:
        repo.close()

    markdown_text = next((home / "cards").glob("*/*.md")).read_text(encoding="utf-8")
    assert "Project link: Demo (used) - Used during project work." in markdown_text
    assert 'status: "used"' in markdown_text
    assert 'reason: "Used during project work."' in markdown_text
    assert 'outcome_note: "Applied in the autoload flow."' in markdown_text
    assert "updated_at:" in markdown_text

    autoload = runner.invoke(
        app,
        [
            "project",
            "autoload",
            str(project_root),
            "--task",
            "autoload flow",
            "--json",
        ],
    )
    assert autoload.exit_code == 0, autoload.output
    autoload_payload = json.loads(autoload.output)
    assert autoload_payload["items"][0]["card"]["id"] == card.id
    assert "Explicitly linked to this project" in autoload_payload["items"][0][
        "relevance_reason"
    ]
    assert autoload_payload["retrieval_metadata"]["network_calls"] == 0
    assert autoload_payload["retrieval_metadata"]["cloud_calls"] == 0
    assert autoload_payload["retrieval_metadata"]["project_links_considered"] == 1


def test_cli_reprocess_preserves_manual_review_signals_and_project_links(
    tmp_path,
    monkeypatch,
):
    home = tmp_path / "home"
    monkeypatch.setenv("CONTEXTBANK_HOME", str(home))
    runner.invoke(app, ["init"])
    project_root = tmp_path / "project"
    project_root.mkdir()
    runner.invoke(app, ["project", "init", str(project_root), "--name", "Demo"])

    repo = ContextBankRepository.from_home(home)
    try:
        _, document, card = seed_card(
            repo,
            title="Reusable project workflow",
            summary="Step 1. Load project context. Step 2. Verify the retrieved evidence.",
            skill_candidate_status="candidate",
        )
        repo.upsert_document(
            document.model_copy(
                update={
                    "text": (
                        "Updated source text says project autoload should preserve reviewed state."
                    )
                }
            )
        )
    finally:
        repo.close()

    linked = runner.invoke(
        app,
        [
            "project",
            "link",
            card.id,
            "--project",
            str(project_root),
            "--reason",
            "Relevant to reprocess preservation.",
            "--status",
            "used",
        ],
    )
    approved = runner.invoke(app, ["review", "approve-skill", card.id])
    marked = runner.invoke(
        app,
        ["mark", card.id, "--priority", "high", "--usefulness", "5", "--pin"],
    )
    reprocessed = runner.invoke(app, ["reprocess", card.id, "--json"])

    assert linked.exit_code == 0, linked.output
    assert approved.exit_code == 0, approved.output
    assert marked.exit_code == 0, marked.output
    assert reprocessed.exit_code == 0, reprocessed.output
    assert json.loads(reprocessed.output)["processed"][0]["item_id"] == card.source_item_id

    repo = ContextBankRepository.from_home(home)
    try:
        updated = repo.get_knowledge_card(card.id)
        assert updated.summary == (
            "Updated source text says project autoload should preserve reviewed state."
        )
        assert updated.personal_priority == "high"
        assert updated.review_status == "approved"
        assert updated.skill_candidate_status == "approved"
        assert updated.provenance["skill_candidate"]["active"] is True
        assert updated.provenance["user_signals"]["pinned"] is True
        assert updated.provenance["user_signals"]["usefulness_rating"] == 5
        assert updated.provenance["project_links"][0]["project_name"] == "Demo"
    finally:
        repo.close()

    markdown_text = next((home / "cards").glob("*/*.md")).read_text(encoding="utf-8")
    assert "review_status: \"approved\"" in markdown_text
    assert "Project link: Demo (used) - Relevant to reprocess preservation." in markdown_text


def test_cli_process_paginates_and_isolates_item_errors(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setenv("CONTEXTBANK_HOME", str(home))
    monkeypatch.setattr(cli_app, "PROCESS_PAGE_SIZE", 2)
    runner.invoke(app, ["init"])

    repo = ContextBankRepository.from_home(home)
    try:
        seed_card(
            repo,
            url="https://example.com/process-a",
            title="Process page A",
        )
        missing_document_source = SourceItem(
            id=source_item_id(SourceType.WEB.value, value="https://example.com/process-b"),
            source_type=SourceType.WEB,
            source_external_id="https://example.com/process-b",
            source_url="https://example.com/process-b",
            availability_status=AvailabilityStatus.AVAILABLE,
        )
        repo.upsert_source_item(missing_document_source)
        seed_card(
            repo,
            url="https://example.com/process-c",
            title="Process page C",
        )
    finally:
        repo.close()

    processed = runner.invoke(app, ["process", "--json"])

    assert processed.exit_code == 0, processed.output
    payload = json.loads(processed.output)
    assert payload["requested"] == 3
    assert payload["processed_count"] == 2
    assert payload["failed"] == 1
    assert payload["pages"] == 2
    assert payload["errors"] == [
        {"item_id": missing_document_source.id, "error": "No documents found."}
    ]


def test_cli_ai_status_is_byok_and_never_prints_key_values(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setenv("CONTEXTBANK_HOME", str(home))
    monkeypatch.setenv("OPENAI_API_KEY", "fake-openai-key-value")
    runner.invoke(app, ["init"])

    set_mode = runner.invoke(app, ["config", "set", "ai.mode", "cloud"])
    set_provider = runner.invoke(app, ["config", "set", "ai.generation_provider", "openai"])
    set_generation_model = runner.invoke(
        app,
        ["config", "set", "ai.generation_model", "gpt-test"],
    )
    set_env = runner.invoke(
        app,
        ["config", "set", "ai.generation_credential_env", "OPENAI_API_KEY"],
    )
    set_embedding_provider = runner.invoke(
        app,
        ["config", "set", "ai.embedding_provider", "openai"],
    )
    set_embedding_model = runner.invoke(
        app,
        ["config", "set", "ai.embedding_model", "text-embedding-3-small"],
    )
    set_embedding_env = runner.invoke(
        app,
        ["config", "set", "ai.embedding_credential_env", "OPENAI_API_KEY"],
    )
    allow_cloud = runner.invoke(app, ["config", "set", "ai.allow_cloud", "true"])

    assert set_mode.exit_code == 0, set_mode.output
    assert set_provider.exit_code == 0, set_provider.output
    assert set_generation_model.exit_code == 0, set_generation_model.output
    assert set_env.exit_code == 0, set_env.output
    assert set_embedding_provider.exit_code == 0, set_embedding_provider.output
    assert set_embedding_model.exit_code == 0, set_embedding_model.output
    assert set_embedding_env.exit_code == 0, set_embedding_env.output
    assert allow_cloud.exit_code == 0, allow_cloud.output

    status = runner.invoke(app, ["ai", "status", "--json"])
    assert status.exit_code == 0, status.output
    payload = json.loads(status.output)
    assert payload["open_source_core"] is True
    assert payload["byok"] is True
    assert payload["bundled_api_keys"] is False
    assert payload["mode"] == "cloud"
    assert payload["available_adapters"]["generation"] == ["openai", "openai-compatible"]
    assert payload["available_adapters"]["embedding"] == [
        "local-hash",
        "openai",
        "openai-compatible",
    ]
    assert payload["generation"]["provider"] == "openai"
    assert payload["generation"]["model"] == "gpt-test"
    assert payload["generation"]["credential_env"] == "OPENAI_API_KEY"
    assert payload["generation"]["credential_present"] is True
    assert payload["generation"]["adapter_implemented"] is True
    assert payload["generation"]["ready"] is True
    assert payload["embedding"]["provider"] == "openai"
    assert payload["embedding"]["model"] == "text-embedding-3-small"
    assert payload["embedding"]["adapter_implemented"] is True
    assert payload["embedding"]["ready"] is True
    assert "fake-openai-key-value" not in status.output

    doctor = runner.invoke(app, ["doctor", "--json"])
    assert doctor.exit_code == 0, doctor.output
    assert json.loads(doctor.output)["ai"]["byok"] is True
    assert "fake-openai-key-value" not in doctor.output


def test_cli_ai_status_lists_available_adapters_when_ai_is_unconfigured(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setenv("CONTEXTBANK_HOME", str(home))
    runner.invoke(app, ["init"])

    status = runner.invoke(app, ["ai", "status", "--json"])

    assert status.exit_code == 0, status.output
    payload = json.loads(status.output)
    assert payload["mode"] == "none"
    assert payload["generation"]["provider"] == "none"
    assert payload["generation"]["adapter_implemented"] is False
    assert payload["embedding"]["provider"] == "none"
    assert payload["embedding"]["adapter_implemented"] is False
    assert payload["available_adapters"]["generation"] == ["openai", "openai-compatible"]
    assert payload["available_adapters"]["embedding"] == [
        "local-hash",
        "openai",
        "openai-compatible",
    ]


def test_cli_ai_generate_card_uses_configured_byok_provider(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setenv("CONTEXTBANK_HOME", str(home))
    runner.invoke(app, ["init"])
    source_file = tmp_path / "source.md"
    source_file.write_text(
        "# Provider source\n\nUse explicit JSON generation for card enrichment.",
        encoding="utf-8",
    )
    imported = runner.invoke(app, ["import", str(source_file), "--json"])
    assert imported.exit_code == 0, imported.output
    item_id = json.loads(imported.output)["item_id"]

    class FakeGenerationProvider:
        name = "fake-openai-compatible"
        model_id = "fake-model"
        base_url = "http://127.0.0.1:11434/v1"

        def __init__(self) -> None:
            self.closed = False

        def generate_structured(self, request):
            assert request.schema_name == "contextbank.knowledge_card.v1"
            assert "CONTEXTBANK_UNTRUSTED_SOURCE_BEGIN" in request.source_text
            return GenerationResponse(
                text=json.dumps(
                    {
                        "title": "Generated Provider Card",
                        "one_line_summary": "Generated card summary.",
                        "summary": "A generated card grounded in the source packet.",
                        "why_useful": "Useful for BYOK generation testing.",
                        "key_insights": ["Validated generated insight"],
                        "confidence": 0.82,
                    }
                ),
                provider=self.name,
                model=self.model_id,
                input_chars=len(request.source_text),
                output_chars=100,
            )

        def close(self):
            self.closed = True

    fake_provider = FakeGenerationProvider()

    def fake_generation_provider(provider_name, *, settings):
        assert provider_name == "configured"
        return fake_provider

    monkeypatch.setattr(cli_app, "_generation_provider", fake_generation_provider)

    generated = runner.invoke(app, ["ai", "generate-card", item_id, "--json"])

    assert generated.exit_code == 0, generated.output
    payload = json.loads(generated.output)
    assert payload["provider"] == "fake-openai-compatible"
    assert payload["model"] == "fake-model"
    assert payload["cloud_call"] is False
    assert fake_provider.closed is True

    repo = ContextBankRepository.from_home(home)
    try:
        card = repo.list_knowledge_cards_for_source(item_id)[0]
        assert card.title == "Generated Provider Card"
        assert card.model_provider == "fake-openai-compatible"
        assert card.model_id == "fake-model"
        assert card.confidence == 0.82
        assert card.provenance["model_generation"]["validated"] is True
    finally:
        repo.close()


def test_cli_ai_generate_card_constructs_configured_openai_provider(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setenv("CONTEXTBANK_HOME", str(home))
    monkeypatch.setenv("OPENAI_API_KEY", "fake-openai-key-value")
    runner.invoke(app, ["init"])
    runner.invoke(app, ["config", "set", "ai.generation_provider", "openai"])
    runner.invoke(app, ["config", "set", "ai.generation_model", "gpt-test"])
    runner.invoke(app, ["config", "set", "ai.generation_credential_env", "OPENAI_API_KEY"])
    runner.invoke(app, ["config", "set", "ai.allow_cloud", "true"])
    source_file = tmp_path / "source.md"
    source_file.write_text("# Provider source\n\nUse configured generation.", encoding="utf-8")
    imported = runner.invoke(app, ["import", str(source_file), "--json"])
    item_id = json.loads(imported.output)["item_id"]
    captured: dict[str, Any] = {}

    class FakeOpenAIProvider:
        def __init__(
            self,
            *,
            name: str,
            model_id: str,
            api_key: str | None,
            base_url: str | None = None,
            allow_cloud: bool = False,
        ) -> None:
            captured.update(
                {
                    "name": name,
                    "model_id": model_id,
                    "api_key": api_key,
                    "base_url": base_url,
                    "allow_cloud": allow_cloud,
                }
            )
            self.name = name
            self.model_id = model_id
            self.base_url = base_url or "https://api.openai.com/v1"
            self.closed = False

        def generate_structured(self, request):
            assert "CONTEXTBANK_UNTRUSTED_SOURCE_BEGIN" in request.source_text
            return GenerationResponse(
                text=json.dumps(
                    {
                        "title": "Factory Generated Card",
                        "one_line_summary": "Generated through the configured factory.",
                        "summary": "The CLI constructed the BYOK provider from config.",
                        "why_useful": "It proves the provider seam is wired.",
                        "key_insights": ["Factory seam covered"],
                    }
                ),
                provider=self.name,
                model=self.model_id,
                input_chars=len(request.source_text),
                output_chars=100,
            )

        def close(self):
            self.closed = True

    monkeypatch.setattr(cli_app, "OpenAICompatibleGenerationProvider", FakeOpenAIProvider)

    generated = runner.invoke(app, ["ai", "generate-card", item_id, "--json"])

    assert generated.exit_code == 0, generated.output
    assert captured == {
        "name": "openai",
        "model_id": "gpt-test",
        "api_key": "fake-openai-key-value",
        "base_url": "",
        "allow_cloud": True,
    }
    assert "fake-openai-key-value" not in generated.output
    payload = json.loads(generated.output)
    assert payload["provider"] == "openai"
    assert payload["model"] == "gpt-test"


def test_cli_external_embedding_index_is_cloud_gated_without_network(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setenv("CONTEXTBANK_HOME", str(home))
    monkeypatch.setenv("OPENAI_API_KEY", "fake-openai-key-value")
    runner.invoke(app, ["init"])
    runner.invoke(app, ["config", "set", "ai.embedding_provider", "openai"])
    runner.invoke(app, ["config", "set", "ai.embedding_model", "text-embedding-3-small"])
    runner.invoke(app, ["config", "set", "ai.embedding_credential_env", "OPENAI_API_KEY"])

    blocked = runner.invoke(app, ["ai", "index-embeddings", "--provider", "configured"])

    assert blocked.exit_code != 0
    assert "Set ai.allow_cloud true" in blocked.output
    assert "fake-openai-key-value" not in blocked.output


def test_cli_external_embedding_index_constructs_configured_provider(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setenv("CONTEXTBANK_HOME", str(home))
    monkeypatch.setenv("OPENAI_API_KEY", "fake-openai-key-value")
    runner.invoke(app, ["init"])
    runner.invoke(app, ["config", "set", "ai.embedding_provider", "openai"])
    runner.invoke(app, ["config", "set", "ai.embedding_model", "text-embedding-3-small"])
    runner.invoke(app, ["config", "set", "ai.embedding_credential_env", "OPENAI_API_KEY"])
    runner.invoke(app, ["config", "set", "ai.allow_cloud", "true"])
    captured: dict[str, Any] = {}

    repo = ContextBankRepository.from_home(home)
    try:
        _, _, card = seed_card(
            repo,
            title="Configured embeddings",
            summary="External BYOK embedding configuration should be wired.",
        )
    finally:
        repo.close()

    class FakeEmbeddingProvider:
        def __init__(
            self,
            *,
            name: str,
            model_id: str,
            api_key: str | None,
            base_url: str | None = None,
            allow_cloud: bool = False,
        ) -> None:
            captured.update(
                {
                    "name": name,
                    "model_id": model_id,
                    "api_key": api_key,
                    "base_url": base_url,
                    "allow_cloud": allow_cloud,
                }
            )
            self.name = name
            self.model_id = model_id
            self.request_count = 1
            self.closed = False

        def embed(self, texts):
            assert len(texts) == 1
            assert "Configured embeddings" in texts[0]
            return [[1.0, 0.0, 0.0]]

        def close(self):
            self.closed = True

    monkeypatch.setattr(
        retrieval_search,
        "OpenAICompatibleEmbeddingProvider",
        FakeEmbeddingProvider,
    )

    indexed = runner.invoke(app, ["ai", "index-embeddings", "--provider", "configured", "--json"])

    assert indexed.exit_code == 0, indexed.output
    assert captured == {
        "name": "openai",
        "model_id": "text-embedding-3-small",
        "api_key": "fake-openai-key-value",
        "base_url": "",
        "allow_cloud": True,
    }
    assert "fake-openai-key-value" not in indexed.output
    payload = json.loads(indexed.output)
    assert payload["provider"] == "openai"
    assert payload["model_id"] == "text-embedding-3-small"
    assert payload["indexed"] == 1
    assert payload["cloud_calls"] == 1

    repo = ContextBankRepository.from_home(home)
    try:
        embeddings = repo.list_card_embeddings(provider="openai", model_id="text-embedding-3-small")
        assert embeddings[0]["card_id"] == card.id
    finally:
        repo.close()


def test_cli_ai_index_embeddings_enables_local_semantic_search(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setenv("CONTEXTBANK_HOME", str(home))
    runner.invoke(app, ["init"])
    repo = ContextBankRepository.from_home(home)
    try:
        _, _, card = seed_card(
            repo,
            title="Semantic retrieval",
            summary="Local hash embeddings help find related saved context.",
            tags=["semantic", "embeddings"],
        )
    finally:
        repo.close()

    indexed = runner.invoke(app, ["ai", "index-embeddings", "--json"])
    semantic = runner.invoke(app, ["search", "hash embeddings", "--semantic", "--json"])

    assert indexed.exit_code == 0, indexed.output
    payload = json.loads(indexed.output)
    assert payload["provider"] == "local-hash"
    assert payload["cloud_calls"] == 0
    assert payload["semantic_search_enabled"] is True
    assert semantic.exit_code == 0, semantic.output
    assert json.loads(semantic.output)[0]["card"]["id"] == card.id


def test_cli_mcp_config_emits_codex_toml_and_json_client_snippet(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setenv("CONTEXTBANK_HOME", str(home))
    monkeypatch.setenv("CONTEXTBANK_X_BEARER_TOKEN", "secret-x-token")
    monkeypatch.setenv("OPENAI_API_KEY", "secret-openai-key")

    codex_config = runner.invoke(app, ["mcp", "config"])
    assert codex_config.exit_code == 0, codex_config.output
    parsed = tomllib.loads(codex_config.output)
    server = parsed["mcp_servers"]["contextbank"]
    assert server["command"] == "contextbank"
    assert server["args"] == ["mcp", "serve"]
    assert server["env"]["CONTEXTBANK_HOME"] == str(home)
    assert set(server["env"]) == {"CONTEXTBANK_HOME"}
    assert "secret-x-token" not in codex_config.output
    assert "secret-openai-key" not in codex_config.output

    generic_config = runner.invoke(
        app,
        [
            "mcp",
            "config",
            "--format",
            "json",
            "--name",
            "cb",
            "--command",
            "/usr/local/bin/contextbank",
        ],
    )
    assert generic_config.exit_code == 0, generic_config.output
    payload = json.loads(generic_config.output)
    server = payload["mcpServers"]["cb"]
    assert server["command"] == "/usr/local/bin/contextbank"
    assert server["args"] == ["mcp", "serve"]
    assert server["env"]["CONTEXTBANK_HOME"] == str(home)
    assert set(server["env"]) == {"CONTEXTBANK_HOME"}

    cursor_config = runner.invoke(app, ["mcp", "config", "--client", "cursor"])
    assert cursor_config.exit_code == 0, cursor_config.output
    assert "mcpServers" in json.loads(cursor_config.output)

    claude_config = runner.invoke(app, ["mcp", "config", "--client", "claude"])
    assert claude_config.exit_code == 0, claude_config.output
    assert set(json.loads(claude_config.output)["mcpServers"]["contextbank"]["env"]) == {
        "CONTEXTBANK_HOME"
    }

    claude_code_config = runner.invoke(app, ["mcp", "config", "--client", "claude-code"])
    assert claude_code_config.exit_code == 0, claude_code_config.output
    assert "mcpServers" in json.loads(claude_code_config.output)

    custom_home = tmp_path / "custom-home"
    json_config = runner.invoke(
        app,
        ["mcp", "config", "--client", "mcp-json", "--home", str(custom_home)],
    )
    assert json_config.exit_code == 0, json_config.output
    assert (
        json.loads(json_config.output)["mcpServers"]["contextbank"]["env"]["CONTEXTBANK_HOME"]
        == str(custom_home)
    )

    invalid = runner.invoke(app, ["mcp", "config", "--client", "not-real"])
    assert invalid.exit_code != 0
    assert "Unknown MCP client/config preset" in invalid.output

    project_root = tmp_path / "project"
    project_root.mkdir()
    instructions = runner.invoke(
        app,
        [
            "mcp",
            "instructions",
            "--client",
            "claude-code",
            "--project",
            str(project_root),
        ],
    )
    assert instructions.exit_code == 0, instructions.output
    assert "autoload_project_context" in instructions.output
    assert str(project_root) in instructions.output
    assert "CLAUDE.md" in instructions.output
    assert "secret-x-token" not in instructions.output
    assert "secret-openai-key" not in instructions.output

    write_instructions = runner.invoke(
        app,
        [
            "mcp",
            "instructions",
            "--client",
            "codex",
            "--project",
            str(project_root),
            "--write",
        ],
    )
    assert write_instructions.exit_code == 0, write_instructions.output
    agents_file = project_root / "AGENTS.md"
    assert agents_file.exists()
    assert agents_file.read_text(encoding="utf-8").count("ContextBank Autoload") == 1

    all_project_root = tmp_path / "all-agent-project"
    all_project_root.mkdir()
    all_instructions = runner.invoke(
        app,
        [
            "mcp",
            "instructions",
            "--client",
            "all",
            "--project",
            str(all_project_root),
            "--write",
        ],
    )
    assert all_instructions.exit_code == 0, all_instructions.output
    expected_instruction_files = {
        all_project_root / "AGENTS.md",
        all_project_root / "CLAUDE.md",
        all_project_root / ".cursor" / "rules" / "contextbank.mdc",
        all_project_root / "CONTEXTBANK_AGENT_INSTRUCTIONS.md",
    }
    assert (
        set(map(Path, all_instructions.output.strip().splitlines()))
        == expected_instruction_files
    )
    for instruction_file in expected_instruction_files:
        text = instruction_file.read_text(encoding="utf-8")
        assert text.count("ContextBank Autoload") == 1
        assert "autoload_project_context" in text

    rewrite_all_instructions = runner.invoke(
        app,
        [
            "mcp",
            "instructions",
            "--client",
            "all",
            "--project",
            str(all_project_root),
            "--write",
            "--token-budget",
            "900",
        ],
    )
    assert rewrite_all_instructions.exit_code == 0, rewrite_all_instructions.output
    for instruction_file in expected_instruction_files:
        text = instruction_file.read_text(encoding="utf-8")
        assert text.count("ContextBank Autoload") == 1
        assert '"token_budget": 900' in text

    all_with_target = runner.invoke(
        app,
        [
            "mcp",
            "instructions",
            "--client",
            "all",
            "--project",
            str(all_project_root),
            "--write",
            "--target",
            str(all_project_root / "CUSTOM.md"),
        ],
    )
    assert all_with_target.exit_code != 0
    assert "--target can only be used with a single --client" in all_with_target.output

    manifest = runner.invoke(app, ["mcp", "manifest", "--json"])
    assert manifest.exit_code == 0, manifest.output
    manifest_payload = json.loads(manifest.output)
    assert manifest_payload["transport"] == "stdio"
    assert manifest_payload["read_only_default"] is True
    assert manifest_payload["env"] == {"CONTEXTBANK_HOME": str(home)}
    assert manifest_payload["security"]["secrets_included"] is False
    assert manifest_payload["security"]["write_tools_enabled_by_default"] is False
    assert manifest_payload["autoload"]["tool"] == "autoload_project_context"
    assert "instructions_command" in manifest_payload["autoload"]
    assert "|all>" in manifest_payload["autoload"]["instructions_command"]
    assert "start of an agent session" in manifest_payload["autoload"]["startup_instruction"]
    assert "autoload_project_context" in {
        tool["name"] for tool in manifest_payload["tools"]
    }


def test_cli_add_reads_manual_text_from_stdin(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setenv("CONTEXTBANK_HOME", str(home))
    runner.invoke(app, ["init"])

    result = runner.invoke(
        app,
        [
            "add",
            "-",
            "--source-url",
            "https://example.com/source",
            "--title",
            "Copied note",
            "--json",
        ],
        input="Copied local-first retrieval note",
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["card_path"].endswith(".md")

    repo = ContextBankRepository.from_home(home)
    try:
        source = repo.get_source_item(payload["item_id"])
        document = repo.get_document(payload["document_id"])
        assert source.source_type == "text"
        assert source.source_url == "https://example.com/source"
        assert document.document_type == "manual_text"
        assert document.title == "Copied note"
    finally:
        repo.close()


def test_cli_add_manual_text_reimport_with_different_source_url_is_idempotent(
    tmp_path,
    monkeypatch,
):
    home = tmp_path / "home"
    monkeypatch.setenv("CONTEXTBANK_HOME", str(home))
    runner.invoke(app, ["init"])
    text = "Same pasted context should dedupe across source URLs."

    first = runner.invoke(
        app,
        ["add", "-", "--source-url", "https://example.com/first", "--json"],
        input=text,
    )
    second = runner.invoke(
        app,
        ["add", "-", "--source-url", "https://example.com/second", "--json"],
        input=text,
    )

    assert first.exit_code == 0, first.output
    assert second.exit_code == 0, second.output
    first_payload = json.loads(first.output)
    second_payload = json.loads(second.output)
    assert second_payload["item_id"] == first_payload["item_id"]
    assert second_payload["document_id"] == first_payload["document_id"]

    repo = ContextBankRepository.from_home(home)
    try:
        source = repo.get_source_item(first_payload["item_id"])
        assert repo.count_rows("source_items") == 1
        assert repo.count_rows("documents") == 1
        assert repo.count_rows("knowledge_cards") == 1
        assert source.source_url == "https://example.com/second"
    finally:
        repo.close()


def test_cli_add_rejects_private_network_web_targets(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setenv("CONTEXTBANK_HOME", str(home))
    runner.invoke(app, ["init"])

    result = runner.invoke(app, ["add", "http://127.0.0.1:8000/private", "--json"])

    assert result.exit_code != 0
    assert "private or local network URLs" in result.output
    repo = ContextBankRepository.from_home(home)
    try:
        assert repo.count_rows("source_items") == 0
    finally:
        repo.close()


def test_cli_import_list_parses_deduped_markdown_urls(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setenv("CONTEXTBANK_HOME", str(home))
    runner.invoke(app, ["init"])
    source_list = tmp_path / "urls.md"
    source_list.write_text(
        "- [First](https://example.com/first)\n"
        "- https://example.com/second\n"
        "- https://example.com/first\n",
        encoding="utf-8",
    )

    def fake_ingest_url(url, paths):
        return import_text(f"Captured from {url}", paths, source_url=url, title=url)

    monkeypatch.setattr(cli_app, "ingest_url", fake_ingest_url)

    result = runner.invoke(app, ["import-list", str(source_list), "--json"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["requested"] == 2
    assert payload["imported"] == 2
    assert payload["failed"] == 0
    assert [item["url"] for item in payload["items"]] == [
        "https://example.com/first",
        "https://example.com/second",
    ]


def test_cli_import_list_parses_headerless_csv_without_dropping_first_row(
    tmp_path,
    monkeypatch,
):
    home = tmp_path / "home"
    monkeypatch.setenv("CONTEXTBANK_HOME", str(home))
    runner.invoke(app, ["init"])
    source_list = tmp_path / "urls.csv"
    source_list.write_text(
        "https://example.com/first,https://example.com/second\n"
        "https://example.com/third\n",
        encoding="utf-8",
    )

    def fake_ingest_url(url, paths):
        return import_text(f"Captured from {url}", paths, source_url=url, title=url)

    monkeypatch.setattr(cli_app, "ingest_url", fake_ingest_url)

    result = runner.invoke(app, ["import-list", str(source_list), "--json"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["requested"] == 3
    assert [item["url"] for item in payload["items"]] == [
        "https://example.com/first",
        "https://example.com/second",
        "https://example.com/third",
    ]


def test_cli_import_list_parses_json_url_fields(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setenv("CONTEXTBANK_HOME", str(home))
    runner.invoke(app, ["init"])
    source_list = tmp_path / "urls.json"
    source_list.write_text(
        json.dumps(
            {
                "items": [
                    {"url": "https://example.com/first"},
                    {"source_url": "https://example.com/second"},
                    {"href": "https://example.com/third"},
                    {"link": "https://example.com/first"},
                ]
            }
        ),
        encoding="utf-8",
    )

    def fake_ingest_url(url, paths):
        return import_text(f"Captured from {url}", paths, source_url=url, title=url)

    monkeypatch.setattr(cli_app, "ingest_url", fake_ingest_url)

    result = runner.invoke(app, ["import-list", str(source_list), "--json"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["requested"] == 3
    assert [item["url"] for item in payload["items"]] == [
        "https://example.com/first",
        "https://example.com/second",
        "https://example.com/third",
    ]


def test_cli_source_adapters_lists_pluggable_connector_contracts():
    result = runner.invoke(app, ["source", "adapters", "--json"])

    assert result.exit_code == 0, result.output
    adapters = json.loads(result.output)["adapters"]
    by_name = {adapter["name"]: adapter for adapter in adapters}
    assert set(by_name) == {"file", "text", "web", "x.bookmarks"}
    assert by_name["x.bookmarks"]["requires_credentials"] is True
    assert by_name["x.bookmarks"]["capabilities"]["continuation_state"] is True
    assert by_name["web"]["capabilities"]["fetch_one"] is True
    assert by_name["file"]["capabilities"]["normalized_documents"] is True


def test_cli_import_list_continue_on_error_survives_persistence_errors(
    tmp_path,
    monkeypatch,
):
    home = tmp_path / "home"
    monkeypatch.setenv("CONTEXTBANK_HOME", str(home))
    runner.invoke(app, ["init"])
    source_list = tmp_path / "urls.md"
    source_list.write_text(
        "- https://example.com/first\n"
        "- https://example.com/broken\n"
        "- https://example.com/third\n",
        encoding="utf-8",
    )

    def fake_ingest_url(url, paths):
        return import_text(f"Captured from {url}", paths, source_url=url, title=url)

    original_persist = cli_app._persist_ingest

    def flaky_persist(repo, cards_path, source_item, document):
        if str(source_item.source_url).endswith("/broken"):
            raise RuntimeError("simulated persistence failure")
        return original_persist(repo, cards_path, source_item, document)

    monkeypatch.setattr(cli_app, "ingest_url", fake_ingest_url)
    monkeypatch.setattr(cli_app, "_persist_ingest", flaky_persist)

    result = runner.invoke(app, ["import-list", str(source_list), "--json"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["requested"] == 3
    assert payload["imported"] == 2
    assert payload["failed"] == 1
    assert payload["errors"][0]["url"] == "https://example.com/broken"
    assert "simulated persistence failure" in payload["errors"][0]["error"]
    assert [item["url"] for item in payload["items"]] == [
        "https://example.com/first",
        "https://example.com/third",
    ]


def test_cli_duplicate_import_is_preserved_but_hidden_from_agent_search(
    tmp_path,
    monkeypatch,
):
    home = tmp_path / "home"
    monkeypatch.setenv("CONTEXTBANK_HOME", str(home))
    runner.invoke(app, ["init"])
    text = "Duplicate context body for local agent retrieval."
    digest = content_hash(text)

    def fake_ingest_url(url, paths):
        item_id = source_item_id(SourceType.WEB.value, value=url)
        return cli_app.CliIngestResult(
            source_item=SourceItem(
                id=item_id,
                source_type=SourceType.WEB,
                source_external_id=url,
                source_url=url,
                content_hash=digest,
                availability_status=AvailabilityStatus.AVAILABLE,
            ),
            document=Document(
                id=document_id(item_id, "linked_page", digest),
                source_item_id=item_id,
                document_type="linked_page",
                canonical_url="https://example.com/canonical-duplicate",
                title="Duplicate context",
                extraction_status=ExtractionStatus.COMPLETE,
                content_hash=digest,
                text=text,
            ),
            warnings=[],
        )

    monkeypatch.setattr(cli_app, "ingest_url", fake_ingest_url)

    first = runner.invoke(app, ["add", "https://example.com/first", "--json"])
    second = runner.invoke(app, ["add", "https://example.com/second", "--json"])
    first_id = json.loads(first.output)["item_id"]
    second_id = json.loads(second.output)["item_id"]
    search_result = runner.invoke(app, ["search", "duplicate context body", "--json"])
    second_item = runner.invoke(app, ["show", second_id, "--json"])
    duplicate_queue = runner.invoke(app, ["review", "list", "--queue", "duplicates", "--json"])

    assert first.exit_code == 0, first.output
    assert second.exit_code == 0, second.output
    assert search_result.exit_code == 0, search_result.output
    results = json.loads(search_result.output)
    assert [result["card"]["source_item_id"] for result in results] == [first_id]

    assert second_item.exit_code == 0, second_item.output
    duplicate_card = json.loads(second_item.output)["card"]
    assert duplicate_card["source_item_id"] == second_id
    assert duplicate_card["provenance"]["duplicate_detection"]["status"] == (
        "possible-duplicate"
    )
    assert duplicate_card["provenance"]["user_signals"]["hidden_from_agent_retrieval"] is True
    assert duplicate_queue.exit_code == 0, duplicate_queue.output
    assert json.loads(duplicate_queue.output)[0]["source_item_id"] == second_id

    approved = runner.invoke(
        app,
        [
            "review",
            "approve-duplicate",
            second_id,
            "--duplicate-of",
            first_id,
            "--reason",
            "Same canonical URL and body.",
            "--json",
        ],
    )
    approved_search = runner.invoke(app, ["search", "duplicate context body", "--json"])

    assert approved.exit_code == 0, approved.output
    approved_payload = json.loads(approved.output)
    assert approved_payload["duplicate_relationship"]["status"] == "duplicate"
    assert approved_payload["duplicate_detection"]["status"] == "confirmed-duplicate"
    assert approved_payload["user_signals"]["hidden_from_agent_retrieval"] is True
    assert [result["card"]["source_item_id"] for result in json.loads(approved_search.output)] == [
        first_id
    ]

    rejected = runner.invoke(
        app,
        [
            "review",
            "reject-duplicate",
            second_id,
            "--not-duplicate-of",
            first_id,
            "--reason",
            "Keep this saved context separately.",
            "--json",
        ],
    )
    unhidden_search = runner.invoke(
        app,
        [
            "search",
            "duplicate context body",
            "--max-output-size",
            "20000",
            "--json",
        ],
    )

    assert rejected.exit_code == 0, rejected.output
    rejected_payload = json.loads(rejected.output)
    assert rejected_payload["duplicate_relationship"]["status"] == "not-duplicate"
    assert rejected_payload["duplicate_detection"]["status"] == "not-duplicate"
    assert rejected_payload["user_signals"]["hidden_from_agent_retrieval"] is False
    assert {result["card"]["source_item_id"] for result in json.loads(unhidden_search.output)} == {
        first_id,
        second_id,
    }


def test_cli_add_x_post_url_creates_partial_record_that_sync_hydrates(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setenv("CONTEXTBANK_HOME", str(home))
    monkeypatch.setenv("CONTEXTBANK_X_BEARER_TOKEN", "user-token")
    runner.invoke(app, ["init"])

    added = runner.invoke(
        app,
        ["add", "https://x.com/example_user/status/1234567890", "--json"],
    )

    assert added.exit_code == 0, added.output
    added_payload = json.loads(added.output)
    repo = ContextBankRepository.from_home(home)
    try:
        source = repo.get_source_item(added_payload["item_id"])
        document = repo.get_document(added_payload["document_id"])
        assert source.source_type == "x"
        assert source.source_external_id == "1234567890"
        assert source.author_handle == "example_user"
        assert source.availability_status == "partial"
        assert document.extraction_status == "partial"
        assert document.metadata["hydration_required"] is True
        assert repo.count_rows("source_items") == 1
    finally:
        repo.close()

    @dataclass(frozen=True)
    class FakePage:
        data: list[dict[str, Any]]
        includes: dict[str, Any]
        next_token: str | None = None

    class FakeClient:
        def __init__(self, token: str) -> None:
            assert token == "user-token"

        def get_authenticated_user(self) -> dict[str, str]:
            return {"id": "123"}

        def iter_bookmark_pages(self, user_id: str, *, limit: int = 100):
            assert user_id == "123"
            yield FakePage(
                data=[
                    {
                        "id": "1234567890",
                        "text": "Hydrated post text from the official X API.",
                        "author_id": "42",
                        "created_at": "2026-06-20T12:00:00Z",
                    }
                ],
                includes={"users": [{"id": "42", "name": "Example", "username": "example_user"}]},
            )

    monkeypatch.setattr(cli_app, "XBookmarkClient", FakeClient)

    synced = runner.invoke(app, ["sync", "x", "--limit", "1", "--json"])

    assert synced.exit_code == 0, synced.output
    sync_payload = json.loads(synced.output)
    assert sync_payload["imported"] == 0
    assert sync_payload["updated"] == 1

    repo = ContextBankRepository.from_home(home)
    try:
        assert repo.count_rows("source_items") == 1
        assert repo.count_rows("documents") == 1
        source = repo.get_source_item(added_payload["item_id"])
        document = repo.get_document(added_payload["document_id"])
        assert source.availability_status == "available"
        assert source.author_handle == "example_user"
        assert document.extraction_status == "complete"
        assert document.text == "Hydrated post text from the official X API."
        assert document.metadata["raw_retained"] is True
    finally:
        repo.close()


def test_cli_rejects_secret_config_keys_and_redacts_existing_values(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setenv("CONTEXTBANK_HOME", str(home))
    runner.invoke(app, ["init"])

    rejected_key = runner.invoke(app, ["config", "set", "x.access_token", "secret-token"])
    rejected_secret_value = "sk-proj-" + "secretvalue0123456789"
    rejected_value = runner.invoke(
        app,
        ["config", "set", "ai.generation_credential_env", rejected_secret_value],
    )
    assert rejected_key.exit_code != 0
    assert "Refusing to store secret-like config keys" in rejected_key.output
    assert rejected_value.exit_code != 0
    assert "Refusing to store a value that looks like a secret" in rejected_value.output

    config_file = home / "config.toml"
    secret_value = "sk-proj-" + "secretvalue0123456789"
    config_file.write_text(
        config_file.read_text(encoding="utf-8").replace(
            'user_id = ""',
            f'user_id = ""\ncredential_env = "{secret_value}"',
        ),
        encoding="utf-8",
    )
    shown = runner.invoke(app, ["config", "show", "--json"])
    assert shown.exit_code == 0, shown.output
    assert secret_value not in shown.output
    assert "sk-p...6789" in shown.output


def test_cli_backup_sanitizes_and_verify_rejects_secret_like_config_values(
    tmp_path,
    monkeypatch,
):
    home = tmp_path / "home"
    monkeypatch.setenv("CONTEXTBANK_HOME", str(home))
    runner.invoke(app, ["init"])
    secret_value = "sk-proj-" + "backupsecret0123456789"
    config_file = home / "config.toml"
    config_file.write_text(
        config_file.read_text(encoding="utf-8").replace(
            'generation_credential_env = ""',
            f'generation_credential_env = "{secret_value}"',
        )
        + '\n[connectors.web]\nuser_agent = "ContextBank Test"\n',
        encoding="utf-8",
    )

    backup = runner.invoke(app, ["backup", "create"])

    assert backup.exit_code == 0, backup.output
    backup_path = Path(backup.output.strip())
    backed_up_config = backup_path / "config.toml"
    backed_up_text = backed_up_config.read_text(encoding="utf-8")
    assert secret_value not in backed_up_text
    assert "[REDACTED]" in backed_up_text
    assert "[connectors.web]" in backed_up_text
    assert 'user_agent = "ContextBank Test"' in backed_up_text

    verified = runner.invoke(app, ["backup", "verify", str(backup_path), "--json"])
    assert verified.exit_code == 0, verified.output
    verified_checks = {
        check["name"]: check["passed"] for check in json.loads(verified.output)["checks"]
    }
    assert verified_checks["config_no_secret_values"] is True

    backed_up_config.write_text(
        backed_up_config.read_text(encoding="utf-8").replace("[REDACTED]", secret_value),
        encoding="utf-8",
    )
    rejected = runner.invoke(app, ["backup", "verify", str(backup_path), "--json"])

    assert rejected.exit_code == 1, rejected.output
    payload = json.loads(rejected.output)
    assert payload["valid"] is False
    checks = {check["name"]: check for check in payload["checks"]}
    assert checks["config_no_secret_values"]["passed"] is False
    assert secret_value not in rejected.output


def test_cli_auth_x_stores_oauth_token_and_logout_deletes_it(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setenv("CONTEXTBANK_HOME", str(home))
    monkeypatch.setenv("CONTEXTBANK_X_CLIENT_ID", "client-123")
    runner.invoke(app, ["init"])

    def fake_flow(**kwargs: Any) -> StoredXToken:
        assert kwargs["client_id"] == "client-123"
        assert kwargs["redirect_uri"] == "http://127.0.0.1:8765/oauth/x/callback"
        assert kwargs["scopes"] == ("tweet.read", "users.read", "bookmark.read")
        return StoredXToken(
            access_token="stored-access-token",
            client_id="client-123",
            expires_at=datetime(2026, 6, 20, tzinfo=UTC),
        )

    monkeypatch.setattr(cli_app, "run_local_authorization_flow", fake_flow)

    auth_result = runner.invoke(app, ["auth", "x", "--json"])
    assert auth_result.exit_code == 0, auth_result.output
    assert json.loads(auth_result.output)["authenticated"] is True

    status = runner.invoke(app, ["auth", "status", "--json"])
    assert status.exit_code == 0, status.output
    assert "stored-access-token" not in status.output
    assert "...oken" in status.output

    logout = runner.invoke(app, ["auth", "logout", "x"])
    assert logout.exit_code == 0, logout.output
    assert "Deleted stored X OAuth token" in logout.output


def test_cli_auth_x_uses_configured_public_client_id(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setenv("CONTEXTBANK_HOME", str(home))
    monkeypatch.delenv("CONTEXTBANK_X_CLIENT_ID", raising=False)
    runner.invoke(app, ["init"])
    configured = runner.invoke(app, ["config", "set", "x.client_id", "client-config"])
    assert configured.exit_code == 0, configured.output

    def fake_flow(**kwargs: Any) -> StoredXToken:
        assert kwargs["client_id"] == "client-config"
        return StoredXToken(
            access_token="configured-client-token",
            client_id="client-config",
            expires_at=datetime(2026, 6, 20, tzinfo=UTC),
        )

    monkeypatch.setattr(cli_app, "run_local_authorization_flow", fake_flow)

    setup = runner.invoke(app, ["auth", "x-setup", "--json"])
    auth_result = runner.invoke(app, ["auth", "x", "--json"])
    shown = runner.invoke(app, ["config", "show", "--json"])

    assert setup.exit_code == 0, setup.output
    setup_payload = json.loads(setup.output)
    assert setup_payload["client"]["env_configured"] is False
    assert setup_payload["client"]["config_configured"] is True
    assert setup_payload["client"]["ready_for_auth"] is True
    setup_actions = {action["id"]: action for action in setup_payload["next_actions"]}
    assert setup_actions["authorize_x_user"]["command"] == "contextbank auth x"

    assert auth_result.exit_code == 0, auth_result.output
    assert json.loads(auth_result.output)["authenticated"] is True
    assert "configured-client-token" not in shown.output
    assert '"client_id": "client-config"' in shown.output


def test_cli_auth_x_setup_reports_oauth_readiness_without_secrets(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setenv("CONTEXTBANK_HOME", str(home))
    monkeypatch.setenv("CONTEXTBANK_X_CLIENT_ID", "client-123")
    runner.invoke(app, ["init"])

    paths = initialize_settings().paths
    save_stored_token(
        paths,
        StoredXToken(
            access_token="stored-secret-token",
            client_id="client-123",
            scope="tweet.read users.read",
            expires_at=datetime(2027, 6, 20, tzinfo=UTC),
        ),
    )

    setup = runner.invoke(app, ["auth", "x-setup", "--json"])
    invalid = runner.invoke(
        app,
        [
            "auth",
            "x-setup",
            "--callback-url",
            "http://localhost:8765/oauth/x/callback",
            "--json",
        ],
    )
    doctor = runner.invoke(app, ["doctor", "--json"])

    assert setup.exit_code == 0, setup.output
    assert "stored-secret-token" not in setup.output
    payload = json.loads(setup.output)
    assert payload["developer_console"]["app_type"] == "native"
    assert payload["developer_console"]["callback_url_valid"] is True
    assert payload["client"]["ready_for_auth"] is True
    assert payload["token"]["ready_for_sync"] is False
    assert payload["token"]["sync_status"] == "missing-required-scopes"
    assert payload["token"]["missing_required_scopes"] == ["bookmark.read"]
    assert payload["token"]["secrets_redacted"] is True
    assert payload["developer_console"]["oauth_2_enabled"] is True
    assert payload["developer_console"]["required_app_permissions"] == "Read"
    assert "ContextBank is an open-source, local-first" in payload["data_use_answer"]
    assert "does not post, like, reply" in payload["data_use_answer"]
    setup_actions = {action["id"]: action for action in payload["next_actions"]}
    assert setup_actions["authorize_x_user"]["required_for"] == ["x_sync"]
    assert setup_actions["authorize_x_user"]["command"] == "contextbank auth x"

    assert invalid.exit_code == 0, invalid.output
    invalid_payload = json.loads(invalid.output)
    assert invalid_payload["developer_console"]["callback_url_valid"] is False
    assert "127.0.0.1" in invalid_payload["developer_console"]["callback_url_issues"][0]
    invalid_actions = {action["id"]: action for action in invalid_payload["next_actions"]}
    assert invalid_actions["fix_x_callback_url"]["required_for"] == ["x_sync"]
    assert "127.0.0.1:8765" in invalid_actions["fix_x_callback_url"]["command"]

    assert doctor.exit_code == 0, doctor.output
    assert "stored-secret-token" not in doctor.output
    assert json.loads(doctor.output)["x"]["oauth_setup"]["client"]["ready_for_auth"] is True


def test_cli_review_approve_skill_updates_card_and_markdown(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setenv("CONTEXTBANK_HOME", str(home))
    runner.invoke(app, ["init"])
    repo = ContextBankRepository.from_home(home)
    try:
        _, _, card = seed_card(
            repo,
            title="Reusable MCP workflow",
            summary="Step 1. Inspect context. Step 2. Verify output.",
            skill_candidate_status="candidate",
        )
    finally:
        repo.close()

    result = runner.invoke(app, ["review", "approve-skill", card.id, "--json"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["skill_candidate_status"] == "approved"
    assert payload["review_status"] == "approved"
    assert payload["active"] is True

    repo = ContextBankRepository.from_home(home)
    try:
        updated = repo.get_knowledge_card(card.id)
        assert updated.skill_candidate_status == "approved"
        assert updated.review_status == "approved"
        assert updated.provenance["skill_candidate"]["active"] is True
    finally:
        repo.close()

    markdown = (home / "cards").glob("*/*.md")
    assert "review_status: \"approved\"" in next(markdown).read_text(encoding="utf-8")


def test_cli_review_list_filters_health_queues(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setenv("CONTEXTBANK_HOME", str(home))
    runner.invoke(app, ["init"])
    repo = ContextBankRepository.from_home(home)
    try:
        _, _, card = seed_card(
            repo,
            title="Needs review",
            summary="This item should appear in multiple review queues.",
            skill_candidate_status="candidate",
        )
        repo.upsert_knowledge_card(
            card.model_copy(
                update={
                    "confidence": 0.2,
                    "freshness_status": "stale-risk",
                    "review_status": "hidden",
                }
            )
        )
    finally:
        repo.close()

    all_items = runner.invoke(app, ["review", "list", "--json"])
    skill_items = runner.invoke(app, ["review", "list", "--queue", "skill", "--json"])
    low_confidence = runner.invoke(
        app,
        ["review", "list", "--queue", "low-confidence", "--json"],
    )
    hidden = runner.invoke(app, ["review", "list", "--queue", "hidden", "--json"])

    assert all_items.exit_code == 0, all_items.output
    assert skill_items.exit_code == 0, skill_items.output
    assert low_confidence.exit_code == 0, low_confidence.output
    assert hidden.exit_code == 0, hidden.output
    assert "skill-review" in json.loads(all_items.output)[0]["reasons"]
    assert json.loads(skill_items.output)[0]["card_id"] == card.id
    assert json.loads(low_confidence.output)[0]["confidence"] == 0.2
    assert json.loads(hidden.output)[0]["review_status"] == "hidden"


def test_cli_review_correct_updates_card_and_preserves_audit_history(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setenv("CONTEXTBANK_HOME", str(home))
    runner.invoke(app, ["init"])
    repo = ContextBankRepository.from_home(home)
    try:
        _, document, card = seed_card(
            repo,
            title="Generated title",
            summary="Generated summary about local retrieval.",
            topics=["generated"],
            tags=["draft"],
        )
        repo.upsert_document(
            document.model_copy(update={"text": "Fresh source text should not erase corrections."})
        )
    finally:
        repo.close()

    corrected = runner.invoke(
        app,
        [
            "review",
            "correct",
            card.id,
            "--title",
            "Human title",
            "--summary",
            "Human summary for retrieval.",
            "--topic",
            "retrieval",
            "--topic",
            "local-first",
            "--tag",
            "human-reviewed",
            "--utility",
            "reference",
            "--list-field",
            "key_insights",
            "--list-item",
            "Human-verified insight",
            "--note",
            "Corrected after review.",
            "--json",
        ],
    )

    assert corrected.exit_code == 0, corrected.output
    payload = json.loads(corrected.output)
    assert set(payload["corrected_fields"]) >= {"title", "summary", "topics", "key_insights"}

    search_result = runner.invoke(app, ["search", "Human-verified", "--json"])
    assert search_result.exit_code == 0, search_result.output
    assert json.loads(search_result.output)[0]["card"]["title"] == "Human title"

    reprocessed = runner.invoke(app, ["reprocess", card.id, "--json"])
    assert reprocessed.exit_code == 0, reprocessed.output

    repo = ContextBankRepository.from_home(home)
    try:
        updated = repo.get_knowledge_card(card.id)
        assert updated.title == "Human title"
        assert updated.summary == "Human summary for retrieval."
        assert updated.topics == ["retrieval", "local-first"]
        assert updated.tags == ["human-reviewed"]
        assert updated.key_insights == ["Human-verified insight"]
        assert updated.review_status == "approved"
        history = updated.provenance["correction_history"]
        assert history[0]["fields"]["title"]["previous"] == "Generated title"
        assert history[0]["fields"]["title"]["corrected"] == "Human title"
    finally:
        repo.close()

    markdown_text = next((home / "cards").glob("*/*.md")).read_text(encoding="utf-8")
    assert "Human corrections:" in markdown_text
    assert "# Human title" in markdown_text


def test_cli_mark_user_signals_and_hide_from_search(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setenv("CONTEXTBANK_HOME", str(home))
    runner.invoke(app, ["init"])
    repo = ContextBankRepository.from_home(home)
    try:
        _, _, card = seed_card(
            repo,
            title="Private retrieval note",
            summary="A hidden retrieval note should not appear in agent search.",
        )
    finally:
        repo.close()

    marked = runner.invoke(
        app,
        [
            "mark",
            card.id,
            "--priority",
            "high",
            "--usefulness",
            "5",
            "--used-in-project",
            "ContextBank",
            "--pin",
            "--obsolete",
            "--hide",
            "--json",
        ],
    )
    assert marked.exit_code == 0, marked.output
    payload = json.loads(marked.output)
    assert payload["personal_priority"] == "high"
    assert payload["freshness_status"] == "stale"
    assert payload["review_status"] == "hidden"
    assert payload["user_signals"]["pinned"] is True
    assert payload["user_signals"]["usefulness_rating"] == 5

    hidden_search = runner.invoke(app, ["search", "hidden retrieval note", "--json"])
    assert hidden_search.exit_code == 0, hidden_search.output
    assert json.loads(hidden_search.output) == []

    show_result = runner.invoke(app, ["show", card.id, "--json"])
    assert show_result.exit_code == 0, show_result.output
    assert json.loads(show_result.output)["card"]["review_status"] == "hidden"

    unhidden = runner.invoke(app, ["mark", card.id, "--unhide", "--json"])
    assert unhidden.exit_code == 0, unhidden.output
    assert json.loads(unhidden.output)["review_status"] == "unreviewed"

    visible_search = runner.invoke(app, ["search", "hidden retrieval note", "--json"])
    assert visible_search.exit_code == 0, visible_search.output
    assert json.loads(visible_search.output)[0]["card"]["id"] == card.id


def test_cli_hidden_signal_keeps_card_out_of_agent_search_after_review_changes(
    tmp_path,
    monkeypatch,
):
    home = tmp_path / "home"
    monkeypatch.setenv("CONTEXTBANK_HOME", str(home))
    runner.invoke(app, ["init"])
    repo = ContextBankRepository.from_home(home)
    try:
        _, _, card = seed_card(
            repo,
            title="Hidden incorrect note",
            summary="A hidden note should stay hidden even after review status changes.",
        )
    finally:
        repo.close()

    hidden = runner.invoke(app, ["mark", card.id, "--hide", "--json"])
    incorrect = runner.invoke(app, ["mark", card.id, "--incorrect", "--json"])
    search_result = runner.invoke(app, ["search", "hidden note", "--json"])
    unhidden = runner.invoke(app, ["mark", card.id, "--unhide", "--json"])
    still_rejected_search = runner.invoke(app, ["search", "hidden note", "--json"])

    assert hidden.exit_code == 0, hidden.output
    assert incorrect.exit_code == 0, incorrect.output
    incorrect_payload = json.loads(incorrect.output)
    assert incorrect_payload["review_status"] == "rejected"
    assert incorrect_payload["user_signals"]["hidden_from_agent_retrieval"] is True
    assert search_result.exit_code == 0, search_result.output
    assert json.loads(search_result.output) == []

    assert unhidden.exit_code == 0, unhidden.output
    assert json.loads(unhidden.output)["review_status"] == "rejected"
    assert still_rejected_search.exit_code == 0, still_rejected_search.output
    assert json.loads(still_rejected_search.output) == []


def test_cli_source_lifecycle_deletes_files_and_detaches_notes(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setenv("CONTEXTBANK_HOME", str(home))
    runner.invoke(app, ["init"])

    added = runner.invoke(
        app,
        ["add", "-", "--title", "Lifecycle note", "--json"],
        input="Lifecycle raw note for local retention testing.",
    )
    assert added.exit_code == 0, added.output
    added_payload = json.loads(added.output)
    item_id = added_payload["item_id"]

    repo = ContextBankRepository.from_home(home)
    try:
        source = repo.get_source_item(item_id)
        document = repo.get_document(added_payload["document_id"])
        card = repo.list_knowledge_cards_for_source(item_id)[0]
        raw_path = Path(source.raw_path)
        extracted_path = Path(document.extracted_text_path)
        markdown_path = next((home / "cards").glob(f"*/{card.id}.md"))
        assert raw_path.exists()
        assert extracted_path.exists()
        assert markdown_path.exists()
    finally:
        repo.close()

    unavailable = runner.invoke(
        app,
        ["source", "unavailable", item_id, "--reason", "Upstream disappeared", "--json"],
    )
    revalidated = runner.invoke(
        app,
        ["source", "revalidate", item_id, "--reason", "Need a fresh check", "--json"],
    )
    marked = runner.invoke(
        app,
        ["mark", item_id, "--priority", "high", "--usefulness", "5", "--json"],
    )
    raw_without_yes = runner.invoke(app, ["source", "delete-raw", item_id])
    deleted_raw = runner.invoke(app, ["source", "delete-raw", item_id, "--yes", "--json"])

    assert unavailable.exit_code == 0, unavailable.output
    assert json.loads(unavailable.output)["source_item"]["availability_status"] == "unavailable"
    assert revalidated.exit_code == 0, revalidated.output
    assert json.loads(revalidated.output)["updated_card_ids"] == [card.id]
    assert marked.exit_code == 0, marked.output
    assert raw_without_yes.exit_code != 0
    assert deleted_raw.exit_code == 0, deleted_raw.output
    raw_payload = json.loads(deleted_raw.output)
    assert str(raw_path) in raw_payload["files"]["deleted"]
    assert not raw_path.exists()
    assert extracted_path.exists()

    repo = ContextBankRepository.from_home(home)
    try:
        source_after_raw_delete = repo.get_source_item(item_id)
        document_after_raw_delete = repo.get_document(added_payload["document_id"])
        assert source_after_raw_delete.raw_path is None
        assert document_after_raw_delete.extracted_text_path == str(extracted_path)
        assert document_after_raw_delete.metadata["raw_retained"] is False
        assert repo.get_knowledge_card(card.id).freshness_status == "recheck-soon"
    finally:
        repo.close()

    delete_without_yes = runner.invoke(app, ["source", "delete", item_id])
    deleted = runner.invoke(
        app,
        [
            "source",
            "delete",
            item_id,
            "--yes",
            "--detach-notes",
            "--reason",
            "Testing complete deletion.",
            "--json",
        ],
    )

    assert delete_without_yes.exit_code != 0
    assert deleted.exit_code == 0, deleted.output
    deleted_payload = json.loads(deleted.output)
    assert deleted_payload["deleted"] is True
    assert deleted_payload["detached_notes_preserved"] is True
    assert str(extracted_path) in deleted_payload["files"]["deleted"]
    assert str(markdown_path) in deleted_payload["files"]["deleted"]
    assert not extracted_path.exists()
    assert not markdown_path.exists()

    repo = ContextBankRepository.from_home(home)
    try:
        assert repo.get_source_item(item_id) is None
        assert repo.get_document(added_payload["document_id"]) is None
        assert repo.get_knowledge_card(card.id) is None
        detached = repo.get_detached_note(deleted_payload["detached_note_id"])
        assert detached["user_notes"]["cards"][0]["user_signals"]["usefulness_rating"] == 5
    finally:
        repo.close()


def test_cli_source_delete_raw_skips_paths_outside_contextbank_home(tmp_path, monkeypatch):
    home = tmp_path / "home"
    outside = tmp_path / "outside-raw.txt"
    outside.write_text("outside raw data must survive", encoding="utf-8")
    monkeypatch.setenv("CONTEXTBANK_HOME", str(home))
    runner.invoke(app, ["init"])

    repo = ContextBankRepository.from_home(home)
    try:
        source_item, _, _ = seed_card(repo)
        repo.upsert_source_item(source_item.model_copy(update={"raw_path": str(outside)}))
    finally:
        repo.close()

    deleted = runner.invoke(app, ["source", "delete-raw", source_item.id, "--yes", "--json"])

    assert deleted.exit_code == 0, deleted.output
    payload = json.loads(deleted.output)
    assert str(outside.resolve()) in payload["files"]["skipped"]
    assert outside.read_text(encoding="utf-8") == "outside raw data must survive"

    repo = ContextBankRepository.from_home(home)
    try:
        assert repo.get_source_item(source_item.id).raw_path is None
    finally:
        repo.close()


def test_cli_source_revalidate_without_fetch_does_not_call_connectors(
    tmp_path,
    monkeypatch,
):
    home = tmp_path / "home"
    monkeypatch.setenv("CONTEXTBANK_HOME", str(home))
    runner.invoke(app, ["init"])

    source_file = tmp_path / "note.md"
    source_file.write_text("# Local file\n\nNo-fetch revalidation.", encoding="utf-8")
    imported_file = runner.invoke(app, ["import", str(source_file), "--json"])
    assert imported_file.exit_code == 0, imported_file.output
    file_item_id = json.loads(imported_file.output)["item_id"]

    added_x = runner.invoke(
        app,
        ["add", "https://x.com/example_user/status/1234567890", "--json"],
    )
    assert added_x.exit_code == 0, added_x.output
    x_item_id = json.loads(added_x.output)["item_id"]

    repo = ContextBankRepository.from_home(home)
    try:
        web_source, _, web_card = seed_card(
            repo,
            url="https://example.com/no-fetch",
            title="No fetch web source",
            summary="Web source should not refetch without the explicit flag.",
        )
        before_counts = {
            web_source.id: len(repo.list_documents_for_source(web_source.id)),
            file_item_id: len(repo.list_documents_for_source(file_item_id)),
            x_item_id: len(repo.list_documents_for_source(x_item_id)),
        }
    finally:
        repo.close()

    def fail_connector(*args, **kwargs):
        raise AssertionError("source revalidate without --fetch must not call connectors")

    monkeypatch.setattr(cli_app, "ingest_url", fail_connector)
    monkeypatch.setattr(cli_app, "import_local_file", fail_connector)
    monkeypatch.setattr(cli_app, "XBookmarkClient", fail_connector)

    for item_id in (web_card.id, file_item_id, x_item_id):
        result = runner.invoke(
            app,
            ["source", "revalidate", item_id, "--reason", "local bookkeeping", "--json"],
        )
        assert result.exit_code == 0, result.output
        payload = json.loads(result.output)
        assert payload["refetch"]["status"] == "not-requested"
        assert payload["refetch"]["requested"] is False
        assert payload["refetch"]["refetched"] is False

    repo = ContextBankRepository.from_home(home)
    try:
        after_counts = {
            web_source.id: len(repo.list_documents_for_source(web_source.id)),
            file_item_id: len(repo.list_documents_for_source(file_item_id)),
            x_item_id: len(repo.list_documents_for_source(x_item_id)),
        }
        assert after_counts == before_counts
        for item_id in before_counts:
            card = repo.list_knowledge_cards_for_source(item_id)[0]
            assert card.freshness_status == "recheck-soon"
    finally:
        repo.close()


def test_cli_source_revalidate_fetch_identity_mismatch_cleans_artifacts(
    tmp_path,
    monkeypatch,
):
    home = tmp_path / "home"
    monkeypatch.setenv("CONTEXTBANK_HOME", str(home))
    runner.invoke(app, ["init"])

    repo = ContextBankRepository.from_home(home)
    try:
        web_source, _, web_card = seed_card(
            repo,
            url="https://example.com/original-refetch",
            title="Original web source",
            summary="Original source should remain when refetch identity differs.",
        )
    finally:
        repo.close()

    source_file = tmp_path / "note.md"
    source_file.write_text("# Original file\n\nOriginal file source.", encoding="utf-8")
    imported_file = runner.invoke(app, ["import", str(source_file), "--json"])
    assert imported_file.exit_code == 0, imported_file.output
    file_item_id = json.loads(imported_file.output)["item_id"]

    wrong_ids: list[str] = []

    def mismatched_ingest(paths, source_type: SourceType, identity: str):
        wrong_id = source_item_id(source_type.value, value=f"{identity}-wrong")
        wrong_ids.append(wrong_id)
        raw_path = paths.raw / "mismatch" / f"{wrong_id}.raw"
        text_path = paths.documents / f"doc_{wrong_id}.txt"
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        text_path.parent.mkdir(parents=True, exist_ok=True)
        raw_path.write_text("temporary raw artifact", encoding="utf-8")
        text_path.write_text("temporary extracted artifact", encoding="utf-8")
        document_type = "linked_page" if source_type == SourceType.WEB else "file"
        return cli_app.CliIngestResult(
            source_item=SourceItem(
                id=wrong_id,
                source_type=source_type,
                source_external_id=identity,
                source_url=identity if source_type == SourceType.WEB else None,
                raw_path=str(raw_path),
                availability_status=AvailabilityStatus.AVAILABLE,
            ),
            document=Document(
                id=document_id(wrong_id, document_type, "mismatch"),
                source_item_id=wrong_id,
                document_type=document_type,
                canonical_url=identity if source_type == SourceType.WEB else None,
                title="Mismatched refetch",
                extracted_text_path=str(text_path),
                extraction_status=ExtractionStatus.COMPLETE,
                text="Mismatched refetch text.",
            ),
            warnings=["temporary mismatch"],
        )

    def fake_ingest_url(url, paths):
        assert url == "https://example.com/original-refetch"
        return mismatched_ingest(paths, SourceType.WEB, "https://example.com/other-refetch")

    def fake_import_file(file_path, paths):
        assert Path(file_path) == source_file
        return mismatched_ingest(paths, SourceType.FILE, str(tmp_path / "other.md"))

    monkeypatch.setattr(cli_app, "ingest_url", fake_ingest_url)
    monkeypatch.setattr(cli_app, "import_local_file", fake_import_file)

    web_result = runner.invoke(
        app,
        ["source", "revalidate", web_card.id, "--fetch", "--json"],
    )
    file_result = runner.invoke(
        app,
        ["source", "revalidate", file_item_id, "--fetch", "--json"],
    )

    for result in (web_result, file_result):
        assert result.exit_code == 0, result.output
        payload = json.loads(result.output)
        assert payload["refetch"]["status"] == "failed"
        assert payload["refetch"]["refetched"] is False
        assert "different ContextBank id" in payload["refetch"]["warnings"][0]
        deleted_paths = payload["refetch"]["cleanup"]["deleted"]
        assert len(deleted_paths) == 2
        assert all(not Path(path).exists() for path in deleted_paths)
        assert payload["refetch"]["cleanup"]["skipped"] == []

    repo = ContextBankRepository.from_home(home)
    try:
        assert repo.get_source_item(web_source.id) is not None
        assert repo.get_source_item(file_item_id) is not None
        for wrong_id in wrong_ids:
            assert repo.get_source_item(wrong_id) is None
            assert repo.list_documents_for_source(wrong_id) == []
    finally:
        repo.close()


def test_cli_source_revalidate_fetches_web_and_preserves_user_signals(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setenv("CONTEXTBANK_HOME", str(home))
    runner.invoke(app, ["init"])
    url = "https://example.com/revalidate"

    repo = ContextBankRepository.from_home(home)
    try:
        source, _, card = seed_card(
            repo,
            url=url,
            title="Old web source",
            summary="Old source text before revalidation.",
        )
    finally:
        repo.close()

    marked = runner.invoke(app, ["mark", card.id, "--priority", "high", "--pin", "--json"])
    assert marked.exit_code == 0, marked.output

    def fake_ingest_url(requested_url, paths):
        assert requested_url == url
        item_id = source_item_id(SourceType.WEB.value, value=url)
        text = (
            "# Updated web source\n\n"
            "Updated source text says project autoload should refresh useful bookmarks."
        )
        return cli_app.CliIngestResult(
            source_item=SourceItem(
                id=item_id,
                source_type=SourceType.WEB,
                source_external_id=url,
                source_url=url,
                content_hash=content_hash(text),
                availability_status=AvailabilityStatus.AVAILABLE,
            ),
            document=Document(
                id=document_id(item_id, "linked_page", "updated"),
                source_item_id=item_id,
                document_type="linked_page",
                canonical_url=url,
                title="Updated web source",
                extraction_status=ExtractionStatus.COMPLETE,
                content_hash=content_hash(text),
                text=text,
            ),
            warnings=["Updated from saved URL."],
        )

    monkeypatch.setattr(cli_app, "ingest_url", fake_ingest_url)

    result = runner.invoke(
        app,
        ["source", "revalidate", source.id, "--fetch", "--reason", "Freshen", "--json"],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["refetch"]["refetched"] is True
    assert payload["refetch"]["status"] == "refetched"
    assert payload["refetch"]["warnings"] == ["Updated from saved URL."]
    assert payload["card_paths"][0].endswith(f"{card.id}.md")

    repo = ContextBankRepository.from_home(home)
    try:
        updated = repo.get_knowledge_card(card.id)
        assert updated.title == "Updated web source"
        assert updated.personal_priority == "high"
        assert updated.provenance["user_signals"]["pinned"] is True
        assert updated.provenance["source_lifecycle_events"][0]["event_type"] == (
            "revalidation-requested"
        )
    finally:
        repo.close()


def test_cli_source_revalidate_fetches_same_local_file(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setenv("CONTEXTBANK_HOME", str(home))
    runner.invoke(app, ["init"])
    source_file = tmp_path / "note.md"
    source_file.write_text("# Old file context\n\nOld local file body.", encoding="utf-8")

    imported = runner.invoke(app, ["import", str(source_file), "--json"])
    assert imported.exit_code == 0, imported.output
    item_id = json.loads(imported.output)["item_id"]

    source_file.write_text(
        "# Updated file context\n\nUpdated local file body for autoload.",
        encoding="utf-8",
    )

    result = runner.invoke(app, ["source", "revalidate", item_id, "--fetch", "--json"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["refetch"]["refetched"] is True
    assert payload["refetch"]["source_type"] == "file"

    repo = ContextBankRepository.from_home(home)
    try:
        updated = repo.list_knowledge_cards_for_source(item_id)[0]
        documents = repo.list_documents_for_source(item_id)
        assert updated.title == "note.md"
        assert "Updated local file body" in updated.summary
        assert any("Updated local file body" in (document.text or "") for document in documents)
    finally:
        repo.close()


def test_cli_source_revalidate_fetch_reports_manual_text_not_refetchable(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setenv("CONTEXTBANK_HOME", str(home))
    runner.invoke(app, ["init"])

    added = runner.invoke(
        app,
        ["add", "-", "--title", "Manual note", "--json"],
        input="Manual text has no external refetch authority.",
    )
    assert added.exit_code == 0, added.output
    item_id = json.loads(added.output)["item_id"]

    result = runner.invoke(app, ["source", "revalidate", item_id, "--fetch", "--json"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["refetch"]["refetched"] is False
    assert payload["refetch"]["status"] == "not-refetchable"
    assert "Manual text sources" in payload["refetch"]["warnings"][0]

    repo = ContextBankRepository.from_home(home)
    try:
        card = repo.list_knowledge_cards_for_source(item_id)[0]
        assert card.freshness_status == "recheck-soon"
    finally:
        repo.close()


def test_cli_source_revalidate_fetches_x_post_by_id_without_bookmark_scan(
    tmp_path,
    monkeypatch,
):
    home = tmp_path / "home"
    monkeypatch.setenv("CONTEXTBANK_HOME", str(home))
    monkeypatch.setenv("CONTEXTBANK_X_BEARER_TOKEN", "user-token")
    runner.invoke(app, ["init"])

    added = runner.invoke(
        app,
        ["add", "https://x.com/example_user/status/1234567890", "--json"],
    )
    assert added.exit_code == 0, added.output
    item_id = json.loads(added.output)["item_id"]

    @dataclass(frozen=True)
    class FakePost:
        data: dict[str, Any]
        includes: dict[str, Any]
        errors: list[dict[str, Any]]

        @property
        def id(self) -> str:
            return str(self.data.get("id") or "")

    class FakeXClient:
        def __init__(self, token: str) -> None:
            assert token == "user-token"

        def get_post(self, tweet_id: str, **kwargs):
            assert tweet_id == "1234567890"
            return FakePost(
                data={
                    "id": tweet_id,
                    "text": "Fetched X post says targeted revalidation should hydrate saved URLs.",
                    "author_id": "42",
                    "conversation_id": tweet_id,
                    "created_at": "2026-06-20T10:00:00Z",
                    "lang": "en",
                    "public_metrics": {"like_count": 5},
                },
                includes={
                    "users": [{"id": "42", "name": "Example User", "username": "example_user"}]
                },
                errors=[],
            )

    monkeypatch.setattr(cli_app, "XBookmarkClient", FakeXClient)

    result = runner.invoke(app, ["source", "revalidate", item_id, "--fetch", "--json"])

    assert result.exit_code == 0, result.output
    assert "user-token" not in result.output
    payload = json.loads(result.output)
    assert payload["refetch"]["refetched"] is True
    assert payload["refetch"]["status"] == "refetched"
    assert payload["refetch"]["source_type"] == "x"

    repo = ContextBankRepository.from_home(home)
    try:
        source = repo.get_source_item(item_id)
        document = repo.list_documents_for_source(item_id)[-1]
        card = repo.list_knowledge_cards_for_source(item_id)[0]
        assert source.availability_status == "available"
        assert source.author_handle == "example_user"
        assert document.extraction_status == "complete"
        assert document.metadata["conversation_id"] == "1234567890"
        assert "targeted revalidation" in card.summary
    finally:
        repo.close()


def test_cli_x_sync_validates_user_and_persists_markdown_and_raw(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setenv("CONTEXTBANK_HOME", str(home))
    monkeypatch.setenv("CONTEXTBANK_X_BEARER_TOKEN", "user-token")
    runner.invoke(app, ["init"])

    @dataclass(frozen=True)
    class FakePage:
        data: list[dict[str, Any]]
        includes: dict[str, Any]

    class FakeClient:
        def __init__(self, token: str) -> None:
            assert token == "user-token"

        def get_authenticated_user(self) -> dict[str, str]:
            return {"id": "123", "name": "Example", "username": "example_user"}

        def iter_bookmark_pages(self, user_id: str, *, limit: int = 100):
            assert user_id == "123"
            assert limit == 1
            yield FakePage(
                data=[
                    {
                        "id": "tweet-1",
                        "text": "Local-first retrieval note",
                        "author_id": "42",
                        "conversation_id": "conv-1",
                        "created_at": "2026-06-20T12:00:00Z",
                        "lang": "en",
                        "entities": {
                            "urls": [
                                {
                                    "url": "https://t.co/example",
                                    "expanded_url": "https://example.com/article",
                                    "display_url": "example.com/article",
                                    "title": "Example Article",
                                }
                            ]
                        },
                        "attachments": {"media_keys": ["media-1"]},
                        "referenced_tweets": [{"type": "quoted", "id": "tweet-0"}],
                        "public_metrics": {"like_count": 1},
                    }
                ],
                includes={
                    "users": [{"id": "42", "name": "Example Author", "username": "author"}],
                    "media": [
                        {
                            "media_key": "media-1",
                            "type": "photo",
                            "url": "https://pbs.twimg.com/media/example.jpg",
                            "alt_text": "Architecture diagram",
                            "width": 1200,
                            "height": 800,
                        }
                    ],
                    "tweets": [
                        {
                            "id": "tweet-0",
                            "text": "Referenced note",
                            "author_id": "99",
                            "conversation_id": "conv-1",
                        }
                    ],
                },
            )

    monkeypatch.setattr(cli_app, "XBookmarkClient", FakeClient)

    result = runner.invoke(app, ["sync", "x", "--limit", "1", "--json"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["imported"] == 1
    assert list((home / "raw" / "x").glob("*.json"))
    assert list((home / "cards").glob("*/*.md"))

    repo = ContextBankRepository.from_home(home)
    try:
        sources = repo.list_source_items()
        assert sources[0].author_handle == "author"
        document = repo.list_documents_for_source(sources[0].id)[0]
        assert document.metadata["conversation_id"] == "conv-1"
        assert document.metadata["links"][0]["expanded_url"] == "https://example.com/article"
        assert document.metadata["media"][0]["alt_text"] == "Architecture diagram"
        assert document.metadata["media_transcription_status"] == (
            "metadata_only_alt_text_available"
        )
        assert document.metadata["referenced_posts"][0]["text"] == "Referenced note"
        assert repo.count_rows("documents") == 1
        assert repo.count_rows("knowledge_cards") == 1
    finally:
        repo.close()


def test_cli_x_sync_fetches_linked_pages_when_enabled(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setenv("CONTEXTBANK_HOME", str(home))
    monkeypatch.setenv("CONTEXTBANK_X_BEARER_TOKEN", "user-token")
    runner.invoke(app, ["init"])
    fetched_urls: list[str] = []

    @dataclass(frozen=True)
    class FakePage:
        data: list[dict[str, Any]]
        includes: dict[str, Any]
        next_token: str | None = None

    class FakeClient:
        def __init__(self, token: str) -> None:
            assert token == "user-token"

        def get_authenticated_user(self) -> dict[str, str]:
            return {"id": "123"}

        def iter_bookmark_pages(self, user_id: str, *, limit: int = 100):
            assert user_id == "123"
            yield FakePage(
                data=[
                    {
                        "id": "tweet-1",
                        "text": "Bookmark with linked articles",
                        "author_id": "42",
                        "entities": {
                            "urls": [
                                {"expanded_url": "https://example.com/article"},
                                {"expanded_url": "https://example.com/article"},
                                {"expanded_url": "http://127.0.0.1/private"},
                            ]
                        },
                    }
                ],
                includes={"users": [{"id": "42", "username": "author"}]},
            )

    def fake_ingest_added_url(url, paths):
        fetched_urls.append(url)
        if url.startswith("http://127.0.0.1"):
            raise ValueError("Refusing private URL")
        return import_text(f"Linked page text for {url}", paths, source_url=url, title="Linked")

    monkeypatch.setattr(cli_app, "XBookmarkClient", FakeClient)
    monkeypatch.setattr(cli_app, "_ingest_added_url", fake_ingest_added_url)

    result = runner.invoke(app, ["sync", "x", "--fetch-linked-pages", "--json"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["imported"] == 1
    assert payload["linked_pages_imported"] == 1
    assert payload["linked_pages_updated"] == 0
    assert payload["linked_page_errors"] == [
        {
            "bookmark_id": "tweet-1",
            "url": "http://127.0.0.1/private",
            "error": "Refusing private URL",
        }
    ]
    assert fetched_urls == ["https://example.com/article", "http://127.0.0.1/private"]

    repo = ContextBankRepository.from_home(home)
    try:
        assert repo.count_rows("source_items") == 2
        assert repo.count_rows("documents") == 2
    finally:
        repo.close()


def test_cli_x_sync_uses_stored_oauth_token(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setenv("CONTEXTBANK_HOME", str(home))
    runner.invoke(app, ["init"])
    settings = initialize_settings(home)
    save_stored_token(
        settings.paths,
        StoredXToken(
            access_token="stored-access-token",
            client_id="client-123",
            expires_at=datetime(2999, 1, 1, tzinfo=UTC),
        ),
    )

    class FakeClient:
        def __init__(self, token: str) -> None:
            assert token == "stored-access-token"

        def get_authenticated_user(self) -> dict[str, str]:
            return {"id": "123"}

        def iter_bookmark_pages(self, user_id: str, *, limit: int = 100):
            assert user_id == "123"
            assert limit == 0
            return iter(())

    monkeypatch.setattr(cli_app, "XBookmarkClient", FakeClient)

    result = runner.invoke(app, ["sync", "x", "--limit", "0", "--json"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["authenticated_user_id"] == "123"


def test_cli_x_sync_persists_and_resumes_pagination_cursor(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setenv("CONTEXTBANK_HOME", str(home))
    monkeypatch.setenv("CONTEXTBANK_X_BEARER_TOKEN", "user-token")
    runner.invoke(app, ["init"])
    seen_cursors: list[str | None] = []

    @dataclass(frozen=True)
    class FakePage:
        data: list[dict[str, Any]]
        includes: dict[str, Any]
        next_token: str | None

    class FakeClient:
        def __init__(self, token: str) -> None:
            assert token == "user-token"

        def get_authenticated_user(self) -> dict[str, str]:
            return {"id": "123"}

        def iter_bookmark_pages(
            self,
            user_id: str,
            *,
            limit: int = 100,
            pagination_token: str | None = None,
        ):
            assert user_id == "123"
            assert limit == 1
            seen_cursors.append(pagination_token)
            if pagination_token is None:
                yield FakePage(
                    data=[{"id": "tweet-1", "text": "first bookmark"}],
                    includes={},
                    next_token="cursor-1",
                )
            else:
                yield FakePage(
                    data=[{"id": "tweet-2", "text": "second bookmark"}],
                    includes={},
                    next_token=None,
                )

    monkeypatch.setattr(cli_app, "XBookmarkClient", FakeClient)

    first = runner.invoke(app, ["sync", "x", "--limit", "1", "--json"])
    assert first.exit_code == 0, first.output
    first_payload = json.loads(first.output)
    assert first_payload["resumed_from_cursor"] is None
    assert first_payload["next_cursor"] == "cursor-1"
    assert first_payload["full_sync_completed"] is False

    status = runner.invoke(app, ["sync", "status", "--json"])
    assert status.exit_code == 0, status.output
    state = json.loads(status.output)["sync_state"][0]
    assert state["cursor"] == "cursor-1"
    assert state["status"] == "paused"
    assert state["last_completed_at"] is not None

    second = runner.invoke(app, ["sync", "x", "--limit", "1", "--json"])
    assert second.exit_code == 0, second.output
    second_payload = json.loads(second.output)
    assert second_payload["resumed_from_cursor"] == "cursor-1"
    assert second_payload["next_cursor"] is None
    assert second_payload["full_sync_completed"] is True
    assert seen_cursors == [None, "cursor-1"]


def test_cli_x_sync_isolates_per_bookmark_errors(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setenv("CONTEXTBANK_HOME", str(home))
    monkeypatch.setenv("CONTEXTBANK_X_BEARER_TOKEN", "user-token")
    runner.invoke(app, ["init"])

    @dataclass(frozen=True)
    class FakePage:
        data: list[dict[str, Any]]
        includes: dict[str, Any]
        next_token: str | None = None

    class FakeClient:
        def __init__(self, token: str) -> None:
            assert token == "user-token"

        def get_authenticated_user(self) -> dict[str, str]:
            return {"id": "123"}

        def iter_bookmark_pages(
            self,
            user_id: str,
            *,
            limit: int = 100,
            pagination_token: str | None = None,
        ):
            assert user_id == "123"
            assert pagination_token is None
            yield FakePage(
                data=[
                    {"id": "bad", "text": "bad bookmark"},
                    {"id": "good", "text": "good bookmark"},
                ],
                includes={},
            )

    original_ingest = cli_app._ingest_x_bookmark

    def flaky_ingest(bookmark, *args, **kwargs):
        if bookmark.get("id") == "bad":
            raise RuntimeError("cannot persist bookmark")
        return original_ingest(bookmark, *args, **kwargs)

    monkeypatch.setattr(cli_app, "XBookmarkClient", FakeClient)
    monkeypatch.setattr(cli_app, "_ingest_x_bookmark", flaky_ingest)

    result = runner.invoke(app, ["sync", "x", "--limit", "2", "--json"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["imported"] == 1
    assert payload["updated"] == 0
    assert payload["full_sync_completed"] is True
    assert payload["bookmark_errors"] == [
        {"id": "bad", "error": "RuntimeError: cannot persist bookmark"}
    ]

    repo = ContextBankRepository.from_home(home)
    try:
        assert repo.count_rows("source_items") == 1
        assert repo.list_sync_states()[0]["status"] == "complete"
    finally:
        repo.close()


def test_cli_x_sync_reconcile_removed_only_after_verified_full_sync(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setenv("CONTEXTBANK_HOME", str(home))
    monkeypatch.setenv("CONTEXTBANK_X_BEARER_TOKEN", "user-token")
    runner.invoke(app, ["init"])
    kept_add = runner.invoke(app, ["add", "https://x.com/ola/status/111", "--json"])
    removed_add = runner.invoke(app, ["add", "https://x.com/ola/status/222", "--json"])
    assert kept_add.exit_code == 0, kept_add.output
    assert removed_add.exit_code == 0, removed_add.output
    kept_id = json.loads(kept_add.output)["item_id"]
    removed_id = json.loads(removed_add.output)["item_id"]

    @dataclass(frozen=True)
    class FakePage:
        data: list[dict[str, Any]]
        includes: dict[str, Any]
        next_token: str | None = None

    class FakeClient:
        def __init__(self, token: str) -> None:
            assert token == "user-token"

        def get_authenticated_user(self) -> dict[str, str]:
            return {"id": "123"}

        def iter_bookmark_pages(
            self,
            user_id: str,
            *,
            limit: int = 100,
            pagination_token: str | None = None,
        ):
            assert user_id == "123"
            assert pagination_token is None
            yield FakePage(
                data=[{"id": "111", "text": "still bookmarked", "author_id": "42"}],
                includes={"users": [{"id": "42", "username": "ola"}]},
                next_token=None,
            )

    monkeypatch.setattr(cli_app, "XBookmarkClient", FakeClient)

    result = runner.invoke(app, ["sync", "x", "--limit", "100", "--reconcile-removed", "--json"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["full_sync_completed"] is True
    assert payload["removed"] == 1
    assert payload["reconcile_skipped_reason"] is None

    repo = ContextBankRepository.from_home(home)
    try:
        kept = repo.get_source_item(kept_id)
        removed = repo.get_source_item(removed_id)
        assert kept.deleted_locally_at is None
        assert kept.availability_status == "available"
        assert removed.deleted_locally_at is not None
        assert removed.availability_status == "unavailable"
    finally:
        repo.close()


def test_cli_x_sync_skips_reconcile_removed_on_partial_sync(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setenv("CONTEXTBANK_HOME", str(home))
    monkeypatch.setenv("CONTEXTBANK_X_BEARER_TOKEN", "user-token")
    runner.invoke(app, ["init"])
    removed_add = runner.invoke(app, ["add", "https://x.com/ola/status/222", "--json"])
    assert removed_add.exit_code == 0, removed_add.output
    removed_id = json.loads(removed_add.output)["item_id"]

    @dataclass(frozen=True)
    class FakePage:
        data: list[dict[str, Any]]
        includes: dict[str, Any]
        next_token: str | None = "cursor-1"

    class FakeClient:
        def __init__(self, token: str) -> None:
            assert token == "user-token"

        def get_authenticated_user(self) -> dict[str, str]:
            return {"id": "123"}

        def iter_bookmark_pages(self, user_id: str, *, limit: int = 100):
            assert user_id == "123"
            yield FakePage(
                data=[{"id": "111", "text": "first page", "author_id": "42"}],
                includes={"users": [{"id": "42", "username": "ola"}]},
            )

    monkeypatch.setattr(cli_app, "XBookmarkClient", FakeClient)

    result = runner.invoke(app, ["sync", "x", "--limit", "1", "--reconcile-removed", "--json"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["full_sync_completed"] is False
    assert payload["removed"] == 0
    assert payload["reconcile_skipped_reason"] == "sync did not reach the end of pagination"

    repo = ContextBankRepository.from_home(home)
    try:
        removed = repo.get_source_item(removed_id)
        assert removed.deleted_locally_at is None
        assert removed.availability_status == "partial"
    finally:
        repo.close()


def test_cli_x_sync_rejects_mismatched_configured_user(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setenv("CONTEXTBANK_HOME", str(home))
    monkeypatch.setenv("CONTEXTBANK_X_BEARER_TOKEN", "user-token")
    runner.invoke(app, ["init"])
    runner.invoke(app, ["config", "set", "x.user_id", "wrong"])

    class FakeClient:
        def __init__(self, token: str) -> None:
            assert token == "user-token"

        def get_authenticated_user(self) -> dict[str, str]:
            return {"id": "123"}

    monkeypatch.setattr(cli_app, "XBookmarkClient", FakeClient)

    result = runner.invoke(app, ["sync", "x", "--limit", "1"])
    assert result.exit_code != 0
    assert "does not match" in result.output
