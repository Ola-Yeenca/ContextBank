from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

FORBIDDEN_SMOKE_OUTPUT = (
    "smoke-parent-secret-token",
    "smoke-parent-secret-openai",
)


def test_python_module_cli_local_happy_path_smoke(tmp_path: Path) -> None:
    home = tmp_path / "home"
    project = tmp_path / "project"
    project.mkdir()
    (project / "pyproject.toml").write_text("[project]\nname = 'smoke'\n", encoding="utf-8")

    original_x_token = os.environ.get("CONTEXTBANK_X_BEARER_TOKEN")
    original_openai_key = os.environ.get("OPENAI_API_KEY")
    os.environ["CONTEXTBANK_X_BEARER_TOKEN"] = FORBIDDEN_SMOKE_OUTPUT[0]
    os.environ["OPENAI_API_KEY"] = FORBIDDEN_SMOKE_OUTPUT[1]
    try:
        env = _smoke_env(home)
    finally:
        _restore_env("CONTEXTBANK_X_BEARER_TOKEN", original_x_token)
        _restore_env("OPENAI_API_KEY", original_openai_key)

    assert "CONTEXTBANK_X_BEARER_TOKEN" not in env
    assert "OPENAI_API_KEY" not in env

    init = _run_contextbank(["init", "--json"], env=env)
    setup = _run_contextbank(["auth", "x-setup", "--json"], env=env)

    source_file = tmp_path / "context.md"
    source_file.write_text("# Old smoke context\n\nOld local note.", encoding="utf-8")
    imported_file = _run_contextbank(["import", str(source_file), "--json"], env=env)
    file_item_id = imported_file["item_id"]
    source_file.write_text(
        "# Updated smoke context\n\nUpdated local context should autoload for the project.",
        encoding="utf-8",
    )
    refetched = _run_contextbank(
        ["source", "revalidate", file_item_id, "--fetch", "--json"],
        env=env,
    )

    captured_x = _run_contextbank(
        ["add", "https://x.com/example_user/status/1234567890", "--json"],
        env=env,
    )
    pasted = _run_contextbank(
        [
            "add",
            "-",
            "--title",
            "Smoke pasted note",
            "--source-url",
            "https://example.com/smoke",
            "--json",
        ],
        env=env,
        input_text="Pasted smoke note for local-first project retrieval.",
    )

    project_init = _run_contextbank(["project", "init", str(project), "--name", "Smoke"], env=env)
    autoload_config = _run_contextbank(
        [
            "project",
            "autoload-config",
            str(project),
            "--enabled",
            "--token-budget",
            "900",
            "--output-type",
            "implementation",
            "--json",
        ],
        env=env,
    )
    instructions = _run_contextbank(
        ["mcp", "instructions", "--client", "codex", "--project", str(project)],
        env=env,
    )
    written_instructions = _run_contextbank(
        ["mcp", "instructions", "--client", "codex", "--project", str(project), "--write"],
        env=env,
    )
    claude_config = json.loads(
        _run_contextbank(
            ["mcp", "config", "--client", "claude-code", "--home", str(home)],
            env=env,
        )
    )
    used = _run_contextbank(
        [
            "project",
            "use",
            file_item_id,
            "--project",
            str(project),
            "--outcome-note",
            "Applied during smoke autoload.",
            "--json",
        ],
        env=env,
    )
    autoload = _run_contextbank(
        [
            "project",
            "autoload",
            str(project),
            "--task",
            "Use refreshed local context",
            "--json",
        ],
        env=env,
    )
    manifest = _run_contextbank(["mcp", "manifest", "--json"], env=env)
    adapters = _run_contextbank(["source", "adapters", "--json"], env=env)
    ai_status = _run_contextbank(["ai", "status", "--json"], env=env)
    embeddings = _run_contextbank(["ai", "index-embeddings", "--json"], env=env)
    backup_path = _run_contextbank(["backup", "create"], env=env)
    backup_verify = _run_contextbank(["backup", "verify", backup_path, "--json"], env=env)
    readiness = _run_contextbank(
        ["readiness", "--project", str(project), "--task", "Use refreshed local context", "--json"],
        env=env,
    )
    doctor = _run_contextbank(["doctor", "--json"], env=env)

    assert init["schema_version"] >= 1
    assert setup["client"]["ready_for_auth"] is True
    assert setup["token"]["sync_status"] == "needs-token"
    assert refetched["refetch"]["refetched"] is True
    assert refetched["refetch"]["source_type"] == "file"
    assert captured_x["item_id"].startswith("cb_x_")
    assert pasted["document_id"].startswith("doc_")
    assert project_init
    assert (project / ".contextbank" / "project.yml").exists()
    assert autoload_config["autoload"]["enabled"] is True
    assert autoload_config["autoload"]["token_budget"] == 900
    assert "autoload_project_context" in instructions
    assert str(project) in instructions
    assert Path(written_instructions).resolve() == (project / "AGENTS.md").resolve()
    assert "ContextBank Autoload" in (project / "AGENTS.md").read_text(encoding="utf-8")
    claude_server = claude_config["mcpServers"]["contextbank"]
    assert claude_server["args"] == ["mcp", "serve"]
    assert claude_server["env"] == {"CONTEXTBANK_HOME": str(home)}
    assert used["project_link"]["project_name"] == "Smoke"
    assert used["project_link"]["status"] == "used"
    assert used["user_signals"]["used_in_projects"] == ["Smoke"]
    assert autoload["items"][0]["card"]["source_item_id"] == file_item_id
    assert "Explicitly linked to this project" in autoload["items"][0]["relevance_reason"]
    assert manifest["autoload"]["tool"] == "autoload_project_context"
    assert manifest["read_only_default"] is True
    assert {adapter["name"] for adapter in adapters["adapters"]} == {
        "file",
        "text",
        "web",
        "x.bookmarks",
    }
    assert ai_status["byok"] is True
    assert ai_status["bundled_api_keys"] is False
    assert ai_status["available_adapters"]["embedding"] == [
        "local-hash",
        "openai",
        "openai-compatible",
    ]
    assert embeddings["provider"] == "local-hash"
    assert embeddings["indexed"] >= 1
    assert backup_verify["valid"] is True
    assert readiness["ready_for_local_agent"] is True
    assert readiness["ready_for_x_sync"] is False
    assert readiness["autoload"]["network_calls"] == 0
    assert readiness["mcp"]["env"] == {"CONTEXTBANK_HOME": str(home)}
    assert doctor["offline_search"] is True
    assert doctor["x"]["oauth_setup"]["client"]["ready_for_auth"] is True


def _smoke_env(home: Path) -> dict[str, str]:
    pythonpath = os.environ.get("PYTHONPATH")
    return {
        "CONTEXTBANK_HOME": str(home),
        "CONTEXTBANK_X_CLIENT_ID": "client-smoke",
        "PATH": os.environ.get("PATH", ""),
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPATH": os.pathsep.join(
            item
            for item in [str(Path.cwd() / "src"), pythonpath]
            if item
        ),
    }


def _restore_env(key: str, value: str | None) -> None:
    if value is None:
        os.environ.pop(key, None)
        return
    os.environ[key] = value


def _run_contextbank(
    args: list[str],
    *,
    env: dict[str, str],
    input_text: str | None = None,
) -> dict[str, object] | str:
    result = subprocess.run(
        [sys.executable, "-m", "contextbank", *args],
        input=input_text,
        text=True,
        capture_output=True,
        env=env,
        check=False,
    )
    combined_output = result.stderr + result.stdout
    for marker in FORBIDDEN_SMOKE_OUTPUT:
        assert marker not in combined_output
    assert result.returncode == 0, result.stderr + result.stdout
    output = result.stdout.strip()
    if args[-1] == "--json":
        return json.loads(output)
    return output
