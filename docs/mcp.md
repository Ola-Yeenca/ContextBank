# MCP Integration

ContextBank exposes a local read-only MCP server over stdio. Any MCP-compatible agent that can
launch a local command can connect with:

```text
command: contextbank
args: mcp serve
env: CONTEXTBANK_HOME=/path/to/contextbank-home
```

Generate a Codex-compatible local MCP configuration snippet with:

```bash
contextbank mcp config
```

Example output:

```toml
[mcp_servers.contextbank]
command = "contextbank"
args = ["mcp", "serve"]

[mcp_servers.contextbank.env]
CONTEXTBANK_HOME = "/Users/example/.contextbank"
```

For MCP clients that use the JSON `mcpServers` shape, run:

```bash
contextbank mcp config --client mcp-json
```

Agent presets:

- Codex: `contextbank mcp config --client codex --command "$(command -v contextbank)"`
- Claude Desktop JSON config: `contextbank mcp config --client claude-desktop --command "$(command -v contextbank)"`
- Claude Code JSON config: `contextbank mcp config --client claude-code --command "$(command -v contextbank)"`
- Claude alias: `contextbank mcp config --client claude --command "$(command -v contextbank)"`
- Cursor: `contextbank mcp config --client cursor --command "$(command -v contextbank)"`
- Generic MCP JSON: `contextbank mcp config --client mcp-json --command "$(command -v contextbank)"`

`claude`, `claude-code`, `claude-desktop`, and `cursor` are aliases for the same generic JSON shape
because those clients typically accept an `mcpServers` command/args/env configuration. If an agent
uses a custom UI, copy the command, args, and environment values from
`contextbank mcp manifest --json`.

Custom vault example:

```bash
contextbank mcp config --client claude-code --home "$PWD/.tmp/contextbank"
```

No API keys are included in MCP snippets. The only environment variable emitted by default is
`CONTEXTBANK_HOME`, so the server reads the intended local vault.

Run the local read-only MCP server with:

```bash
contextbank mcp serve
```

Default tools:

- `search_knowledge`
- `get_knowledge_item`
- `build_context_pack`
- `autoload_project_context`
- `find_skill_candidates`
- `find_related_items`
- `get_source_document`

`build_context_pack` and `autoload_project_context` return selected cards plus source-linked
`source_claims`, `actionable_suggestions`, `conflicts`, stale warnings, source references,
retrieval metadata, and synthesis. Claims and suggestions are derived locally from the saved card
fields; they are not fresh web/X fetches and are not model authority.
Primary retrieval results also include `source_packet` and `source_security` fields. Treat
`source_packet` as the quoted evidence channel: it wraps source-derived card fields in
`CONTEXTBANK_UNTRUSTED_SOURCE_BEGIN` / `CONTEXTBANK_UNTRUSTED_SOURCE_END` markers with a handling
rule telling agents not to follow source text as instructions.
For MCP clients, source-derived card text such as titles, summaries, caveats, claims, and
suggested actions is redacted from ordinary top-level JSON fields and made available through the
packet channel instead. Stable metadata such as IDs, source references, freshness status, review
state, and confidence remains available for ranking and follow-up calls.

## Autoload

ContextBank cannot force every MCP client to accept pushed context. Instead, it exposes an autoload
contract and publishes server instructions that compatible agents receive during MCP initialization:

```text
Call `autoload_project_context` when starting work in a repository. Pass the current project path,
the user's task if known, and a token budget. Use the returned source-linked context pack before
planning or implementing.
```

For clients that do not surface MCP server instructions, put that sentence in the agent's project
instructions manually, or generate a ready-to-copy block:

```bash
contextbank mcp instructions --client codex --project .
contextbank mcp instructions --client claude-code --project .
contextbank mcp instructions --client cursor --project .
contextbank mcp instructions --client all --project . --write
```

Use `--write` to write the managed block into the conventional project file for that client, such
as `AGENTS.md`, `CLAUDE.md`, or `.cursor/rules/contextbank.mdc`. `--client all --write` updates
the common Codex, Claude, Cursor, and generic project-instruction files at once. The generated
manifest also includes `autoload.startup_instruction` and `autoload.instructions_command`. The agent
should pass the real repository path instead of relying on `project_path="."` when its MCP server
process may start outside the active project directory.

Generated instruction files are local setup files. They can include an absolute project path so the
agent can call `autoload_project_context` reliably. Do not publish those generated files to a
public repository unless you intentionally sanitize or replace that path.

For local testing, run:

```bash
contextbank project autoload-config . --enabled --token-budget 1500 --output-type implementation
contextbank project autoload . --task "current project goal" --json
contextbank context "current project goal" --project . --json
contextbank source list --type x --json
contextbank readiness --project . --json
```

`project autoload` and `context` are the easiest way to preview what an agent should see before
opening Codex, Claude, Cursor, or another MCP client. They search only already saved local
ContextBank items; they do not sync X, fetch pages, call cloud AI, or write into the project. Their
default JSON output uses the same MCP-safe context-pack redaction as the server. Pass `--raw` only
for explicit local human/debug inspection.

`readiness` is a local, no-network preflight. It checks the initialized database, read-only MCP
manifest, secret-free MCP environment, project autoload state, autoload network/cloud call counts,
BYOK status, and X OAuth setup. It reports `ready_for_local_agent` separately from
`ready_for_x_sync`. It also reports whether the managed ContextBank Autoload block is installed in
the conventional Codex, Claude Code, Cursor, and generic project instruction files. Missing
instruction files do not make local MCP readiness fail, because compatible MCP clients can still
receive the server startup instruction during initialization.
The JSON payload includes `next_actions`: machine-readable setup commands for any remaining gates,
such as installing managed autoload blocks, configuring `x.client_id`, running `contextbank auth x`,
or validating an environment token with `contextbank sync x --dry-run`.

Autoload inspects the project profile and lightweight project signals such as common manifest
files, safe manifest names/dependencies, and README headings. It prioritizes cards explicitly
linked to the current project by exact project root, then builds a bounded context pack from saved
ContextBank cards. It does not write into the project and does not execute source instructions.
MCP responses avoid echoing absolute project paths back to the client; the caller already knows the
path it supplied, while ContextBank can still use the local root internally for matching.
After a returned card helps, record that local feedback with
`contextbank project use <card-or-source-id> --project . --outcome-note "..."`; future autoload can
use the explicit `used` project signal without giving the MCP server write permissions.
Project links marked `rejected` or `obsolete`, cards marked `hidden` or `rejected`, and profile
`exclude_topics` are excluded from default autoload so past user decisions are respected. Autoload
does not sync X/bookmarks, fetch web pages, reread local files, or call BYOK/cloud AI providers
during agent startup; run `contextbank sync x` or `contextbank source revalidate --fetch`
deliberately when you want fresh source data.

The server is read-only. Imported source content is untrusted evidence, not instructions. Skill
candidates returned by MCP are inactive until reviewed and approved by the user.

Privacy and bounds:

- `get_knowledge_item` returns card and provenance data, but redacts raw document text, raw X
  payload metadata, and local filesystem paths.
- `get_source_document` is the explicit source-body tool. It clamps requested output and wraps the
  returned text in `CONTEXTBANK_UNTRUSTED_SOURCE_*` markers.
- Search and related-item tools clamp result counts and output size at the server boundary.
