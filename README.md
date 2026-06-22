# ContextBank

ContextBank is a local-first personal context layer for AI-assisted work. It stores saved links,
files, and X bookmarks in SQLite, mirrors them into readable Markdown cards, and exposes bounded
read-only retrieval tools through MCP.

This repository starts with the CLI-first MVP spine from the product brief:

- SQLite as the canonical store.
- Markdown cards as a transparent, regenerable mirror.
- Manual URL, pasted text, bulk URL list, and file import that work without X credentials.
- Cost-aware X connector scaffolding for the official bookmark API.
- Offline full-text search and context-pack generation.
- Read-only MCP server tools for agents.
- Open-source, BYOK AI configuration with no bundled provider API keys.

## Quick Start

```bash
python -m pip install -e ".[dev,mcp]"
contextbank init
contextbank add https://docs.x.com/x-api/users/get-bookmarks
contextbank add https://x.com/example_user/status/1234567890
printf "A useful pasted note" | contextbank add - --title "Pasted note"
contextbank import-list urls.md
contextbank import docs/example.md
contextbank search "local-first retrieval"
contextbank review list
contextbank mark <card-or-source-id> --priority high --pin
contextbank project link <card-or-source-id> --project . --reason "Useful for this repo"
contextbank project use <card-or-source-id> --project . --outcome-note "Applied to the agent flow"
contextbank mcp config
contextbank mcp instructions --client all --project . --write
contextbank project autoload . --task "current project goal"
contextbank readiness --project . --json
contextbank backup create
contextbank backup verify ~/.contextbank/exports/backup --json
contextbank mcp serve
```

By default, data lives in `~/.contextbank`. Set `CONTEXTBANK_HOME` to use another directory.

## See What ContextBank Has

Use `sync status` for a quick inventory of the local vault:

```bash
contextbank sync status --json
contextbank source list --counts-only --json
contextbank source list --type x --json
```

`sync status` includes `source_items`, `documents`, `knowledge_cards`, and per-connector sync
state. For X bookmarks, a successful limited sync should show an `x.bookmarks` entry with `status`
such as `paused` or `complete`, `last_error: null`, and either a saved `cursor` for resumable sync
or `full_sync_completed: true`. `source list` is the safer human inventory view: it shows counts by
source type and availability, and can filter to `--type x` to confirm how many X bookmark sources
are currently saved without exposing raw local paths.

Search the saved library directly:

```bash
contextbank search "agent memory" --limit 10
contextbank search "agent memory" --limit 10 --json
contextbank show <card-or-source-id> --json
```

`search` returns source-linked cards. The CLI can show local card text because it runs on the
user's machine; MCP responses are stricter and expose source-authored text through delimited
`source_packet` fields so agents treat it as evidence, not instructions.

Preview exactly what an agent should receive for a project task without using MCP:

```bash
contextbank project autoload /path/to/project --task "what I am working on" --json
contextbank context "what I am working on" --project /path/to/project --json
```

Those commands do not sync X, fetch web pages, call cloud AI, or write to the project. They only
search already saved local ContextBank items and return a bounded MCP-safe context pack. Source
authored card fields are redacted from ordinary JSON fields and available through `source_packet`.
Pass `--raw` only for explicit local human/debug inspection.

`contextbank import-list` accepts newline text, Markdown links, CSV files with a URL-like column,
or JSON arrays/objects containing `url`, `source_url`, `href`, or `link`. Web capture only fetches
`http(s)` URLs and blocks obvious local/private network targets by default.
`contextbank source adapters --json` lists the built-in source adapter contract and capabilities
for `web`, `file`, `text`, and `x.bookmarks`, including which adapters require credentials.
X post URLs are captured as partial X records keyed by post id, so later OAuth sync can hydrate the
same item instead of creating a duplicate.
When a new import matches an existing saved item by source hash, document hash, or canonical URL,
ContextBank preserves the new record for provenance but marks it as a possible duplicate and hides
it from normal agent retrieval. Review the relationship with
`contextbank review approve-duplicate <new-id> --duplicate-of <existing-id>` or
`contextbank review reject-duplicate <new-id> --not-duplicate-of <existing-id>`.

`contextbank mcp config` prints a local MCP client snippet. By default it emits the Codex TOML
`[mcp_servers.contextbank]` shape; pass `--client mcp-json`, `--client claude`,
`--client claude-code`, `--client claude-desktop`, or `--client cursor` for agents that use the
JSON `mcpServers` shape. Use `--home /path/to/vault` when an agent should read a specific
ContextBank vault instead of the default `~/.contextbank`. MCP snippets include only
`CONTEXTBANK_HOME`; provider and X credentials remain BYOK environment or token-store concerns.
`contextbank mcp manifest --json` prints the command, args, tools, autoload contract, and read-only
safety metadata for custom MCP setup flows.
`contextbank readiness --project . --json` runs a no-network preflight that reports whether the
local SQLite store, MCP manifest, project autoload, BYOK config, and X OAuth setup are ready. It
separates `ready_for_local_agent` from `ready_for_x_sync`, so local agent use can be ready even
before the live X OAuth flow is finished. The JSON payload also includes `next_actions`, a
machine-readable list of commands that unblock any remaining local-agent or X-sync gates.

For project startup, `contextbank project autoload . --task "..."` builds the same source-linked
context pack that MCP agents can request with `autoload_project_context`. The MCP server publishes
startup instructions telling compatible agents to call this tool at the start of a project session,
then use the returned cards before planning or implementing. If a client does not surface server
instructions, run `contextbank mcp instructions --client codex --project .` or
`contextbank mcp instructions --client claude-code --project .` and paste the managed block into
the agent's project instructions. Add `--write` to write the block into the conventional local file
such as `AGENTS.md` or `CLAUDE.md`. Use
`contextbank mcp instructions --client all --project . --write` to install managed autoload blocks
for the common Codex, Claude, Cursor, and generic project-instruction files in one step.

Generated instruction files are local setup files. They may contain an absolute path to the user's
repository, so do not publish them to a public repo unless you intentionally sanitize or replace
that path.

### Using ContextBank From Codex Or Another Agent

First connect the MCP server to the agent. For Codex-style TOML:

```bash
contextbank mcp config --client codex --command "$(command -v contextbank)"
```

For JSON-based MCP clients such as Claude Code, Claude Desktop, Cursor, or generic MCP UIs:

```bash
contextbank mcp config --client claude-code --command "$(command -v contextbank)"
contextbank mcp config --client cursor --command "$(command -v contextbank)"
```

Then initialize the target project and install local startup instructions if the client does not
automatically surface MCP server instructions:

```bash
cd /path/to/project
contextbank project init . --name "My Project" --autoload
contextbank project autoload-config . --enabled --token-budget 1500 --output-type implementation
contextbank mcp instructions --client codex --project . --write
```

At the start of a session, the agent should call the MCP tool:

```text
autoload_project_context({
  "project_path": "/path/to/project",
  "task": "the user's current request",
  "token_budget": 1500,
  "desired_output_type": "implementation"
})
```

ContextBank inspects the project profile, lightweight project signals, explicit project links, and
the saved local vault, then returns source-linked cards, stale warnings, source references, and
retrieval metadata. The agent uses that pack before planning or editing. It does not receive write
tools by default and autoload does not perform network or cloud calls.

Autoload can be controlled per repository through `.contextbank/project.yml`:

```bash
contextbank project init . --name "My App" --autoload
contextbank project autoload-config . --disabled
contextbank project autoload-config . --enabled --token-budget 1200 --output-type implementation
```

Autoload reads already saved/imported/synced ContextBank items, explicit project links, the project
profile, and safe local project signals such as common manifest metadata and README headings. It
does not sync X, fetch bookmarks, call cloud AI providers, or refresh web pages during agent
startup. Project links marked `rejected` or `obsolete`, cards marked `rejected` or `hidden`, and
profile `exclude_topics` are excluded from default agent retrieval.

Use `contextbank project link <card-or-source-id> --project . --reason "..."` to persist why a
saved item matters to a repository. Use `contextbank project use <card-or-source-id> --project .`
after an item actually helps; it marks the project link `used`, records an optional outcome note,
and teaches future autoload that this item was useful for that repository. Project links are
explicit local writes, mirrored into the Markdown card, and searchable with
`contextbank search "query" --project <name>`.
`contextbank reprocess <card-or-source-id>` regenerates source-derived fields while preserving
explicit user decisions such as project links, pin/usefulness signals, hidden/rejected review state,
and approved or rejected skill review.

Skill candidates stay inactive until reviewed. Use `contextbank review approve-skill <id>` or
`contextbank review reject-skill <id>` to persist the decision. User signals such as `--pin`,
`--usefulness`, `--used-in-project`, `--obsolete`, `--incorrect`, and `--hide` are recorded through
`contextbank mark`; hidden items remain showable by direct id but are excluded from normal agent
retrieval.
Use `contextbank review correct <card-or-source-id>` to override generated card fields after human
review. Corrections are stored in provenance with previous and corrected values, update the
Markdown card, and survive later reprocessing.
`contextbank review list --queue all` shows local library-health queues including skill review,
duplicates, low confidence, stale, partial, failed, missing-source, hidden, and unprocessed items.

Source lifecycle commands keep local retention explicit:

```bash
contextbank source unavailable <card-or-source-id> --reason "gone upstream"
contextbank source revalidate <card-or-source-id> --reason "check freshness"
contextbank source revalidate <card-or-source-id> --fetch --reason "refresh saved source"
contextbank source delete-raw <card-or-source-id> --yes
contextbank source delete <card-or-source-id> --yes --detach-notes
```

`source revalidate` without `--fetch` is local lifecycle bookkeeping only. `--fetch` rereads the
exact saved web URL or local file path, and can refresh one X post by ID through the official Post
lookup endpoint when an OAuth user token is configured. It preserves user/project review signals;
manual text reports not-refetchable because there is no external source to reread.
Raw-data and full-item deletion require `--yes`. User notes, corrections, and project-use signals
are preserved after source deletion only when `--detach-notes` is supplied.

Backups copy the SQLite database with SQLite's online backup API, include the Markdown card mirror,
and write a sanitized `config.toml` without secret-like keys or secret-looking scalar values. Run
`contextbank backup verify <backup-dir> --json` after creating or moving a backup to check database
integrity, schema version, config parseability, secret-key/value absence, and card-file presence.

## X Credentials

> **You connect your own X account.** ContextBank ingests *your* bookmarks through *your own* X
> Developer app and account — it bundles no credentials and never uses the maintainer's account or
> any shared key. You create a free X Developer app, use its public OAuth Client ID, and authorize
> your own account; the resulting token is stored locally on your machine (`~/.contextbank/`), never
> in the repo. **X sync is optional** — manual URL, file, pasted-text, and bulk-list import all work
> with no X account at all (see [Quick Start](#quick-start)). Current X API access and pricing change
> over time, so check them when you set this up.

ContextBank does not store X access tokens in config files. X bookmarks require an OAuth 2.0
user access token, not an app-only Bearer token. The token needs `tweet.read`, `users.read`, and
`bookmark.read`; add `offline.access` only if you want refresh-token sync.

In the X Developer Console, configure OAuth 2.0 as a Native App and allowlist:

```text
http://127.0.0.1:8765/oauth/x/callback
```

Then run:

```bash
export CONTEXTBANK_X_CLIENT_ID="..."
# or persist the public OAuth Client ID locally:
contextbank config set x.client_id "..."
contextbank auth x-setup
contextbank auth x
contextbank sync x --dry-run --limit 25
contextbank sync x --limit 25
contextbank sync status
```

For automation, `CONTEXTBANK_X_BEARER_TOKEN` is still supported as an environment-token override.
`contextbank auth x-setup --json` includes the secret-free Developer Console checklist,
`next_actions`, and a `data_use_answer` draft for X's data-protection form.
`contextbank sync x` validates that token by calling X. Local readiness commands intentionally
report env-only bearer tokens as scope-unverified because they do not contact X; `contextbank auth x`
stores local scope metadata that readiness can verify.

`contextbank sync x` calls `/2/users/me` first and uses that authenticated user id unless you set
`x.user_id`; if you do set it, ContextBank verifies it matches the token before fetching bookmarks.
Pagination cursors are persisted after each page so interrupted or deliberately limited syncs can
resume on the next run. Use `--no-resume` to start from the beginning without clearing saved state,
or `--reset-cursor` to discard the saved cursor before syncing.
Use `--reconcile-removed` only after a full sync from the beginning; it soft-marks local X bookmark
records that are absent from the completed API result instead of deleting source data.
Use `--fetch-linked-pages` to also import expanded URLs found inside synced bookmarks. Linked pages
use the same guarded web ingestion path as `contextbank add`, so blocked/private URLs are reported
without stopping the rest of the sync.
Rate-limit responses are retried with bounded exponential backoff before surfacing a typed error
that includes reset timing when X returns it.
When available from X, sync preserves conversation IDs, referenced posts, expanded links, media
metadata, image alt text, and public metrics as metadata. Media transcription/OCR is not performed;
that absence is recorded instead of pretending media was fully processed.

The X API uses authenticated user access for bookmarks and current pay-per-use pricing. Always run
dry runs before large backfills. Use `--cost-basis standard` when estimating sync for users who do
not own the developer app.

## BYOK AI Providers

ContextBank’s core is open source and works with `ai.mode = "none"` by default. Optional local or
cloud AI providers are bring-your-own-key: configure provider names, models, local endpoints, and
credential environment-variable names in `config.toml`, but keep actual API key values in your
shell, OS keychain, or secret manager.

```bash
contextbank config set ai.mode cloud
contextbank config set ai.generation_provider openai
contextbank config set ai.generation_model gpt-4.1-mini
contextbank config set ai.generation_credential_env OPENAI_API_KEY
contextbank ai status
```

`contextbank ai status` reports whether the named environment variables are present, but never
prints their values. `contextbank config set` refuses secret-like config keys such as tokens,
passwords, and API-key fields, and rejects values that look like accidentally pasted API keys or
bearer tokens. Configure environment variable names, not the secret values themselves.
When generation is configured, enrich one saved item explicitly with:

```bash
contextbank ai generate-card <card-or-source-id>
```

This calls the configured BYOK provider, requires `ai.allow_cloud = true` for non-loopback
endpoints, wraps source text as untrusted evidence, and validates the returned JSON against
ContextBank's card schema before saving it.

For offline semantic-style retrieval, build the local hash embedding index:

```bash
contextbank ai index-embeddings
contextbank search "related implementation idea" --semantic
```

The built-in `local-hash` embedding provider is deterministic and local. It makes no cloud calls and
uses no bundled API keys.

External OpenAI-compatible embedding providers are explicit BYOK adapters. Configure a provider,
model, endpoint if needed, and the name of an environment variable that already contains your key:

```bash
contextbank config set ai.embedding_provider openai
contextbank config set ai.embedding_model text-embedding-3-small
contextbank config set ai.embedding_credential_env OPENAI_API_KEY
contextbank config set ai.allow_cloud true
contextbank ai index-embeddings --provider configured
```

`ai.allow_cloud` is required for non-loopback endpoints. Loopback OpenAI-compatible endpoints can
use `openai-compatible` with `ai.embedding_base_url` and no bundled key.

## Credits

ContextBank was built and hardened in public with AI pair-programming:

- **Codex** (OpenAI) — primary implementation: ingestion, retrieval, MCP server, connectors, storage, and CLI.
- **Claude** (Anthropic, Opus) — multi-agent security/PRD audits, fixes, and review-gated pull requests.

It's a BYOK, local-first project with no bundled secrets. Contributions are welcome — see
[`CONTRIBUTING.md`](CONTRIBUTING.md) and the security policy in [`SECURITY.md`](SECURITY.md).
