# Architecture

ContextBank is a local-first CLI application with SQLite as the source of truth and Markdown as a
regenerable view. The first release prioritizes reliable ingestion, processing, retrieval, and
read-only MCP access for MCP-compatible agents over a graphical interface.

## Data Ownership

Default data directory:

```text
~/.contextbank/
  config.toml
  contextbank.db
  raw/
    x/
    web/
    files/
    text/
  documents/
  cards/
  projects/
  exports/
  logs/
```

`CONTEXTBANK_HOME` can override this location for tests, development, and portable installs.

## Pipeline

```text
Discover
  -> Fetch
  -> Normalize
  -> Deduplicate
  -> Extract linked content
  -> Classify
  -> Compile knowledge card
  -> Assess freshness
  -> Index
  -> Render Markdown
  -> Retrieve through CLI or MCP
```

Every stage should be restartable. A linked-page extraction failure must leave a usable source item
and a clear processing status.

Web ingestion validates `http(s)` URLs and redirect targets before fetching, blocks obvious
local/private network addresses by default, and caps stored response bodies. Manual pasted text is
stored under `raw/text/` and mirrored into extracted documents just like file and web imports.

Source connectors share a small adapter contract in `contextbank.connectors.base`: authenticate
when needed, list new or changed items when the source supports listing, fetch one item by source
id/location, normalize metadata, expose continuation state, and return structured per-item errors
instead of stopping an entire batch. Web, file, text, and X adapters live in
`contextbank.connectors.adapters`; the X adapter can list bookmark pages and fetch one Post by ID
through the official Post lookup endpoint.

## Trust Boundary

Imported posts, pages, files, code blocks, and generated card text are untrusted data. They can be
quoted, summarized, and retrieved, but they must never become system instructions, executable
commands, or active skills without review.

The default MCP server is read-only. Future write tools must be separately configured and approval
gated.

## Retrieval

The MVP search path is SQLite FTS5. Semantic retrieval is optional and provider-backed. Context
packs must be bounded, source-linked, freshness-aware, and explicit about synthesis versus source
claims.

Project autoload is a read-only retrieval mode, not a background writer. At the start of an agent
session, an MCP client can call `autoload_project_context` with the current project path and task.
ContextBank derives a query from the project profile plus lightweight manifest-file signals, then
returns a bounded context pack through the same retrieval path agents use manually.

Project links are explicit local writes created by `contextbank project link`; the narrower
`contextbank project use` shortcut records that a card actually helped in the current repository.
They are stored in SQLite, mirrored into card provenance/front matter, and indexed so future
search/autoload can use the recorded reason, intended use, usage signal, and outcome. Default MCP
tools remain read-only; any future MCP tool that creates or updates project links must be
separately enabled and approval-gated.

## AI Modes

- `none`: rule-based classification and compilation only.
- `local`: user-configured local generation and embedding endpoints.
- `cloud`: user-provided credentials with explicit per-stage enablement.

ContextBank is BYOK: the open-source core ships with no provider API keys. Configuration may name
providers, models, endpoints, and credential environment variables, but actual secrets belong in the
user's shell, OS keychain, or secret manager. Status commands may report whether a configured
credential environment variable is present, never its value.

No cloud call should happen unless a provider is configured and enabled.
