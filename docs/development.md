# Development

Use Python 3.11 or newer.

```bash
python -m pip install -e ".[dev,mcp]"
pytest
ruff check src tests
```

`tests/test_e2e_smoke.py` runs a real `python -m contextbank` subprocess journey against an
isolated local vault: init, X OAuth setup preflight, manual X capture, file import/refetch, project
linking, autoload, source adapter manifest, local embedding indexing, backup verification, doctor,
and MCP manifest. It does not contact X or any cloud provider.

See `docs/e2e-readiness.md` for the current project-level E2E validation matrix, browser callback
smoke evidence, active MCP autoload state, and the remaining live X OAuth gate.

For isolated local data during development:

```bash
export CONTEXTBANK_HOME="$(pwd)/.tmp/contextbank"
contextbank init
```

Useful local smoke commands:

```bash
contextbank add https://example.com/article
contextbank add https://x.com/example_user/status/1234567890
printf "Copied research note" | contextbank add - --title "Copied note"
contextbank import-list urls.md
contextbank import ./notes.md
contextbank search "research note"
contextbank review list
contextbank source adapters --json
contextbank review approve-skill <card-or-source-id>
contextbank review approve-duplicate <card-or-source-id> --duplicate-of <existing-card-or-source-id>
contextbank review reject-duplicate <card-or-source-id> --not-duplicate-of <existing-card-or-source-id>
contextbank mark <card-or-source-id> --priority high --usefulness 5 --pin
contextbank source revalidate <card-or-source-id> --reason "needs freshness check"
contextbank source revalidate <card-or-source-id> --fetch --reason "refresh saved source"
contextbank ai status
contextbank backup create
contextbank backup verify "$CONTEXTBANK_HOME/exports/backup" --json
```

MCP/autoload smoke commands:

```bash
contextbank project init . --name "ContextBank Dev" --autoload
contextbank project use <card-or-source-id> --project . --outcome-note "Applied during smoke"
contextbank project autoload-config . --enabled --token-budget 1500
contextbank project autoload . --task "current project goal" --json
contextbank mcp config --client codex
contextbank mcp config --client claude-code
contextbank mcp instructions --client codex --project .
contextbank mcp instructions --client all --project . --write
contextbank mcp manifest --json
contextbank readiness --project . --json
```

Review queues:

```bash
contextbank review list --queue all --json
contextbank review list --queue low-confidence --json
contextbank review list --queue failed --json
```

Supported queues are `all`, `skill-review`, `low-confidence`, `stale`, `partial`, `failed`,
`missing-source`, `duplicates`, `hidden`, and `unprocessed`. The queue is derived from current local source,
document, and card state; it does not contact external services.

`contextbank import-list` parses plain text, Markdown links, CSV URL columns, and JSON fields named
`url`, `source_url`, `href`, or `link`. URL ingestion blocks obvious local/private network targets
by default and stores at most the configured response-body cap.
Manual X post URLs are parsed into partial `source_type=x` records using the post id; X OAuth sync
uses the same ids and can later hydrate the source text and metadata.

`contextbank review approve-skill` and `contextbank review reject-skill` accept either a card id or
source item id. `contextbank mark --hide` removes a card from normal search/context retrieval while
leaving direct `contextbank show <id>` access available for audit and restoration with `--unhide`.
Possible duplicate imports appear in `contextbank review list --queue duplicates`; approve a match
to keep it hidden as a confirmed duplicate, or reject the match to make the preserved item eligible
for normal retrieval again.
`contextbank review correct` updates generated card fields after human review while storing the
previous values in `provenance.correction_history`.
`contextbank reprocess <card-or-source-id>` refreshes the generated card from source documents, but
must preserve manual review decisions, project links, and user signals such as pinned/usefulness
state.
`contextbank source revalidate <card-or-source-id>` only marks freshness locally unless `--fetch` is
passed. With `--fetch`, web and file sources are refreshed from the exact saved URL/path, and X
sources are refreshed by one targeted Post lookup when an OAuth user token is configured. Manual
text reports not-refetchable because it has no external authority to reread.

For BYOK provider checks, configure environment-variable names rather than key values:

```bash
contextbank config set ai.mode cloud
contextbank config set ai.generation_provider openai
contextbank config set ai.generation_model gpt-4.1-mini
contextbank config set ai.generation_credential_env OPENAI_API_KEY
contextbank ai status
contextbank ai generate-card <card-or-source-id> --json
```

`ai generate-card` is the explicit BYOK text-generation path. It wraps source text in
`CONTEXTBANK_UNTRUSTED_SOURCE_*` evidence markers, requests JSON output from an OpenAI-compatible
chat-completions endpoint, validates the returned card fields locally, and then writes the Markdown
mirror. Non-loopback endpoints require `ai.allow_cloud = true`; loopback `openai-compatible`
endpoints can run without cloud mode.

Semantic-style local retrieval can be exercised without any cloud provider:

```bash
contextbank ai index-embeddings --json
contextbank search "local retrieval" --semantic --json
```

The built-in `local-hash` embedding provider is deterministic, offline, and reports
`cloud_calls: 0`. It proves the vector index and hybrid search path while preserving the BYOK rule
for future external providers.

External embedding adapters are BYOK and opt-in:

```bash
contextbank config set ai.embedding_provider openai
contextbank config set ai.embedding_model text-embedding-3-small
contextbank config set ai.embedding_credential_env OPENAI_API_KEY
contextbank config set ai.allow_cloud true
contextbank ai index-embeddings --provider configured --json
```

Do not put the key value in `config.toml`; only store the environment variable name. Non-loopback
endpoints require `ai.allow_cloud = true`. Loopback OpenAI-compatible endpoints can use provider
`openai-compatible` plus `ai.embedding_base_url`.

Live X sync requires a user access token with bookmark read access:

```bash
export CONTEXTBANK_X_CLIENT_ID="<oauth-client-id>"
# or persist the public OAuth Client ID locally:
contextbank config set x.client_id "<oauth-client-id>"
contextbank auth x-setup --json
contextbank auth x
contextbank sync x --dry-run --limit 100
contextbank sync x --limit 100
contextbank sync status
```

The token must be OAuth 2.0 user context with `tweet.read`, `users.read`, and `bookmark.read`.
ContextBank discovers the authenticated user with `/2/users/me`. Setting `x.user_id` is optional;
when set, sync fails if it does not match the token owner.
`contextbank auth x-setup` is a local preflight: it checks the callback URL, client ID presence,
stored token state, and required scopes without contacting X or printing secrets.
Its JSON output also includes `next_actions` and `data_use_answer`, so the X Developer Console
setup steps and data-protection form answer can be copied from a local, secret-free command.

Use `contextbank auth x --offline` to request `offline.access` and allow refresh-token sync.
For CI or short-lived automation, `CONTEXTBANK_X_BEARER_TOKEN` can override the stored token.
Because readiness does not call X, env-only bearer tokens are reported as
`env-token-present-scopes-unverified`; run `contextbank sync x` to validate the token against X or
`contextbank auth x` to store local scope metadata for future readiness checks.
X bookmark pagination cursors are saved in local `sync_state`; use `--no-resume` to ignore the
cursor for one run or `--reset-cursor` to clear it before syncing.
`contextbank sync x --reconcile-removed` only soft-marks missing local X records after a verified
full sync from the beginning. Partial or resumed syncs skip reconciliation so absent pages are not
mistaken for removed bookmarks.
`contextbank sync x --fetch-linked-pages` imports expanded URLs from bookmark entities through the
normal guarded URL ingestion path. Per-link failures are returned in the sync result and do not stop
other bookmarks or links from processing.
HTTP 429 responses retry with bounded exponential backoff before surfacing an `XRateLimitError`;
long platform reset windows are reported rather than slept through indefinitely.
The X connector requests post fields, media fields, and expansions needed for conversation IDs,
referenced posts, links, media URLs, media dimensions, public media metrics, and image alt text.
Public metrics remain metadata only and are not used as truth or quality scores.

Connector contributors should implement the adapter protocol in `contextbank.connectors.base`.
Adapters must return structured warnings/errors for per-item failures, expose continuation state
when listing is supported, and provide `fetch_one` for revalidation or targeted imports. The current
web, file, text, and X adapters are covered in `tests/test_ingestion.py`.

Do not commit local `.contextbank` data or credentials.
