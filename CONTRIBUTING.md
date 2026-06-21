# Contributing To ContextBank

Thanks for helping build ContextBank.

## Development

```bash
python -m pip install -e ".[dev,mcp]"
python -m pytest -q -p no:cacheprovider
python -m ruff check src tests
```

## Privacy And Secrets

Before committing, check that you are not adding local runtime data:

```bash
git status --short --ignored
git ls-files -o --exclude-standard
```

Do not commit:

- `.env` files or shell exports
- OAuth access or refresh tokens
- X app client secrets
- OpenAI or other provider API keys
- `~/.contextbank` vault databases, raw files, cards, exports, or logs
- generated local agent instruction files such as `AGENTS.md`, `CLAUDE.md`, or `.cursor/` unless
  they are deliberately sanitized examples
- local audit/evidence files containing account IDs, cursors, browser paths, or private source IDs

Use fake example handles and IDs in tests and docs. Do not use a real personal account, browser
profile, local path, bookmark, or generated credential as a fixture.

## Security-Sensitive Changes

Add focused tests when changing:

- MCP source redaction and source-packet delimiting
- X OAuth or bookmark sync
- BYOK provider calls
- web fetching and SSRF controls
- backup/export redaction
- local file path handling

Run the full test suite and Ruff before opening a pull request.
