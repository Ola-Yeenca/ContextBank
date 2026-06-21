# Research Notes

These notes capture volatile platform details checked on 2026-06-21. Recheck before a public
release because API access, pricing, and SDK APIs can change.

## X Bookmark API

- Official endpoint: `GET /2/users/{id}/bookmarks`.
- The path `{id}` must match the authenticated source user.
- `max_results` supports `1 <= x <= 100`.
- Pagination uses `pagination_token` and response metadata can include `next_token`.
- Authorization must be OAuth 2.0 user context, not an app-only Bearer token. Required scopes are
  `tweet.read`, `users.read`, and `bookmark.read`.
- Use `GET /2/users/me` to discover the authenticated user id before fetching bookmarks.
- Use `GET /2/tweets/{id}` for targeted Post lookup when a connector needs to fetch one saved X
  item by ID without scanning bookmark pages.
- Access tokens created by OAuth 2.0 Authorization Code with PKCE last two hours by default unless
  the app requests `offline.access`, which enables refresh tokens.
- For local development callback URLs, X docs say to use `http://127.0.0.1`, not `localhost`, and
  the callback must match the Developer Console allowlist exactly.
- X currently describes pay-per-use pricing. Owned Reads for your own app accessing your own
  bookmarks are listed at `$0.001` per resource, while standard Post reads are listed at `$0.005`
  per resource. The Developer Console is the authoritative pricing source.

Sources:

- https://docs.x.com/x-api/users/get-bookmarks
- https://docs.x.com/x-api/posts/get-post-by-id
- https://docs.x.com/x-api/posts/bookmarks/quickstart/bookmarks-lookup
- https://docs.x.com/fundamentals/developer-apps
- https://docs.x.com/fundamentals/authentication/oauth-2-0/authorization-code
- https://docs.x.com/x-api/getting-started/pricing

## MCP Python SDK

- The Python SDK supports building servers that expose tools, resources, and prompts.
- Current examples still show `mcp.server.fastmcp.FastMCP` for simple servers.
- ContextBank's MVP server should use stdio by default for agent integration and keep every tool
  read-only unless the user later enables write tools explicitly.

Sources:

- https://py.sdk.modelcontextprotocol.io/
- Context7 `/modelcontextprotocol/python-sdk` docs query on FastMCP/MCPServer patterns.

## OpenAI-Compatible Embeddings

- The current OpenAI embeddings endpoint is `POST /v1/embeddings`.
- The request includes `input`, `model`, and can request `encoding_format: "float"`.
- The response returns `data[]` items with `index` and `embedding` vectors plus model/usage
  metadata.
- ContextBank should treat this as an explicit BYOK adapter: no key values in config, no bundled
  keys, and no non-loopback provider call unless the user enables `ai.allow_cloud`.

Sources:

- https://platform.openai.com/docs/api-reference/embeddings/create
- https://platform.openai.com/docs/api-reference/chat/create

## OpenAI-Compatible Generation

- OpenAI currently recommends the Responses API for new OpenAI-only integrations, but many local
  and hosted OpenAI-compatible providers still expose `/v1/chat/completions`.
- Chat Completions JSON mode uses `response_format: {"type": "json_object"}` and still requires
  the prompt to explicitly ask for JSON. JSON mode is a syntax guarantee, not a schema guarantee,
  so ContextBank validates the returned object locally before saving generated card fields.
- ContextBank's generation adapter is BYOK and opt-in: no bundled keys, no non-loopback calls
  unless `ai.allow_cloud = true`, and source text is sent as delimited untrusted evidence.

Sources:

- https://developers.openai.com/api/docs/guides/structured-outputs
- https://developers.openai.com/api/docs/guides/migrate-to-responses
- https://platform.openai.com/docs/api-reference/chat/create
