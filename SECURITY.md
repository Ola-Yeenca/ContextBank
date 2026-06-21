# Security Policy

## Supported Versions

ContextBank is pre-1.0. Please report security issues against the current `main` branch unless a
release branch exists for the affected version.

## Reporting A Vulnerability

Do not open a public issue for suspected secrets exposure, token handling flaws, SSRF bypasses,
prompt-injection trust-boundary bugs, local file disclosure, or authentication issues.

Report privately through GitHub's private vulnerability reporting feature when it is enabled for
the repository. If private reporting is not available yet, open a minimal public issue that says a
private security contact is needed, without exploit details or sensitive data.

Useful reports include:

- affected command, MCP tool, or connector
- expected and observed behavior
- minimal reproduction steps
- whether the issue can expose local files, credentials, source text, or private bookmarks
- relevant ContextBank version or commit

## Security Model

ContextBank is local-first and BYOK. The project should not ship provider API keys, X client
secrets, OAuth access tokens, vault databases, raw source archives, or private bookmark data.
Imported content is treated as untrusted evidence, not as executable instructions.
