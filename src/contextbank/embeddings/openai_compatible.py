from __future__ import annotations

from urllib.parse import urlsplit

import httpx


class EmbeddingProviderConfigurationError(ValueError):
    """Raised when a configured embedding provider is missing safe BYOK settings."""


class EmbeddingProviderCallError(RuntimeError):
    """Raised when an embedding provider call fails or returns an invalid response."""


class OpenAICompatibleEmbeddingProvider:
    """BYOK embedding provider for OpenAI-compatible `/embeddings` APIs."""

    def __init__(
        self,
        *,
        name: str,
        model_id: str,
        api_key: str | None,
        base_url: str | None = None,
        allow_cloud: bool = False,
        http_client: httpx.Client | None = None,
        timeout: float = 30.0,
    ) -> None:
        cleaned_name = name.strip().lower()
        if cleaned_name not in {"openai", "openai-compatible"}:
            raise EmbeddingProviderConfigurationError(
                "External embedding provider must be 'openai' or 'openai-compatible'."
            )
        if not model_id.strip():
            raise EmbeddingProviderConfigurationError(
                "Set ai.embedding_model before using an external embedding provider."
            )

        self.name = cleaned_name
        self.model_id = model_id.strip()
        self.base_url = (base_url or "https://api.openai.com/v1").rstrip("/")
        self.api_key = (api_key or "").strip()
        self.request_count = 0
        self._client = http_client or httpx.Client(timeout=timeout, trust_env=False)
        self._owns_client = http_client is None

        if not allow_cloud and not _is_loopback_url(self.base_url):
            raise EmbeddingProviderConfigurationError(
                "External embedding provider would contact a non-loopback endpoint. "
                "Set ai.allow_cloud true only after configuring a user-owned credential."
            )
        if self.name == "openai" and not self.api_key:
            raise EmbeddingProviderConfigurationError(
                "Set ai.embedding_credential_env to an environment variable containing "
                "your OpenAI API key before using provider 'openai'."
            )

    @property
    def endpoint(self) -> str:
        return f"{self.base_url}/embeddings"

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        payload = {
            "input": texts,
            "model": self.model_id,
            "encoding_format": "float",
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        try:
            response = self._client.post(self.endpoint, headers=headers, json=payload)
        except httpx.HTTPError as exc:
            raise EmbeddingProviderCallError(
                f"Embedding provider request failed: {exc.__class__.__name__}"
            ) from exc
        self.request_count += 1
        if response.status_code >= 400:
            raise EmbeddingProviderCallError(
                f"Embedding provider returned HTTP {response.status_code}."
            )
        try:
            body = response.json()
        except ValueError as exc:
            raise EmbeddingProviderCallError(
                "Embedding provider returned non-JSON response."
            ) from exc

        data = body.get("data") if isinstance(body, dict) else None
        if not isinstance(data, list):
            raise EmbeddingProviderCallError("Embedding provider response did not include data[].")

        indexed: list[tuple[int, list[float]]] = []
        for item in data:
            if not isinstance(item, dict):
                raise EmbeddingProviderCallError("Embedding provider returned invalid data item.")
            index = item.get("index")
            embedding = item.get("embedding")
            if not isinstance(index, int) or not isinstance(embedding, list):
                raise EmbeddingProviderCallError(
                    "Embedding provider data item is missing index or embedding."
                )
            vector = [float(value) for value in embedding]
            indexed.append((index, vector))

        indexed.sort(key=lambda item: item[0])
        vectors = [vector for _, vector in indexed]
        if len(vectors) != len(texts):
            raise EmbeddingProviderCallError(
                "Embedding provider returned a different number of vectors than inputs."
            )
        return vectors


def _is_loopback_url(url: str) -> bool:
    parsed = urlsplit(url)
    host = (parsed.hostname or "").lower()
    return host in {"localhost", "127.0.0.1", "::1"}
