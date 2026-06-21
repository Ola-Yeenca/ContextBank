from __future__ import annotations

import json

import httpx
import pytest

from contextbank.embeddings import (
    EmbeddingProviderCallError,
    EmbeddingProviderConfigurationError,
    LocalHashEmbeddingProvider,
    OpenAICompatibleEmbeddingProvider,
)


def test_local_hash_embedding_is_deterministic_and_normalized():
    provider = LocalHashEmbeddingProvider(dimensions=16)

    first = provider.embed(["ContextBank semantic retrieval"])[0]
    second = provider.embed(["ContextBank semantic retrieval"])[0]
    empty = provider.embed([""])[0]

    assert first == second
    assert len(first) == 16
    assert round(sum(value * value for value in first), 6) == 1.0
    assert empty == [0.0] * 16


def test_openai_compatible_embedding_provider_uses_byok_request_shape():
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["authorization"] = request.headers.get("authorization")
        seen["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "object": "list",
                "data": [
                    {"object": "embedding", "index": 1, "embedding": [0.0, 1.0]},
                    {"object": "embedding", "index": 0, "embedding": [1.0, 0.0]},
                ],
                "model": "text-embedding-3-small",
                "usage": {"prompt_tokens": 4, "total_tokens": 4},
            },
        )

    provider = OpenAICompatibleEmbeddingProvider(
        name="openai",
        model_id="text-embedding-3-small",
        api_key="fakekey",
        allow_cloud=True,
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    vectors = provider.embed(["alpha", "beta"])

    assert vectors == [[1.0, 0.0], [0.0, 1.0]]
    assert seen["url"] == "https://api.openai.com/v1/embeddings"
    assert seen["authorization"] == "Bearer fakekey"
    assert seen["body"] == {
        "input": ["alpha", "beta"],
        "model": "text-embedding-3-small",
        "encoding_format": "float",
    }
    assert provider.request_count == 1


def test_openai_compatible_embedding_provider_disables_env_proxy_by_default(monkeypatch):
    captured: dict[str, object] = {}

    class FakeClient:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def close(self):
            return None

    monkeypatch.setattr("contextbank.embeddings.openai_compatible.httpx.Client", FakeClient)

    provider = OpenAICompatibleEmbeddingProvider(
        name="openai",
        model_id="text-embedding-3-small",
        api_key="fakekey",
        allow_cloud=True,
    )
    provider.close()

    assert captured["trust_env"] is False


def test_openai_compatible_embedding_provider_is_cloud_gated_and_validates_response():
    with pytest.raises(EmbeddingProviderConfigurationError):
        OpenAICompatibleEmbeddingProvider(
            name="openai",
            model_id="text-embedding-3-small",
            api_key="fakekey",
            allow_cloud=False,
        )

    provider = OpenAICompatibleEmbeddingProvider(
        name="openai-compatible",
        model_id="local-embeddings",
        api_key=None,
        base_url="http://127.0.0.1:11434/v1",
        allow_cloud=False,
        http_client=httpx.Client(
            transport=httpx.MockTransport(lambda request: httpx.Response(200, json={"data": []}))
        ),
    )

    with pytest.raises(EmbeddingProviderCallError):
        provider.embed(["one input"])
