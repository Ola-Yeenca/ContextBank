"""Embedding provider interfaces."""

from contextbank.embeddings.local import LocalHashEmbeddingProvider
from contextbank.embeddings.openai_compatible import (
    EmbeddingProviderCallError,
    EmbeddingProviderConfigurationError,
    OpenAICompatibleEmbeddingProvider,
)

__all__ = [
    "EmbeddingProviderCallError",
    "EmbeddingProviderConfigurationError",
    "LocalHashEmbeddingProvider",
    "OpenAICompatibleEmbeddingProvider",
]
