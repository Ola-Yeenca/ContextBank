from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class GenerationRequest:
    source_text: str
    task: str
    schema_name: str
    max_output_chars: int = 8000


@dataclass(frozen=True)
class GenerationResponse:
    text: str
    provider: str
    model: str
    input_chars: int
    output_chars: int
    estimated_cost_usd: float | None = None


class TextGenerationProvider(Protocol):
    name: str
    model_id: str

    def generate_structured(self, request: GenerationRequest) -> GenerationResponse:
        """Generate source-grounded structured text."""


class EmbeddingProvider(Protocol):
    name: str
    model_id: str

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one embedding vector per input text."""


class NoAIProvider:
    name = "none"
    model_id = "rule-based"

    def generate_structured(self, request: GenerationRequest) -> GenerationResponse:
        return GenerationResponse(
            text="",
            provider=self.name,
            model=self.model_id,
            input_chars=len(request.source_text),
            output_chars=0,
            estimated_cost_usd=0.0,
        )

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[] for _ in texts]
