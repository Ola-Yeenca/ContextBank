from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


def utc_now() -> datetime:
    return datetime.now(UTC)


class SourceType(StrEnum):
    X = "x"
    WEB = "web"
    FILE = "file"
    TEXT = "text"


class AvailabilityStatus(StrEnum):
    AVAILABLE = "available"
    PARTIAL = "partial"
    UNAVAILABLE = "unavailable"
    UNKNOWN = "unknown"


class ExtractionStatus(StrEnum):
    PENDING = "pending"
    COMPLETE = "complete"
    PARTIAL = "partial"
    FAILED = "failed"
    SKIPPED = "skipped"


class ReviewStatus(StrEnum):
    UNREVIEWED = "unreviewed"
    APPROVED = "approved"
    REJECTED = "rejected"
    HIDDEN = "hidden"


class SkillCandidateStatus(StrEnum):
    NONE = "none"
    CANDIDATE = "candidate"
    APPROVED = "approved"
    REJECTED = "rejected"


class FreshnessStatus(StrEnum):
    CURRENT = "current"
    RECHECK_SOON = "recheck-soon"
    STALE_RISK = "stale-risk"
    STALE = "stale"
    UNKNOWN = "unknown"


class SourceItem(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    id: str
    source_type: SourceType
    source_external_id: str | None = None
    source_url: str | None = None
    author_id: str | None = None
    author_name: str | None = None
    author_handle: str | None = None
    source_created_at: datetime | None = None
    first_captured_at: datetime = Field(default_factory=utc_now)
    last_synced_at: datetime | None = None
    raw_path: str | None = None
    content_hash: str | None = None
    language: str | None = None
    availability_status: AvailabilityStatus = AvailabilityStatus.UNKNOWN
    deleted_locally_at: datetime | None = None


class Document(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    id: str
    source_item_id: str
    document_type: Literal["source", "linked_page", "file", "manual_text"]
    canonical_url: str | None = None
    title: str | None = None
    author: str | None = None
    published_at: datetime | None = None
    extracted_text_path: str | None = None
    html_snapshot_path: str | None = None
    extraction_status: ExtractionStatus = ExtractionStatus.PENDING
    extraction_error: str | None = None
    extractor_version: str = "builtin-0.1"
    content_hash: str | None = None
    text: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class KnowledgeCard(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    id: str
    source_item_id: str
    schema_version: str = "card.v1"
    title: str
    one_line_summary: str
    summary: str = ""
    key_insights: list[str] = Field(default_factory=list)
    why_useful: str = ""
    possible_applications: list[str] = Field(default_factory=list)
    actionable_next_steps: list[str] = Field(default_factory=list)
    implementation_notes: list[str] = Field(default_factory=list)
    prerequisites: list[str] = Field(default_factory=list)
    patterns: list[str] = Field(default_factory=list)
    caveats: list[str] = Field(default_factory=list)
    source_quality_notes: list[str] = Field(default_factory=list)
    content_type: str = "other"
    utility: list[str] = Field(default_factory=list)
    topics: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    personal_priority: str = "normal"
    freshness_status: FreshnessStatus = FreshnessStatus.UNKNOWN
    freshness_reason: str = ""
    confidence: float = 0.0
    skill_candidate_status: SkillCandidateStatus = SkillCandidateStatus.NONE
    review_status: ReviewStatus = ReviewStatus.UNREVIEWED
    generated_at: datetime = Field(default_factory=utc_now)
    model_provider: str = "none"
    model_id: str = "rule-based"
    prompt_version: str = "rule-based-0.1"
    provenance: dict[str, Any] = Field(default_factory=dict)


class SearchFilters(BaseModel):
    source: str | None = None
    author: str | None = None
    topic: str | None = None
    content_type: str | None = None
    utility: str | None = None
    freshness_status: str | None = None
    skill_candidate_status: str | None = None
    review_status: str | None = None
    project: str | None = None
    min_confidence: float | None = None
    date_from: datetime | None = None
    date_to: datetime | None = None
    used: bool | None = None


class SearchResult(BaseModel):
    card: KnowledgeCard
    score: float
    relevance_reason: str
    source_url: str | None = None
    source_type: str | None = None
    source_created_at: datetime | None = None
    captured_at: datetime | None = None
    excerpt: str | None = None
    source_packet: str | None = None
    source_security: dict[str, Any] = Field(default_factory=dict)


class ContextPack(BaseModel):
    task: str
    output_type: str = "research"
    items: list[SearchResult]
    synthesis: str
    source_packets: list[str] = Field(default_factory=list)
    source_security: dict[str, Any] = Field(default_factory=dict)
    source_claims: list[dict[str, Any]] = Field(default_factory=list)
    actionable_suggestions: list[dict[str, Any]] = Field(default_factory=list)
    conflicts: list[dict[str, Any]] = Field(default_factory=list)
    project_context: dict[str, Any] | None = None
    retrieval_metadata: dict[str, Any] = Field(default_factory=dict)
    stale_warnings: list[str] = Field(default_factory=list)
    source_references: list[str] = Field(default_factory=list)
    budget_used: int
    truncated: bool = False


class IngestResult(BaseModel):
    item_id: str
    document_id: str | None = None
    card_id: str | None = None
    created: bool = True
    status: str = "ok"
    warnings: list[str] = Field(default_factory=list)


class XSyncEstimate(BaseModel):
    requested_limit: int
    estimated_resources: int
    estimated_cost_usd: float
    pricing_note: str
    dry_run: bool = True
