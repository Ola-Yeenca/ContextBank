from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field

from contextbank.models import ReviewStatus, SkillCandidateStatus

_WORD_RE = re.compile(r"[a-z][a-z0-9+#.\-]{1,}", re.IGNORECASE)


@dataclass(frozen=True)
class ClassificationResult:
    content_type: str
    utility: list[str]
    topics: list[str]
    tags: list[str]
    personal_priority: str
    confidence: float
    skill_candidate_status: SkillCandidateStatus
    review_status: ReviewStatus
    skill_candidate_reasons: list[str] = field(default_factory=list)
    signals: dict[str, list[str]] = field(default_factory=dict)

    @property
    def skill_candidate_active(self) -> bool:
        return (
            self.skill_candidate_status == SkillCandidateStatus.APPROVED
            and self.review_status == ReviewStatus.APPROVED
        )


_CONTENT_TYPE_KEYWORDS: dict[str, tuple[str, ...]] = {
    "implementation_guide": (
        "how to",
        "step",
        "install",
        "setup",
        "configure",
        "implementation",
        "code",
        "api",
        "sdk",
        "cli",
        "example",
        "tutorial",
    ),
    "troubleshooting": (
        "debug",
        "error",
        "fix",
        "failure",
        "failed",
        "traceback",
        "exception",
        "incident",
        "root cause",
        "workaround",
    ),
    "reference": (
        "reference",
        "documentation",
        "docs",
        "specification",
        "schema",
        "contract",
        "parameters",
        "options",
    ),
    "research_note": (
        "research",
        "study",
        "paper",
        "analysis",
        "finding",
        "evidence",
        "benchmark",
        "evaluation",
        "survey",
    ),
    "product_idea": (
        "idea",
        "opportunity",
        "user problem",
        "customer",
        "roadmap",
        "feature",
        "prototype",
        "mvp",
    ),
    "business_strategy": (
        "strategy",
        "pricing",
        "market",
        "positioning",
        "go-to-market",
        "revenue",
        "sales",
        "growth",
    ),
    "announcement": (
        "announcing",
        "launched",
        "release",
        "released",
        "changelog",
        "available today",
        "new version",
    ),
    "opinion": (
        "opinion",
        "i think",
        "essay",
        "perspective",
        "takeaway",
        "lesson",
        "reflection",
    ),
    "prompt_template": (
        "prompt",
        "system message",
        "developer message",
        "instructions for",
        "llm",
        "agent prompt",
    ),
    "dataset": (
        "dataset",
        "csv",
        "spreadsheet",
        "table",
        "rows",
        "columns",
        "data export",
    ),
}


_UTILITY_KEYWORDS: dict[str, tuple[str, ...]] = {
    "build": ("build", "implement", "code", "api", "sdk", "cli", "component", "workflow"),
    "debug": ("debug", "fix", "error", "failure", "traceback", "incident", "workaround"),
    "learn": ("learn", "explain", "concept", "guide", "tutorial", "course", "primer"),
    "reference": ("reference", "docs", "documentation", "schema", "contract", "checklist"),
    "decide": ("compare", "tradeoff", "decision", "strategy", "pricing", "evaluate"),
    "monitor": ("trend", "metric", "dashboard", "alert", "freshness", "risk"),
    "extract_as_skill": ("skill", "playbook", "procedure", "standard operating", "runbook"),
}


_TOPIC_KEYWORDS: dict[str, tuple[str, ...]] = {
    "ai": ("ai", "llm", "model", "embedding", "rag", "agent", "prompt", "openai", "claude"),
    "agents": ("agent", "mcp", "tool call", "context pack", "retrieval", "codex"),
    "python": ("python", "pytest", "pydantic", "typer", "pip", "sqlite"),
    "javascript": ("javascript", "typescript", "react", "node", "npm", "vite", "next.js"),
    "security": (
        "security",
        "prompt injection",
        "trust boundary",
        "sandbox",
        "secrets",
        "auth",
    ),
    "local-first": ("local-first", "offline", "sqlite", "sync", "personal knowledge", "filesystem"),
    "databases": ("database", "sqlite", "postgres", "sql", "vector", "index"),
    "cli": ("cli", "command line", "terminal", "typer", "argparse", "shell"),
    "testing": ("test", "pytest", "unit test", "integration test", "coverage", "fixture"),
    "web": ("web", "browser", "http", "html", "css", "url", "crawler"),
    "product": ("product", "mvp", "roadmap", "feature", "user", "customer"),
    "design": ("design", "ux", "ui", "prototype", "figma", "interaction"),
    "business": ("sales", "market", "pricing", "revenue", "growth", "positioning"),
}


_HIGH_PRIORITY_TERMS = (
    "urgent",
    "critical",
    "must",
    "blocker",
    "security",
    "compliance",
    "breaking change",
    "migration required",
)

_LOW_PRIORITY_TERMS = ("nice to have", "interesting", "someday", "optional", "opinion")

_SKILL_CANDIDATE_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"\bskill\b", "mentions a skill"),
    (r"\bplaybook\b", "looks like a reusable playbook"),
    (r"\brunbook\b", "looks like an operational runbook"),
    (r"\bstandard operating procedure\b|\bsop\b", "describes a reusable procedure"),
    (r"\bworkflow\b", "describes a workflow"),
    (r"\bchecklist\b", "contains checklist language"),
    (r"\bwhen to use\b", "defines when to apply it"),
    (r"\bstep\s+\d+\b|\bsteps?:\b", "contains ordered steps"),
    (r"\btemplate\b", "contains a reusable template"),
    (r"\brubric\b", "contains review criteria"),
)


def classify_text(
    text: str,
    *,
    title: str | None = None,
    source_url: str | None = None,
    metadata: dict[str, object] | None = None,
) -> ClassificationResult:
    """Classify source content with deterministic keyword rules.

    The classifier is intentionally small and inspectable: it produces enough taxonomy for the
    MVP without calling an AI model or treating source text as executable instructions.
    """

    joined = " ".join(part for part in (title or "", text or "", source_url or "") if part)
    lowered = _normalize(joined)
    metadata = metadata or {}

    content_scores = _score_keyword_groups(lowered, _CONTENT_TYPE_KEYWORDS)
    content_type = _best_label(content_scores, fallback="other")

    utility_scores = _score_keyword_groups(lowered, _UTILITY_KEYWORDS)
    utility = _top_labels(utility_scores, limit=4)

    topic_scores = _score_keyword_groups(lowered, _TOPIC_KEYWORDS)
    topics = _top_labels(topic_scores, limit=6)

    skill_status, review_status, skill_reasons = detect_skill_candidate(text, title=title)
    if skill_status == SkillCandidateStatus.CANDIDATE and "extract_as_skill" not in utility:
        utility.append("extract_as_skill")

    tags = _build_tags(
        content_type=content_type,
        utility=utility,
        topics=topics,
        text=lowered,
        source_url=source_url,
        metadata=metadata,
    )
    priority = _priority(lowered, skill_status)
    confidence = _confidence(
        text=text,
        content_scores=content_scores,
        utility=utility,
        topics=topics,
        skill_reasons=skill_reasons,
    )

    return ClassificationResult(
        content_type=content_type,
        utility=utility or ["reference"],
        topics=topics,
        tags=tags,
        personal_priority=priority,
        confidence=confidence,
        skill_candidate_status=skill_status,
        review_status=review_status,
        skill_candidate_reasons=skill_reasons,
        signals={
            "content_type": _matched_terms(lowered, _CONTENT_TYPE_KEYWORDS.get(content_type, ())),
            "utility": _signals_for_labels(lowered, utility, _UTILITY_KEYWORDS),
            "topics": _signals_for_labels(lowered, topics, _TOPIC_KEYWORDS),
        },
    )


def detect_skill_candidate(
    text: str,
    *,
    title: str | None = None,
) -> tuple[SkillCandidateStatus, ReviewStatus, list[str]]:
    """Detect skill candidates while keeping them inactive until human review."""

    lowered = _normalize(" ".join(part for part in (title or "", text or "") if part))
    reasons: list[str] = []
    for pattern, reason in _SKILL_CANDIDATE_PATTERNS:
        if re.search(pattern, lowered):
            reasons.append(reason)

    has_structure = len(re.findall(r"(?m)^\s*(?:[-*]|\d+[.)])\s+", text or "")) >= 2
    has_application_rule = bool(re.search(r"\bwhen to use\b|\buse this when\b", lowered))
    has_steps = bool(re.search(r"\bstep\s+\d+\b|\bsteps?:\b", lowered))
    if has_structure and has_steps and "contains ordered steps" not in reasons:
        reasons.append("contains ordered steps")
    if has_structure and has_application_rule and "defines when to apply it" not in reasons:
        reasons.append("defines when to apply it")

    if len(reasons) >= 2 or (has_steps and has_application_rule):
        return SkillCandidateStatus.CANDIDATE, ReviewStatus.UNREVIEWED, _dedupe(reasons)

    return SkillCandidateStatus.NONE, ReviewStatus.UNREVIEWED, []


def _normalize(value: str) -> str:
    return re.sub(r"\s+", " ", value.lower()).strip()


def _score_keyword_groups(text: str, groups: dict[str, tuple[str, ...]]) -> Counter[str]:
    scores: Counter[str] = Counter()
    for label, keywords in groups.items():
        for keyword in keywords:
            if _keyword_present(text, keyword):
                scores[label] += 2 if " " in keyword else 1
    return scores


def _keyword_present(text: str, keyword: str) -> bool:
    if not keyword:
        return False
    if re.search(r"[^a-z0-9]", keyword):
        return keyword in text
    return bool(re.search(rf"\b{re.escape(keyword)}\b", text))


def _best_label(scores: Counter[str], *, fallback: str) -> str:
    if not scores:
        return fallback
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))[0][0]


def _top_labels(scores: Counter[str], *, limit: int) -> list[str]:
    ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    return [label for label, score in ranked[:limit] if score > 0]


def _signals_for_labels(
    text: str,
    labels: Iterable[str],
    groups: dict[str, tuple[str, ...]],
) -> list[str]:
    signals: list[str] = []
    for label in labels:
        signals.extend(_matched_terms(text, groups.get(label, ())))
    return _dedupe(signals)[:12]


def _matched_terms(text: str, keywords: Iterable[str]) -> list[str]:
    return [keyword for keyword in keywords if _keyword_present(text, keyword)]


def _build_tags(
    *,
    content_type: str,
    utility: list[str],
    topics: list[str],
    text: str,
    source_url: str | None,
    metadata: dict[str, object],
) -> list[str]:
    tags = [content_type, *utility, *topics]

    domain_match = re.search(r"https?://(?:www\.)?([^/\s]+)", source_url or "")
    if domain_match:
        tags.append(domain_match.group(1).lower())

    for key in ("project", "author", "language"):
        value = metadata.get(key)
        if isinstance(value, str) and value.strip():
            tags.append(value.strip().lower())

    for token in _WORD_RE.findall(text):
        if token in {"mcp", "rag", "sqlite", "pydantic", "typer", "pytest", "openai"}:
            tags.append(token)

    return _dedupe(_slug_tag(tag) for tag in tags if tag)[:20]


def _priority(text: str, skill_status: SkillCandidateStatus) -> str:
    if any(term in text for term in _HIGH_PRIORITY_TERMS):
        return "high"
    if skill_status == SkillCandidateStatus.CANDIDATE:
        return "review"
    if any(term in text for term in _LOW_PRIORITY_TERMS):
        return "low"
    return "normal"


def _confidence(
    *,
    text: str,
    content_scores: Counter[str],
    utility: list[str],
    topics: list[str],
    skill_reasons: list[str],
) -> float:
    length_bonus = min(len((text or "").split()) / 400, 0.18)
    signal_bonus = min((sum(content_scores.values()) + len(utility) + len(topics)) / 30, 0.35)
    skill_bonus = 0.08 if skill_reasons else 0.0
    return round(min(0.35 + length_bonus + signal_bonus + skill_bonus, 0.92), 2)


def _slug_tag(value: str) -> str:
    value = value.lower().strip()
    value = re.sub(r"[^a-z0-9.+#-]+", "-", value).strip("-")
    return value or "uncategorized"


def _dedupe(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result
