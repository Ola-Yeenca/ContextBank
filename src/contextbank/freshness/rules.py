from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import UTC, datetime

from contextbank.models import FreshnessStatus


@dataclass(frozen=True)
class FreshnessResult:
    status: FreshnessStatus
    score: float
    reason: str
    content_type: str
    age_days: int | None = None
    date_used: datetime | None = None
    date_cues: list[str] = field(default_factory=list)
    version_cues: list[str] = field(default_factory=list)


_CONTENT_TYPE_WINDOWS: dict[str, tuple[int, int, int]] = {
    "announcement": (30, 90, 180),
    "implementation_guide": (180, 365, 730),
    "troubleshooting": (180, 365, 730),
    "reference": (180, 365, 730),
    "prompt_template": (180, 365, 730),
    "product_idea": (365, 730, 1095),
    "business_strategy": (365, 730, 1095),
    "research_note": (365, 730, 1460),
    "dataset": (365, 730, 1095),
    "opinion": (730, 1460, 2190),
    "other": (365, 730, 1095),
}

_MONTHS = {
    "january": 1,
    "jan": 1,
    "february": 2,
    "feb": 2,
    "march": 3,
    "mar": 3,
    "april": 4,
    "apr": 4,
    "may": 5,
    "june": 6,
    "jun": 6,
    "july": 7,
    "jul": 7,
    "august": 8,
    "aug": 8,
    "september": 9,
    "sep": 9,
    "sept": 9,
    "october": 10,
    "oct": 10,
    "november": 11,
    "nov": 11,
    "december": 12,
    "dec": 12,
}

_VERSION_RE = re.compile(
    r"\b(?:v(?:ersion)?\s*\d+\.\d+(?:\.\d+)?|\d+\.\d+\.\d+)(?:-[a-z0-9.]+)?\b",
    re.IGNORECASE,
)
_NEGATIVE_VERSION_CUES = (
    "deprecated",
    "unmaintained",
    "archived",
    "sunset",
    "end-of-life",
    "end of life",
    "no longer supported",
    "replaced by",
    "outdated",
    "out of date",
)
_VOLATILE_CUES = (
    "api",
    "sdk",
    "cli",
    "pricing",
    "changelog",
    "release notes",
    "migration",
    "breaking change",
    "latest",
)


def score_freshness(
    *,
    content_type: str,
    text: str = "",
    published_at: datetime | None = None,
    source_created_at: datetime | None = None,
    captured_at: datetime | None = None,
    as_of: datetime | None = None,
) -> FreshnessResult:
    """Score freshness using dates, content type volatility, and version language."""

    now = _ensure_aware(as_of) or datetime.now(UTC)

    def _past(values: list[datetime | None]) -> list[datetime]:
        return [date for date in values if date is not None and date <= now]

    # Authoritative publication dates. Text cues and capture time must never inflate these:
    # a bare year or copyright footer mentioned in a 2020 doc must not make it look current.
    explicit_dates = _past([_ensure_aware(published_at), _ensure_aware(source_created_at)])
    precise_text_dates, bare_year_dates = _text_dates(text, now)
    precise_text_dates = _past(precise_text_dates)
    bare_year_dates = _past(bare_year_dates)
    captured_dates = _past([_ensure_aware(captured_at)])

    if explicit_dates:
        date_used: datetime | None = max(explicit_dates)
    elif precise_text_dates:
        date_used = max(precise_text_dates)
    elif bare_year_dates:
        date_used = max(bare_year_dates)
    elif captured_dates:
        date_used = max(captured_dates)
    else:
        date_used = None

    date_candidates = explicit_dates + precise_text_dates + bare_year_dates + captured_dates
    version_cues = _version_cues(text)
    negative_cues = [cue for cue in _NEGATIVE_VERSION_CUES if cue in text.lower()]

    if negative_cues:
        return FreshnessResult(
            status=FreshnessStatus.STALE,
            score=0.05,
            reason=f"Marked stale by cue: {negative_cues[0]}.",
            content_type=content_type,
            age_days=(now - date_used).days if date_used else None,
            date_used=date_used,
            date_cues=_format_dates(date_candidates),
            version_cues=version_cues + negative_cues,
        )

    windows = _CONTENT_TYPE_WINDOWS.get(content_type, _CONTENT_TYPE_WINDOWS["other"])
    volatile = _has_any(text, _VOLATILE_CUES) or bool(version_cues)

    if date_used is None:
        if volatile:
            return FreshnessResult(
                status=FreshnessStatus.RECHECK_SOON,
                score=0.48,
                reason="Technical or version-sensitive content has no reliable date.",
                content_type=content_type,
                date_cues=[],
                version_cues=version_cues,
            )
        return FreshnessResult(
            status=FreshnessStatus.UNKNOWN,
            score=0.4,
            reason="No reliable freshness date was found.",
            content_type=content_type,
            date_cues=[],
            version_cues=version_cues,
        )

    age_days = max((now - date_used).days, 0)
    status = _status_for_age(age_days, windows)
    score = _score_for_age(age_days, windows)
    reason = _reason_for_status(status, content_type, age_days, windows)

    if volatile and status == FreshnessStatus.CURRENT and age_days > windows[0] // 2:
        status = FreshnessStatus.RECHECK_SOON
        score = min(score, 0.68)
        reason = "Version-sensitive content is still recent but should be rechecked soon."

    return FreshnessResult(
        status=status,
        score=score,
        reason=reason,
        content_type=content_type,
        age_days=age_days,
        date_used=date_used,
        date_cues=_format_dates(date_candidates),
        version_cues=version_cues,
    )


def _status_for_age(age_days: int, windows: tuple[int, int, int]) -> FreshnessStatus:
    current, recheck, stale_risk = windows
    if age_days <= current:
        return FreshnessStatus.CURRENT
    if age_days <= recheck:
        return FreshnessStatus.RECHECK_SOON
    if age_days <= stale_risk:
        return FreshnessStatus.STALE_RISK
    return FreshnessStatus.STALE


def _score_for_age(age_days: int, windows: tuple[int, int, int]) -> float:
    _, _, stale_risk = windows
    ratio = min(age_days / max(stale_risk, 1), 1)
    return round(max(1 - ratio, 0.05), 2)


def _reason_for_status(
    status: FreshnessStatus,
    content_type: str,
    age_days: int,
    windows: tuple[int, int, int],
) -> str:
    current, recheck, stale_risk = windows
    if status == FreshnessStatus.CURRENT:
        return f"{content_type} is {age_days} days old, within the {current}-day current window."
    if status == FreshnessStatus.RECHECK_SOON:
        return f"{content_type} is {age_days} days old; recheck after {current} days."
    if status == FreshnessStatus.STALE_RISK:
        return f"{content_type} is {age_days} days old and beyond the {recheck}-day comfort window."
    return f"{content_type} is {age_days} days old, beyond the {stale_risk}-day stale window."


def _text_dates(text: str, now: datetime) -> tuple[list[datetime], list[datetime]]:
    """Split in-text dates into precise (full Y-M-D / Month D, Y) and bare-year cues.

    Bare years are imprecise and easily produced by copyright footers, current-year
    mentions, or roadmap targets, so they are kept separate and used only as a last
    resort when no explicit or precise date is available.
    """

    precise: list[datetime] = []
    for match in re.finditer(r"\b(20\d{2})[-/](0?[1-9]|1[0-2])[-/](0?[1-9]|[12]\d|3[01])\b", text):
        date = _safe_date(int(match.group(1)), int(match.group(2)), int(match.group(3)))
        if date is not None:
            precise.append(date)

    month_names = "|".join(_MONTHS)
    month_pattern = re.compile(
        rf"\b({month_names})\s+(0?[1-9]|[12]\d|3[01])(?:st|nd|rd|th)?,?\s+(20\d{{2}})\b",
        re.IGNORECASE,
    )
    for match in month_pattern.finditer(text):
        date = _safe_date(int(match.group(3)), _MONTHS[match.group(1).lower()], int(match.group(2)))
        if date is not None:
            precise.append(date)

    bare: list[datetime] = []
    for match in re.finditer(r"\b(20\d{2})\b", text):
        year = int(match.group(1))
        if 2000 <= year <= now.year:
            date = _safe_date(year, 1, 1)
            if date is not None:
                bare.append(date)

    return precise, bare


def _safe_date(year: int, month: int, day: int) -> datetime | None:
    try:
        return datetime(year, month, day, tzinfo=UTC)
    except ValueError:
        return None


def _version_cues(text: str) -> list[str]:
    lowered = text.lower()
    cues = [match.group(0) for match in _VERSION_RE.finditer(text)]
    cues.extend(cue for cue in _VOLATILE_CUES if cue in lowered)
    return _dedupe(cues)[:12]


def _has_any(text: str, cues: tuple[str, ...]) -> bool:
    lowered = text.lower()
    return any(cue in lowered for cue in cues)


def _ensure_aware(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _format_dates(dates: list[datetime]) -> list[str]:
    return _dedupe(date.date().isoformat() for date in dates)


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result
