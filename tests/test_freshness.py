from __future__ import annotations

from datetime import UTC, datetime

from contextbank.freshness.rules import score_freshness
from contextbank.models import FreshnessStatus

NOW = datetime(2026, 6, 20, tzinfo=UTC)


def test_bare_year_in_body_does_not_override_explicit_publish_date() -> None:
    # A 2020 reference that merely mentions a future-goal year / copyright footer must not
    # be rated CURRENT just because "2026" appears in the prose.
    result = score_freshness(
        content_type="reference",
        text="Written long ago. Roadmap to 2026 goals, planning for 2026, (c) 2026.",
        published_at=datetime(2020, 1, 1, tzinfo=UTC),
        as_of=NOW,
    )
    assert result.date_used == datetime(2020, 1, 1, tzinfo=UTC)
    assert result.status != FreshnessStatus.CURRENT
    # reference windows are (180, 365, 730); ~2335 days old => STALE
    assert result.status == FreshnessStatus.STALE


def test_capture_date_does_not_override_explicit_publish_date() -> None:
    result = score_freshness(
        content_type="reference",
        text="Some evergreen notes.",
        published_at=datetime(2020, 1, 1, tzinfo=UTC),
        captured_at=datetime(2026, 6, 1, tzinfo=UTC),
        as_of=NOW,
    )
    assert result.date_used == datetime(2020, 1, 1, tzinfo=UTC)
    assert result.status == FreshnessStatus.STALE


def test_bare_year_is_used_only_as_a_last_resort_when_no_explicit_date() -> None:
    result = score_freshness(
        content_type="reference",
        text="Notes from 2026 about local-first retrieval.",
        as_of=NOW,
    )
    assert result.date_used == datetime(2026, 1, 1, tzinfo=UTC)
    assert result.status == FreshnessStatus.CURRENT


def test_precise_text_date_still_used_when_no_explicit_date() -> None:
    result = score_freshness(
        content_type="reference",
        text="Updated 2026-05-01 with new guidance.",
        as_of=NOW,
    )
    assert result.date_used == datetime(2026, 5, 1, tzinfo=UTC)
    assert result.status == FreshnessStatus.CURRENT


def test_recent_explicit_publish_date_stays_current() -> None:
    result = score_freshness(
        content_type="implementation_guide",
        text="A current Python CLI guide.",
        published_at=datetime(2026, 5, 20, tzinfo=UTC),
        as_of=NOW,
    )
    assert result.status == FreshnessStatus.CURRENT
