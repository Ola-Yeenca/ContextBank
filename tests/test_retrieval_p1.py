from __future__ import annotations

import math
from datetime import UTC, datetime

from contextbank.models import SearchFilters
from contextbank.retrieval.search import _cosine_similarity
from contextbank.storage.repository import ContextBankRepository
from tests.helpers import seed_card


# ---- M-2: cosine similarity must be normalized, not a raw dot product ----
def test_cosine_similarity_is_normalized() -> None:
    # Parallel vectors of different magnitude → cosine 1.0 (raw dot product would give 10).
    assert _cosine_similarity([2.0, 0.0, 0.0], [5.0, 0.0, 0.0]) == 1.0
    # Orthogonal → 0.
    assert _cosine_similarity([1.0, 0.0], [0.0, 1.0]) == 0.0
    # 45 degrees → 1/sqrt(2).
    assert abs(_cosine_similarity([1.0, 1.0], [1.0, 0.0]) - (1 / math.sqrt(2))) < 1e-9
    # Zero vector → 0, no ZeroDivisionError.
    assert _cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0


# ---- M-10: FR-29 date-range + used/unused filters ----
def test_search_filters_have_date_range_and_used_fields() -> None:
    f = SearchFilters(date_from=datetime(2026, 1, 1, tzinfo=UTC), used=True)
    assert f.date_from == datetime(2026, 1, 1, tzinfo=UTC)
    assert f.used is True
    assert SearchFilters().used is None  # default = no filter


def test_date_range_filter_excludes_older_cards(tmp_path) -> None:
    repo = ContextBankRepository.from_home(tmp_path)
    old_s, _, _ = seed_card(repo, url="https://example.com/old", title="retrieval old note")
    new_s, _, _ = seed_card(repo, url="https://example.com/new", title="retrieval new note")
    repo.upsert_source_item(
        old_s.model_copy(update={"source_created_at": datetime(2020, 1, 1, tzinfo=UTC)})
    )
    repo.upsert_source_item(
        new_s.model_copy(update={"source_created_at": datetime(2026, 6, 1, tzinfo=UTC)})
    )
    try:
        hits = repo.search_cards(
            "retrieval",
            filters=SearchFilters(date_from=datetime(2026, 1, 1, tzinfo=UTC)),
            limit=10,
            default_visible=True,
        )
        urls = {r.source_url for r in hits}
        assert "https://example.com/new" in urls
        assert "https://example.com/old" not in urls
    finally:
        repo.close()


def test_used_unused_filter(tmp_path) -> None:
    repo = ContextBankRepository.from_home(tmp_path)
    seed_card(repo, url="https://example.com/u1", title="retrieval one")
    seed_card(repo, url="https://example.com/u2", title="retrieval two")
    try:
        # No project links exist → unused returns matches, used returns none.
        unused = repo.search_cards(
            "retrieval", filters=SearchFilters(used=False), limit=10, default_visible=True
        )
        used = repo.search_cards(
            "retrieval", filters=SearchFilters(used=True), limit=10, default_visible=True
        )
        assert len(unused) >= 2
        assert len(used) == 0
    finally:
        repo.close()
