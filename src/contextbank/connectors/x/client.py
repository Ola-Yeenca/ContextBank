from __future__ import annotations

import math
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import httpx

from contextbank.models import XSyncEstimate

X_API_BASE_URL = "https://api.x.com"
BOOKMARKS_ENDPOINT = "/2/users/{id}/bookmarks"
ME_ENDPOINT = "/2/users/me"
POST_LOOKUP_ENDPOINT = "/2/tweets/{id}"
BOOKMARKS_DOCS_URL = "https://docs.x.com/x-api/users/get-bookmarks"
POST_LOOKUP_DOCS_URL = "https://docs.x.com/x-api/posts/get-post-by-id"
PRICING_DOCS_URL = "https://docs.x.com/x-api/getting-started/pricing"
RATE_LIMIT_DOCS_URL = "https://docs.x.com/x-api/fundamentals/rate-limits"
PRICING_CHECKED_ON = "2026-06-21"
OWNED_READ_UNIT_COST_USD = 0.001
STANDARD_POST_READ_UNIT_COST_USD = 0.005
MAX_BOOKMARKS_PAGE_SIZE = 100
BOOKMARKS_USER_RATE_LIMIT_PER_15_MIN = 180
REQUIRED_BOOKMARK_SCOPES = ("tweet.read", "users.read", "bookmark.read")


DEFAULT_TWEET_FIELDS = (
    "id",
    "text",
    "author_id",
    "conversation_id",
    "created_at",
    "lang",
    "attachments",
    "entities",
    "referenced_tweets",
    "in_reply_to_user_id",
    "possibly_sensitive",
    "public_metrics",
)

DEFAULT_EXPANSIONS = (
    "author_id",
    "attachments.media_keys",
    "referenced_tweets.id",
    "referenced_tweets.id.author_id",
)

DEFAULT_USER_FIELDS = ("id", "name", "username")

DEFAULT_MEDIA_FIELDS = (
    "media_key",
    "type",
    "url",
    "preview_image_url",
    "alt_text",
    "duration_ms",
    "height",
    "width",
    "public_metrics",
    "variants",
)


@dataclass(frozen=True)
class RateLimitInfo:
    limit: int | None = None
    remaining: int | None = None
    reset_epoch: int | None = None

    @property
    def reset_at(self) -> datetime | None:
        if self.reset_epoch is None:
            return None
        return datetime.fromtimestamp(self.reset_epoch, tz=UTC)


@dataclass(frozen=True)
class XBookmarkPage:
    data: list[dict[str, Any]]
    meta: dict[str, Any]
    includes: dict[str, Any]
    errors: list[dict[str, Any]]
    rate_limit: RateLimitInfo
    raw: dict[str, Any]

    @property
    def next_token(self) -> str | None:
        token = self.meta.get("next_token")
        return str(token) if token else None

    @property
    def result_count(self) -> int:
        value = self.meta.get("result_count")
        return int(value) if isinstance(value, int) or str(value).isdigit() else len(self.data)


@dataclass(frozen=True)
class XPostLookup:
    data: dict[str, Any]
    includes: dict[str, Any]
    errors: list[dict[str, Any]]
    rate_limit: RateLimitInfo
    raw: dict[str, Any]

    @property
    def id(self) -> str:
        return str(self.data.get("id") or "")


class XApiError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        errors: list[dict[str, Any]] | None = None,
        rate_limit: RateLimitInfo | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.errors = errors or []
        self.rate_limit = rate_limit


class XRateLimitError(XApiError):
    pass


class XBookmarkClient:
    def __init__(
        self,
        bearer_token: str,
        *,
        base_url: str = X_API_BASE_URL,
        client: httpx.Client | None = None,
        timeout: float = 20.0,
        max_retries: int = 2,
        backoff_seconds: float = 1.0,
        sleep: Callable[[float], None] | None = None,
    ) -> None:
        if not bearer_token or not bearer_token.strip():
            raise ValueError("An OAuth 2.0 user access bearer token is required.")
        self.bearer_token = bearer_token.strip()
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._client = client
        self.max_retries = max(max_retries, 0)
        self.backoff_seconds = max(backoff_seconds, 0.0)
        self._sleep = sleep or time.sleep

    @property
    def auth_assumptions(self) -> dict[str, Any]:
        return {
            "token_type": "OAuth 2.0 user access bearer token",
            "required_scopes": list(REQUIRED_BOOKMARK_SCOPES),
            "endpoint": BOOKMARKS_ENDPOINT,
            "bookmark_owner_requirement": "{id} must match the authenticated source user.",
            "docs": BOOKMARKS_DOCS_URL,
        }

    def get_authenticated_user(
        self,
        *,
        user_fields: Iterable[str] | None = ("id", "name", "username"),
    ) -> dict[str, Any]:
        params: dict[str, str | int] = {}
        _add_csv_param(params, "user.fields", user_fields)
        payload, _ = self._get_json(ME_ENDPOINT, params=params)
        data = payload.get("data")
        if not isinstance(data, dict):
            raise XApiError("X API /2/users/me response did not include a user object.")
        return data

    def get_bookmarks(
        self,
        user_id: str,
        *,
        max_results: int = MAX_BOOKMARKS_PAGE_SIZE,
        pagination_token: str | None = None,
        tweet_fields: Iterable[str] | None = DEFAULT_TWEET_FIELDS,
        expansions: Iterable[str] | None = DEFAULT_EXPANSIONS,
        user_fields: Iterable[str] | None = DEFAULT_USER_FIELDS,
        media_fields: Iterable[str] | None = DEFAULT_MEDIA_FIELDS,
    ) -> XBookmarkPage:
        if not user_id or not user_id.strip():
            raise ValueError("user_id is required.")
        if max_results < 1 or max_results > MAX_BOOKMARKS_PAGE_SIZE:
            raise ValueError("max_results must be between 1 and 100 for the bookmarks endpoint.")

        params: dict[str, str | int] = {"max_results": max_results}
        if pagination_token:
            params["pagination_token"] = pagination_token
        _add_csv_param(params, "tweet.fields", tweet_fields)
        _add_csv_param(params, "expansions", expansions)
        _add_csv_param(params, "user.fields", user_fields)
        _add_csv_param(params, "media.fields", media_fields)

        path = BOOKMARKS_ENDPOINT.format(id=user_id.strip())
        payload, rate_limit = self._get_json(path, params=params)
        return XBookmarkPage(
            data=list(payload.get("data") or []),
            meta=dict(payload.get("meta") or {}),
            includes=dict(payload.get("includes") or {}),
            errors=list(payload.get("errors") or []),
            rate_limit=rate_limit,
            raw=payload,
        )

    def get_post(
        self,
        tweet_id: str,
        *,
        tweet_fields: Iterable[str] | None = DEFAULT_TWEET_FIELDS,
        expansions: Iterable[str] | None = DEFAULT_EXPANSIONS,
        user_fields: Iterable[str] | None = DEFAULT_USER_FIELDS,
        media_fields: Iterable[str] | None = DEFAULT_MEDIA_FIELDS,
    ) -> XPostLookup:
        if not tweet_id or not tweet_id.strip():
            raise ValueError("tweet_id is required.")

        params: dict[str, str | int] = {}
        _add_csv_param(params, "tweet.fields", tweet_fields)
        _add_csv_param(params, "expansions", expansions)
        _add_csv_param(params, "user.fields", user_fields)
        _add_csv_param(params, "media.fields", media_fields)

        path = POST_LOOKUP_ENDPOINT.format(id=tweet_id.strip())
        payload, rate_limit = self._get_json(path, params=params)
        data = payload.get("data")
        if not isinstance(data, dict):
            raise XApiError("X API post lookup response did not include a post object.")
        return XPostLookup(
            data=data,
            includes=dict(payload.get("includes") or {}),
            errors=list(payload.get("errors") or []),
            rate_limit=rate_limit,
            raw=payload,
        )

    def iter_bookmark_pages(
        self,
        user_id: str,
        *,
        limit: int | None = None,
        page_size: int = MAX_BOOKMARKS_PAGE_SIZE,
        pagination_token: str | None = None,
        **kwargs: Any,
    ) -> Iterable[XBookmarkPage]:
        if limit is not None and limit < 0:
            raise ValueError("limit must be non-negative or None.")
        if page_size < 1 or page_size > MAX_BOOKMARKS_PAGE_SIZE:
            raise ValueError("page_size must be between 1 and 100.")

        remaining = limit
        next_token = pagination_token
        while remaining is None or remaining > 0:
            page_limit = page_size if remaining is None else min(page_size, remaining)
            page = self.get_bookmarks(
                user_id,
                max_results=page_limit,
                pagination_token=next_token,
                **kwargs,
            )
            yield page

            fetched_count = len(page.data)
            if remaining is not None:
                remaining -= fetched_count
            next_token = page.next_token
            if not next_token or fetched_count == 0:
                break

    def iter_bookmarks(
        self,
        user_id: str,
        *,
        limit: int | None = None,
        page_size: int = MAX_BOOKMARKS_PAGE_SIZE,
        pagination_token: str | None = None,
        **kwargs: Any,
    ) -> Iterable[dict[str, Any]]:
        emitted = 0
        for page in self.iter_bookmark_pages(
            user_id,
            limit=limit,
            page_size=page_size,
            pagination_token=pagination_token,
            **kwargs,
        ):
            for bookmark in page.data:
                if limit is not None and emitted >= limit:
                    return
                emitted += 1
                yield bookmark

    def _get_json(
        self,
        path: str,
        *,
        params: dict[str, str | int],
    ) -> tuple[dict[str, Any], RateLimitInfo]:
        created_client = self._client is None
        client = self._client or httpx.Client(timeout=self.timeout, trust_env=False)
        try:
            for attempt in range(self.max_retries + 1):
                response = client.get(
                    f"{self.base_url}{path}",
                    params=params,
                    headers={
                        "Authorization": f"Bearer {self.bearer_token}",
                        "Accept": "application/json",
                        "User-Agent": "ContextBank/0.1",
                    },
                )
                rate_limit = _rate_limit_from_headers(response.headers)
                if response.status_code != 429 or attempt >= self.max_retries:
                    break
                self._sleep(_retry_delay(rate_limit, attempt, self.backoff_seconds))
        finally:
            if created_client:
                client.close()

        payload = _safe_json(response)
        errors = list(payload.get("errors") or []) if isinstance(payload, dict) else []

        if response.status_code == 429:
            raise XRateLimitError(
                "X API rate limit exceeded for bookmarks endpoint.",
                status_code=response.status_code,
                errors=errors,
                rate_limit=rate_limit,
            )
        if response.status_code >= 400:
            message = (
                _error_message(payload)
                or f"X API request failed with HTTP {response.status_code}."
            )
            raise XApiError(
                message,
                status_code=response.status_code,
                errors=errors,
                rate_limit=rate_limit,
            )
        if not isinstance(payload, dict):
            raise XApiError(
                "X API response was not a JSON object.",
                status_code=response.status_code,
                rate_limit=rate_limit,
            )
        return payload, rate_limit


def estimate_bookmark_sync(
    requested_limit: int,
    *,
    owned_read: bool = True,
) -> XSyncEstimate:
    if requested_limit < 0:
        raise ValueError("requested_limit must be non-negative.")

    unit_cost = OWNED_READ_UNIT_COST_USD if owned_read else STANDARD_POST_READ_UNIT_COST_USD
    estimated_cost = requested_limit * unit_cost
    estimated_requests = (
        math.ceil(requested_limit / MAX_BOOKMARKS_PAGE_SIZE) if requested_limit else 0
    )
    pricing_basis = "Owned Read" if owned_read else "standard Post: Read"
    return XSyncEstimate(
        requested_limit=requested_limit,
        estimated_resources=requested_limit,
        estimated_cost_usd=round(estimated_cost, 6),
        pricing_note=(
            f"Dry-run estimate only, checked against X docs on {PRICING_CHECKED_ON}. "
            f"Uses {pricing_basis} unit cost ${unit_cost:.3f}/resource for "
            f"{BOOKMARKS_ENDPOINT}; estimated API requests: {estimated_requests}. "
            f"Current rates and eligibility should be verified in the X Developer Console. "
            f"Docs: {PRICING_DOCS_URL}"
        ),
        dry_run=True,
    )


dry_run_bookmark_sync_estimate = estimate_bookmark_sync


def _add_csv_param(
    params: dict[str, str | int],
    name: str,
    values: Iterable[str] | None,
) -> None:
    if values is None:
        return
    csv = ",".join(value for value in values if value)
    if csv:
        params[name] = csv


def _rate_limit_from_headers(headers: httpx.Headers) -> RateLimitInfo:
    return RateLimitInfo(
        limit=_header_int(headers, "x-rate-limit-limit"),
        remaining=_header_int(headers, "x-rate-limit-remaining"),
        reset_epoch=_header_int(headers, "x-rate-limit-reset"),
    )


def _retry_delay(rate_limit: RateLimitInfo, attempt: int, base_delay: float) -> float:
    exponential = base_delay * (2**attempt)
    if rate_limit.reset_at is None:
        return min(exponential, 30.0)
    seconds_until_reset = max((rate_limit.reset_at - datetime.now(UTC)).total_seconds(), 0.0)
    if seconds_until_reset <= 30.0:
        return max(seconds_until_reset, exponential)
    return min(exponential, 30.0)


def _header_int(headers: httpx.Headers, name: str) -> int | None:
    value = headers.get(name)
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _safe_json(response: httpx.Response) -> dict[str, Any] | list[Any] | None:
    if not response.content:
        return {}
    try:
        return response.json()
    except ValueError:
        return {
            "errors": [
                {
                    "title": "Invalid JSON response",
                    "detail": response.text[:500],
                    "status": response.status_code,
                }
            ]
        }


def _error_message(payload: dict[str, Any] | list[Any] | None) -> str | None:
    if not isinstance(payload, dict):
        return None
    errors = payload.get("errors")
    if not isinstance(errors, list) or not errors:
        return None
    first = errors[0]
    if not isinstance(first, dict):
        return str(first)
    return str(first.get("detail") or first.get("title") or first)
