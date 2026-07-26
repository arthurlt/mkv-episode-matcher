"""Shared plumbing for the metadata providers: retries, caching, series matching."""

from __future__ import annotations

import contextlib
import logging
import re
from dataclasses import dataclass
from time import sleep

import httpx

from .cache import JsonCache

__all__ = [
    "JsonApiClient",
    "ProviderError",
    "QueryParams",
    "SeriesMatch",
    "SeriesNotFoundError",
    "normalize_title",
    "squash_title",
]

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT_S = 30.0
_BACKOFF_BASE_S = 0.5
_MAX_BACKOFF_S = 8.0
_RETRYABLE_STATUS = {408, 425, 429, 500, 502, 503, 504}
_PUNCTUATION = re.compile(r"[^a-z0-9]+")

#: Query string values the provider APIs accept.
QueryParams = dict[str, str | int | float | None]


class ProviderError(RuntimeError):
    """Raised when a metadata provider cannot satisfy a request."""


class SeriesNotFoundError(ProviderError):
    """Raised when no series on a provider plausibly matches the given name."""


@dataclass(frozen=True, slots=True)
class SeriesMatch:
    """A series resolved on one provider."""

    provider: str
    provider_id: str
    name: str
    year: int | None = None

    def __str__(self) -> str:
        """Return a human-readable description of the match."""
        suffix = f" ({self.year})" if self.year else ""
        return f"{self.name}{suffix} [{self.provider}:{self.provider_id}]"


def normalize_title(title: str) -> str:
    """Reduce a series title to a comparable form.

    Examples
    --------
    >>> normalize_title("Test Precinct: The Reunion!")
    'test precinct the reunion'
    """
    return _PUNCTUATION.sub(" ", title.lower()).strip()


def squash_title(title: str) -> str:
    """Reduce a title to letters and digits only.

    Disc labels and provider titles punctuate differently for the same show,
    so exact matching compares this form as well as :func:`normalize_title`.

    Examples
    --------
    >>> squash_title("M*A*S*H") == squash_title("MASH")
    True
    >>> squash_title("Test Precinct: The Reunion!")
    'testprecinctthereunion'
    """
    return normalize_title(title).replace(" ", "")


def choose_series(
    provider: str, candidates: list[SeriesMatch], query: str, year: int | None
) -> SeriesMatch:
    """Pick the candidate that best matches ``query`` (and ``year`` if given).

    An exact title match always wins. Failing that, a candidate whose title
    contains the query is accepted -- discs are usually labelled with the short
    form of a title. Anything looser is rejected rather than guessed at, because
    silently matching the wrong series would poison every downstream still.

    Raises
    ------
    SeriesNotFoundError
        If nothing plausibly matches.
    """
    pool = candidates
    if year is not None:
        pool = [candidate for candidate in pool if candidate.year == year]

    wanted = squash_title(query)
    exact = [candidate for candidate in pool if squash_title(candidate.name) == wanted]
    if exact:
        return exact[0]

    wanted = normalize_title(query)

    partial = [
        candidate for candidate in pool if wanted and wanted in normalize_title(candidate.name)
    ]
    if partial:
        return partial[0]

    raise SeriesNotFoundError(
        f"{provider} has no series matching {query!r}"
        + (f" from {year}" if year is not None else "")
    )


class JsonApiClient:
    """A small JSON HTTP client with disk caching and bounded retries."""

    provider: str = "provider"
    base_url: str = ""

    def __init__(
        self,
        *,
        client: httpx.Client,
        cache: JsonCache,
        max_retries: int = 3,
        timeout_s: float = _DEFAULT_TIMEOUT_S,
    ) -> None:
        self.client = client
        self.cache = cache
        self.max_retries = max_retries
        self.timeout_s = timeout_s

    def get_json(
        self,
        path: str,
        *,
        params: QueryParams | None = None,
        headers: dict[str, str] | None = None,
        cache_key: str | None = None,
    ) -> dict:
        """Fetch and parse a JSON document, using the disk cache when possible."""
        if cache_key is not None:
            cached = self.cache.get(cache_key)
            if cached is not None:
                logger.debug("%s cache hit: %s", self.provider, cache_key)
                return cached

        payload = self.request_json("GET", path, params=params, headers=headers)
        if cache_key is not None:
            self.cache.put(cache_key, payload)
        return payload

    def request_json(
        self,
        method: str,
        path: str,
        *,
        params: QueryParams | None = None,
        headers: dict[str, str] | None = None,
        json_body: dict[str, object] | None = None,
    ) -> dict:
        """Perform an HTTP request, retrying transient failures.

        Raises
        ------
        ProviderError
            On a non-retryable status, on exhausted retries, or on a body that
            is not a JSON object.
        """
        url = path if path.startswith("http") else f"{self.base_url}{path}"
        last_detail = "no attempts were made"

        for attempt in range(self.max_retries + 1):
            try:
                response = self.client.request(
                    method,
                    url,
                    params=params,
                    headers=headers,
                    json=json_body,
                    timeout=self.timeout_s,
                )
            except httpx.HTTPError as error:
                last_detail = f"{type(error).__name__}: {error}"
                logger.warning("%s request to %s failed (%s)", self.provider, url, last_detail)
            else:
                if response.status_code < 400:
                    return self._parse(response, url)
                last_detail = f"HTTP {response.status_code}"
                if response.status_code not in _RETRYABLE_STATUS:
                    raise ProviderError(f"{self.provider} {method} {url} failed: {last_detail}")
                logger.warning(
                    "%s %s returned %s (attempt %d/%d)",
                    self.provider,
                    url,
                    response.status_code,
                    attempt + 1,
                    self.max_retries + 1,
                )
                self._wait(response, attempt)
                continue

            self._wait(None, attempt)

        raise ProviderError(
            f"{self.provider} {method} {url} failed after "
            f"{self.max_retries + 1} attempts: {last_detail}"
        )

    def _wait(self, response: httpx.Response | None, attempt: int) -> None:
        """Sleep before the next retry, honouring ``Retry-After`` when present."""
        delay = min(_BACKOFF_BASE_S * (2**attempt), _MAX_BACKOFF_S)
        if response is not None:
            header = response.headers.get("retry-after")
            if header is not None:
                with contextlib.suppress(ValueError):
                    delay = float(header)
        sleep(delay)

    def _parse(self, response: httpx.Response, url: str) -> dict:
        """Decode a JSON object body, or raise :class:`ProviderError`."""
        try:
            payload = response.json()
        except ValueError as error:
            raise ProviderError(f"{self.provider} returned invalid JSON from {url}") from error
        if not isinstance(payload, dict):
            raise ProviderError(f"{self.provider} returned an unexpected body from {url}")
        return payload
