"""Request rate limiting for the public AI API."""
from __future__ import annotations

import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Callable, Deque

LOGGER = logging.getLogger("dinoboard.ai.rate_limit")


@dataclass(frozen=True)
class RateLimitRule:
    name: str
    window_seconds: int
    max_requests: int
    matcher: Callable[[str, str], bool]


class SlidingWindowRateLimiter:
    def __init__(self, rules: list[RateLimitRule]):
        self._rules = rules
        self._hits: dict[tuple[str, str], Deque[float]] = defaultdict(deque)

    def check(self, ip: str, method: str, path: str, now: float | None = None) -> tuple[RateLimitRule, int] | None:
        now = time.monotonic() if now is None else now
        for rule in self._rules:
            if not rule.matcher(method, path):
                continue
            key = (rule.name, ip)
            hits = self._hits[key]
            cutoff = now - rule.window_seconds
            while hits and hits[0] <= cutoff:
                hits.popleft()
            if len(hits) >= rule.max_requests:
                retry_after = max(1, int(rule.window_seconds - (now - hits[0])))
                return rule, retry_after
            hits.append(now)
        return None


def _normalize_path(path: str) -> str:
    if path.startswith("/api/dinoboard/"):
        return path.removeprefix("/api/dinoboard")
    return path


def _is_ai_path(path: str) -> bool:
    normalized = _normalize_path(path)
    return normalized == "/ai/sessions" or normalized.startswith("/ai/sessions/")


def _is_create_session(method: str, path: str) -> bool:
    return method == "POST" and _normalize_path(path) == "/ai/sessions"


def _is_decide(method: str, path: str) -> bool:
    return method == "POST" and _normalize_path(path).startswith("/ai/sessions/") and path.endswith("/decide")


def _is_observe(method: str, path: str) -> bool:
    return method == "POST" and _normalize_path(path).startswith("/ai/sessions/") and path.endswith("/observe")


DEFAULT_RULES = [
    RateLimitRule("ai_global", 60, 180, lambda _method, path: _is_ai_path(path)),
    RateLimitRule("ai_create_session", 600, 10, _is_create_session),
    RateLimitRule("ai_decide", 60, 30, _is_decide),
    RateLimitRule("ai_observe", 60, 120, _is_observe),
]

LIMITER = SlidingWindowRateLimiter(DEFAULT_RULES)


def client_ip(request) -> str:
    host = request.client.host if request.client else ""
    forwarded_for = request.headers.get("x-forwarded-for", "")
    if forwarded_for and host in {"127.0.0.1", "::1", "testclient", ""}:
        return forwarded_for.split(",", 1)[0].strip()
    return host or "unknown"


async def rate_limit_middleware(request, call_next):
    path = request.url.path
    if not _is_ai_path(path):
        return await call_next(request)

    ip = client_ip(request)
    limited = LIMITER.check(ip, request.method, path)
    if limited is not None:
        from starlette.responses import JSONResponse

        rule, retry_after = limited
        LOGGER.warning("RATE_LIMIT ip=%s route=%s limit=%s", ip, path, rule.name)
        return JSONResponse(
            {"detail": "Too many requests"},
            status_code=429,
            headers={"Retry-After": str(retry_after)},
        )
    return await call_next(request)
