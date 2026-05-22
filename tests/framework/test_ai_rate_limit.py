from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "platform"))

from ai_service.rate_limit import (  # noqa: E402
    RateLimitRule,
    SlidingWindowRateLimiter,
    _is_create_session,
    _is_decide,
    _is_observe,
)


def test_specific_ai_routes_match_expected_methods():
    assert _is_create_session("POST", "/ai/sessions")
    assert not _is_create_session("GET", "/ai/sessions")
    assert _is_decide("POST", "/ai/sessions/abc/decide")
    assert not _is_decide("GET", "/ai/sessions/abc/decide")
    assert _is_observe("POST", "/api/dinoboard/ai/sessions/abc/observe")


def test_rate_limiter_blocks_after_window_quota():
    limiter = SlidingWindowRateLimiter([
        RateLimitRule("test", 10, 2, lambda _method, _path: True),
    ])

    assert limiter.check("203.0.113.10", "POST", "/ai/sessions", now=100.0) is None
    assert limiter.check("203.0.113.10", "POST", "/ai/sessions", now=101.0) is None

    limited = limiter.check("203.0.113.10", "POST", "/ai/sessions", now=102.0)
    assert limited is not None
    rule, retry_after = limited
    assert rule.name == "test"
    assert retry_after == 8

    assert limiter.check("203.0.113.10", "POST", "/ai/sessions", now=111.0) is None


def test_rate_limiter_isolated_by_ip():
    limiter = SlidingWindowRateLimiter([
        RateLimitRule("test", 10, 1, lambda _method, _path: True),
    ])

    assert limiter.check("203.0.113.10", "POST", "/ai/sessions", now=100.0) is None
    assert limiter.check("203.0.113.11", "POST", "/ai/sessions", now=100.0) is None
    assert limiter.check("203.0.113.10", "POST", "/ai/sessions", now=101.0) is not None
