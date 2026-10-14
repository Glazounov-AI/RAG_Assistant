import sqlite3
import sys
from pathlib import Path

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1] / "assistant_api"),
)

from rate_limiter import RateLimiter
from concurrent.futures import ThreadPoolExecutor

def test_rate_limit_allows_requests(tmp_path):
    limiter = RateLimiter(tmp_path / "rate_limits.db")

    assert limiter.check("client-1", "support", limit=2) is True
    assert limiter.check("client-1", "support", limit=2) is True


def test_rate_limit_blocks_excess_requests(tmp_path):
    limiter = RateLimiter(tmp_path / "rate_limits.db")

    assert limiter.check("client-1", "support", limit=2) is True
    assert limiter.check("client-1", "support", limit=2) is True
    assert limiter.check("client-1", "support", limit=2) is False


def test_rate_limit_separates_modes(tmp_path):
    limiter = RateLimiter(tmp_path / "rate_limits.db")

    assert limiter.check("client-1", "support", limit=1) is True
    assert limiter.check("client-1", "support", limit=1) is False

    assert limiter.check("client-1", "portfolio", limit=1) is True


def test_rate_limit_separates_clients(tmp_path):
    limiter = RateLimiter(tmp_path / "rate_limits.db")

    assert limiter.check("client-1", "support", limit=1) is True
    assert limiter.check("client-2", "support", limit=1) is True


def test_rate_limit_persists_between_instances(tmp_path):
    db_path = tmp_path / "rate_limits.db"

    first = RateLimiter(db_path)
    assert first.check("client-1", "support", limit=1) is True

    second = RateLimiter(db_path)
    assert second.check("client-1", "support", limit=1) is False


def test_rate_limit_does_not_increment_blocked_requests(tmp_path):
    db_path = tmp_path / "rate_limits.db"
    limiter = RateLimiter(db_path)

    assert limiter.check("client-1", "support", limit=1) is True
    assert limiter.check("client-1", "support", limit=1) is False

    with sqlite3.connect(db_path) as conn:
        count = conn.execute(
            """
            SELECT request_count
            FROM rate_limits
            WHERE client_key = ? AND mode = ?
            """,
            ("client-1", "support"),
        ).fetchone()[0]

    assert count == 1


def test_rate_limit_concurrent_requests(tmp_path):
    limiter = RateLimiter(tmp_path / "rate_limits.db")

    def send_request(_):
        return limiter.check(
            "client-1",
            "support",
            limit=10,
        )

    with ThreadPoolExecutor(max_workers=20) as executor:
        results = list(executor.map(send_request, range(30)))

    assert results.count(True) == 10
    assert results.count(False) == 20