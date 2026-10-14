"""Ограничение частоты HTTP-запросов с хранением счётчиков в SQLite."""

import sqlite3
import time
from pathlib import Path

from paths import DEFAULT_RATE_LIMITS_DB_PATH


class RateLimiter:
    def __init__(self, db_path=None):
        self.db_path = str(db_path or DEFAULT_RATE_LIMITS_DB_PATH)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS rate_limits (
                    client_key TEXT NOT NULL,
                    mode TEXT NOT NULL,
                    window_start INTEGER NOT NULL,
                    request_count INTEGER NOT NULL,
                    PRIMARY KEY (client_key, mode, window_start)
                )
            """)

    def check(self, client_key: str, mode: str, limit: int) -> bool:
        """Атомарно учитывает запрос. True — разрешён, False — лимит исчерпан."""
        now = int(time.time())
        window_start = now - (now % 60)

        with sqlite3.connect(self.db_path, timeout=5) as conn:
            conn.execute("BEGIN IMMEDIATE")

            conn.execute(
                "DELETE FROM rate_limits WHERE window_start < ?",
                (window_start - 60,),
            )

            row = conn.execute(
                """
                SELECT request_count
                FROM rate_limits
                WHERE client_key = ? AND mode = ? AND window_start = ?
                """,
                (client_key, mode, window_start),
            ).fetchone()

            count = row[0] if row else 0

            if count >= limit:
                return False

            conn.execute(
                """
                INSERT INTO rate_limits
                    (client_key, mode, window_start, request_count)
                VALUES (?, ?, ?, 1)
                ON CONFLICT(client_key, mode, window_start)
                DO UPDATE SET request_count = request_count + 1
                """,
                (client_key, mode, window_start),
            )

        return True