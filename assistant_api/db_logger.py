"""
Модуль логирования пользовательских запросов RAG-ассистента.
Использует отдельную SQLite-базу logs.db.
"""

import sqlite3
from datetime import datetime
from typing import Optional, Union
from pathlib import Path

from paths import ensure_db_dirs
import config


class QueryLogger:
    """Журнал пользовательских запросов в SQLite."""

    def __init__(self, db_path: Union[str, Path, None] = None):
        """
        Инициализация логгера.

        Args:
            db_path: путь к файлу базы данных SQLite
        """
        ensure_db_dirs()
        self.db_path = str(db_path or config.LOGS_DB_PATH)
        self._init_db()

    def _init_db(self):
        """Создание таблицы логов, если она не существует."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                user_id TEXT,
                username TEXT,
                source TEXT,
                query TEXT,
                response TEXT,
                from_cache INTEGER,
                response_time_ms REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        conn.commit()
        conn.close()

    def log(
        self,
        query: str,
        response: str,
        from_cache: bool,
        response_time_ms: float,
        source: str = "console",
        user_id: Optional[str] = None,
        username: Optional[str] = None,
    ) -> None:
        """
        Сохранение отдельной записи о пользовательском запросе.

        Args:
            query: текст запроса
            response: текст ответа
            from_cache: ответ получен из кеша
            response_time_ms: время обработки запроса в миллисекундах
            source: источник запроса
            user_id: идентификатор пользователя
            username: имя пользователя
        """
        now = datetime.now().isoformat()

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO logs (
                timestamp, user_id, username, source, query,
                response, from_cache, response_time_ms, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                now,
                user_id,
                username,
                source,
                query,
                response,
                1 if from_cache else 0,
                response_time_ms,
                now,
            ),
        )

        conn.commit()
        conn.close()
