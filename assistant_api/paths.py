"""Пути к локальным данным проекта (кэш, логи и ChromaDB)."""

import os
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
# Базовый каталог данных можно переопределить через переменную окружения
# RAG_DB_DIR (например, для Railway persistent volume: RAG_DB_DIR=/data/db)
DB_DIR = Path(os.getenv("RAG_DB_DIR", str(PROJECT_DIR / "db")))
CACHE_DIR = DB_DIR / "cache"
LOGS_DIR = DB_DIR / "logs"
CHROMADB_DIR = DB_DIR / "chromadb"
RATE_LIMITS_DIR = DB_DIR / "rate_limits"

DEFAULT_CACHE_DB_PATH = CACHE_DIR / "rag_cache.db"
DEFAULT_LOGS_DB_PATH = LOGS_DIR / "logs.db"
DEFAULT_CHROMADB_PATH = CHROMADB_DIR
DEFAULT_RATE_LIMITS_DB_PATH = RATE_LIMITS_DIR / "rate_limits.db"


def ensure_db_dirs() -> None:
    """Создаёт каталоги для кеша, логов, ChromaDB и ограничителя запросов."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    CHROMADB_DIR.mkdir(parents=True, exist_ok=True)
    RATE_LIMITS_DIR.mkdir(parents=True, exist_ok=True)
