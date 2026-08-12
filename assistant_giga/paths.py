"""Пути к локальным данным проекта (кэш и ChromaDB)."""

from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
DB_DIR = PROJECT_DIR / "db"
CACHE_DIR = DB_DIR / "cache"
CHROMADB_DIR = DB_DIR / "chromadb"

DEFAULT_CACHE_DB_PATH = CACHE_DIR / "rag_cache.db"
DEFAULT_CHROMADB_PATH = CHROMADB_DIR


def ensure_db_dirs() -> None:
    """Создаёт подпапки db/cache и db/chromadb при запуске проекта."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    CHROMADB_DIR.mkdir(parents=True, exist_ok=True)
