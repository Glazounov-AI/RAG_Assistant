"""Настройки RAG-ассистента (GigaChat)."""

from paths import PROJECT_DIR, DEFAULT_CACHE_DB_PATH, DEFAULT_LOGS_DB_PATH, DEFAULT_CHROMADB_PATH

# ChromaDB
COLLECTION_NAME = "gigachat_rag_collection"
CHROMADB_PATH = DEFAULT_CHROMADB_PATH

# Кэш
CACHE_DB_PATH = DEFAULT_CACHE_DB_PATH

# Логи запросов
LOGS_DB_PATH = DEFAULT_LOGS_DB_PATH

# Документы для индексации (сканируется при каждом запуске)
DATA_DIR = PROJECT_DIR / "data"
DATA_EXTENSIONS = {".txt", ".md", ".markdown"}

# LLM
MODEL = "GigaChat"
EMBEDDING_MODEL = "Embeddings"
TEMPERATURE = 0.3
MAX_TOKENS = 500

# Поиск
TOP_K = 3

# Разбиение текста на чанки
CHUNK_SIZE = 500        # целевой размер чанка (символы)
CHUNK_OVERLAP = 100     # перекрытие между соседними чанками (символы)
MIN_CHUNK_SIZE = 50     # минимальный размер чанка (символы)

# Промпты
SYSTEM_PROMPT = (
    "Ты - полезный AI ассистент, который отвечает на вопросы "
    "на основе предоставленного контекста."
)
