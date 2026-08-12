"""Настройки RAG-ассистента (GigaChat)."""

from paths import PROJECT_DIR, DEFAULT_CACHE_DB_PATH, DEFAULT_CHROMADB_PATH

# ChromaDB
COLLECTION_NAME = "gigachat_rag_collection"
CHROMADB_PATH = DEFAULT_CHROMADB_PATH

# Кэш
CACHE_DB_PATH = DEFAULT_CACHE_DB_PATH

# Документы
DATA_FILE = PROJECT_DIR / "data" / "docs.txt"

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
