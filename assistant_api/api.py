"""
Минимальный HTTP API для RAG ассистента (FastAPI).

Дополнительная точка входа параллельно консольному app.py.
Запуск: uvicorn api:app --host 127.0.0.1 --port 8000
"""

import time
from contextlib import asynccontextmanager

from pydantic import BaseModel, Field, field_validator, model_validator
from fastapi import FastAPI, HTTPException, Request
from config import KNOWLEDGE_BASES, DEFAULT_MODE, RATE_LIMIT_ENABLED
from rate_limiter import RateLimiter

from rag_pipeline import RAGPipeline
from db_logger import QueryLogger


class QueryRequest(BaseModel):
    """Тело запроса POST /query."""

    query: str = Field(..., min_length=1)
    mode: str = Field(default=DEFAULT_MODE)

    @field_validator("query")
    @classmethod
    def validate_query(cls, value: str) -> str:
        """Убирает пробелы по краям и отклоняет пустые строки."""
        stripped = value.strip()
        if not stripped:
            raise ValueError("query не должна быть пустой")
        return stripped

    @model_validator(mode="after")
    def validate_request(self):
        mode_config = KNOWLEDGE_BASES.get(self.mode)

        if mode_config is None:
            raise ValueError("Неизвестный режим")

        max_length = mode_config["max_query_length"]

        if len(self.query) > max_length:
            raise ValueError(
                f"Максимальная длина запроса для режима "
                f"'{self.mode}' — {max_length} символов"
            )

        return self

class QueryResponse(BaseModel):
    """Ответ на успешный POST /query."""

    query: str
    answer: str
    from_cache: bool
    model: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Однократная инициализация RAG pipeline и логгера при старте."""
    print("🚀 Инициализация HTTP API...")

    # Тяжёлая инициализация ( ChromaDB, кеш, индексация ) — один раз
    app.state.pipeline = RAGPipeline()
    app.state.rate_limiter = RateLimiter() if RATE_LIMIT_ENABLED else None

    # Логгер создаётся один раз; ошибка не останавливает API
    try:
        app.state.logger = QueryLogger()
    except Exception as e:
        print(f"⚠️  Не удалось инициализировать логирование запросов: {e}")
        app.state.logger = None

    print("✅ HTTP API готово к работе")
    yield


app = FastAPI(
    title="RAG Assistant API",
    lifespan=lifespan,
)


@app.get("/health")
def health():
    """Проверка доступности сервиса. Без pipeline и без записи в logs.db."""
    return {"status": "ok", "service": "rag-assistant-api"}


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest, http_request: Request):
    """Обработка вопроса через существующий RAGPipeline."""
    pipeline: RAGPipeline = app.state.pipeline
    logger: QueryLogger | None = app.state.logger

    limiter = app.state.rate_limiter

    if limiter is not None:
        client_key = (
            http_request.client.host
            if http_request.client is not None
            else "unknown"
        )

        limit = KNOWLEDGE_BASES[request.mode]["rate_limit_per_minute"]

        if not limiter.check(client_key, request.mode, limit):
            raise HTTPException(
                status_code=429,
                detail="Превышен лимит запросов. Повторите попытку позже.",
            )

    # Замер только выполнения pipeline.query()
    started_at = time.perf_counter()
    try:
        result = pipeline.query(request.query, mode=request.mode)
    except Exception:
        raise HTTPException(
            status_code=500,
            detail="Ошибка обработки запроса",
        ) from None

    response_time_ms = (time.perf_counter() - started_at) * 1000

    # Логирование: ошибка логгера не влияет на HTTP-ответ
    if logger is not None:
        try:
            logger.log(
                query=result["query"],
                response=result["answer"],
                from_cache=result["from_cache"],
                response_time_ms=response_time_ms,
                source="api",
                user_id=None,
                username=None,
            )
        except Exception as e:
            print(f"⚠️  Не удалось записать лог запроса: {e}")

    return QueryResponse(
        query=result["query"],
        answer=result["answer"],
        from_cache=result["from_cache"],
        model=result.get("model", pipeline.model),
    )
