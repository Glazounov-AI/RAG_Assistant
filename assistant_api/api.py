"""
Минимальный HTTP API для RAG ассистента (FastAPI).

Дополнительная точка входа параллельно консольному app.py.
Запуск: uvicorn api:app --host 127.0.0.1 --port 8000
"""

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

from rag_pipeline import RAGPipeline
from db_logger import QueryLogger


class QueryRequest(BaseModel):
    """Тело запроса POST /query."""

    query: str = Field(..., min_length=1)

    @field_validator("query")
    @classmethod
    def validate_query(cls, value: str) -> str:
        """Убирает пробелы по краям и отклоняет пустые строки."""
        stripped = value.strip()
        if not stripped:
            raise ValueError("query не должна быть пустой")
        return stripped


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
def query(request: QueryRequest):
    """Обработка вопроса через существующий RAGPipeline."""
    pipeline: RAGPipeline = app.state.pipeline
    logger: QueryLogger | None = app.state.logger

    # Замер только выполнения pipeline.query()
    started_at = time.perf_counter()
    try:
        result = pipeline.query(request.query)
    except Exception:
        raise HTTPException(status_code=500, detail="Ошибка обработки запроса")
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
