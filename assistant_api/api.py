"""
Минимальный HTTP API для RAG ассистента (FastAPI).

Дополнительная точка входа параллельно консольному app.py.
Запуск: uvicorn api:app --host 127.0.0.1 --port 8000
"""

import secrets
import json
from generation import generate, serialize_input
from generation_models import GenerationRequest, GenerationResponse
from generation_profiles import load_generation_profiles
from config import GENERATION_PROFILES_DIR, GENERATION_ACCESS_KEY
from openai import APITimeoutError
import time
from contextlib import asynccontextmanager

from pydantic import BaseModel, Field, field_validator, model_validator
from fastapi import FastAPI, HTTPException, Request
from config import (
    KNOWLEDGE_BASES,
    DEFAULT_MODE,
    RATE_LIMIT_ENABLED,
    API_AUTH_ENABLED,
    API_ACCESS_KEY,
)
from rate_limiter import RateLimiter

from rag_pipeline import RAGPipeline
from db_logger import QueryLogger


class QueryRequest(BaseModel):
    """Тело запроса POST /query."""

    query: str = Field(..., min_length=1)
    mode: str = Field(default=DEFAULT_MODE)

    sources: list[str] | None = Field(default=None, min_length=1, max_length=20)

    @field_validator("sources")
    @classmethod
    def validate_sources(cls, value):
        if value is not None:
            if any(not x.strip() or x != x.strip() or len(x) > 255 or any(c in x for c in ("/", "\\", "\x00")) or x in (".", "..") for x in value):
                raise ValueError("Invalid source filename")
            if len(set(value)) != len(value):
                raise ValueError("Duplicate sources")
        return value

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
    retrieved_sources: list[dict] = Field(default_factory=list)
    usage: dict = Field(default_factory=dict)
    strategy: str = "semantic"
    total: int | None = None
    page: int | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Однократная инициализация RAG pipeline и логгера при старте."""
    print("🚀 Инициализация HTTP API...")

    if API_AUTH_ENABLED and not API_ACCESS_KEY:
        raise RuntimeError(
            "RAG_API_ACCESS_KEY must be configured when API authentication is enabled"
        )

    app.state.generation_profiles = load_generation_profiles(GENERATION_PROFILES_DIR)
    if app.state.generation_profiles and not GENERATION_ACCESS_KEY:
        raise RuntimeError("RAG_GENERATION_ACCESS_KEY must be configured")

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
    if API_AUTH_ENABLED:
        provided_key = http_request.headers.get("X-API-Key", "")

        if not secrets.compare_digest(provided_key, API_ACCESS_KEY):
            raise HTTPException(
                status_code=401,
                detail="Неверный или отсутствующий API-ключ",
            )

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
        result = pipeline.query(request.query, mode=request.mode, **({"sources": request.sources} if request.sources is not None else {}))
    except APITimeoutError:
        raise HTTPException(status_code=504, detail="Превышено время ожидания LLM") from None
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
                query=result["query"] if KNOWLEDGE_BASES[request.mode].get("log_content", True) else "[redacted]",
                response=result["answer"] if KNOWLEDGE_BASES[request.mode].get("log_content", True) else "[redacted]",
                from_cache=result["from_cache"],
                response_time_ms=response_time_ms,
                source="api",
                user_id=None,
                username=None,
                mode=request.mode,
                model=result.get("model", pipeline.model),
                usage=result.get("usage"),
            )
        except Exception as e:
            print(f"⚠️  Не удалось записать лог запроса: {e}")

    return QueryResponse(
        query=result["query"],
        answer=result["answer"],
        from_cache=result["from_cache"],
        model=result.get("model", pipeline.model),
        retrieved_sources=result.get("retrieved_sources", []),
        usage=result.get("usage", {}),
        strategy=result.get("strategy", "semantic"),
        total=result.get("total"),
        page=result.get("page"),
    )


@app.post("/generate", response_model=GenerationResponse, responses={code: {"description": description} for code, description in [(401,"Invalid API key"),(403,"Generation access denied"),(422,"Invalid input or profile"),(429,"Rate limit"),(502,"Invalid provider response"),(503,"Provider unavailable"),(504,"Provider timeout")]})
def generate_endpoint(request: GenerationRequest, http_request: Request):
    if API_AUTH_ENABLED and not secrets.compare_digest(http_request.headers.get("X-API-Key", "").encode(), API_ACCESS_KEY.encode()):
        raise HTTPException(401, "Invalid API key")
    if not GENERATION_ACCESS_KEY or not secrets.compare_digest(http_request.headers.get("X-Generation-Key", "").encode(), GENERATION_ACCESS_KEY.encode()):
        raise HTTPException(403, "Generation access denied")
    entry = app.state.generation_profiles.get(request.profile)
    if entry is None:
        raise HTTPException(422, "Unknown generation profile")
    profile = entry[0]
    text = serialize_input(request.input)
    if len(text) > profile.max_input_length:
        raise HTTPException(422, "Generation input too long")
    limiter = app.state.rate_limiter
    if limiter is not None and not limiter.check(http_request.client.host if http_request.client else "unknown", "generate:" + profile.profile, profile.rate_limit_per_minute):
        raise HTTPException(429, "Generation rate limit exceeded")
    result = generate(entry, text)
    logger = app.state.logger
    if logger is not None:
        try:
            logger.log(query=text if profile.log_content else "[redacted]", response=json.dumps(result["result"], ensure_ascii=False) if profile.log_content else "[redacted]", from_cache=False, response_time_ms=result["response_time_ms"], source="generate", mode=profile.profile, model=result["model"], usage=result["usage"])
        except Exception:
            print("Generation logging failed")
    return result
