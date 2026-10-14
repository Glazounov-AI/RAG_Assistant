import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1] / "assistant_api"),
)

from api import app


class FakePipeline:
    model = "test-model"

    def __init__(self):
        self.calls = []

    def query(self, question, mode=None):
        self.calls.append((question, mode))

        if mode not in ("support", "portfolio"):
            raise ValueError(f"Неизвестный режим: {mode}")

        if question == "Ошибка":
            raise RuntimeError("Тестовая внутренняя ошибка")

        return {
            "query": question,
            "answer": f"Ответ режима {mode}",
            "from_cache": False,
            "model": self.model,
        }


@pytest.fixture
def client():
    pipeline = FakePipeline()
    app.state.pipeline = pipeline
    app.state.logger = None
    app.state.rate_limiter = None

    test_client = TestClient(app, raise_server_exceptions=False)

    try:
        yield test_client, pipeline
    finally:
        test_client.close()


def test_default_mode(client):
    http, pipeline = client

    response = http.post("/query", json={"query": "VPN"})

    assert response.status_code == 200
    assert pipeline.calls == [("VPN", "support")]
    assert response.json()["answer"] == "Ответ режима support"


def test_portfolio_mode(client):
    http, pipeline = client

    response = http.post(
        "/query",
        json={"query": "Проекты", "mode": "portfolio"},
    )

    assert response.status_code == 200
    assert pipeline.calls == [("Проекты", "portfolio")]
    assert response.json()["answer"] == "Ответ режима portfolio"


def test_unknown_mode(client):
    http, _ = client

    response = http.post(
        "/query",
        json={"query": "Тест", "mode": "unknown"},
    )

    assert response.status_code == 422
    assert any(
        "Неизвестный режим" in error["msg"]
        for error in response.json()["detail"]
    )


def test_internal_error(client):
    http, _ = client

    response = http.post(
        "/query",
        json={"query": "Ошибка", "mode": "support"},
    )

    assert response.status_code == 500
    assert response.json()["detail"] == "Ошибка обработки запроса"

@pytest.mark.parametrize(
    "mode, max_length",
    [
        ("support", 2000),
        ("portfolio", 8000),
    ],
)

def test_query_length_limits(client, mode, max_length):
    http, pipeline = client

    valid_response = http.post(
        "/query",
        json={"query": "А" * max_length, "mode": mode},
    )
    assert valid_response.status_code == 200

    invalid_response = http.post(
        "/query",
        json={"query": "А" * (max_length + 1), "mode": mode},
    )
    assert invalid_response.status_code == 422

    # Превышающий лимит запрос не должен попадать в RAG.
    assert len(pipeline.calls) == 1

def test_internal_value_error_is_not_exposed(client):
    http, pipeline = client

    def raise_sensitive_error(question, mode=None):
        raise ValueError(
            "SECRET_INTERNAL_DATABASE_CONNECTION_STRING"
        )

    pipeline.query = raise_sensitive_error

    response = http.post(
        "/query",
        json={"query": "Проверка ошибки", "mode": "support"},
    )

    assert response.status_code == 500
    assert response.json()["detail"] == "Ошибка обработки запроса"
    assert "SECRET_INTERNAL_DATABASE_CONNECTION_STRING" not in response.text

def test_rate_limit_returns_429(client, tmp_path):
    from rate_limiter import RateLimiter

    http, pipeline = client

    app.state.rate_limiter = RateLimiter(
        tmp_path / "rate_limits.db"
    )

    # В режиме portfolio настроен лимит 5 запросов в минуту.
    for _ in range(5):
        response = http.post(
            "/query",
            json={"query": "Проекты", "mode": "portfolio"},
        )
        assert response.status_code == 200

    response = http.post(
        "/query",
        json={"query": "Проекты", "mode": "portfolio"},
    )

    assert response.status_code == 429
    assert len(pipeline.calls) == 5
    assert "Превышен лимит" in response.json()["detail"]

def test_rate_limit_modes_are_independent(client, tmp_path):
    from rate_limiter import RateLimiter
    from config import KNOWLEDGE_BASES

    http, pipeline = client

    app.state.rate_limiter = RateLimiter(
        tmp_path / "rate_limits.db"
    )

    portfolio_limit = KNOWLEDGE_BASES["portfolio"]["rate_limit_per_minute"]

    for _ in range(portfolio_limit):
        response = http.post(
            "/query",
            json={"query": "Проекты", "mode": "portfolio"},
        )
        assert response.status_code == 200

    blocked = http.post(
        "/query",
        json={"query": "Проекты", "mode": "portfolio"},
    )
    assert blocked.status_code == 429

    support = http.post(
        "/query",
        json={"query": "VPN", "mode": "support"},
    )
    assert support.status_code == 200
    assert support.json()["answer"] == "Ответ режима support"

    assert len(pipeline.calls) == portfolio_limit + 1

def test_rate_limit_disabled(client):
    from config import KNOWLEDGE_BASES

    http, pipeline = client

    app.state.rate_limiter = None

    limit = KNOWLEDGE_BASES["portfolio"]["rate_limit_per_minute"]

    for _ in range(limit + 1):
        response = http.post(
            "/query",
            json={"query": "Проекты", "mode": "portfolio"},
        )
        assert response.status_code == 200

    assert len(pipeline.calls) == limit + 1