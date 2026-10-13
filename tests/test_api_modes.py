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

    assert response.status_code == 400
    assert "Неизвестный режим" in response.json()["detail"]


def test_internal_error(client):
    http, _ = client

    response = http.post(
        "/query",
        json={"query": "Ошибка", "mode": "support"},
    )

    assert response.status_code == 500
    assert response.json()["detail"] == "Ошибка обработки запроса"