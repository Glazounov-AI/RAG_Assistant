
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1] / "assistant_api"),
)

import api
from test_api_modes import FakePipeline


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(api, "API_AUTH_ENABLED", True)
    monkeypatch.setattr(api, "API_ACCESS_KEY", "test-secret-key")

    pipeline = FakePipeline()
    api.app.state.pipeline = pipeline
    api.app.state.logger = None
    api.app.state.rate_limiter = None

    http = TestClient(api.app, raise_server_exceptions=False)
    try:
        yield http, pipeline
    finally:
        http.close()

def test_missing_api_key(client):
    http, pipeline = client

    response = http.post("/query", json={"query": "VPN"})

    assert response.status_code == 401
    assert pipeline.calls == []


def test_invalid_api_key(client):
    http, pipeline = client

    response = http.post(
        "/query",
        json={"query": "VPN"},
        headers={"X-API-Key": "wrong-key"},
    )

    assert response.status_code == 401
    assert pipeline.calls == []


def test_valid_api_key(client):
    http, pipeline = client

    response = http.post(
        "/query",
        json={"query": "VPN"},
        headers={"X-API-Key": "test-secret-key"},
    )

    assert response.status_code == 200
    assert pipeline.calls == [("VPN", "support")]


def test_missing_configured_key(monkeypatch):
    monkeypatch.setattr(api, "API_AUTH_ENABLED", True)
    monkeypatch.setattr(api, "API_ACCESS_KEY", "")

    with pytest.raises(RuntimeError, match="RAG_API_ACCESS_KEY"):
        with TestClient(api.app):
            pass
