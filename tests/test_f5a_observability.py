import time
import pytest
from observability import deadline, remaining, install
from fastapi import FastAPI
from fastapi.testclient import TestClient

def test_budget():
    token = deadline.set(time.monotonic()+0.5)
    try:
        assert 0 < remaining(10) <= 0.5
    finally:
        deadline.reset(token)

def test_expired():
    token = deadline.set(time.monotonic()-1)
    try:
        with pytest.raises(TimeoutError):
            remaining(10)
    finally:
        deadline.reset(token)

def test_trace():
    app = FastAPI()
    install(app)
    @app.get('/')
    def endpoint():
        return {'ok': True}
    client = TestClient(app)
    assert client.get('/', headers={'X-Request-ID':'a'*32}).headers['X-Request-ID'] == 'a'*32
    assert len(client.get('/', headers={'X-Request-ID':'invalid'}).headers['X-Request-ID']) == 32
