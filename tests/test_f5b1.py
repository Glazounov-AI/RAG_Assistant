from unittest.mock import Mock
import pytest
from test_f2_regression import pipeline
from retrieval import retrieve
from api import app, RetrieveRequest
import api
from fastapi.testclient import TestClient


def test_retrieval_only(pipeline):
    result = retrieve(pipeline, 'question', 'support')
    assert result['usage'] == dict(input_tokens=0, output_tokens=0, embedding_tokens=12)
    assert result['chunks'][0]['text'] == 'support'
    pipeline.openai_client.chat.completions.create.assert_not_called()
    assert pipeline.cache.get_stats()['total_entries'] == 0


def test_bounded_targeted(pipeline):
    store = pipeline.vector_stores['support']
    store.search.side_effect = None
    store.search.return_value = [{'text':'abcdef','source':'a.md','distance':0.1}]
    result = retrieve(pipeline, 'question', 'support', ['a.md'], 3)
    assert result['chunks'][0]['text'] == 'abc'
    assert result['truncated'] and result['context_chars'] == 3
    assert result['usage']['embedding_tokens'] is None
    store.search.assert_called_once_with('question', top_k=3, sources=['a.md'])

@pytest.mark.parametrize('extra', [{'max_context_chars':0}, {'max_context_chars':24001}, {'top_k':100}, {'sources':['../a']}])
def test_validation(extra):
    with pytest.raises(ValueError):
        RetrieveRequest(query='test', **extra)

@pytest.fixture
def client(monkeypatch, pipeline):
    monkeypatch.setattr(api, 'API_AUTH_ENABLED', False)
    monkeypatch.setattr(api, 'API_ACCESS_KEY', 'test-key')
    monkeypatch.setattr(app.state, 'pipeline', pipeline, raising=False)
    monkeypatch.setattr(app.state, 'logger', None, raising=False)
    monkeypatch.setattr(app.state, 'rate_limiter', None, raising=False)
    http = TestClient(app)
    yield http
    http.close()


def test_access(client):
    assert client.post('/retrieve', json={'query':'test'}).status_code == 403


def test_success(client):
    response = client.post('/retrieve', json={'query':'test'}, headers={'X-API-Key':'test-key'})
    assert response.status_code == 200
    assert response.json()['usage']['output_tokens'] == 0
    assert response.headers['X-Request-ID']


def test_not_ready(client, monkeypatch):
    monkeypatch.setattr(app.state, 'pipeline', None)
    assert client.post('/retrieve', json={'query':'test'}, headers={'X-API-Key':'test-key'}).status_code == 503


def test_shared_rate_limit(client, monkeypatch):
    limiter = Mock()
    limiter.check.return_value = False
    monkeypatch.setattr(app.state, 'rate_limiter', limiter)
    assert client.post('/retrieve', json={'query':'test'}, headers={'X-API-Key':'test-key'}).status_code == 429
    assert limiter.check.call_args.args[1] == 'support'
