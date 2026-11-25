"""Offline F2 regressions: real SQLite, fake retrieval and OpenAI."""
import hashlib
import sqlite3
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'assistant_api'))
from cache import RAGCache
from db_logger import QueryLogger
from rag_pipeline import RAGPipeline, embedding_tokens


@pytest.fixture
def pipeline(tmp_path):
    p = RAGPipeline.__new__(RAGPipeline)
    p.default_mode = 'support'
    p.model = 'base-model'
    p.knowledge_bases = {
        'support': {'model': 'support-model', 'system_prompt': 'support prompt', 'top_k': 3},
        'custom': {'model': 'custom-model', 'system_prompt': 'custom prompt', 'top_k': 5, 'cache_enabled': False},
    }
    p.cache = RAGCache(tmp_path / 'cache.db')
    p.vector_stores = {}
    for mode in p.knowledge_bases:
        store = Mock()
        store.get_index_version.return_value = 'same-version'
        def search(query, top_k, mode=mode):
            embedding_tokens.set(12)
            return [{'source': mode + '.md', 'text': mode, 'distance': 0.2}]
        store.search.side_effect = search
        p.vector_stores[mode] = store
    p.openai_client = Mock()
    p.openai_client.chat.completions.create.return_value = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content='answer'))],
        usage=SimpleNamespace(prompt_tokens=100, completion_tokens=20),
    )
    return p


def test_profile_isolation(pipeline):
    pipeline.knowledge_bases['custom']['cache_enabled'] = True
    a = pipeline.query('same', mode='support')
    b = pipeline.query('same', mode='custom')
    assert not a['from_cache'] and not b['from_cache']
    assert a['retrieved_sources'][0]['source'] == 'support.md'
    assert b['retrieved_sources'][0]['source'] == 'custom.md'
    assert pipeline.query('same', mode='custom')['from_cache']


def test_profile_generation_settings(pipeline):
    pipeline.knowledge_bases['custom'].update(temperature=0.2, max_tokens=800)
    result = pipeline.query('question', mode='custom')
    kwargs = pipeline.openai_client.chat.completions.create.call_args.kwargs
    assert kwargs['model'] == result['model'] == 'custom-model'
    assert kwargs['messages'][0]['content'] == 'custom prompt'
    assert kwargs['temperature'] == 0.2 and kwargs['max_tokens'] == 800
    pipeline.vector_stores['custom'].search.assert_called_once_with('question', top_k=5)


def test_sources_and_usage(pipeline):
    result = pipeline.query('question')
    assert result['retrieved_sources'] == [{'source': 'support.md', 'distance': 0.2}]
    assert result['usage'] == {'input_tokens': 100, 'output_tokens': 20, 'embedding_tokens': 12}


def test_cache_hit_preserves_sources_and_zero_usage(pipeline):
    first = pipeline.query('question')
    second = pipeline.query('question')
    assert second['from_cache']
    assert second['answer'] == first['answer']
    assert second['retrieved_sources'] == first['retrieved_sources']
    assert second['usage'] == dict(input_tokens=0, output_tokens=0, embedding_tokens=0)
    assert pipeline.openai_client.chat.completions.create.call_count == 1


def test_disabled_cache(pipeline):
    for _ in range(2):
        assert not pipeline.query('question', mode='custom')['from_cache']
    assert pipeline.cache.get_stats()['total_entries'] == 0
    assert pipeline.openai_client.chat.completions.create.call_count == 2


@pytest.mark.parametrize('field,value', [('model', 'new-model'), ('system_prompt', 'new prompt'), ('top_k', 4)])
def test_cache_invalidation(pipeline, field, value):
    pipeline.query('question')
    pipeline.knowledge_bases['support'][field] = value
    assert not pipeline.query('question')['from_cache']


def test_legacy_cache(tmp_path):
    path = tmp_path / 'old-cache.db'
    with sqlite3.connect(path) as conn:
        conn.execute('CREATE TABLE cache (query_hash TEXT PRIMARY KEY, query TEXT NOT NULL, answer TEXT NOT NULL, context TEXT, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)')
        conn.execute('INSERT INTO cache (query_hash, query, answer) VALUES (?, ?, ?)', (hashlib.sha256(b'old').hexdigest(), 'old', 'answer'))
    cache = RAGCache(path)
    assert cache.get('old')['answer'] == 'answer'
    assert cache.get('old', mode='support', kb_version='new') is None
    cache.set('old', 'new answer', mode='support', kb_version='new')
    assert cache.get_stats()['total_entries'] == 2


def test_legacy_logs_migration(tmp_path):
    path = tmp_path / 'old-logs.db'
    with sqlite3.connect(path) as conn:
        conn.execute('CREATE TABLE logs (id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT, user_id TEXT, username TEXT, source TEXT, query TEXT, response TEXT, from_cache INTEGER, response_time_ms REAL, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)')
        conn.execute("INSERT INTO logs (query, response) VALUES ('old', 'answer')")
    QueryLogger(path)
    logger = QueryLogger(path)
    logger.log('new', 'answer', False, 10, mode='custom', model='custom-model', usage={'input_tokens': 100, 'output_tokens': 20, 'embedding_tokens': 12})
    with sqlite3.connect(path) as conn:
        rows = conn.execute('SELECT query, mode, model, input_tokens, output_tokens, embedding_tokens FROM logs ORDER BY id').fetchall()
    assert rows == [('old', None, None, None, None, None), ('new', 'custom', 'custom-model', 100, 20, 12)]
